"""
database.py
───────────
MongoDB persistence layer for the Gut Health API.

Collections:
  users            — user profiles and onboarding scores
  food_logs        — every food entry logged by a user
  symptom_logs     — every symptom entry logged by a user
  user_memories    — serialised Bayesian learning state per user

All CRUD functions are synchronous (Motor async version trivial to add).
Connection is established once at import via MongoDBClient singleton.

Usage:
  from database import db
  db.insert_food_log(...)
  db.get_food_logs(user_id, days=7)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from pymongo import MongoClient, DESCENDING, ASCENDING
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, DuplicateKeyError

from user_symptom_memory import UserMemory, UserMemoryStore

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config — read from env or use defaults
# ─────────────────────────────────────────────────────────────────────────────

MONGO_URI  = os.getenv("MONGO_URI",  "mongodb://localhost:27017")
DB_NAME    = os.getenv("MONGO_DB",   "gut_health")


# ─────────────────────────────────────────────────────────────────────────────
# MongoDB client singleton
# ─────────────────────────────────────────────────────────────────────────────

class _MongoClient:
    def __init__(self):
        self._client: Optional[MongoClient] = None

    def connect(self) -> None:
        log.info(f"Connecting to MongoDB: {MONGO_URI} / {DB_NAME}")
        self._client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Verify connection
        self._client.admin.command("ping")
        log.info("MongoDB connected.")
        self._ensure_indexes()

    def disconnect(self) -> None:
        if self._client:
            self._client.close()
            log.info("MongoDB disconnected.")

    def _db(self):
        return self._client[DB_NAME]

    def _col(self, name: str) -> Collection:
        return self._db()[name]

    # ── Index creation (idempotent) ───────────────────────────────────────────

    def _ensure_indexes(self) -> None:
        # users: unique on user_id
        self._col("users").create_index("user_id", unique=True)

        # food_logs: query by user_id + date range (most common query)
        self._col("food_logs").create_index(
            [("user_id", ASCENDING), ("logged_at", DESCENDING)]
        )

        # symptom_logs: query by user_id + date range
        self._col("symptom_logs").create_index(
            [("user_id", ASCENDING), ("logged_at", DESCENDING)]
        )

        # user_memories: unique per user
        self._col("user_memories").create_index("user_id", unique=True)

        # usda_cache: unique on fdc_id — O(1) nutrient lookups
        # food_category stored here is used by POST /food/tags
        self._col("usda_cache").create_index("fdc_id", unique=True)

        # usda_search_cache: USDA text-search results keyed by normalised query string
        # Permanent — "Chicken, broiler, breast, cooked, roasted" always returns the same ID
        self._col("usda_search_cache").create_index("query", unique=True)

        # food_parse_cache: Claude parse results keyed by SHA-256 of input text
        # Permanent — nutritional identity of a food never changes
        self._col("food_parse_cache").create_index("cache_key", unique=True)

        # meal_score_cache: Claude meal score keyed by SHA-256 of foods+meal_type
        # Permanent — same foods at same weights for same meal type always score the same
        self._col("meal_score_cache").create_index("cache_key", unique=True)

        # gentle_note_cache: one-sentence Claude note keyed by SHA-256 of trigger summary
        # Permanent — same trigger pattern always produces equivalent advice
        self._col("gentle_note_cache").create_index("cache_key", unique=True)

        # composite_food_cache: maps a parse cache_key → user's original food name
        # e.g. "chicken biryani" → [171477, 169708, ...]
        # Permanent — user-given names never change for the same input text
        self._col("composite_food_cache").create_index("cache_key", unique=True)

        log.info("MongoDB indexes ensured.")

    # ══════════════════════════════════════════════════════════════════════════
    # USERS
    # ══════════════════════════════════════════════════════════════════════════

    def upsert_user(self, user: dict) -> None:
        """
        Insert or update a user profile.
        user must contain: user_id, name, age, gender, sleep_hours, weight_kg
        Also stores: onboarding_score, grade, created_at
        """
        user["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._col("users").update_one(
            {"user_id": user["user_id"]},
            {"$set": user, "$setOnInsert": {"created_at": datetime.now(timezone.utc).isoformat()}},
            upsert=True,
        )
        log.info(f"Upserted user: {user['user_id']}")

    def get_user(self, user_id: str) -> Optional[dict]:
        doc = self._col("users").find_one({"user_id": user_id}, {"_id": 0})
        return doc

    def set_user_score(self, user_id: str, current_score: int, grade: str) -> None:
        """
        Persist the user's latest current_score + grade.
        Creates a minimal user shell on first write if the profile does not exist yet.
        """
        now = datetime.now(timezone.utc).isoformat()
        self._col("users").update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "current_score": int(current_score),
                    "grade": grade,
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "created_at": now,
                    "name": "Unknown",
                },
            },
            upsert=True,
        )

    def list_users(self) -> list[dict]:
        return list(self._col("users").find({}, {"_id": 0}))

    # ══════════════════════════════════════════════════════════════════════════
    # FOOD LOGS
    # ══════════════════════════════════════════════════════════════════════════

    def insert_food_log(self, log_entry: dict) -> str:
        """
        Insert one food log entry.
        Required keys: user_id, usda_id, usda_description, quantity_g, unit,
                       meal_type, logged_at (ISO string or datetime)
        Optional:      score_modifier, nutrient_profile, hf_digestibility
        Returns inserted _id as string.
        """
        entry = dict(log_entry)
        if isinstance(entry.get("logged_at"), datetime):
            entry["logged_at"] = entry["logged_at"].isoformat()
        result = self._col("food_logs").insert_one(entry)
        return str(result.inserted_id)

    def insert_food_logs_bulk(self, entries: list[dict]) -> int:
        """Bulk insert food logs. Returns count inserted."""
        if not entries:
            return 0
        docs = []
        for e in entries:
            d = dict(e)
            if isinstance(d.get("logged_at"), datetime):
                d["logged_at"] = d["logged_at"].isoformat()
            docs.append(d)
        result = self._col("food_logs").insert_many(docs)
        return len(result.inserted_ids)

    def get_food_logs(
        self,
        user_id: str,
        days:    int = 7,
        limit:   int = 200,
    ) -> list[dict]:
        """
        Retrieve food logs for a user from the last N days.
        Returns newest first.
        """
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cursor = (
            self._col("food_logs")
            .find(
                {"user_id": user_id, "logged_at": {"$gte": since}},
                {"_id": 0},
            )
            .sort("logged_at", DESCENDING)
            .limit(limit)
        )
        return list(cursor)

    def get_all_food_logs(self, user_id: str, limit: int = 500) -> list[dict]:
        """All food logs for a user, newest first."""
        return list(
            self._col("food_logs")
            .find({"user_id": user_id}, {"_id": 0})
            .sort("logged_at", DESCENDING)
            .limit(limit)
        )

    def count_food_logs(self, user_id: str) -> int:
        return self._col("food_logs").count_documents({"user_id": user_id})

    # ══════════════════════════════════════════════════════════════════════════
    # SYMPTOM LOGS
    # ══════════════════════════════════════════════════════════════════════════

    def insert_symptom_log(self, log_entry: dict) -> str:
        """
        Insert one symptom log entry.
        Required keys: user_id, symptom, severity, logged_at
        Optional:      score_penalty, notes
        Returns inserted _id as string.
        """
        entry = dict(log_entry)
        if isinstance(entry.get("logged_at"), datetime):
            entry["logged_at"] = entry["logged_at"].isoformat()
        result = self._col("symptom_logs").insert_one(entry)
        return str(result.inserted_id)

    def insert_symptom_logs_bulk(self, entries: list[dict]) -> int:
        if not entries:
            return 0
        docs = []
        for e in entries:
            d = dict(e)
            if isinstance(d.get("logged_at"), datetime):
                d["logged_at"] = d["logged_at"].isoformat()
            docs.append(d)
        result = self._col("symptom_logs").insert_many(docs)
        return len(result.inserted_ids)

    def get_symptom_logs(
        self,
        user_id: str,
        days:    int = 7,
        limit:   int = 100,
    ) -> list[dict]:
        """Symptom logs from the last N days, newest first."""
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cursor = (
            self._col("symptom_logs")
            .find(
                {"user_id": user_id, "logged_at": {"$gte": since}},
                {"_id": 0},
            )
            .sort("logged_at", DESCENDING)
            .limit(limit)
        )
        return list(cursor)

    def count_symptom_logs(self, user_id: str) -> int:
        return self._col("symptom_logs").count_documents({"user_id": user_id})

    # ══════════════════════════════════════════════════════════════════════════
    # USER MEMORIES  (Bayesian learning state)
    # ══════════════════════════════════════════════════════════════════════════

    def save_memory(self, memory: UserMemory) -> None:
        """Persist a UserMemory object to MongoDB."""
        doc = memory.to_dict()
        doc["saved_at"] = datetime.now(timezone.utc).isoformat()
        self._col("user_memories").update_one(
            {"user_id": memory.user_id},
            {"$set": doc},
            upsert=True,
        )

    def load_memory(self, user_id: str) -> UserMemory:
        """Load UserMemory from MongoDB, or create a fresh one."""
        doc = self._col("user_memories").find_one({"user_id": user_id}, {"_id": 0})
        if doc:
            return UserMemory.from_dict(doc)
        return UserMemory.new(user_id)

    def memory_exists(self, user_id: str) -> bool:
        return self._col("user_memories").count_documents({"user_id": user_id}) > 0

    # ══════════════════════════════════════════════════════════════════════════
    # DIGESTION SCORE HISTORY
    # ══════════════════════════════════════════════════════════════════════════

    def insert_score_event(self, user_id: str, event: dict) -> str:
        """
        Log a score change event (food log, symptom log, onboarding).
        event keys: event_type, previous_score, new_score, modifier, timestamp, details
        """
        doc = {
            "user_id":   user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        result = self._col("score_history").insert_one(doc)
        return str(result.inserted_id)

    def get_score_history(self, user_id: str, days: int = 7) -> list[dict]:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        return list(
            self._col("score_history")
            .find(
                {"user_id": user_id, "timestamp": {"$gte": since}},
                {"_id": 0},
            )
            .sort("timestamp", ASCENDING)
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY QUERIES
    # ══════════════════════════════════════════════════════════════════════════

    def user_dashboard(self, user_id: str) -> dict:
        """
        Single call that fetches everything needed for the user's dashboard.
        Returns aggregated stats for the last 7 days.
        """
        user         = self.get_user(user_id) or {}
        food_logs    = self.get_food_logs(user_id, days=7)
        symptom_logs = self.get_symptom_logs(user_id, days=7)
        score_hist   = self.get_score_history(user_id, days=7)
        memory       = self.load_memory(user_id)

        # Unique foods eaten this week
        unique_foods = list({f["usda_description"] for f in food_logs if "usda_description" in f})

        # Symptom frequency this week
        sym_counts: dict[str, int] = {}
        for s in symptom_logs:
            sym_counts[s.get("symptom", "Unknown")] = sym_counts.get(s.get("symptom", "Unknown"), 0) + 1

        return {
            "user_id":               user_id,
            "name":                  user.get("name", "Unknown"),
            "current_score":         user.get("current_score", 0),
            "grade":                 user.get("grade", "Unknown"),
            "food_logs_this_week":   len(food_logs),
            "symptom_logs_this_week":len(symptom_logs),
            "unique_foods_eaten":    unique_foods[:10],
            "symptom_frequency":     sym_counts,
            "score_history":         score_hist,
            "personalisation_weight":memory.personalisation_weight,
            "top_sensitivities":     [
                {
                    "pair":        k,
                    "probability": round(v.posterior_mean, 3),
                    "observations":v.observations,
                }
                for k, v in sorted(
                    memory.priors.items(),
                    key=lambda x: x[1].posterior_mean,
                    reverse=True,
                )
                if v.observations > 0
            ][:5],
        }


    # ══════════════════════════════════════════════════════════════════════════
    # USDA SEARCH CACHE
    # ══════════════════════════════════════════════════════════════════════════

    def get_usda_search(self, query: str) -> Optional[list]:
        """
        Return cached USDA search results for query, or None on miss.
        Results are a list of {rank, usda_id, usda_description, similarity} dicts.
        """
        doc = self._col("usda_search_cache").find_one(
            {"query": query}, {"_id": 0, "query": 0, "cached_at": 0}
        )
        return doc.get("results") if doc else None

    def set_usda_search(self, query: str, results: list) -> None:
        """Persist USDA search results for a query string."""
        self._col("usda_search_cache").update_one(
            {"query": query},
            {"$set": {
                "query":     query,
                "results":   results,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }},
            upsert=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # FOOD PARSE CACHE
    # ══════════════════════════════════════════════════════════════════════════

    def get_food_parse(self, cache_key: str) -> Optional[list]:
        """
        Return cached parse result for cache_key, or None on miss.
        Result is a list of food dicts (without logged_at — caller adds that).
        """
        doc = self._col("food_parse_cache").find_one(
            {"cache_key": cache_key}, {"_id": 0, "cache_key": 0, "cached_at": 0}
        )
        if doc:
            return doc.get("foods")
        return None

    def set_food_parse(self, cache_key: str, foods: list) -> None:
        """
        Persist a parse result. foods is the list of food dicts from text_to_usda(),
        with logged_at stripped out before storing.
        """
        self._col("food_parse_cache").update_one(
            {"cache_key": cache_key},
            {"$set": {
                "cache_key": cache_key,
                "foods":     foods,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }},
            upsert=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # MEAL SCORE CACHE
    # ══════════════════════════════════════════════════════════════════════════

    def get_meal_score(self, cache_key: str) -> Optional[int]:
        """Return cached raw_score for cache_key, or None on miss."""
        doc = self._col("meal_score_cache").find_one(
            {"cache_key": cache_key}, {"_id": 0}
        )
        return int(doc["raw_score"]) if doc and "raw_score" in doc else None

    def set_meal_score(self, cache_key: str, raw_score: int) -> None:
        """Persist a meal raw_score result."""
        self._col("meal_score_cache").update_one(
            {"cache_key": cache_key},
            {"$set": {
                "cache_key": cache_key,
                "raw_score": raw_score,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }},
            upsert=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # GENTLE NOTE CACHE
    # ══════════════════════════════════════════════════════════════════════════

    def get_gentle_note(self, cache_key: str) -> Optional[str]:
        """Return cached gentle note string, or None on miss."""
        doc = self._col("gentle_note_cache").find_one(
            {"cache_key": cache_key}, {"_id": 0, "note": 1}
        )
        return doc["note"] if doc else None

    def set_gentle_note(self, cache_key: str, note: str) -> None:
        """Persist a gentle note result."""
        self._col("gentle_note_cache").update_one(
            {"cache_key": cache_key},
            {"$set": {
                "cache_key": cache_key,
                "note":      note,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }},
            upsert=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # COMPOSITE FOOD CACHE
    # ══════════════════════════════════════════════════════════════════════════

    def get_composite_food_name(self, cache_key: str) -> Optional[str]:
        """
        Return the user's original food name for a composite parse result, or None.
        e.g. "chicken biryani" when usda_ids=[171477, 169708].
        """
        doc = self._col("composite_food_cache").find_one(
            {"cache_key": cache_key}, {"_id": 0, "user_given_name": 1}
        )
        return doc["user_given_name"] if doc else None

    def set_composite_food_name(
        self,
        cache_key:       str,
        user_given_name: str,
        usda_ids:        list,
    ) -> None:
        """
        Store the mapping: parse cache_key → user's original food name + usda_ids.
        Called when a single food text resolves to 2+ USDA IDs (composite dish).
        """
        self._col("composite_food_cache").update_one(
            {"cache_key": cache_key},
            {"$set": {
                "cache_key":       cache_key,
                "user_given_name": user_given_name,
                "usda_ids":        usda_ids,
                "cached_at":       datetime.now(timezone.utc).isoformat(),
            }},
            upsert=True,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
db = _MongoClient()


# ─────────────────────────────────────────────────────────────────────────────
# MongoDB-backed UserMemoryStore (replaces in-memory store from user_symptom_memory.py)
# ─────────────────────────────────────────────────────────────────────────────

class MongoUserMemoryStore(UserMemoryStore):
    """
    Drop-in replacement for the in-memory UserMemoryStore.
    Persists every UserMemory to the `user_memories` MongoDB collection.
    """

    def load(self, user_id: str) -> UserMemory:
        return db.load_memory(user_id)

    def save(self, memory: UserMemory) -> None:
        db.save_memory(memory)

    def exists(self, user_id: str) -> bool:
        return db.memory_exists(user_id)

    def summary(self, user_id: str) -> dict:
        if not self.exists(user_id):
            return {"user_id": user_id, "status": "no data yet"}
        m = self.load(user_id)
        learned = [
            {
                "food_symptom_pair":     k,
                "causation_probability": round(v.posterior_mean, 3),
                "observations":          v.observations,
                "confirmations":         v.confirmations,
                "confidence":            round(v.confidence, 3),
                "last_updated":          v.last_updated,
            }
            for k, v in m.priors.items()
            if v.observations > 0
        ]
        learned.sort(key=lambda x: x["causation_probability"], reverse=True)
        return {
            "user_id":                m.user_id,
            "total_food_logs":        m.total_food_logs,
            "total_symptom_logs":     m.total_symptom_logs,
            "personalisation_weight": m.personalisation_weight,
            "model_weight":           m.model_weight,
            "learned_pairs":          len(learned),
            "top_sensitivities":      learned[:5],
        }
