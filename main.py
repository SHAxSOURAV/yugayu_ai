"""
main.py
───────
Gut Health API — 14 endpoints.

Startup: < 2 seconds (was 60 s). No model downloads. No RAM for ML weights.
AI:      Claude API (claude-sonnet-4-6) via ANTHROPIC key from .env
USDA:    FoodData Central REST API via USDA_API_KEY from .env
"""

from __future__ import annotations

import json
import hashlib
import re
import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from bisect import bisect_left, bisect_right
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

_MOCK_MODE: bool = os.getenv("MOCK_MODE", "false").lower() == "true"

import anthropic
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scorer import DigestiveInput, DigestiveResult, calculate_score
from text_context_parser import utc_now_iso
from culprit_food_finder import find_culprit_foods, CulpritResult, CulpritFood

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# App state
# ─────────────────────────────────────────────────────────────────────────────

class _State:
    claude_client                  = None   # anthropic.Anthropic
    usda_client                    = None   # USDAClient

    usda_ready:       bool         = False
    usda_error:       Optional[str] = None
    text_to_usda                   = None   # callable

    nutrition_ready:  bool         = False
    nutrition_error:  Optional[str] = None
    analyse_food_log_batch         = None
    analyse_symptom_log            = None

    predictor_ready:  bool         = False
    predictor_error:  Optional[str] = None
    predict_food_symptom_causes    = None

    memory_store                   = None
    auto_update_from_logs          = None

    recommender_ready: bool        = False
    recommender_error: Optional[str] = None
    recommend_safe_foods           = None

    meal_forecast_ready: bool      = False
    meal_forecast_error: Optional[str] = None
    forecast_meal_symptoms         = None

    analyse_symptom_note           = None

    _mongo_db                      = None

_state = _State()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — all startup logic
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()

    if _MOCK_MODE:
        log.info("⚠️  MOCK_MODE=true — Claude/AI endpoints return fake data. USDA still uses real API.")
        # USDA is free — always init it even in mock mode so real food data is available
        usda_key = os.getenv("USDA_API_KEY")
        try:
            from usda_client import USDAClient
            _state.usda_client = USDAClient(api_key=usda_key, mongo_col=None)
            _state.usda_ready  = True
            log.info("USDA client ready (mock mode — no MongoDB cache).")
        except Exception as exc:
            log.warning(f"USDA client failed in mock mode: {exc}")
        yield
        log.info("Shutdown (mock mode).")
        return


    # ── 1. Claude client ──────────────────────────────────────────────────────
    claude_key = os.getenv("CLAUDE_API_key")
    if not claude_key:
        log.error("CLAUDE_API_key is missing from .env — all AI features will fail.")
    else:
        _state.claude_client = anthropic.Anthropic(api_key=claude_key)
        log.info("Claude client ready.")


    # ── 2. USDA client ────────────────────────────────────────────────────────
    usda_key = os.getenv("USDA_API_KEY")
    
    try:
        from usda_client import USDAClient
        from database import db as _mongo_db_ref
        # Try to connect to MongoDB to get the cache collection; gracefully fall back
        try:
            _mongo_db_ref.connect()
            _state._mongo_db = _mongo_db_ref
            usda_col        = _mongo_db_ref._col("usda_cache")
            usda_search_col = _mongo_db_ref._col("usda_search_cache")
            log.info("MongoDB connected. USDA nutrient + search cache enabled.")
        except Exception as mongo_exc:
            log.warning(f"MongoDB unavailable ({mongo_exc}) — USDA cache disabled, in-process cache only.")
            usda_col        = None
            usda_search_col = None

        _state.usda_client = USDAClient(
            api_key    = usda_key,
            mongo_col  = usda_col,
            search_col = usda_search_col,
        )
        _state.usda_ready  = True
        log.info("USDA client ready.")
    except Exception as exc:
        _state.usda_error = str(exc)
        log.error(f"USDA client failed: {exc}")

    # ── 3. Food text-to-USDA pipeline ────────────────────────────────────────
    try:
        import food_text_to_usda as _ft
        _ft.init(_state.claude_client, _state.usda_client,
                 mongo_db=getattr(_state, "_mongo_db", None))
        _state.text_to_usda = _ft.text_to_usda
        log.info("food_text_to_usda ready.")
    except Exception as exc:
        _state.usda_error = str(exc)
        log.error(f"food_text_to_usda failed: {exc}")

    # ── 4. Nutrition scorer ───────────────────────────────────────────────────
    try:
        import nutrition_scorer as _ns
        _ns.init(_state.claude_client, _state.usda_client,
                 mongo_db=getattr(_state, "_mongo_db", None))
        _state.analyse_food_log_batch = _ns.analyse_food_log_batch
        _state.analyse_symptom_log    = _ns.analyse_symptom_log
        _state.nutrition_ready        = True
        log.info("nutrition_scorer ready.")
    except Exception as exc:
        _state.nutrition_error = str(exc)
        log.error(f"nutrition_scorer failed: {exc}")

    # ── 5. Food-symptom predictor ─────────────────────────────────────────────
    try:
        import food_symptom_predictor as _fsp
        _fsp.init(_state.claude_client, _state.usda_client)
        _state.predict_food_symptom_causes = _fsp.predict_food_symptom_causes
        _state.predictor_ready             = True
        log.info("food_symptom_predictor ready.")
    except Exception as exc:
        _state.predictor_error = str(exc)
        log.error(f"food_symptom_predictor failed: {exc}")

    # ── 6. MongoDB + Bayesian memory store ────────────────────────────────────
    if _state._mongo_db is not None:
        try:
            from database import MongoUserMemoryStore
            from user_symptom_memory import auto_update_from_logs
            _state.memory_store          = MongoUserMemoryStore()
            _state.auto_update_from_logs = auto_update_from_logs
            log.info("MongoUserMemoryStore ready.")
        except Exception as exc:
            log.warning(f"MongoUserMemoryStore failed ({exc}) — falling back to in-memory.")

    if _state.memory_store is None:
        try:
            from user_symptom_memory import UserMemoryStore, auto_update_from_logs
            _state.memory_store          = UserMemoryStore()
            _state.auto_update_from_logs = auto_update_from_logs
            log.info("In-memory UserMemoryStore active (no persistence).")
        except Exception as exc:
            log.error(f"Memory store failed: {exc}")

    # ── 7. Food recommender ───────────────────────────────────────────────────
    try:
        import food_recommender as _fr
        _fr.init(_state.claude_client, _state.usda_client)
        _state.recommend_safe_foods = _fr.recommend_safe_foods
        _state.recommender_ready    = True
        log.info("food_recommender ready.")
    except Exception as exc:
        _state.recommender_error = str(exc)
        log.error(f"food_recommender failed: {exc}")

    # ── 8. Meal symptom forecast ──────────────────────────────────────────────
    try:
        import meal_symptom_forecast as _msf
        _msf.init(_state.usda_client)
        _state.forecast_meal_symptoms = _msf.forecast_meal_symptoms
        _state.meal_forecast_ready    = True
        log.info("meal_symptom_forecast ready.")
    except Exception as exc:
        _state.meal_forecast_error = str(exc)
        log.error(f"meal_symptom_forecast failed: {exc}")

    # ── 9. Symptom note analyser (keyword-based, no model) ────────────────────
    try:
        import symptom_note_analyser as _sna
        _sna.init_classifier(None)   # no-op; keyword matching only
        _state.analyse_symptom_note = _sna.analyse_symptom_note
        log.info("symptom_note_analyser ready.")
    except Exception as exc:
        log.warning(f"symptom_note_analyser failed ({exc}) — notes stored but not scored.")

    # ── 10. Food tag classifier ───────────────────────────────────────────────
    try:
        import food_tag_classifier as _ftc
        _ftc.init(_state.claude_client, _state.usda_client)
        log.info("food_tag_classifier ready.")
    except Exception as exc:
        log.warning(f"food_tag_classifier failed: {exc}")

    # ── 11. Diet symptom risk ─────────────────────────────────────────────────
    try:
        import diet_symptom_risk as _dsr
        _dsr.init(_state.claude_client)
        log.info("diet_symptom_risk ready.")
    except Exception as exc:
        log.warning(f"diet_symptom_risk failed: {exc}")

    log.info(f"✅ All modules ready in {time.time()-t0:.1f}s")
    yield
    log.info("Shutdown.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

# On Railway, RAILWAY_PUBLIC_DOMAIN is set automatically (e.g. "your-app.up.railway.app").
# FastAPI needs to know this so the Swagger UI sends requests to the correct
# HTTPS URL instead of the internal http://0.0.0.0:PORT — which browsers block
# as mixed content, causing the "Failed to fetch" error in Swagger.
_railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN")
_servers = (
    [{"url": f"https://{_railway_domain}", "description": "Production (Railway)"}]
    if _railway_domain else []
)

app = FastAPI(
    title    = "Gut Health API",
    version  = "4.0.0",
    lifespan = lifespan,
    servers  = _servers or None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# MOCK MODE — one fake-data function per endpoint (activated by MOCK_MODE=true)
# ─────────────────────────────────────────────────────────────────────────────

def _mk_mock_score():
    from scorer import ScoreBreakdown, Concern, Recommendation, DigestiveResult
    return DigestiveResult(
        score=72, grade="Good",
        tagline="[MOCK] Minor improvements in diet or lifestyle could make a real difference.",
        breakdown=ScoreBreakdown(base_score=100, age_penalty=4, sleep_penalty=6,
                                  weight_penalty=0, gender_penalty=3, food_penalty=5,
                                  symptom_penalty=6, compound_penalty=4, final_score=72),
        concerns=[Concern(category="lifestyle",
                           description="[MOCK] Sample gut-health concern for demo purposes.")],
        recommendations=[Recommendation(priority=1,
                                         advice="[MOCK] Stay hydrated: 2–2.5 L water per day.")],
    )


def _mk_mock_log_food(req):
    from main import FoodLogResponse, FoodLogItem  # forward-ref; use local names below
    return {
        "updated_score":    max(40, min(99, req.current_score + 3)),
        "food_detected":    2,
        "logged_at":        utc_now_iso(),
        "normalised_names": ["Chicken", "Rice"],
        "results": [
            {"normalised_name": "Chicken", "usda_id": 171477, "weight_g": 150.0},
            {"normalised_name": "Rice",    "usda_id": 169704, "weight_g": 200.0},
        ],
        "meal_type": req.meal_type,
    }


def _mk_mock_log_symptom(req):
    deduped = list(dict.fromkeys(req.symptoms))
    return {
        "updated_score":     max(0, min(100, req.current_score - 8)),
        "detected_symptoms": deduped,
        "logged_at":         utc_now_iso(),
    }


def _mk_mock_food_parse(req):
    return {
        "Food_detected":    2,
        "Logged_at":        utc_now_iso(),
        "normalised_names": ["Chicken", "Broccoli"],
        "results": [
            {"normalised_name": "Chicken",  "usda_id": 171477, "weight_g": 150.0},
            {"normalised_name": "Broccoli", "usda_id": 169967, "weight_g": 100.0},
        ],
        "meal_type":    "Lunch",
        "score_impact": 3  if req.current_score is not None else None,
        "updated_score": max(40, min(99, (req.current_score or 70) + 3))
                         if req.current_score is not None else None,
    }


def _mk_mock_food_lookup(req):
    items = []
    for item in req.usda_ids:
        items.append({
            "usda_id":         item.usda_id,
            "normalised_name": f"MOCK Food #{item.usda_id}",
            "weight_g":        item.weight_g,
            "calories":        round(item.weight_g * 1.5, 2),
            "carbohydrate":    round(item.weight_g * 0.3, 2),
            "protein":         round(item.weight_g * 0.2, 2),
            "fat":             round(item.weight_g * 0.1, 2),
        })
    total_w = round(sum(i["weight_g"] for i in items), 1)
    return {
        "food_detected":         len(items),
        "foods_macros":          items,
        "total_calories":        round(sum(i["calories"]     or 0 for i in items), 4),
        "total_carb":            round(sum(i["carbohydrate"] or 0 for i in items), 4),
        "total_protein":         round(sum(i["protein"]      or 0 for i in items), 4),
        "total_fat":             round(sum(i["fat"]          or 0 for i in items), 4),
        "total_normalised_name": " + ".join(f"MOCK Food #{i['usda_id']}" for i in items)
                                  + f" — {total_w} g",
        "total_weight_g":        total_w,
    }


def _mk_mock_food_tags(req):
    return {
        "foods_analysed":    len(req.usda_ids),
        "categorised_count": len(req.usda_ids),
        "top_categories": [
            {"category": "Poultry Products",     "food_count": 3,
             "insight": "[MOCK] Lean protein, generally gut-friendly", "severity": "Low"},
            {"category": "Cereal Grains",        "food_count": 2,
             "insight": "[MOCK] Moderate fibre, watch portion size",   "severity": "Low"},
            {"category": "Vegetables and Products","food_count": 2,
             "insight": "[MOCK] High fibre supports gut microbiome",   "severity": "Low"},
        ],
    }


def _mk_mock_risky_food(req):
    return {
        "predictions":              {"Bloating": ["MOCK Food A"], "Heartburn": ["MOCK Food B"]},
        "food_logs_processed":      len(req.food_logs),
        "symptom_logs_processed":   len(req.symptom_logs),
        "composite_meals_detected": 1,
        "evaluated_at":             datetime.now(timezone.utc).isoformat(),
    }


def _mk_mock_safe_food(req):
    return {
        "safe_foods": [
            "[MOCK] Plain steamed rice",
            "[MOCK] Boiled chicken breast",
            "[MOCK] Banana",
            "[MOCK] Plain oatmeal",
            "[MOCK] Boiled sweet potato",
        ][:req.n],
        "foods_analysed":           len(req.food_logs),
        "composite_meals_detected": 0,
        "symptoms_considered":      len(req.symptom_logs),
        "source_note": "[MOCK] Fake recommendations — enable real mode by removing MOCK_MODE=true.",
    }


def _mk_mock_triggers_food(req):
    return {
        "symptom_name":  req.symptom_name,
        "trigger_foods": req.food_name,
        "insight":       f"[MOCK] {', '.join(req.food_name[:2])} commonly aggravate {req.symptom_name} in sensitive individuals.",
    }


def _mk_mock_feedback(req):
    return {
        "user_id":                req.user_id,
        "usda_id":                req.usda_id,
        "symptom":                req.symptom,
        "confirmed":              req.confirmed,
        "updated_prior":          0.70 if req.confirmed else 0.30,
        "prior_observations":     5,
        "prior_confirmations":    3 if req.confirmed else 1,
        "prior_confidence":       0.55,
        "personalisation_weight": 0.40,
        "message":                "[MOCK] Feedback recorded successfully.",
    }


def _mk_mock_forecast(req):
    return {
        "user_id":              req.user_id,
        "proposed_foods_count": len(req.proposed_foods),
        "food_logs_used":       10,
        "symptom_logs_used":    5,
        "personalised":         False,
        "personalisation_weight": 0.20,
        "high_risk_symptoms":   ["Bloating"],
        "medium_risk_symptoms": ["Gas"],
        "forecasts": [
            {
                "symptom": "Bloating", "risk_score": 0.62, "risk_level": "High",
                "risk_pct": "62%", "top_trigger_food": "MOCK Food",
                "top_trigger_usda_id": req.proposed_foods[0].usda_id,
                "top_trigger_score": 0.62, "top_risk_nutrients": ["sodium", "fat"],
                "per_food_scores": [], "personalised": False,
                "explanation": "[MOCK] HIGH risk of Bloating (62%).",
            },
            {
                "symptom": "Gas", "risk_score": 0.42, "risk_level": "Medium",
                "risk_pct": "42%", "top_trigger_food": "MOCK Food",
                "top_trigger_usda_id": req.proposed_foods[0].usda_id,
                "top_trigger_score": 0.42, "top_risk_nutrients": ["fiber"],
                "per_food_scores": [], "personalised": False,
                "explanation": "[MOCK] MODERATE risk of Gas (42%).",
            },
        ],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def _mk_mock_culprit(req):
    return {
        "total_foods":    len(req.food_logs),
        "total_symptoms": len(req.symptom_logs),
        "culprit_foods": [
            {
                "usda_id": req.food_logs[0].usda_id if req.food_logs else 0,
                "food_name": "[MOCK] Food A", "score": 0.75,
                "confidence": "High", "occurrence_count": 3,
                "linked_symptoms": ["Bloating", "Gas"], "top_symptom": "Bloating",
            },
        ],
        "summary": "[MOCK] 1 culprit food identified in demo mode.",
    }


def _mk_mock_learning_summary(user_id):
    return {
        "user_id": user_id, "total_food_logs": 42, "total_symptom_logs": 12,
        "total_log_entries": 54, "personalisation_weight": 0.45, "model_weight": 0.55,
        "personalisation_stage": "[MOCK] Learning — personal patterns emerging",
        "learned_pairs": 3,
        "top_sensitivities": [
            {
                "food_symptom_pair": "Dairy → Bloating",
                "causation_probability": 0.78, "observations": 8,
                "confirmations": 6, "confidence": 0.70,
                "last_updated": utc_now_iso(),
            },
        ],
        "learning_message": "[MOCK] 3 food→symptom patterns identified.",
    }


def _mk_mock_dashboard(user_id):
    return {
        "user_id": user_id, "name": "Demo User", "current_score": 72, "grade": "Good",
        "food_logs_this_week": 14, "symptom_logs_this_week": 3,
        "unique_foods_eaten": ["Chicken", "Rice", "Broccoli", "Banana"],
        "symptom_frequency": {"Bloating": 2, "Gas": 1},
        "score_history": [
            {"date": "2026-04-07", "score": 68, "event": "food"},
            {"date": "2026-04-10", "score": 72, "event": "food"},
        ],
        "personalisation_weight": 0.45,
        "personalisation_stage": "[MOCK] Learning — personal patterns emerging",
        "top_sensitivities": [
            {"pair": "Dairy → Bloating", "probability": 0.78, "observations": 8},
        ],
    }


def _mk_mock_db_health():
    return {
        "status":   "mock",
        "note":     "MOCK_MODE=true — no real database connection. Remove MOCK_MODE to connect.",
        "database": "gut_health_mock",
        "collections": {
            "users": 3, "food_logs": 84, "symptom_logs": 36,
            "user_memories": 3, "score_history": 12, "usda_cache": 150,
        },
        "users": [
            {"user_id": "user_anika",  "name": "Anika Rahman",   "score": 74, "grade": "Good"},
            {"user_id": "user_rafiul", "name": "Rafiul Islam",    "score": 61, "grade": "Fair"},
            {"user_id": "user_sadia",  "name": "Sadia Hossain",   "score": 55, "grade": "Fair"},
        ],
    }


def _mk_mock_barcode(req):
    return {
        "barcode": req.code.strip(),
        "product_name": f"[MOCK] Product for barcode {req.code.strip()}",
        "quantity": "250g",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grade(score: int) -> str:
    if   score >= 85: return "Excellent"
    elif score >= 70: return "Good"
    elif score >= 50: return "Fair"
    elif score >= 30: return "Poor"
    else:             return "Critical"

def _grade_summary(score: int) -> str:
    if   score >= 85: return "Gut health is in great shape."
    elif score >= 70: return "Minor improvements could help further."
    elif score >= 50: return "Several areas need attention."
    elif score >= 30: return "Significant digestive stress detected."
    else:             return "Please consult a healthcare professional."


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — POST /score
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/score", response_model=DigestiveResult,
          summary="Calculate onboarding digestion score (call once at onboarding)")
def score(data: DigestiveInput) -> DigestiveResult:
    if _MOCK_MODE:
        return _mk_mock_score()
    return calculate_score(data)

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — POST /log/food
# ─────────────────────────────────────────────────────────────────────────────

_VALID_MEAL_TYPES = {"Breakfast", "Lunch", "Dinner", "Snack"}
_SCORE_FLOOR      = 0
_SCORE_CEIL       = 100
_SCORE_BLEND_ALPHA = 0.70


def _blend_with_meal_quality(current_score: int, meal_quality: int) -> int:
    blended = round(
        _SCORE_BLEND_ALPHA * current_score + (1.0 - _SCORE_BLEND_ALPHA) * meal_quality
    )
    return max(_SCORE_FLOOR, min(_SCORE_CEIL, blended))


def _light_normalise_text(text: str) -> str:
    norm = text.lower().strip()
    norm = re.sub(r"[^\w\s]", " ", norm)
    norm = re.sub(r"\b(had\s+eaten|had\s+eat|ate)\b", "eat", norm)
    norm = re.sub(r"\bfried\b", "fry", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm




class FoodLogRequest(BaseModel):
    current_score: int = Field(..., ge=0, le=100,
                               description="User's current gut health score")
    meal_type:     str = Field(..., description="Breakfast | Lunch | Dinner | Snack")
    foods:         str = Field(..., min_length=2, max_length=500,
                               description="Any meal text — single or composite food",
                               examples=["chicken biriyani", "rice and egg fry"])
    quantity:      str = Field(..., min_length=1, max_length=100,
                               description="Amount the user ate",
                               examples=["400g", "1 cup", "4 pieces"])


class FoodLogItem(BaseModel):
    normalised_name: str
    usda_id:         int
    weight_g:        float


class FoodLogResponse(BaseModel):
    updated_score:        int
    food_detected:        int
    logged_at:            str
    normalised_names:     List[str]
    results:              List[FoodLogItem]
    meal_type:            str
    users_given_food_name: Optional[str] = None


@app.post(
    "/log/food",
    response_model=FoodLogResponse,
    summary="Log a meal in plain text and get an updated gut health score",
    description=(
        "Send a meal description and quantity in plain text. Claude parses and decomposes "
        "the food, looks up USDA IDs, evaluates gut health impact (meal timing, nutritional "
        "quality, gut irritants), applies a score multiplier that neutralises around 80-90, "
        "and returns the updated score with a 10-12 word note."
    ),
)
def log_food(req: FoodLogRequest) -> FoodLogResponse:
    if _MOCK_MODE:
        d = _mk_mock_log_food(req)
        return FoodLogResponse(**d)
    if req.meal_type not in _VALID_MEAL_TYPES:
        raise HTTPException(422,
            f"Invalid meal_type '{req.meal_type}'. Must be one of: {sorted(_VALID_MEAL_TYPES)}")
    if not _state.usda_ready:
        raise HTTPException(503, f"USDA pipeline unavailable: {_state.usda_error or 'unknown'}")

    # ── Step 1: parse food text via existing text_to_usda ────────────────────
    # Combine quantity + food so Claude can distribute weights intelligently
    combined_text = f"{req.quantity} {req.foods} for {req.meal_type.lower()}"
    try:
        raw = _state.text_to_usda(combined_text)
    except RuntimeError as exc:
        # Claude auth errors are raised as RuntimeError by food_text_to_usda
        raise HTTPException(503, str(exc))
    except Exception as exc:
        log.exception(f"log/food parse error: {combined_text!r}")
        raise HTTPException(500, f"Food parsing error: {exc}")

    if not raw:
        raise HTTPException(
            422,
            "No recognisable food found in the description. "
            "If this keeps happening check your Claude API key is valid and "
            "matches the variable name in your .env (Claude_API_key or CLAUDE_API_KEY).",
        )

    logged_at        = raw[0].get("logged_at", utc_now_iso())
    normalised_names = [r["normalised_name"].split(",")[0].strip() for r in raw]

    # ── Step 2: score the meal via USDA-only deterministic logic ─────────────
    meal_quality = 50
    if _state.nutrition_ready:
        try:
            import nutrition_scorer as _ns
            foods_for_scoring = [
                {"usda_id": r["usda_id"], "weight_g": r["weight_g"]}
                for r in raw
            ]
            meal_quality, _ = _ns.score_meal_usda(
                foods=foods_for_scoring,
                meal_type=req.meal_type,
            )
        except Exception as exc:
            log.warning(f"log/food scoring failed: {exc}")

    # ── Step 3: blend current score with USDA meal quality ───────────────────
    updated_score = _blend_with_meal_quality(req.current_score, meal_quality)

    # ── Step 4: composite food detection + cache ──────────────────────────────
    # All items from one parse call share the same logged_at — if there are 2+
    # items they came from decomposing one composite dish (e.g. "chicken biryani").
    # Store the user's original food name so future responses can surface it.
    users_given_food_name: Optional[str] = None
    mongo_db = getattr(_state, "_mongo_db", None)
    if mongo_db is not None:
        cache_key = hashlib.sha256(
            _light_normalise_text(combined_text).encode()
        ).hexdigest()
        try:
            if len(raw) >= 2:
                mongo_db.set_composite_food_name(
                    cache_key       = cache_key,
                    user_given_name = req.foods.strip(),
                    usda_ids        = [r["usda_id"] for r in raw],
                )
            users_given_food_name = mongo_db.get_composite_food_name(cache_key)
        except Exception as exc:
            log.warning(f"composite_food_cache error: {exc}")

    return FoodLogResponse(
        updated_score         = updated_score,
        food_detected         = len(raw),
        logged_at             = logged_at,
        normalised_names      = normalised_names,
        results               = [
            FoodLogItem(
                normalised_name = r["normalised_name"].split(",")[0].strip(),
                usda_id         = r["usda_id"],
                weight_g        = r["weight_g"],
            )
            for r in raw
        ],
        meal_type             = req.meal_type,
        users_given_food_name = users_given_food_name,
    )
# ENDPOINT 4 — POST /log/symptom
# ─────────────────────────────────────────────────────────────────────────────

_VALID_SYMPTOMS   = {"Bloating","Abdominal Pain","Nausea","Constipation","Heartburn",
                     "Gas","Fatigue","Acid Reflux","Cramps","Diarrhea"}
_VALID_SEVERITIES = {"Mild","Moderate","Severe"}


class SymptomLogRequest(BaseModel):
    current_score: int         = Field(..., ge=0, le=100,
                                       description="User's current digestion score (0–100).")
    symptoms:      List[str]   = Field(..., min_length=1, max_length=12,
                                       description="One or more symptoms. Max 12.",
                                       examples=[["Bloating", "Heartburn"]])
    severity:      str         = Field(..., description="Mild | Moderate | Severe",
                                       examples=["Moderate"])
    note:          Optional[str] = Field(default=None, max_length=1000,
                                         description="Optional free-text description.")
    logged_at:     Optional[datetime] = Field(default=None,
                                              description="When the symptom occurred. Defaults to now.")


class SymptomLogResponse(BaseModel):
    updated_score:      int        = Field(..., description="Score after penalty applied (0–100).")
    detected_symptoms:  List[str]  = Field(..., description="Validated, deduplicated symptom list.")
    logged_at:          str        = Field(..., description="UTC ISO-8601 timestamp used for this entry.")


@app.post("/log/symptom", response_model=SymptomLogResponse,
          summary="Log symptoms and get an updated digestion score")
def log_symptom(req: SymptomLogRequest) -> SymptomLogResponse:
    """
    Log one or more gut symptoms and receive an **updated digestion score**.

    Claude scores the full symptom picture in one call — taking into account
    symptom count, severity, and the optional free-text note together — and
    returns a penalty in [0, 40] that is subtracted from the current score.

    **Penalty scale (approximate):**

    | Situation | Penalty |
    |---|---|
    | 1 symptom, Mild, no note | 3–5 |
    | 1 symptom, Moderate, no note | 6–9 |
    | 1 symptom, Severe, no note | 10–13 |
    | 2–3 symptoms, Moderate | 10–16 |
    | 4–6 symptoms, Moderate | 17–24 |
    | 7–9 symptoms, Severe | 25–32 |
    | 10–12 symptoms, Severe + severe note | 33–40 |

    **Allowed `symptoms`:** `Bloating` · `Abdominal Pain` · `Nausea` · `Constipation`
    · `Heartburn` · `Gas` · `Fatigue` · `Acid Reflux` · `Cramps` · `Diarrhea`

    **Allowed `severity`:** `Mild` · `Moderate` · `Severe`
    """
    if _MOCK_MODE:
        return SymptomLogResponse(**_mk_mock_log_symptom(req))
    # ── Validate ──────────────────────────────────────────────────────────────
    invalid = [s for s in req.symptoms if s not in _VALID_SYMPTOMS]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid symptom(s): {invalid}. Allowed: {sorted(_VALID_SYMPTOMS)}",
        )
    if req.severity not in _VALID_SEVERITIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid severity '{req.severity}'. Allowed: {sorted(_VALID_SEVERITIES)}",
        )

    # Deduplicate while preserving order
    seen: set[str] = set(); deduped: list[str] = []
    for s in req.symptoms:
        if s not in seen:
            seen.add(s); deduped.append(s)

    logged_at = req.logged_at or datetime.now(timezone.utc)
    logged_at_str = (
        logged_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        if hasattr(logged_at, "strftime") else str(logged_at)
    )

    # ── Score with deterministic symptom rules ───────────────────────────────
    try:
        from nutrition_scorer import score_symptom_log_rules
        penalty, _ = score_symptom_log_rules(
            symptoms  = deduped,
            severity  = req.severity,
            note      = req.note,
        )
    except Exception as exc:
        log.exception("score_symptom_log_rules failed")
        raise HTTPException(500, f"Scoring error: {exc}")

    updated_score = max(0, min(100, req.current_score - penalty))

    return SymptomLogResponse(
        updated_score     = updated_score,
        detected_symptoms = deduped,
        logged_at         = logged_at_str,
    )
# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — POST /food/parse
# ─────────────────────────────────────────────────────────────────────────────

class FoodParseRequest(BaseModel):
    text:          str = Field(..., min_length=3, max_length=1000,
                               examples=["I had eat 200g rice and egg fry in lunch"])
    current_score: Optional[int] = Field(None, ge=0, le=100,
                               description="Pass your current score to also get score_impact and updated_score")


class FoodParseItem(BaseModel):
    normalised_name: str
    usda_id:         int
    weight_g:        float


class FoodParseResponse(BaseModel):
    Food_detected:         int
    Logged_at:             str
    normalised_names:      List[str]
    results:               List[FoodParseItem]
    meal_type:             Optional[str] = None
    score_impact:          Optional[int] = None
    updated_score:         Optional[int] = None
    users_given_food_name: Optional[str] = None


@app.post(
    "/food/parse",
    response_model=FoodParseResponse,
    summary="Parse natural meal text → USDA IDs + optional gut health score",
    description=(
        "Parse any meal description. Pass current_score to also receive "
        "score_impact, updated_score, and a gut health note."
    ),
)
def food_parse(req: FoodParseRequest) -> FoodParseResponse:
    if _MOCK_MODE:
        return FoodParseResponse(**_mk_mock_food_parse(req))
    if not _state.usda_ready:
        raise HTTPException(503, f"USDA pipeline unavailable: {_state.usda_error or 'unknown'}")
    try:
        raw = _state.text_to_usda(req.text)
    except Exception as exc:
        log.exception(f"food/parse pipeline error for: {req.text!r}")
        raise HTTPException(500, f"Pipeline error: {exc}")

    if not raw:
        return FoodParseResponse(
            Food_detected=0, Logged_at=utc_now_iso(),
            normalised_names=[], results=[], meal_type=None,
        )

    logged_at        = raw[0].get("logged_at", utc_now_iso())
    meal_type        = raw[0].get("meal_type")
    normalised_names = [r["normalised_name"].split(",")[0].strip() for r in raw]

    score_impact:  Optional[int] = None
    updated_score: Optional[int] = None

    if req.current_score is not None and _state.nutrition_ready:
        try:
            import nutrition_scorer as _ns
            foods_for_scoring = [
                {"usda_id": r["usda_id"], "weight_g": r["weight_g"]}
                for r in raw
            ]
            meal_quality, _ = _ns.score_meal_usda(
                foods=foods_for_scoring,
                meal_type=meal_type or "Lunch",
            )
            updated_score = _blend_with_meal_quality(req.current_score, meal_quality)
            score_impact = updated_score - req.current_score
        except Exception as exc:
            log.warning(f"food/parse scoring failed: {exc}")

    # ── Composite food detection + cache ─────────────────────────────────────
    # 2+ items from one parse call → composite dish. Store user's original text.
    users_given_food_name: Optional[str] = None
    mongo_db = getattr(_state, "_mongo_db", None)
    if mongo_db is not None:
        parse_key = hashlib.sha256(
            _light_normalise_text(req.text).encode()
        ).hexdigest()
        try:
            if len(raw) >= 2:
                mongo_db.set_composite_food_name(
                    cache_key       = parse_key,
                    user_given_name = req.text.strip(),
                    usda_ids        = [r["usda_id"] for r in raw],
                )
            users_given_food_name = mongo_db.get_composite_food_name(parse_key)
        except Exception as exc:
            log.warning(f"food/parse composite_food_cache error: {exc}")

    return FoodParseResponse(
        Food_detected         = len(raw),
        Logged_at             = logged_at,
        normalised_names      = normalised_names,
        results               = [
            FoodParseItem(
                normalised_name = r["normalised_name"].split(",")[0].strip(),
                usda_id         = r["usda_id"],
                weight_g        = r["weight_g"],
            )
            for r in raw
        ],
        meal_type             = meal_type,
        score_impact          = score_impact,
        updated_score         = updated_score,
        users_given_food_name = users_given_food_name,
    )

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 6 — GET /food/lookup/{usda_id}
# ─────────────────────────────────────────────────────────────────────────────

def _sf(v) -> Optional[float]:
    """Safe-float: coerce to float or return None."""
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


class FoodLookupInputItem(BaseModel):
    usda_id:  int
    weight_g: float = Field(..., gt=0, description="Portion weight in grams")


class FoodLookupRequest(BaseModel):
    usda_ids: List[FoodLookupInputItem]


class FoodLookupItem(BaseModel):
    usda_id:         int
    normalised_name: str
    weight_g:        float
    calories:        Optional[float] = None
    carbohydrate:    Optional[float] = None
    protein:         Optional[float] = None
    fat:             Optional[float] = None


class FoodLookupResponse(BaseModel):
    food_detected:         int
    foods_macros:          List[FoodLookupItem]
    total_calories:        float
    total_carb:            float
    total_protein:         float
    total_fat:             float
    total_normalised_name: str
    total_weight_g:        float


@app.post("/food/lookup", response_model=FoodLookupResponse,
          summary="Lookup macros for one or multiple USDA IDs, scaled to portion weight")
def food_lookup(req: FoodLookupRequest) -> FoodLookupResponse:
    if _MOCK_MODE:
        return FoodLookupResponse(**_mk_mock_food_lookup(req))
    if not _state.usda_ready:
        raise HTTPException(503, f"USDA dataset unavailable: {_state.usda_error or 'unknown'}")
    if not req.usda_ids:
        raise HTTPException(422, "usda_ids list is empty.")

    foods_macros: List[FoodLookupItem] = []
    names: List[str] = []

    for item in req.usda_ids:
        data = _state.usda_client.get_nutrients(item.usda_id)
        if data is None:
            raise HTTPException(404, f"USDA ID {item.usda_id} not found.")

        ratio = item.weight_g / 100.0

        def _scale(key: str) -> Optional[float]:
            val = _sf(data.get(key))
            return round(val * ratio, 4) if val is not None else None

        name = data.get("description", f"USDA ID {item.usda_id}")
        names.append(name)

        foods_macros.append(
            FoodLookupItem(
                usda_id         = item.usda_id,
                normalised_name = name,
                weight_g        = item.weight_g,
                calories        = _scale("calories"),
                carbohydrate    = _scale("carbs"),
                protein         = _scale("protein"),
                fat             = _scale("total_fat"),
            )
        )

    total_weight_g = round(sum(i.weight_g for i in foods_macros), 1)
    total_normalised_name = ", ".join(n.split()[0] for n in names) + f" — {total_weight_g} g"

    return FoodLookupResponse(
        food_detected         = len(foods_macros),
        foods_macros          = foods_macros,
        total_calories        = round(sum(i.calories     or 0.0 for i in foods_macros), 4),
        total_carb            = round(sum(i.carbohydrate or 0.0 for i in foods_macros), 4),
        total_protein         = round(sum(i.protein      or 0.0 for i in foods_macros), 4),
        total_fat             = round(sum(i.fat          or 0.0 for i in foods_macros), 4),
        total_normalised_name = total_normalised_name,
        total_weight_g        = total_weight_g,
    )

# _____________________
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 7 — POST /food/tags
# ─────────────────────────────────────────────────────────────────────────────

class FoodTagsRequest(BaseModel):
    usda_ids: List[int] = Field(
        ..., min_length=1, max_length=500,
        description="USDA food IDs to analyse (e.g. all foods that caused gut problems).",
        examples=[[171477, 169704, 172687, 173944, 170379]],
    )


class CategoryItem(BaseModel):
    category:   str
    food_count: int
    insight:    str
    severity:   str   # "Low" | "Medium" | "High" — AI-generated


class FoodTagsResponse(BaseModel):
    foods_analysed:     int
    categorised_count:  int   # foods that had a USDA category
    top_categories:     List[CategoryItem]


# Claude evaluates USDA category names → insight (5-7 words) + severity
_CATEGORY_EVAL_SYSTEM = (
    "You are a clinical gut-health dietitian AI.\n\n"
    "You will receive a list of USDA food category names with their food counts.\n"
    "For each category, return:\n\n"
    "  insight  — exactly 5-7 words, specific gut effect, no punctuation at end\n"
    "             Good: 'Elevates intestinal permeability and inflammation risk'\n"
    "             Bad:  'This is bad for your gut'\n\n"
    "  severity — gut-health severity for most people:\n"
    "     'Low'    → gut-friendly or neutral (vegetables, fruits, lean proteins, whole grains)\n"
    "     'Medium' → moderate concern for sensitive individuals (dairy, eggs, legumes, nuts)\n"
    "     'High'   → significant gut irritant (fried foods, fast food, processed snacks, sweets)\n\n"
    "Return ONLY valid JSON — no markdown:\n"
    '{"results": [{"category": "<name>", "insight": "<5-7 words>", "severity": "Low|Medium|High"}, ...]}\n'
    "Include ALL categories provided, in the same order."
)


def _claude_category_eval(
    category_counts: list[tuple[str, int]],
    claude_client,
) -> dict[str, dict]:
    """
    Send USDA category names + counts to Claude.
    Returns dict: category_name → {insight, severity}
    """
    fallback = {
        cat: {"insight": "Monitor intake for gut sensitivity", "severity": "Medium"}
        for cat, _ in category_counts
    }
    if not category_counts or claude_client is None:
        return fallback

    lines = "\n".join(
        f"{i+1}. {cat} ({count} food{'s' if count != 1 else ''})"
        for i, (cat, count) in enumerate(category_counts)
    )
    try:
        msg = claude_client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 60 * len(category_counts) + 80,
            system     = _CATEGORY_EVAL_SYSTEM,
            messages   = [{"role": "user", "content": f"Evaluate these USDA food categories:\n\n{lines}"}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        results = json.loads(raw).get("results", [])
        return {
            r["category"]: {
                "insight":  r.get("insight", fallback.get(r["category"], {}).get("insight", "")),
                "severity": r.get("severity", "Medium"),
            }
            for r in results
            if "category" in r
        }
    except Exception as exc:
        log.warning(f"Claude category evaluation failed: {exc}")
        return fallback


@app.post(
    "/food/tags",
    response_model=FoodTagsResponse,
    summary="Top USDA food categories from a food list with AI-generated gut-health insight and severity",
)
def food_tags_batch(req: FoodTagsRequest) -> FoodTagsResponse:
    """
    Pass a list of USDA food IDs — typically all foods linked to a user's gut
    symptoms. Each ID is resolved via the USDA API (cache-first) to get its
    official **USDA food category** (e.g. "Dairy and Egg Products", "Fast Foods").

    The top 5 categories by frequency are then sent to **Claude** in one call,
    which returns:
    - a **5-7 word clinical gut-health insight** per category
    - a **severity rating** (Low / Medium / High) based on how harmful that
      category typically is for gut health

    Both insight and severity are fully AI-generated — nothing is hardcoded.
    """
    if _MOCK_MODE:
        return FoodTagsResponse(**_mk_mock_food_tags(req))
    if not _state.usda_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"USDA client unavailable: {_state.usda_error or 'unknown'}",
        )
    if _state.claude_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Claude client not available — check Claude_API_key in .env",
        )

    from collections import Counter
    category_counter: Counter = Counter()
    resolved = 0

    for usda_id in req.usda_ids:
        try:
            data = _state.usda_client.get_nutrients(usda_id)
            if not data:
                continue
            resolved += 1
            cat = data.get("food_category")
            if cat:
                category_counter[cat] += 1
        except Exception as exc:
            log.warning(f"food_tags_batch: USDA lookup failed for {usda_id}: {exc}")

    if not category_counter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No USDA food categories found for the provided IDs. "
                "This usually means the foods are not yet in the USDA cache — "
                "try fetching them via /food/lookup/{usda_id} first."
            ),
        )

    top5: list[tuple[str, int]] = category_counter.most_common(5)
    claude_data = _claude_category_eval(top5, _state.claude_client)

    return FoodTagsResponse(
        foods_analysed    = len(req.usda_ids),
        categorised_count = sum(category_counter.values()),
        top_categories    = [
            CategoryItem(
                category   = cat,
                food_count = count,
                insight    = claude_data.get(cat, {}).get("insight", ""),
                severity   = claude_data.get(cat, {}).get("severity", "Medium"),
            )
            for cat, count in top5
        ],
    )

from food_symptom_predictor import (FoodLogEntry, SymptomLogEntry, FoodCausationResult,SymptomPrediction, SYMPTOM_WINDOWS,predict_causation_by_time,group_composite_meals,)

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 8 — POST /predict/food-symptom
# ─────────────────────────────────────────────────────────────────────────────

class FoodLogInput(BaseModel):
    usda_id:   int
    weight_g:  float = Field(..., gt=0)
    logged_at: datetime
 
 
class SymptomLogInput(BaseModel):
    symptom:   str
    intensity: str   # Mild | Moderate | Severe  (normalised before use)
    logged_at: datetime
 
 
# ── Shared helper: usda_id → short display name ───────────────────────────────
 
def _usda_id_to_short_name(usda_id: int) -> str:
    """
    Memory cache → MongoDB → USDA API (in that order).
    Returns first comma-segment of the USDA description, e.g.:
        "Chicken, broiler, breast, cooked" → "Chicken"
    """
    if _state.usda_client is None:
        return f"Food #{usda_id}"
    full = _state.usda_client.get_description(usda_id)
    return full.split(",")[0].strip()
 
 
# ── Shared helper: build food_logs_named list from request ────────────────────
 
def _resolve_food_logs(food_logs: list[FoodLogInput]) -> list[dict]:
    """
    Convert a list of FoodLogInput (with usda_id) to the named format
    expected by predict_causation_by_time and recommend_safe_from_logs.
    Builds a usda_id name-cache to avoid duplicate lookups.
    """
    name_cache: dict[int, str] = {}
    for fl in food_logs:
        if fl.usda_id not in name_cache:
            name_cache[fl.usda_id] = _usda_id_to_short_name(fl.usda_id)
 
    return [
        {
            "food_name": name_cache[fl.usda_id],
            "usda_id":   fl.usda_id,          # required by nutrient danger scoring
            "weight_g":  fl.weight_g,
            "logged_at": fl.logged_at,
        }
        for fl in food_logs
    ]


class FoodNameLogInput(BaseModel):
    food_name: str = Field(..., min_length=1, max_length=300)
    weight_g: float = Field(..., gt=0)
    logged_at: datetime


def _clean_food_label(name: str) -> str:
    base = " ".join(name.strip().split())
    if not base:
        return "Unknown food"
    return base[0].upper() + base[1:]


def _natural_join(items: list[str]) -> str:
    parts = [p for p in items if p]
    if not parts:
        return "No food"
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


_FAST_ENDPOINT_CACHE_TTL_SEC = 300
_FAST_ENDPOINT_CACHE: dict[str, tuple[float, dict]] = {}


def _endpoint_cache_get(key: str) -> Optional[dict]:
    hit = _FAST_ENDPOINT_CACHE.get(key)
    if not hit:
        return None
    expires_at, payload = hit
    if time.time() > expires_at:
        _FAST_ENDPOINT_CACHE.pop(key, None)
        return None
    return payload


def _endpoint_cache_set(key: str, payload: dict) -> None:
    _FAST_ENDPOINT_CACHE[key] = (time.time() + _FAST_ENDPOINT_CACHE_TTL_SEC, payload)


def _request_cache_key(
    endpoint: str,
    food_logs: list,
    symptom_logs: list,
    n: Optional[int] = None,
) -> str:
    compact = {
        "endpoint": endpoint,
        "n": n,
        "food_logs": [
            {
                "food_name": _light_normalise_text(getattr(f, "food_name", "")),
                "weight_g": round(float(getattr(f, "weight_g", 0)), 2),
                "logged_at": getattr(f, "logged_at").isoformat(),
            }
            for f in food_logs
        ],
        "symptom_logs": [
            {
                "symptom": getattr(s, "symptom", ""),
                "intensity": _normalise_intensity(getattr(s, "intensity", "")),
                "logged_at": getattr(s, "logged_at").isoformat(),
            }
            for s in symptom_logs
        ],
    }
    blob = json.dumps(compact, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _looks_like_sentence_meal_text(name: str) -> bool:
    txt = _light_normalise_text(name)
    words = txt.split()
    if len(words) < 5:
        return False
    markers = {"had", "eat", "with", "and", "for", "breakfast", "lunch", "dinner", "snack"}
    return any(m in words for m in markers)


def _expand_food_name_log(fl: FoodNameLogInput, mongo_db) -> list[dict]:
    """
    Accept either plain food names or sentence-style meal text.
    Expands sentence input via existing parser and resolves each food to USDA ids.
    """
    if _state.usda_ready and _state.text_to_usda is not None and _looks_like_sentence_meal_text(fl.food_name):
        try:
            parsed = _state.text_to_usda(fl.food_name)
            if parsed:
                per_weight = fl.weight_g / max(1, len(parsed))
                expanded: list[dict] = []
                for item in parsed:
                    clean = _clean_food_label(item.get("raw_food") or item.get("normalised_name") or fl.food_name)
                    expanded.append({
                        "food_name": clean,
                        "usda_id": int(item.get("usda_id", 0)),
                        "weight_g": per_weight,
                        "logged_at": fl.logged_at,
                        "display_name": clean,
                    })
                return expanded
        except Exception as exc:
            log.warning(f"food_name sentence parse fallback for '{fl.food_name}': {exc}")

    usda_id, _ = _resolve_food_name_to_usda(fl.food_name, mongo_db)
    clean = _clean_food_label(fl.food_name)
    return [{
        "food_name": clean,
        "usda_id": usda_id or 0,
        "weight_g": fl.weight_g,
        "logged_at": fl.logged_at,
        "display_name": clean,
    }]


def _expand_food_name_logs_batch(food_logs: list[FoodNameLogInput], mongo_db) -> list[dict]:
    """
    Fast path:
    - Resolve unique simple food names once (in parallel).
    - Parse sentence-style logs only when needed.
    """
    if not food_logs:
        return []

    simple_keys: dict[str, str] = {}
    sentence_logs: list[FoodNameLogInput] = []
    for fl in food_logs:
        raw = " ".join(fl.food_name.split())
        if _looks_like_sentence_meal_text(raw):
            sentence_logs.append(fl)
        else:
            key = _light_normalise_text(raw)
            if key and key not in simple_keys:
                simple_keys[key] = raw

    resolved_simple: dict[str, tuple[int, str]] = {}
    if simple_keys:
        def _resolve_one(item: tuple[str, str]) -> tuple[str, tuple[int, str]]:
            key, raw = item
            usda_id, _ = _resolve_food_name_to_usda(raw, mongo_db)
            clean = _clean_food_label(raw)
            return key, (usda_id or 0, clean)

        with ThreadPoolExecutor(max_workers=min(8, max(2, len(simple_keys)))) as ex:
            for key, value in ex.map(_resolve_one, simple_keys.items()):
                resolved_simple[key] = value

    expanded: list[dict] = []
    for fl in food_logs:
        key = _light_normalise_text(fl.food_name)
        if _looks_like_sentence_meal_text(fl.food_name):
            expanded.extend(_expand_food_name_log(fl, mongo_db))
            continue
        usda_id, clean = resolved_simple.get(key, (0, _clean_food_label(fl.food_name)))
        expanded.append({
            "food_name": clean,
            "usda_id": usda_id,
            "weight_g": fl.weight_g,
            "logged_at": fl.logged_at,
            "display_name": clean,
        })
    return expanded
 
 
# ── Shared helper: normalise intensity case ───────────────────────────────────
 
_INTENSITY_NORMALISE = {v.lower(): v for v in ("Mild", "Moderate", "Severe")}
 
def _normalise_intensity(raw: str) -> str:
    """'mild' → 'Mild', 'MODERATE' → 'Moderate', 'high' → kept as-is."""
    return _INTENSITY_NORMALISE.get(raw.lower(), raw)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — ENDPOINT 8  POST /predict/food-symptom
# ENDPOINT — POST /recommend/safe_food
# ─────────────────────────────────────────────────────────────────────────────

from food_recommender import recommend_safe_from_logs, safe_history_foods_only


# Shared request model — used by /recommend/risky_food
class FoodAnalysisRequest(BaseModel):
    food_logs:    List[FoodLogInput]    = Field(..., min_length=1,
                                               description="Food entries (usda_id + weight_g + logged_at)")
    symptom_logs: List[SymptomLogInput] = Field(default_factory=list,
                                               description="Symptom entries (symptom + intensity + logged_at)")
    n:            int                   = Field(default=5, ge=1, le=10,
                                               description="Number of safe food recommendations (safe_food only)")


class SafeFoodRequest(BaseModel):
    food_logs:    List[FoodNameLogInput] = Field(
        default_factory=list,
        description="Food entries (food_name + weight_g + logged_at)",
    )
    symptom_logs: List[SymptomLogInput] = Field(default_factory=list)
    n:            int = Field(default=5, ge=1, le=20)
 
 
class SafeFoodResponse(BaseModel):
    safe_foods:              List[str]
    foods_analysed:          int
    composite_meals_detected: int
    symptoms_considered:     int
    source_note:             str
 
 
@app.post(
    "/recommend/safe_food",
    response_model=SafeFoodResponse,
    summary="Get safe food recommendations based on your food and symptom history",
    description=(
        "Pass your food logs (food_name + weight_g + logged_at) and symptom logs. "
        "Foods logged at the same timestamp are grouped as composite meals. "
        "Deterministic digestion-window logic identifies foods from your history that did NOT "
        "trigger symptoms and returns up to top 10 when enough safe-history data exists. "
        "Returns a list of 5 safe food names. "
        "Intensity is case-insensitive."
    ),
)
def recommend_safe_foods_endpoint(req: SafeFoodRequest) -> SafeFoodResponse:
 
    if _MOCK_MODE:
        return SafeFoodResponse(**_mk_mock_safe_food(req))
    # ── Validate ──────────────────────────────────────────────────────────────
    if not _state.recommender_ready:
        raise HTTPException(503, f"Recommender unavailable: {_state.recommender_error or 'unknown'}")
    if not _state.usda_ready:
        raise HTTPException(503, "USDA client unavailable — cannot resolve food names.")
 
    for sl in req.symptom_logs:
        normalised = _normalise_intensity(sl.intensity)
        if normalised not in _VALID_PREDICT_SEVERITIES:
            raise HTTPException(
                422,
                f"Invalid intensity '{sl.intensity}'. "
                f"Accepted: Mild, Moderate, Severe (case-insensitive).",
            )
 
    cache_key = _request_cache_key("recommend_safe_food", req.food_logs, req.symptom_logs, req.n)
    cached_payload = _endpoint_cache_get(cache_key)
    if cached_payload is not None:
        return SafeFoodResponse(**cached_payload)
 
    mongo_db = getattr(_state, "_mongo_db", None)
    food_logs_named = _expand_food_name_logs_batch(req.food_logs, mongo_db)
 
    symptom_logs_plain = [
        {
            "symptom":   sl.symptom,
            "intensity": _normalise_intensity(sl.intensity),
            "logged_at": sl.logged_at,
        }
        for sl in req.symptom_logs
    ]
 
    # ── Count composites for metadata ─────────────────────────────────────────
    grouped       = group_composite_meals(food_logs_named)
    composite_cnt = sum(1 for m in grouped if m["is_composite"])
 
    # ── History-only safe foods (count < 3 associations) ───────────────────────
    try:
        # Respect user's n, cap at 20
        n_limit = min(req.n, 20)
        safe_foods = recommend_safe_from_logs(
            food_logs_named=food_logs_named,
            symptom_logs=symptom_logs_plain,
            n=n_limit,
        )
    except Exception as exc:
        log.exception("recommend/safe-foods — recommendation failed")
        raise HTTPException(500, f"Recommendation error: {exc}")

    # ── Source Note Logic ─────────────────────────────────────────────────────
    # User requirements:
    # 1. if no food log: "You dont have enough logs, add more food and symptom to find safe food."
    # 2. if safe food list < 5: "add more food and symptom logs to find more safe food."
    # 3. else: default note
    if not req.food_logs:
        source_note = "You dont have enough logs, add more food and symptom to find safe food."
    elif len(safe_foods) < 5:
        source_note = "add more food and symptom logs to find more safe food."
    else:
        source_note = (
            "Recommendations based on foods logged at least 3 times with zero symptom associations."
            + (f" {composite_cnt} composite meal(s) detected." if composite_cnt else "")
        )

    response_payload = {
        "safe_foods": safe_foods,
        "foods_analysed": len(req.food_logs),
        "composite_meals_detected": composite_cnt,
        "symptoms_considered": len(req.symptom_logs),
        "source_note": source_note,
    }
    _endpoint_cache_set(cache_key, response_payload)
    return SafeFoodResponse(**response_payload)
 
# ══════════════════════════════════════════════════════════════════════════════
 
class FoodSymptomPredictRequest(BaseModel):
    food_logs:    List[FoodNameLogInput] = Field(
        ..., min_length=1,
        description="Food entries (food_name + weight_g + logged_at)",
    )
    symptom_logs: List[SymptomLogInput] = Field(default_factory=list)
 
 
class FoodSymptomPredictResponse(BaseModel):
    predictions:             dict[str, List[str]]
    food_logs_processed:     int
    symptom_logs_processed:  int
    composite_meals_detected: int
    evaluated_at:            str
 
 
_VALID_PREDICT_SEVERITIES = {"Mild", "Moderate", "Severe"}
 
 
@app.post(
    "/recommend/risky_food",
    response_model=FoodSymptomPredictResponse,
    summary="Predict which foods triggered which symptoms (risky food analysis)",
    description=(
        "Pass food logs (food_name + weight_g + logged_at) and symptom logs "
        "(symptom + intensity + logged_at). Foods logged at the same timestamp "
        "are automatically grouped as a composite meal. Claude reads the full "
        "timeline and uses clinical digestion windows to identify which foods "
        "most plausibly caused each symptom. "
        "Returns {symptom: [foods]}. "
        "Intensity is case-insensitive (mild/Mild/MILD all accepted)."
    ),
)
def predict_food_symptom(req: FoodSymptomPredictRequest) -> FoodSymptomPredictResponse:
 
    if _MOCK_MODE:
        return FoodSymptomPredictResponse(**_mk_mock_risky_food(req))
    # ── Validate ──────────────────────────────────────────────────────────────
    if not _state.predictor_ready:
        raise HTTPException(503, f"Predictor unavailable: {_state.predictor_error or 'unknown'}")
    if not _state.usda_ready:
        raise HTTPException(503, "USDA client unavailable — cannot resolve food names.")
 
    for sl in req.symptom_logs:
        normalised = _normalise_intensity(sl.intensity)
        if normalised not in _VALID_PREDICT_SEVERITIES:
            raise HTTPException(
                422,
                f"Invalid intensity '{sl.intensity}'. "
                f"Accepted values: Mild, Moderate, Severe (case-insensitive).",
            )
 
    cache_key = _request_cache_key("recommend_risky_food", req.food_logs, req.symptom_logs)
    cached_payload = _endpoint_cache_get(cache_key)
    if cached_payload is not None:
        return FoodSymptomPredictResponse(**cached_payload)
 
    # ── Resolve food_name inputs (supports sentence-style meal text) ─────────
    mongo_db = getattr(_state, "_mongo_db", None)
    food_logs_named = _expand_food_name_logs_batch(req.food_logs, mongo_db)
 
    symptom_logs_plain = [
        {
            "symptom":   sl.symptom,
            "intensity": _normalise_intensity(sl.intensity),
            "logged_at": sl.logged_at,
        }
        for sl in req.symptom_logs
    ]
 
    # ── Detect composites (for response metadata) ─────────────────────────────
    grouped = group_composite_meals(food_logs_named)
    composite_count = sum(1 for m in grouped if m["is_composite"])
 
    # ── Claude temporal causation ─────────────────────────────────────────────
    try:
        predictions = predict_causation_by_time(
            food_logs_named = food_logs_named,
            symptom_logs    = symptom_logs_plain,
        )
    except Exception as exc:
        log.exception("predict/food-symptom — temporal analysis failed")
        raise HTTPException(500, f"Prediction error: {exc}")

    response_payload = {
        "predictions": predictions,
        "food_logs_processed": len(req.food_logs),
        "symptom_logs_processed": len(req.symptom_logs),
        "composite_meals_detected": composite_count,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    _endpoint_cache_set(cache_key, response_payload)
    return FoodSymptomPredictResponse(**response_payload)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT — POST /recommend/triggers_food
# ─────────────────────────────────────────────────────────────────────────────

class TriggersFoodRequest(BaseModel):
    symptom_name: str = Field(..., min_length=2, max_length=100,
                               description="The gut symptom being investigated",
                               examples=["Bloating"])
    food_name:    List[str] = Field(..., min_length=1, max_length=20,
                               description="Foods suspected of triggering this symptom",
                               examples=[["Beans", "Onion", "Dairy"]])


class TriggersFoodResponse(BaseModel):
    symptom_name: str
    trigger_foods: List[str]
    insight:       str   # 2-line, 17-20 word Claude-generated summary


_TRIGGERS_SYSTEM = (
    "You are a clinical gut-health dietitian AI. "
    "Given a symptom and a list of foods that trigger it, write exactly 1 lines "
    "totalling 17-20 words. Be direct and clinical. No bullet points, no markdown, "
    "no preamble. Return only the two lines of text, nothing else."
)


@app.post(
    "/recommend/triggers_food",
    response_model=TriggersFoodResponse,
    summary="Get a 1-line AI insight about foods that trigger a specific symptom",
)
def recommend_triggers_food(req: TriggersFoodRequest) -> TriggersFoodResponse:
    if _MOCK_MODE:
        return TriggersFoodResponse(**_mk_mock_triggers_food(req))
    if _state.claude_client is None:
        raise HTTPException(503, "Claude client not available — check Claude_API_key in .env")

    foods_str = ", ".join(req.food_name)
    user_msg  = (
        f"Symptom: {req.symptom_name}\n"
        f"Trigger foods: {foods_str}"
    )

    try:
        msg = _state.claude_client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 60,
            system     = _TRIGGERS_SYSTEM,
            messages   = [{"role": "user", "content": user_msg}],
        )
        insight = msg.content[0].text.strip()
    except Exception as exc:
        log.exception("recommend/triggers_food — Claude call failed")
        raise HTTPException(500, f"AI insight generation failed: {exc}")

    return TriggersFoodResponse(
        symptom_name  = req.symptom_name,
        trigger_foods = req.food_name,
        insight       = insight,
    )

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT — POST /recommend/gentle_note
# ─────────────────────────────────────────────────────────────────────────────

import hashlib as _hashlib

_GENTLE_NOTE_SYSTEM = (
    "You are a warm, encouraging gut-health coach. "
    "Given a short summary of a user's recent diet and symptoms, "
    "write exactly ONE gentle, supportive sentence of advice (15-20 words). "
    "Be specific to what they ate and felt. No lists, no markdown, no preamble. "
    "Return only the sentence."
)


class GentleNoteResponse(BaseModel):
    note:             str   # the one gentle sentence
    symptoms_found:   int
    trigger_foods:    int
    cached:           bool


@app.post(
    "/recommend/gentle_note",
    response_model=GentleNoteResponse,
    summary="One gentle sentence of gut-health advice based on food and symptom logs",
    description=(
        "Pass food_logs (usda_id + weight_g + logged_at) and symptom_logs "
        "(symptom + intensity + logged_at). Returns a single warm, encouraging "
        "sentence of advice tailored to the user's actual trigger foods and symptoms. "
        "Zero Claude calls on a cache hit. Haiku model on a miss."
    ),
)
def recommend_gentle_note(req: FoodAnalysisRequest) -> GentleNoteResponse:

    if _MOCK_MODE:
        return GentleNoteResponse(
            note="Keep up the great work — your balanced meals are supporting your gut health well.",
            symptoms_found=len(req.symptom_logs),
            trigger_foods=0,
            cached=False,
        )

    if not _state.predictor_ready:
        raise HTTPException(503, f"Predictor unavailable: {_state.predictor_error or 'unknown'}")
    if not _state.usda_ready:
        raise HTTPException(503, "USDA client unavailable.")

    # ── Resolve usda_id → food names (uses USDA search cache) ────────────────
    food_logs_named = _resolve_food_logs(req.food_logs)

    symptom_logs_plain = [
        {
            "symptom":   sl.symptom,
            "intensity": _normalise_intensity(sl.intensity),
            "logged_at": sl.logged_at,
        }
        for sl in req.symptom_logs
    ]

    # ── Run pure-logic trigger finder (zero Claude calls) ─────────────────────
    from food_symptom_predictor import predict_causation_by_time
    predictions: dict[str, list[str]] = predict_causation_by_time(
        food_logs_named, symptom_logs_plain
    )

    # Build compact trigger summary: top 2 symptoms + their top 2 trigger foods
    # This is what we feed Claude — never the full 168-entry log
    trigger_lines: list[str] = []
    all_trigger_foods: set[str] = set()
    for symptom, foods in list(predictions.items())[:2]:
        top_foods = foods[:2]
        all_trigger_foods.update(top_foods)
        if top_foods:
            trigger_lines.append(f"{symptom}: {', '.join(top_foods)}")

    n_symptoms = len([s for s, f in predictions.items() if f])
    n_triggers = len(all_trigger_foods)

    # If no triggers found — a gentle positive note still helps
    if not trigger_lines:
        dominant_symptom = req.symptom_logs[0].symptom if req.symptom_logs else None
        summary = (
            f"The user logged {len(req.food_logs)} food entries "
            + (f"and reported {dominant_symptom}. " if dominant_symptom else "with no symptoms reported. ")
            + "No clear dietary trigger was identified."
        )
    else:
        summary = (
            f"The user logged {len(req.food_logs)} food entries. "
            f"Likely triggers — {'; '.join(trigger_lines)}."
        )

    # ── Cache key: SHA-256 of the compact trigger summary (not the raw logs) ──
    cache_key = _hashlib.sha256(summary.encode()).hexdigest()

    # ── Cache lookup ──────────────────────────────────────────────────────────
    mongo_db = getattr(_state, "_mongo_db", None)
    if mongo_db is not None:
        try:
            cached_note = mongo_db.get_gentle_note(cache_key)
            if cached_note:
                log.info(f"gentle_note cache HIT key={cache_key[:12]}…")
                return GentleNoteResponse(
                    note=cached_note,
                    symptoms_found=n_symptoms,
                    trigger_foods=n_triggers,
                    cached=True,
                )
        except Exception as exc:
            log.warning(f"gentle_note cache read failed: {exc}")

    # ── Claude call — Haiku, max 60 tokens (one sentence) ────────────────────
    if _state.claude_client is None:
        raise HTTPException(503, "Claude client unavailable.")

    try:
        msg = _state.claude_client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 60,
            system     = _GENTLE_NOTE_SYSTEM,
            messages   = [{"role": "user", "content": summary}],
        )
        note = msg.content[0].text.strip().strip('"')
    except Exception as exc:
        log.exception("gentle_note — Claude call failed")
        raise HTTPException(500, f"Note generation failed: {exc}")

    # ── Store in cache ────────────────────────────────────────────────────────
    if mongo_db is not None:
        try:
            mongo_db.set_gentle_note(cache_key, note)
            log.info(f"gentle_note cache SET key={cache_key[:12]}…")
        except Exception as exc:
            log.warning(f"gentle_note cache write failed: {exc}")

    return GentleNoteResponse(
        note=note,
        symptoms_found=n_symptoms,
        trigger_foods=n_triggers,
        cached=False,
    )

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT — POST /recommend/food_trigger_check
# ─────────────────────────────────────────────────────────────────────────────

import hashlib as _hashlib2

_TRIGGER_NOTE_SYSTEM = (
    "You are a concise gut-health coach. "
    "Write exactly ONE sentence (15-22 words) explaining why a specific food "
    "may trigger specific symptoms, based on its nutritional content. "
    "Be specific and warm. No markdown, no preamble, no lists. Return only the sentence."
)


class FoodTriggerCheckRequest(BaseModel):
    predictions: dict[str, list[str]] = Field(

        description="Symptom → list of food names (output from /recommend/risky_food)",
        example={
            "Heartburn": ["Fish"],
            "Bloating":  ["Beans"],
            "Gas":       ["Beans", "Egg"],
            "Fatigue":   ["Nuts + Snacks", "Chicken + Rice"],
        },
    )
    target_food: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Food name to check (case-insensitive, partial match supported)",
        example="Egg",
    )


class SymptomRiskDetail(BaseModel):
    symptom:          str
    nutrient_risk:    float   # 0-1 from SYMPTOM_NUTRIENT_RISK logic
    keyword_risk:     float   # 0-1 from _FOOD_RISK_KEYWORDS heuristic
    combined_risk:    float   # weighted blend
    risk_level:       str     # Low / Moderate / High
    risk_nutrients:   list[str]  # which nutrients are over threshold


class FoodTriggerCheckResponse(BaseModel):
    target_food:          str
    usda_description:     str         # resolved USDA name (or food name if not found)
    triggered_count:      int         # x — symptoms where this food appeared
    total_symptoms:       int         # y — total symptoms in predictions
    triggered_symptoms:   list[str]   # which symptoms the food appeared in
    symptom_risks:        list[SymptomRiskDetail]  # nutrient risk per triggered symptom
    note:                 str         # one Claude Haiku sentence
    cached:               bool


def _risk_level_label(score: float) -> str:
    if score >= 0.60: return "High"
    if score >= 0.35: return "Moderate"
    return "Low"


def _build_nutrient_snapshot_from_usda(
    data: dict, portion_g: float = 100.0
) -> "NutrientSnapshot":
    """Build NutrientSnapshot from usda_client.get_nutrients() result at 100g."""
    from food_symptom_predictor import NutrientSnapshot
    def sv(k):
        v = data.get(k)
        return round(float(v), 3) if v is not None else None
    return NutrientSnapshot(
        description  = data.get("description", ""),
        calories     = sv("calories"),   protein  = sv("protein"),
        total_fat    = sv("total_fat"),  carbs    = sv("carbs"),
        sodium       = sv("sodium"),     sat_fat  = sv("sat_fat"),
        cholesterol  = sv("cholesterol"),sugar    = sv("sugar"),
        portion_g    = portion_g,
    )


@app.post(
    "/recommend/food_trigger_check",
    response_model=FoodTriggerCheckResponse,
    summary="Check how many symptoms a specific food triggered and why (nutrient-based, no Claude except 1-sentence note)",
    description=(
        "Pass the predictions dict from /recommend/risky_food and a target food name. "
        "Returns x/y symptom count, per-symptom nutrient risk scores from USDA data, "
        "and one Claude Haiku sentence summarising the finding. "
        "All risk scoring is pure logic — no Claude for analysis."
    ),
)
def food_trigger_check(req: FoodTriggerCheckRequest) -> FoodTriggerCheckResponse:

    if _MOCK_MODE:
        return FoodTriggerCheckResponse(
            target_food="Egg", usda_description="Egg, whole, raw",
            triggered_count=1, total_symptoms=4,
            triggered_symptoms=["Gas"],
            symptom_risks=[SymptomRiskDetail(
                symptom="Gas", nutrient_risk=0.32, keyword_risk=0.0,
                combined_risk=0.32, risk_level="Low", risk_nutrients=[],
            )],
            note="Eggs are generally gentle on the gut but can cause mild gas in sensitive individuals.",
            cached=False,
        )

    if not _state.usda_ready:
        raise HTTPException(503, f"USDA client unavailable: {_state.usda_error or 'unknown'}")

    target = req.target_food.strip()
    target_lower = target.lower()

    # ── Step 1: find which symptoms contain the target food (substring match) ──
    triggered_symptoms: list[str] = []
    for symptom, foods in req.predictions.items():
        for food in foods:
            if target_lower in food.lower() or food.lower() in target_lower:
                triggered_symptoms.append(symptom)
                break

    total_symptoms   = len(req.predictions)
    triggered_count  = len(triggered_symptoms)

    # ── Step 2: USDA search → get nutrients (uses existing search cache) ───────
    usda_data        = None
    usda_description = target
    usda_matches     = _state.usda_client.search(target, top_k=1)

    if usda_matches:
        best_id  = usda_matches[0]["usda_id"]
        usda_data = _state.usda_client.get_nutrients(best_id)
        if usda_data:
            usda_description = usda_data.get("description", target)

    # Build NutrientSnapshot at 100g baseline
    if usda_data:
        nutrients = _build_nutrient_snapshot_from_usda(usda_data, portion_g=100.0)
    else:
        from food_symptom_predictor import NutrientSnapshot
        nutrients = NutrientSnapshot(description=target, portion_g=100.0)

    # ── Step 3: pure-logic risk scoring per triggered symptom ─────────────────
    from food_symptom_predictor import _nutrient_risk, _keyword_risk_score

    symptom_risks: list[SymptomRiskDetail] = []
    for symptom in triggered_symptoms:
        n_risk, risk_notes = _nutrient_risk(nutrients, symptom)
        k_risk             = _keyword_risk_score(usda_description, symptom)
        # weighted blend: nutrient data is primary when available
        if usda_data:
            combined = round(n_risk * 0.70 + k_risk * 0.30, 4)
        else:
            # no USDA data — rely entirely on keyword heuristic
            combined = round(k_risk, 4)

        symptom_risks.append(SymptomRiskDetail(
            symptom       = symptom,
            nutrient_risk = round(n_risk, 4),
            keyword_risk  = round(k_risk, 4),
            combined_risk = combined,
            risk_level    = _risk_level_label(combined),
            risk_nutrients= risk_notes,
        ))

    # Sort by combined risk descending
    symptom_risks.sort(key=lambda r: r.combined_risk, reverse=True)

    # ── Step 4: build compact summary for Claude (not the raw logs) ────────────
    top_symptom   = symptom_risks[0].symptom if symptom_risks else (triggered_symptoms[0] if triggered_symptoms else "no symptoms")
    top_nutrients = symptom_risks[0].risk_nutrients[:2] if symptom_risks else []

    nutrient_hint = (
        f" Key nutrients involved: {'; '.join(top_nutrients)}." if top_nutrients else ""
    )
    summary = (
        f"Food: {usda_description}. "
        f"Triggered {triggered_count} out of {total_symptoms} tracked symptoms "
        f"({', '.join(triggered_symptoms) if triggered_symptoms else 'none'}). "
        f"Highest risk symptom: {top_symptom}.{nutrient_hint}"
    )

    # ── Step 5: cache key based on compact summary (not raw input) ─────────────
    cache_key  = _hashlib2.sha256(summary.encode()).hexdigest()
    mongo_db   = getattr(_state, "_mongo_db", None)

    if mongo_db is not None:
        try:
            cached_note = mongo_db.get_gentle_note(cache_key)
            if cached_note:
                log.info(f"food_trigger_check cache HIT key={cache_key[:12]}…")
                return FoodTriggerCheckResponse(
                    target_food       = target,
                    usda_description  = usda_description,
                    triggered_count   = triggered_count,
                    total_symptoms    = total_symptoms,
                    triggered_symptoms= triggered_symptoms,
                    symptom_risks     = symptom_risks,
                    note              = cached_note,
                    cached            = True,
                )
        except Exception as exc:
            log.warning(f"food_trigger_check cache read failed: {exc}")

    # ── Step 6: Claude Haiku — one sentence only ───────────────────────────────
    if _state.claude_client is None:
        note = (
            f"{target} appeared in {triggered_count} of {total_symptoms} tracked symptoms."
        )
    else:
        try:
            msg = _state.claude_client.messages.create(
                model      = "claude-haiku-4-5-20251001",
                max_tokens = 60,
                system     = _TRIGGER_NOTE_SYSTEM,
                messages   = [{"role": "user", "content": summary}],
            )
            note = msg.content[0].text.strip().strip('"')
        except Exception as exc:
            log.exception("food_trigger_check — Claude call failed")
            note = (
                f"{target} appeared in {triggered_count} of {total_symptoms} tracked symptoms."
            )

    # ── Step 7: store in cache ────────────────────────────────────────────────
    if mongo_db is not None:
        try:
            mongo_db.set_gentle_note(cache_key, note)
            log.info(f"food_trigger_check cache SET key={cache_key[:12]}…")
        except Exception as exc:
            log.warning(f"food_trigger_check cache write failed: {exc}")

    return FoodTriggerCheckResponse(
        target_food        = target,
        usda_description   = usda_description,
        triggered_count    = triggered_count,
        total_symptoms     = total_symptoms,
        triggered_symptoms = triggered_symptoms,
        symptom_risks      = symptom_risks,
        note               = note,
        cached             = False,
    )

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT — POST /recommend/symptom_culprit
# ─────────────────────────────────────────────────────────────────────────────

class SymptomCulpritRequest(BaseModel):
    food_logs:    List[FoodNameLogInput]    = Field(
        default_factory=list,
        description="Food entries (food_name + weight_g + logged_at)",
    )
    symptom_logs: List[SymptomLogInput] = Field(
        ...,
        min_length=1,
        description="One or more symptom events to investigate (symptom + intensity + logged_at)",
    )


class CulpritFoodDetail(BaseModel):
    usda_id:       int
    food_name:     str
    weight_g:      float
    hours_before:  float   # how many hours before the symptom this food was eaten
    nutrient_risk: float   # 0-1 from SYMPTOM_NUTRIENT_RISK logic
    keyword_risk:  float   # 0-1 from _FOOD_RISK_KEYWORDS heuristic
    combined_risk: float   # weighted blend
    risk_level:    str     # Low / Moderate / High
    risk_nutrients: list[str]


class SymptomCulpritResponse(BaseModel):
    symptom:          str
    intensity:        str
    culprit_foods:    list[int]              # just the usda_ids as requested
    culprit_details:  list[CulpritFoodDetail]
    foods_in_window:  int
    foods_scanned:    int
    message:          str


# Digestion windows in MINUTES — per clinical reference (image)
_CULPRIT_WINDOW: dict[str, tuple[int, int]] = {
    "Heartburn":      (15,  180),   # 0.25–3 h
    "Acid Reflux":    (15,  180),   # 0.25–3 h
    "Bloating":       (30,  360),   # 0.5–6 h
    "Gas":            (30,  360),   # 0.5–6 h
    "Cramps":         (30,  360),   # 0.5–6 h
    "Abdominal Pain": (30,  480),   # 0.5–8 h
    "Nausea":         (30,  240),   # 0.5–4 h
    "Diarrhea":       (60,  720),   # 1–12 h
    "Constipation":   (720, 4320),  # 12–72 h
    "Fatigue":        (60,  720),   # 1–12 h
}
_CULPRIT_DEFAULT_WINDOW = (30, 360)


@app.post(
    "/recommend/symptom_culprit",
    response_model=SymptomCulpritResponse,
    summary="Given food logs and one symptom event, identify which foods likely caused it",
    description=(
        "Filters foods eaten within the clinical digestion window before the symptom, "
        "then ranks them by nutrient-risk + keyword-risk scores using USDA data. "
        "Zero Claude calls — pure logic only."
    ),
)
def symptom_culprit(req: SymptomCulpritRequest) -> SymptomCulpritResponse:
    first_symptom = req.symptom_logs[0].symptom.strip()
    first_intensity = _normalise_intensity(req.symptom_logs[0].intensity)
    symptom_label = first_symptom if len(req.symptom_logs) == 1 else "Multiple"
    intensity_label = first_intensity if len(req.symptom_logs) == 1 else "Mixed"

    # ── No food logs provided ────────────────────────────────────────────────
    if not req.food_logs:
        return SymptomCulpritResponse(
            symptom          = symptom_label,
            intensity        = intensity_label,
            culprit_foods    = [],
            culprit_details  = [],
            foods_in_window  = 0,
            foods_scanned    = 0,
            message          = "No food logs provided. Cannot identify culprit foods.",
        )

    # ── Request-level cache (fast replay) ─────────────────────────────────
    culprit_cache_key = _request_cache_key(
        "recommend_symptom_culprit",
        req.food_logs,
        req.symptom_logs,
    )
    cached_payload = _endpoint_cache_get(culprit_cache_key)
    if cached_payload is not None:
        return SymptomCulpritResponse(**cached_payload)

    mongo_db = getattr(_state, "_mongo_db", None)
    expanded_logs: list[dict] = _expand_food_name_logs_batch(req.food_logs, mongo_db)

    # ── Step 1: collect candidates across all symptom windows ─────────────────
    foods_sorted: list[tuple[datetime, dict]] = []
    for fl in expanded_logs:
        ft = fl.get("logged_at")
        if ft is None:
            continue
        if ft.tzinfo is None:
            ft = ft.replace(tzinfo=timezone.utc)
        foods_sorted.append((ft, fl))
    foods_sorted.sort(key=lambda x: x[0])
    food_times = [t for t, _ in foods_sorted]

    candidates: list[dict] = []
    for sl in req.symptom_logs:
        symptom = sl.symptom.strip()
        sym_time = sl.logged_at
        if sym_time.tzinfo is None:
            sym_time = sym_time.replace(tzinfo=timezone.utc)
        win_min, win_max = _CULPRIT_WINDOW.get(symptom, _CULPRIT_DEFAULT_WINDOW)  # minutes

        # food_time satisfies: win_min <= (sym_time - food_time) <= win_max
        t_start = sym_time - timedelta(minutes=win_max)
        t_end = sym_time - timedelta(minutes=win_min)

        l = bisect_left(food_times, t_start)
        r = bisect_right(food_times, t_end)

        intensity = _normalise_intensity(sl.intensity)
        for i in range(l, r):
            food_time, fl = foods_sorted[i]
            delta_minutes = (sym_time - food_time).total_seconds() / 60.0
            if win_min <= delta_minutes <= win_max:
                original_name = fl.get("display_name") or fl.get("food_name", "Unknown food")
                candidates.append({
                    "symptom": symptom,
                    "intensity": intensity,
                    "window_min": win_min,
                    "window_max": win_max,
                    "usda_id": fl.get("usda_id", 0),
                    "food_name": original_name,
                    "original_food_name": original_name,
                    "weight_g": fl.get("weight_g", 0),
                    "logged_at": food_time,
                    "hours_before": round(delta_minutes / 60.0, 2),
                })

    foods_scanned = len(expanded_logs)
    unique_in_window = {(c["usda_id"], c["food_name"]) for c in candidates}
    foods_in_window = len(unique_in_window)

    if not candidates:
        # message references first symptom window for readability
        win_min, win_max = _CULPRIT_WINDOW.get(first_symptom, _CULPRIT_DEFAULT_WINDOW)
        win_h_min = int(win_min)
        win_h_max = int(win_max / 60)
        return SymptomCulpritResponse(
            symptom          = symptom_label,
            intensity        = intensity_label,
            culprit_foods    = [],
            culprit_details  = [],
            foods_in_window  = 0,
            foods_scanned    = foods_scanned,
            message          = (
                f"No food was eaten within the clinical digestion window "
                f"last {win_h_min} minutes-{win_h_max} hours before this {first_symptom}."
            ),
        )

    # ── Step 2: resolve USDA descriptions (search cache → MongoDB → API) ──────
    if not _state.usda_ready:
        raise HTTPException(503, f"USDA client unavailable: {_state.usda_error or 'unknown'}")

    name_cache: dict[int, str] = {}
    nutrient_cache: dict[int, Optional[dict]] = {}
    unique_uids = {c.get("usda_id", 0) for c in candidates}
    for uid in unique_uids:
        if uid and uid > 0:
            name_cache[uid] = _state.usda_client.get_description(uid)
            nutrient_cache[uid] = _state.usda_client.get_nutrients(uid)

    # ── Step 3: nutrient-risk + keyword-risk per candidate ────────────────────
    from food_symptom_predictor import (
        NutrientSnapshot, _nutrient_risk, _keyword_risk_score,
    )

    scored_raw: list[dict] = []
    for c in candidates:
        uid       = c["usda_id"]
        food_name = name_cache.get(uid, c["food_name"]) or "Unknown food"
        original_food_name = c.get("original_food_name", c["food_name"])
        weight_g  = c["weight_g"]
        symptom   = c["symptom"]
        sev_mult  = {"Mild": 0.9, "Moderate": 1.0, "Severe": 1.15}.get(c["intensity"], 1.0)

        # Fetch USDA nutrients (3-tier cache: in-process → MongoDB → API)
        usda_data = nutrient_cache.get(uid) if uid and uid > 0 else None
        if usda_data:
            scale = weight_g / 100.0
            def sv(k):
                v = usda_data.get(k)
                return round(float(v) * scale, 3) if v is not None else None
            nutrients = NutrientSnapshot(
                description  = food_name,
                calories     = sv("calories"),    protein  = sv("protein"),
                total_fat    = sv("total_fat"),   carbs    = sv("carbs"),
                sodium       = sv("sodium"),      sat_fat  = sv("sat_fat"),
                cholesterol  = sv("cholesterol"), sugar    = sv("sugar"),
                portion_g    = weight_g,
            )
            n_risk, risk_notes = _nutrient_risk(nutrients, symptom)
            k_risk             = _keyword_risk_score(food_name, symptom)
            combined           = round((n_risk * 0.70 + k_risk * 0.30) * sev_mult, 4)
        else:
            # No USDA data — keyword heuristic only
            nutrients  = NutrientSnapshot(description=food_name, portion_g=weight_g)
            n_risk     = 0.0
            risk_notes = []
            k_risk     = _keyword_risk_score(food_name, symptom)
            combined   = round(k_risk * sev_mult, 4)

        scored_raw.append({
            "usda_id": uid,
            "food_name": food_name,
            "original_food_name": original_food_name,
            "weight_g": weight_g,
            "hours_before": c["hours_before"],
            "nutrient_risk": round(n_risk, 4),
            "keyword_risk": round(k_risk, 4),
            "combined_risk": combined,
            "risk_nutrients": risk_notes,
        })

    # ── Step 4: aggregate across multiple symptom events by food ──────────────
    merged: dict[tuple[int, str], dict] = {}
    for r in scored_raw:
        key = (r["usda_id"], r["food_name"])
        if key not in merged:
            merged[key] = {
                "usda_id": r["usda_id"],
                "food_name": r["food_name"],
                "original_food_name": r.get("original_food_name", r["food_name"]),
                "weight_g": r["weight_g"],
                "hours_before": r["hours_before"],
                "nutrient_risk_sum": 0.0,
                "keyword_risk_sum": 0.0,
                "combined_risk_sum": 0.0,
                "count": 0,
                "risk_nutrients": set(),
            }
        m = merged[key]
        m["weight_g"] = max(m["weight_g"], r["weight_g"])
        m["hours_before"] = min(m["hours_before"], r["hours_before"])
        m["nutrient_risk_sum"] += r["nutrient_risk"]
        m["keyword_risk_sum"] += r["keyword_risk"]
        m["combined_risk_sum"] += r["combined_risk"]
        m["count"] += 1
        m["risk_nutrients"].update(r["risk_nutrients"])

    scored: list[CulpritFoodDetail] = []
    original_names_map: dict[str, str] = {}  # Map USDA name -> original name
    for m in merged.values():
        cnt = max(1, m["count"])
        combined = round(m["combined_risk_sum"] / cnt, 4)
        scored.append(CulpritFoodDetail(
            usda_id       = m["usda_id"],
            food_name     = m["food_name"],
            weight_g      = m["weight_g"],
            hours_before  = m["hours_before"],
            nutrient_risk = round(m["nutrient_risk_sum"] / cnt, 4),
            keyword_risk  = round(m["keyword_risk_sum"] / cnt, 4),
            combined_risk = combined,
            risk_level    = _risk_level_label(combined),
            risk_nutrients= sorted(m["risk_nutrients"]),
        ))
        # Track original food name for message generation
        original_names_map[m["food_name"]] = m.get("original_food_name", m["food_name"])

    scored.sort(key=lambda x: x.combined_risk, reverse=True)

    culprit_ids = [d.usda_id for d in scored]

    # Use original food names in message, not normalized USDA names
    original_names = [original_names_map.get(d.food_name, d.food_name) for d in scored]
    associated = _natural_join(list(dict.fromkeys(original_names)))
    message = f"{associated} is associated with {symptom.lower()}."

    response_payload = {
        "symptom": symptom_label,
        "intensity": intensity_label,
        "culprit_foods": culprit_ids,
        "culprit_details": scored,
        "foods_in_window": foods_in_window,
        "foods_scanned": foods_scanned,
        "message": message,
    }
    _endpoint_cache_set(culprit_cache_key, response_payload)
    return SymptomCulpritResponse(**response_payload)

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT — POST /recommend/food_note
# ─────────────────────────────────────────────────────────────────────────────

_FOOD_NOTE_NO_SYMPTOMS_SYSTEM = (
    "You are a concise gut-health coach. "
    "Write exactly ONE sentence of 8-10 words warning a user about a food "
    "they eat frequently but have no symptom history for yet. "
    "Always mention the food name and its likely gut symptom by name. "
    "No markdown, no preamble. Return only the sentence."
)


class FoodNoteLogItem(BaseModel):
    food_name: str      = Field(..., min_length=1, max_length=200,
                                description="Food name as typed by the user")
    weight_g:  float    = Field(..., gt=0)
    logged_at: datetime


class FoodNoteRequest(BaseModel):
    food_logs:    List[FoodNoteLogItem]   = Field(
        default_factory=list,
        description="Last 3 months of food logs with food_name (not usda_id)",
    )
    symptom_logs: List[SymptomLogInput]  = Field(
        default_factory=list,
        description="Symptom logs. Empty = user has not logged any symptoms yet.",
    )
    target_food:  str                    = Field(
        ..., min_length=1, max_length=200,
        description="The food name to generate a note about",
    )


class FoodNoteResponse(BaseModel):
    target_food: str
    case:        str    # "with_symptoms" | "no_symptoms" | "no_logs"
    severity:    str    # "Low" | "Medium" | "High" | "Early Signal - Log more to confirm"
    note:        str
    cached:      bool


def _food_name_key(name: str) -> str:
    """Normalised cache key for a food name string."""
    return hashlib.sha256(" ".join(name.lower().split()).encode()).hexdigest()


def _compute_food_note_severity(top_sym: str, x: int, pct: float) -> str:
    """
    Compute severity label for /recommend/food_note case 1.

    Rules (applied in order):
      1. Fatigue symptom        → always "Low"
      2. x < 4 occurrences     → "Early Signal - Log more to confirm"
      3. pct < 40%             → "Low"
      4. 40% <= pct < 70%      → "Medium"
      5. pct >= 70%            → "High"
    """
    if top_sym.lower() == "fatigue":
        return "Low"
    if x < 4:
        return "Early Signal - Log more to confirm"
    if pct < 40.0:
        return "Low"
    if pct < 70.0:
        return "Medium"
    return "High"


def _resolve_food_name_to_usda(
    food_name: str,
    mongo_db,
) -> tuple[Optional[int], Optional[str]]:
    """
    food_name → (usda_id, usda_description).
    Cache order: MongoDB food_name_usda_cache → USDA search API.
    Returns (None, None) if nothing found.
    """
    name_key = _food_name_key(food_name)

    # 1. MongoDB food_name_usda_cache
    if mongo_db is not None:
        try:
            cached = mongo_db.get_food_name_usda(name_key)
            if cached:
                return cached["usda_id"], cached["usda_description"]
        except Exception as exc:
            log.warning(f"food_name_usda_cache read failed: {exc}")

    # 2. USDA search — top_k=3, then pick the match whose description most
    #    closely matches the user's food name (shortest description length
    #    is a reliable proxy for "plain" food vs processed variant).
    #    E.g. "Rice" → prefer "Rice, white, long-grain, cooked" over "Rice crackers"
    if not _state.usda_ready:
        return None, None

    matches = _state.usda_client.search(food_name, top_k=3)
    if not matches:
        return None, None

    # Score each candidate: prefer descriptions where the query appears as a
    # standalone word (not part of a compound like "Rice crackers").
    food_name_lower = food_name.lower().strip()

    def _match_score(desc: str) -> tuple:
        dl = desc.lower()
        words = dl.replace(",", " ").split()
        # Exact word match in description is best
        exact = food_name_lower in words
        # Shorter description = less processed / more generic
        return (not exact, len(desc))

    best = min(matches, key=lambda m: _match_score(m["usda_description"]))
    usda_id   = best["usda_id"]
    usda_desc = best["usda_description"]

    if mongo_db is not None:
        try:
            mongo_db.set_food_name_usda(name_key, usda_id, usda_desc)
        except Exception as exc:
            log.warning(f"food_name_usda_cache write failed: {exc}")

    return usda_id, usda_desc


@app.post(
    "/recommend/food_note",
    response_model=FoodNoteResponse,
    summary="Generate a personalised note about a target food based on 3-month food + symptom logs",
    description=(
        "Three cases handled:\n"
        "1. food_logs + symptom_logs → temporal analysis, count how often target_food "
        "preceded symptoms, return factual note. Zero Claude calls.\n"
        "2. food_logs but no symptom_logs → USDA category + nutrient risk → "
        "Claude Haiku 15-18 word warning. Cached by food+frequency.\n"
        "3. No food_logs → static response, no Claude, no USDA.\n"
        "All food_name → usda_id resolutions are cached in MongoDB permanently."
    ),
)
def recommend_food_note(req: FoodNoteRequest) -> FoodNoteResponse:

    if _MOCK_MODE:
        return FoodNoteResponse(
            target_food = "chicken",
            case        = "with_symptoms",
            severity    = "Low",
            note        = "Maybe 'chicken' triggered Heartburn about 2 times in the last 30 days.",
            cached      = False,
        )

    mongo_db   = getattr(_state, "_mongo_db", None)
    target     = req.target_food.strip()
    target_low = target.lower()

    # ── CASE 3: no food logs at all ──────────────────────────────────────────
    if not req.food_logs:
        return FoodNoteResponse(
            target_food = target,
            case        = "no_logs",
            severity    = "Early Signal - Log more to confirm",
            note        = f"Not enough logs to analyse '{target}' yet.",
            cached      = False,
        )

    # ── Request-level cache (fast replay) ─────────────────────────────────
    endpoint_cache_key = _request_cache_key(
        "recommend_food_note|" + _light_normalise_text(target),
        req.food_logs,
        req.symptom_logs,
    )
    cached_payload = _endpoint_cache_get(endpoint_cache_key)
    if cached_payload is not None:
        return FoodNoteResponse(**cached_payload)

    # Fast early return: no symptom logs => we don't need USDA resolution.
    if not req.symptom_logs:
        payload = {
            "target_food": target,
            "case": "no_logs",
            "severity": "Early Signal - Log more to confirm",
            "note": f"Not enough symptom logs to analyse '{target}' yet.",
            "cached": False,
        }
        _endpoint_cache_set(endpoint_cache_key, payload)
        return FoodNoteResponse(**payload)

    # ── Resolve target_food → USDA (database first, then USDA API) ───────────
    target_usda_id, target_usda_desc = _resolve_food_name_to_usda(target, mongo_db)

    # ── Pre-filter only food logs matching the target substring (fast) ──────
    target_food_times: list[datetime] = []
    target_count: int = 0
    for fl in req.food_logs:
        fn_low = fl.food_name.lower()
        if target_low in fn_low or fn_low in target_low:
            target_count += 1
            ft = fl.logged_at
            if ft.tzinfo is None:
                ft = ft.replace(tzinfo=timezone.utc)
            target_food_times.append(ft)

    # ── Date range from first to last food log entry ──────────────────────────
    timestamps  = sorted(fl.logged_at for fl in req.food_logs)
    date_start  = timestamps[0]
    date_end    = timestamps[-1]
    delta_days  = max(1, (date_end - date_start).days)
    days_label  = f"{delta_days} day{'s' if delta_days != 1 else ''}"

    # ── CASE 1: food_logs + symptom_logs present ──────────────────────────────

    # Temporal window: which symptom events had target_food eaten before them?
    # Track per-symptom: trigger count AND the actual delta minutes for avg time.
    from culprit_food_finder import _DIGESTION_WINDOW, _DEFAULT_WINDOW

    trigger_count:    int              = 0
    matched_symptoms: dict[str, int]  = {}
    delta_minutes_list: list[float]   = []   # collect deltas to compute avg time

    for sl in req.symptom_logs:
        symptom  = sl.symptom.strip()
        sym_time = sl.logged_at
        if sym_time.tzinfo is None:
            sym_time = sym_time.replace(tzinfo=timezone.utc)
        win_min, win_max = _DIGESTION_WINDOW.get(symptom, _DEFAULT_WINDOW)

        for food_time in target_food_times:
            delta_min = (sym_time - food_time).total_seconds() / 60.0
            if win_min <= delta_min <= win_max:
                trigger_count += 1
                matched_symptoms[symptom] = matched_symptoms.get(symptom, 0) + 1
                delta_minutes_list.append(delta_min)
                break  # count once per symptom event

    # Top triggered symptom
    top_sym = (
        max(matched_symptoms, key=lambda k: matched_symptoms[k])
        if matched_symptoms else req.symptom_logs[0].symptom
    )

    # X = trigger_count, Y = target_count (total times target food appears in logs)
    x = trigger_count
    y = target_count
    pct = round((x / y * 100) if y > 0 else 0.0, 1)

    # Average time from eating target food to symptom
    if delta_minutes_list:
        avg_min = sum(delta_minutes_list) / len(delta_minutes_list)
        if avg_min >= 60:
            avg_time_str = f"{round(avg_min / 60, 1)} hour{'s' if round(avg_min/60,1) != 1.0 else ''}"
        else:
            avg_time_str = f"{round(avg_min)} minutes"
    else:
        avg_time_str = None

    # Build cache key
    days_bucket   = delta_days // 7
    sym_sig       = "|".join(sorted(matched_symptoms.keys()))
    note_key_raw  = f"with_symptoms|{target_usda_id or target_low}|{sym_sig}|{x}|{y}|d={days_bucket}"
    note_cache_key = hashlib.sha256(note_key_raw.encode()).hexdigest()

    if mongo_db is not None:
        try:
            cached_note = mongo_db.get_food_note(note_cache_key)
            if cached_note:
                payload = {
                    "target_food": target,
                    "case": "with_symptoms",
                    "severity": _compute_food_note_severity(top_sym, x, pct),
                    "note": cached_note,
                    "cached": True,
                }
                _endpoint_cache_set(endpoint_cache_key, payload)
                return FoodNoteResponse(**payload)
        except Exception as exc:
            log.warning(f"food_note cache read failed (case1): {exc}")

    # Build deterministic note — zero Claude
    if x == 0:
        note = (
            f"'{target}' was not linked to any symptoms in {days_label}."
        )
    elif x < 4:
        note = (
            f"Early signal: '{target}' appeared before {top_sym} {x} out of {y} times "
            f"({pct}%). Log more meals and symptoms to confirm this pattern."
        )
    else:
        time_part = f", within {avg_time_str}" if avg_time_str else ""
        note = (
            f"'{target}' was associated with {top_sym} {x} out of {y} times "
            f"({pct}%){time_part} in the last {days_label}."
        )

    if mongo_db is not None:
        try:
            mongo_db.set_food_note(note_cache_key, note)
        except Exception as exc:
            log.warning(f"food_note cache write failed (case1): {exc}")

    payload = {
        "target_food": target,
        "case": "with_symptoms",
        "severity": _compute_food_note_severity(top_sym, x, pct),
        "note": note,
        "cached": False,
    }
    _endpoint_cache_set(endpoint_cache_key, payload)
    return FoodNoteResponse(**payload)

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 9 — POST /predict/feedback
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    user_id: str; usda_id: int; symptom: str; confirmed: bool


class FeedbackResponse(BaseModel):
    user_id: str; usda_id: int; symptom: str; confirmed: bool
    updated_prior: float; prior_observations: int; prior_confirmations: int
    prior_confidence: float; personalisation_weight: float; message: str


@app.post("/predict/feedback", response_model=FeedbackResponse,
          summary="Submit explicit feedback to improve personalised predictions")
def predict_feedback(req: FeedbackRequest) -> FeedbackResponse:
    if _MOCK_MODE:
        return FeedbackResponse(**_mk_mock_feedback(req))
    if _state.memory_store is None:
        raise HTTPException(503, "Memory store unavailable.")
    memory = _state.memory_store.load(req.user_id)
    memory.apply_explicit_feedback(req.usda_id, req.symptom, req.confirmed)
    _state.memory_store.save(memory)
    prior  = memory.get_prior(req.usda_id, req.symptom)
    action = "confirmed ✅" if req.confirmed else "denied ✗"
    return FeedbackResponse(
        user_id=req.user_id, usda_id=req.usda_id, symptom=req.symptom,
        confirmed=req.confirmed, updated_prior=round(prior.posterior_mean, 4),
        prior_observations=prior.observations, prior_confirmations=prior.confirmations,
        prior_confidence=round(prior.confidence, 4),
        personalisation_weight=memory.personalisation_weight,
        message=(
            f"Feedback recorded — {action}. Prior for this food→{req.symptom} pair "
            f"updated to {prior.posterior_mean:.0%} based on {prior.observations} observations."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 10 — GET /user/{user_id}/learning-summary
# ─────────────────────────────────────────────────────────────────────────────

class SensitivityItem(BaseModel):
    food_symptom_pair: str; causation_probability: float
    observations: int; confirmations: int; confidence: float; last_updated: str


class LearningSummaryResponse(BaseModel):
    user_id: str; total_food_logs: int; total_symptom_logs: int; total_log_entries: int
    personalisation_weight: float; model_weight: float; personalisation_stage: str
    learned_pairs: int; top_sensitivities: List[SensitivityItem]; learning_message: str


def _personalisation_stage(weight: float) -> str:
    if   weight >= 0.75: return "Personalised — strong personal data, minimal model reliance"
    elif weight >= 0.60: return "Developing — personal data growing, balanced with model"
    elif weight >= 0.40: return "Learning — personal patterns emerging"
    else:                return "New user — model-led predictions, keep logging daily"


@app.get("/user/{user_id}/learning-summary", response_model=LearningSummaryResponse,
         summary="View what the model has learned about this user's food sensitivities")
def learning_summary(user_id: str) -> LearningSummaryResponse:
    if _MOCK_MODE:
        return LearningSummaryResponse(**_mk_mock_learning_summary(user_id))
    if _state.memory_store is None:
        raise HTTPException(503, "Memory store unavailable.")
    summary = _state.memory_store.summary(user_id)
    if summary.get("status") == "no data yet":
        return LearningSummaryResponse(
            user_id=user_id, total_food_logs=0, total_symptom_logs=0, total_log_entries=0,
            personalisation_weight=0.20, model_weight=0.80,
            personalisation_stage="New user — model-led predictions, keep logging daily",
            learned_pairs=0, top_sensitivities=[],
            learning_message="No data yet. Start logging meals and symptoms.",
        )
    total = summary["total_food_logs"] + summary["total_symptom_logs"]
    pw    = summary["personalisation_weight"]
    return LearningSummaryResponse(
        user_id=user_id, total_food_logs=summary["total_food_logs"],
        total_symptom_logs=summary["total_symptom_logs"], total_log_entries=total,
        personalisation_weight=pw, model_weight=round(1.0 - pw, 2),
        personalisation_stage=_personalisation_stage(pw),
        learned_pairs=summary["learned_pairs"],
        top_sensitivities=[SensitivityItem(**p) for p in summary["top_sensitivities"]],
        learning_message=(
            f"Model is {_personalisation_stage(pw).lower()}. "
            f"{summary['learned_pairs']} food→symptom pattern(s) identified. "
            + ("Keep logging — predictions improve daily." if pw < 0.60
               else "Strong personal profile — predictions are highly personalised.")
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 12 — GET /user/{user_id}/dashboard
# ─────────────────────────────────────────────────────────────────────────────

class SensitivitySummary(BaseModel):
    pair: str; probability: float; observations: int


class DashboardResponse(BaseModel):
    user_id: str; name: str; current_score: int; grade: str
    food_logs_this_week: int; symptom_logs_this_week: int
    unique_foods_eaten: List[str]; symptom_frequency: dict
    score_history: List[dict]; personalisation_weight: float
    personalisation_stage: str; top_sensitivities: List[SensitivitySummary]


@app.get("/user/{user_id}/dashboard", response_model=DashboardResponse,
         summary="Full 7-day user dashboard from MongoDB")
def user_dashboard(user_id: str) -> DashboardResponse:
    if _MOCK_MODE:
        d = _mk_mock_dashboard(user_id)
        return DashboardResponse(**d)
    mongo = _state._mongo_db
    if mongo is None:
        raise HTTPException(503, "MongoDB not connected.")
    user = mongo.get_user(user_id)
    if user is None:
        raise HTTPException(404, f"User '{user_id}' not found.")
    data  = mongo.user_dashboard(user_id)
    pw    = data["personalisation_weight"]
    return DashboardResponse(
        user_id=user_id, name=data["name"], current_score=data["current_score"],
        grade=data["grade"], food_logs_this_week=data["food_logs_this_week"],
        symptom_logs_this_week=data["symptom_logs_this_week"],
        unique_foods_eaten=data["unique_foods_eaten"],
        symptom_frequency=data["symptom_frequency"],
        score_history=data["score_history"], personalisation_weight=pw,
        personalisation_stage=_personalisation_stage(pw),
        top_sensitivities=[SensitivitySummary(**s) for s in data["top_sensitivities"]],
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 13 — GET /db/health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/db/health", summary="MongoDB connection health and collection stats")
def db_health():
    if _MOCK_MODE:
        return _mk_mock_db_health()
    mongo = _state._mongo_db
    if mongo is None:
        return {"status": "disconnected", "error": "MongoDB not initialised"}
    try:
        mongo._client.admin.command("ping")
        users = mongo.list_users()
        return {
            "status": "connected",
            "database": mongo._db().name,
            "collections": {
                "users":         len(users),
                "food_logs":     mongo._col("food_logs").count_documents({}),
                "symptom_logs":  mongo._col("symptom_logs").count_documents({}),
                "user_memories": mongo._col("user_memories").count_documents({}),
                "score_history": mongo._col("score_history").count_documents({}),
                "usda_cache":    mongo._col("usda_cache").count_documents({}),
            },
            "users": [
                {"user_id": u["user_id"], "name": u.get("name"),
                 "score": u.get("current_score"), "grade": u.get("grade"),
                 "food_logs": mongo.count_food_logs(u["user_id"]),
                 "symptom_logs": mongo.count_symptom_logs(u["user_id"])}
                for u in users
            ],
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 14 — POST /predict/meal-symptom-forecast
# ─────────────────────────────────────────────────────────────────────────────

from meal_symptom_forecast import ProposedFoodItem


class ProposedFoodIn(BaseModel):
    usda_id: int; quantity_g: float = Field(..., gt=0)


class FoodScoreBreakdown(BaseModel):
    usda_id: int; food_name: str; quantity_g: float
    nli_entailment: float; nutrient_risk_score: float; quantity_weight: float
    base_score: float; personalised_score: float; personal_prior: float
    prior_confidence: float; prior_observations: int; top_risk_nutrients: List[str]


class MealSymptomForecastOut(BaseModel):
    symptom: str; risk_score: float; risk_level: str; risk_pct: str
    top_trigger_food: str; top_trigger_usda_id: int; top_trigger_score: float
    top_risk_nutrients: List[str]; per_food_scores: List[FoodScoreBreakdown]
    personalised: bool; explanation: str


class MealForecastRequest(BaseModel):
    user_id: str
    proposed_foods: List[ProposedFoodIn] = Field(..., min_length=1, max_length=20)


class MealForecastResponse(BaseModel):
    user_id: str; proposed_foods_count: int; food_logs_used: int
    symptom_logs_used: int; personalised: bool; personalisation_weight: float
    high_risk_symptoms: List[str]; medium_risk_symptoms: List[str]
    forecasts: List[MealSymptomForecastOut]; evaluated_at: str


@app.post("/predict/meal-symptom-forecast", response_model=MealForecastResponse,
          summary="Predict which symptoms a hypothetical uneaten meal might cause")
def meal_symptom_forecast(req: MealForecastRequest) -> MealForecastResponse:
    if _MOCK_MODE:
        d = _mk_mock_forecast(req)
        forecasts = [MealSymptomForecastOut(**f) for f in d.pop("forecasts")]
        return MealForecastResponse(**d, forecasts=forecasts)
    if not _state.meal_forecast_ready:
        raise HTTPException(503, f"Meal forecast unavailable: {_state.meal_forecast_error or 'unknown'}")
    if not _state.usda_ready:
        raise HTTPException(503, "USDA dataset unavailable.")
    mongo = _state._mongo_db
    if mongo is None:
        raise HTTPException(503, "MongoDB not connected — user history unavailable.")

    try:
        food_logs    = mongo.get_food_logs(req.user_id, days=90, limit=400)
        symptom_logs = mongo.get_symptom_logs(req.user_id, days=90, limit=400)
    except Exception as exc:
        log.exception(f"MongoDB fetch failed for user={req.user_id!r}")
        raise HTTPException(500, f"Database error: {exc}")

    user_memory = None; p_weight = 0.20; is_personalised = False
    if _state.memory_store is not None:
        try:
            user_memory     = _state.memory_store.load(req.user_id)
            p_weight        = getattr(user_memory, "personalisation_weight", 0.20)
            is_personalised = getattr(user_memory, "total_food_logs", 0) > 0
        except Exception as exc:
            log.warning(f"Memory load failed for {req.user_id!r}: {exc}")

    proposed = [ProposedFoodItem(usda_id=f.usda_id, quantity_g=f.quantity_g)
                for f in req.proposed_foods]

    try:
        raw_forecasts = _state.forecast_meal_symptoms(
            proposed_foods=proposed, user_memory=user_memory,
        )
    except Exception as exc:
        log.exception(f"Meal forecast failed for user={req.user_id!r}")
        raise HTTPException(500, f"Forecast error: {exc}")

    return MealForecastResponse(
        user_id=req.user_id, proposed_foods_count=len(req.proposed_foods),
        food_logs_used=len(food_logs), symptom_logs_used=len(symptom_logs),
        personalised=is_personalised, personalisation_weight=round(p_weight, 2),
        high_risk_symptoms=[f.symptom for f in raw_forecasts if f.risk_level == "High"],
        medium_risk_symptoms=[f.symptom for f in raw_forecasts if f.risk_level == "Medium"],
        forecasts=[
            MealSymptomForecastOut(
                symptom=f.symptom, risk_score=f.risk_score, risk_level=f.risk_level,
                risk_pct=f.risk_pct, top_trigger_food=f.top_trigger_food,
                top_trigger_usda_id=f.top_trigger_usda_id, top_trigger_score=f.top_trigger_score,
                top_risk_nutrients=f.top_risk_nutrients,
                per_food_scores=[
                    FoodScoreBreakdown(
                        usda_id=s.usda_id, food_name=s.food_name, quantity_g=s.quantity_g,
                        nli_entailment=s.nli_entailment, nutrient_risk_score=s.nutrient_risk_score,
                        quantity_weight=s.quantity_weight, base_score=s.base_score,
                        personalised_score=s.personalised_score, personal_prior=s.personal_prior,
                        prior_confidence=s.prior_confidence, prior_observations=s.prior_observations,
                        top_risk_nutrients=s.top_risk_nutrients,
                    )
                    for s in f.per_food_scores
                ],
                personalised=f.personalised, explanation=f.explanation,
            )
            for f in raw_forecasts
        ],
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT (existing) — POST /scan/barcode
# ─────────────────────────────────────────────────────────────────────────────

from scanner import fetch_product


class BarcodeRequest(BaseModel):
    code: str


class BarcodeResponse(BaseModel):
    barcode: str; product_name: str; quantity: Optional[str]


@app.post("/scan/barcode", response_model=BarcodeResponse)
def scan_barcode(req: BarcodeRequest):
    if _MOCK_MODE:
        return BarcodeResponse(**_mk_mock_barcode(req))
    try:
        product = fetch_product(req.code.strip())
    except Exception as e:
        raise HTTPException(404, str(e))
    return BarcodeResponse(barcode=req.code.strip(),
                            product_name=product["name"],
                            quantity=product.get("quantity"))


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT (existing) — POST /culprit-foods
# ─────────────────────────────────────────────────────────────────────────────

class SimpleFoodLog(BaseModel):
    usda_id: int; quantity_g: float; logged_at: datetime


class SimpleSymptomLog(BaseModel):
    symptom: str; severity: str = "Moderate"; logged_at: datetime


class CulpritRequest(BaseModel):
    food_logs: List[SimpleFoodLog]; symptom_logs: List[SimpleSymptomLog]


class CulpritFoodOut(BaseModel):
    usda_id: int; food_name: str; score: float; confidence: str
    occurrence_count: int; linked_symptoms: List[str]; top_symptom: str


class CulpritResponse(BaseModel):
    total_foods: int; total_symptoms: int
    culprit_foods: List[CulpritFoodOut]; summary: str


@app.post("/culprit-foods", response_model=CulpritResponse,
          summary="Find which foods most likely caused symptoms (NO user_id needed)")
def culprit_foods(req: CulpritRequest) -> CulpritResponse:
    if _MOCK_MODE:
        d = _mk_mock_culprit(req)
        culprits = [CulpritFoodOut(**f) for f in d.pop("culprit_foods")]
        return CulpritResponse(**d, culprit_foods=culprits)
    if not _state.usda_ready:
        raise HTTPException(503, "USDA client unavailable.")
    try:
        food_logs = []
        for f in req.food_logs:
            food_name = _state.usda_client.get_description(f.usda_id)
            food_logs.append({
                "usda_id": f.usda_id, "usda_description": food_name,
                "quantity_g": f.quantity_g, "logged_at": f.logged_at,
            })
        result = find_culprit_foods(
            food_logs=food_logs,
            symptom_logs=[s.model_dump() for s in req.symptom_logs],
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    return CulpritResponse(
        total_foods=len(req.food_logs), total_symptoms=len(req.symptom_logs),
        culprit_foods=[
            CulpritFoodOut(usda_id=f.usda_id, food_name=f.food_name, score=f.aggregate_score,
                           confidence=f.confidence_label, occurrence_count=f.occurrence_count,
                           linked_symptoms=f.linked_symptoms, top_symptom=f.top_symptom)
            for f in result.culprit_foods
        ],
        summary=result.method_summary,
    )
