"""
usda_client.py
──────────────
Thin wrapper around the USDA FoodData Central REST API.
Replaces the Kaggle CSV + pandas DataFrame + SentenceTransformer pipeline.

Public API
──────────
    client = USDAClient(api_key, mongo_db=None)
    client.search(query, top_k)     -> list[dict]   # text → food candidates
    client.get_nutrients(fdc_id)    -> dict | None   # fdc_id → nutrient profile
    client.get_description(fdc_id)  -> str           # fdc_id → food name

All results are cached in MongoDB (collection: usda_cache) so each USDA ID
is fetched from the network only once per server lifetime.

Nutrient keys returned (all per 100 g, matching the old Kaggle column names):
    calories, protein, total_fat, carbs, sodium, sat_fat,
    cholesterol, sugar, calcium, iron, potassium,
    vitamin_c, vitamin_e, vitamin_d, fiber
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

log = logging.getLogger(__name__)

_BASE = "https://api.nal.usda.gov/fdc/v1"

# Map our internal key → list of USDA nutrient name variants
_NUTRIENT_MAP: dict[str, list[str]] = {
    "calories":    ["Energy"],
    "protein":     ["Protein"],
    "total_fat":   ["Total lipid (fat)"],
    "carbs":       ["Carbohydrate, by difference"],
    "sodium":      ["Sodium, Na"],
    "sat_fat":     ["Fatty acids, total saturated"],
    "cholesterol": ["Cholesterol"],
    "sugar":       ["Sugars, total including NLEA", "Sugars, Total",
                    "Sugars, total"],
    "calcium":     ["Calcium, Ca"],
    "iron":        ["Iron, Fe"],
    "potassium":   ["Potassium, K"],
    "vitamin_c":   ["Vitamin C, total ascorbic acid"],
    "vitamin_e":   ["Vitamin E (alpha-tocopherol)"],
    "vitamin_d":   ["Vitamin D (D2 + D3)", "Vitamin D (D2 + D3), International Units",
                    "Vitamin D"],
    "fiber":       ["Fiber, total dietary"],
}


def _extract_nutrients(food_nutrients: list[dict]) -> dict[str, Optional[float]]:
    """Parse USDA foodNutrients list -> our standard nutrient dict.

    USDA returns two 'Energy' rows per food: one in kcal (nutrient ID 1008)
    and one in kJ (nutrient ID 1062).  The kJ value is ~4.184x larger so we
    must always prefer the kcal row for calories.

    Strategy:
      - Build a kcal-only lookup alongside the general lookup.
      - For the 'calories' key, try the kcal-only lookup first.
      - All other nutrients use the general lookup as before.
    """
    name_to_val: dict[str, float] = {}       # all units
    name_to_val_kcal: dict[str, float] = {}  # kcal-unit entries only

    for fn in food_nutrients:
        nutrient = fn.get("nutrient") or fn
        name   = nutrient.get("name", "").strip()
        unit   = (nutrient.get("unitName") or fn.get("unitName") or "").strip().upper()
        amount = fn.get("amount")
        if amount is None or not name:
            continue
        amount = float(amount)
        name_l = name.lower()
        if name_l not in name_to_val:        # first occurrence wins
            name_to_val[name_l] = amount
        if unit == "KCAL":
            name_to_val_kcal[name_l] = amount

    out: dict[str, Optional[float]] = {}
    for key, variants in _NUTRIENT_MAP.items():
        val = None
        # For calories try kcal-only bucket first to avoid kJ values
        buckets = [name_to_val_kcal, name_to_val] if key == "calories" else [name_to_val]
        for bucket in buckets:
            for v in variants:
                val = bucket.get(v.lower())
                if val is not None:
                    break
            if val is not None:
                break
        out[key] = val
    return out


class USDAClient:
    """
    Thread-safe USDA FoodData Central client with MongoDB nutrient cache.
    Pass mongo_col=None to disable nutrient caching.
    Pass search_col=None to disable search result caching.
    """

    def __init__(self, api_key: str, mongo_col=None, search_col=None):
        self._key        = api_key
        self._col        = mongo_col           # pymongo Collection or None (nutrient cache)
        self._search_col = search_col          # pymongo Collection or None (search cache)
        self._http       = httpx.Client(timeout=10.0)
        self._mem:        dict[int, dict] = {}  # in-process nutrient cache
        self._search_mem: dict[str, list] = {}  # in-process search cache

    # ── Public: search ────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search USDA FDC by food name.
        Returns list of {rank, usda_id, usda_description, similarity} dicts.

        Cache strategy (two layers):
          1. In-process dict  — zero latency, lives for the process lifetime
          2. MongoDB          — survives restarts, shared across workers
          3. USDA API         — only reached on a true cold miss

        Results are permanent: "Chicken, broiler, breast, cooked, roasted"
        will always resolve to the same FDC ID.
        """
        norm_query = " ".join(query.lower().split())  # normalise before key lookup

        # 1. In-process cache
        if norm_query in self._search_mem:
            return self._search_mem[norm_query]

        # 2. MongoDB cache
        if self._search_col is not None:
            try:
                doc = self._search_col.find_one(
                    {"query": norm_query}, {"_id": 0, "query": 0, "cached_at": 0}
                )
                if doc and doc.get("results"):
                    self._search_mem[norm_query] = doc["results"]
                    return doc["results"]
            except Exception as exc:
                log.warning(f"USDA search cache read failed: {exc}")

        # 3. USDA API
        try:
            resp = self._http.get(
                f"{_BASE}/foods/search",
                params={
                    "query":    query,
                    "api_key":  self._key,
                    "pageSize": min(top_k, 10),
                    "dataType": "Foundation,SR Legacy",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning(f"USDA search failed for '{query}': {exc}")
            return []

        foods = data.get("foods", [])
        if not foods:
            return []

        max_score = max((f.get("score", 1.0) for f in foods), default=1.0) or 1.0
        results = [
            {
                "rank":             rank,
                "usda_id":          int(f["fdcId"]),
                "usda_description": f.get("description", ""),
                "similarity":       round(f.get("score", max_score) / max_score, 4),
            }
            for rank, f in enumerate(foods[:top_k], 1)
        ]

        # Store in both caches
        self._search_mem[norm_query] = results
        if self._search_col is not None:
            try:
                self._search_col.update_one(
                    {"query": norm_query},
                    {"$set": {
                        "query":     norm_query,
                        "results":   results,
                        "cached_at": __import__("datetime").datetime.utcnow().isoformat(),
                    }},
                    upsert=True,
                )
            except Exception as exc:
                log.warning(f"USDA search cache write failed: {exc}")

        return results

    # ── Public: get_nutrients ─────────────────────────────────────────────────

    def get_nutrients(self, fdc_id: int) -> Optional[dict]:
        """
        Return nutrient profile for fdc_id (all values per 100 g).
        Returns None if ID not found.
        """
        fdc_id = int(fdc_id)

        # 1. In-process memory cache
        if fdc_id in self._mem:
            return self._mem[fdc_id]

        # 2. MongoDB cache
        if self._col is not None:
            doc = self._col.find_one({"fdc_id": fdc_id}, {"_id": 0})
            if doc:
                self._mem[fdc_id] = doc
                return doc

        # 3. USDA API
        try:
            resp = self._http.get(
                f"{_BASE}/food/{fdc_id}",
                params={"api_key": self._key},
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning(f"USDA nutrient fetch failed for fdc_id={fdc_id}: {exc}")
            return None

        nutrients = _extract_nutrients(data.get("foodNutrients", []))
        doc = {
            "fdc_id": fdc_id,
            "description": data.get("description", f"USDA ID {fdc_id}"),
            "food_category": (
                data.get("foodCategory", {}).get("description")
            ),
            **nutrients,
        }

        # Store in caches
        if self._col is not None:
            try:
                self._col.update_one(
                    {"fdc_id": fdc_id},
                    {"$set": doc},
                    upsert=True,
                )
            except Exception as exc:
                log.warning(f"MongoDB usda_cache write failed: {exc}")

        self._mem[fdc_id] = doc
        return doc

    # ── Public: get_description ───────────────────────────────────────────────

    def get_description(self, fdc_id: int) -> str:
        doc = self.get_nutrients(fdc_id)
        if doc:
            return doc.get("description", f"USDA ID {fdc_id}")
        return f"USDA ID {fdc_id}"