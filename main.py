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
import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

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

    # ── 1. Claude client ──────────────────────────────────────────────────────
    claude_key = os.getenv("Claude_API_key", "")
    if not claude_key:
        log.error("Claude_API_key not set in .env — AI features will fail.")
    _state.claude_client = anthropic.Anthropic(api_key=claude_key)
    log.info("Claude client ready.")

    # ── 2. USDA client ────────────────────────────────────────────────────────
    usda_key = os.getenv("USDA_API_KEY", "DEMO_KEY")
    log.info(f"USDA API key: {'custom' if usda_key != 'DEMO_KEY' else 'DEMO_KEY (rate-limited)'}")

    try:
        from usda_client import USDAClient
        from database import db as _mongo_db_ref
        # Try to connect to MongoDB to get the cache collection; gracefully fall back
        try:
            _mongo_db_ref.connect()
            _state._mongo_db = _mongo_db_ref
            usda_col = _mongo_db_ref._col("usda_cache")
            log.info("MongoDB connected. USDA cache enabled.")
        except Exception as mongo_exc:
            log.warning(f"MongoDB unavailable ({mongo_exc}) — USDA cache disabled, in-process cache only.")
            usda_col = None

        _state.usda_client = USDAClient(api_key=usda_key, mongo_col=usda_col)
        _state.usda_ready  = True
        log.info("USDA client ready.")
    except Exception as exc:
        _state.usda_error = str(exc)
        log.error(f"USDA client failed: {exc}")

    # ── 3. Food text-to-USDA pipeline ────────────────────────────────────────
    try:
        import food_text_to_usda as _ft
        _ft.init(_state.claude_client, _state.usda_client)
        _state.text_to_usda = _ft.text_to_usda
        log.info("food_text_to_usda ready.")
    except Exception as exc:
        _state.usda_error = str(exc)
        log.error(f"food_text_to_usda failed: {exc}")

    # ── 4. Nutrition scorer ───────────────────────────────────────────────────
    try:
        import nutrition_scorer as _ns
        _ns.init(_state.claude_client, _state.usda_client)
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

app = FastAPI(title="Gut Health API", version="4.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    Food_detected:    int
    Logged_at:        str
    normalised_names: List[str]
    results:          List[FoodParseItem]
    meal_type:        Optional[str] = None
    score_impact:     Optional[int] = None
    updated_score:    Optional[int] = None
    note:             Optional[str] = None


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
    note:          Optional[str] = None

    if req.current_score is not None and _state.nutrition_ready:
        try:
            import nutrition_scorer as _ns
            foods_for_scoring = [
                {"usda_description": r["usda_description"], "weight_g": r["weight_g"]}
                for r in raw
            ]
            raw_score, note = _ns.score_meal_claude(
                foods=foods_for_scoring,
                meal_type=meal_type or "Lunch",
            )
            score_impact  = _calculate_score_impact(raw_score, req.current_score)
            updated_score = max(_SCORE_FLOOR, min(_SCORE_CEIL, req.current_score + score_impact))
        except Exception as exc:
            log.warning(f"food/parse scoring failed: {exc}")

    return FoodParseResponse(
        Food_detected    = len(raw),
        Logged_at        = logged_at,
        normalised_names = normalised_names,
        results          = [
            FoodParseItem(
                normalised_name = r["normalised_name"].split(",")[0].strip(),
                usda_id         = r["usda_id"],
                weight_g        = r["weight_g"],
            )
            for r in raw
        ],
        meal_type     = meal_type,
        score_impact  = score_impact,
        updated_score = updated_score,
        note          = note,
    )


# ENDPOINT 2 — POST /score
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/score", response_model=DigestiveResult,
          summary="Calculate onboarding digestion score (call once at onboarding)")
def score(data: DigestiveInput) -> DigestiveResult:
    return calculate_score(data)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — POST /log/food
# ─────────────────────────────────────────────────────────────────────────────

_VALID_MEAL_TYPES = {"Breakfast", "Lunch", "Dinner", "Snack"}
_SCORE_FLOOR      = 40
_SCORE_CEIL       = 99


def _calculate_score_impact(raw_score: int, current_score: int) -> int:
    """
    Apply a neutralising multiplier so the gut score drifts toward the
    healthy band (80-90) rather than runaway highs or lows.

    Positive foods have MORE impact when the score is LOW (the user
    benefits more from good food when they are struggling).
    Negative foods have MORE impact when the score is HIGH (a healthy
    gut has more to lose from junk food).

    Formula (derived from user example: +3 at score=60 → +5, +3 at score=90 → +1):
        positive multiplier = (100 - current_score) / 24
        negative multiplier = (current_score - 40)  / 24
    """
    clamped = max(_SCORE_FLOOR, min(_SCORE_CEIL, current_score))
    if raw_score >= 0:
        multiplier = (100 - clamped) / 24
    else:
        multiplier = (clamped - _SCORE_FLOOR) / 24
    return int(round(raw_score * max(0.05, multiplier)))




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
    updated_score:    int
    food_detected:    int
    logged_at:        str
    normalised_names: List[str]
    results:          List[FoodLogItem]
    meal_type:        str
    note:             str


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
    except Exception as exc:
        log.exception(f"log/food parse error: {combined_text!r}")
        raise HTTPException(500, f"Food parsing error: {exc}")

    if not raw:
        raise HTTPException(422, "No recognisable food found in the description.")

    logged_at        = raw[0].get("logged_at", utc_now_iso())
    normalised_names = [r["normalised_name"].split(",")[0].strip() for r in raw]

    # ── Step 2: score the meal via Claude ────────────────────────────────────
    raw_score = 0
    note      = "Meal logged successfully."
    if _state.nutrition_ready:
        try:
            import nutrition_scorer as _ns
            foods_for_scoring = [
                {"usda_description": r["usda_description"], "weight_g": r["weight_g"]}
                for r in raw
            ]
            raw_score, note = _ns.score_meal_claude(
                foods=foods_for_scoring,
                meal_type=req.meal_type,
            )
        except Exception as exc:
            log.warning(f"log/food scoring failed: {exc}")

    # ── Step 3: apply neutralising multiplier + clamp ────────────────────────
    score_impact  = _calculate_score_impact(raw_score, req.current_score)
    updated_score = max(_SCORE_FLOOR, min(_SCORE_CEIL, req.current_score + score_impact))

    return FoodLogResponse(
        updated_score    = updated_score,
        food_detected    = len(raw),
        logged_at        = logged_at,
        normalised_names = normalised_names,
        results          = [
            FoodLogItem(
                normalised_name = r["normalised_name"].split(",")[0].strip(),
                usda_id         = r["usda_id"],
                weight_g        = r["weight_g"],
            )
            for r in raw
        ],
        meal_type = req.meal_type,
        note      = note,
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
    note:               str        = Field(..., description="7–10 word clinical summary of what was reported.")


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

    # ── Score with Claude ─────────────────────────────────────────────────────
    if _state.claude_client is None:
        raise HTTPException(503, "Claude client not available.")

    try:
        from nutrition_scorer import score_symptom_log_claude
        penalty, note_text = score_symptom_log_claude(
            symptoms  = deduped,
            severity  = req.severity,
            note      = req.note,
        )
    except Exception as exc:
        log.exception("score_symptom_log_claude failed")
        raise HTTPException(500, f"Scoring error: {exc}")

    updated_score = max(0, min(100, req.current_score - penalty))

    return SymptomLogResponse(
        updated_score     = updated_score,
        detected_symptoms = deduped,
        logged_at         = logged_at_str,
        note              = note_text,
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

#

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
            model      = "claude-sonnet-4-6",
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
            "weight_g":  fl.weight_g,
            "logged_at": fl.logged_at,
        }
        for fl in food_logs
    ]
 
 
# ── Shared helper: normalise intensity case ───────────────────────────────────
 
_INTENSITY_NORMALISE = {v.lower(): v for v in ("Mild", "Moderate", "Severe")}
 
def _normalise_intensity(raw: str) -> str:
    """'mild' → 'Mild', 'MODERATE' → 'Moderate', 'high' → kept as-is."""
    return _INTENSITY_NORMALISE.get(raw.lower(), raw)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — ENDPOINT 8  POST /predict/food-symptom
# ══════════════════════════════════════════════════════════════════════════════
 
class FoodSymptomPredictRequest(BaseModel):
    food_logs:    List[FoodLogInput]    = Field(..., min_length=1)
    symptom_logs: List[SymptomLogInput] = Field(..., min_length=1)
    user_id:      Optional[str]         = Field(
        None,
        description="Optional — supply to enable Bayesian personalisation memory updates.",
    )
 
 
class FoodSymptomPredictResponse(BaseModel):
    predictions:             dict[str, List[str]]
    food_logs_processed:     int
    symptom_logs_processed:  int
    composite_meals_detected: int
    evaluated_at:            str
 
 
_VALID_PREDICT_SEVERITIES = {"Mild", "Moderate", "Severe"}
 
 
@app.post(
    "/predict/food-symptom",
    response_model=FoodSymptomPredictResponse,
    summary="Predict which foods caused which symptoms using Claude temporal analysis",
    description=(
        "Pass food logs (usda_id + weight_g + logged_at) and symptom logs "
        "(symptom + intensity + logged_at). Foods logged at the same timestamp "
        "are automatically grouped as a composite meal. Claude reads the full "
        "timeline and uses clinical digestion windows to identify which foods "
        "most plausibly caused each symptom. "
        "Returns {symptom: [foods]}. "
        "Intensity is case-insensitive (mild/Mild/MILD all accepted). "
        "Supply user_id to enable Bayesian personalisation memory updates."
    ),
)
def predict_food_symptom(req: FoodSymptomPredictRequest) -> FoodSymptomPredictResponse:
 
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
 
    # ── Resolve usda_id → food names ─────────────────────────────────────────
    food_logs_named = _resolve_food_logs(req.food_logs)
 
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
 
    # ── Bayesian memory auto-update (only when user_id is provided) ───────────
    if req.user_id and _state.auto_update_from_logs and _state.memory_store:
        try:
            food_entries = [
                FoodLogEntry(
                    user_id    = req.user_id,
                    usda_id    = fl.usda_id,
                    logged_at  = fl.logged_at,
                    quantity_g = fl.weight_g,
                )
                for fl in req.food_logs
            ]
            symptom_entries = [
                SymptomLogEntry(
                    user_id   = req.user_id,
                    symptom   = sl.symptom,
                    logged_at = sl.logged_at,
                    intensity = _normalise_intensity(sl.intensity),
                )
                for sl in req.symptom_logs
            ]
            _state.auto_update_from_logs(
                req.user_id, food_entries, symptom_entries,
                store=_state.memory_store,
            )
        except Exception as exc:
            log.warning(f"Memory auto-update failed (non-fatal): {exc}")
 
    return FoodSymptomPredictResponse(
        predictions              = predictions,
        food_logs_processed      = len(req.food_logs),
        symptom_logs_processed   = len(req.symptom_logs),
        composite_meals_detected = composite_count,
        evaluated_at             = datetime.now(timezone.utc).isoformat(),
    )
 
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
# ENDPOINT 11 — POST /recommend/safe-foods
# ─────────────────────────────────────────────────────────────────────────────

from food_recommender import recommend_safe_from_logs


class SafeFoodRequest(BaseModel):
    food_logs:    List[FoodLogInput]    = Field(..., min_length=1)
    symptom_logs: List[SymptomLogInput] = Field(default_factory=list)
    user_id:      Optional[str]         = None
    n:            int                   = Field(default=5, ge=1, le=10)
 
 
class SafeFoodResponse(BaseModel):
    safe_foods:              List[str]
    foods_analysed:          int
    composite_meals_detected: int
    symptoms_considered:     int
    source_note:             str
 
 
@app.post(
    "/recommend/safe-foods",
    response_model=SafeFoodResponse,
    summary="Get safe food recommendations based on your food and symptom history",
    description=(
        "Pass your food logs (usda_id + weight_g + logged_at) and symptom logs. "
        "Foods logged at the same timestamp are grouped as composite meals. "
        "Claude identifies which foods from your history did NOT trigger symptoms, "
        "then fills remaining slots with new gut-friendly food suggestions. "
        "Returns a list of 5 safe food names. "
        "Intensity is case-insensitive. user_id is optional."
    ),
)
def recommend_safe_foods_endpoint(req: SafeFoodRequest) -> SafeFoodResponse:
 
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
 
    # ── Resolve usda_id → food names ─────────────────────────────────────────
    food_logs_named = _resolve_food_logs(req.food_logs)
 
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
 
    # ── Claude safe food recommendation ──────────────────────────────────────
    try:
        safe_foods = recommend_safe_from_logs(
            food_logs_named = food_logs_named,
            symptom_logs    = symptom_logs_plain,
            n               = req.n,
        )
    except Exception as exc:
        log.exception("recommend/safe-foods — recommendation failed")
        raise HTTPException(500, f"Recommendation error: {exc}")
 
    if not safe_foods:
        raise HTTPException(404, "No safe food recommendations could be generated.")
 
    source_note = (
        "Recommendations generated by Claude AI. "
        "Safe history foods are returned verbatim; "
        "new suggestions are gut-friendly foods unlikely to trigger your reported symptoms."
        + (f" {composite_cnt} composite meal(s) detected and grouped." if composite_cnt else "")
    )
 
    return SafeFoodResponse(
        safe_foods               = safe_foods,
        foods_analysed           = len(req.food_logs),
        composite_meals_detected = composite_cnt,
        symptoms_considered      = len(req.symptom_logs),
        source_note              = source_note,
    )
 
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