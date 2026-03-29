"""
main.py
───────
Seven endpoints — each does exactly one job:

  POST /food/parse              — Natural meal text → USDA food IDs (NER + T5 + SentenceTransformer)
  POST /food/text-to-id         — Natural food text → best-match USDA ID only (clean alias)
  GET  /food/lookup/{id}        — USDA food ID → name + all 16 nutrients from dataset
  POST /score                   — Onboarding questionnaire → base digestion score (call once)
  POST /log/food                — Log a meal → updated score via USDA nutrients + HuggingFace
  POST /log/symptom             — Log a symptom → updated score via severity + time-of-day
  POST /predict/food-symptom    — Given food logs + symptom logs → which food caused which symptom

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload

Docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import sys
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
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
# App state — all heavy objects loaded once at startup
# ─────────────────────────────────────────────────────────────────────────────

class _State:
    usda_ready:      bool          = False
    usda_error:      Optional[str] = None
    text_to_usda                   = None

    nutrition_ready: bool          = False
    nutrition_error: Optional[str] = None
    analyse_food_log_batch         = None
    analyse_symptom_log            = None
    usda_df                        = None
    id_col:          str           = ""
    desc_col:        str           = ""

    predictor_ready: bool          = False
    predictor_error: Optional[str] = None
    predict_food_symptom_causes    = None

    # Per-user Bayesian memory store
    memory_store                   = None   # UserMemoryStore instance
    auto_update_from_logs          = None   # callable

    # Food recommender
    recommender_ready: bool        = False
    recommender_error: Optional[str] = None
    recommend_safe_foods           = None   # callable

    # Meal symptom forecast (pre-emptive prediction for uneaten meal)
    meal_forecast_ready: bool          = False
    meal_forecast_error: Optional[str] = None
    forecast_meal_symptoms             = None   # callable

    # Symptom note NLP analyser (reuses DeBERTa — no extra RAM)
    analyse_symptom_note               = None   # callable

    # Culprit food cross-encoder (reused from food_symptom_predictor)
    culprit_cross_encoder = None

_state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading USDA food pipeline (NER + T5 + SentenceTransformer)...")
    t0 = time.time()
    try:
        import food_text_to_usda as _ft
        _state.text_to_usda = _ft.text_to_usda
        _state.usda_df      = _ft.usda_df
        _state.id_col       = _ft._id_col
        _state.desc_col     = _ft._desc_col
        _state.usda_ready   = True
        log.info(f"USDA pipeline ready in {time.time()-t0:.1f}s")
    except Exception as exc:
        _state.usda_error = str(exc)
        log.error(f"USDA pipeline failed: {exc}")

    log.info("Loading HuggingFace nutrition scorer...")
    t1 = time.time()
    try:
        import nutrition_scorer as _ns
        _state.analyse_food_log_batch = _ns.analyse_food_log_batch
        _state.analyse_symptom_log    = _ns.analyse_symptom_log
        _state.nutrition_ready        = True
        log.info(f"Nutrition scorer ready in {time.time()-t1:.1f}s")
    except Exception as exc:
        _state.nutrition_error = str(exc)
        log.error(f"Nutrition scorer failed: {exc}")

    log.info("Loading food-symptom causation predictor (cross-encoder/nli-deberta-v3-small)...")
    t2 = time.time()
    try:
        import food_symptom_predictor as _fsp
        _state.predict_food_symptom_causes = _fsp.predict_food_symptom_causes
        _state.predictor_ready             = True
        log.info(f"Food-symptom predictor ready in {time.time()-t2:.1f}s")
    except Exception as exc:
        _state.predictor_error = str(exc)
        log.error(f"Food-symptom predictor failed: {exc}")

    log.info("Connecting to MongoDB...")
    try:
        from database import db as _mongo_db, MongoUserMemoryStore
        from user_symptom_memory import auto_update_from_logs
        import user_symptom_memory as _usm
        _mongo_db.connect()
        _state.memory_store          = MongoUserMemoryStore()
        _state.auto_update_from_logs = auto_update_from_logs
        _state._mongo_db             = _mongo_db    # attach for endpoint use
        # Patch the module-level singleton so any direct callers also hit MongoDB
        _usm.user_memory_store       = _state.memory_store
        log.info("MongoDB connected. MongoUserMemoryStore ready.")
    except Exception as exc:
        log.warning(f"MongoDB unavailable ({exc}) — falling back to in-memory store.")
        try:
            from user_symptom_memory import UserMemoryStore, auto_update_from_logs
            _state.memory_store          = UserMemoryStore()
            _state.auto_update_from_logs = auto_update_from_logs
            log.info("In-memory store active (no persistence).")
        except Exception as exc2:
            log.error(f"Memory store failed: {exc2}")

    log.info("Loading food recommender (Flan-T5 + USDA similarity)...")
    t3 = time.time()
    try:
        import food_recommender as _fr
        # Inject already-loaded T5 to avoid loading a second copy
        try:
            import food_text_to_usda as _ft2
            _fr.init_t5(_ft2.t5_tokenizer, _ft2.t5_model)
        except Exception:
            log.warning("T5 not injected into recommender — will use rule-based fallback.")
        _state.recommend_safe_foods = _fr.recommend_safe_foods
        _state.recommender_ready    = True
        log.info(f"Food recommender ready in {time.time()-t3:.1f}s")
    except Exception as exc:
        _state.recommender_error = str(exc)
        log.error(f"Food recommender failed: {exc}")

    log.info("Loading meal symptom forecast module...")
    try:
        import meal_symptom_forecast as _msf
        _state.forecast_meal_symptoms = _msf.forecast_meal_symptoms
        _state.meal_forecast_ready    = True
        log.info("Meal symptom forecast ready (reuses cross-encoder NLI — no extra RAM).")
    except Exception as exc:
        _state.meal_forecast_error = str(exc)
        log.error(f"Meal symptom forecast failed to load: {exc}")

    log.info("Loading symptom note analyser...")
    try:
        import symptom_note_analyser as _sna
        import nutrition_scorer as _ns_ref
        _sna.init_classifier(_ns_ref._classifier)   # inject same DeBERTa — zero extra RAM
        _state.analyse_symptom_note = _sna.analyse_symptom_note
        log.info("Symptom note analyser ready (reuses DeBERTa — no extra RAM).")
    except Exception as exc:
        log.warning(f"Symptom note analyser failed to load ({exc}) — notes stored but not scored.")
        _state.analyse_symptom_note = None

    log.info("Loading culprit food cross-encoder model...")
    # ─────────────────────────────────────────────────────────────
    # Culprit food model (ONLY ONE CLEAN BLOCK)
    # ─────────────────────────────────────────────────────────────

    log.info("Loading culprit food cross-encoder model...")

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small")
        model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-small")

        class CrossEncoder:
            def predict(self, pairs):
                inputs = tokenizer(
                    [p[0] for p in pairs],
                    [p[1] for p in pairs],
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                outputs = model(**inputs)
                return outputs.logits.detach().tolist()

        _state.culprit_cross_encoder = CrossEncoder()
        log.info("✅ Culprit model loaded successfully")

    except Exception as e:
        _state.culprit_cross_encoder = None
        log.error(f"❌ Failed to load culprit model: {e}")


    # ─────────────────────────────────────────────────────────────
    # IMPORTANT: KEEP THIS EXACTLY HERE
    # ─────────────────────────────────────────────────────────────
    yield
    log.info("Shutdown.")

    

app = FastAPI(
    title    = "Gut Health API",
    version  = "3.0.0",
    lifespan = lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
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
# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 1 ── POST /food/parse
#  Natural meal text  →  USDA food IDs + descriptions
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class FoodParseRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3, max_length=1000,
        description="Any natural language meal description.",
        examples=["I had grilled chicken breast with brown rice and broccoli"],
    )
    top_k: int = Field(
        default=3, ge=1, le=10,
        description="Number of USDA candidate matches per detected food.",
    )


class USDAMatch(BaseModel):
    rank:             int
    usda_id:          int
    usda_description: str
    similarity:       float


class DetectedFood(BaseModel):
    raw_food:         str
    ner_confidence:   float
    normalised_name:  str
    usda_id:          int   = Field(..., description="Use this ID in POST /log/food")
    usda_description: str
    usda_similarity:  float
    usda_top_matches: List[USDAMatch]
    # ── context fields ────────────────────────────────────────────────────────
    meal_type: Optional[str] = Field(
        None,
        description="Meal type detected from text: Breakfast | Lunch | Dinner | Snack | null",
    )
    quantity: Optional[float] = Field(
        None,
        description="Numeric quantity found near this food in the text (e.g. 200).",
    )
    unit: Optional[str] = Field(
        None,
        description="Unit found near this food (e.g. 'g', 'cup', 'piece'). null when no unit present.",
    )
    logged_at: str = Field(
        ...,
        description="UTC ISO-8601 timestamp of when this parse was performed.",
    )


class FoodParseResponse(BaseModel):
    input_text:     str
    foods_detected: int
    meal_type:      Optional[str] = Field(
        None,
        description="Meal type detected from the full text, or null if not mentioned.",
    )
    logged_at:      str = Field(
        ...,
        description="UTC ISO-8601 timestamp of the parse call.",
    )
    results:        List[DetectedFood]


@app.post("/food/parse", response_model=FoodParseResponse,
          summary="Convert natural meal text to USDA food IDs")
def food_parse(req: FoodParseRequest) -> FoodParseResponse:
    """
    Converts free-form meal text into structured USDA food records.

    **3-stage pipeline:**
    1. `InstaFoodRoBERTa-NER` — extracts food words from the sentence
    2. `Flan-T5-base` — normalises to USDA-style food name
    3. `all-MiniLM-L6-v2` — semantic search over the full USDA dataset

    Pass the returned `usda_id` values into `POST /log/food`.

    **Example:**
    ```json
    Request:  { "text": "I ate oatmeal with banana and Greek yogurt", "top_k": 3 }
    Response: { "foods_detected": 3, "results": [{ "usda_id": 8121, ... }] }
    ```
    """
    if not _state.usda_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"USDA pipeline unavailable. Error: {_state.usda_error or 'unknown'}",
        )
    try:
        raw = _state.text_to_usda(req.text, top_k=req.top_k)
    except Exception as exc:
        log.exception(f"Pipeline error for: {req.text!r}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    if not raw:
        return FoodParseResponse(
            input_text     = req.text,
            foods_detected = 0,
            meal_type      = None,
            logged_at      = utc_now_iso(),
            results        = [],
        )

    # meal_type and logged_at are identical across all items — take from first result
    meal_type = raw[0].get("meal_type")
    logged_at = raw[0].get("logged_at", utc_now_iso())

    return FoodParseResponse(
        input_text     = req.text,
        foods_detected = len(raw),
        meal_type      = meal_type,
        logged_at      = logged_at,
        results = [
            DetectedFood(
                raw_food         = r["raw_food"],
                ner_confidence   = r["ner_confidence"],
                normalised_name  = r["normalised_name"],
                usda_id          = r["usda_id"],
                usda_description = r["usda_description"],
                usda_similarity  = r["usda_similarity"],
                usda_top_matches = [USDAMatch(**m) for m in r["usda_top_matches"]],
                meal_type        = r.get("meal_type"),
                quantity         = r.get("quantity"),
                unit             = r.get("unit"),
                logged_at        = r.get("logged_at", logged_at),
            )
            for r in raw
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 2 ── POST /score
#  Onboarding questionnaire  →  baseline digestion score (call ONCE)
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/score", response_model=DigestiveResult,
          summary="Calculate onboarding digestion score (call once at onboarding)")
def score(data: DigestiveInput) -> DigestiveResult:
    """
    Calculates the user's **baseline** digestive health score (0–100) from the onboarding questionnaire.

    **Call this once when the user completes onboarding. Store the returned `score`.**
    All future updates come from `POST /log/food` and `POST /log/symptom`.

    Rule-based engine — instant, no ML, no USDA lookup needed.

    ---
    **Allowed `foods`:** `Dairy` · `Gluten` · `Spicy` · `Fried` · `Sugar` · `Caffeine` · `Processed Food` · `Other`

    **Allowed `symptoms`:** `Bloating` · `Gas` · `Abdominal Pain` · `Nausea` · `Heartburn` · `Diarrhea` · `Constipation` · `Fatigue`

    ```json
    {
      "gender": "Female", "age": 34, "sleep_hours": 5.5, "weight_kg": 68,
      "foods": ["Spicy", "Fried", "Dairy"],
      "symptoms": ["Bloating", "Heartburn"]
    }
    ```
    """
    return calculate_score(data)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 3 ── POST /log/food
#  Log a meal  →  updated score via USDA nutrients + HuggingFace
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class FoodItem(BaseModel):
    usda_id:  int   = Field(..., description="USDA food ID from POST /food/parse")
    quantity: float = Field(..., gt=0, description="Amount consumed (numeric)")
    unit:     str   = Field(
        ...,
        description="Unit of quantity.",
        examples=["g", "ml", "pieces", "cup", "tbsp", "oz"],
    )


class FoodLogRequest(BaseModel):
    current_score: int = Field(
        ..., ge=0, le=100,
        description="User's current digestion score (from /score or last /log call).",
    )
    meal_type: str = Field(
        ...,
        description="Type of meal.",
        examples=["Breakfast", "Lunch", "Dinner", "Snack"],
    )
    foods: List[FoodItem] = Field(
        ..., min_length=1,
        description="List of foods eaten in this meal. Each needs a usda_id, quantity, and unit.",
    )


class NutrientProfileOut(BaseModel):
    # All 16 USDA columns — scaled to actual portion
    calories:     Optional[float] = None
    protein:      Optional[float] = None
    total_fat:    Optional[float] = None
    carbs:        Optional[float] = None
    sodium:       Optional[float] = None
    sat_fat:      Optional[float] = None
    cholesterol:  Optional[float] = None
    sugar:        Optional[float] = None
    calcium:      Optional[float] = None
    iron:         Optional[float] = None
    potassium:    Optional[float] = None
    vitamin_c:    Optional[float] = None
    vitamin_e:    Optional[float] = None
    vitamin_d:    Optional[float] = None
    portion_grams: float


class NutrientImpactOut(BaseModel):
    nutrient:  str
    value:     float
    unit:      str
    delta:     int    = Field(..., description="Score points contributed by this nutrient (+/-)")
    note:      str
    direction: str    = Field(..., description="'beneficial' or 'harmful'")


class FoodLogDetail(BaseModel):
    usda_id:           int
    food_description:  str
    portion_grams:     float
    score_modifier:    int
    nutrient_modifier: int
    hf_modifier:       int
    digestibility:     str
    hf_confidence:     float
    meal_type_note:    str
    nutrient_profile:  NutrientProfileOut
    nutrient_impacts:  List[NutrientImpactOut]   = Field(..., description="Per-nutrient score breakdown")
    nutrient_notes:    List[str]
    hf_note:           str


class FoodLogResponse(BaseModel):
    meal_type:      str
    previous_score: int
    score_modifier: int   = Field(..., description="Total score change from this meal")
    updated_score:  int
    grade:          str
    summary:        str
    food_details:   List[FoodLogDetail]
    recommendations: List[str]


_VALID_MEAL_TYPES = {"Breakfast", "Lunch", "Dinner", "Snack"}


@app.post("/log/food", response_model=FoodLogResponse,
          summary="Log a meal and get an updated digestion score")
def log_food(req: FoodLogRequest) -> FoodLogResponse:
    """
    Log what the user ate and get an **updated digestion score**.

    **How the score changes:**
    - Each food is looked up in the USDA dataset using its `usda_id`
    - Nutritional values are **scaled to the actual quantity** (e.g. 250g portion ≠ 100g USDA default)
    - A HuggingFace zero-shot classifier assesses each food's digestibility
    - Meal type applies a context multiplier (Dinner = 1.15×, Breakfast = 0.85×)
    - Final modifier = (nutrient score + HF score) × meal type multiplier

    **Workflow:**
    ```
    1. User types meal text
       → POST /food/parse  →  get usda_id per food

    2. Log the meal with quantities
       → POST /log/food {
           "current_score": 68,
           "meal_type": "Dinner",
           "foods": [
             { "usda_id": 5064, "quantity": 200, "unit": "g" },
             { "usda_id": 20037, "quantity": 150, "unit": "g" }
           ]
         }
    ```

    **Allowed `unit` values:** `g` · `gram` · `kg` · `ml` · `l` · `oz` · `lb` · `piece` · `pieces` · `cup` · `tbsp` · `tsp` · `serving`

    **Allowed `meal_type` values:** `Breakfast` · `Lunch` · `Dinner` · `Snack`
    """
    if req.meal_type not in _VALID_MEAL_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid meal_type '{req.meal_type}'. Allowed: {sorted(_VALID_MEAL_TYPES)}",
        )

    if not _state.nutrition_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Nutrition scorer unavailable. Error: {_state.nutrition_error or 'unknown'}",
        )

    items = [{"usda_id": f.usda_id, "quantity": f.quantity, "unit": f.unit} for f in req.foods]

    try:
        total_modifier, raw_details = _state.analyse_food_log_batch(
            items     = items,
            meal_type = req.meal_type,
            usda_df   = _state.usda_df,
            id_col    = _state.id_col,
            desc_col  = _state.desc_col,
        )
    except Exception as exc:
        log.exception("Food log analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis error: {exc}")

    updated_score = max(0, min(100, req.current_score + total_modifier))

    # Collect unique recommendations from nutrient notes
    seen_recs: set[str] = set()
    recommendations: list[str] = []
    for d in raw_details:
        for note in d.nutrient_notes:
            if note not in seen_recs:
                seen_recs.add(note)
                recommendations.append(note)

    food_details = [
        FoodLogDetail(
            usda_id           = d.usda_id,
            food_description  = d.food_description,
            portion_grams     = d.portion_grams,
            score_modifier    = d.score_modifier,
            nutrient_modifier = d.nutrient_modifier,
            hf_modifier       = d.hf_modifier,
            digestibility     = d.digestibility,
            hf_confidence     = d.hf_confidence,
            meal_type_note    = d.meal_type_note,
            nutrient_profile  = NutrientProfileOut(
                calories      = d.nutrient_profile.calories,
                protein       = d.nutrient_profile.protein,
                total_fat     = d.nutrient_profile.total_fat,
                carbs         = d.nutrient_profile.carbs,
                sodium        = d.nutrient_profile.sodium,
                sat_fat       = d.nutrient_profile.sat_fat,
                cholesterol   = d.nutrient_profile.cholesterol,
                sugar         = d.nutrient_profile.sugar,
                calcium       = d.nutrient_profile.calcium,
                iron          = d.nutrient_profile.iron,
                potassium     = d.nutrient_profile.potassium,
                vitamin_c     = d.nutrient_profile.vitamin_c,
                vitamin_e     = d.nutrient_profile.vitamin_e,
                vitamin_d     = d.nutrient_profile.vitamin_d,
                portion_grams = d.nutrient_profile.portion_grams,
            ),
            nutrient_impacts = [
                NutrientImpactOut(
                    nutrient  = i.nutrient,
                    value     = i.value,
                    unit      = i.unit,
                    delta     = i.delta,
                    note      = i.note,
                    direction = i.direction,
                )
                for i in d.nutrient_impacts
            ],
            nutrient_notes = d.nutrient_notes,
            hf_note        = d.hf_note,
        )
        for d in raw_details
    ]

    return FoodLogResponse(
        meal_type      = req.meal_type,
        previous_score = req.current_score,
        score_modifier = total_modifier,
        updated_score  = updated_score,
        grade          = _grade(updated_score),
        summary        = _grade_summary(updated_score),
        food_details   = food_details,
        recommendations= recommendations,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 4 ── POST /log/symptom
#  Log one or more symptoms + optional free-text note → updated score
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

_VALID_SYMPTOMS = {
    "Bloating", "Abdominal Pain", "Nausea", "Constipation",
    "Heartburn", "Gas", "Fatigue", "Acid Reflux", "Cramps", "Diarrhea",
}
_VALID_SEVERITIES = {"Mild", "Moderate", "Severe"}


class SymptomLogRequest(BaseModel):
    current_score: int = Field(
        ..., ge=0, le=100,
        description="User's current digestion score.",
    )
    symptom: List[str] = Field(
        ...,
        min_length=1,
        description=(
            "One or more symptoms to log simultaneously. "
            "All share the same severity and timestamp."
        ),
        examples=[["Bloating", "Nausea"], ["Heartburn"]],
    )
    severity: str = Field(
        ...,
        description="Severity that applies to ALL listed symptoms.",
        examples=["Mild", "Moderate", "Severe"],
    )
    note: Optional[str] = Field(
        default=None,
        max_length=1000,
        description=(
            "Optional free-text description of how the user feels. "
            "The system NLP-analyses this note using DeBERTa zero-shot classification to: "
            "(1) detect any additional gut symptoms not explicitly listed — added at 50% penalty weight; "
            "(2) flag novel symptoms outside the 10-label taxonomy (e.g. headache, dizziness) "
            "with a small flat penalty and a descriptive hint."
        ),
        examples=["I also feel a strong headache and my stomach feels very tight"],
    )
    logged_at: Optional[datetime] = Field(
        default=None,
        description=(
            "ISO 8601 datetime when the symptom(s) occurred. "
            "If omitted, server uses current UTC time."
        ),
        examples=["2024-03-15T22:30:00"],
    )


# ── Per-symptom breakdown row ─────────────────────────────────────────────────

class SymptomDetail(BaseModel):
    symptom:       str
    score_penalty: int    = Field(..., description="Points deducted by this symptom (always ≤ 0)")
    severity_note: str
    time_note:     str
    clinical_note: str
    source:        str    = Field(
        ...,
        description="'user_logged' — explicitly submitted | 'note_detected' — inferred from note text",
    )
    confidence:    Optional[float] = Field(
        None,
        description="NLI confidence 0–1 for note-detected symptoms. null for user_logged.",
    )


# ── Note analysis block in the response ──────────────────────────────────────

class NoteAnalysisOut(BaseModel):
    analysed:               bool
    detected_symptom_count: int
    novel_symptom_flag:     bool
    novel_symptom_hint:     str
    additional_penalty:     int
    analysis_summary:       str
    raw_scores:             dict = Field(
        default_factory=dict,
        description="NLI score per symptom label from note analysis (0–1).",
    )


class SymptomLogResponse(BaseModel):
    symptoms:            List[str]           = Field(..., description="All symptoms scored (user-logged + note-detected)")
    severity:            str
    logged_at:           str                 = Field(..., description="ISO timestamp used for time-of-day scoring")
    previous_score:      int
    score_penalty:       int                 = Field(..., description="Total points deducted across all symptoms")
    updated_score:       int
    grade:               str
    summary:             str
    per_symptom_details: List[SymptomDetail] = Field(..., description="Breakdown per symptom with individual penalties")
    note_analysis:       Optional[NoteAnalysisOut] = Field(
        None,
        description="Present only when a note was provided.",
    )


@app.post("/log/symptom", response_model=SymptomLogResponse,
          summary="Log one or more symptoms (+ optional note) and get an updated digestion score")
def log_symptom(req: SymptomLogRequest) -> SymptomLogResponse:
    """
    Log one or more digestive symptoms and get an **updated score**.

    ---

    ### Multi-symptom support
    Pass any number of symptoms in the `symptom` list — all share the same
    `severity` and `logged_at`. Each symptom is scored independently and
    all penalties are summed.

    ```json
    {
      "current_score": 80,
      "symptom": ["Bloating", "Nausea"],
      "severity": "Moderate",
      "logged_at": "2024-03-15T22:30:00"
    }
    ```

    ---

    ### Note-based symptom detection
    Include a free-text `note` describing how the user feels.
    The DeBERTa NLI model (already loaded, zero extra RAM) analyses the note to:

    1. **Detect additional known symptoms** not explicitly listed.
       e.g. note says *"stomach is cramping tight"* → `Cramps` detected,
       added at **50% penalty weight** (inferred, not confirmed).

    2. **Flag novel symptoms** outside the gut taxonomy (e.g. *"strong headache"*).
       These get a small flat penalty (−3) and a descriptive `novel_symptom_hint`
       so the frontend can surface them to the user.

    ```json
    {
      "current_score": 80,
      "symptom": ["Bloating"],
      "severity": "Mild",
      "note": "I also feel a strong headache and my stomach is cramping",
      "logged_at": "2024-03-15T22:30:00"
    }
    ```

    The response `per_symptom_details` marks each entry as
    `user_logged` or `note_detected` so the frontend can display them differently.

    ---

    ### How the score changes (per symptom)
    - Each symptom has a clinical base penalty (e.g. `Diarrhea` = −12, `Gas` = −5)
    - Severity multiplier: `Mild` × 0.5, `Moderate` × 1.0, `Severe` × 1.6
    - Time-of-day multiplier: night (10 PM–6 AM) = × 1.25
    - Note-detected symptoms: same formula but × 0.50 confidence weight
    - Final `score_penalty` = sum of all individual symptom penalties

    **Allowed `symptom` values:**
    `Bloating` · `Abdominal Pain` · `Nausea` · `Constipation` · `Heartburn` ·
    `Gas` · `Fatigue` · `Acid Reflux` · `Cramps` · `Diarrhea`

    **Allowed `severity` values:** `Mild` · `Moderate` · `Severe`
    """
    # ── Validate every symptom in the list ───────────────────────────────────
    invalid = [s for s in req.symptom if s not in _VALID_SYMPTOMS]
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

    if not _state.nutrition_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Scorer unavailable. Error: {_state.nutrition_error or 'unknown'}",
        )

    # Deduplicate while preserving order
    seen_syms: set[str] = set()
    deduped: list[str] = []
    for s in req.symptom:
        if s not in seen_syms:
            seen_syms.add(s)
            deduped.append(s)

    logged_at = req.logged_at or datetime.now(timezone.utc)

    # ── Score each explicitly logged symptom ─────────────────────────────────
    per_symptom_details: list[SymptomDetail] = []
    total_penalty = 0

    for sym in deduped:
        try:
            result = _state.analyse_symptom_log(
                symptom   = sym,
                severity  = req.severity,
                logged_at = logged_at,
            )
        except Exception as exc:
            log.exception(f"Symptom log analysis failed for {sym!r}")
            raise HTTPException(status_code=500, detail=f"Analysis error for {sym!r}: {exc}")

        total_penalty += result.score_penalty
        per_symptom_details.append(SymptomDetail(
            symptom       = result.symptom,
            score_penalty = result.score_penalty,
            severity_note = result.severity_note,
            time_note     = result.time_note,
            clinical_note = result.clinical_note,
            source        = "user_logged",
            confidence    = None,
        ))

    # ── Analyse the optional note ─────────────────────────────────────────────
    note_analysis_out: Optional[NoteAnalysisOut] = None

    if req.note and req.note.strip():
        if _state.analyse_symptom_note is not None:
            try:
                note_result = _state.analyse_symptom_note(
                    note_text      = req.note,
                    already_logged = deduped,
                    severity       = req.severity,
                )

                # Add note-detected symptoms to breakdown and score
                for ds in note_result.detected_symptoms:
                    total_penalty += ds.penalty
                    deduped.append(ds.symptom)
                    per_symptom_details.append(SymptomDetail(
                        symptom       = ds.symptom,
                        score_penalty = ds.penalty,
                        severity_note = (
                            f"Inferred from note at {ds.confidence:.0%} confidence. "
                            f"Penalty reduced to 50% weight."
                        ),
                        time_note     = "Time-of-day multiplier not applied to note-inferred symptoms.",
                        clinical_note = ds.note,
                        source        = "note_detected",
                        confidence    = ds.confidence,
                    ))

                # Novel-symptom flat penalty (total includes detected penalties already,
                # so subtract them to avoid double-counting)
                novel_penalty = (
                    note_result.total_additional_penalty
                    - sum(ds.penalty for ds in note_result.detected_symptoms)
                )
                total_penalty += novel_penalty

                note_analysis_out = NoteAnalysisOut(
                    analysed               = True,
                    detected_symptom_count = len(note_result.detected_symptoms),
                    novel_symptom_flag     = note_result.novel_symptom_flag,
                    novel_symptom_hint     = note_result.novel_symptom_hint,
                    additional_penalty     = note_result.total_additional_penalty,
                    analysis_summary       = note_result.analysis_summary,
                    raw_scores             = note_result.raw_scores,
                )

            except Exception as exc:
                log.warning(f"Note analysis failed (non-fatal): {exc}")
                note_analysis_out = NoteAnalysisOut(
                    analysed               = False,
                    detected_symptom_count = 0,
                    novel_symptom_flag     = False,
                    novel_symptom_hint     = "",
                    additional_penalty     = 0,
                    analysis_summary       = f"Note stored but analysis failed: {exc}",
                    raw_scores             = {},
                )
        else:
            # Analyser not loaded — acknowledge note without scoring it
            note_analysis_out = NoteAnalysisOut(
                analysed               = False,
                detected_symptom_count = 0,
                novel_symptom_flag     = False,
                novel_symptom_hint     = "",
                additional_penalty     = 0,
                analysis_summary       = (
                    "Note received and stored. "
                    "NLP analyser not available — note was not scored."
                ),
                raw_scores             = {},
            )

    # ── Build final response ──────────────────────────────────────────────────
    updated_score = max(0, min(100, req.current_score + total_penalty))

    logged_at_str = (
        logged_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        if hasattr(logged_at, "strftime")
        else str(logged_at)
    )

    return SymptomLogResponse(
        symptoms             = deduped,
        severity             = req.severity,
        logged_at            = logged_at_str,
        previous_score       = req.current_score,
        score_penalty        = total_penalty,
        updated_score        = updated_score,
        grade                = _grade(updated_score),
        summary              = _grade_summary(updated_score),
        per_symptom_details  = per_symptom_details,
        note_analysis        = note_analysis_out,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 5 ── POST /food/text-to-id
#  Natural food text  →  best-match USDA ID   (clean alias of /food/parse)
#  Returns only the top-1 USDA ID per food — no top_k candidates noise.
#  Intended as the lightweight "I just want the ID" call before /log/food.
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class TextToIdRequest(BaseModel):
    text: str = Field(
        ...,
        min_length  = 3,
        max_length  = 1000,
        description = "Any natural language description of a food or meal.",
        examples    = ["grilled salmon with steamed broccoli"],
    )


class FoodIdItem(BaseModel):
    raw_food:         str   = Field(..., description="Food word extracted from text by NER")
    ner_confidence:   float = Field(..., description="NER model confidence 0–1")
    normalised_name:  str   = Field(..., description="USDA-style name produced by Flan-T5")
    usda_id:          int   = Field(..., description="Best-match USDA food ID — use in /log/food or /food/lookup/{id}")
    usda_description: str   = Field(..., description="Full USDA description for this ID")
    similarity:       float = Field(..., description="Semantic similarity score 0–1")
    # ── context fields ────────────────────────────────────────────────────────
    quantity: Optional[float] = Field(
        None,
        description="Numeric quantity found near this food in the text (e.g. 200). null if not mentioned.",
    )
    unit: Optional[str] = Field(
        None,
        description="Unit found near this food (e.g. 'g', 'cup', 'piece'). null when no unit present.",
    )


class TextToIdResponse(BaseModel):
    input_text:     str
    foods_detected: int
    meal_type:      Optional[str] = Field(
        None,
        description="Meal type detected from the full text: Breakfast | Lunch | Dinner | Snack | null",
    )
    logged_at:      str = Field(
        ...,
        description="UTC ISO-8601 timestamp of when this parse was performed.",
    )
    food_list:      dict[str, str] = Field(
        ...,
        description='Quick lookup: raw food name → quantity string (e.g. {"salmon": "300g", "broccoli": "200g"}). '
                    'Value is "Xunit" when both are present, "X" when quantity only, or "-" when neither.',
    )
    foods:          List[FoodIdItem]


@app.post(
    "/food/text-to-id",
    response_model = TextToIdResponse,
    summary        = "Convert natural food text to USDA IDs (top-1 match per food)",
)
def food_text_to_id(req: TextToIdRequest) -> TextToIdResponse:
    """
    Converts free-form food or meal text into the best-matching **USDA food ID** per food.

    Unlike `/food/parse` which returns top-k candidates, this endpoint returns
    **only the single best USDA ID** per detected food — cleaner for use with `/log/food`.

    **Pipeline (from `food_text_to_usda.py`):**
    1. `InstaFoodRoBERTa-NER` — extracts food words from the text
    2. `Flan-T5-base` — normalises each food name to USDA-style
    3. `all-MiniLM-L6-v2` — semantic search → best USDA row by cosine similarity

    **Typical usage:**
    ```
    POST /food/text-to-id  { "text": "I had oatmeal and a banana" }
    → foods: [{ "usda_id": 8121, ... }, { "usda_id": 9040, ... }]

    then:
    POST /log/food  { "current_score": 70, "meal_type": "Breakfast",
                      "foods": [{ "usda_id": 8121, "quantity": 250, "unit": "g" },
                                { "usda_id": 9040, "quantity": 1,   "unit": "piece" }] }
    ```
    """
    if not _state.usda_ready:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = f"USDA pipeline unavailable. Error: {_state.usda_error or 'unknown'}",
        )

    try:
        raw = _state.text_to_usda(req.text, top_k=1)   # top_k=1 → best match only
    except Exception as exc:
        log.exception(f"text-to-id pipeline error for: {req.text!r}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    if not raw:
        return TextToIdResponse(
            input_text     = req.text,
            foods_detected = 0,
            meal_type      = None,
            logged_at      = utc_now_iso(),
            food_list      = {},
            foods          = [],
        )

    meal_type = raw[0].get("meal_type")
    logged_at = raw[0].get("logged_at", utc_now_iso())

    def _qty_str(r: dict) -> str:
        q, u = r.get("quantity"), r.get("unit")
        if q is not None and u is not None:
            return f"{int(q) if q == int(q) else q}{u}"
        if q is not None:
            return str(int(q) if q == int(q) else q)
        return "-"

    food_list = {r["raw_food"]: _qty_str(r) for r in raw}

    return TextToIdResponse(
        input_text     = req.text,
        foods_detected = len(raw),
        meal_type      = meal_type,
        logged_at      = logged_at,
        food_list      = food_list,
        foods = [
            FoodIdItem(
                raw_food         = r["raw_food"],
                ner_confidence   = r["ner_confidence"],
                normalised_name  = r["normalised_name"],
                usda_id          = r["usda_id"],
                usda_description = r["usda_description"],
                similarity       = r["usda_similarity"],
                quantity         = r.get("quantity"),
                unit             = r.get("unit"),
            )
            for r in raw
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 6 ── GET /food/lookup/{usda_id}
#  USDA food ID  →  food name + Calories, Protein, Carbs, Fat
#  Data served directly from the USDA dataset loaded in memory.
#  No ML model involved — pure DataFrame lookup, instant response.
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class FoodLookupResponse(BaseModel):
    usda_id:     int
    name:        str   = Field(..., description="USDA food description / name")
    calories:    Optional[float] = Field(None, description="kcal per 100g")
    protein:     Optional[float] = Field(None, description="Protein in grams per 100g")
    carbs:       Optional[float] = Field(None, description="Carbohydrates in grams per 100g")
    fat:         Optional[float] = Field(None, description="Total fat in grams per 100g")
    # Bonus fields from the USDA dataset if available
    sodium:      Optional[float] = Field(None, description="Sodium in mg per 100g")
    sugar:       Optional[float] = Field(None, description="Sugar in grams per 100g")
    sat_fat:     Optional[float] = Field(None, description="Saturated fat in grams per 100g")
    cholesterol: Optional[float] = Field(None, description="Cholesterol in mg per 100g")
    calcium:     Optional[float] = Field(None, description="Calcium in mg per 100g")
    iron:        Optional[float] = Field(None, description="Iron in mg per 100g")
    potassium:   Optional[float] = Field(None, description="Potassium in mg per 100g")
    vitamin_c:   Optional[float] = Field(None, description="Vitamin C in mg per 100g")
    vitamin_e:   Optional[float] = Field(None, description="Vitamin E in mg per 100g")
    vitamin_d:   Optional[float] = Field(None, description="Vitamin D in µg per 100g")
    per:         str = "100g"


# Reuse the column resolver from nutrition_scorer
def _resolve_usda_col(df, key: str) -> Optional[str]:
    """Resolve a nutrient key to actual column name using the same alias table as nutrition_scorer."""
    from nutrition_scorer import _resolve_col
    return _resolve_col(df, key)


def _safe_float(val) -> Optional[float]:
    try:
        import math
        v = float(val)
        return round(v, 3) if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


@app.get(
    "/food/lookup/{usda_id}",
    response_model = FoodLookupResponse,
    summary        = "Look up food name and nutrition facts by USDA ID",
)
def food_lookup(usda_id: int) -> FoodLookupResponse:
    """
    Retrieve the **food name and macro/micronutrient values** for a USDA food ID.

    All values are **per 100g** as stored in the USDA dataset.
    Data is served directly from the in-memory USDA DataFrame — no ML, instant response.

    **Primary fields always returned (if available in dataset):**
    - `name` — USDA food description
    - `calories` — kcal per 100g
    - `protein` — grams per 100g
    - `carbs` — grams per 100g
    - `fat` — grams per 100g

    **All 16 USDA nutrient columns returned** when available in the dataset.

    **Example:**
    ```
    GET /food/lookup/8121
    → {
        "usda_id": 8121,
        "name": "OATMEAL,INST,FORT,PLAIN,PREP W/WATER",
        "calories": 71.0,
        "protein": 2.5,
        "carbs": 12.0,
        "fat": 1.5,
        "sodium": 218.0,
        "sugar": 0.3,
        ...
      }
    ```

    Get a USDA ID from `POST /food/text-to-id` or `POST /food/parse`.
    """
    if not _state.usda_ready:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = f"USDA dataset unavailable. Error: {_state.usda_error or 'unknown'}",
        )

    df = _state.usda_df

    # Resolve id column
    id_col = _state.id_col
    mask   = df[id_col].astype(str) == str(usda_id)

    if not mask.any():
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = f"USDA food ID {usda_id} not found in dataset.",
        )

    row = df[mask].iloc[0]

    # Description / name
    desc_col = _state.desc_col
    name     = str(row[desc_col]) if desc_col in df.columns else "Unknown"

    def gv(key: str) -> Optional[float]:
        col = _resolve_usda_col(df, key)
        if col is None:
            return None
        return _safe_float(row.get(col))

    return FoodLookupResponse(
        usda_id     = usda_id,
        name        = name,
        calories    = gv("calories"),
        protein     = gv("protein"),
        carbs       = gv("carbs"),
        fat         = gv("total_fat"),
        sodium      = gv("sodium"),
        sugar       = gv("sugar"),
        sat_fat     = gv("sat_fat"),
        cholesterol = gv("cholesterol"),
        calcium     = gv("calcium"),
        iron        = gv("iron"),
        potassium   = gv("potassium"),
        vitamin_c   = gv("vitamin_c"),
        vitamin_e   = gv("vitamin_e"),
        vitamin_d   = gv("vitamin_d"),
        per         = "100g",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 7 ── GET /food/tags/{usda_id}
#  USDA food ID → meal category tags
#  Tags: Dairy | Gluten | Spicy | Fried | Sugar | Caffeine | Processed Food | Others
#
#  Uses DeBERTa-v3-base zero-shot NLI (reused from nutrition_scorer — no extra RAM)
#  plus nutrient-based heuristic boosts for higher accuracy.
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class TagScoreOut(BaseModel):
    tag:              str
    score:            float = Field(..., description="Final confidence score 0–1 (NLI + heuristic)")
    nli_score:        float = Field(..., description="Raw DeBERTa NLI score before heuristic boosts")
    heuristic_boost:  float = Field(..., description="Total boost applied by nutrient rules")
    active:           bool  = Field(..., description="True if score ≥ 0.35 threshold")
    reasons:          List[str] = Field(..., description="Explanation of how this score was reached")


class FoodTagResponse(BaseModel):
    usda_id:              int
    food_description:     str
    primary_tag:          str   = Field(
        ..., description="Highest-confidence tag. 'Others' if nothing cleared the threshold."
    )
    active_tags:          List[str] = Field(
        ..., description="All tags that cleared the 0.35 confidence threshold."
    )
    tag_scores:           List[TagScoreOut] = Field(
        ..., description="Full score breakdown for every tag, sorted highest first."
    )
    confidence:           float = Field(..., description="Primary tag confidence (0–1)")
    is_others:            bool  = Field(..., description="True when no specific tag was matched.")
    classification_method:str   = Field(
        ..., description="'nli+heuristic' | 'heuristic_only'"
    )
    note:                 str


@app.get(
    "/food/tags/{usda_id}",
    response_model = FoodTagResponse,
    summary        = "Classify a USDA food ID into meal category tags",
)
def food_tags(usda_id: int) -> FoodTagResponse:
    """
    Classifies a USDA food into one or more **meal category tags**.

    **Available tags:**
    `Dairy` · `Gluten` · `Spicy` · `Fried` · `Sugar` · `Caffeine` · `Processed Food` · `Others`

    A food can have **multiple active tags** (e.g. a cheesy pizza → Dairy + Gluten + Processed Food).
    `Others` is assigned only when no tag clears the 0.35 confidence threshold.

    ---

    ### How classification works

    **Layer 1 — DeBERTa NLI (zero-shot)**
    The USDA food description is the NLI premise.
    Each tag is expressed as a natural-language hypothesis, e.g.:
    - `"Dairy"` → *"This food is a dairy product or contains milk, cheese, cream, yogurt, or butter."*
    - `"Fried"` → *"This food is fried, deep-fried, or cooked by submerging in hot oil."*

    The DeBERTa model already loaded by the nutrition scorer is **reused — no extra RAM or startup cost.**

    **Layer 2 — Nutrient heuristic boosts**
    Hard rules adjust scores using USDA nutrient values, for example:
    - Calcium ≥ 150 mg/100g → Dairy score +0.20
    - Sugar ≥ 20 g/100g → Sugar score +0.15
    - Sodium ≥ 400 mg/100g → Processed Food score +0.15

    Final score = NLI score + heuristic boost, clamped to [0, 1].

    ---

    ### Example responses
    ```
    GET /food/tags/1009  (Butter, salted)
    → primary_tag: "Dairy"
    → active_tags: ["Dairy", "Processed Food"]

    GET /food/tags/11819  (Pepper, hot chili)
    → primary_tag: "Spicy"
    → active_tags: ["Spicy"]

    GET /food/tags/19335  (Sugars, granulated)
    → primary_tag: "Sugar"
    → active_tags: ["Sugar"]
    ```

    Get a `usda_id` from `POST /food/parse` or `POST /food/text-to-id`.
    """
    if not _state.usda_ready:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = f"USDA dataset unavailable. Error: {_state.usda_error or 'unknown'}",
        )

    try:
        from food_tag_classifier import classify_food_tags
        result = classify_food_tags(
            usda_id  = usda_id,
            usda_df  = _state.usda_df,
            id_col   = _state.id_col,
            desc_col = _state.desc_col,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = str(exc),
        )
    except Exception as exc:
        log.exception(f"food_tags classification failed for usda_id={usda_id}")
        raise HTTPException(
            status_code = 500,
            detail      = f"Classification error: {exc}",
        )

    return FoodTagResponse(
        usda_id               = result.usda_id,
        food_description      = result.food_description,
        primary_tag           = result.primary_tag,
        active_tags           = result.active_tags,
        tag_scores            = [
            TagScoreOut(
                tag             = ts.tag,
                score           = ts.score,
                nli_score       = ts.nli_score,
                heuristic_boost = ts.heuristic_boost,
                active          = ts.active,
                reasons         = ts.reasons,
            )
            for ts in result.tag_scores
        ],
        confidence            = result.confidence,
        is_others             = result.is_others,
        classification_method = result.classification_method,
        note                  = result.note,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 8 ── POST /predict/food-symptom
#  Given recent food logs + symptom logs for a user,
#  predict which food(s) most likely caused each symptom.
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────════════════════════════════

from food_symptom_predictor import (
    FoodLogEntry,
    SymptomLogEntry,
    FoodCausationResult,
    SymptomPrediction,
    SYMPTOM_WINDOWS,
)


# ── Request models ────────────────────────────────────────────────────────────

class FoodLogInput(BaseModel):
    """One food log row from your database."""
    user_id:    str   = Field(..., description="User identifier")
    usda_id:    int   = Field(..., description="USDA food ID (from /food/text-to-id)")
    logged_at:  datetime = Field(..., description="When the food was eaten (ISO 8601)")
    quantity_g: float = Field(..., gt=0, description="Quantity eaten in grams")


class SymptomLogInput(BaseModel):
    """One symptom log row from your database."""
    user_id:   str      = Field(..., description="User identifier (must match food log user_id)")
    symptom:   str      = Field(..., description="Symptom name",
                                examples=["Bloating","Heartburn","Diarrhea"])
    logged_at: datetime = Field(..., description="When the symptom was felt (ISO 8601)")
    intensity: str      = Field(..., description="Symptom intensity",
                                examples=["Mild","Moderate","Severe"])


class FoodSymptomPredictRequest(BaseModel):
    food_logs:    List[FoodLogInput]    = Field(
        ..., min_length=1,
        description=(
            "Food log entries from your database for this user. "
            "Include all meals from the past 48 hours for best coverage. "
            "quantity_g must already be in grams (convert from pieces/cups/ml before sending)."
        ),
    )
    symptom_logs: List[SymptomLogInput] = Field(
        ..., min_length=1,
        description="Symptom log entries to analyse. Each symptom is scored independently.",
    )


# ── Response models ───────────────────────────────────────────────────────────

class NutrientRiskItem(BaseModel):
    nutrient:     str
    value:        float
    unit:         str
    delta:        int
    note:         str
    direction:    str


class FoodCausationOut(BaseModel):
    usda_id:             int
    food_name:           str
    quantity_g:          float
    logged_at:           str
    hours_before:        float   = Field(..., description="Hours before the symptom this food was eaten")
    entailment_score:    float   = Field(..., description="Raw NLI model confidence (0–1)")
    contradiction_score: float
    neutral_score:       float
    nutrient_risk_score: float   = Field(..., description="How risky this food's nutrients are for this symptom (0–1)")
    temporal_score:      float   = Field(..., description="Temporal fit score within digestion window (0–1)")
    quantity_score:      float   = Field(..., description="Portion-size weight (0–1)")
    causation_score:     float   = Field(..., description="Final personalised causation score (0–1)")
    causation_label:     str     = Field(..., description="'Likely' | 'Possible' | 'Unlikely'")
    causation_pct:       str     = Field(..., description="Human-readable score e.g. '78%'")
    # Personalisation fields
    personal_prior:         float = Field(0.5,  description="User's learned probability for this food→symptom pair")
    prior_confidence:       float = Field(0.0,  description="How much real data backs the personal prior (0–1)")
    prior_observations:     int   = Field(0,    description="Number of times this food was in window for this symptom")
    prior_confirmations:    int   = Field(0,    description="Number of confirmed co-occurrences logged")
    personalisation_weight: float = Field(0.0,  description="Weight given to personal prior in final score")
    model_weight:           float = Field(1.0,  description="Weight given to NLI model in final score")
    top_risk_nutrients:  List[str]
    explanation:         str


class SymptomPredictionOut(BaseModel):
    symptom:              str
    intensity:            str
    symptom_logged_at:    str
    digestion_window:     str    = Field(..., description="Clinical window used to filter foods")
    foods_in_window:      int    = Field(..., description="Number of foods found within digestion window")
    foods_outside_window: int    = Field(..., description="Number of foods outside window (not considered)")
    top_cause:            Optional[FoodCausationOut] = Field(None, description="Most likely culprit food")
    all_candidates:       List[FoodCausationOut]     = Field(..., description="All candidate foods ranked by causation score")
    no_foods_found:       bool
    note:                 str = ""


class FoodSymptomPredictResponse(BaseModel):
    user_id:     str
    total_symptoms_analysed: int
    total_food_logs_provided: int
    predictions: List[SymptomPredictionOut]


_VALID_PREDICT_SYMPTOMS  = set(SYMPTOM_WINDOWS.keys())
_VALID_PREDICT_SEVERITIES = {"Mild", "Moderate", "Severe"}


@app.post(
    "/predict/food-symptom",
    response_model = FoodSymptomPredictResponse,
    summary        = "Predict which logged food caused which symptom",
)
def predict_food_symptom(req: FoodSymptomPredictRequest) -> FoodSymptomPredictResponse:
    """
    **Analyses your food log history and symptom logs to predict causation.**

    Food and symptoms are never logged at the same time — this endpoint reasons
    across time to find which food most likely caused each reported symptom.

    ---

    ### How it works

    **Stage 1 — Temporal window filter**
    Each symptom has a clinical digestion window:
    - `Heartburn` / `Acid Reflux` → foods eaten **0.25h – 3h** before
    - `Bloating` / `Gas` → foods eaten **0.5h – 8h** before
    - `Diarrhea` → foods eaten **1h – 16h** before
    - `Constipation` → foods eaten **12h – 48h** before
    - `Nausea` → foods eaten **0.5h – 4h** before
    - `Cramps` / `Abdominal Pain` → foods eaten **0.5h – 8h** before
    - `Fatigue` → foods eaten **1h – 12h** before

    Only foods eaten within this window are considered candidates.

    **Stage 2 — HuggingFace NLI causation scoring**
    Model: `cross-encoder/nli-deberta-v3-small`
    - Builds a rich premise: *"Person ate [food] containing [nutrients]"*
    - Hypothesis: *"Eating this food caused [symptom] at [intensity] severity"*
    - Cross-encoder reads both together → outputs entailment probability

    **Stage 3 — Combined score**
    ```
    causation_score =
        35% × NLI entailment score       (model confidence)
      + 35% × nutrient risk score        (domain knowledge: which nutrients cause this symptom)
      + 20% × temporal proximity         (foods at window centre scored higher)
      + 10% × quantity weight            (larger portions = higher weight)
    ```

    **Causation labels:**
    - `Likely` → score ≥ 0.70
    - `Possible` → score 0.45 – 0.69
    - `Unlikely` → score < 0.45

    ---

    ### Important notes
    - Send **all food logs from the past 48h** for best accuracy (covers Constipation window)
    - `quantity_g` must already be converted to grams before sending
    - All `user_id` values in food_logs and symptom_logs must match

    ---

    ### Example request
    ```json
    {
      "food_logs": [
        { "user_id": "u123", "usda_id": 5064, "logged_at": "2024-03-15T19:00:00", "quantity_g": 250 },
        { "user_id": "u123", "usda_id": 11090, "logged_at": "2024-03-15T19:05:00", "quantity_g": 150 }
      ],
      "symptom_logs": [
        { "user_id": "u123", "symptom": "Heartburn", "logged_at": "2024-03-15T21:30:00", "intensity": "Severe" }
      ]
    }
    ```
    """
    if not _state.predictor_ready:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = f"Food-symptom predictor unavailable. Error: {_state.predictor_error or 'unknown'}",
        )

    # Validate symptoms and intensities
    for sl in req.symptom_logs:
        if sl.symptom not in _VALID_PREDICT_SYMPTOMS:
            raise HTTPException(
                status_code = 422,
                detail      = f"Invalid symptom '{sl.symptom}'. Allowed: {sorted(_VALID_PREDICT_SYMPTOMS)}",
            )
        if sl.intensity not in _VALID_PREDICT_SEVERITIES:
            raise HTTPException(
                status_code = 422,
                detail      = f"Invalid intensity '{sl.intensity}'. Allowed: {sorted(_VALID_PREDICT_SEVERITIES)}",
            )

    # Derive user_id from first food log
    user_id = req.food_logs[0].user_id

    # Convert to internal data classes
    food_entries = [
        FoodLogEntry(
            user_id    = f.user_id,
            usda_id    = f.usda_id,
            logged_at  = f.logged_at,
            quantity_g = f.quantity_g,
        )
        for f in req.food_logs
    ]
    symptom_entries = [
        SymptomLogEntry(
            user_id   = s.user_id,
            symptom   = s.symptom,
            logged_at = s.logged_at,
            intensity = s.intensity,
        )
        for s in req.symptom_logs
    ]

    # Load user memory for personalisation
    user_memory = None
    if _state.memory_store is not None:
        user_memory = _state.memory_store.load(user_id)

    try:
        raw_predictions = _state.predict_food_symptom_causes(
            food_logs    = food_entries,
            symptom_logs = symptom_entries,
            usda_df      = _state.usda_df,
            id_col       = _state.id_col,
            desc_col     = _state.desc_col,
            user_memory  = user_memory,
        )
    except Exception as exc:
        log.exception("Food-symptom prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # Auto-update Bayesian priors from this log session (learning happens here)
    if _state.auto_update_from_logs is not None and _state.memory_store is not None:
        try:
            _state.auto_update_from_logs(
                user_id,
                food_entries,
                symptom_entries,
                store=_state.memory_store,   # ← explicitly use the active persistent store
            )
        except Exception as exc:
            log.warning(f"Memory auto-update failed (non-fatal): {exc}")

    def _map_candidate(c: FoodCausationResult) -> FoodCausationOut:
        return FoodCausationOut(
            usda_id                 = c.usda_id,
            food_name               = c.food_name,
            quantity_g              = c.quantity_g,
            logged_at               = c.logged_at,
            hours_before            = c.hours_before,
            entailment_score        = c.entailment_score,
            contradiction_score     = c.contradiction_score,
            neutral_score           = c.neutral_score,
            nutrient_risk_score     = c.nutrient_risk_score,
            temporal_score          = c.temporal_score,
            quantity_score          = c.quantity_score,
            causation_score         = c.causation_score,
            causation_label         = c.causation_label,
            causation_pct           = c.causation_pct,
            personal_prior          = c.personal_prior,
            prior_confidence        = c.prior_confidence,
            prior_observations      = c.prior_observations,
            prior_confirmations     = c.prior_confirmations,
            personalisation_weight  = c.personalisation_weight,
            model_weight            = c.model_weight,
            top_risk_nutrients      = c.top_risk_nutrients,
            explanation             = c.explanation,
        )

    output = [
        SymptomPredictionOut(
            symptom              = p.symptom,
            intensity            = p.intensity,
            symptom_logged_at    = p.symptom_logged_at,
            digestion_window     = p.digestion_window,
            foods_in_window      = p.foods_in_window,
            foods_outside_window = p.foods_outside_window,
            top_cause            = _map_candidate(p.top_cause) if p.top_cause else None,
            all_candidates       = [_map_candidate(c) for c in p.all_candidates],
            no_foods_found       = p.no_foods_found,
            note                 = p.note,
        )
        for p in raw_predictions
    ]

    return FoodSymptomPredictResponse(
        user_id                  = user_id,
        total_symptoms_analysed  = len(output),
        total_food_logs_provided = len(req.food_logs),
        predictions              = output,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 8 ── POST /predict/feedback
#  User confirms or denies a prediction → strongest learning signal
#  Called when user taps "Yes, this caused it" or "No, not this food"
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    user_id:   str  = Field(..., description="User identifier")
    usda_id:   int  = Field(..., description="USDA food ID being confirmed or denied")
    symptom:   str  = Field(..., description="Symptom being linked or unlinked",
                            examples=["Heartburn", "Bloating"])
    confirmed: bool = Field(..., description=(
        "True = user confirms this food caused the symptom. "
        "False = user denies — this food did NOT cause it."
    ))


class FeedbackResponse(BaseModel):
    user_id:              str
    usda_id:              int
    symptom:              str
    confirmed:            bool
    updated_prior:        float  = Field(..., description="Updated personal probability for this food→symptom pair")
    prior_observations:   int    = Field(..., description="Total co-occurrence observations for this pair")
    prior_confirmations:  int    = Field(..., description="Number of confirmed causations")
    prior_confidence:     float  = Field(..., description="Model confidence in this prior (0–1)")
    personalisation_weight: float
    message:              str


@app.post(
    "/predict/feedback",
    response_model = FeedbackResponse,
    summary        = "Submit explicit feedback to improve personalised predictions",
)
def predict_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """
    **The strongest learning signal.** Call this when the user explicitly confirms
    or denies that a specific food caused a specific symptom.

    Each feedback submission applies a **weight-2.0 Bayesian update**
    (twice as strong as an automatic co-occurrence update from daily logging).

    **After ~5 feedbacks on the same food→symptom pair**, the personal prior
    starts dominating the NLI model score, making predictions highly personalised.

    ---
    ### When to call this
    After `POST /predict/food-symptom` returns a prediction, show the user:
    > *"We think the Fried Chicken you ate 1.5h before caused your Heartburn — was that right?"*
    - User taps **Yes** → `confirmed: true`
    - User taps **No** → `confirmed: false`

    ---
    ### Example — user confirms heartburn was caused by fried chicken (usda_id: 5064)
    ```json
    {
      "user_id": "u123",
      "usda_id": 5064,
      "symptom": "Heartburn",
      "confirmed": true
    }
    ```
    """
    if _state.memory_store is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "Memory store unavailable.",
        )

    memory = _state.memory_store.load(req.user_id)
    memory.apply_explicit_feedback(req.usda_id, req.symptom, req.confirmed)
    _state.memory_store.save(memory)

    prior   = memory.get_prior(req.usda_id, req.symptom)
    action  = "confirmed ✅" if req.confirmed else "denied ✗"
    message = (
        f"Feedback recorded — {action}. "
        f"Personal prior for this food→{req.symptom} pair updated to "
        f"{prior.posterior_mean:.0%} based on {prior.observations} observations. "
        f"Personalisation weight: {memory.personalisation_weight:.0%}."
    )

    return FeedbackResponse(
        user_id               = req.user_id,
        usda_id               = req.usda_id,
        symptom               = req.symptom,
        confirmed             = req.confirmed,
        updated_prior         = round(prior.posterior_mean, 4),
        prior_observations    = prior.observations,
        prior_confirmations   = prior.confirmations,
        prior_confidence      = round(prior.confidence, 4),
        personalisation_weight= memory.personalisation_weight,
        message               = message,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 9 ── GET /user/{user_id}/learning-summary
#  View what the model has learned about a specific user
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class SensitivityItem(BaseModel):
    food_symptom_pair:       str
    causation_probability:   float  = Field(..., description="Learned probability 0–1")
    observations:            int
    confirmations:           int
    confidence:              float
    last_updated:            str


class LearningSummaryResponse(BaseModel):
    user_id:                 str
    total_food_logs:         int
    total_symptom_logs:      int
    total_log_entries:       int
    personalisation_weight:  float = Field(..., description=(
        "How much weight personal data carries in predictions. "
        "Grows from 0.20 (new user) to 0.75 (30+ logs)."
    ))
    model_weight:            float
    personalisation_stage:   str   = Field(..., description=(
        "'New user (model-led)' | 'Learning' | 'Developing' | 'Personalised'"
    ))
    learned_pairs:           int   = Field(..., description="Number of food→symptom pairs with real data")
    top_sensitivities:       List[SensitivityItem] = Field(..., description=(
        "Top 5 food→symptom pairs sorted by learned causation probability"
    ))
    learning_message:        str


def _personalisation_stage(weight: float) -> str:
    if   weight >= 0.75: return "Personalised — strong personal data, minimal model reliance"
    elif weight >= 0.60: return "Developing — personal data growing, balanced with model"
    elif weight >= 0.40: return "Learning — personal patterns emerging"
    else:                return "New user — model-led predictions, keep logging daily"


@app.get(
    "/user/{user_id}/learning-summary",
    response_model = LearningSummaryResponse,
    summary        = "View what the model has learned about this user's food sensitivities",
)
def learning_summary(user_id: str) -> LearningSummaryResponse:
    """
    Returns a full summary of what the personalisation engine has learned
    about this user's specific food→symptom sensitivities.

    **Use this to show users their personalised food sensitivity profile.**

    ---
    ### What it shows
    - How many food + symptom logs have been recorded
    - Current personalisation stage and weight
    - Top 5 learned food→symptom sensitivities ranked by probability
    - Each pair shows: causation probability, observation count, confidence

    ### Personalisation stages
    | Total logs | Stage | Personal weight |
    |------------|-------|-----------------|
    | 0–4        | New user (model-led) | 20% |
    | 5–14       | Learning | 40% |
    | 15–29      | Developing | 60% |
    | 30+        | Personalised | 75% |

    ### Example response
    ```json
    {
      "total_log_entries": 18,
      "personalisation_weight": 0.60,
      "personalisation_stage": "Developing — personal data growing",
      "top_sensitivities": [
        { "food_symptom_pair": "5064:Heartburn", "causation_probability": 0.82, "observations": 6 },
        { "food_symptom_pair": "11090:Bloating", "causation_probability": 0.71, "observations": 4 }
      ]
    }
    ```
    """
    if _state.memory_store is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "Memory store unavailable.",
        )

    summary = _state.memory_store.summary(user_id)

    if summary.get("status") == "no data yet":
        return LearningSummaryResponse(
            user_id                = user_id,
            total_food_logs        = 0,
            total_symptom_logs     = 0,
            total_log_entries      = 0,
            personalisation_weight = 0.20,
            model_weight           = 0.80,
            personalisation_stage  = "New user — model-led predictions, keep logging daily",
            learned_pairs          = 0,
            top_sensitivities      = [],
            learning_message       = (
                "No data yet for this user. "
                "Start logging meals with /log/food and symptoms with /log/symptom. "
                "Run /predict/food-symptom after each log session to trigger learning."
            ),
        )

    total = summary["total_food_logs"] + summary["total_symptom_logs"]
    pw    = summary["personalisation_weight"]

    return LearningSummaryResponse(
        user_id                = user_id,
        total_food_logs        = summary["total_food_logs"],
        total_symptom_logs     = summary["total_symptom_logs"],
        total_log_entries      = total,
        personalisation_weight = pw,
        model_weight           = round(1.0 - pw, 2),
        personalisation_stage  = _personalisation_stage(pw),
        learned_pairs          = summary["learned_pairs"],
        top_sensitivities      = [
            SensitivityItem(
                food_symptom_pair     = p["food_symptom_pair"],
                causation_probability = p["causation_probability"],
                observations          = p["observations"],
                confirmations         = p["confirmations"],
                confidence            = p["confidence"],
                last_updated          = p["last_updated"],
            )
            for p in summary["top_sensitivities"]
        ],
        learning_message = (
            f"Model is {_personalisation_stage(pw).lower()}. "
            f"{summary['learned_pairs']} food→symptom pattern(s) identified. "
            + (
                f"Keep logging daily — predictions improve with each session."
                if pw < 0.60 else
                f"Strong personal profile established — predictions are now highly personalised."
            )
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 10 ── POST /recommend/safe-foods
#  Returns 5 personalised safe food recommendations in natural language.
#  Foods are nutritionally similar to what the user eats, but safer for
#  their specific gut sensitivities learned from their log history.
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class SafeFoodRequest(BaseModel):
    user_id: str = Field(
        ...,
        description = "User identifier — used to load personal symptom memory.",
    )
    eaten_usda_ids: List[int] = Field(
        default_factory = list,
        description = (
            "USDA IDs of foods this user has already eaten (from food logs). "
            "These are excluded from recommendations — only new foods are returned. "
            "Pass the last 7–14 days of food log usda_ids for best results."
        ),
        examples = [[8121, 5064, 20037, 11090]],
    )
    user_symptoms: List[str] = Field(
        default_factory = list,
        description = (
            "Symptoms this user has reported. Used to identify which nutrients "
            "to avoid in recommendations. Pass all symptoms from their symptom log."
        ),
        examples = [["Heartburn", "Bloating"]],
    )
    n: int = Field(
        default     = 5,
        ge          = 1,
        le          = 10,
        description = "Number of recommendations to return (default 5, max 10).",
    )


class KeyNutrientsOut(BaseModel):
    model_config = {"extra": "allow"}   # dynamic nutrient keys


class SafeFoodOut(BaseModel):
    rank:             int   = Field(..., description="1 = best match")
    food_name:        str   = Field(..., description="Clean natural food name")
    recommendation:   str   = Field(..., description="Flan-T5 generated friendly recommendation")
    safety_score:     float = Field(..., description="How safe for this user's gut (0–1)")
    similarity_score: float = Field(..., description="How similar to user's eating pattern (0–1)")
    key_nutrients:    dict  = Field(..., description="Key nutrients per 100g from USDA dataset")
    safe_reasons:     List[str] = Field(..., description="Why this food is recommended")
    avoid_reasons:    List[str] = Field(..., description="Any mild caveats about this food")


class SafeFoodResponse(BaseModel):
    user_id:              str
    total_eaten_excluded: int    = Field(..., description="Number of already-eaten foods excluded from search")
    user_symptoms:        List[str]
    personalised:         bool   = Field(..., description="True if personal memory was used to personalise results")
    recommendations:      List[SafeFoodOut]
    model_note:           str    = Field(..., description="Which model generated the descriptions")


_VALID_RECOMMENDATION_SYMPTOMS = {
    "Heartburn", "Acid Reflux", "Bloating", "Gas", "Nausea",
    "Cramps", "Abdominal Pain", "Diarrhea", "Constipation", "Fatigue",
}


@app.post(
    "/recommend/safe-foods",
    response_model = SafeFoodResponse,
    summary        = "Get 5 personalised safe food recommendations",
)
def recommend_safe_foods(req: SafeFoodRequest) -> SafeFoodResponse:
    """
    Returns **5 personalised gut-friendly food recommendations** based on the user's
    food log history and symptom patterns.

    ---

    ### What makes a recommendation personalised?

    **1. Nutritional similarity** — finds foods with a similar macro profile to what
    the user already eats (calories, protein, carbs, fat range). They won't get
    recommendations for foods completely unlike their diet.

    **2. Symptom-aware safety filtering** — identifies which nutrients are known to
    trigger the user's reported symptoms and filters for foods LOW in those nutrients.
    - User has `Heartburn` → recommends foods low in fat, sodium, and sugar
    - User has `Bloating` → recommends foods low in fermentable carbs and sugar
    - User has `Constipation` → recommends foods high in fibre, low in saturated fat

    **3. Personal memory** — if the user has enough log history, their Bayesian
    causation priors (from `/predict/food-symptom`) are used to penalise foods
    that are similar to their personal trigger foods.

    **4. No repeats** — all foods in `eaten_usda_ids` are excluded. Every recommendation
    is genuinely new.

    **5. Natural language** — descriptions are generated by **Flan-T5** as warm,
    friendly sentences rather than raw USDA codes like "CHICKEN,BRST,CKD,RSTD".

    ---

    ### Workflow
    ```
    1. Pull user's last 14 days of food logs → extract usda_ids
    2. Pull user's symptom log → extract symptom names
    3. POST /recommend/safe-foods {
         "user_id": "u123",
         "eaten_usda_ids": [8121, 5064, 11090],
         "user_symptoms": ["Heartburn", "Bloating"]
       }
    4. Show 5 recommendations with friendly descriptions to the user
    ```

    ---

    ### Allowed `user_symptoms` values
    `Heartburn` · `Acid Reflux` · `Bloating` · `Gas` · `Nausea` ·
    `Cramps` · `Abdominal Pain` · `Diarrhea` · `Constipation` · `Fatigue`

    ---

    ### Example response (one item)
    ```json
    {
      "rank": 1,
      "food_name": "Steamed white fish, lemon herb",
      "recommendation": "A light, easy-to-digest protein source that is very low in fat
                         and sodium — a great choice if you struggle with heartburn or
                         acid reflux after heavy meals.",
      "safety_score": 0.91,
      "similarity_score": 0.73,
      "key_nutrients": {
        "Calories (kcal/100g)": 96.0,
        "Protein (g/100g)": 20.1,
        "Total Fat (g/100g)": 1.2,
        "Sodium (mg/100g)": 68.0
      },
      "safe_reasons": ["Very low total fat (1.2g/100g)", "Very low sodium (68mg/100g)"],
      "avoid_reasons": []
    }
    ```
    """
    if not _state.recommender_ready:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = f"Recommender unavailable. Error: {_state.recommender_error or 'unknown'}",
        )

    if not _state.usda_ready:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "USDA dataset unavailable.",
        )

    # Validate symptoms
    invalid_syms = [s for s in req.user_symptoms if s not in _VALID_RECOMMENDATION_SYMPTOMS]
    if invalid_syms:
        raise HTTPException(
            status_code = 422,
            detail      = f"Invalid symptoms: {invalid_syms}. Allowed: {sorted(_VALID_RECOMMENDATION_SYMPTOMS)}",
        )

    # Load user memory for personalisation
    user_memory  = None
    personalised = False
    if _state.memory_store is not None:
        user_memory = _state.memory_store.load(req.user_id)
        if user_memory.total_food_logs > 0:
            personalised = True

    try:
        from food_recommender import recommend_safe_foods as _rec
        raw_results = _rec(
            eaten_usda_ids = req.eaten_usda_ids,
            user_symptoms  = req.user_symptoms,
            user_memory    = user_memory,
            usda_df        = _state.usda_df,
            id_col         = _state.id_col,
            desc_col       = _state.desc_col,
            n              = req.n,
        )
    except Exception as exc:
        log.exception("Food recommendation failed")
        raise HTTPException(status_code=500, detail=f"Recommendation error: {exc}")

    if not raw_results:
        raise HTTPException(
            status_code = 404,
            detail      = (
                "No suitable food recommendations found. "
                "Try providing fewer eaten_usda_ids or broadening user_symptoms."
            ),
        )

    return SafeFoodResponse(
        user_id              = req.user_id,
        total_eaten_excluded = len(req.eaten_usda_ids),
        user_symptoms        = req.user_symptoms,
        personalised         = personalised,
        recommendations      = [
            SafeFoodOut(
                rank             = r.rank,
                food_name        = r.food_name,
                recommendation   = r.recommendation,
                safety_score     = r.safety_score,
                similarity_score = r.similarity_score,
                key_nutrients    = r.key_nutrients,
                safe_reasons     = r.safe_reasons,
                avoid_reasons    = r.avoid_reasons,
            )
            for r in raw_results
        ],
        model_note = (
            "Descriptions generated by google/flan-t5-base with rule-based fallback. "
            "Nutritional data sourced from USDA National Nutrient Database. "
            f"{'Personalised using user log history.' if personalised else 'Generic recommendations — log more meals for personalisation.'}"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 11 — GET /user/{user_id}/dashboard
#  Full 7-day summary from MongoDB: food logs, symptom logs,
#  score history, and top learned sensitivities.
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class SensitivitySummary(BaseModel):
    pair:        str
    probability: float
    observations:int

class DashboardResponse(BaseModel):
    user_id:                 str
    name:                    str
    current_score:           int
    grade:                   str
    food_logs_this_week:     int
    symptom_logs_this_week:  int
    unique_foods_eaten:      List[str]
    symptom_frequency:       dict
    score_history:           List[dict]
    personalisation_weight:  float
    personalisation_stage:   str
    top_sensitivities:       List[SensitivitySummary]


@app.get(
    "/user/{user_id}/dashboard",
    response_model = DashboardResponse,
    summary        = "Full 7-day user dashboard from MongoDB",
)
def user_dashboard(user_id: str) -> DashboardResponse:
    """
    Returns a full week-in-review dashboard for the user,
    sourced directly from MongoDB.

    Includes food log count, symptom frequency, score history,
    and the top learned food→symptom sensitivities from their
    Bayesian memory.
    """
    mongo = getattr(_state, "_mongo_db", None)
    if mongo is None:
        raise HTTPException(503, "MongoDB not connected.")

    user = mongo.get_user(user_id)
    if user is None:
        raise HTTPException(404, f"User '{user_id}' not found.")

    data = mongo.user_dashboard(user_id)

    pw    = data["personalisation_weight"]
    stage = _personalisation_stage(pw)

    return DashboardResponse(
        user_id                = user_id,
        name                   = data["name"],
        current_score          = data["current_score"],
        grade                  = data["grade"],
        food_logs_this_week    = data["food_logs_this_week"],
        symptom_logs_this_week = data["symptom_logs_this_week"],
        unique_foods_eaten     = data["unique_foods_eaten"],
        symptom_frequency      = data["symptom_frequency"],
        score_history          = data["score_history"],
        personalisation_weight = pw,
        personalisation_stage  = stage,
        top_sensitivities      = [
            SensitivitySummary(**s) for s in data["top_sensitivities"]
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 12 — GET /db/health
#  MongoDB connection status + collection counts
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/db/health", summary="MongoDB connection health and collection stats")
def db_health():
    """
    Returns MongoDB connection status and row counts for all collections.
    Use this to verify the demo seed ran correctly.
    """
    mongo = getattr(_state, "_mongo_db", None)
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
            },
            "users": [
                {
                    "user_id":       u["user_id"],
                    "name":          u.get("name"),
                    "score":         u.get("current_score"),
                    "grade":         u.get("grade"),
                    "food_logs":     mongo.count_food_logs(u["user_id"]),
                    "symptom_logs":  mongo.count_symptom_logs(u["user_id"]),
                }
                for u in users
            ],
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 13 — POST /predict/meal-symptom-forecast
#
#  The user provides a meal they HAVEN'T eaten yet (USDA IDs + quantities).
#  The API pulls their full food + symptom log history from MongoDB (≤ 400 each),
#  builds their personal Bayesian sensitivity profile, then predicts which of
#  the 10 gut symptoms they are at risk of triggering — before they eat.
#
#  Model: cross-encoder/nli-deberta-v3-small (reused from food_symptom_predictor)
#  No extra RAM. No extra startup cost.
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

from meal_symptom_forecast import ProposedFoodItem


# ── Request / response models ─────────────────────────────────────────────────

class ProposedFoodIn(BaseModel):
    usda_id:    int   = Field(
        ...,
        description="USDA food ID. Get from POST /food/parse or POST /food/text-to-id.",
    )
    quantity_g: float = Field(
        ..., gt=0,
        description="Portion size in grams. Convert cups/pieces/ml to grams before sending.",
        examples=[200.0, 150.0, 100.0],
    )


class FoodScoreBreakdown(BaseModel):
    usda_id:             int
    food_name:           str
    quantity_g:          float
    nli_entailment:      float = Field(..., description="NLI model confidence this food causes the symptom (0–1)")
    nutrient_risk_score: float = Field(..., description="Nutrient-based risk score for this symptom (0–1)")
    quantity_weight:     float = Field(..., description="Portion size weight (0–1)")
    base_score:          float = Field(..., description="Score without personalisation")
    personalised_score:  float = Field(..., description="Final score after Bayesian prior blend")
    personal_prior:      float = Field(..., description="User's learned probability for this food→symptom pair")
    prior_confidence:    float = Field(..., description="How much history backs the personal prior (0–1)")
    prior_observations:  int   = Field(..., description="Number of co-occurrence observations logged")
    top_risk_nutrients:  List[str] = Field(..., description="Nutrients contributing most to this symptom risk")


class MealSymptomForecastOut(BaseModel):
    symptom:              str
    risk_score:           float = Field(..., description="Aggregated risk score across the full meal (0–1)")
    risk_level:           str   = Field(..., description="'High' (≥0.60) | 'Medium' (≥0.38) | 'Low'")
    risk_pct:             str   = Field(..., description="Human-readable score, e.g. '72%'")
    top_trigger_food:     str   = Field(..., description="Food in the meal most likely to cause this symptom")
    top_trigger_usda_id:  int
    top_trigger_score:    float = Field(..., description="That food's individual contribution score")
    top_risk_nutrients:   List[str]
    per_food_scores:      List[FoodScoreBreakdown]
    personalised:         bool  = Field(..., description="True if user history was used to personalise scores")
    explanation:          str


class MealForecastRequest(BaseModel):
    user_id: str = Field(
        ...,
        description=(
            "User identifier. Used to pull their food + symptom log history "
            "from MongoDB (up to 400 entries each) to personalise predictions."
        ),
    )
    proposed_foods: List[ProposedFoodIn] = Field(
        ..., min_length=1, max_length=20,
        description=(
            "The hypothetical meal to evaluate. Pass 1–20 food items with USDA IDs and "
            "portion sizes in grams. Use POST /food/text-to-id to convert food names to IDs."
        ),
    )


class MealForecastResponse(BaseModel):
    user_id:                  str
    proposed_foods_count:     int   = Field(..., description="Number of food items evaluated")
    food_logs_used:           int   = Field(..., description="User's food log entries pulled from MongoDB")
    symptom_logs_used:        int   = Field(..., description="User's symptom log entries pulled from MongoDB")
    personalised:             bool  = Field(..., description="True if personal history influenced predictions")
    personalisation_weight:   float = Field(..., description="Weight given to personal data vs model (0–1)")
    high_risk_symptoms:       List[str] = Field(..., description="Symptoms with risk_level = 'High'")
    medium_risk_symptoms:     List[str] = Field(..., description="Symptoms with risk_level = 'Medium'")
    forecasts:                List[MealSymptomForecastOut] = Field(
        ..., description="All 10 symptoms ranked by risk score (highest first)"
    )
    evaluated_at:             str


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post(
    "/predict/meal-symptom-forecast",
    response_model = MealForecastResponse,
    summary        = "Predict which symptoms a hypothetical uneaten meal might cause",
)
def meal_symptom_forecast(req: MealForecastRequest) -> MealForecastResponse:
    """
    **Before eating, check if a meal is safe for your gut.**

    Provide a meal you're planning to eat (USDA food IDs + portion sizes).
    The API analyses your personal gut history from MongoDB and predicts
    which of the 10 gut symptoms you are at risk of triggering.

    ---

    ### How personalisation works

    The system pulls up to **400 food logs** and **400 symptom logs** from
    MongoDB for this user. These are used to build a personal Bayesian sensitivity
    profile — how often a specific food appeared in a digestion window before a
    specific symptom for THIS user. New users get generic NLI-model predictions.
    Users with 30+ logs get 75% weight on their personal history.

    | Total logs | Personal weight | Model weight |
    |------------|-----------------|--------------|
    | 0–4        | 20%             | 80%          |
    | 5–14       | 40%             | 60%          |
    | 15–29      | 60%             | 40%          |
    | 30+        | 75%             | 25%          |

    ---

    ### Scoring formula (per food × per symptom)
    ```
    base_score = 0.45 × NLI entailment + 0.45 × nutrient_risk + 0.10 × quantity_weight
    ```
    - **NLI entailment** — cross-encoder/nli-deberta-v3-small reads:
      - Premise: *"The person ate [food] containing [nutrients]"*
      - Hypothesis: *"Eating this food caused [symptom] (moderate intensity)"*
    - **Nutrient risk** — domain rules: e.g. high fat + sodium = Heartburn risk
    - **Quantity weight** — larger portions contribute more risk
    - **Personal prior** — blended in via Bayesian weight schedule above

    Multi-food aggregation: `max(weighted_avg_by_portion, 0.85 × max_single_food_score)`

    ---

    ### Risk levels
    | Score  | Level  |
    |--------|--------|
    | ≥ 0.60 | High   |
    | ≥ 0.38 | Medium |
    | < 0.38 | Low    |

    ---

    ### Workflow
    ```
    1. User types: "I'm thinking of eating fried chicken with white rice"
       → POST /food/text-to-id  →  get USDA IDs

    2. Check if it's safe:
       → POST /predict/meal-symptom-forecast {
           "user_id": "u123",
           "proposed_foods": [
             { "usda_id": 5064, "quantity_g": 300 },
             { "usda_id": 20050, "quantity_g": 200 }
           ]
         }

    3. Response shows: Heartburn HIGH 74%, Acid Reflux HIGH 69%, Bloating MEDIUM 51%
       → Warn the user before they eat
    ```

    ---

    ### Difference from other prediction endpoints
    | Endpoint | When | Needs symptom logs |
    |---|---|---|
    | `/predict/meal-symptom-forecast` | Before eating | ❌ No |
    | `/predict/symptom-risk/{user_id}` | After eating (from DB) | ❌ No |
    | `/predict/food-symptom` | After eating + felt symptom | ✅ Yes |
    """
    # ── Guards ────────────────────────────────────────────────────────────────
    if not _state.meal_forecast_ready:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = f"Meal forecast unavailable: {_state.meal_forecast_error or 'unknown'}",
        )
    if not _state.usda_ready:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "USDA dataset unavailable.",
        )

    mongo = getattr(_state, "_mongo_db", None)
    if mongo is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "MongoDB not connected — user history unavailable.",
        )

    # ── Pull user history from MongoDB (max 400 each) ─────────────────────────
    try:
        # 400 food logs covers ~57 days at 7 logs/day — sufficient for full history
        food_logs    = mongo.get_food_logs(req.user_id,    days=90, limit=400)
        symptom_logs = mongo.get_symptom_logs(req.user_id, days=90, limit=400)
    except Exception as exc:
        log.exception(f"MongoDB fetch failed for user={req.user_id!r}")
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    # ── Load Bayesian memory built from that history ───────────────────────────
    user_memory   = None
    p_weight      = 0.20
    is_personalised = False

    if _state.memory_store is not None:
        try:
            user_memory     = _state.memory_store.load(req.user_id)
            p_weight        = getattr(user_memory, "personalisation_weight", 0.20)
            is_personalised = getattr(user_memory, "total_food_logs", 0) > 0
        except Exception as exc:
            log.warning(f"Memory load failed for {req.user_id!r} (non-fatal): {exc}")

    # ── Convert request foods to internal data objects ────────────────────────
    proposed = [
        ProposedFoodItem(usda_id=f.usda_id, quantity_g=f.quantity_g)
        for f in req.proposed_foods
    ]

    # ── Run forecast ──────────────────────────────────────────────────────────
    try:
        raw_forecasts = _state.forecast_meal_symptoms(
            proposed_foods = proposed,
            usda_df        = _state.usda_df,
            id_col         = _state.id_col,
            desc_col       = _state.desc_col,
            user_memory    = user_memory,
        )
    except Exception as exc:
        log.exception(f"Meal forecast failed for user={req.user_id!r}")
        raise HTTPException(status_code=500, detail=f"Forecast error: {exc}")

    # ── Build response ─────────────────────────────────────────────────────────
    high_risk   = [f.symptom for f in raw_forecasts if f.risk_level == "High"]
    medium_risk = [f.symptom for f in raw_forecasts if f.risk_level == "Medium"]

    forecasts_out = [
        MealSymptomForecastOut(
            symptom             = f.symptom,
            risk_score          = f.risk_score,
            risk_level          = f.risk_level,
            risk_pct            = f.risk_pct,
            top_trigger_food    = f.top_trigger_food,
            top_trigger_usda_id = f.top_trigger_usda_id,
            top_trigger_score   = f.top_trigger_score,
            top_risk_nutrients  = f.top_risk_nutrients,
            per_food_scores     = [
                FoodScoreBreakdown(
                    usda_id             = s.usda_id,
                    food_name           = s.food_name,
                    quantity_g          = s.quantity_g,
                    nli_entailment      = s.nli_entailment,
                    nutrient_risk_score = s.nutrient_risk_score,
                    quantity_weight     = s.quantity_weight,
                    base_score          = s.base_score,
                    personalised_score  = s.personalised_score,
                    personal_prior      = s.personal_prior,
                    prior_confidence    = s.prior_confidence,
                    prior_observations  = s.prior_observations,
                    top_risk_nutrients  = s.top_risk_nutrients,
                )
                for s in f.per_food_scores
            ],
            personalised  = f.personalised,
            explanation   = f.explanation,
        )
        for f in raw_forecasts
    ]

    return MealForecastResponse(
        user_id                = req.user_id,
        proposed_foods_count   = len(req.proposed_foods),
        food_logs_used         = len(food_logs),
        symptom_logs_used      = len(symptom_logs),
        personalised           = is_personalised,
        personalisation_weight = round(p_weight, 2),
        high_risk_symptoms     = high_risk,
        medium_risk_symptoms   = medium_risk,
        forecasts              = forecasts_out,
        evaluated_at           = datetime.now(timezone.utc).isoformat(),
    )


from scanner import fetch_product


class BarcodeRequest(BaseModel):
    code: str


class BarcodeResponse(BaseModel):
    barcode: str
    product_name: str
    quantity: Optional[str]


@app.post("/scan/barcode", response_model=BarcodeResponse)
def scan_barcode(req: BarcodeRequest):

    code = req.code.strip()

    try:
        product = fetch_product(code)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    return BarcodeResponse(
        barcode=code,
        product_name=product["name"],
        quantity=product.get("quantity")
    )

# ─────────────────────────────────────────────────────────────
#  ENDPOINT 14 ── POST /culprit-foods
#  Simple AI analysis → which foods caused symptoms
# ─────────────────────────────────────────────────────────────

class SimpleFoodLog(BaseModel):
    usda_id: int
    quantity_g: float
    logged_at: datetime


class SimpleSymptomLog(BaseModel):
    symptom: str
    severity: str = "Moderate"
    logged_at: datetime


class CulpritRequest(BaseModel):
    food_logs: List[SimpleFoodLog]
    symptom_logs: List[SimpleSymptomLog]


class CulpritFoodOut(BaseModel):
    usda_id: int
    food_name: str
    score: float
    confidence: str
    occurrence_count: int
    linked_symptoms: List[str]
    top_symptom: str


class CulpritResponse(BaseModel):
    total_foods: int
    total_symptoms: int
    culprit_foods: List[CulpritFoodOut]
    summary: str


@app.post("/culprit-foods", response_model=CulpritResponse,
          summary="Find which foods most likely caused symptoms (NO user_id needed)")
def culprit_foods(req: CulpritRequest) -> CulpritResponse:

    if _state.culprit_cross_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Culprit model not loaded"
        )

    try:
        food_logs = []

        for f in req.food_logs:
            df = _state.usda_df
            id_col = _state.id_col
            desc_col = _state.desc_col

            row = df[df[id_col].astype(str) == str(f.usda_id)]

            if row.empty:
                food_name = "Unknown food"
            else:
                food_name = str(row.iloc[0][desc_col])

            food_logs.append({
                "usda_id": f.usda_id,
                "usda_description": food_name,
                "quantity_g": f.quantity_g,
                "logged_at": f.logged_at
            })

        result = find_culprit_foods(
            food_logs=food_logs,
            symptom_logs=[s.dict() for s in req.symptom_logs],
            cross_encoder=_state.culprit_cross_encoder
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return CulpritResponse(
        total_foods=len(req.food_logs),
        total_symptoms=len(req.symptom_logs),
        culprit_foods=[
            CulpritFoodOut(
                usda_id=f.usda_id,
                food_name=f.food_name,
                score=f.aggregate_score,
                confidence=f.confidence_label,
                occurrence_count=f.occurrence_count,
                linked_symptoms=f.linked_symptoms,
                top_symptom=f.top_symptom
            )
            for f in result.culprit_foods
        ],
        summary=result.method_summary
    )