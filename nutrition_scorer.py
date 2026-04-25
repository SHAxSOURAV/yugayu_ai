"""
nutrition_scorer.py
───────────────────
Gut-health scoring engine.

Rule-based nutrient scoring is 100% unchanged.
DeBERTa zero-shot digestibility classifier replaced with a single Claude API call.

Two public functions (signatures unchanged):
    analyse_food_log_batch(items, meal_type) -> (total_modifier, list[FoodLogScore])
    analyse_symptom_log(symptom, severity, logged_at) -> SymptomLogScore
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_claude_client = None
_usda_client   = None
_mongo_db      = None   # set by init() — optional, cache disabled if None


def init(claude_client, usda_client, mongo_db=None) -> None:
    global _claude_client, _usda_client, _mongo_db
    _claude_client = claude_client
    _usda_client   = usda_client
    _mongo_db      = mongo_db
    status = "MongoDB cache enabled" if mongo_db is not None else "no cache"
    log.info(f"nutrition_scorer: Claude + USDA client ready ({status}).")


# ─────────────────────────────────────────────────────────────────────────────
# Nutrient config (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

_NUTRIENT_CONFIG: dict[str, dict] = {
    "protein": {
        "direction": "beneficial", "low_threshold": 8.0, "high_threshold": 20.0,
        "max_delta": +4, "unit": "g",
        "note_good": "Good protein content supports gut tissue repair and enzyme production.",
        "note_excellent": "High protein supports gut lining integrity and digestive enzyme synthesis.",
    },
    "calcium": {
        "direction": "beneficial", "low_threshold": 80.0, "high_threshold": 200.0,
        "max_delta": +3, "unit": "mg",
        "note_good": "Calcium supports smooth muscle contractions in the gut.",
        "note_excellent": "High calcium aids gut muscle function and may reduce colorectal risk.",
    },
    "potassium": {
        "direction": "beneficial", "low_threshold": 200.0, "high_threshold": 400.0,
        "max_delta": +4, "unit": "mg",
        "note_good": "Potassium supports gut muscle contractions and motility.",
        "note_excellent": "High potassium is strongly linked to improved gut motility and reduced constipation.",
    },
    "vitamin_c": {
        "direction": "beneficial", "low_threshold": 10.0, "high_threshold": 40.0,
        "max_delta": +3, "unit": "mg",
        "note_good": "Vitamin C supports gut antioxidant defence and iron absorption.",
        "note_excellent": "High Vitamin C strongly supports gut lining integrity and microbiome diversity.",
    },
    "vitamin_e": {
        "direction": "beneficial", "low_threshold": 1.0, "high_threshold": 5.0,
        "max_delta": +2, "unit": "mg",
        "note_good": "Vitamin E protects gut cell membranes from oxidative stress.",
        "note_excellent": "High Vitamin E provides strong anti-inflammatory protection for the gut lining.",
    },
    "vitamin_d": {
        "direction": "beneficial", "low_threshold": 1.0, "high_threshold": 5.0,
        "max_delta": +3, "unit": "µg",
        "note_good": "Vitamin D supports gut barrier function and immune-microbiome regulation.",
        "note_excellent": "High Vitamin D is strongly linked to reduced intestinal permeability.",
    },
    "iron": {
        "direction": "dual", "good_low": 2.0, "good_high": 8.0,
        "excess_threshold": 15.0, "max_delta": +2, "min_delta": -3, "unit": "mg",
        "note_good": "Adequate iron supports gut tissue oxygenation and cell renewal.",
        "note_excess": "Very high iron can cause constipation and oxidative gut stress.",
    },
    "total_fat": {
        "direction": "harmful", "mild_threshold": 12.0, "high_threshold": 20.0,
        "max_delta": -5, "unit": "g",
        "note_mild": "Elevated fat content slows gastric emptying, may worsen reflux.",
        "note_high": "Very high fat significantly delays digestion and increases acid reflux risk.",
    },
    "sat_fat": {
        "direction": "harmful", "mild_threshold": 4.0, "high_threshold": 8.0,
        "max_delta": -5, "unit": "g",
        "note_mild": "Elevated saturated fat promotes gut inflammation and reduces microbiome diversity.",
        "note_high": "High saturated fat is strongly linked to gut dysbiosis and increased intestinal permeability.",
    },
    "cholesterol": {
        "direction": "harmful", "mild_threshold": 60.0, "high_threshold": 120.0,
        "max_delta": -3, "unit": "mg",
        "note_mild": "Elevated cholesterol content may contribute to bile overproduction and digestive stress.",
        "note_high": "High cholesterol is associated with increased gallbladder and digestive burden.",
    },
    "sodium": {
        "direction": "harmful", "mild_threshold": 300.0, "high_threshold": 600.0,
        "max_delta": -5, "unit": "mg",
        "note_mild": "Elevated sodium promotes gut wall inflammation and disrupts gut microbiome balance.",
        "note_high": "Very high sodium significantly damages gut barrier function and drives microbiome dysbiosis.",
    },
    "sugar": {
        "direction": "harmful", "mild_threshold": 8.0, "high_threshold": 18.0,
        "max_delta": -5, "unit": "g",
        "note_mild": "Elevated sugar feeds harmful gut bacteria and may worsen bloating and gas.",
        "note_high": "Very high sugar rapidly disrupts gut microbiome diversity and promotes gut dysbiosis.",
    },
    "carbs": {
        "direction": "harmful", "mild_threshold": 35.0, "high_threshold": 65.0,
        "max_delta": -3, "unit": "g",
        "note_mild": "High refined carbohydrate content may spike gut fermentation and bloating.",
        "note_high": "Very high carbohydrate load drives excessive gut fermentation.",
    },
    "calories": {
        "direction": "harmful", "mild_threshold": 250.0, "high_threshold": 450.0,
        "max_delta": -3, "unit": "kcal",
        "note_mild": "High caloric density slows digestion and increases digestive workload.",
        "note_high": "Very high caloric density places significant burden on digestive enzymes and motility.",
    },
}

_MEAL_MULTIPLIER = {"Breakfast": 0.85, "Lunch": 1.00, "Dinner": 1.15, "Snack": 0.90}
_MEAL_NOTE = {
    "Breakfast": "Breakfast multiplier (×0.85) — morning digestion is active, gut tolerates more.",
    "Lunch":     "Lunch multiplier (×1.00) — neutral baseline context.",
    "Dinner":    "Dinner multiplier (×1.15) — evening digestion is slower; impact amplified.",
    "Snack":     "Snack multiplier (×0.90) — smaller portion context, slightly reduced impact.",
}

_UNIT_TO_GRAMS = {
    "g": 1.0, "gram": 1.0, "grams": 1.0, "kg": 1000.0,
    "ml": 1.0, "l": 1000.0,
    "oz": 28.3495, "lb": 453.592,
    "piece": 100.0, "pieces": 100.0,
    "cup": 240.0, "tbsp": 15.0, "tsp": 5.0, "serving": 100.0,
}


def _to_grams(quantity: float, unit: str) -> float:
    return quantity * _UNIT_TO_GRAMS.get(unit.lower().strip(), 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NutrientProfile:
    calories: Optional[float] = None; protein: Optional[float] = None
    total_fat: Optional[float] = None; carbs: Optional[float] = None
    sodium: Optional[float] = None; sat_fat: Optional[float] = None
    cholesterol: Optional[float] = None; sugar: Optional[float] = None
    calcium: Optional[float] = None; iron: Optional[float] = None
    potassium: Optional[float] = None; vitamin_c: Optional[float] = None
    vitamin_e: Optional[float] = None; vitamin_d: Optional[float] = None
    fiber: Optional[float] = None
    portion_grams: float = 100.0


@dataclass
class NutrientImpact:
    nutrient: str; value: float; unit: str
    delta: int; note: str; direction: str


@dataclass
class FoodLogScore:
    usda_id: int; food_description: str; portion_grams: float; meal_type: str
    score_modifier: int; nutrient_modifier: int; hf_modifier: int
    meal_type_note: str; digestibility: str; hf_confidence: float; hf_note: str
    nutrient_profile: NutrientProfile
    nutrient_impacts: list[NutrientImpact] = field(default_factory=list)
    nutrient_notes: list[str] = field(default_factory=list)


@dataclass
class SymptomLogScore:
    symptom: str; severity: str; logged_at: str; hour: int
    score_penalty: int; severity_note: str; time_note: str; clinical_note: str


# ─────────────────────────────────────────────────────────────────────────────
# Nutrient scoring (unchanged pure-Python logic)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_nutrient_modifier(
    profile: NutrientProfile, scale: float,
) -> tuple[int, list[NutrientImpact], list[str]]:
    total_delta = 0
    impacts: list[NutrientImpact] = []
    notes:   list[str]            = []

    val_map: dict[str, Optional[float]] = {
        "protein": profile.protein, "calcium": profile.calcium,
        "potassium": profile.potassium, "vitamin_c": profile.vitamin_c,
        "vitamin_e": profile.vitamin_e, "vitamin_d": profile.vitamin_d,
        "iron": profile.iron, "total_fat": profile.total_fat,
        "sat_fat": profile.sat_fat, "cholesterol": profile.cholesterol,
        "sodium": profile.sodium, "sugar": profile.sugar,
        "carbs": profile.carbs, "calories": profile.calories,
    }

    for key, val in val_map.items():
        if val is None:
            continue
        cfg       = _NUTRIENT_CONFIG[key]
        direction = cfg["direction"]
        unit      = cfg["unit"]
        s         = scale

        if direction == "beneficial":
            low_t = cfg["low_threshold"] * s; high_t = cfg["high_threshold"] * s
            max_d = cfg["max_delta"]
            if val >= high_t:
                delta = max_d; note = f"✅ {val:.1f}{unit} {key.replace('_',' ')} — {cfg['note_excellent']}"; direction_label = "beneficial"
            elif val >= low_t:
                delta = round(max_d * 0.55); note = f"⚠ {val:.1f}{unit} {key.replace('_',' ')} — {cfg['note_good']}"; direction_label = "beneficial"
            else:
                continue

        elif direction == "harmful":
            mild_t = cfg["mild_threshold"] * s; high_t = cfg["high_threshold"] * s
            max_d  = cfg["max_delta"]
            if val >= high_t:
                delta = max_d; note = f"❌ {val:.1f}{unit} {key.replace('_',' ')} — {cfg['note_high']}"; direction_label = "harmful"
            elif val >= mild_t:
                delta = round(max_d * 0.55); note = f"⚠ {val:.1f}{unit} {key.replace('_',' ')} — {cfg['note_mild']}"; direction_label = "harmful"
            else:
                continue

        elif direction == "dual":
            good_lo = cfg["good_low"] * s; good_hi = cfg["good_high"] * s
            excess  = cfg["excess_threshold"] * s
            if val > excess:
                delta = cfg["min_delta"]; note = f"❌ {val:.1f}{unit} iron — {cfg['note_excess']}"; direction_label = "harmful"
            elif val >= good_lo:
                delta = cfg["max_delta"] if val >= good_hi else round(cfg["max_delta"] * 0.55)
                note  = f"✅ {val:.1f}{unit} iron — {cfg['note_good']}"; direction_label = "beneficial"
            else:
                continue
        else:
            continue

        total_delta += delta
        notes.append(note)
        impacts.append(NutrientImpact(
            nutrient=key.replace("_", " ").title(), value=round(val, 2),
            unit=unit, delta=delta, note=note, direction=direction_label,
        ))

    return total_delta, impacts, notes


# ─────────────────────────────────────────────────────────────────────────────
# Claude digestibility classification (replaces DeBERTa)
# ─────────────────────────────────────────────────────────────────────────────

_DIGESTIBILITY_SYSTEM = (
    "You are a clinical dietitian AI. Given a food description and its nutrients, "
    "classify its effect on human digestion. "
    "Return ONLY valid JSON with exactly this shape (no markdown, no extra keys):\n"
    '{"label": "easy to digest"|"difficult to digest", '
    '"confidence": 0.0-1.0, '
    '"score_delta": integer -5 to +5}'
)


def _run_claude_classification(desc: str, profile: NutrientProfile) -> tuple[int, str, float, str]:
    """Replace DeBERTa. Returns (hf_modifier, digestibility_label, confidence, hf_note)."""
    if _claude_client is None:
        return 0, "unknown", 0.0, "Claude client not available."

    parts = []
    if profile.calories    is not None: parts.append(f"{profile.calories:.0f}kcal")
    if profile.protein     is not None: parts.append(f"{profile.protein:.1f}g protein")
    if profile.total_fat   is not None: parts.append(f"{profile.total_fat:.1f}g fat")
    if profile.sat_fat     is not None: parts.append(f"{profile.sat_fat:.1f}g sat.fat")
    if profile.sugar       is not None: parts.append(f"{profile.sugar:.1f}g sugar")
    if profile.sodium      is not None: parts.append(f"{profile.sodium:.0f}mg sodium")
    if profile.fiber       is not None: parts.append(f"{profile.fiber:.1f}g fiber")  # type: ignore[attr-defined]

    nutrient_ctx = ", ".join(parts) if parts else "composition unknown"
    user_msg = (
        f"Food: {desc}. Nutrients per portion: {nutrient_ctx}. "
        f"Classify digestibility and return JSON."
    )

    try:
        msg = _claude_client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 120,
            system     = _DIGESTIBILITY_SYSTEM,
            messages   = [{"role": "user", "content": user_msg}],
        )
        raw  = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)

        label      = data.get("label", "unknown")
        confidence = float(data.get("confidence", 0.5))
        delta      = int(data.get("score_delta", 0))
        delta      = max(-6, min(+5, delta))

        note = (
            f"Claude classified as '{label}' "
            f"(confidence {confidence:.0%}) → adjustment: {delta:+d} pts"
        )
        return delta, label, round(confidence, 4), note

    except Exception as exc:
        log.warning(f"Claude digestibility classification failed: {exc}")
        return 0, "unknown", 0.0, f"Classification unavailable: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Public: analyse_food_log
# ─────────────────────────────────────────────────────────────────────────────

def analyse_food_log(
    usda_id: int, quantity: float, unit: str, meal_type: str,
) -> FoodLogScore:
    if _usda_client is None:
        raise RuntimeError("nutrition_scorer not initialised.")

    food_data = _usda_client.get_nutrients(usda_id)
    portion_g = _to_grams(quantity, unit)
    scale     = portion_g / 100.0

    if food_data is None:
        log.warning(f"USDA ID {usda_id} not found.")
        empty = NutrientProfile(portion_grams=portion_g)
        return FoodLogScore(
            usda_id=usda_id, food_description="Unknown", portion_grams=portion_g,
            meal_type=meal_type, score_modifier=0, nutrient_modifier=0, hf_modifier=0,
            meal_type_note="", digestibility="unknown", hf_confidence=0.0,
            nutrient_profile=empty, hf_note="Food not found in USDA dataset.",
        )

    desc = food_data.get("description", f"USDA ID {usda_id}")

    def sv(key: str) -> Optional[float]:
        v = food_data.get(key)
        return round(float(v) * scale, 3) if v is not None else None

    profile = NutrientProfile(
        calories=sv("calories"), protein=sv("protein"), total_fat=sv("total_fat"),
        carbs=sv("carbs"), sodium=sv("sodium"), sat_fat=sv("sat_fat"),
        cholesterol=sv("cholesterol"), sugar=sv("sugar"), calcium=sv("calcium"),
        iron=sv("iron"), potassium=sv("potassium"), vitamin_c=sv("vitamin_c"),
        vitamin_e=sv("vitamin_e"), vitamin_d=sv("vitamin_d"),
        fiber=sv("fiber"), portion_grams=portion_g,
    )

    nutrient_modifier, nutrient_impacts, nutrient_notes = _compute_nutrient_modifier(profile, scale)
    hf_modifier, digestibility, hf_confidence, hf_note   = _run_claude_classification(desc, profile)

    mt_mult  = _MEAL_MULTIPLIER.get(meal_type, 1.0)
    adjusted = round((nutrient_modifier + hf_modifier) * mt_mult)
    final    = max(-18, min(+12, adjusted))

    return FoodLogScore(
        usda_id=usda_id, food_description=desc, portion_grams=portion_g,
        meal_type=meal_type, score_modifier=final, nutrient_modifier=nutrient_modifier,
        hf_modifier=hf_modifier, meal_type_note=_MEAL_NOTE.get(meal_type, ""),
        digestibility=digestibility, hf_confidence=hf_confidence, hf_note=hf_note,
        nutrient_profile=profile, nutrient_impacts=nutrient_impacts,
        nutrient_notes=nutrient_notes,
    )


def analyse_food_log_batch(
    items: list[dict], meal_type: str,
) -> tuple[int, list[FoodLogScore]]:
    results = [
        analyse_food_log(
            usda_id=item["usda_id"], quantity=item["quantity"],
            unit=item["unit"], meal_type=meal_type,
        )
        for item in items
    ]
    total = sum(r.score_modifier for r in results)
    return max(-45, min(+35, total)), results


# ─────────────────────────────────────────────────────────────────────────────
# Symptom scoring (pure Python — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

_SYMPTOM_BASE_PENALTY: dict[str, int] = {
    "Bloating": -7, "Abdominal Pain": -12, "Nausea": -9, "Constipation": -10,
    "Heartburn": -10, "Gas": -5, "Fatigue": -6, "Acid Reflux": -10,
    "Cramps": -11, "Diarrhea": -12,
}
_SEVERITY_MULT = {"Mild": 0.5, "Moderate": 1.0, "Severe": 1.65}
_SEVERITY_NOTE = {
    "Mild":     "Mild symptom — minor gut irritation. Monitor if recurring.",
    "Moderate": "Moderate symptom — notable digestive stress. Track frequency.",
    "Severe":   "Severe symptom — significant gut health signal. Consult a doctor if persistent.",
}
_SYMPTOM_CLINICAL: dict[str, str] = {
    "Bloating":       "Bloating suggests fermentation imbalance, gut dysbiosis, or motility issues.",
    "Abdominal Pain": "Abdominal pain is a primary IBS/IBD indicator — warrants evaluation if recurring.",
    "Nausea":         "Nausea may indicate gastritis, H. pylori, or delayed gastric emptying.",
    "Constipation":   "Constipation reflects low fibre, poor hydration, or slow gut transit time.",
    "Heartburn":      "Heartburn indicates acid reflux / GERD — dietary + positional changes help.",
    "Gas":            "Excess gas is linked to microbiome imbalance or fermentable carbohydrate intake.",
    "Fatigue":        "Post-meal fatigue indicates gut-brain axis stress or nutrient malabsorption.",
    "Acid Reflux":    "Acid reflux indicates lower oesophageal sphincter weakness — avoid trigger foods.",
    "Cramps":         "Gut cramps suggest intestinal spasm — possibly IBS, infection, or food sensitivity.",
    "Diarrhea":       "Diarrhea severely disrupts absorption and electrolytes — urgent hydration needed.",
}


def _time_context(hour: int) -> tuple[float, str]:
    if 22 <= hour or hour < 6:
        return 1.30, "Nocturnal symptom (10 PM–6 AM) — sleep disruption × 1.30 amplifier applied."
    elif 6 <= hour < 10:
        return 1.08, "Morning symptom (6–10 AM) — may reflect overnight gut activity or fasting irritation."
    elif 18 <= hour < 22:
        return 1.12, "Evening symptom (6–10 PM) — slower post-dinner digestion × 1.12 amplifier applied."
    else:
        return 1.0, "Daytime symptom — standard gut stress context (no time amplifier)."


def analyse_symptom_log(symptom: str, severity: str, logged_at: datetime) -> SymptomLogScore:
    base      = _SYMPTOM_BASE_PENALTY.get(symptom, -6)
    sev_mult  = _SEVERITY_MULT.get(severity, 1.0)
    hour      = logged_at.hour
    time_mult, time_note = _time_context(hour)

    raw_penalty   = base * sev_mult * time_mult
    final_penalty = max(-22, round(raw_penalty))

    return SymptomLogScore(
        symptom=symptom, severity=severity, logged_at=logged_at.isoformat(),
        hour=hour, score_penalty=final_penalty,
        severity_note=_SEVERITY_NOTE.get(severity, ""),
        time_note=time_note,
        clinical_note=_SYMPTOM_CLINICAL.get(symptom, ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# score_meal_claude — single-call gut health score for a parsed meal
# ─────────────────────────────────────────────────────────────────────────────

_MEAL_SCORE_SYSTEM = """\
You are a clinical gut health dietitian AI evaluating a meal's overall impact
on gut health. You receive a list of foods with their weights and the meal type.

Assess three things:
1. Nutritional quality — fibre, fat, sugar, sodium, protein balance
2. Meal timing appropriateness — is this food right for this time of day?
3. Specific gut risks — fried food, spicy food, high fat at dinner, etc.

MEAL TIMING RULES:
- Breakfast: Light, easily digestible foods ideal. Fried or heavy = penalise.
- Lunch: Widest tolerance. Moderate portions of anything mostly fine.
- Dinner: Spicy, fatty, heavy foods are WORSE here (digestion slows during sleep).
- Snack: Light portions ideal. High sugar or heavy snacks penalised.

SCORING GUIDE (-5 to +5 integer):
+5  Excellent — gut-friendly, perfectly timed, anti-inflammatory
+3/+4  Good foods, well-timed, minor concerns only
+1/+2  Decent — some positives, mild concerns
 0  Neutral — mixed bag, roughly cancels out
-1/-2  Problematic elements — some gut irritants present
-3/-4  Clearly gut-unfriendly — fried/high-fat/high-sugar/poorly timed
-5  Very harmful — multiple gut stressors combined badly

Return ONLY valid JSON — no markdown, no explanation:
{
  "raw_score": <integer -5 to +5>,
  "note": "<exactly 10 to 12 words, clinically warm, specific to this meal>"
}
"""


def _meal_score_cache_key(foods: list[dict], meal_type: str) -> str:
    """
    SHA-256 key for a meal score cache entry.
    Deterministic: sorted by usda_description so order doesn't matter.
    Rounds weight_g to nearest 10g so 158g and 162g share the same cache slot.
    """
    parts = sorted(
        f"{f.get('usda_description', '')}:{round(f.get('weight_g', 0) / 10) * 10}g"
        for f in foods
        if f.get("weight_g", 0) > 0
    )
    payload = meal_type.lower() + "|" + "|".join(parts)
    return hashlib.sha256(payload.encode()).hexdigest()


def score_meal_claude(
    foods: list[dict],   # [{usda_description, weight_g}, ...]
    meal_type: str,
) -> tuple[int, str]:
    """
    Evaluate the gut health impact of a parsed meal via Claude.

    Cache strategy (MongoDB):
      - Key: SHA-256 of sorted food descriptions + rounded weights + meal_type.
      - On HIT:  return cached (raw_score, note) instantly — zero Claude call.
      - On MISS: call Claude, store result, return.
      - Weights rounded to nearest 10g so "chicken 158g" and "chicken 162g"
        share one cache entry without affecting scoring accuracy meaningfully.

    Parameters
    ----------
    foods     : list of {usda_description: str, weight_g: float}
    meal_type : "Breakfast" | "Lunch" | "Dinner" | "Snack"

    Returns
    -------
    (raw_score, note)
        raw_score : int in [-5, +5]
        note      : 10-12 word plain-English gut health observation
    """
    if _claude_client is None:
        return 0, "Scoring unavailable — Claude client not initialised."

    food_lines = "\n".join(
        f"  - {f.get('usda_description', 'Unknown food')} — {f.get('weight_g', 0):.0f}g"
        for f in foods
        if f.get("weight_g", 0) > 0
    )
    if not food_lines:
        return 0, "No foods to evaluate."

    # ── Cache lookup ──────────────────────────────────────────────────────────
    cache_key = _meal_score_cache_key(foods, meal_type)
    if _mongo_db is not None:
        try:
            cached = _mongo_db.get_meal_score(cache_key)
            if cached:
                log.info(f"meal_score cache HIT for key={cache_key[:12]}…")
                return cached   # (raw_score, note)
        except Exception as exc:
            log.warning(f"meal_score cache read failed: {exc}")

    # ── Cache miss: Claude call ───────────────────────────────────────────────
    user_msg = (
        f"Meal type: {meal_type}\n"
        f"Foods consumed:\n{food_lines}\n\n"
        f"Return the JSON score and note."
    )

    try:
        msg = _claude_client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 120,
            system     = _MEAL_SCORE_SYSTEM,
            messages   = [{"role": "user", "content": user_msg}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)

        raw_score = int(data.get("raw_score", 0))
        raw_score = max(-5, min(+5, raw_score))
        note      = str(data.get("note", "Meal evaluated.")).strip()

        # ── Store in cache ────────────────────────────────────────────────────
        if _mongo_db is not None:
            try:
                _mongo_db.set_meal_score(cache_key, raw_score, note)
                log.info(f"meal_score cache SET for key={cache_key[:12]}…")
            except Exception as exc:
                log.warning(f"meal_score cache write failed: {exc}")

        return raw_score, note

    except Exception as exc:
        log.warning(f"score_meal_claude failed: {exc}")
        return 0, "Unable to evaluate meal at this time."


# ─────────────────────────────────────────────────────────────────────────────
# Claude symptom log scorer
# Replaces all rule-based per-symptom penalty tables.
# Single call → (penalty: int 0-40, note: str 7-10 words)
# ─────────────────────────────────────────────────────────────────────────────

_SYMPTOM_SCORE_SYSTEM = """\
You are a digestive health scoring AI. A user has reported gut symptoms.
Generate a penalty score (0–40) that will be subtracted from their digestion score,
and a short clinical note (7–10 words exactly) summarising what was reported.

PENALTY SCALE — commit to it precisely:
  • 1 symptom, Mild, no note           →  3–5
  • 1 symptom, Moderate, no note       →  6–9
  • 1 symptom, Severe, no note         → 10–13
  • 2–3 symptoms, Moderate, no note    → 10–16
  • 4–6 symptoms, Moderate, no note    → 17–24
  • 7–9 symptoms, Severe, no note      → 25–32
  • 10–12 symptoms, Severe + bad note  → 33–40

NOTE RULES:
  • Exactly 7–10 words. No more, no less.
  • Clinical, neutral tone. No exclamation marks.
  • Must reference why the dominant symptom or severity may appear.
  • Example: "Severe bloating are may be caused by high-fat intake."

Return ONLY valid JSON, no markdown:
{"penalty": <integer 0-40>, "note": "<7-10 word string>"}
"""


def score_symptom_log_claude(
    symptoms: list[str],
    severity: str,
    note:     Optional[str],
) -> tuple[int, str]:
    """
    Call Claude to score a symptom log entry.

    Returns
    -------
    (penalty, note_text)
      penalty   : int in [0, 40] — subtracted from current score
      note_text : 7–10 word clinical summary string
    """
    if _claude_client is None:
        # Deterministic fallback if Claude is unavailable
        base = {"Mild": 4, "Moderate": 8, "Severe": 13}.get(severity, 6)
        penalty = min(40, base * max(1, len(symptoms)))
        return penalty, f"{severity.lower()} {symptoms[0].lower()} and related symptoms reported."

    symptom_str = ", ".join(symptoms) if symptoms else "unspecified"
    note_str    = f'User note: "{note.strip()}"' if note and note.strip() else "No additional note."

    user_msg = (
        f"Symptoms reported: {symptom_str}\n"
        f"Severity: {severity}\n"
        f"Number of symptoms: {len(symptoms)}\n"
        f"{note_str}\n\n"
        f"Return the JSON penalty and note."
    )

    try:
        msg = _claude_client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 80,
            system     = _SYMPTOM_SCORE_SYSTEM,
            messages   = [{"role": "user", "content": user_msg}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        data    = json.loads(raw)
        penalty = int(data.get("penalty", 5))
        penalty = max(0, min(40, penalty))          # hard clamp to [0, 40]
        note_out = str(data.get("note", "")).strip()
        if not note_out:
            note_out = f"{severity.lower()} {symptoms[0].lower()} symptoms logged by user."
        return penalty, note_out

    except Exception as exc:
        log.warning(f"score_symptom_log_claude failed: {exc}")
        base = {"Mild": 4, "Moderate": 8, "Severe": 13}.get(severity, 6)
        return min(40, base * max(1, len(symptoms))), "Symptom scoring temporarily unavailable, defaults applied."