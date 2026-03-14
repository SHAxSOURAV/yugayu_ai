"""
nutrition_scorer.py
───────────────────
HuggingFace-powered gut-health scoring engine.

Model: MoritzLaurer/deberta-v3-base-mnli-fever-anli
  ─ State-of-the-art NLI zero-shot classifier
  ─ Significantly more accurate than distilbert/BART for nuanced labels
  ─ ~180 MB, CPU-friendly, no fine-tuning needed

Why DeBERTa-v3-base over distilbert-mnli?
  • DeBERTa uses disentangled attention (position + content separate)
    → better at understanding nutrient-description context pairs
  • Trained on MNLI + FEVER + ANLI → more robust to adversarial phrasing
  • Consistently ranks #1 on zero-shot NLI benchmarks (SuperGLUE, ANLI)

USDA columns used (exact names from demomaster/usda-national-nutrient-database):
  ID, Description, Calories, Protein, TotalFat, Carbohydrate, Sodium,
  SaturatedFat, Cholesterol, Sugar, Calcium, Iron, Potassium,
  VitaminC, VitaminE, VitaminD

Two public functions:
  analyse_food_log_batch()   — called from POST /log/food
  analyse_symptom_log()      — called from POST /log/symptom
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from transformers import pipeline

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace Model
# MoritzLaurer/deberta-v3-base-mnli-fever-anli
# Best-in-class zero-shot NLI on CPU — trained on MNLI + FEVER + ANLI
# ─────────────────────────────────────────────────────────────────────────────

HF_MODEL = "MoritzLaurer/deberta-v3-base-mnli-fever-anli"

log.info(f"Loading zero-shot classifier: {HF_MODEL} ...")
_classifier = pipeline(
    "zero-shot-classification",
    model  = HF_MODEL,
    device = -1,   # -1 = CPU; set to 0 for GPU
)
log.info("Zero-shot classifier ready.")

# Fine-grained digestibility labels — DeBERTa handles these nuanced pairs well
_DIGESTIBILITY_LABELS = [
    "easy to digest and gut-friendly",
    "difficult to digest and gut-irritating",
    "high in gut-beneficial nutrients like fiber and vitamins",
    "high in gut-harmful nutrients like saturated fat and sodium",
    "promotes healthy gut microbiome",
    "causes digestive inflammation or irritation",
    "supports gut lining and motility",
    "triggers bloating, reflux, or bowel disturbance",
]

# Score delta per label, scaled later by model confidence
_HF_LABEL_DELTA: dict[str, float] = {
    "easy to digest and gut-friendly":                        +4.0,
    "high in gut-beneficial nutrients like fiber and vitamins":+3.5,
    "promotes healthy gut microbiome":                         +3.0,
    "supports gut lining and motility":                        +2.5,
    "difficult to digest and gut-irritating":                 -4.0,
    "high in gut-harmful nutrients like saturated fat and sodium": -3.5,
    "causes digestive inflammation or irritation":            -4.0,
    "triggers bloating, reflux, or bowel disturbance":        -3.5,
}


# ─────────────────────────────────────────────────────────────────────────────
# USDA Column Map — exact column names from the Kaggle dataset
# Fallback aliases handle minor naming variations across dataset versions
# ─────────────────────────────────────────────────────────────────────────────

# Primary names (from demomaster/usda-national-nutrient-database)
USDA_COLS = {
    "id":          "ID",
    "description": "Description",
    "calories":    "Calories",
    "protein":     "Protein",
    "total_fat":   "TotalFat",
    "carbs":       "Carbohydrate",
    "sodium":      "Sodium",
    "sat_fat":     "SaturatedFat",
    "cholesterol": "Cholesterol",
    "sugar":       "Sugar",
    "calcium":     "Calcium",
    "iron":        "Iron",
    "potassium":   "Potassium",
    "vitamin_c":   "VitaminC",
    "vitamin_e":   "VitaminE",
    "vitamin_d":   "VitaminD",
}

# Fallback aliases if dataset uses different casing or naming
_FALLBACK_ALIASES: dict[str, list[str]] = {
    "id":          ["ID","id","NDB_No","fdc_id","FDC_ID","FOOD_ID"],
    "description": ["Description","description","DESCRIPTION","Long_Desc","long_desc","NAME","name"],
    "calories":    ["Calories","calories","Energ_Kcal","ENERG_KCAL","Energy","energy","kcal"],
    "protein":     ["Protein","protein","Protein_g","PROTEIN"],
    "total_fat":   ["TotalFat","Total_Fat","total_fat","Lipid_Tot","LIPID_TOT","Fat","fat"],
    "carbs":       ["Carbohydrate","carbohydrate","Carbohydrt","CARBOHYDRT","Carbs","carbs"],
    "sodium":      ["Sodium","sodium","SODIUM","Sodium_mg"],
    "sat_fat":     ["SaturatedFat","Saturated_Fat","saturated_fat","FA_Sat","FA_SAT","fa_sat"],
    "cholesterol": ["Cholesterol","cholesterol","CHOLESTEROL","Cholestrl","cholestrl"],
    "sugar":       ["Sugar","sugar","Sugar_Tot","SUGAR_TOT","Sugars","sugars"],
    "calcium":     ["Calcium","calcium","CALCIUM","Calcium_mg"],
    "iron":        ["Iron","iron","IRON","Iron_mg"],
    "potassium":   ["Potassium","potassium","POTASSIUM","Potassium_mg"],
    "vitamin_c":   ["VitaminC","Vitamin_C","vitamin_c","Vit_C","VIT_C","Vitamin C"],
    "vitamin_e":   ["VitaminE","Vitamin_E","vitamin_e","Vit_E","VIT_E","Vitamin E"],
    "vitamin_d":   ["VitaminD","Vitamin_D","vitamin_d","Vit_D","VIT_D","Vitamin D"],
}

def _resolve_col(df: pd.DataFrame, key: str) -> Optional[str]:
    """
    Resolve a nutrient key to the actual column name in df.
    Tries primary name first, then all fallback aliases, then case-insensitive match.
    """
    primary = USDA_COLS.get(key)
    if primary and primary in df.columns:
        return primary
    lower_map = {c.lower(): c for c in df.columns}
    for alias in _FALLBACK_ALIASES.get(key, []):
        if alias in df.columns:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def _get_val(row: pd.Series, df: pd.DataFrame, key: str, scale: float = 1.0) -> Optional[float]:
    """Extract and scale a nutrient value from a USDA row."""
    col = _resolve_col(df, key)
    if col is None:
        return None
    try:
        v = float(row.get(col, float("nan")))
        return round(v * scale, 3) if not np.isnan(v) else None
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Clinical Nutrient Thresholds
# All thresholds are per 100g in USDA; scaled to actual portion in code.
#
# Research references (per 100g):
#   Fiber: WHO recommends 25g/day total → >5g/100g = high
#   Sodium: WHO <2000mg/day → >400mg/100g = high; >800mg = very high
#   SaturatedFat: AHA <13g/day → >5g/100g = high; >10g = very high
#   Cholesterol: AHA <300mg/day → >100mg/100g = elevated
#   Sugar: WHO <25g/day free sugars → >10g/100g = high; >20g = very high
#   Potassium: adequate intake 3500mg/day → >300mg/100g = good
#   Vitamins: presence in meaningful amounts = gut-protective
# ─────────────────────────────────────────────────────────────────────────────

# Per-nutrient: (beneficial_threshold, harmful_threshold, max_delta)
# beneficial_threshold → value above which it HELPS digestion
# harmful_threshold    → value above which it HURTS digestion
# max_delta            → max absolute score change from this nutrient

_NUTRIENT_CONFIG: dict[str, dict] = {
    # ── GUT-BENEFICIAL nutrients ─────────────────────────────────────────────
    "protein": {
        "direction": "beneficial",
        "low_threshold":  8.0,    # >8g/100g = good protein
        "high_threshold": 20.0,   # >20g/100g = excellent
        "max_delta":      +4,
        "unit": "g",
        "note_good": "Good protein content supports gut tissue repair and enzyme production.",
        "note_excellent": "High protein supports gut lining integrity and digestive enzyme synthesis.",
    },
    "calcium": {
        "direction": "beneficial",
        "low_threshold":  80.0,   # >80mg/100g = meaningful
        "high_threshold": 200.0,  # >200mg/100g = excellent
        "max_delta":      +3,
        "unit": "mg",
        "note_good": "Calcium supports smooth muscle contractions in the gut.",
        "note_excellent": "High calcium aids gut muscle function and may reduce colorectal risk.",
    },
    "potassium": {
        "direction": "beneficial",
        "low_threshold":  200.0,  # >200mg/100g = good
        "high_threshold": 400.0,  # >400mg/100g = excellent
        "max_delta":      +4,
        "unit": "mg",
        "note_good": "Potassium supports gut muscle contractions and motility.",
        "note_excellent": "High potassium is strongly linked to improved gut motility and reduced constipation.",
    },
    "vitamin_c": {
        "direction": "beneficial",
        "low_threshold":  10.0,   # >10mg/100g = meaningful
        "high_threshold": 40.0,   # >40mg/100g = high
        "max_delta":      +3,
        "unit": "mg",
        "note_good": "Vitamin C supports gut antioxidant defence and iron absorption.",
        "note_excellent": "High Vitamin C strongly supports gut lining integrity and microbiome diversity.",
    },
    "vitamin_e": {
        "direction": "beneficial",
        "low_threshold":  1.0,    # >1mg/100g = meaningful
        "high_threshold": 5.0,    # >5mg/100g = high
        "max_delta":      +2,
        "unit": "mg",
        "note_good": "Vitamin E protects gut cell membranes from oxidative stress.",
        "note_excellent": "High Vitamin E provides strong anti-inflammatory protection for the gut lining.",
    },
    "vitamin_d": {
        "direction": "beneficial",
        "low_threshold":  1.0,    # >1µg/100g = meaningful
        "high_threshold": 5.0,    # >5µg/100g = high
        "max_delta":      +3,
        "unit": "µg",
        "note_good": "Vitamin D supports gut barrier function and immune-microbiome regulation.",
        "note_excellent": "High Vitamin D is strongly linked to reduced intestinal permeability ('leaky gut').",
    },
    "iron": {
        "direction": "dual",      # moderate = good, excess = bad (constipation)
        "good_low":  2.0,
        "good_high": 8.0,
        "excess_threshold": 15.0,
        "max_delta":  +2,
        "min_delta":  -3,
        "unit": "mg",
        "note_good":    "Adequate iron supports gut tissue oxygenation and cell renewal.",
        "note_excess":  "Very high iron can cause constipation and oxidative gut stress.",
    },
    # ── GUT-HARMFUL nutrients ────────────────────────────────────────────────
    "total_fat": {
        "direction": "harmful",
        "mild_threshold":  12.0,  # >12g/100g = elevated
        "high_threshold":  20.0,  # >20g/100g = very high
        "max_delta":       -5,
        "unit": "g",
        "note_mild": "Elevated fat content slows gastric emptying, may worsen reflux.",
        "note_high": "Very high fat significantly delays digestion and increases acid reflux risk.",
    },
    "sat_fat": {
        "direction": "harmful",
        "mild_threshold":  4.0,   # >4g/100g = elevated
        "high_threshold":  8.0,   # >8g/100g = very high
        "max_delta":       -5,
        "unit": "g",
        "note_mild": "Elevated saturated fat promotes gut inflammation and reduces microbiome diversity.",
        "note_high": "High saturated fat is strongly linked to gut dysbiosis and increased intestinal permeability.",
    },
    "cholesterol": {
        "direction": "harmful",
        "mild_threshold":  60.0,  # >60mg/100g = elevated
        "high_threshold":  120.0, # >120mg/100g = very high
        "max_delta":       -3,
        "unit": "mg",
        "note_mild": "Elevated cholesterol content may contribute to bile overproduction and digestive stress.",
        "note_high": "High cholesterol is associated with increased gallbladder and digestive burden.",
    },
    "sodium": {
        "direction": "harmful",
        "mild_threshold":  300.0, # >300mg/100g = elevated
        "high_threshold":  600.0, # >600mg/100g = very high
        "max_delta":       -5,
        "unit": "mg",
        "note_mild": "Elevated sodium promotes gut wall inflammation and disrupts gut microbiome balance.",
        "note_high": "Very high sodium significantly damages gut barrier function and drives microbiome dysbiosis.",
    },
    "sugar": {
        "direction": "harmful",
        "mild_threshold":  8.0,   # >8g/100g = elevated
        "high_threshold":  18.0,  # >18g/100g = very high
        "max_delta":       -5,
        "unit": "g",
        "note_mild": "Elevated sugar feeds harmful gut bacteria and may worsen bloating and gas.",
        "note_high": "Very high sugar rapidly disrupts gut microbiome diversity and promotes gut dysbiosis.",
    },
    "carbs": {
        "direction": "harmful",
        "mild_threshold":  35.0,  # >35g/100g = elevated (refined carbs)
        "high_threshold":  65.0,  # >65g/100g = very high
        "max_delta":       -3,
        "unit": "g",
        "note_mild": "High refined carbohydrate content may spike gut fermentation and bloating.",
        "note_high": "Very high carbohydrate load (likely refined) drives excessive gut fermentation.",
    },
    "calories": {
        "direction": "harmful",
        "mild_threshold":  250.0, # >250 kcal/100g = energy-dense
        "high_threshold":  450.0, # >450 kcal/100g = very dense
        "max_delta":       -3,
        "unit": "kcal",
        "note_mild": "High caloric density slows digestion and increases digestive workload.",
        "note_high": "Very high caloric density places significant burden on digestive enzymes and motility.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Meal-type multipliers — same meal hits differently at different times
# ─────────────────────────────────────────────────────────────────────────────

_MEAL_MULTIPLIER = {
    "Breakfast": 0.85,  # active gut, tolerates slightly more
    "Lunch":     1.00,  # neutral baseline
    "Dinner":    1.15,  # slower evening peristalsis
    "Snack":     0.90,  # smaller context
}

_MEAL_NOTE = {
    "Breakfast": "Breakfast multiplier (×0.85) — morning digestion is active, gut tolerates more.",
    "Lunch":     "Lunch multiplier (×1.00) — neutral baseline context.",
    "Dinner":    "Dinner multiplier (×1.15) — evening digestion is slower; impact amplified.",
    "Snack":     "Snack multiplier (×0.90) — smaller portion context, slightly reduced impact.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Unit → grams conversion table
# ─────────────────────────────────────────────────────────────────────────────

_UNIT_TO_GRAMS = {
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "kg": 1000.0,
    "ml": 1.0, "l": 1000.0,           # water-density approx for liquids
    "oz": 28.3495, "lb": 453.592,
    "piece": 100.0, "pieces": 100.0,
    "cup": 240.0,
    "tbsp": 15.0, "tsp": 5.0,
    "serving": 100.0,
}

def _to_grams(quantity: float, unit: str) -> float:
    return quantity * _UNIT_TO_GRAMS.get(unit.lower().strip(), 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NutrientProfile:
    """All 16 USDA columns scaled to actual portion."""
    calories:    Optional[float] = None
    protein:     Optional[float] = None
    total_fat:   Optional[float] = None
    carbs:       Optional[float] = None
    sodium:      Optional[float] = None
    sat_fat:     Optional[float] = None
    cholesterol: Optional[float] = None
    sugar:       Optional[float] = None
    calcium:     Optional[float] = None
    iron:        Optional[float] = None
    potassium:   Optional[float] = None
    vitamin_c:   Optional[float] = None
    vitamin_e:   Optional[float] = None
    vitamin_d:   Optional[float] = None
    portion_grams: float = 100.0


@dataclass
class NutrientImpact:
    """Per-nutrient score contribution with reasoning."""
    nutrient:  str
    value:     float
    unit:      str
    delta:     int     # score points this nutrient contributes
    note:      str
    direction: str     # "beneficial" | "harmful" | "neutral"


@dataclass
class FoodLogScore:
    usda_id:           int
    food_description:  str
    portion_grams:     float
    meal_type:         str
    # Final combined modifier
    score_modifier:    int
    # Sub-components
    nutrient_modifier: int
    hf_modifier:       int
    meal_type_note:    str
    # HF details
    digestibility:     str
    hf_confidence:     float
    hf_note:           str
    # Nutrients
    nutrient_profile:  NutrientProfile
    nutrient_impacts:  list[NutrientImpact] = field(default_factory=list)
    nutrient_notes:    list[str]            = field(default_factory=list)


@dataclass
class SymptomLogScore:
    symptom:       str
    severity:      str
    logged_at:     str
    hour:          int
    score_penalty: int
    severity_note: str
    time_note:     str
    clinical_note: str


# ─────────────────────────────────────────────────────────────────────────────
# Food nutrient scoring — all 16 USDA columns
# ─────────────────────────────────────────────────────────────────────────────

def _compute_nutrient_modifier(
    profile: NutrientProfile,
    scale: float,
) -> tuple[int, list[NutrientImpact], list[str]]:
    """
    Compute score modifier from all 16 USDA nutrient columns.

    scale = portion_grams / 100.0
    Thresholds in _NUTRIENT_CONFIG are per 100g; we scale them by portion.
    """
    total_delta   = 0
    impacts:  list[NutrientImpact] = []
    notes:    list[str]            = []

    # Map nutrient key → current (scaled) value
    val_map: dict[str, Optional[float]] = {
        "protein":     profile.protein,
        "calcium":     profile.calcium,
        "potassium":   profile.potassium,
        "vitamin_c":   profile.vitamin_c,
        "vitamin_e":   profile.vitamin_e,
        "vitamin_d":   profile.vitamin_d,
        "iron":        profile.iron,
        "total_fat":   profile.total_fat,
        "sat_fat":     profile.sat_fat,
        "cholesterol": profile.cholesterol,
        "sodium":      profile.sodium,
        "sugar":       profile.sugar,
        "carbs":       profile.carbs,
        "calories":    profile.calories,
    }

    for key, val in val_map.items():
        if val is None:
            continue

        cfg = _NUTRIENT_CONFIG[key]
        direction = cfg["direction"]
        unit      = cfg["unit"]

        # Scale thresholds to actual portion
        s = scale

        if direction == "beneficial":
            low_t  = cfg["low_threshold"]  * s
            high_t = cfg["high_threshold"] * s
            max_d  = cfg["max_delta"]

            if val >= high_t:
                delta = max_d
                note  = f"✅ {val:.1f}{unit} {key.replace('_',' ')} — {cfg['note_excellent']}"
                direction_label = "beneficial"
            elif val >= low_t:
                delta = round(max_d * 0.55)
                note  = f"⚠ {val:.1f}{unit} {key.replace('_',' ')} — {cfg['note_good']}"
                direction_label = "beneficial"
            else:
                delta = 0
                continue

        elif direction == "harmful":
            mild_t = cfg["mild_threshold"] * s
            high_t = cfg["high_threshold"] * s
            max_d  = cfg["max_delta"]

            if val >= high_t:
                delta = max_d
                note  = f"❌ {val:.1f}{unit} {key.replace('_',' ')} — {cfg['note_high']}"
                direction_label = "harmful"
            elif val >= mild_t:
                delta = round(max_d * 0.55)
                note  = f"⚠ {val:.1f}{unit} {key.replace('_',' ')} — {cfg['note_mild']}"
                direction_label = "harmful"
            else:
                delta = 0
                continue

        elif direction == "dual":
            good_lo = cfg["good_low"]  * s
            good_hi = cfg["good_high"] * s
            excess  = cfg["excess_threshold"] * s

            if val > excess:
                delta = cfg["min_delta"]
                note  = f"❌ {val:.1f}{unit} iron — {cfg['note_excess']}"
                direction_label = "harmful"
            elif val >= good_lo:
                delta = cfg["max_delta"] if val >= good_hi else round(cfg["max_delta"] * 0.55)
                note  = f"✅ {val:.1f}{unit} iron — {cfg['note_good']}"
                direction_label = "beneficial"
            else:
                delta = 0
                continue
        else:
            continue

        total_delta += delta
        notes.append(note)
        impacts.append(NutrientImpact(
            nutrient  = key.replace("_", " ").title(),
            value     = round(val, 2),
            unit      = unit,
            delta     = delta,
            note      = note,
            direction = direction_label,
        ))

    return total_delta, impacts, notes


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace classification with rich nutrient context
# ─────────────────────────────────────────────────────────────────────────────

def _run_hf_classification(desc: str, profile: NutrientProfile) -> tuple[int, str, float, str]:
    """
    Build a rich nutrient-aware input string for the DeBERTa classifier.
    Returns (hf_modifier, top_label, confidence, hf_note)
    """
    parts = []
    if profile.calories    is not None: parts.append(f"{profile.calories:.0f}kcal")
    if profile.protein     is not None: parts.append(f"{profile.protein:.1f}g protein")
    if profile.total_fat   is not None: parts.append(f"{profile.total_fat:.1f}g fat")
    if profile.sat_fat     is not None: parts.append(f"{profile.sat_fat:.1f}g sat.fat")
    if profile.sugar       is not None: parts.append(f"{profile.sugar:.1f}g sugar")
    if profile.sodium      is not None: parts.append(f"{profile.sodium:.0f}mg sodium")
    if profile.cholesterol is not None: parts.append(f"{profile.cholesterol:.0f}mg cholesterol")
    if profile.calcium     is not None: parts.append(f"{profile.calcium:.0f}mg calcium")
    if profile.potassium   is not None: parts.append(f"{profile.potassium:.0f}mg potassium")
    if profile.iron        is not None: parts.append(f"{profile.iron:.1f}mg iron")
    if profile.vitamin_c   is not None: parts.append(f"{profile.vitamin_c:.1f}mg VitC")
    if profile.vitamin_d   is not None: parts.append(f"{profile.vitamin_d:.1f}µg VitD")

    nutrient_ctx = ", ".join(parts) if parts else "nutritional composition unknown"
    hf_input     = (
        f"Food: {desc}. "
        f"Nutritional composition per portion: {nutrient_ctx}. "
        f"Effect on human digestive system:"
    )

    result    = _classifier(hf_input, candidate_labels=_DIGESTIBILITY_LABELS, multi_label=True)
    top_label = result["labels"][0]
    top_conf  = float(result["scores"][0])

    # Weighted modifier: top 2 labels contribute, scaled by confidence
    hf_modifier = 0.0
    for label, score in zip(result["labels"][:3], result["scores"][:3]):
        hf_modifier += _HF_LABEL_DELTA.get(label, 0) * float(score)
    hf_modifier_int = max(-6, min(+5, round(hf_modifier)))

    hf_note = (
        f"DeBERTa-v3 classified as '{top_label}' "
        f"(confidence {top_conf:.0%}) → HF adjustment: {hf_modifier_int:+d} pts"
    )
    return hf_modifier_int, top_label, round(top_conf, 4), hf_note


# ─────────────────────────────────────────────────────────────────────────────
# Public: analyse_food_log — single food item
# ─────────────────────────────────────────────────────────────────────────────

def analyse_food_log(
    usda_id:   int,
    quantity:  float,
    unit:      str,
    meal_type: str,
    usda_df:   pd.DataFrame,
    id_col:    str,
    desc_col:  str,
) -> FoodLogScore:
    """
    Full nutritional + HuggingFace analysis for one logged food item.

    Steps
    -----
    1. Fetch all 16 nutrient columns from USDA by usda_id
    2. Scale all values from /100g to actual portion (quantity × unit → grams)
    3. Score each of the 14 nutrients against clinical thresholds
    4. Run DeBERTa zero-shot with full nutrient context for digestibility label
    5. Apply meal-type multiplier
    6. Clamp to [-18, +12] per food
    """
    # ── Row lookup ────────────────────────────────────────────────────────────
    id_col_actual   = _resolve_col(usda_df, "id")   or id_col
    desc_col_actual = _resolve_col(usda_df, "description") or desc_col

    mask = usda_df[id_col_actual].astype(str) == str(usda_id)
    if not mask.any():
        log.warning(f"USDA ID {usda_id} not found.")
        empty_profile = NutrientProfile(portion_grams=_to_grams(quantity, unit))
        return FoodLogScore(
            usda_id=usda_id, food_description="Unknown",
            portion_grams=empty_profile.portion_grams, meal_type=meal_type,
            score_modifier=0, nutrient_modifier=0, hf_modifier=0,
            meal_type_note="", digestibility="unknown", hf_confidence=0.0,
            nutrient_profile=empty_profile, hf_note="Food not found in USDA dataset.",
        )

    row  = usda_df[mask].iloc[0]
    desc = str(row[desc_col_actual])

    # ── Portion scaling ───────────────────────────────────────────────────────
    portion_g = _to_grams(quantity, unit)
    scale     = portion_g / 100.0

    def gv(key: str) -> Optional[float]:
        return _get_val(row, usda_df, key, scale=scale)

    profile = NutrientProfile(
        calories     = gv("calories"),
        protein      = gv("protein"),
        total_fat    = gv("total_fat"),
        carbs        = gv("carbs"),
        sodium       = gv("sodium"),
        sat_fat      = gv("sat_fat"),
        cholesterol  = gv("cholesterol"),
        sugar        = gv("sugar"),
        calcium      = gv("calcium"),
        iron         = gv("iron"),
        potassium    = gv("potassium"),
        vitamin_c    = gv("vitamin_c"),
        vitamin_e    = gv("vitamin_e"),
        vitamin_d    = gv("vitamin_d"),
        portion_grams= portion_g,
    )

    # ── Nutrient scoring (all 16 columns) ─────────────────────────────────────
    nutrient_modifier, nutrient_impacts, nutrient_notes = _compute_nutrient_modifier(profile, scale)

    # ── HuggingFace DeBERTa classification ────────────────────────────────────
    hf_modifier, digestibility, hf_confidence, hf_note = _run_hf_classification(desc, profile)

    # ── Meal-type multiplier ──────────────────────────────────────────────────
    mt_mult    = _MEAL_MULTIPLIER.get(meal_type, 1.0)
    raw        = nutrient_modifier + hf_modifier
    adjusted   = round(raw * mt_mult)
    final      = max(-18, min(+12, adjusted))

    return FoodLogScore(
        usda_id           = usda_id,
        food_description  = desc,
        portion_grams     = portion_g,
        meal_type         = meal_type,
        score_modifier    = final,
        nutrient_modifier = nutrient_modifier,
        hf_modifier       = hf_modifier,
        meal_type_note    = _MEAL_NOTE.get(meal_type, ""),
        digestibility     = digestibility,
        hf_confidence     = hf_confidence,
        hf_note           = hf_note,
        nutrient_profile  = profile,
        nutrient_impacts  = nutrient_impacts,
        nutrient_notes    = nutrient_notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public: analyse_food_log_batch — full meal
# ─────────────────────────────────────────────────────────────────────────────

def analyse_food_log_batch(
    items:     list[dict],
    meal_type: str,
    usda_df:   pd.DataFrame,
    id_col:    str,
    desc_col:  str,
) -> tuple[int, list[FoodLogScore]]:
    """
    Analyse all foods in a meal.
    items = [{"usda_id": int, "quantity": float, "unit": str}, ...]
    Returns (total_modifier clamped to [-45, +35], list[FoodLogScore])
    """
    results = [
        analyse_food_log(
            usda_id   = item["usda_id"],
            quantity  = item["quantity"],
            unit      = item["unit"],
            meal_type = meal_type,
            usda_df   = usda_df,
            id_col    = id_col,
            desc_col  = desc_col,
        )
        for item in items
    ]
    total = sum(r.score_modifier for r in results)
    return max(-45, min(+35, total)), results


# ─────────────────────────────────────────────────────────────────────────────
# Symptom scoring tables
# ─────────────────────────────────────────────────────────────────────────────

_SYMPTOM_BASE_PENALTY: dict[str, int] = {
    "Bloating":       -7,
    "Abdominal Pain": -12,
    "Nausea":         -9,
    "Constipation":   -10,
    "Heartburn":      -10,
    "Gas":            -5,
    "Fatigue":        -6,
    "Acid Reflux":    -10,
    "Cramps":         -11,
    "Diarrhea":       -12,
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
        return 1.30, (
            "Nocturnal symptom (10 PM–6 AM) — sleep disruption × 1.30 amplifier applied. "
            "Night symptoms indicate the gut is stressed during its recovery window."
        )
    elif 6 <= hour < 10:
        return 1.08, (
            "Morning symptom (6–10 AM) — may reflect overnight gut activity or fasting irritation."
        )
    elif 18 <= hour < 22:
        return 1.12, (
            "Evening symptom (6–10 PM) — slower post-dinner digestion × 1.12 amplifier applied."
        )
    else:
        return 1.0, "Daytime symptom — standard gut stress context (no time amplifier)."


# ─────────────────────────────────────────────────────────────────────────────
# Public: analyse_symptom_log
# ─────────────────────────────────────────────────────────────────────────────

def analyse_symptom_log(
    symptom:   str,
    severity:  str,
    logged_at: datetime,
) -> SymptomLogScore:
    """
    Compute score penalty for a reported symptom.

    Penalty = base_penalty × severity_multiplier × time_of_day_multiplier
    Capped at -22 per symptom entry.
    """
    base      = _SYMPTOM_BASE_PENALTY.get(symptom, -6)
    sev_mult  = _SEVERITY_MULT.get(severity, 1.0)
    hour      = logged_at.hour
    time_mult, time_note = _time_context(hour)

    raw_penalty   = base * sev_mult * time_mult
    final_penalty = max(-22, round(raw_penalty))

    return SymptomLogScore(
        symptom       = symptom,
        severity      = severity,
        logged_at     = logged_at.isoformat(),
        hour          = hour,
        score_penalty = final_penalty,
        severity_note = _SEVERITY_NOTE.get(severity, ""),
        time_note     = time_note,
        clinical_note = _SYMPTOM_CLINICAL.get(symptom, ""),
    )