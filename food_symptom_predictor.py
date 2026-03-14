"""
food_symptom_predictor.py
─────────────────────────
Predicts which logged food(s) most likely caused a reported symptom.

━━━ THE PROBLEM ━━━
Food and symptoms are never logged at the same time.
A person eats at 1 PM, then feels bloated at 4 PM.
The model must reason: "Which food eaten BEFORE this symptom, within the
clinically correct digestion window, is the most likely cause?"

━━━ APPROACH ━━━

Stage 1 — Temporal window filter
  Each symptom has a known clinical digestion window (e.g. Heartburn
  appears 30 min–2 h after eating; Constipation appears 12–48 h later).
  Only foods eaten within that window before the symptom are candidates.

Stage 2 — HuggingFace NLI causation scorer
  Model: cross-encoder/nli-deberta-v3-small
  ─ Specifically trained for Natural Language Inference (entailment/
    contradiction/neutral) — ideal for hypothesis testing:
    "Does eating [food with these nutrients] ENTAIL [this symptom]?"
  ─ cross-encoder architecture reads premise + hypothesis together →
    richer contextual understanding than bi-encoder or zero-shot
  ─ ~170 MB, fast CPU inference, no fine-tuning needed

  Why cross-encoder/nli-deberta-v3-small over the existing zero-shot pipeline?
  • The existing pipeline (deberta-v3-base-mnli) is optimised for
    zero-shot classification with predefined labels.
  • For food→symptom causation we need PAIRWISE reasoning between a
    specific food description and a specific symptom — exactly what
    cross-encoder NLI is built for.
  • The cross-encoder processes the concatenated [food premise] [SEP]
    [symptom hypothesis] in a single pass → captures token-level
    interactions between food properties and symptom patterns.
  • deberta-v3-small is ~2× faster than base with <5% accuracy loss.

Stage 3 — Evidence scoring
  Each candidate food gets a final causation_score combining:
  • NLI entailment probability (HuggingFace model)
  • Nutrient risk score (how pro-inflammatory/irritating the food is)
  • Temporal proximity score (sooner within window = higher weight)
  • Quantity weight (larger portion = higher likelihood of causation)

Public functions:
  predict_food_symptom_causes()  — called from POST /predict/food-symptom
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace Model
# cross-encoder/nli-deberta-v3-small
# Pairwise NLI: reads [food premise] + [symptom hypothesis] together
# Returns: entailment / neutral / contradiction probabilities
# ─────────────────────────────────────────────────────────────────────────────

HF_MODEL = "cross-encoder/nli-deberta-v3-small"

log.info(f"Loading food-symptom NLI model: {HF_MODEL} ...")

_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
_nli_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
_nli_model.eval()

# DeBERTa NLI label order: contradiction=0, entailment=1, neutral=2
# (confirmed from model card — cross-encoder/nli-deberta-v3-small)
_LABEL_ORDER = {0: "contradiction", 1: "entailment", 2: "neutral"}

log.info("Food-symptom NLI model ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Clinical digestion windows per symptom
# ─────────────────────────────────────────────────────────────────────────────
# Based on gastroenterology literature:
#   • Upper GI symptoms (heartburn, nausea, acid reflux) appear fast: 30min–3h
#   • Fermentation symptoms (bloating, gas, cramps) appear in mid range: 1h–8h
#   • Lower GI symptoms (diarrhea, constipation) appear later: 2h–48h
#   • Systemic symptoms (fatigue) can span wide: 1h–12h
#
# Tuple: (min_hours_before, max_hours_before)
# Food must have been eaten between min and max hours BEFORE the symptom.

SYMPTOM_WINDOWS: dict[str, tuple[float, float]] = {
    "Heartburn":      (0.25, 3.0),    # 15 min – 3 h
    "Acid Reflux":    (0.25, 3.0),    # 15 min – 3 h
    "Nausea":         (0.5,  4.0),    # 30 min – 4 h
    "Bloating":       (0.5,  8.0),    # 30 min – 8 h
    "Gas":            (1.0,  8.0),    # 1 h – 8 h
    "Cramps":         (0.5,  6.0),    # 30 min – 6 h
    "Abdominal Pain": (0.5,  8.0),    # 30 min – 8 h
    "Diarrhea":       (1.0, 16.0),    # 1 h – 16 h
    "Constipation":   (12.0,48.0),    # 12 h – 48 h
    "Fatigue":        (1.0, 12.0),    # 1 h – 12 h
}

# Symptom-specific nutrient risk factors
# These nutrients are clinically linked to each symptom
# Used to boost/reduce NLI score with domain knowledge
SYMPTOM_NUTRIENT_RISK: dict[str, dict[str, float]] = {
    "Heartburn":      {"total_fat": 0.30, "sat_fat": 0.25, "sodium": 0.20, "sugar": 0.15, "calories": 0.10},
    "Acid Reflux":    {"total_fat": 0.30, "sat_fat": 0.25, "sodium": 0.20, "sugar": 0.15, "calories": 0.10},
    "Bloating":       {"carbs": 0.30, "sugar": 0.25, "sodium": 0.20, "total_fat": 0.15, "cholesterol": 0.10},
    "Gas":            {"carbs": 0.35, "sugar": 0.30, "sodium": 0.15, "total_fat": 0.10, "cholesterol": 0.10},
    "Cramps":         {"total_fat": 0.25, "sat_fat": 0.20, "sodium": 0.25, "sugar": 0.20, "cholesterol": 0.10},
    "Abdominal Pain": {"total_fat": 0.25, "sat_fat": 0.20, "sodium": 0.20, "sugar": 0.20, "cholesterol": 0.15},
    "Nausea":         {"total_fat": 0.35, "sat_fat": 0.25, "cholesterol": 0.20, "sodium": 0.10, "calories": 0.10},
    "Diarrhea":       {"sugar": 0.30, "carbs": 0.25, "total_fat": 0.20, "sodium": 0.15, "cholesterol": 0.10},
    "Constipation":   {"total_fat": 0.30, "sat_fat": 0.25, "sodium": 0.20, "calories": 0.15, "cholesterol": 0.10},
    "Fatigue":        {"sugar": 0.35, "carbs": 0.30, "calories": 0.20, "total_fat": 0.10, "sodium": 0.05},
}

# Nutrient thresholds per 100g — above these values = meaningful risk
_NUTRIENT_RISK_THRESHOLDS: dict[str, float] = {
    "total_fat":   10.0,
    "sat_fat":     4.0,
    "sodium":      300.0,
    "sugar":       8.0,
    "carbs":       30.0,
    "cholesterol": 60.0,
    "calories":    250.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# USDA column aliases — reuse from nutrition_scorer
# ─────────────────────────────────────────────────────────────────────────────

_COL_ALIASES: dict[str, list[str]] = {
    "description": ["Description","description","DESCRIPTION","Long_Desc","NAME"],
    "calories":    ["Calories","calories","Energ_Kcal","ENERG_KCAL","Energy"],
    "protein":     ["Protein","protein","Protein_g","PROTEIN"],
    "total_fat":   ["TotalFat","Total_Fat","total_fat","Lipid_Tot","Fat","fat"],
    "carbs":       ["Carbohydrate","carbohydrate","Carbohydrt","Carbs","carbs"],
    "sodium":      ["Sodium","sodium","SODIUM","Sodium_mg"],
    "sat_fat":     ["SaturatedFat","Saturated_Fat","saturated_fat","FA_Sat"],
    "cholesterol": ["Cholesterol","cholesterol","CHOLESTEROL","Cholestrl"],
    "sugar":       ["Sugar","sugar","Sugar_Tot","SUGAR_TOT","Sugars"],
}

def _find_col(df: pd.DataFrame, key: str) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for alias in _COL_ALIASES.get(key, []):
        if alias in df.columns:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None

def _get_nutrient(row: pd.Series, df: pd.DataFrame, key: str, scale: float = 1.0) -> Optional[float]:
    col = _find_col(df, key)
    if col is None:
        return None
    try:
        v = float(row.get(col, float("nan")))
        return round(v * scale, 3) if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FoodLogEntry:
    """One row from the food_logs table."""
    user_id:    str
    usda_id:    int
    logged_at:  datetime
    quantity_g: float          # already converted to grams by caller


@dataclass
class SymptomLogEntry:
    """One row from the symptom_logs table."""
    user_id:      str
    symptom:      str
    logged_at:    datetime
    intensity:    str          # Mild | Moderate | Severe


@dataclass
class NutrientSnapshot:
    """Key nutrients for a food at its logged quantity."""
    description: str
    calories:    Optional[float] = None
    protein:     Optional[float] = None
    total_fat:   Optional[float] = None
    carbs:       Optional[float] = None
    sodium:      Optional[float] = None
    sat_fat:     Optional[float] = None
    cholesterol: Optional[float] = None
    sugar:       Optional[float] = None
    portion_g:   float = 100.0


@dataclass
class FoodCausationResult:
    """Causation prediction for one food → one symptom pair."""
    usda_id:           int
    food_name:         str
    quantity_g:        float
    logged_at:         str
    hours_before:      float            # hours before the symptom

    # NLI model output
    entailment_score:  float            # 0–1: model confidence that food caused symptom
    contradiction_score: float
    neutral_score:     float

    # Nutrient risk score for this symptom
    nutrient_risk_score: float          # 0–1

    # Temporal proximity score
    temporal_score:    float            # 0–1 (closer to window centre = higher)

    # Quantity weight
    quantity_score:    float            # 0–1

    # Final combined causation score
    causation_score:   float            # 0–1
    causation_label:   str              # "Likely", "Possible", "Unlikely"
    causation_pct:     str              # e.g. "78%"

    # Personalisation layer
    personal_prior:         float = 0.5   # user's learned probability for this pair
    prior_confidence:       float = 0.0   # how much data backs the prior
    prior_observations:     int   = 0
    prior_confirmations:    int   = 0
    personalisation_weight: float = 0.0   # how much personal data influenced score
    model_weight:           float = 1.0

    # Rich explanation
    explanation:       str = ""
    top_risk_nutrients:list[str] = field(default_factory=list)


@dataclass
class SymptomPrediction:
    """Full prediction result for one symptom entry."""
    symptom:          str
    intensity:        str
    symptom_logged_at:str
    digestion_window: str               # e.g. "0.25h – 3.0h before symptom"
    foods_in_window:  int               # how many foods were in the window
    foods_outside_window: int           # foods that were too early/late
    top_cause:        Optional[FoodCausationResult]    # most likely culprit
    all_candidates:   list[FoodCausationResult]        # ranked list
    no_foods_found:   bool = False
    note:             str = ""


# ─────────────────────────────────────────────────────────────────────────────
# NLI causation scorer
# ─────────────────────────────────────────────────────────────────────────────

def _build_premise(food_name: str, nutrients: NutrientSnapshot) -> str:
    """
    Build a rich food description as the NLI premise.
    The more nutrients we include, the better the model can reason.
    """
    parts = [f"The person ate {food_name}"]
    if nutrients.portion_g:
        parts.append(f"({nutrients.portion_g:.0f}g portion)")
    details = []
    if nutrients.calories    is not None: details.append(f"{nutrients.calories:.0f} kcal")
    if nutrients.total_fat   is not None: details.append(f"{nutrients.total_fat:.1f}g total fat")
    if nutrients.sat_fat     is not None: details.append(f"{nutrients.sat_fat:.1f}g saturated fat")
    if nutrients.sugar       is not None: details.append(f"{nutrients.sugar:.1f}g sugar")
    if nutrients.sodium      is not None: details.append(f"{nutrients.sodium:.0f}mg sodium")
    if nutrients.carbs       is not None: details.append(f"{nutrients.carbs:.1f}g carbohydrates")
    if nutrients.cholesterol is not None: details.append(f"{nutrients.cholesterol:.0f}mg cholesterol")
    if nutrients.protein     is not None: details.append(f"{nutrients.protein:.1f}g protein")
    if details:
        parts.append("containing " + ", ".join(details))
    return " ".join(parts) + "."


def _build_hypothesis(symptom: str, intensity: str) -> str:
    """
    Build the NLI hypothesis: "This food caused the symptom."
    Phrasing matters for NLI — entailment-style is best.
    """
    intensity_adv = {
        "Mild":     "mild",
        "Moderate": "moderate",
        "Severe":   "severe",
    }.get(intensity, "")

    symptom_phrases = {
        "Heartburn":      "caused heartburn and acid burning sensation",
        "Acid Reflux":    "triggered acid reflux and stomach acid backflow",
        "Bloating":       "caused abdominal bloating and distension",
        "Gas":            "triggered excessive intestinal gas and flatulence",
        "Nausea":         "caused nausea and stomach discomfort",
        "Cramps":         "caused intestinal cramps and spasms",
        "Abdominal Pain": "caused abdominal pain and gut discomfort",
        "Diarrhea":       "triggered diarrhea and loose stools",
        "Constipation":   "contributed to constipation and slow bowel transit",
        "Fatigue":        "caused post-meal fatigue and low energy due to poor digestion",
    }
    phrase = symptom_phrases.get(symptom, f"caused {symptom.lower()}")
    adv    = f"{intensity_adv} " if intensity_adv else ""
    return f"Eating this food {phrase} ({adv}intensity)."


def _nli_score(premise: str, hypothesis: str) -> dict[str, float]:
    """
    Run cross-encoder NLI on premise + hypothesis.
    Returns {"entailment": float, "neutral": float, "contradiction": float}
    """
    inputs = _tokenizer(
        premise,
        hypothesis,
        truncation      = True,
        max_length      = 512,
        return_tensors  = "pt",
        padding         = True,
    )
    with torch.no_grad():
        logits = _nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    # cross-encoder/nli-deberta-v3-small label order: contradiction=0, entailment=1, neutral=2
    return {
        "contradiction": round(probs[0], 4),
        "entailment":    round(probs[1], 4),
        "neutral":       round(probs[2], 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Nutrient risk scorer
# ─────────────────────────────────────────────────────────────────────────────

def _nutrient_risk(nutrients: NutrientSnapshot, symptom: str) -> tuple[float, list[str]]:
    """
    Score how risky this food's nutrients are for this specific symptom.
    Returns (0–1 score, list of top risk nutrient notes).
    """
    risk_weights = SYMPTOM_NUTRIENT_RISK.get(symptom, {})
    scale        = nutrients.portion_g / 100.0

    val_map: dict[str, Optional[float]] = {
        "total_fat":   nutrients.total_fat,
        "sat_fat":     nutrients.sat_fat,
        "sodium":      nutrients.sodium,
        "sugar":       nutrients.sugar,
        "carbs":       nutrients.carbs,
        "cholesterol": nutrients.cholesterol,
        "calories":    nutrients.calories,
    }

    total_risk     = 0.0
    total_weight   = sum(risk_weights.values()) or 1.0
    risk_notes: list[str] = []

    for nutrient, weight in risk_weights.items():
        val = val_map.get(nutrient)
        if val is None:
            continue
        threshold = _NUTRIENT_RISK_THRESHOLDS.get(nutrient, 1.0) * scale
        ratio     = min(val / threshold, 2.0)   # cap at 2× threshold
        contribution = ratio * weight
        total_risk  += contribution

        if ratio >= 1.0:
            unit = "mg" if nutrient in ("sodium", "cholesterol") else ("kcal" if nutrient == "calories" else "g")
            risk_notes.append(f"{nutrient.replace('_',' ').title()} {val:.1f}{unit} — {ratio:.1f}× above threshold for {symptom}")

    normalised = min(total_risk / total_weight, 1.0)
    return round(normalised, 4), risk_notes[:3]   # top 3 risk nutrients


# ─────────────────────────────────────────────────────────────────────────────
# Temporal proximity scorer
# ─────────────────────────────────────────────────────────────────────────────

def _temporal_score(hours_before: float, window_min: float, window_max: float) -> float:
    """
    Score temporal proximity within the digestion window.
    Foods eaten near the centre of the window get the highest score.
    Foods at the very edges of the window get lower scores.
    """
    if hours_before < window_min or hours_before > window_max:
        return 0.0
    window_span   = window_max - window_min
    window_centre = window_min + window_span / 2.0
    distance      = abs(hours_before - window_centre)
    max_distance  = window_span / 2.0
    return round(1.0 - (distance / max_distance) * 0.5, 4)   # 0.5–1.0 range


# ─────────────────────────────────────────────────────────────────────────────
# Quantity score
# ─────────────────────────────────────────────────────────────────────────────

def _quantity_score(portion_g: float) -> float:
    """
    Larger portions = higher causation likelihood.
    Sigmoid-like scaling: 50g=0.3, 100g=0.5, 200g=0.7, 400g=0.9
    """
    return round(min(portion_g / (portion_g + 150.0), 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Causation label
# ─────────────────────────────────────────────────────────────────────────────

def _causation_label(score: float) -> tuple[str, str]:
    if   score >= 0.70: return "Likely",   f"{int(score*100)}%"
    elif score >= 0.45: return "Possible", f"{int(score*100)}%"
    else:               return "Unlikely", f"{int(score*100)}%"


# ─────────────────────────────────────────────────────────────────────────────
# Explanation builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_explanation(
    food_name:      str,
    symptom:        str,
    hours_before:   float,
    nli_score:      float,
    nutrient_risk:  float,
    temporal:       float,
    quantity:       float,
    risk_nutrients: list[str],
    causation_label:str,
) -> str:
    lines = [
        f"{food_name} eaten {hours_before:.1f}h before {symptom} — causation assessment: {causation_label}.",
    ]
    lines.append(
        f"NLI model confidence: {nli_score:.0%} | "
        f"Nutrient risk: {nutrient_risk:.0%} | "
        f"Temporal fit: {temporal:.0%} | "
        f"Portion size: {quantity:.0%}"
    )
    if risk_nutrients:
        lines.append("Key risk nutrients: " + "; ".join(risk_nutrients[:2]) + ".")
    if causation_label == "Likely":
        lines.append(f"Strong evidence that this food contributed to your {symptom}.")
    elif causation_label == "Possible":
        lines.append(f"This food may have contributed to your {symptom} — log more entries to confirm.")
    else:
        lines.append(f"Weak evidence — {symptom} may have another cause.")
    return " ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public function
# ─────────────────────────────────────────────────────────────────────────────

def predict_food_symptom_causes(
    food_logs:    list[FoodLogEntry],
    symptom_logs: list[SymptomLogEntry],
    usda_df:      pd.DataFrame,
    id_col:       str,
    desc_col:     str,
    user_memory   = None,   # Optional[UserMemory] from user_symptom_memory.py
) -> list[SymptomPrediction]:
    """
    For each symptom in symptom_logs, find which food(s) in food_logs
    are most likely to have caused it.

    Parameters
    ----------
    food_logs    : food entries from DB (same user_id, recent history)
    symptom_logs : symptom entries from DB (same user_id)
    usda_df      : USDA DataFrame (already loaded in memory)
    id_col       : ID column name
    desc_col     : description column name
    user_memory  : Optional UserMemory — if provided, personalises the prediction
                   using the user's Bayesian prior for each (food, symptom) pair

    Returns
    -------
    list[SymptomPrediction] — one per symptom entry, ranked candidates inside
    """
    predictions: list[SymptomPrediction] = []

    for sym_entry in symptom_logs:
        symptom   = sym_entry.symptom
        intensity = sym_entry.intensity
        sym_time  = sym_entry.logged_at

        win_min, win_max = SYMPTOM_WINDOWS.get(symptom, (0.5, 8.0))

        # ── Stage 1: Temporal window filter ───────────────────────────────────
        candidates: list[tuple[FoodLogEntry, float]] = []   # (food, hours_before)
        outside_count = 0

        for food in food_logs:
            if food.user_id != sym_entry.user_id:
                continue
            if food.logged_at >= sym_time:
                continue   # food logged AFTER symptom — irrelevant
            hours_before = (sym_time - food.logged_at).total_seconds() / 3600.0
            if win_min <= hours_before <= win_max:
                candidates.append((food, hours_before))
            else:
                outside_count += 1

        if not candidates:
            predictions.append(SymptomPrediction(
                symptom           = symptom,
                intensity         = intensity,
                symptom_logged_at = sym_time.isoformat(),
                digestion_window  = f"{win_min}h – {win_max}h before symptom",
                foods_in_window   = 0,
                foods_outside_window = outside_count,
                top_cause         = None,
                all_candidates    = [],
                no_foods_found    = True,
                note = (
                    f"No food logs found within the {win_min}–{win_max}h digestion window "
                    f"before this {symptom}. {outside_count} food(s) were outside the window. "
                    f"Log more meals for accurate predictions."
                ),
            ))
            continue

        # ── Stage 2 + 3: NLI + nutrient risk + temporal + quantity scoring ───
        results: list[FoodCausationResult] = []

        for food_entry, hours_before in candidates:
            # Fetch USDA nutrients
            mask = usda_df[id_col].astype(str) == str(food_entry.usda_id)
            if not mask.any():
                food_name = f"USDA ID {food_entry.usda_id}"
                nutrients = NutrientSnapshot(
                    description = food_name,
                    portion_g   = food_entry.quantity_g,
                )
            else:
                row       = usda_df[mask].iloc[0]
                food_name = str(row.get(desc_col, f"USDA ID {food_entry.usda_id}"))
                scale     = food_entry.quantity_g / 100.0

                nutrients = NutrientSnapshot(
                    description = food_name,
                    calories    = _get_nutrient(row, usda_df, "calories",    scale),
                    protein     = _get_nutrient(row, usda_df, "protein",     scale),
                    total_fat   = _get_nutrient(row, usda_df, "total_fat",   scale),
                    carbs       = _get_nutrient(row, usda_df, "carbs",       scale),
                    sodium      = _get_nutrient(row, usda_df, "sodium",      scale),
                    sat_fat     = _get_nutrient(row, usda_df, "sat_fat",     scale),
                    cholesterol = _get_nutrient(row, usda_df, "cholesterol", scale),
                    sugar       = _get_nutrient(row, usda_df, "sugar",       scale),
                    portion_g   = food_entry.quantity_g,
                )

            # NLI causation
            premise    = _build_premise(food_name, nutrients)
            hypothesis = _build_hypothesis(symptom, intensity)
            nli_scores = _nli_score(premise, hypothesis)
            nli_ent    = nli_scores["entailment"]

            # Nutrient risk
            n_risk, risk_notes = _nutrient_risk(nutrients, symptom)

            # Temporal + quantity
            t_score = _temporal_score(hours_before, win_min, win_max)
            q_score = _quantity_score(food_entry.quantity_g)

            # ── Base causation score (generic, no personalisation) ────────
            # NLI entailment  35% — primary model signal
            # Nutrient risk   35% — domain knowledge anchor
            # Temporal fit    20% — timing within window
            # Quantity        10% — portion size weight
            base_causation = (
                0.35 * nli_ent +
                0.35 * n_risk  +
                0.20 * t_score +
                0.10 * q_score
            )
            base_causation = round(min(base_causation, 1.0), 4)

            # ── Personalisation layer ─────────────────────────────────────────
            personal_prior   = 0.5
            prior_conf       = 0.0
            prior_obs        = 0
            prior_conf_int   = 0
            p_weight         = 0.0
            m_weight         = 1.0

            if user_memory is not None:
                blended, breakdown = user_memory.blend_score(
                    usda_id   = food_entry.usda_id,
                    symptom   = symptom,
                    nli_score = nli_ent,
                )
                # Replace the NLI component with the personalised blend
                causation = (
                    0.35 * blended +   # personalised NLI+prior replaces raw NLI
                    0.35 * n_risk   +
                    0.20 * t_score  +
                    0.10 * q_score
                )
                personal_prior = breakdown["personal_prior"]
                prior_conf     = breakdown["prior_confidence"]
                prior_obs      = breakdown["prior_observations"]
                prior_conf_int = breakdown["prior_confirmations"]
                p_weight       = breakdown["personalisation_weight"]
                m_weight       = breakdown["model_weight"]
            else:
                causation = base_causation

            causation = round(min(causation, 1.0), 4)
            c_label, c_pct = _causation_label(causation)

            results.append(FoodCausationResult(
                usda_id                 = food_entry.usda_id,
                food_name               = food_name,
                quantity_g              = food_entry.quantity_g,
                logged_at               = food_entry.logged_at.isoformat(),
                hours_before            = round(hours_before, 2),
                entailment_score        = nli_ent,
                contradiction_score     = nli_scores["contradiction"],
                neutral_score           = nli_scores["neutral"],
                nutrient_risk_score     = n_risk,
                temporal_score          = t_score,
                quantity_score          = q_score,
                causation_score         = causation,
                causation_label         = c_label,
                causation_pct           = c_pct,
                personal_prior          = personal_prior,
                prior_confidence        = prior_conf,
                prior_observations      = prior_obs,
                prior_confirmations     = prior_conf_int,
                personalisation_weight  = p_weight,
                model_weight            = m_weight,
                top_risk_nutrients      = risk_notes,
                explanation             = _build_explanation(
                    food_name, symptom, hours_before,
                    nli_ent, n_risk, t_score, q_score,
                    risk_notes, c_label,
                ),
            ))

        # Sort by causation score descending
        results.sort(key=lambda r: r.causation_score, reverse=True)

        predictions.append(SymptomPrediction(
            symptom              = symptom,
            intensity            = intensity,
            symptom_logged_at    = sym_time.isoformat(),
            digestion_window     = f"{win_min}h – {win_max}h before symptom",
            foods_in_window      = len(results),
            foods_outside_window = outside_count,
            top_cause            = results[0] if results else None,
            all_candidates       = results,
        ))

    return predictions