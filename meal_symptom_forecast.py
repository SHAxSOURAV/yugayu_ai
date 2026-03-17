"""
meal_symptom_forecast.py
────────────────────────
Predicts which gut symptoms a user might experience if they eat a hypothetical
meal — BEFORE they eat it.

━━━ HOW IT DIFFERS FROM food_symptom_predictor.py ━━━

  food_symptom_predictor.py  — POST HOC: user already ate and felt a symptom.
                               "Which food I already ate most likely caused this?"
                               Uses temporal window + already-known symptom.

  meal_symptom_forecast.py   — PRE HOC: user hasn't eaten yet.
                               "If I eat this meal, what symptoms might I get?"
                               No temporal signal (food not yet eaten).
                               Scores all 10 symptoms independently.

━━━ SCORING APPROACH ━━━

For each food in the proposed meal × each of the 10 symptom labels:

  Stage 1 — NLI causation (cross-encoder/nli-deberta-v3-small)
    Premise:    "The person ate [food] containing [nutrients] ([Xg portion])"
    Hypothesis: "Eating this food caused [symptom] (moderate intensity)"
    → entailment probability (0–1)

  Stage 2 — Nutrient risk (domain knowledge from food_symptom_predictor)
    Nutrient values are looked up from USDA and scored against
    symptom-specific risk thresholds (e.g. fat+sodium → Heartburn).

  Stage 3 — Quantity weight
    Larger portion = higher risk weight.

  Per-food score per symptom:
    score = 0.45 × NLI entailment + 0.45 × nutrient_risk + 0.10 × quantity_weight

  Stage 4 — Personalisation (Bayesian prior from user history)
    If the user has logged food + symptoms before, their personal
    prior is blended in via the same mechanism as food_symptom_predictor.
    A user with 30+ logs has 75% weight on their personal prior.

  Stage 5 — Multi-food meal aggregation per symptom
    score_symptom = weighted_average(food_scores, weights=quantity_g)
    If any single food scores high, the symptom surface risk is elevated.
    Final score = max(weighted_avg, 0.85 × max_single_food_score)

━━━ HISTORY USAGE ━━━

  Up to 400 food logs + 400 symptom logs are pulled from MongoDB.
  These are used ONLY to personalise predictions via the Bayesian prior
  (UserMemory). The raw logs themselves are not used in the forecast score.

Public function:
  forecast_meal_symptoms(proposed_foods, user_memory, usda_df, id_col, desc_col)
    → list[MealSymptomForecast]
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# Reuse NLI infrastructure from food_symptom_predictor — same models, no extra RAM
from food_symptom_predictor import (
    _build_premise,
    _build_hypothesis,
    _nli_score,
    _nutrient_risk,
    _quantity_score,
    NutrientSnapshot,
    SYMPTOM_WINDOWS,
    SYMPTOM_NUTRIENT_RISK,
    _find_col,
    _get_nutrient,
)

log = logging.getLogger(__name__)


# ── All 10 symptoms — must match _VALID_SYMPTOMS in main.py ──────────────────
ALL_SYMPTOMS = list(SYMPTOM_WINDOWS.keys())   # consistent ordering

# Fixed intensity for forecast — "Moderate" is the neutral/median assumption
_FORECAST_INTENSITY = "Moderate"

# Risk label thresholds
_RISK_THRESHOLDS: list[tuple[str, float]] = [
    ("High",   0.60),
    ("Medium", 0.38),
    ("Low",    0.0),
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ProposedFoodItem:
    """One food in the hypothetical meal."""
    usda_id:    int
    quantity_g: float          # portion in grams


@dataclass
class FoodSymptomScore:
    """
    NLI + nutrient risk score for one food → one symptom pair.
    Used internally to build the per-symptom aggregate.
    """
    usda_id:             int
    food_name:           str
    quantity_g:          float
    nli_entailment:      float
    nutrient_risk_score: float
    quantity_weight:     float
    base_score:          float           # 0.45×NLI + 0.45×nutrient + 0.10×qty
    personalised_score:  float           # after Bayesian prior blend
    personal_prior:      float
    prior_confidence:    float
    prior_observations:  int
    top_risk_nutrients:  list[str]


@dataclass
class MealSymptomForecast:
    """
    Final predicted risk for one symptom across the full proposed meal.
    """
    symptom:              str
    risk_score:           float          # 0–1 aggregated across all foods
    risk_level:           str            # "High" | "Medium" | "Low"
    risk_pct:             str            # e.g. "72%"
    top_trigger_food:     str            # food name with highest contribution
    top_trigger_usda_id:  int
    top_trigger_score:    float          # that food's individual score
    top_risk_nutrients:   list[str]      # top risk nutrients from top trigger
    per_food_scores:      list[FoodSymptomScore]   # detailed breakdown
    personalised:         bool
    explanation:          str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk_label(score: float) -> str:
    for level, threshold in _RISK_THRESHOLDS:
        if score >= threshold:
            return level
    return "Low"


def _fetch_nutrients(
    usda_id:    int,
    quantity_g: float,
    usda_df:    pd.DataFrame,
    id_col:     str,
    desc_col:   str,
) -> tuple[str, NutrientSnapshot]:
    """
    Look up USDA nutrients for a food, scaled to the given portion.
    Returns (food_name, NutrientSnapshot).
    """
    mask = usda_df[id_col].astype(str) == str(usda_id)
    if not mask.any():
        name = f"USDA ID {usda_id}"
        return name, NutrientSnapshot(description=name, portion_g=quantity_g)

    row       = usda_df[mask].iloc[0]
    food_name = str(row.get(desc_col, f"USDA ID {usda_id}"))
    scale     = quantity_g / 100.0

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
        portion_g   = quantity_g,
    )
    return food_name, nutrients


def _score_food_for_symptom(
    food_name:   str,
    nutrients:   NutrientSnapshot,
    usda_id:     int,
    quantity_g:  float,
    symptom:     str,
    user_memory,                    # Optional UserMemory
) -> FoodSymptomScore:
    """
    Score one food against one symptom using NLI + nutrient risk + quantity.
    Optionally blends with the user's Bayesian prior if memory is available.
    """
    # NLI — uses "Moderate" intensity as neutral baseline for uneaten food
    premise    = _build_premise(food_name, nutrients)
    hypothesis = _build_hypothesis(symptom, _FORECAST_INTENSITY)
    nli_scores = _nli_score(premise, hypothesis)
    nli_ent    = nli_scores["entailment"]

    # Nutrient risk
    n_risk, risk_notes = _nutrient_risk(nutrients, symptom)

    # Quantity weight
    q_weight = _quantity_score(quantity_g)

    # Base score — no temporal component (uneaten food has no time signal)
    # Redistribute temporal 20% equally to NLI and nutrient risk
    base = round(min(0.45 * nli_ent + 0.45 * n_risk + 0.10 * q_weight, 1.0), 4)

    # Personalisation
    personalised_score = base
    personal_prior     = 0.5
    prior_conf         = 0.0
    prior_obs          = 0

    if user_memory is not None:
        blended, breakdown = user_memory.blend_score(
            usda_id   = usda_id,
            symptom   = symptom,
            nli_score = nli_ent,
        )
        # Replace NLI component with personalised blend
        personalised_score = round(
            min(0.45 * blended + 0.45 * n_risk + 0.10 * q_weight, 1.0), 4
        )
        personal_prior = breakdown["personal_prior"]
        prior_conf     = breakdown["prior_confidence"]
        prior_obs      = breakdown["prior_observations"]

    return FoodSymptomScore(
        usda_id             = usda_id,
        food_name           = food_name,
        quantity_g          = quantity_g,
        nli_entailment      = nli_ent,
        nutrient_risk_score = n_risk,
        quantity_weight     = q_weight,
        base_score          = base,
        personalised_score  = personalised_score,
        personal_prior      = personal_prior,
        prior_confidence    = prior_conf,
        prior_observations  = prior_obs,
        top_risk_nutrients  = risk_notes,
    )


def _aggregate_food_scores(
    food_scores: list[FoodSymptomScore],
) -> tuple[float, FoodSymptomScore]:
    """
    Aggregate per-food scores for a single symptom across the full meal.

    Strategy:
    - Weighted average by quantity_g (larger portions have more influence)
    - Also compute max single-food score
    - Final = max(weighted_avg, 0.85 × max_score)
      This ensures one high-risk food can elevate the overall meal risk.
    """
    if not food_scores:
        return 0.0, None

    total_qty = sum(f.quantity_g for f in food_scores)
    if total_qty == 0:
        total_qty = 1.0

    weighted_avg = sum(
        f.personalised_score * (f.quantity_g / total_qty)
        for f in food_scores
    )

    max_score = max(f.personalised_score for f in food_scores)
    top_food  = max(food_scores, key=lambda f: f.personalised_score)

    # Blend weighted average with single-food max
    final = max(weighted_avg, 0.85 * max_score)
    return round(min(final, 1.0), 4), top_food


def _build_forecast_explanation(
    symptom:       str,
    risk_score:    float,
    risk_level:    str,
    top_food:      str,
    top_score:     float,
    risk_nutrients:list[str],
    personalised:  bool,
) -> str:
    lines = []
    if risk_level == "High":
        lines.append(
            f"This meal carries a HIGH risk of causing {symptom} ({int(risk_score*100)}%). "
            f"The main concern is {top_food} (individual risk {int(top_score*100)}%)."
        )
    elif risk_level == "Medium":
        lines.append(
            f"This meal has a MODERATE risk of triggering {symptom} ({int(risk_score*100)}%). "
            f"{top_food} is the most likely contributor ({int(top_score*100)}%)."
        )
    else:
        lines.append(
            f"Low risk of {symptom} from this meal ({int(risk_score*100)}%). "
            "No strong gut-irritating factors detected."
        )
    if risk_nutrients:
        lines.append("Key risk factors: " + "; ".join(risk_nutrients[:2]) + ".")
    if personalised:
        lines.append("Score includes your personal gut sensitivity history.")
    else:
        lines.append("Log more meals and symptoms to personalise this forecast.")
    return " ".join(lines)


# ── Public function ───────────────────────────────────────────────────────────

def forecast_meal_symptoms(
    proposed_foods: list[ProposedFoodItem],
    usda_df:        pd.DataFrame,
    id_col:         str,
    desc_col:       str,
    user_memory     = None,   # Optional UserMemory from user_symptom_memory.py
) -> list[MealSymptomForecast]:
    """
    Predict which gut symptoms a user is at risk of after eating a proposed meal.

    Parameters
    ----------
    proposed_foods : list of ProposedFoodItem (usda_id + quantity_g)
    usda_df        : USDA DataFrame loaded at startup
    id_col         : USDA ID column name
    desc_col       : USDA description column name
    user_memory    : Optional UserMemory — if provided, blends personal priors
                     from the user's food+symptom log history into the score

    Returns
    -------
    list[MealSymptomForecast] — one per symptom, sorted by risk_score descending
    """
    if not proposed_foods:
        return []

    is_personalised = (
        user_memory is not None and
        getattr(user_memory, "total_food_logs", 0) > 0
    )

    # ── Pre-fetch nutrients for all foods once (avoid repeated DF lookups) ────
    food_nutrient_cache: dict[int, tuple[str, NutrientSnapshot]] = {}
    for item in proposed_foods:
        if item.usda_id not in food_nutrient_cache:
            food_nutrient_cache[item.usda_id] = _fetch_nutrients(
                usda_id    = item.usda_id,
                quantity_g = item.quantity_g,
                usda_df    = usda_df,
                id_col     = id_col,
                desc_col   = desc_col,
            )

    # ── Score all foods × all symptoms ───────────────────────────────────────
    # Structure: {symptom: [FoodSymptomScore, ...]}
    symptom_food_scores: dict[str, list[FoodSymptomScore]] = {s: [] for s in ALL_SYMPTOMS}

    for item in proposed_foods:
        food_name, nutrients = food_nutrient_cache[item.usda_id]
        # Rebuild nutrients with THIS item's quantity (cache may be for different qty)
        _, nutrients = _fetch_nutrients(
            usda_id    = item.usda_id,
            quantity_g = item.quantity_g,
            usda_df    = usda_df,
            id_col     = id_col,
            desc_col   = desc_col,
        )

        for symptom in ALL_SYMPTOMS:
            score = _score_food_for_symptom(
                food_name   = food_name,
                nutrients   = nutrients,
                usda_id     = item.usda_id,
                quantity_g  = item.quantity_g,
                symptom     = symptom,
                user_memory = user_memory,
            )
            symptom_food_scores[symptom].append(score)
            log.debug(f"  {food_name} → {symptom}: {score.personalised_score:.3f}")

    # ── Aggregate per symptom and build output ────────────────────────────────
    forecasts: list[MealSymptomForecast] = []

    for symptom in ALL_SYMPTOMS:
        food_scores = symptom_food_scores[symptom]
        agg_score, top_food = _aggregate_food_scores(food_scores)

        if top_food is None:
            continue

        risk_level = _risk_label(agg_score)
        risk_pct   = f"{int(agg_score * 100)}%"

        forecasts.append(MealSymptomForecast(
            symptom             = symptom,
            risk_score          = agg_score,
            risk_level          = risk_level,
            risk_pct            = risk_pct,
            top_trigger_food    = top_food.food_name,
            top_trigger_usda_id = top_food.usda_id,
            top_trigger_score   = top_food.personalised_score,
            top_risk_nutrients  = top_food.top_risk_nutrients,
            per_food_scores     = food_scores,
            personalised        = is_personalised,
            explanation         = _build_forecast_explanation(
                symptom        = symptom,
                risk_score     = agg_score,
                risk_level     = risk_level,
                top_food       = top_food.food_name,
                top_score      = top_food.personalised_score,
                risk_nutrients = top_food.top_risk_nutrients,
                personalised   = is_personalised,
            ),
        ))

    # Sort by risk_score descending
    forecasts.sort(key=lambda f: f.risk_score, reverse=True)
    return forecasts