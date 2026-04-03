"""
meal_symptom_forecast.py
────────────────────────
Predicts which gut symptoms a user might experience if they eat a hypothetical
meal — BEFORE they eat it.

All scoring logic unchanged. Imports updated to match new food_symptom_predictor
(which now uses Claude API for NLI instead of cross-encoder).

Public function:
    forecast_meal_symptoms(proposed_foods, usda_client, user_memory)
        → list[MealSymptomForecast]
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

# Reuse from food_symptom_predictor — NLI now uses Claude API (no extra cost)
from food_symptom_predictor import (
    _build_premise,
    _build_hypothesis,
    _nli_score,
    _nutrient_risk,
    _quantity_score,
    NutrientSnapshot,
    SYMPTOM_WINDOWS,
    SYMPTOM_NUTRIENT_RISK,
)

log = logging.getLogger(__name__)

_usda_client = None


def init(usda_client) -> None:
    global _usda_client
    _usda_client = usda_client
    log.info("meal_symptom_forecast: USDA client ready.")


ALL_SYMPTOMS = list(SYMPTOM_WINDOWS.keys())
_FORECAST_INTENSITY = "Moderate"

_RISK_THRESHOLDS: list[tuple[str, float]] = [
    ("High",   0.60),
    ("Medium", 0.38),
    ("Low",    0.0),
]


# ── Data classes (unchanged) ──────────────────────────────────────────────────

@dataclass
class ProposedFoodItem:
    usda_id: int; quantity_g: float


@dataclass
class FoodSymptomScore:
    usda_id: int; food_name: str; quantity_g: float
    nli_entailment: float; nutrient_risk_score: float; quantity_weight: float
    base_score: float; personalised_score: float; personal_prior: float
    prior_confidence: float; prior_observations: int; top_risk_nutrients: list[str]


@dataclass
class MealSymptomForecast:
    symptom: str; risk_score: float; risk_level: str; risk_pct: str
    top_trigger_food: str; top_trigger_usda_id: int; top_trigger_score: float
    top_risk_nutrients: list[str]; per_food_scores: list[FoodSymptomScore]
    personalised: bool; explanation: str


# ── Helpers (unchanged) ───────────────────────────────────────────────────────

def _risk_label(score: float) -> str:
    for level, threshold in _RISK_THRESHOLDS:
        if score >= threshold:
            return level
    return "Low"


def _fetch_nutrients(usda_id: int, quantity_g: float) -> tuple[str, NutrientSnapshot]:
    if _usda_client is None:
        name = f"USDA ID {usda_id}"
        return name, NutrientSnapshot(description=name, portion_g=quantity_g)

    data = _usda_client.get_nutrients(usda_id)
    if data is None:
        name = f"USDA ID {usda_id}"
        return name, NutrientSnapshot(description=name, portion_g=quantity_g)

    food_name = data.get("description", f"USDA ID {usda_id}")
    scale     = quantity_g / 100.0
    def sv(k): v = data.get(k); return round(float(v)*scale, 3) if v is not None else None
    nutrients = NutrientSnapshot(
        description=food_name,
        calories=sv("calories"), protein=sv("protein"), total_fat=sv("total_fat"),
        carbs=sv("carbs"), sodium=sv("sodium"), sat_fat=sv("sat_fat"),
        cholesterol=sv("cholesterol"), sugar=sv("sugar"), portion_g=quantity_g,
    )
    return food_name, nutrients


def _score_food_for_symptom(
    food_name: str, nutrients: NutrientSnapshot, usda_id: int,
    quantity_g: float, symptom: str, user_memory,
) -> FoodSymptomScore:
    premise    = _build_premise(food_name, nutrients)
    hypothesis = _build_hypothesis(symptom, _FORECAST_INTENSITY)
    nli_scores = _nli_score(premise, hypothesis)
    nli_ent    = nli_scores["entailment"]

    n_risk, risk_notes = _nutrient_risk(nutrients, symptom)
    q_weight           = _quantity_score(quantity_g)

    base = round(min(0.45 * nli_ent + 0.45 * n_risk + 0.10 * q_weight, 1.0), 4)

    personalised_score = base
    personal_prior     = 0.5
    prior_conf         = 0.0
    prior_obs          = 0

    if user_memory is not None:
        blended, breakdown = user_memory.blend_score(
            usda_id=usda_id, symptom=symptom, nli_score=nli_ent,
        )
        personalised_score = round(
            min(0.45 * blended + 0.45 * n_risk + 0.10 * q_weight, 1.0), 4,
        )
        personal_prior = breakdown["personal_prior"]
        prior_conf     = breakdown["prior_confidence"]
        prior_obs      = breakdown["prior_observations"]

    return FoodSymptomScore(
        usda_id=usda_id, food_name=food_name, quantity_g=quantity_g,
        nli_entailment=nli_ent, nutrient_risk_score=n_risk, quantity_weight=q_weight,
        base_score=base, personalised_score=personalised_score,
        personal_prior=personal_prior, prior_confidence=prior_conf,
        prior_observations=prior_obs, top_risk_nutrients=risk_notes,
    )


def _aggregate_food_scores(food_scores: list[FoodSymptomScore]) -> tuple[float, Optional[FoodSymptomScore]]:
    if not food_scores:
        return 0.0, None
    total_qty = sum(f.quantity_g for f in food_scores) or 1.0
    weighted_avg = sum(f.personalised_score * (f.quantity_g / total_qty) for f in food_scores)
    max_score    = max(f.personalised_score for f in food_scores)
    top_food     = max(food_scores, key=lambda f: f.personalised_score)
    final        = max(weighted_avg, 0.85 * max_score)
    return round(min(final, 1.0), 4), top_food


# ── Public function ───────────────────────────────────────────────────────────

def forecast_meal_symptoms(
    proposed_foods: list[ProposedFoodItem],
    user_memory     = None,
    # Legacy params accepted but ignored (usda_df, id_col, desc_col)
    **_kwargs,
) -> list[MealSymptomForecast]:

    if not proposed_foods:
        return []

    is_personalised = (
        user_memory is not None and
        getattr(user_memory, "total_food_logs", 0) > 0
    )

    # Fetch nutrients once per unique food
    nutrient_cache: dict[int, tuple[str, NutrientSnapshot]] = {}
    for item in proposed_foods:
        if item.usda_id not in nutrient_cache:
            nutrient_cache[item.usda_id] = _fetch_nutrients(item.usda_id, item.quantity_g)

    # Score all foods × all symptoms
    symptom_food_scores: dict[str, list[FoodSymptomScore]] = {s: [] for s in ALL_SYMPTOMS}

    for item in proposed_foods:
        food_name, _ = nutrient_cache[item.usda_id]
        _, nutrients = _fetch_nutrients(item.usda_id, item.quantity_g)

        for symptom in ALL_SYMPTOMS:
            score = _score_food_for_symptom(
                food_name=food_name, nutrients=nutrients, usda_id=item.usda_id,
                quantity_g=item.quantity_g, symptom=symptom, user_memory=user_memory,
            )
            symptom_food_scores[symptom].append(score)

    # Aggregate and build output
    forecasts: list[MealSymptomForecast] = []

    for symptom in ALL_SYMPTOMS:
        agg_score, top_food = _aggregate_food_scores(symptom_food_scores[symptom])
        if top_food is None:
            continue

        risk_level = _risk_label(agg_score)

        explanation_parts = []
        if risk_level == "High":
            explanation_parts.append(
                f"HIGH risk of {symptom} ({int(agg_score*100)}%). "
                f"Main concern: {top_food.food_name} (individual {int(top_food.personalised_score*100)}%)."
            )
        elif risk_level == "Medium":
            explanation_parts.append(
                f"MODERATE risk of {symptom} ({int(agg_score*100)}%). "
                f"{top_food.food_name} is the likely contributor."
            )
        else:
            explanation_parts.append(f"Low risk of {symptom} ({int(agg_score*100)}%).")

        if top_food.top_risk_nutrients:
            explanation_parts.append("Risk factors: " + "; ".join(top_food.top_risk_nutrients[:2]) + ".")
        if is_personalised:
            explanation_parts.append("Score includes your personal gut sensitivity history.")

        forecasts.append(MealSymptomForecast(
            symptom=symptom, risk_score=agg_score, risk_level=risk_level,
            risk_pct=f"{int(agg_score*100)}%",
            top_trigger_food=top_food.food_name, top_trigger_usda_id=top_food.usda_id,
            top_trigger_score=top_food.personalised_score,
            top_risk_nutrients=top_food.top_risk_nutrients,
            per_food_scores=symptom_food_scores[symptom],
            personalised=is_personalised,
            explanation=" ".join(explanation_parts),
        ))

    forecasts.sort(key=lambda f: f.risk_score, reverse=True)
    return forecasts
