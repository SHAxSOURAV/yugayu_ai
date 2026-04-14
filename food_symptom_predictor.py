"""
food_symptom_predictor.py
─────────────────────────
Logic-based food → symptom causation predictor.
Zero Claude/ML API calls.  Zero additional cost.

Replaces Claude NLI with three deterministic layers:
  1. Temporal window matching   — clinical digestion windows per symptom
  2. Nutrient-risk scoring      — USDA nutrients vs per-symptom thresholds
  3. Food descriptor heuristics — keyword lookup (fried/spicy/dairy/…)

All public symbols are identical to the Claude version so no caller changes.

Public API (unchanged):
    init(claude_client, usda_client)
    predict_food_symptom_causes(food_logs, symptom_logs, user_memory) -> list[SymptomPrediction]
    predict_causation_by_time(food_logs_named, symptom_logs) -> dict[str, list[str]]
    group_composite_meals(food_logs_named)                   -> list[dict]
    batch_nli_score(pairs)                                   -> list[dict]

    # Used by meal_symptom_forecast
    NutrientSnapshot, SYMPTOM_WINDOWS, SYMPTOM_NUTRIENT_RISK
    _build_premise, _build_hypothesis, _nli_score
    _nutrient_risk, _quantity_score

    # Used by food_recommender
    _DIGESTION_WINDOWS_H, _DEFAULT_WINDOW_H, _parse_dt, _build_timeline_prompt
"""

from __future__ import annotations

import json
import logging
import re as _re
from collections import defaultdict as _defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# Claude client accepted for API compatibility but never used
_claude_client = None
_usda_client   = None


def init(claude_client, usda_client) -> None:
    global _claude_client, _usda_client
    _claude_client = claude_client   # kept for signature compat — not called
    _usda_client   = usda_client
    log.info("food_symptom_predictor: logic-based engine ready (no Claude API).")


# ─────────────────────────────────────────────────────────────────────────────
# Clinical digestion windows  (hours)
# ─────────────────────────────────────────────────────────────────────────────

SYMPTOM_WINDOWS: dict[str, tuple[float, float]] = {
    "Heartburn":      (0.25,  3.0),
    "Acid Reflux":    (0.25,  3.0),
    "Nausea":         (0.5,   4.0),
    "Bloating":       (0.5,   8.0),
    "Gas":            (1.0,   8.0),
    "Cramps":         (0.5,   6.0),
    "Abdominal Pain": (0.5,   8.0),
    "Diarrhea":       (1.0,  16.0),
    "Constipation":   (12.0, 48.0),
    "Fatigue":        (1.0,  12.0),
}

# Alias used by food_recommender
_DIGESTION_WINDOWS_H = SYMPTOM_WINDOWS
_DEFAULT_WINDOW_H    = (0.5, 8.0)


# ─────────────────────────────────────────────────────────────────────────────
# Nutrient-risk configuration
# ─────────────────────────────────────────────────────────────────────────────

# Per-symptom nutrient weights (must sum to 1.0 per symptom)
SYMPTOM_NUTRIENT_RISK: dict[str, dict[str, float]] = {
    "Heartburn":      {"total_fat": 0.30, "sat_fat": 0.25, "sodium": 0.20, "sugar": 0.15, "calories": 0.10},
    "Acid Reflux":    {"total_fat": 0.30, "sat_fat": 0.25, "sodium": 0.20, "sugar": 0.15, "calories": 0.10},
    "Bloating":       {"carbs":     0.30, "sugar":   0.25, "sodium": 0.20, "total_fat": 0.15, "cholesterol": 0.10},
    "Gas":            {"carbs":     0.35, "sugar":   0.30, "sodium": 0.15, "total_fat": 0.10, "cholesterol": 0.10},
    "Cramps":         {"total_fat": 0.25, "sat_fat": 0.20, "sodium": 0.25, "sugar": 0.20, "cholesterol": 0.10},
    "Abdominal Pain": {"total_fat": 0.25, "sat_fat": 0.20, "sodium": 0.20, "sugar": 0.20, "cholesterol": 0.15},
    "Nausea":         {"total_fat": 0.35, "sat_fat": 0.25, "cholesterol": 0.20, "sodium": 0.10, "calories": 0.10},
    "Diarrhea":       {"sugar":     0.30, "carbs":   0.25, "total_fat": 0.20, "sodium": 0.15, "cholesterol": 0.10},
    "Constipation":   {"total_fat": 0.30, "sat_fat": 0.25, "sodium": 0.20, "calories": 0.15, "cholesterol": 0.10},
    "Fatigue":        {"sugar":     0.35, "carbs":   0.30, "calories": 0.20, "total_fat": 0.10, "sodium": 0.05},
}

# Nutrient danger thresholds per 100 g baseline (scaled by portion)
_NUTRIENT_RISK_THRESHOLDS: dict[str, float] = {
    "total_fat":   10.0,   # g
    "sat_fat":      4.0,   # g
    "sodium":     300.0,   # mg
    "sugar":        8.0,   # g
    "carbs":       30.0,   # g
    "cholesterol": 60.0,   # mg
    "calories":   250.0,   # kcal
}


# ─────────────────────────────────────────────────────────────────────────────
# Food-descriptor keyword heuristics
# (used when USDA nutrients are unavailable or as a supplementary signal)
# ─────────────────────────────────────────────────────────────────────────────

# keyword → {symptom: risk_score (0-1)}
_FOOD_RISK_KEYWORDS: dict[str, dict[str, float]] = {
    # cooking styles
    "fried":      {"Heartburn": 0.80, "Acid Reflux": 0.80, "Nausea": 0.70, "Bloating": 0.60, "Abdominal Pain": 0.60},
    "deep fried": {"Heartburn": 0.85, "Acid Reflux": 0.85, "Nausea": 0.75, "Bloating": 0.65},
    "greasy":     {"Heartburn": 0.75, "Nausea": 0.70, "Abdominal Pain": 0.60},
    "fatty":      {"Heartburn": 0.65, "Nausea": 0.65, "Bloating": 0.50, "Constipation": 0.50},
    "spicy":      {"Heartburn": 0.85, "Acid Reflux": 0.85, "Abdominal Pain": 0.80, "Diarrhea": 0.70, "Cramps": 0.65},
    # dairy
    "dairy":      {"Bloating": 0.70, "Gas": 0.70, "Diarrhea": 0.60, "Cramps": 0.55},
    "milk":       {"Bloating": 0.70, "Gas": 0.65, "Diarrhea": 0.55, "Cramps": 0.50},
    "cheese":     {"Bloating": 0.60, "Gas": 0.50, "Heartburn": 0.55, "Constipation": 0.45},
    "cream":      {"Heartburn": 0.65, "Acid Reflux": 0.60, "Bloating": 0.55},
    "ice cream":  {"Bloating": 0.65, "Gas": 0.55, "Diarrhea": 0.50},
    "yogurt":     {"Bloating": 0.30},          # generally gut-friendly but can bloat
    # high-sugar / processed
    "sugar":      {"Bloating": 0.55, "Diarrhea": 0.60, "Gas": 0.50, "Fatigue": 0.55},
    "sweet":      {"Bloating": 0.50, "Diarrhea": 0.50, "Fatigue": 0.45},
    "chocolate":  {"Heartburn": 0.70, "Acid Reflux": 0.65, "Diarrhea": 0.45},
    "candy":      {"Diarrhea": 0.55, "Bloating": 0.50, "Fatigue": 0.50},
    "soda":       {"Bloating": 0.70, "Gas": 0.75, "Heartburn": 0.60, "Acid Reflux": 0.60},
    "carbonated": {"Bloating": 0.70, "Gas": 0.75},
    # caffeine / alcohol
    "coffee":     {"Heartburn": 0.75, "Acid Reflux": 0.70, "Diarrhea": 0.55},
    "caffeine":   {"Heartburn": 0.70, "Acid Reflux": 0.70, "Diarrhea": 0.50},
    "alcohol":    {"Heartburn": 0.80, "Acid Reflux": 0.80, "Diarrhea": 0.70, "Nausea": 0.65},
    # high-fibre / fermentable
    "bean":       {"Gas": 0.85, "Bloating": 0.80, "Cramps": 0.55},
    "beans":      {"Gas": 0.85, "Bloating": 0.80, "Cramps": 0.55},
    "lentil":     {"Gas": 0.75, "Bloating": 0.70},
    "legume":     {"Gas": 0.80, "Bloating": 0.75},
    "cabbage":    {"Gas": 0.75, "Bloating": 0.65},
    "broccoli":   {"Gas": 0.65, "Bloating": 0.55},
    "onion":      {"Gas": 0.70, "Bloating": 0.65, "Heartburn": 0.55},
    "garlic":     {"Gas": 0.55, "Heartburn": 0.50},
    "cauliflower":{"Gas": 0.65, "Bloating": 0.60},
    # processed / fast food
    "processed":  {"Bloating": 0.60, "Gas": 0.55, "Abdominal Pain": 0.55, "Fatigue": 0.50},
    "fast food":  {"Heartburn": 0.70, "Bloating": 0.65, "Nausea": 0.55, "Fatigue": 0.50},
    "burger":     {"Heartburn": 0.65, "Bloating": 0.55, "Nausea": 0.50},
    "pizza":      {"Heartburn": 0.65, "Bloating": 0.60, "Gas": 0.50},
    # gluten / wheat
    "wheat":      {"Bloating": 0.55, "Gas": 0.50, "Abdominal Pain": 0.45},
    "gluten":     {"Bloating": 0.70, "Abdominal Pain": 0.65, "Diarrhea": 0.55},
    "bread":      {"Bloating": 0.50, "Gas": 0.45},
    # gut-safe foods (low risk → low entailment is appropriate)
    "banana":     {},
    "oat":        {},
    "ginger":     {},
    "rice":       {},
    "chicken":    {},
    "turkey":     {},
    "salmon":     {},
    "carrot":     {},
    "sweet potato": {},
    "apple":      {},
    "blueberry":  {},
}


def _keyword_risk_score(food_name: str, symptom: str) -> float:
    """
    Returns a 0-1 risk score for a food name → symptom pair
    based purely on keyword matching.  0.0 means no keyword matched.

    A safe-food keyword (empty dict, e.g. "sweet potato" -> {}) suppresses a
    risky keyword ONLY when the safe keyword contains the risky keyword as a
    sub-string — meaning the safe keyword is the more specific description.

    Examples:
      "Sweet potato, baked":  safe "sweet potato" contains risky "sweet"
                              → risky keyword suppressed → 0.0
      "Fried chicken":        safe "chicken" does NOT contain risky "fried"
                              → "fried" contributes its risk score normally
      "Ice cream, vanilla":   no safe keyword matches
                              → best of "ice cream" and "cream" scores used
    """
    name_lower     = food_name.lower()
    matched_safe:  list[str]             = []
    matched_risky: list[tuple[str, float]] = []

    for kw, sym_scores in _FOOD_RISK_KEYWORDS.items():
        if kw not in name_lower:
            continue
        if not sym_scores:                      # explicitly safe keyword
            matched_safe.append(kw)
        else:
            score = sym_scores.get(symptom, 0.0)
            if score > 0.0:
                matched_risky.append((kw, score))

    if not matched_risky:
        return 0.0

    # A risky keyword is suppressed only when a matched safe keyword
    # subsumes it (i.e. the safe keyword is the longer, more specific form).
    best = 0.0
    for rk, score in matched_risky:
        suppressed = any(rk in sk for sk in matched_safe)
        if not suppressed:
            best = max(best, score)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Data classes (identical to Claude version)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NutrientSnapshot:
    description: str
    calories:    Optional[float] = None
    protein:     Optional[float] = None
    total_fat:   Optional[float] = None
    carbs:       Optional[float] = None
    sodium:      Optional[float] = None
    sat_fat:     Optional[float] = None
    cholesterol: Optional[float] = None
    sugar:       Optional[float] = None
    portion_g:   float           = 100.0


@dataclass
class FoodLogEntry:
    user_id: str; usda_id: int; logged_at: datetime; quantity_g: float


@dataclass
class SymptomLogEntry:
    user_id: str; symptom: str; logged_at: datetime; intensity: str


@dataclass
class FoodCausationResult:
    usda_id: int; food_name: str; quantity_g: float; logged_at: str
    hours_before: float; entailment_score: float; contradiction_score: float
    neutral_score: float; nutrient_risk_score: float; temporal_score: float
    quantity_score: float; causation_score: float; causation_label: str
    causation_pct: str
    personal_prior: float       = 0.5
    prior_confidence: float     = 0.0
    prior_observations: int     = 0
    prior_confirmations: int    = 0
    personalisation_weight: float = 0.0
    model_weight: float         = 1.0
    explanation: str            = ""
    top_risk_nutrients: list[str] = field(default_factory=list)


@dataclass
class SymptomPrediction:
    symptom: str; intensity: str; symptom_logged_at: str; digestion_window: str
    foods_in_window: int; foods_outside_window: int
    top_cause: Optional[FoodCausationResult]
    all_candidates: list[FoodCausationResult]
    no_foods_found: bool = False
    note: str            = ""


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python helpers (unchanged logic from original)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dt(ts) -> datetime:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(str(ts).rstrip("Z"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _fmt_time(dt: datetime) -> str:
    return dt.strftime("%b %d %H:%M")


def _get_nutrient_snapshot(usda_id: int, quantity_g: float) -> NutrientSnapshot:
    if _usda_client is None:
        return NutrientSnapshot(description=f"USDA ID {usda_id}", portion_g=quantity_g)
    data = _usda_client.get_nutrients(usda_id)
    if data is None:
        return NutrientSnapshot(description=f"USDA ID {usda_id}", portion_g=quantity_g)
    scale = quantity_g / 100.0
    def sv(k):
        v = data.get(k)
        return round(float(v) * scale, 3) if v is not None else None
    return NutrientSnapshot(
        description  = data.get("description", f"USDA ID {usda_id}"),
        calories     = sv("calories"), protein  = sv("protein"),
        total_fat    = sv("total_fat"), carbs   = sv("carbs"),
        sodium       = sv("sodium"),   sat_fat  = sv("sat_fat"),
        cholesterol  = sv("cholesterol"), sugar = sv("sugar"),
        portion_g    = quantity_g,
    )


def _build_premise(food_name: str, nutrients: NutrientSnapshot) -> str:
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


_SYMPTOM_PHRASES: dict[str, str] = {
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

# Reverse map: phrase fragment → canonical symptom name (for parsing hypotheses)
_PHRASE_TO_SYMPTOM: dict[str, str] = {phrase: sym for sym, phrase in _SYMPTOM_PHRASES.items()}


def _build_hypothesis(symptom: str, intensity: str) -> str:
    phrase = _SYMPTOM_PHRASES.get(symptom, f"caused {symptom.lower()}")
    adv    = {"Mild": "mild ", "Moderate": "moderate ", "Severe": "severe "}.get(intensity, "")
    # Embed symptom name in brackets so _nli_score can parse it without regex fragility
    return f"Eating this food {phrase} ({adv}intensity). [symptom:{symptom}]"


def _extract_symptom_from_hypothesis(hypothesis: str) -> str:
    """Extract canonical symptom name embedded by _build_hypothesis."""
    m = _re.search(r"\[symptom:([^\]]+)\]", hypothesis)
    if m:
        return m.group(1).strip()
    # Fallback: scan known symptom names in the text
    h_lower = hypothesis.lower()
    for sym in SYMPTOM_WINDOWS:
        if sym.lower() in h_lower:
            return sym
    # Fallback for culprit_food_finder format: "This food caused the user's {symptom}."
    m2 = _re.search(r"user's ([a-z ]+)\.", hypothesis, _re.IGNORECASE)
    if m2:
        candidate = m2.group(1).strip().title()
        for sym in SYMPTOM_WINDOWS:
            if sym.lower() == candidate.lower():
                return sym
        return candidate
    return ""


def _extract_food_from_premise(premise: str) -> str:
    """Extract food name from a premise string."""
    # _build_premise format: "The person ate FOOD_NAME (Xg portion) ..."
    m = _re.search(r"The person ate (.+?)(?:\s*\(\d+g portion\)|\.)", premise, _re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # culprit_food_finder format: "The user recently ate FOOD_NAME (Xg)."
    m2 = _re.search(r"(?:ate|eaten)\s+(.+?)(?:\s*\(\d+g\)|\s*\.)", premise, _re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return premise


def _parse_nutrients_from_premise(premise: str) -> dict[str, float]:
    """
    Parse nutrient values embedded in a _build_premise() string.
    Returns {nutrient_key: value} for any nutrients found.
    """
    nutrients: dict[str, float] = {}
    patterns = [
        (r"([\d.]+)\s*kcal",              "calories"),
        (r"([\d.]+)g\s*total fat",        "total_fat"),
        (r"([\d.]+)g\s*saturated fat",    "sat_fat"),
        (r"([\d.]+)g\s*sugar",            "sugar"),
        (r"([\d.]+)mg\s*sodium",          "sodium"),
        (r"([\d.]+)g\s*carbohydrates",    "carbs"),
        (r"([\d.]+)mg\s*cholesterol",     "cholesterol"),
        (r"([\d.]+)g\s*protein",          "protein"),
    ]
    for pattern, key in patterns:
        m = _re.search(pattern, premise, _re.IGNORECASE)
        if m:
            nutrients[key] = float(m.group(1))
    return nutrients


def _nutrient_risk(nutrients: NutrientSnapshot, symptom: str) -> tuple[float, list[str]]:
    """
    Compute a 0-1 nutrient-risk score for a food vs. a symptom.
    Returns (risk_score, risk_notes).
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
    total_risk   = 0.0
    total_weight = sum(risk_weights.values()) or 1.0
    risk_notes:  list[str] = []

    for nutrient, weight in risk_weights.items():
        val = val_map.get(nutrient)
        if val is None:
            continue
        threshold = _NUTRIENT_RISK_THRESHOLDS.get(nutrient, 1.0) * scale
        ratio     = min(val / max(threshold, 0.001), 2.0)
        total_risk += ratio * weight
        if ratio >= 1.0:
            unit = ("mg" if nutrient in ("sodium", "cholesterol")
                    else ("kcal" if nutrient == "calories" else "g"))
            risk_notes.append(
                f"{nutrient.replace('_',' ').title()} {val:.1f}{unit} "
                f"— {ratio:.1f}× threshold for {symptom}"
            )

    return round(min(total_risk / total_weight, 1.0), 4), risk_notes[:3]


def _temporal_score(hours_before: float, window_min: float, window_max: float) -> float:
    """Higher score = food was eaten closer to the centre of the digestion window."""
    if hours_before < window_min or hours_before > window_max:
        return 0.0
    span   = window_max - window_min
    centre = window_min + span / 2.0
    dist   = abs(hours_before - centre)
    return round(1.0 - (dist / (span / 2.0)) * 0.5, 4)


def _quantity_score(portion_g: float) -> float:
    return round(min(portion_g / (portion_g + 150.0), 1.0), 4)


def _causation_label(score: float) -> tuple[str, str]:
    if   score >= 0.70: return "Likely",   f"{int(score * 100)}%"
    elif score >= 0.45: return "Possible", f"{int(score * 100)}%"
    else:               return "Unlikely", f"{int(score * 100)}%"


# ─────────────────────────────────────────────────────────────────────────────
# Logic-based NLI scoring  (replaces Claude batch call)
# ─────────────────────────────────────────────────────────────────────────────

def _score_pair_logically(premise: str, hypothesis: str) -> float:
    """
    Compute a 0-1 entailment score without any external model or API.

    Strategy (in order of signal quality):
    1. Parse nutrient values from premise (if built by _build_premise).
       Apply SYMPTOM_NUTRIENT_RISK weighting.
    2. Try USDA lookup by food name for full nutrient profile.
    3. Apply food-descriptor keyword heuristics.
    Final score = weighted blend of available signals.
    """
    symptom   = _extract_symptom_from_hypothesis(hypothesis)
    food_name = _extract_food_from_premise(premise)

    # ── Signal 1: inline nutrient values (fast, no API call) ─────────────────
    inline_nutrients = _parse_nutrients_from_premise(premise)
    nutrient_score   = 0.0
    has_inline       = bool(inline_nutrients)

    if has_inline and symptom:
        risk_weights  = SYMPTOM_NUTRIENT_RISK.get(symptom, {})
        total_weight  = sum(risk_weights.values()) or 1.0
        total_risk    = 0.0
        for nutrient, weight in risk_weights.items():
            val = inline_nutrients.get(nutrient)
            if val is None:
                continue
            threshold  = _NUTRIENT_RISK_THRESHOLDS.get(nutrient, 1.0)
            total_risk += min(val / max(threshold, 0.001), 2.0) * weight
        nutrient_score = round(min(total_risk / total_weight, 1.0), 4)

    # ── Signal 2: USDA nutrient lookup (if inline is absent & client ready) ──
    usda_score   = 0.0
    has_usda     = False
    if not has_inline and _usda_client and food_name and symptom:
        try:
            hits = _usda_client.search(food_name, top_k=1)
            if hits:
                fdc_id = hits[0]["usda_id"]
                data   = _usda_client.get_nutrients(fdc_id)
                if data:
                    nuts = NutrientSnapshot(
                        description  = data.get("description", food_name),
                        calories     = data.get("calories"),
                        total_fat    = data.get("total_fat"),
                        sat_fat      = data.get("sat_fat"),
                        sugar        = data.get("sugar"),
                        sodium       = data.get("sodium"),
                        carbs        = data.get("carbs"),
                        cholesterol  = data.get("cholesterol"),
                        portion_g    = 100.0,
                    )
                    usda_score, _ = _nutrient_risk(nuts, symptom)
                    has_usda = True
        except Exception as exc:
            log.debug(f"USDA lookup in batch_nli_score failed for '{food_name}': {exc}")

    # ── Signal 3: food descriptor keyword heuristics ──────────────────────────
    kw_score = _keyword_risk_score(food_name, symptom) if symptom else 0.0

    # ── Blend signals ─────────────────────────────────────────────────────────
    if has_inline:
        # Inline nutrients are the best signal; keywords as a supplement
        final = 0.75 * nutrient_score + 0.25 * kw_score
    elif has_usda:
        final = 0.70 * usda_score + 0.30 * kw_score
    elif kw_score > 0:
        # Keywords only — less confident; scale down slightly
        final = 0.65 * kw_score
    else:
        # No signal at all → neutral/moderate default
        final = 0.25

    return round(min(max(final, 0.0), 1.0), 4)


def batch_nli_score(pairs: list[tuple[str, str]]) -> list[dict]:
    """
    Score multiple (premise, hypothesis) pairs using deterministic logic.
    Replaces the Claude batch NLI call.  Same return shape.

    Returns list of {"entailment", "neutral", "contradiction"} dicts.
    """
    if not pairs:
        return []

    results = []
    for premise, hypothesis in pairs:
        entail = _score_pair_logically(premise, hypothesis)
        # Contradiction should be high when entailment is very low
        contra  = round(max(0.0, (0.4 - entail) * 0.5), 4)
        neutral = round(max(0.0, 1.0 - entail - contra), 4)
        results.append({
            "entailment":    entail,
            "neutral":       neutral,
            "contradiction": contra,
        })
    return results


def _nli_score(premise: str, hypothesis: str) -> dict:
    """Single-pair wrapper over batch_nli_score.  Used by meal_symptom_forecast."""
    return batch_nli_score([(premise, hypothesis)])[0]


# Legacy no-ops kept for import compatibility
def _find_col(df, key):                            return None
def _get_nutrient(row, df, key, scale=1.0):        return None


# ─────────────────────────────────────────────────────────────────────────────
# Composite meal grouping  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def group_composite_meals(food_logs_named: list[dict]) -> list[dict]:
    """
    Group food-log entries sharing the same logged_at timestamp into a single
    composite meal entry (e.g. chicken + rice + oil all logged at 12:00).
    """
    def _ts_key(entry: dict) -> str:
        val = entry.get("logged_at")
        return val.isoformat() if isinstance(val, datetime) else str(val)

    groups: dict[str, list[dict]] = _defaultdict(list)
    for entry in food_logs_named:
        groups[_ts_key(entry)].append(entry)

    result: list[dict] = []
    for ts_key in sorted(groups):
        foods = groups[ts_key]
        if len(foods) == 1:
            f = foods[0]
            result.append({
                "food_name":    f["food_name"],
                "weight_g":     float(f.get("weight_g") or f.get("quantity_g") or 0),
                "logged_at":    f["logged_at"],
                "is_composite": False,
                "components":   [f["food_name"]],
            })
        else:
            total_w = sum(float(f.get("weight_g") or f.get("quantity_g") or 0) for f in foods)
            names   = list(dict.fromkeys(f["food_name"] for f in foods))
            result.append({
                "food_name":    " + ".join(names),
                "weight_g":     round(total_w, 1),
                "logged_at":    foods[0]["logged_at"],
                "is_composite": True,
                "components":   names,
            })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Timeline string builder  (used by food_recommender's rule-based path)
# ─────────────────────────────────────────────────────────────────────────────

def _build_timeline_prompt(grouped_meals: list[dict], symptom_logs: list[dict]) -> str:
    """Compact chronological timeline string (kept for food_recommender import)."""
    lines = ["=== FOOD / MEAL LOG ==="]
    for m in grouped_meals:
        dt      = _parse_dt(m["logged_at"])
        w       = m.get("weight_g", 0)
        wt_part = f" ({round(w)}g)" if w else ""
        prefix  = "[MEAL] " if m["is_composite"] else ""
        lines.append(f"  {_fmt_time(dt)}  |  {prefix}{m['food_name']}{wt_part}")

    lines += ["", "=== SYMPTOM LOG ==="]
    for s in sorted(symptom_logs, key=lambda x: _parse_dt(x["logged_at"])):
        dt        = _parse_dt(s["logged_at"])
        intensity = s.get("intensity", "")
        int_part  = f" [{intensity}]" if intensity else ""
        lines.append(f"  {_fmt_time(dt)}  |  {s['symptom']}{int_part}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# predict_causation_by_time  — fully logic-based, no Claude
# ─────────────────────────────────────────────────────────────────────────────

def predict_causation_by_time(
    food_logs_named: list[dict],
    symptom_logs:    list[dict],
) -> dict[str, list[str]]:
    """
    Identify which foods most plausibly caused each reported symptom.

    Algorithm (no external API calls):
    ─────────────────────────────────
    1. Group foods into composite meals (same timestamp → one meal event).
    2. For each symptom occurrence:
         a. Temporal filter: keep only foods eaten within the symptom's
            clinical digestion window.
         b. Score each candidate:
              temporal_score  × 0.40  (closeness to window centre)
              nutrient_risk   × 0.40  (USDA nutrients vs danger thresholds)
              keyword_score   × 0.20  (food descriptor heuristics)
         c. Keep candidates with score ≥ 0.20; sort descending.
    3. Aggregate across all occurrences of the same symptom:
         - A food name that appears for multiple occurrences of the same
           symptom gets a frequency boost.
    4. Return {symptom: [food_name, ...]} for every symptom in the input.

    Parameters
    ----------
    food_logs_named : list[dict]  — {food_name, usda_id, weight_g, logged_at}
    symptom_logs    : list[dict]  — {symptom, intensity, logged_at}

    Returns
    -------
    dict[str, list[str]]  — {symptom: [food_names]}  (ordered by score)
    """
    unique_symptoms = list(dict.fromkeys(
        s.get("symptom", "") for s in symptom_logs if s.get("symptom")
    ))

    if not food_logs_named or not symptom_logs:
        return {sym: [] for sym in unique_symptoms}

    grouped = group_composite_meals(food_logs_named)

    # Build usda_id lookup from original food_logs_named for nutrient fetching
    # (group_composite_meals loses usda_id since we only pass food_name)
    name_to_id: dict[str, int] = {}
    for entry in food_logs_named:
        fn = entry.get("food_name", "")
        if fn and entry.get("usda_id"):
            name_to_id[fn] = int(entry["usda_id"])

    # Accumulate per-symptom scores across all symptom events
    # {symptom → {food_name → [score, ...]}}
    sym_food_scores: dict[str, dict[str, list[float]]] = {
        sym: _defaultdict(list) for sym in unique_symptoms
    }

    for s in symptom_logs:
        symptom  = s.get("symptom", "")
        if symptom not in unique_symptoms:
            continue
        sym_time = _parse_dt(s["logged_at"])
        win_min, win_max = SYMPTOM_WINDOWS.get(symptom, _DEFAULT_WINDOW_H)

        for meal in grouped:
            ft = _parse_dt(meal["logged_at"])
            if ft >= sym_time:
                continue
            hours = (sym_time - ft).total_seconds() / 3600.0
            if not (win_min <= hours <= win_max):
                continue

            t_score = _temporal_score(hours, win_min, win_max)

            # Nutrient risk: try USDA lookup for each component of the meal
            comp_names = meal.get("components", [meal["food_name"]])
            best_nutrient_score = 0.0
            for comp in comp_names:
                uid = name_to_id.get(comp)
                if uid and _usda_client:
                    data = _usda_client.get_nutrients(uid)
                    if data:
                        qty_g = meal["weight_g"] / max(len(comp_names), 1)
                        nuts  = NutrientSnapshot(
                            description  = data.get("description", comp),
                            calories     = _scale(data.get("calories"),     qty_g),
                            total_fat    = _scale(data.get("total_fat"),    qty_g),
                            sat_fat      = _scale(data.get("sat_fat"),      qty_g),
                            sugar        = _scale(data.get("sugar"),        qty_g),
                            sodium       = _scale(data.get("sodium"),       qty_g),
                            carbs        = _scale(data.get("carbs"),        qty_g),
                            cholesterol  = _scale(data.get("cholesterol"),  qty_g),
                            portion_g    = qty_g,
                        )
                        n_risk, _ = _nutrient_risk(nuts, symptom)
                        best_nutrient_score = max(best_nutrient_score, n_risk)

            kw_score = _keyword_risk_score(meal["food_name"], symptom)

            # Combined score
            combined = (0.40 * t_score
                        + 0.40 * best_nutrient_score
                        + 0.20 * kw_score)

            # If nutrient data was absent, fall back to temporal + keyword only
            if best_nutrient_score == 0.0:
                combined = 0.60 * t_score + 0.40 * kw_score

            if combined >= 0.20:  # discard very low-confidence candidates
                sym_food_scores[symptom][meal["food_name"]].append(combined)

    # Aggregate: mean score per (symptom, food), sort descending
    output: dict[str, list[str]] = {}
    for sym in unique_symptoms:
        food_agg: dict[str, float] = {}
        for food_name, scores in sym_food_scores[sym].items():
            # Frequency boost: appearing in multiple symptom events raises confidence
            freq_boost = min((len(scores) - 1) * 0.05, 0.15)
            food_agg[food_name] = min(sum(scores) / len(scores) + freq_boost, 1.0)

        sorted_foods = sorted(food_agg.items(), key=lambda kv: kv[1], reverse=True)
        output[sym] = [fn for fn, _ in sorted_foods]

    log.info(
        f"predict_causation_by_time (logic): triggers found for "
        f"{sum(bool(v) for v in output.values())}/{len(unique_symptoms)} symptoms "
        f"from {len(grouped)} meal events."
    )
    return output


def _scale(val, qty_g: float) -> Optional[float]:
    """Scale a per-100g nutrient value to the actual portion."""
    if val is None:
        return None
    return round(float(val) * qty_g / 100.0, 3)


# ─────────────────────────────────────────────────────────────────────────────
# predict_food_symptom_causes  (unchanged logic; NLI now uses _nli_score above)
# ─────────────────────────────────────────────────────────────────────────────

def predict_food_symptom_causes(
    food_logs:    list[FoodLogEntry],
    symptom_logs: list[SymptomLogEntry],
    user_memory   = None,
    **_kwargs,
) -> list[SymptomPrediction]:
    """
    Per-symptom causation scoring using nutrient risk + temporal windows.
    Identical output shape to the Claude version.
    """
    predictions: list[SymptomPrediction] = []

    for sym_entry in symptom_logs:
        symptom   = sym_entry.symptom
        sym_time  = sym_entry.logged_at
        intensity = sym_entry.intensity
        win_min, win_max = SYMPTOM_WINDOWS.get(symptom, (0.5, 8.0))

        # ── Temporal filter ──────────────────────────────────────────────────
        candidates:    list[tuple[FoodLogEntry, float]] = []
        outside_count: int = 0

        for food in food_logs:
            if food.user_id != sym_entry.user_id:
                continue
            if food.logged_at >= sym_time:
                continue
            hours_before = (sym_time - food.logged_at).total_seconds() / 3600.0
            if win_min <= hours_before <= win_max:
                candidates.append((food, hours_before))
            else:
                outside_count += 1

        if not candidates:
            predictions.append(SymptomPrediction(
                symptom=symptom, intensity=intensity,
                symptom_logged_at=sym_time.isoformat(),
                digestion_window=f"{win_min}h – {win_max}h before symptom",
                foods_in_window=0, foods_outside_window=outside_count,
                top_cause=None, all_candidates=[], no_foods_found=True,
                note=(
                    f"No food logs found within the {win_min}–{win_max}h "
                    f"digestion window. {outside_count} food(s) outside window."
                ),
            ))
            continue

        # ── Fetch nutrients for all unique foods in one pass ─────────────────
        nutrient_cache: dict[int, NutrientSnapshot] = {}
        for food_entry, _ in candidates:
            if food_entry.usda_id not in nutrient_cache:
                nutrient_cache[food_entry.usda_id] = _get_nutrient_snapshot(
                    food_entry.usda_id, food_entry.quantity_g,
                )

        # ── Score each candidate ─────────────────────────────────────────────
        results: list[FoodCausationResult] = []
        for food_entry, hours_before in candidates:
            nuts = nutrient_cache[food_entry.usda_id]

            # Logic-based NLI score via premise→hypothesis scoring
            premise    = _build_premise(nuts.description, nuts)
            hypothesis = _build_hypothesis(symptom, intensity)
            nli_scores = _nli_score(premise, hypothesis)
            nli_ent    = nli_scores["entailment"]

            n_risk, risk_notes = _nutrient_risk(nuts, symptom)
            t_score            = _temporal_score(hours_before, win_min, win_max)
            q_score            = _quantity_score(food_entry.quantity_g)

            base_causation = round(min(
                0.35 * nli_ent + 0.35 * n_risk + 0.20 * t_score + 0.10 * q_score, 1.0,
            ), 4)

            personal_prior  = 0.5
            prior_conf      = 0.0
            prior_obs       = 0
            prior_conf_int  = 0
            p_weight        = 0.0
            m_weight        = 1.0
            causation       = base_causation

            if user_memory is not None:
                blended, breakdown = user_memory.blend_score(
                    usda_id=food_entry.usda_id, symptom=symptom, nli_score=nli_ent,
                )
                causation     = round(min(
                    0.35 * blended + 0.35 * n_risk + 0.20 * t_score + 0.10 * q_score, 1.0,
                ), 4)
                personal_prior = breakdown["personal_prior"]
                prior_conf     = breakdown["prior_confidence"]
                prior_obs      = breakdown["prior_observations"]
                prior_conf_int = breakdown["prior_confirmations"]
                p_weight       = breakdown.get("personalisation_weight", 0.0)
                m_weight       = 1.0 - p_weight

            label, pct = _causation_label(causation)

            results.append(FoodCausationResult(
                usda_id=food_entry.usda_id, food_name=nuts.description,
                quantity_g=food_entry.quantity_g,
                logged_at=food_entry.logged_at.isoformat(),
                hours_before=round(hours_before, 2),
                entailment_score=nli_ent,
                contradiction_score=nli_scores["contradiction"],
                neutral_score=nli_scores["neutral"],
                nutrient_risk_score=n_risk,
                temporal_score=t_score,
                quantity_score=q_score,
                causation_score=causation,
                causation_label=label,
                causation_pct=pct,
                personal_prior=personal_prior,
                prior_confidence=prior_conf,
                prior_observations=prior_obs,
                prior_confirmations=prior_conf_int,
                personalisation_weight=p_weight,
                model_weight=m_weight,
                explanation=(
                    f"Scored {pct} causation probability. "
                    f"Eaten {hours_before:.1f}h before symptom "
                    f"(window: {win_min}–{win_max}h). "
                    + ("; ".join(risk_notes[:2]) if risk_notes else "No specific nutrient risks flagged.")
                ),
                top_risk_nutrients=risk_notes,
            ))

        results.sort(key=lambda r: r.causation_score, reverse=True)
        top = results[0] if results else None

        predictions.append(SymptomPrediction(
            symptom=symptom, intensity=intensity,
            symptom_logged_at=sym_time.isoformat(),
            digestion_window=f"{win_min}h – {win_max}h before symptom",
            foods_in_window=len(candidates),
            foods_outside_window=outside_count,
            top_cause=top,
            all_candidates=results,
            no_foods_found=False,
            note=(
                f"Analysed {len(candidates)} food(s) in digestion window. "
                f"Top cause: {top.food_name if top else 'none'} "
                f"({top.causation_pct if top else '—'})."
            ),
        ))

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# JSON helper (kept for any callers that imported it)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_json_parse(raw: str) -> Optional[dict]:
    cleaned = _re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        return json.loads(cleaned)
    except Exception:
        return None