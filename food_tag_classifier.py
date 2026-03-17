"""
food_tag_classifier.py
──────────────────────
Classifies a USDA food ID into one or more meal category tags.

Tags: Dairy | Gluten | Spicy | Fried | Sugar | Caffeine | Processed Food | Others

━━━ APPROACH ━━━
Two-layer classification — both layers run for every food:

  Layer 1 — Zero-shot NLI (DeBERTa-v3-base, already loaded by nutrition_scorer.py)
    The USDA food description is used as the NLI premise.
    Each tag is expressed as a natural-language entailment hypothesis.
    multi_label=True → each tag scored independently (0–1).
    No new model is loaded — reuses the pipeline from nutrition_scorer.

  Layer 2 — Nutrient heuristic boosts
    Hard clinical rules boost or suppress tag scores based on USDA
    nutrient values. This corrects for USDA's ALL-CAPS terse naming
    (e.g. "BUTTER,WITH SALT" — DeBERTa reads this fine, but nutrient
    evidence makes Dairy near-certain regardless).

━━━ THRESHOLDS ━━━
  tag_threshold  = 0.35  → any tag above this is returned as active
  primary_tag    = tag with highest final score
  "Others"       = assigned only when zero tags clear the threshold

━━━ PUBLIC FUNCTION ━━━
  classify_food_tags(usda_id, usda_df, id_col, desc_col) → FoodTagResult

━━━ DESIGN NOTES ━━━
  • The DeBERTa classifier is imported lazily at first call — if
    nutrition_scorer failed to load, this module degrades gracefully
    to heuristic-only mode.
  • All scores are clamped to [0.0, 1.0] after boosting.
  • Score breakdown is always returned for API transparency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tag definitions
# Each tag has:
#   hypothesis  — NLI hypothesis fed to DeBERTa (must be a clear entailment)
#   boost_rules — list of (nutrient_key, operator, threshold, boost_amount)
#                 applied after NLI to adjust the score with hard evidence
# ─────────────────────────────────────────────────────────────────────────────

VALID_TAGS = [
    "Dairy",
    "Gluten",
    "Spicy",
    "Fried",
    "Sugar",
    "Caffeine",
    "Processed Food",
    "Others",
]

# NLI hypotheses — phrased as facts the model should confirm or deny
_TAG_HYPOTHESES: dict[str, str] = {
    "Dairy":         "This food is a dairy product or contains milk, cheese, cream, yogurt, or butter.",
    "Gluten":        "This food contains gluten, wheat, barley, rye, bread, pasta, or flour.",
    "Spicy":         "This food is spicy or hot, or contains chili peppers, hot sauce, or cayenne.",
    "Fried":         "This food is fried, deep-fried, or cooked by submerging in hot oil.",
    "Sugar":         "This food is high in sugar, sweeteners, syrup, or added sugars.",
    "Caffeine":      "This food contains caffeine, coffee, tea, chocolate, or energy-boosting stimulants.",
    "Processed Food":"This food is highly processed, packaged, preserved, or industrially manufactured.",
}
# "Others" has no NLI hypothesis — it is the fallback label, not scored by the model.

# ── Nutrient heuristic boost rules ───────────────────────────────────────────
# Format: (nutrient_key, operator, threshold, boost)
#   operator: "gte" (≥) or "lte" (≤)
#   boost: positive = raises score, negative = lowers score
# nutrient_key matches USDA_COLS in nutrition_scorer.py

_NUTRIENT_BOOSTS: dict[str, list[tuple]] = {
    "Dairy": [
        ("calcium",  "gte", 150.0,  +0.20),   # high calcium → very likely dairy
        ("calcium",  "gte", 250.0,  +0.15),   # extra boost for very high calcium
        ("sat_fat",  "gte",  10.0,  +0.10),   # butter/cheese signature
    ],
    "Gluten": [
        # No reliable nutrient signal — carbs are too broad
        # NLI carries the full weight for Gluten
    ],
    "Spicy": [
        # No reliable nutrient signal for spiciness in USDA data
        # NLI carries the full weight for Spicy
    ],
    "Fried": [
        ("total_fat","gte",  15.0,  +0.15),   # very high fat → likely fried
        ("total_fat","gte",  25.0,  +0.15),   # extra boost at very high fat
        ("sat_fat",  "gte",   8.0,  +0.10),   # fried foods high in sat fat
        ("calories", "gte", 300.0,  +0.05),   # calorie density signal
    ],
    "Sugar": [
        ("sugar",    "gte",  10.0,  +0.15),   # >10g/100g = high sugar
        ("sugar",    "gte",  20.0,  +0.15),   # >20g/100g = very high
        ("sugar",    "gte",  40.0,  +0.10),   # >40g/100g = extreme (candy/syrups)
        ("sugar",    "lte",   1.0,  -0.10),   # very low sugar → suppress tag
    ],
    "Caffeine": [
        # No caffeine column in USDA — NLI carries full weight
    ],
    "Processed Food": [
        ("sodium",   "gte", 400.0,  +0.15),   # high sodium → processed
        ("sodium",   "gte", 800.0,  +0.10),   # extra boost at very high sodium
        ("sat_fat",  "gte",   5.0,  +0.05),   # processed foods often high sat fat
    ],
}

# Confidence threshold — tags below this are considered inactive
_TAG_THRESHOLD = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TagScore:
    tag:          str
    score:        float           # final score after NLI + heuristic boosts
    nli_score:    float           # raw NLI score before boosts
    heuristic_boost: float        # total boost applied by nutrient rules
    active:       bool            # True if score >= _TAG_THRESHOLD
    reasons:      list[str] = field(default_factory=list)  # human-readable explanations


@dataclass
class FoodTagResult:
    usda_id:          int
    food_description: str
    primary_tag:      str            # highest-scoring active tag (or "Others")
    active_tags:      list[str]      # all tags above threshold
    tag_scores:       list[TagScore] # full breakdown for every tag
    confidence:       float          # primary tag score
    is_others:        bool           # True when no tag cleared the threshold
    classification_method: str       # "nli+heuristic" | "heuristic_only"
    note:             str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Lazy classifier loader — reuses nutrition_scorer's already-loaded pipeline
# ─────────────────────────────────────────────────────────────────────────────

_classifier = None   # set on first call


def _get_classifier():
    """
    Return the zero-shot DeBERTa pipeline.
    Tries to reuse the one loaded by nutrition_scorer first.
    Falls back to loading a fresh instance if nutrition_scorer is unavailable.
    """
    global _classifier
    if _classifier is not None:
        return _classifier

    # Attempt 1: reuse already-loaded instance from nutrition_scorer
    try:
        from nutrition_scorer import _classifier as _ns_clf
        if _ns_clf is not None:
            _classifier = _ns_clf
            log.info("food_tag_classifier: reusing DeBERTa pipeline from nutrition_scorer.")
            return _classifier
    except Exception:
        pass

    # Attempt 2: load fresh (fallback — costs RAM if nutrition_scorer is also running)
    try:
        from transformers import pipeline as hf_pipeline
        _classifier = hf_pipeline(
            "zero-shot-classification",
            model  = "MoritzLaurer/deberta-v3-base-mnli-fever-anli",
            device = -1,
        )
        log.info("food_tag_classifier: loaded fresh DeBERTa pipeline.")
    except Exception as exc:
        log.warning(f"food_tag_classifier: could not load DeBERTa ({exc}). Heuristic-only mode.")
        _classifier = None

    return _classifier


# ─────────────────────────────────────────────────────────────────────────────
# Nutrient helpers — safe column resolution matching nutrition_scorer's approach
# ─────────────────────────────────────────────────────────────────────────────

_NUTRIENT_COL_ALIASES: dict[str, list[str]] = {
    "calories":  ["Calories","calories","Energ_Kcal","Energy","energy","kcal"],
    "total_fat": ["TotalFat","Total_Fat","total_fat","Lipid_Tot","Fat","fat"],
    "sat_fat":   ["SaturatedFat","Saturated_Fat","saturated_fat","FA_Sat","FA_SAT"],
    "sodium":    ["Sodium","sodium","SODIUM","Sodium_mg"],
    "sugar":     ["Sugar","sugar","Sugar_Tot","Sugars","sugars"],
    "calcium":   ["Calcium","calcium","CALCIUM","Calcium_mg"],
}


def _resolve(df: pd.DataFrame, key: str):
    """Resolve a nutrient key to the actual column name in df."""
    lower_map = {c.lower(): c for c in df.columns}
    for alias in _NUTRIENT_COL_ALIASES.get(key, []):
        if alias in df.columns:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def _safe_float(row: pd.Series, col: str) -> float:
    """Extract a float from a row, returning 0.0 on failure."""
    import math
    try:
        v = float(row.get(col, 0.0))
        return v if not math.isnan(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _get_nutrients(row: pd.Series, df: pd.DataFrame) -> dict[str, float]:
    """Extract all nutrient values needed for heuristic boosts."""
    out = {}
    for key in _NUTRIENT_COL_ALIASES:
        col = _resolve(df, key)
        out[key] = _safe_float(row, col) if col else 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic boost engine
# ─────────────────────────────────────────────────────────────────────────────

def _apply_boosts(
    tag:       str,
    nutrients: dict[str, float],
) -> tuple[float, list[str]]:
    """
    Apply nutrient-based heuristic boosts for a tag.
    Returns (total_boost, list_of_reason_strings).
    """
    rules = _NUTRIENT_BOOSTS.get(tag, [])
    total = 0.0
    reasons = []

    for nutrient_key, operator, threshold, boost in rules:
        value = nutrients.get(nutrient_key, 0.0)
        triggered = (
            (operator == "gte" and value >= threshold) or
            (operator == "lte" and value <= threshold)
        )
        if triggered:
            total += boost
            direction = "≥" if operator == "gte" else "≤"
            sign      = f"+{boost:.2f}" if boost >= 0 else f"{boost:.2f}"
            reasons.append(
                f"{nutrient_key.replace('_', ' ').title()} "
                f"{direction}{threshold} ({value:.1f}) → {sign} score"
            )

    return round(total, 4), reasons


# ─────────────────────────────────────────────────────────────────────────────
# NLI classification
# ─────────────────────────────────────────────────────────────────────────────

def _run_nli(food_description: str) -> dict[str, float] | None:
    """
    Run zero-shot multi-label NLI on the food description.
    Returns {tag: score} or None if classifier unavailable.
    """
    clf = _get_classifier()
    if clf is None:
        return None

    hypotheses = list(_TAG_HYPOTHESES.values())
    tag_names  = list(_TAG_HYPOTHESES.keys())

    try:
        result = clf(
            food_description,
            candidate_labels = hypotheses,
            multi_label      = True,   # each tag scored independently
            hypothesis_template = "{}",
        )
        # Map back from hypothesis string to tag name
        hyp_to_tag = {h: t for t, h in _TAG_HYPOTHESES.items()}
        scores = {}
        for label, score in zip(result["labels"], result["scores"]):
            tag = hyp_to_tag.get(label)
            if tag:
                scores[tag] = round(float(score), 4)
        return scores
    except Exception as exc:
        log.warning(f"NLI classification failed for '{food_description}': {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public function
# ─────────────────────────────────────────────────────────────────────────────

def classify_food_tags(
    usda_id:   int,
    usda_df:   pd.DataFrame,
    id_col:    str,
    desc_col:  str,
) -> FoodTagResult:
    """
    Classify a USDA food ID into meal category tags.

    Args:
        usda_id:  USDA food ID (integer)
        usda_df:  Loaded USDA DataFrame (from food_text_to_usda or main._state.usda_df)
        id_col:   Name of the ID column in usda_df (e.g. "ID")
        desc_col: Name of the description column (e.g. "Description")

    Returns:
        FoodTagResult with primary_tag, active_tags, and full score breakdown.

    Raises:
        ValueError if usda_id not found in dataset.
    """
    # ── 1. Look up food in USDA dataset ──────────────────────────────────────
    mask = usda_df[id_col].astype(str) == str(usda_id)
    if not mask.any():
        raise ValueError(f"USDA ID {usda_id} not found in dataset.")

    row  = usda_df[mask].iloc[0]
    desc = str(row[desc_col])

    # ── 2. Get nutrients for heuristic layer ─────────────────────────────────
    nutrients = _get_nutrients(row, usda_df)

    # ── 3. Run NLI classification ─────────────────────────────────────────────
    nli_scores = _run_nli(desc)
    method     = "nli+heuristic" if nli_scores is not None else "heuristic_only"

    # ── 4. Combine NLI + heuristic boosts for each tag ───────────────────────
    tag_scores: list[TagScore] = []

    for tag in VALID_TAGS:
        if tag == "Others":
            continue  # handled separately below

        nli  = nli_scores.get(tag, 0.0) if nli_scores else 0.0
        boost, boost_reasons = _apply_boosts(tag, nutrients)

        final = min(1.0, max(0.0, nli + boost))

        reasons = []
        if nli_scores is not None:
            reasons.append(f"NLI model score: {nli:.2f}")
        reasons.extend(boost_reasons)

        tag_scores.append(TagScore(
            tag              = tag,
            score            = round(final, 4),
            nli_score        = nli,
            heuristic_boost  = boost,
            active           = final >= _TAG_THRESHOLD,
            reasons          = reasons,
        ))

    # ── 5. Determine active tags and primary tag ──────────────────────────────
    active = [ts for ts in tag_scores if ts.active]
    active.sort(key=lambda x: x.score, reverse=True)

    if active:
        primary_tag    = active[0].tag
        confidence     = active[0].score
        active_tag_names = [ts.tag for ts in active]
        is_others      = False
        # Add the Others TagScore as inactive placeholder
        tag_scores.append(TagScore(
            tag="Others", score=0.0, nli_score=0.0,
            heuristic_boost=0.0, active=False,
            reasons=["Others is the fallback — assigned only when no other tag qualifies."],
        ))
    else:
        primary_tag      = "Others"
        confidence       = 1.0
        active_tag_names = ["Others"]
        is_others        = True
        tag_scores.append(TagScore(
            tag="Others", score=1.0, nli_score=0.0,
            heuristic_boost=0.0, active=True,
            reasons=["No specific tag cleared the confidence threshold."],
        ))

    # ── 6. Build note ─────────────────────────────────────────────────────────
    if is_others:
        note = (
            f"No tag cleared the {_TAG_THRESHOLD:.0%} confidence threshold. "
            f"The food may be ambiguous or a mixed/composite item. "
            f"Top NLI score was {max((ts.nli_score for ts in tag_scores if ts.tag != 'Others'), default=0.0):.2f}."
        )
    elif method == "heuristic_only":
        note = (
            "NLI model unavailable — classification based on nutrient heuristics only. "
            "Accuracy may be lower for ambiguous foods."
        )
    else:
        note = (
            f"Classified using DeBERTa NLI + nutrient heuristics. "
            f"{len(active)} tag(s) active above {_TAG_THRESHOLD:.0%} threshold."
        )

    return FoodTagResult(
        usda_id               = usda_id,
        food_description      = desc,
        primary_tag           = primary_tag,
        active_tags           = active_tag_names,
        tag_scores            = sorted(tag_scores, key=lambda x: x.score, reverse=True),
        confidence            = round(confidence, 4),
        is_others             = is_others,
        classification_method = method,
        note                  = note,
    )