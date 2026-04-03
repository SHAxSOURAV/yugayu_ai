"""
food_tag_classifier.py
──────────────────────
Classifies a USDA food ID into meal category tags.
Tags: Dairy | Gluten | Spicy | Fried | Sugar | Caffeine | Processed Food | Others

Old approach: DeBERTa NLI (zero-shot) + nutrient heuristic boosts.
New approach: Claude API (zero-shot) + nutrient heuristic boosts (unchanged).
              No extra RAM, no model download.

Public function:
    classify_food_tags(usda_id) -> FoodTagResult
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

_claude_client = None
_usda_client   = None


def init(claude_client, usda_client) -> None:
    global _claude_client, _usda_client
    _claude_client = claude_client
    _usda_client   = usda_client
    log.info("food_tag_classifier: Claude + USDA client ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Tag definitions (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

VALID_TAGS = ["Dairy", "Gluten", "Spicy", "Fried", "Sugar", "Caffeine", "Processed Food", "Others"]

_TAG_HYPOTHESES: dict[str, str] = {
    "Dairy":          "This food is a dairy product or contains milk, cheese, cream, yogurt, or butter.",
    "Gluten":         "This food contains gluten, wheat, barley, rye, bread, pasta, or flour.",
    "Spicy":          "This food is spicy or hot, or contains chili peppers, hot sauce, or cayenne.",
    "Fried":          "This food is fried, deep-fried, or cooked by submerging in hot oil.",
    "Sugar":          "This food is high in sugar, sweeteners, syrup, or added sugars.",
    "Caffeine":       "This food contains caffeine, coffee, tea, chocolate, or energy-boosting stimulants.",
    "Processed Food": "This food is highly processed, packaged, preserved, or industrially manufactured.",
}

_NUTRIENT_BOOSTS: dict[str, list[tuple]] = {
    "Dairy":         [("calcium", "gte", 150.0, +0.20), ("calcium", "gte", 250.0, +0.15), ("sat_fat", "gte", 10.0, +0.10)],
    "Gluten":        [],
    "Spicy":         [],
    "Fried":         [("total_fat", "gte", 15.0, +0.15), ("total_fat", "gte", 25.0, +0.15), ("sat_fat", "gte", 8.0, +0.10), ("calories", "gte", 300.0, +0.05)],
    "Sugar":         [("sugar", "gte", 10.0, +0.15), ("sugar", "gte", 20.0, +0.15), ("sugar", "gte", 40.0, +0.10), ("sugar", "lte", 1.0, -0.10)],
    "Caffeine":      [],
    "Processed Food":[("sodium", "gte", 400.0, +0.15), ("sodium", "gte", 800.0, +0.10), ("sat_fat", "gte", 5.0, +0.05)],
}

_TAG_THRESHOLD = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Data classes (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TagScore:
    tag: str; score: float; nli_score: float; heuristic_boost: float
    active: bool; reasons: list[str] = field(default_factory=list)


@dataclass
class FoodTagResult:
    usda_id: int; food_description: str; primary_tag: str
    active_tags: list[str]; tag_scores: list[TagScore]
    confidence: float; is_others: bool; classification_method: str; note: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic boost engine (unchanged pure-Python)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_boosts(tag: str, nutrients: dict[str, float]) -> tuple[float, list[str]]:
    rules = _NUTRIENT_BOOSTS.get(tag, [])
    total = 0.0; reasons = []
    for nutrient_key, operator, threshold, boost in rules:
        value     = nutrients.get(nutrient_key, 0.0)
        triggered = (operator == "gte" and value >= threshold) or (operator == "lte" and value <= threshold)
        if triggered:
            total += boost
            direction = "≥" if operator == "gte" else "≤"
            sign      = f"+{boost:.2f}" if boost >= 0 else f"{boost:.2f}"
            reasons.append(f"{nutrient_key.replace('_', ' ').title()} {direction}{threshold} ({value:.1f}) → {sign} score")
    return round(total, 4), reasons


# ─────────────────────────────────────────────────────────────────────────────
# Claude classification (replaces DeBERTa)
# ─────────────────────────────────────────────────────────────────────────────

_TAG_SYSTEM = (
    "You are a food classification AI. Given a food description, score each tag. "
    "Return ONLY a JSON object (no markdown): "
    '{"Dairy": 0.0-1.0, "Gluten": 0.0-1.0, "Spicy": 0.0-1.0, '
    '"Fried": 0.0-1.0, "Sugar": 0.0-1.0, "Caffeine": 0.0-1.0, "Processed Food": 0.0-1.0}'
)


def _run_claude_tags(food_description: str) -> Optional[dict[str, float]]:
    if _claude_client is None:
        return None
    try:
        msg = _claude_client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 150,
            system     = _TAG_SYSTEM,
            messages   = [{"role": "user", "content": food_description}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)
        return {k: round(float(v), 4) for k, v in data.items() if k in _TAG_HYPOTHESES}
    except Exception as exc:
        log.warning(f"Claude tag classification failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public function
# ─────────────────────────────────────────────────────────────────────────────

def classify_food_tags(
    usda_id: int,
    # Legacy params accepted but ignored (usda_df, id_col, desc_col)
    **_kwargs,
) -> FoodTagResult:
    if _usda_client is None:
        raise RuntimeError("food_tag_classifier not initialised.")

    food_data = _usda_client.get_nutrients(usda_id)
    if food_data is None:
        raise ValueError(f"USDA ID {usda_id} not found.")

    desc      = food_data.get("description", f"USDA ID {usda_id}")
    nutrients = {
        k: float(food_data[k])
        for k in ("calories", "total_fat", "sat_fat", "sodium", "sugar", "calcium")
        if food_data.get(k) is not None
    }

    nli_scores = _run_claude_tags(desc)
    method     = "nli+heuristic" if nli_scores is not None else "heuristic_only"

    tag_scores: list[TagScore] = []

    for tag in VALID_TAGS:
        if tag == "Others":
            continue
        nli   = (nli_scores or {}).get(tag, 0.0)
        boost, boost_reasons = _apply_boosts(tag, nutrients)
        final = min(1.0, max(0.0, nli + boost))

        reasons = []
        if nli_scores is not None:
            reasons.append(f"Claude NLI score: {nli:.2f}")
        reasons.extend(boost_reasons)

        tag_scores.append(TagScore(
            tag=tag, score=round(final, 4), nli_score=nli,
            heuristic_boost=boost, active=final >= _TAG_THRESHOLD, reasons=reasons,
        ))

    active = sorted([ts for ts in tag_scores if ts.active], key=lambda x: x.score, reverse=True)

    if active:
        primary_tag      = active[0].tag
        confidence       = active[0].score
        active_tag_names = [ts.tag for ts in active]
        is_others        = False
        tag_scores.append(TagScore(
            tag="Others", score=0.0, nli_score=0.0, heuristic_boost=0.0,
            active=False, reasons=["Others is the fallback — no specific tag qualified."],
        ))
    else:
        primary_tag      = "Others"
        confidence       = 1.0
        active_tag_names = ["Others"]
        is_others        = True
        tag_scores.append(TagScore(
            tag="Others", score=1.0, nli_score=0.0, heuristic_boost=0.0,
            active=True, reasons=["No specific tag cleared the confidence threshold."],
        ))

    note = (
        f"Classified using Claude AI + nutrient heuristics. "
        f"{len(active)} tag(s) active above {_TAG_THRESHOLD:.0%} threshold."
        if not is_others else
        f"No tag cleared the {_TAG_THRESHOLD:.0%} threshold. Food may be ambiguous."
    )

    return FoodTagResult(
        usda_id=usda_id, food_description=desc, primary_tag=primary_tag,
        active_tags=active_tag_names,
        tag_scores=sorted(tag_scores, key=lambda x: x.score, reverse=True),
        confidence=round(confidence, 4), is_others=is_others,
        classification_method=method, note=note,
    )
