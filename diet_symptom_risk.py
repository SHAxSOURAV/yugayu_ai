"""
diet_symptom_risk.py
────────────────────
Predicts gut symptom probabilities from a user's food log history.

Old approach: DeBERTa zero-shot NLI pipeline (requires loaded model).
New approach: Claude API (one batched call for all 10 symptoms).

Public API (unchanged):
    predict_symptom_risk(food_logs) -> (diet_summary, list[dict])
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

_claude_client = None


def init(claude_client) -> None:
    global _claude_client
    _claude_client = claude_client
    log.info("diet_symptom_risk: Claude client ready.")


SYMPTOM_LABELS = [
    "Bloating", "Gas", "Constipation", "Diarrhea", "Nausea",
    "Heartburn", "Abdominal Pain", "Cramps", "Fatigue", "Acid Reflux",
]

_RISK_THRESHOLDS: list[tuple[str, float]] = [
    ("high",   0.55),
    ("medium", 0.35),
    ("low",    0.0),
]


def _build_diet_summary(food_logs: list[dict]) -> str:
    if not food_logs:
        return "The user has not logged any food recently."
    meal_groups: dict[str, list[str]] = {}
    for entry in food_logs:
        meal    = (entry.get("meal_type") or "meal").strip().lower()
        desc    = entry.get("usda_description") or "unknown food"
        qty     = entry.get("quantity_g")
        readable = desc.replace(",", " ").lower().strip()
        portion  = f"{round(qty)}g {readable}" if qty else readable
        meal_groups.setdefault(meal, []).append(portion)
    parts = [f"{meal}: {', '.join(items)}" for meal, items in meal_groups.items()]
    return "The user recently consumed — " + "; ".join(parts) + "."


def _bucket_risk(probability: float) -> str:
    for level, threshold in _RISK_THRESHOLDS:
        if probability >= threshold:
            return level
    return "low"


_RISK_SYSTEM = (
    "You are a clinical gut-health AI. Given a diet summary, estimate the probability "
    "(0.0–1.0) that each gut symptom will occur. "
    "Return ONLY a JSON object (no markdown) with each symptom as a key and its "
    "probability as the value. Symptoms: "
    "Bloating, Gas, Constipation, Diarrhea, Nausea, Heartburn, Abdominal Pain, Cramps, Fatigue, Acid Reflux."
)


def predict_symptom_risk(
    food_logs:  list[dict],
    classifier: Any = None,   # kept for API compatibility — not used
) -> tuple[str, list[dict]]:
    """
    Predict symptom probabilities from MongoDB food log records.
    Returns (diet_summary, predictions_list).
    """
    diet_summary = _build_diet_summary(food_logs)

    if not food_logs:
        return diet_summary, [
            {"symptom": s, "probability": 0.0, "risk_level": "low"}
            for s in SYMPTOM_LABELS
        ]

    if _claude_client is None:
        log.warning("diet_symptom_risk: Claude client not initialised. Returning neutral.")
        return diet_summary, [
            {"symptom": s, "probability": 0.0, "risk_level": "low"}
            for s in SYMPTOM_LABELS
        ]

    try:
        msg = _claude_client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 180,
            system     = _RISK_SYSTEM,
            messages   = [{"role": "user", "content": diet_summary}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)
    except Exception as exc:
        log.warning(f"Claude diet symptom risk failed: {exc}")
        return diet_summary, [
            {"symptom": s, "probability": 0.0, "risk_level": "low"}
            for s in SYMPTOM_LABELS
        ]

    predictions = []
    for label in SYMPTOM_LABELS:
        prob = float(data.get(label, 0.0))
        prob = max(0.0, min(1.0, prob))
        predictions.append({
            "symptom":     label,
            "probability": round(prob, 4),
            "risk_level":  _bucket_risk(prob),
        })

    predictions.sort(key=lambda x: x["probability"], reverse=True)
    return diet_summary, predictions
