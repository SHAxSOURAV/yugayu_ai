"""
diet_symptom_risk.py
────────────────────
Predicts gut symptom probabilities from a user's stored food log history
using zero-shot NLI (MoritzLaurer/deberta-v3-base-mnli-fever-anli).

The classifier is the SAME instance already loaded by nutrition_scorer.py —
no extra RAM, no extra startup cost.

Public API
──────────
    predict_symptom_risk(food_logs, classifier) -> list[dict]

Each returned dict:
    {
        "symptom":     str,
        "probability": float,   # 0–1 independent sigmoid score
        "risk_level":  str,     # "high" | "medium" | "low"
    }

Sorted highest probability first.
"""

from __future__ import annotations

from typing import Any


# ── Symptom taxonomy — must match _VALID_SYMPTOMS in main.py ─────────────────
SYMPTOM_LABELS = [
    "Bloating",
    "Gas",
    "Constipation",
    "Diarrhea",
    "Nausea",
    "Heartburn",
    "Abdominal Pain",
    "Cramps",
    "Fatigue",
    "Acid Reflux",
]

# Hypothesis template: each label is substituted into {}
# Framed as a clinical NLI hypothesis the model scores against the diet premise
_HYPOTHESIS_TEMPLATE = "This diet is likely to cause {}."

# Risk bucketing
_RISK_THRESHOLDS: list[tuple[str, float]] = [
    ("high",   0.55),
    ("medium", 0.35),
    ("low",    0.0),
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_diet_summary(food_logs: list[dict]) -> str:
    """
    Build a natural-language diet description from MongoDB food log dicts.

    Expected keys per log entry (as stored by database.py):
        usda_description : str   (e.g. "CHICKEN,BRST,CKD,RSTD")
        quantity_g       : float
        meal_type        : str   (Breakfast / Lunch / Dinner / Snack)
        logged_at        : str   (ISO timestamp — used for ordering, not displayed)

    Falls back gracefully when optional keys are missing.
    """
    if not food_logs:
        return "The user has not logged any food recently."

    # Group by meal_type for a more natural sentence
    meal_groups: dict[str, list[str]] = {}
    for entry in food_logs:
        meal = (entry.get("meal_type") or "meal").strip().lower()
        desc = entry.get("usda_description") or "unknown food"
        qty  = entry.get("quantity_g")

        # Humanise the USDA description: "CHICKEN,BRST,CKD,RSTD" → "chicken breast"
        readable = _humanise(desc)

        portion = f"{round(qty)}g {readable}" if qty else readable
        meal_groups.setdefault(meal, []).append(portion)

    parts = [f"{meal}: {', '.join(items)}" for meal, items in meal_groups.items()]
    return "The user recently consumed — " + "; ".join(parts) + "."


def _humanise(usda_desc: str) -> str:
    """
    Convert USDA ALL-CAPS comma-separated description to a readable name.
    e.g. "CHICKEN,BRST,CKD,RSTD" → "chicken brst ckd rstd"
    e.g. "BUTTER,WITH SALT"       → "butter with salt"
    """
    return usda_desc.replace(",", " ").lower().strip()


def _bucket_risk(probability: float) -> str:
    for level, threshold in _RISK_THRESHOLDS:
        if probability >= threshold:
            return level
    return "low"


# ── Main function ─────────────────────────────────────────────────────────────

def predict_symptom_risk(
    food_logs:  list[dict],
    classifier: Any,   # the transformers zero-shot-classification pipeline
) -> tuple[str, list[dict]]:
    """
    Predict symptom probabilities from MongoDB food log records.

    Parameters
    ----------
    food_logs   : list of dicts from db.get_food_logs() — may be empty
    classifier  : the zero-shot-classification pipeline from nutrition_scorer._classifier

    Returns
    -------
    (diet_summary, predictions)

    predictions is a list of dicts sorted by probability (descending):
        [{"symptom": str, "probability": float, "risk_level": str}, ...]
    """
    diet_summary = _build_diet_summary(food_logs)

    if not food_logs:
        # No logs — return neutral 0.0 scores for all symptoms
        return diet_summary, [
            {"symptom": s, "probability": 0.0, "risk_level": "low"}
            for s in SYMPTOM_LABELS
        ]

    # multi_label=True → independent sigmoid per label (not softmax).
    # This is correct: multiple symptoms can have elevated risk simultaneously.
    result = classifier(
        diet_summary,
        candidate_labels    = SYMPTOM_LABELS,
        hypothesis_template = _HYPOTHESIS_TEMPLATE,
        multi_label         = True,
    )

    # result["labels"] and result["scores"] are parallel lists,
    # already sorted descending by score by the pipeline.
    predictions = [
        {
            "symptom":     label,
            "probability": round(float(score), 4),
            "risk_level":  _bucket_risk(float(score)),
        }
        for label, score in zip(result["labels"], result["scores"])
    ]

    return diet_summary, predictions