"""
culprit_food_finder.py
──────────────────────
Identifies the top 3–6 foods most likely responsible for a user's
reported symptoms over a 7-day window.

Algorithm
─────────
  For each symptom event:
  1. Temporal filter  — keep foods eaten within the clinical digestion window
  2. NLI scoring      — cross-encoder/nli-deberta-v3-small rates
                        "The user ate {food}" → "This food caused {symptom}"
  3. Severity weight  — Severe 1.6×, Moderate 1.0×, Mild 0.6×
  4. Aggregate        — sum(NLI × severity) per food across all events
  5. Normalise        — 0–1 relative to top food
  6. Select 3–6       — always top 3; include rank 4–6 if score >= 40% of top
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


_DIGESTION_WINDOW: dict[str, tuple[int, int]] = {
    "Heartburn":      (15,   180),
    "Acid Reflux":    (15,   180),
    "Bloating":       (30,   480),
    "Gas":            (30,   480),
    "Nausea":         (30,   240),
    "Cramps":         (30,   480),
    "Abdominal Pain": (30,   480),
    "Diarrhea":       (60,   960),
    "Constipation":   (720, 2880),
    "Fatigue":        (60,   720),
}
_DEFAULT_WINDOW = (30, 480)

_SEVERITY_WEIGHT: dict[str, float] = {
    "Severe": 1.6, "Moderate": 1.0, "Mild": 0.6,
}

_ENTAILMENT_IDX        = 1   # cross-encoder: [contradiction, entailment, neutral]
_MIN_FOODS             = 3
_MAX_FOODS             = 6
_SCORE_RATIO_THRESHOLD = 0.40


@dataclass
class CulpritFood:
    usda_id:          int
    food_name:        str
    aggregate_score:  float      # 0–1 normalised
    raw_score:        float      # sum(NLI × severity) across all events
    occurrence_count: int        # times food fell inside a temporal window
    linked_symptoms:  list[str]  # unique symptoms this food preceded
    top_symptom:      str        # symptom it scored highest against
    confidence_label: str        # "High" | "Moderate" | "Low"


@dataclass
class CulpritResult:
    culprit_foods:       list[CulpritFood]
    method_summary:      str
    symptom_events_used: int
    food_events_scanned: int


def _parse_iso(ts) -> datetime:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(str(ts).rstrip("Z"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _foods_in_window(food_logs, symptom_time, win_min, win_max):
    result = []
    for f in food_logs:
        delta = (symptom_time - _parse_iso(f["logged_at"])).total_seconds() / 60
        if win_min <= delta <= win_max:
            result.append(f)
    return result


def _softmax(logits):
    m = max(logits)
    e = [math.exp(l - m) for l in logits]
    t = sum(e)
    return [x / t for x in e]


def _nli_score(cross_encoder, food_name, qty_g, symptom) -> float:
    qty_part   = f" ({round(qty_g)}g)" if qty_g else ""
    premise    = f"The user recently ate {food_name.lower()}{qty_part}."
    hypothesis = f"This food caused the user's {symptom.lower()}."
    logits     = cross_encoder.predict([[premise, hypothesis]])
    if hasattr(logits, "tolist"):
        logits = logits.tolist()
    if isinstance(logits[0], list):
        logits = logits[0]
    return _softmax(logits)[_ENTAILMENT_IDX]


def _confidence_label(score: float) -> str:
    return "High" if score >= 0.70 else "Moderate" if score >= 0.45 else "Low"


def find_culprit_foods(
    food_logs:     list[dict],
    symptom_logs:  list[dict],
    cross_encoder: Any,
) -> CulpritResult:
    if not symptom_logs:
        return CulpritResult([], "No symptom logs found.", 0, len(food_logs))
    if not food_logs:
        return CulpritResult([], "No food logs found.", len(symptom_logs), 0)

    acc: dict[tuple, dict] = defaultdict(lambda: {
        "raw_score": 0.0, "occurrence_count": 0,
        "symptom_scores": defaultdict(float),
        "food_name": "", "usda_id": 0,
    })

    seen_pairs:          set[tuple] = set()
    symptom_events_used: int        = 0

    for sym in symptom_logs:
        symptom    = sym.get("symptom", "")
        sym_time   = _parse_iso(sym["logged_at"])
        sev_weight = _SEVERITY_WEIGHT.get(sym.get("severity", "Moderate"), 1.0)
        win_min, win_max = _DIGESTION_WINDOW.get(symptom, _DEFAULT_WINDOW)

        candidates = _foods_in_window(food_logs, sym_time, win_min, win_max)
        if not candidates:
            continue
        symptom_events_used += 1

        for food in candidates:
            usda_id   = food.get("usda_id", 0)
            food_name = food.get("usda_description", "Unknown food")
            food_key  = (usda_id, food_name)
            pair_key  = (id(sym), food_key, symptom)

            score = _nli_score(cross_encoder, food_name, food.get("quantity_g"), symptom)

            a = acc[food_key]
            a["food_name"]  = food_name
            a["usda_id"]    = usda_id
            a["raw_score"] += score * sev_weight

            if pair_key not in seen_pairs:
                a["occurrence_count"] += 1
                seen_pairs.add(pair_key)

            if score > a["symptom_scores"][symptom]:
                a["symptom_scores"][symptom] = score

    if not acc:
        return CulpritResult(
            [], "No food eaten within digestion windows before any symptom.",
            symptom_events_used, len(food_logs),
        )

    ranked    = sorted(acc.values(), key=lambda v: v["raw_score"], reverse=True)
    top_score = ranked[0]["raw_score"]

    selected = []
    for i, a in enumerate(ranked):
        if i < _MIN_FOODS:
            selected.append(a)
        elif i < _MAX_FOODS and (a["raw_score"] / top_score) >= _SCORE_RATIO_THRESHOLD:
            selected.append(a)
        else:
            break

    culprit_foods = []
    for a in selected:
        sym_scores  = dict(a["symptom_scores"])
        top_symptom = max(sym_scores, key=sym_scores.get) if sym_scores else "Unknown"
        norm        = a["raw_score"] / top_score if top_score > 0 else 0.0
        culprit_foods.append(CulpritFood(
            usda_id          = a["usda_id"],
            food_name        = a["food_name"],
            aggregate_score  = round(norm, 4),
            raw_score        = round(a["raw_score"], 4),
            occurrence_count = a["occurrence_count"],
            linked_symptoms  = sorted(sym_scores.keys()),
            top_symptom      = top_symptom,
            confidence_label = _confidence_label(norm),
        ))

    n_f, n_s = len(food_logs), len(symptom_logs)
    return CulpritResult(
        culprit_foods       = culprit_foods,
        method_summary      = (
            f"Analysed {n_f} food logs and {n_s} symptom logs over 7 days. "
            f"{symptom_events_used} symptom event(s) had temporally matching food entries. "
            "Scored using cross-encoder/nli-deberta-v3-small entailment probability, "
            f"weighted by severity and aggregated across all events. "
            f"Returned top {len(culprit_foods)} food(s) by aggregate score."
        ),
        symptom_events_used = symptom_events_used,
        food_events_scanned = n_f,
    )