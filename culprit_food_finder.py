"""
culprit_food_finder.py
──────────────────────
Identifies top 3–6 foods most likely responsible for a user's symptoms.

All temporal filtering, severity weighting, and aggregation logic is unchanged.
Only the cross-encoder NLI call is replaced with Claude API (batched).

Public API (unchanged):
    find_culprit_foods(food_logs, symptom_logs) -> CulpritResult
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from food_symptom_predictor import batch_nli_score

log = logging.getLogger(__name__)

_DIGESTION_WINDOW: dict[str, tuple[int, int]] = {
    "Heartburn":      (15,  180),   # 0.25–3 h
    "Acid Reflux":    (15,  180),   # 0.25–3 h
    "Bloating":       (30,  360),   # 0.5–6 h
    "Gas":            (30,  360),   # 0.5–6 h
    "Cramps":         (30,  360),   # 0.5–6 h
    "Abdominal Pain": (30,  480),   # 0.5–8 h
    "Nausea":         (30,  240),   # 0.5–4 h
    "Diarrhea":       (60,  720),   # 1–12 h
    "Constipation":   (720, 4320),  # 12–72 h
    "Fatigue":        (60,  720),   # 1–12 h
}
_DEFAULT_WINDOW    = (30, 360)
_SEVERITY_WEIGHT   = {"Severe": 1.6, "Moderate": 1.0, "Mild": 0.6}
_MIN_FOODS         = 3
_MAX_FOODS         = 6
_SCORE_RATIO_THRESHOLD = 0.40


@dataclass
class CulpritFood:
    usda_id: int; food_name: str; aggregate_score: float; raw_score: float
    occurrence_count: int; linked_symptoms: list[str]; top_symptom: str
    confidence_label: str


@dataclass
class CulpritResult:
    culprit_foods: list[CulpritFood]; method_summary: str
    symptom_events_used: int; food_events_scanned: int


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


def _confidence_label(score: float) -> str:
    return "High" if score >= 0.70 else "Moderate" if score >= 0.45 else "Low"


def find_culprit_foods(
    food_logs:     list[dict],
    symptom_logs:  list[dict],
    cross_encoder: Any = None,   # legacy param — not used
) -> CulpritResult:

    if not symptom_logs:
        return CulpritResult([], "You do not have any symptom logs.", 0, len(food_logs))
    if not food_logs:
        return CulpritResult([], "You do not have any food logs.", len(symptom_logs), 0)

    # Collect all food×symptom pairs to score in one batched Claude call
    acc: dict[tuple, dict] = defaultdict(lambda: {
        "raw_score": 0.0, "occurrence_count": 0,
        "symptom_scores": defaultdict(float),
        "food_name": "", "usda_id": 0,
    })

    # Gather all pairs first
    pending_pairs: list[tuple[tuple, str, float, str]] = []  # (food_key, symptom, sev_weight, pair_uid)
    seen_pairs: set = set()
    symptom_events_used = 0

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
            pair_uid  = (id(sym), food_key, symptom)

            a = acc[food_key]
            a["food_name"] = food_name
            a["usda_id"]   = usda_id

            if pair_uid not in seen_pairs:
                a["occurrence_count"] += 1
                seen_pairs.add(pair_uid)

            pending_pairs.append((food_key, symptom, sev_weight, food.get("quantity_g", 100), pair_uid))

    if not pending_pairs:
        return CulpritResult(
            [], "No food was eaten within digestion windows before any symptom.",
            symptom_events_used, len(food_logs),
        )

    # Build NLI premises + hypotheses for batch call
    nli_input_pairs = []
    for food_key, symptom, sev_weight, qty_g, _ in pending_pairs:
        food_name = food_key[1]
        qty_part  = f" ({round(qty_g)}g)" if qty_g else ""
        premise   = f"The user recently ate {food_name.lower()}{qty_part}."
        hypothesis = f"This food caused the user's {symptom.lower()}."
        nli_input_pairs.append((premise, hypothesis))

    nli_results = batch_nli_score(nli_input_pairs)

    # Accumulate scores
    for (food_key, symptom, sev_weight, qty_g, _), nli in zip(pending_pairs, nli_results):
        entailment_score = nli["entailment"]
        a = acc[food_key]
        a["raw_score"] += entailment_score * sev_weight
        if entailment_score > a["symptom_scores"][symptom]:
            a["symptom_scores"][symptom] = entailment_score

    if not acc:
        return CulpritResult(
            [], "No food was eaten within digestion windows before any symptom.",
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
            usda_id=a["usda_id"], food_name=a["food_name"],
            aggregate_score=round(norm, 4), raw_score=round(a["raw_score"], 4),
            occurrence_count=a["occurrence_count"],
            linked_symptoms=sorted(sym_scores.keys()),
            top_symptom=top_symptom, confidence_label=_confidence_label(norm),
        ))

    n_f, n_s = len(food_logs), len(symptom_logs)
    return CulpritResult(
        culprit_foods=culprit_foods,
        method_summary=(
            f"Analysed {n_f} food logs and {n_s} symptom logs. "
            f"{symptom_events_used} symptom event(s) had temporally matching food entries. "
        ),
        symptom_events_used=symptom_events_used,
        food_events_scanned=n_f,
    )
