"""
food_symptom_predictor.py
─────────────────────────
Predicts which logged food caused which symptom.

All pure-Python logic (temporal windows, nutrient risk math, Bayesian blending)
is unchanged. Only the cross-encoder NLI model is replaced with Claude API.
Claude is called once per predict request with ALL food×symptom pairs batched
into a single message — one round-trip regardless of how many pairs exist.

Public API (unchanged):
    predict_food_symptom_causes(food_logs, symptom_logs, user_memory)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

_claude_client = None
_usda_client   = None


def init(claude_client, usda_client) -> None:
    global _claude_client, _usda_client
    _claude_client = claude_client
    _usda_client   = usda_client
    log.info("food_symptom_predictor: Claude + USDA client ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Clinical digestion windows (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

SYMPTOM_WINDOWS: dict[str, tuple[float, float]] = {
    "Heartburn":      (0.25, 3.0),
    "Acid Reflux":    (0.25, 3.0),
    "Nausea":         (0.5,  4.0),
    "Bloating":       (0.5,  8.0),
    "Gas":            (1.0,  8.0),
    "Cramps":         (0.5,  6.0),
    "Abdominal Pain": (0.5,  8.0),
    "Diarrhea":       (1.0, 16.0),
    "Constipation":   (12.0,48.0),
    "Fatigue":        (1.0, 12.0),
}

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

_NUTRIENT_RISK_THRESHOLDS: dict[str, float] = {
    "total_fat": 10.0, "sat_fat": 4.0, "sodium": 300.0, "sugar": 8.0,
    "carbs": 30.0, "cholesterol": 60.0, "calories": 250.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes (unchanged)
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
    portion_g:   float = 100.0


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
    personal_prior: float = 0.5; prior_confidence: float = 0.0
    prior_observations: int = 0; prior_confirmations: int = 0
    personalisation_weight: float = 0.0; model_weight: float = 1.0
    explanation: str = ""; top_risk_nutrients: list[str] = field(default_factory=list)


@dataclass
class SymptomPrediction:
    symptom: str; intensity: str; symptom_logged_at: str; digestion_window: str
    foods_in_window: int; foods_outside_window: int
    top_cause: Optional[FoodCausationResult]; all_candidates: list[FoodCausationResult]
    no_foods_found: bool = False; note: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (unchanged pure-Python)
# ─────────────────────────────────────────────────────────────────────────────

def _get_nutrient_snapshot(usda_id: int, quantity_g: float) -> NutrientSnapshot:
    if _usda_client is None:
        return NutrientSnapshot(description=f"USDA ID {usda_id}", portion_g=quantity_g)
    data = _usda_client.get_nutrients(usda_id)
    if data is None:
        return NutrientSnapshot(description=f"USDA ID {usda_id}", portion_g=quantity_g)
    scale = quantity_g / 100.0
    def sv(k): v = data.get(k); return round(float(v)*scale, 3) if v is not None else None
    return NutrientSnapshot(
        description=data.get("description", f"USDA ID {usda_id}"),
        calories=sv("calories"), protein=sv("protein"), total_fat=sv("total_fat"),
        carbs=sv("carbs"), sodium=sv("sodium"), sat_fat=sv("sat_fat"),
        cholesterol=sv("cholesterol"), sugar=sv("sugar"), portion_g=quantity_g,
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


_SYMPTOM_PHRASES = {
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


def _build_hypothesis(symptom: str, intensity: str) -> str:
    phrase = _SYMPTOM_PHRASES.get(symptom, f"caused {symptom.lower()}")
    adv    = {"Mild": "mild ", "Moderate": "moderate ", "Severe": "severe "}.get(intensity, "")
    return f"Eating this food {phrase} ({adv}intensity)."


def _nutrient_risk(nutrients: NutrientSnapshot, symptom: str) -> tuple[float, list[str]]:
    risk_weights = SYMPTOM_NUTRIENT_RISK.get(symptom, {})
    scale        = nutrients.portion_g / 100.0
    val_map = {
        "total_fat": nutrients.total_fat, "sat_fat": nutrients.sat_fat,
        "sodium": nutrients.sodium, "sugar": nutrients.sugar,
        "carbs": nutrients.carbs, "cholesterol": nutrients.cholesterol,
        "calories": nutrients.calories,
    }
    total_risk   = 0.0
    total_weight = sum(risk_weights.values()) or 1.0
    risk_notes:  list[str] = []
    for nutrient, weight in risk_weights.items():
        val = val_map.get(nutrient)
        if val is None:
            continue
        threshold    = _NUTRIENT_RISK_THRESHOLDS.get(nutrient, 1.0) * scale
        ratio        = min(val / threshold, 2.0)
        total_risk  += ratio * weight
        if ratio >= 1.0:
            unit = "mg" if nutrient in ("sodium", "cholesterol") else ("kcal" if nutrient == "calories" else "g")
            risk_notes.append(f"{nutrient.replace('_',' ').title()} {val:.1f}{unit} — {ratio:.1f}× above threshold for {symptom}")
    return round(min(total_risk / total_weight, 1.0), 4), risk_notes[:3]


def _temporal_score(hours_before: float, window_min: float, window_max: float) -> float:
    if hours_before < window_min or hours_before > window_max:
        return 0.0
    span    = window_max - window_min
    centre  = window_min + span / 2.0
    dist    = abs(hours_before - centre)
    return round(1.0 - (dist / (span / 2.0)) * 0.5, 4)


def _quantity_score(portion_g: float) -> float:
    return round(min(portion_g / (portion_g + 150.0), 1.0), 4)


def _causation_label(score: float) -> tuple[str, str]:
    if   score >= 0.70: return "Likely",   f"{int(score*100)}%"
    elif score >= 0.45: return "Possible", f"{int(score*100)}%"
    else:               return "Unlikely", f"{int(score*100)}%"


# ─────────────────────────────────────────────────────────────────────────────
# Claude batched NLI (replaces cross-encoder)
# ─────────────────────────────────────────────────────────────────────────────

_NLI_SYSTEM = (
    "You are a clinical NLI (Natural Language Inference) model specialised in "
    "food-symptom causation. For each numbered premise-hypothesis pair, return "
    "entailment/neutral/contradiction probabilities that sum to 1.0. "
    "Return ONLY a JSON array in the same order: "
    '[{"entailment": 0.XX, "neutral": 0.XX, "contradiction": 0.XX}, ...]'
)


def batch_nli_score(pairs: list[tuple[str, str]]) -> list[dict]:
    """
    Score multiple (premise, hypothesis) pairs in one Claude call.
    Returns list of {"entailment", "neutral", "contradiction"} dicts.
    Falls back to neutral 0.33 if Claude call fails.
    """
    if not pairs:
        return []

    fallback = [{"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33}] * len(pairs)

    if _claude_client is None:
        return fallback

    lines = [f"{i+1}. Premise: \"{p}\" Hypothesis: \"{h}\"" for i, (p, h) in enumerate(pairs)]
    user_msg = "Score these food-symptom pairs:\n" + "\n".join(lines)

    try:
        msg = _claude_client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = max(80 * len(pairs), 200),
            system     = _NLI_SYSTEM,
            messages   = [{"role": "user", "content": user_msg}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        results = json.loads(raw)
        if len(results) != len(pairs):
            log.warning("Claude NLI returned wrong number of results; using fallback.")
            return fallback
        # Normalise each to sum=1
        out = []
        for r in results:
            e = max(0.0, float(r.get("entailment",    0.33)))
            n = max(0.0, float(r.get("neutral",       0.34)))
            c = max(0.0, float(r.get("contradiction", 0.33)))
            total = e + n + c or 1.0
            out.append({
                "entailment":    round(e / total, 4),
                "neutral":       round(n / total, 4),
                "contradiction": round(c / total, 4),
            })
        return out
    except Exception as exc:
        log.warning(f"Claude batched NLI failed: {exc}")
        return fallback


# Kept for import compatibility with meal_symptom_forecast.py
def _nli_score(premise: str, hypothesis: str) -> dict:
    return batch_nli_score([(premise, hypothesis)])[0]


def _find_col(df, key):        return None  # no-op; kept for import compat
def _get_nutrient(row, df, key, scale=1.0): return None  # no-op; kept for import compat


# ─────────────────────────────────────────────────────────────────────────────
# Public function
# ─────────────────────────────────────────────────────────────────────────────

def predict_food_symptom_causes(
    food_logs:    list[FoodLogEntry],
    symptom_logs: list[SymptomLogEntry],
    user_memory   = None,
    # Legacy params accepted but ignored (usda_df, id_col, desc_col)
    **_kwargs,
) -> list[SymptomPrediction]:

    predictions: list[SymptomPrediction] = []

    for sym_entry in symptom_logs:
        symptom  = sym_entry.symptom
        sym_time = sym_entry.logged_at
        intensity = sym_entry.intensity
        win_min, win_max = SYMPTOM_WINDOWS.get(symptom, (0.5, 8.0))

        # Stage 1: temporal filter
        candidates: list[tuple[FoodLogEntry, float]] = []
        outside_count = 0

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
                    f"No food logs found within the {win_min}–{win_max}h digestion window. "
                    f"{outside_count} food(s) were outside the window."
                ),
            ))
            continue

        # Fetch nutrients for all candidates
        nutrient_cache: dict[int, NutrientSnapshot] = {}
        for food_entry, _ in candidates:
            if food_entry.usda_id not in nutrient_cache:
                nutrient_cache[food_entry.usda_id] = _get_nutrient_snapshot(
                    food_entry.usda_id, food_entry.quantity_g,
                )

        # Build NLI pairs for batch call
        nli_pairs = []
        for food_entry, _ in candidates:
            nuts = nutrient_cache[food_entry.usda_id]
            premise    = _build_premise(nuts.description, nuts)
            hypothesis = _build_hypothesis(symptom, intensity)
            nli_pairs.append((premise, hypothesis))

        nli_results = batch_nli_score(nli_pairs)

        results: list[FoodCausationResult] = []
        for (food_entry, hours_before), nli_scores in zip(candidates, nli_results):
            nuts   = nutrient_cache[food_entry.usda_id]
            nli_ent = nli_scores["entailment"]

            n_risk, risk_notes = _nutrient_risk(nuts, symptom)
            t_score = _temporal_score(hours_before, win_min, win_max)
            q_score = _quantity_score(food_entry.quantity_g)

            base_causation = round(min(
                0.35 * nli_ent + 0.35 * n_risk + 0.20 * t_score + 0.10 * q_score, 1.0,
            ), 4)

            personal_prior = 0.5; prior_conf = 0.0; prior_obs = 0
            prior_conf_int = 0;   p_weight   = 0.0; m_weight = 1.0

            if user_memory is not None:
                blended, breakdown = user_memory.blend_score(
                    usda_id=food_entry.usda_id, symptom=symptom, nli_score=nli_ent,
                )
                causation      = round(min(
                    0.35 * blended + 0.35 * n_risk + 0.20 * t_score + 0.10 * q_score, 1.0,
                ), 4)
                personal_prior = breakdown["personal_prior"]
                prior_conf     = breakdown["prior_confidence"]
                prior_obs      = breakdown["prior_observations"]
                prior_conf_int = breakdown["prior_confirmations"]
                p_weight       = breakdown["personalisation_weight"]
                m_weight       = breakdown["model_weight"]
            else:
                causation = base_causation

            c_label, c_pct = _causation_label(causation)

            explanation = (
                f"{nuts.description} eaten {hours_before:.1f}h before {symptom} — "
                f"{c_label}. NLI: {nli_ent:.0%} | Nutrient risk: {n_risk:.0%} | "
                f"Temporal fit: {t_score:.0%} | Portion: {q_score:.0%}."
            )
            if risk_notes:
                explanation += " Key risk nutrients: " + "; ".join(risk_notes[:2]) + "."

            results.append(FoodCausationResult(
                usda_id=food_entry.usda_id, food_name=nuts.description,
                quantity_g=food_entry.quantity_g, logged_at=food_entry.logged_at.isoformat(),
                hours_before=round(hours_before, 2),
                entailment_score=nli_ent,
                contradiction_score=nli_scores["contradiction"],
                neutral_score=nli_scores["neutral"],
                nutrient_risk_score=n_risk, temporal_score=t_score, quantity_score=q_score,
                causation_score=causation, causation_label=c_label, causation_pct=c_pct,
                personal_prior=personal_prior, prior_confidence=prior_conf,
                prior_observations=prior_obs, prior_confirmations=prior_conf_int,
                personalisation_weight=p_weight, model_weight=m_weight,
                top_risk_nutrients=risk_notes, explanation=explanation,
            ))

        results.sort(key=lambda r: r.causation_score, reverse=True)

        predictions.append(SymptomPrediction(
            symptom=symptom, intensity=intensity,
            symptom_logged_at=sym_time.isoformat(),
            digestion_window=f"{win_min}h – {win_max}h before symptom",
            foods_in_window=len(results), foods_outside_window=outside_count,
            top_cause=results[0] if results else None, all_candidates=results,
        ))

    return predictions

import re as _re
from datetime import datetime, timezone
from typing import Optional
 
 
# ─────────────────────────────────────────────────────────────────────────────
# System prompt — teaches Claude the digestion window rules and output format
# ─────────────────────────────────────────────────────────────────────────────
 
_TEMPORAL_SYSTEM = """\
You are a clinical gut-health AI analysing a food and symptom timeline.
 
━━━ DIGESTION WINDOWS ━━━
Each symptom has a known time window after eating during which food can cause it:
 
  Heartburn        →  15 min – 3 h
  Acid Reflux      →  15 min – 3 h
  Nausea           →  30 min – 4 h
  Cramps           →  30 min – 6 h
  Bloating         →  30 min – 8 h
  Gas              →  30 min – 8 h
  Abdominal Pain   →  30 min – 8 h
  Diarrhea         →  1 h – 16 h
  Fatigue          →  1 h – 12 h
  Constipation     →  12 h – 48 h
 
━━━ YOUR TASK ━━━
1. For every symptom in the log, look backwards in time from when it was
   reported and find all foods that were eaten within the symptom's window.
2. From those candidate foods, identify which ones most plausibly CAUSED
   the symptom, based on:
     • How close the timing is to the centre of the window (ideal match → stronger candidate)
     • Whether the food is commonly known to trigger that symptom
     • If a food appears before MULTIPLE instances of the same symptom, it
       is a stronger candidate — increase its confidence.
3. A single symptom log entry may have multiple trigger foods — include ALL
   plausible ones, not just the top one.
4. If NO food falls within the digestion window for a symptom, return an
   empty list for that symptom.
 
━━━ STRICT RULES ━━━
• Use food names EXACTLY as provided — do NOT rename, translate, or
  normalise them.  "Chicken" stays "Chicken", "Rice" stays "Rice".
• Only include symptoms that appear in the symptom log.
• Do not invent symptoms or foods.
 
━━━ OUTPUT FORMAT ━━━
Return ONLY a valid JSON object — no markdown, no explanation:
{
  "Symptom Name": ["Food A", "Food B"],
  "Symptom Name": ["Food C"],
  ...
}
If a symptom had no foods in its window, still include it with [].
"""
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _parse_dt(ts) -> datetime:
    """Parse ISO string or datetime to UTC-aware datetime."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(str(ts).rstrip("Z"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
 
 
def _fmt_time(dt: datetime) -> str:
    """Format as 'Apr 03 08:47' for the prompt timeline."""
    return dt.strftime("%b %d %H:%M")
 
 
def _build_timeline_prompt(
    food_logs_named: list[dict],
    symptom_logs:    list[dict],
) -> str:
    """
    Build a compact, chronologically sorted timeline string for Claude.
    Each food log dict: {food_name, weight_g, logged_at}
    Each symptom log dict: {symptom, intensity, logged_at}
    """
    # Sort both by time
    foods = sorted(food_logs_named, key=lambda x: _parse_dt(x["logged_at"]))
    syms  = sorted(symptom_logs,    key=lambda x: _parse_dt(x["logged_at"]))
 
    lines = ["=== FOOD LOG ==="]
    for f in foods:
        dt       = _parse_dt(f["logged_at"])
        weight   = f.get("weight_g") or f.get("quantity_g") or ""
        wt_part  = f" ({round(float(weight))}g)" if weight else ""
        lines.append(f"  {_fmt_time(dt)}  |  {f['food_name']}{wt_part}")
 
    lines.append("")
    lines.append("=== SYMPTOM LOG ===")
    for s in syms:
        dt        = _parse_dt(s["logged_at"])
        intensity = s.get("intensity", "")
        int_part  = f" [{intensity}]" if intensity else ""
        lines.append(f"  {_fmt_time(dt)}  |  {s['symptom']}{int_part}")
 
    return "\n".join(lines)
 
 
def _safe_json_parse(raw: str) -> Optional[dict]:
    """Strip markdown fences and parse JSON, returning None on failure."""
    cleaned = _re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        return json.loads(cleaned)
    except Exception:
        return None
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Public function
# ─────────────────────────────────────────────────────────────────────────────
 
def predict_causation_by_time(
    food_logs_named: list[dict],
    symptom_logs:    list[dict],
) -> dict[str, list[str]]:
    """
    Claude temporal causation analysis.
 
    Parameters
    ----------
    food_logs_named : list of dicts with keys:
        food_name  (str)   — short display name, already converted from usda_id
        weight_g   (float) — portion size in grams
        logged_at  (str | datetime)
    symptom_logs : list of dicts with keys:
        symptom    (str)
        intensity  (str)   — Mild | Moderate | Severe
        logged_at  (str | datetime)
 
    Returns
    -------
    dict[str, list[str]]
        {symptom_name: [food_name, ...]}
        Every symptom in symptom_logs appears as a key.
        Food names are returned EXACTLY as provided — Claude is instructed
        not to rename them.
        Empty list means no food fell within the digestion window.
 
    Fallback
    --------
    If Claude is unavailable or returns malformed JSON, returns a fallback
    dict with every symptom mapped to all candidate foods within its
    digestion window using a simple rule-based filter.
    """
    # Collect all unique symptom names for guaranteed coverage in output
    unique_symptoms = list(dict.fromkeys(
        s.get("symptom", "") for s in symptom_logs if s.get("symptom")
    ))
 
    # ── Rule-based fallback (no Claude needed) ────────────────────────────────
    def _rule_based_fallback() -> dict[str, list[str]]:
        result: dict[str, list[str]] = {sym: [] for sym in unique_symptoms}
        for s in symptom_logs:
            symptom   = s.get("symptom", "")
            if not symptom:
                continue
            sym_time  = _parse_dt(s["logged_at"])
            win_min_h, win_max_h = {
                "Heartburn":      (0.25, 3.0),
                "Acid Reflux":    (0.25, 3.0),
                "Nausea":         (0.5,  4.0),
                "Cramps":         (0.5,  6.0),
                "Bloating":       (0.5,  8.0),
                "Gas":            (0.5,  8.0),
                "Abdominal Pain": (0.5,  8.0),
                "Diarrhea":       (1.0,  16.0),
                "Fatigue":        (1.0,  12.0),
                "Constipation":   (12.0, 48.0),
            }.get(symptom, (0.5, 8.0))
 
            for f in food_logs_named:
                food_time = _parse_dt(f["logged_at"])
                if food_time >= sym_time:
                    continue
                hours = (sym_time - food_time).total_seconds() / 3600.0
                if win_min_h <= hours <= win_max_h:
                    name = f.get("food_name", "")
                    if name and name not in result[symptom]:
                        result[symptom].append(name)
        return result
 
    # Early return if Claude unavailable
    if _claude_client is None:
        log.warning("predict_causation_by_time: Claude client not initialised; using rule-based fallback.")
        return _rule_based_fallback()
 
    if not food_logs_named or not symptom_logs:
        return {sym: [] for sym in unique_symptoms}
 
    timeline_text = _build_timeline_prompt(food_logs_named, symptom_logs)
 
    try:
        msg = _claude_client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = max(200, 60 * len(unique_symptoms) + 80 * len(food_logs_named)),
            system     = _TEMPORAL_SYSTEM,
            messages   = [{"role": "user", "content": timeline_text}],
        )
        raw    = msg.content[0].text.strip()
        parsed = _safe_json_parse(raw)
 
        if not isinstance(parsed, dict):
            log.warning("predict_causation_by_time: Claude returned non-dict; using rule-based fallback.")
            return _rule_based_fallback()
 
        # Guarantee every symptom in the request appears in the output,
        # and deduplicate food names per symptom
        output: dict[str, list[str]] = {}
        for sym in unique_symptoms:
            raw_foods = parsed.get(sym, [])
            # Deduplicate while preserving order
            seen: set[str] = set()
            clean: list[str] = []
            for fn in raw_foods:
                if isinstance(fn, str) and fn and fn not in seen:
                    seen.add(fn)
                    clean.append(fn)
            output[sym] = clean
 
        log.info(
            f"predict_causation_by_time: Claude identified triggers for "
            f"{sum(bool(v) for v in output.values())}/{len(unique_symptoms)} symptoms "
            f"from {len(food_logs_named)} food logs."
        )
        return output
 
    except Exception as exc:
        log.warning(f"predict_causation_by_time: Claude call failed ({exc}); using rule-based fallback.")
        return _rule_based_fallback()
 

  
import re as _re
from collections import defaultdict as _defaultdict
from datetime import datetime, timezone
from typing import Optional
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1.  Composite meal grouping  (shared utility)
# ─────────────────────────────────────────────────────────────────────────────
 
def group_composite_meals(food_logs_named: list[dict]) -> list[dict]:
    """
    Group food-log entries that share the same logged_at timestamp into a
    single composite meal entry.
 
    Why: composite/ethnic dishes are decomposed by Claude into individual USDA
    components that all carry the same logged_at.  Keeping them separate
    floods the timeline with redundant entries and confuses temporal analysis.
 
    Parameters
    ----------
    food_logs_named : list of dicts, each with:
        food_name  (str)
        weight_g   (float)
        logged_at  (str | datetime)
 
    Returns
    -------
    list of dicts sorted chronologically, each with:
        food_name     (str)   — single name or "A + B + C" for composites
        weight_g      (float) — individual weight or summed total
        logged_at     — original value from the first entry in the group
        is_composite  (bool)
        components    (list[str]) — individual food names in the group
    """
    def _ts_key(entry: dict) -> str:
        val = entry.get("logged_at")
        if isinstance(val, datetime):
            return val.isoformat()
        return str(val)
 
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
            total_w = sum(
                float(f.get("weight_g") or f.get("quantity_g") or 0) for f in foods
            )
            # Deduplicate component names while preserving insertion order
            names = list(dict.fromkeys(f["food_name"] for f in foods))
            result.append({
                "food_name":    " + ".join(names),
                "weight_g":     round(total_w, 1),
                "logged_at":    foods[0]["logged_at"],
                "is_composite": True,
                "components":   names,
            })
 
    return result
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2.  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _parse_dt(ts) -> datetime:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(str(ts).rstrip("Z"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
 
 
def _fmt_time(dt: datetime) -> str:
    return dt.strftime("%b %d %H:%M")
 
 
def _build_timeline_prompt(
    grouped_meals: list[dict],
    symptom_logs:  list[dict],
) -> str:
    """Compact chronological timeline string for Claude."""
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
 
 
def _safe_parse(raw: str) -> Optional[dict]:
    cleaned = _re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        return json.loads(cleaned)
    except Exception:
        return None
 
 
_DIGESTION_WINDOWS_H = {
    "Heartburn":      (0.25, 3.0),  "Acid Reflux":    (0.25, 3.0),
    "Nausea":         (0.5,  4.0),  "Cramps":         (0.5,  6.0),
    "Bloating":       (0.5,  8.0),  "Gas":            (0.5,  8.0),
    "Abdominal Pain": (0.5,  8.0),  "Diarrhea":       (1.0,  16.0),
    "Fatigue":        (1.0,  12.0), "Constipation":   (12.0, 48.0),
}
_DEFAULT_WINDOW_H = (0.5, 8.0)
 
 
def _rule_based_fallback(
    grouped_meals:   list[dict],
    symptom_logs:    list[dict],
    unique_symptoms: list[str],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {sym: [] for sym in unique_symptoms}
    for s in symptom_logs:
        symptom = s.get("symptom", "")
        if not symptom:
            continue
        sym_time       = _parse_dt(s["logged_at"])
        wmin, wmax     = _DIGESTION_WINDOWS_H.get(symptom, _DEFAULT_WINDOW_H)
        for m in grouped_meals:
            ft = _parse_dt(m["logged_at"])
            if ft >= sym_time:
                continue
            h = (sym_time - ft).total_seconds() / 3600.0
            if wmin <= h <= wmax:
                name = m["food_name"]
                if name and name not in result.get(symptom, []):
                    result.setdefault(symptom, []).append(name)
    return result
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3.  System prompt
# ─────────────────────────────────────────────────────────────────────────────
 
_TEMPORAL_SYSTEM = """\
You are a clinical gut-health AI analysing a food and symptom timeline.
 
━━━ DIGESTION WINDOWS ━━━
  Heartburn / Acid Reflux  →  15 min – 3 h
  Nausea                   →  30 min – 4 h
  Cramps                   →  30 min – 6 h
  Bloating / Gas / Abdominal Pain  →  30 min – 8 h
  Diarrhea                 →  1 h – 16 h
  Fatigue                  →  1 h – 12 h
  Constipation             →  12 h – 48 h
  Any other symptom        →  default 30 min – 8 h
 
━━━ COMPOSITE MEALS ━━━
Entries marked [MEAL] are composite dishes logged at the same time
(e.g. chicken + rice + oil). Treat the entire [MEAL] as ONE food event.
 
━━━ TASK ━━━
For each symptom in the log:
  1. Look backwards from its timestamp and find all foods / meals eaten
     within the symptom's digestion window.
  2. From those candidates, pick the most plausible triggers based on:
       • Timing closeness to the centre of the window
       • Frequency — food eaten before the SAME symptom on multiple days
       • Clinical likelihood (e.g. high-fat foods → Heartburn)
  3. Include ALL plausible triggers per symptom.
  4. Return [] if no food falls within the window for a symptom.
 
━━━ RULES ━━━
• Use food / meal names EXACTLY as shown — do NOT rename or shorten them.
• Only return symptoms that appear in the symptom log.
• Return ONLY valid JSON — no markdown:
 
{
  "Symptom Name": ["Food A", "Food B"],
  "Symptom Name": ["Food C"],
  "Symptom Name": []
}
"""
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4.  Public function
# ─────────────────────────────────────────────────────────────────────────────
 
def predict_causation_by_time(
    food_logs_named: list[dict],
    symptom_logs:    list[dict],
) -> dict[str, list[str]]:
    """
    Claude temporal causation analysis.
 
    Parameters
    ----------
    food_logs_named : list[dict]  — {food_name, weight_g, logged_at}
        usda_id must be converted to food_name before calling this.
    symptom_logs : list[dict]     — {symptom, intensity, logged_at}
 
    Returns
    -------
    dict[str, list[str]]
        {symptom: [food_or_meal_name, ...]}
        Every unique symptom in symptom_logs appears as a key.
        Composite meals are returned as "A + B + C" strings.
        Falls back to rule-based windowing if Claude is unavailable.
    """
    unique_symptoms = list(dict.fromkeys(
        s.get("symptom", "") for s in symptom_logs if s.get("symptom")
    ))
    grouped = group_composite_meals(food_logs_named)
 
    if _claude_client is None:
        log.warning("predict_causation_by_time: Claude unavailable — rule-based fallback.")
        return _rule_based_fallback(grouped, symptom_logs, unique_symptoms)
 
    if not food_logs_named or not symptom_logs:
        return {sym: [] for sym in unique_symptoms}
 
    prompt = _build_timeline_prompt(grouped, symptom_logs)
 
    try:
        msg = _claude_client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = max(300, 80 * len(unique_symptoms) + 60 * len(grouped)),
            system     = _TEMPORAL_SYSTEM,
            messages   = [{"role": "user", "content": prompt}],
        )
        parsed = _safe_parse(msg.content[0].text.strip())
 
        if not isinstance(parsed, dict):
            log.warning("predict_causation_by_time: non-dict response — rule-based fallback.")
            return _rule_based_fallback(grouped, symptom_logs, unique_symptoms)
 
        output: dict[str, list[str]] = {}
        for sym in unique_symptoms:
            raw_foods = parsed.get(sym, [])
            seen: set[str] = set()
            clean: list[str] = []
            for fn in raw_foods:
                if isinstance(fn, str) and fn and fn not in seen:
                    seen.add(fn)
                    clean.append(fn)
            output[sym] = clean
 
        log.info(
            f"predict_causation_by_time: triggers found for "
            f"{sum(bool(v) for v in output.values())}/{len(unique_symptoms)} symptoms "
            f"from {len(grouped)} meal events."
        )
        return output
 
    except Exception as exc:
        log.warning(f"predict_causation_by_time: Claude failed ({exc}) — rule-based fallback.")
        return _rule_based_fallback(grouped, symptom_logs, unique_symptoms)
 