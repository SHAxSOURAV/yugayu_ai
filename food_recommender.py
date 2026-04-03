"""
food_recommender.py
───────────────────
Personalised safe food recommendation engine.

Old approach: Flan-T5 + cosine similarity over full USDA dataset (3000+ row scan).
New approach: Claude API generates 5 personalised safe food recommendations
             in one call, given the user's eating history and symptom profile.

Response shape (SafeFoodRecommendation) is identical — no API breaking change.

Public function:
    recommend_safe_foods(eaten_usda_ids, user_symptoms, user_memory, n) -> list[SafeFoodRecommendation]
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
    log.info("food_recommender: Claude + USDA client ready.")


# ── Symptom → danger nutrients (kept for prompt context) ─────────────────────

_SYMPTOM_DANGER_NUTRIENTS: dict[str, list[str]] = {
    "Heartburn":      ["total fat", "saturated fat", "sodium", "sugar"],
    "Acid Reflux":    ["total fat", "saturated fat", "sodium", "sugar"],
    "Bloating":       ["fermentable carbs", "sugar", "sodium", "fibre-rich foods"],
    "Gas":            ["fermentable carbs", "sugar"],
    "Nausea":         ["total fat", "saturated fat", "cholesterol"],
    "Cramps":         ["total fat", "saturated fat", "sodium", "sugar"],
    "Abdominal Pain": ["total fat", "sodium", "sugar", "cholesterol"],
    "Diarrhea":       ["sugar", "refined carbs", "total fat", "sodium"],
    "Constipation":   ["total fat", "saturated fat", "sodium"],
    "Fatigue":        ["refined sugar", "refined carbs", "calories"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data class (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SafeFoodRecommendation:
    rank:             int
    food_name:        str
    recommendation:   str
    safety_score:     float
    similarity_score: float
    combined_score:   float
    key_nutrients:    dict
    safe_reasons:     list[str]
    avoid_reasons:    list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Claude recommendation generation
# ─────────────────────────────────────────────────────────────────────────────

_REC_SYSTEM = (
    "You are a clinical gut-health dietitian AI. "
    "Return ONLY a JSON object with a 'recommendations' array (no markdown, no extra text). "
    "Each item must have exactly these keys: "
    "food_name (str), recommendation (str, 1-2 warm friendly sentences), "
    "safety_score (float 0-1, how safe for this user's gut), "
    "similarity_score (float 0-1, how similar to their eating pattern), "
    "combined_score (float 0-1), "
    "key_nutrients (object: Calories kcal/100g, Protein g/100g, Total Fat g/100g, "
    "Carbohydrates g/100g, Sodium mg/100g, Sugar g/100g — use realistic USDA values), "
    "safe_reasons (array of 1-3 short strings), "
    "avoid_reasons (array of 0-2 short strings or empty)."
)


def _build_prompt(
    eaten_names: list[str],
    user_symptoms: list[str],
    danger_nutrients: list[str],
    n: int,
) -> str:
    symptom_str  = ", ".join(user_symptoms) if user_symptoms else "no specific symptoms"
    danger_str   = ", ".join(danger_nutrients) if danger_nutrients else "none identified"
    eaten_str    = (", ".join(eaten_names[:15]) if eaten_names else "nothing logged yet")

    return (
        f"User's recent foods: {eaten_str}.\n"
        f"User's digestive symptoms: {symptom_str}.\n"
        f"Nutrients to minimise: {danger_str}.\n\n"
        f"Recommend {n} gut-friendly foods the user has NOT already eaten. "
        f"Foods must be:\n"
        f"1. Nutritionally similar in style to what they already eat\n"
        f"2. Low in their danger nutrients\n"
        f"3. Genuinely gut-friendly (high in fibre, low in fat/sodium/sugar where possible)\n"
        f"4. Described warmly with WHY they are safe for this user's specific symptoms\n\n"
        f"Return the JSON object now."
    )


def recommend_safe_foods(
    eaten_usda_ids: list[int],
    user_symptoms:  list[str],
    user_memory,
    n:              int = 5,
    # Legacy params accepted but ignored
    **_kwargs,
) -> list[SafeFoodRecommendation]:

    if _claude_client is None:
        raise RuntimeError("food_recommender not initialised.")

    # Get eaten food names from USDA client for context
    eaten_names: list[str] = []
    if _usda_client is not None:
        for uid in eaten_usda_ids[:20]:
            desc = _usda_client.get_description(uid)
            if desc and "USDA ID" not in desc:
                eaten_names.append(desc)

    # Compile danger nutrients from symptoms
    danger: set[str] = set()
    for sym in user_symptoms:
        danger.update(_SYMPTOM_DANGER_NUTRIENTS.get(sym, []))

    # Add danger nutrients from user memory if available
    if user_memory is not None:
        for key, prior in user_memory.priors.items():
            if prior.posterior_mean > 0.60:
                try:
                    sym = key.split(":", 1)[1]
                    danger.update(_SYMPTOM_DANGER_NUTRIENTS.get(sym, []))
                except (IndexError, ValueError):
                    pass

    prompt = _build_prompt(eaten_names, user_symptoms, sorted(danger), n)

    try:
        msg = _claude_client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 200 * n + 200,
            system     = _REC_SYSTEM,
            messages   = [{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        data  = json.loads(raw)
        items = data.get("recommendations", [])
    except Exception as exc:
        log.error(f"Claude recommendation failed: {exc}")
        return []

    results: list[SafeFoodRecommendation] = []
    for rank, item in enumerate(items[:n], 1):
        results.append(SafeFoodRecommendation(
            rank             = rank,
            food_name        = str(item.get("food_name", "")),
            recommendation   = str(item.get("recommendation", "")),
            safety_score     = float(item.get("safety_score", 0.7)),
            similarity_score = float(item.get("similarity_score", 0.6)),
            combined_score   = float(item.get("combined_score", 0.65)),
            key_nutrients    = item.get("key_nutrients", {}),
            safe_reasons     = list(item.get("safe_reasons", [])),
            avoid_reasons    = list(item.get("avoid_reasons", [])),
        ))

    log.info(f"Generated {len(results)} food recommendations via Claude.")
    return results

import re as _re
from collections import defaultdict as _defaultdict
from datetime import datetime, timezone
 
# Import the composite grouping utility from food_symptom_predictor
# (both modules are always initialised before requests are served)
from food_symptom_predictor import (
    group_composite_meals,
    _DIGESTION_WINDOWS_H,
    _DEFAULT_WINDOW_H,
    _parse_dt as _pdt,
    _build_timeline_prompt,
)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────
 
_SAFE_REC_SYSTEM = """\
You are a clinical gut-health dietitian AI.
 
You will receive a user's food and symptom timeline spanning one or more days.
Your job is to recommend exactly 5 safe foods for this user.
 
━━━ DIGESTION WINDOWS ━━━
  Heartburn / Acid Reflux  →  15 min – 3 h
  Nausea                   →  30 min – 4 h
  Cramps / Bloating / Gas / Abdominal Pain  →  30 min – 8 h
  Diarrhea                 →  1 h – 16 h
  Fatigue                  →  1 h – 12 h
  Constipation             →  12 h – 48 h
  Any other symptom        →  default 30 min – 8 h
 
━━━ HOW TO IDENTIFY SAFE FOODS ━━━
Step 1 — Find SYMPTOM-LINKED foods:
  For each symptom, find foods eaten within its digestion window.
  These are SUSPECT foods — likely contributed to the symptom.
 
Step 2 — Find SAFE foods from history:
  Foods that appear in the log but NEVER fell within any symptom's
  digestion window across the entire timeline are SAFE foods.
  Prioritise these in your recommendations.
 
Step 3 — Suggest NEW gut-friendly foods:
  If fewer than 5 safe history foods exist, fill remaining slots with
  NEW foods that are:
    • Unlikely to cause any of the user's reported symptoms
    • Gut-friendly (high fibre, low fat/sodium/sugar where relevant)
    • Realistic and commonly available
    • Nutritionally complementary to what the user already eats
 
━━━ RULES ━━━
• Safe foods from HISTORY: use the name EXACTLY as in the food log.
• NEW suggested foods: use clear, common food names (e.g. "Banana", "Oatmeal").
• Do NOT include any food that was suspect for any symptom.
• Return exactly 5 foods — no more, no less.
• Return ONLY valid JSON — no markdown, no explanation:
 
{"safe_foods": ["Food 1", "Food 2", "Food 3", "Food 4", "Food 5"]}
"""
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Rule-based fallback: finds history foods not linked to any symptom
# ─────────────────────────────────────────────────────────────────────────────
 
def _rule_based_safe(
    grouped_meals: list[dict],
    symptom_logs:  list[dict],
    n:             int,
) -> list[str]:
    """Return up to n safe food names using rule-based window matching."""
    suspect: set[str] = set()
    for s in symptom_logs:
        symptom  = s.get("symptom", "")
        sym_time = _pdt(s["logged_at"])
        wmin, wmax = _DIGESTION_WINDOWS_H.get(symptom, _DEFAULT_WINDOW_H)
        for m in grouped_meals:
            ft = _pdt(m["logged_at"])
            if ft >= sym_time:
                continue
            h = (sym_time - ft).total_seconds() / 3600.0
            if wmin <= h <= wmax:
                suspect.add(m["food_name"])
 
    safe = [
        m["food_name"] for m in grouped_meals
        if m["food_name"] not in suspect
    ]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_safe: list[str] = []
    for name in safe:
        if name not in seen:
            seen.add(name)
            unique_safe.append(name)
 
    return unique_safe[:n]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Public function
# ─────────────────────────────────────────────────────────────────────────────
 
def recommend_safe_from_logs(
    food_logs_named: list[dict],
    symptom_logs:    list[dict],
    n:               int = 5,
) -> list[str]:
    """
    Analyse a food + symptom timeline and return n safe food names.
 
    Parameters
    ----------
    food_logs_named : list[dict]  — {food_name, weight_g, logged_at}
        usda_id must already be converted to food_name by the endpoint.
    symptom_logs : list[dict]     — {symptom, intensity, logged_at}
    n : int                       — number of safe foods to return (default 5)
 
    Returns
    -------
    list[str]
        Exactly n food names (or fewer if Claude returns fewer despite instruction).
        Names from history are verbatim; newly suggested names are plain strings.
        Falls back to rule-based safe-food detection if Claude is unavailable.
    """
    grouped = group_composite_meals(food_logs_named)
 
    if _claude_client is None:
        log.warning("recommend_safe_from_logs: Claude unavailable — rule-based fallback.")
        return _rule_based_safe(grouped, symptom_logs, n)
 
    if not food_logs_named:
        log.info("recommend_safe_from_logs: no food logs provided.")
        return []
 
    # Build the same timeline format used by predict_causation_by_time
    timeline = _build_timeline_prompt(grouped, symptom_logs)
 
    user_msg = (
        f"{timeline}\n\n"
        f"Based on the above timeline, recommend exactly {n} safe foods for this user."
    )
 
    try:
        msg = _claude_client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 150,
            system     = _SAFE_REC_SYSTEM,
            messages   = [{"role": "user", "content": user_msg}],
        )
        raw     = msg.content[0].text.strip()
        cleaned = _re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        data    = json.loads(cleaned)
        foods   = data.get("safe_foods", [])
 
        # Sanitise: ensure strings, strip blanks, deduplicate
        seen: set[str] = set()
        result: list[str] = []
        for item in foods:
            if isinstance(item, str) and item.strip() and item not in seen:
                seen.add(item)
                result.append(item.strip())
 
        log.info(f"recommend_safe_from_logs: Claude returned {len(result)} safe foods.")
        return result[:n]
 
    except Exception as exc:
        log.warning(f"recommend_safe_from_logs: Claude failed ({exc}) — rule-based fallback.")
        return _rule_based_safe(grouped, symptom_logs, n)
 