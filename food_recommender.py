"""
food_recommender.py
───────────────────
Personalised safe food recommendation engine.

╔══════════════════════════════════════════════════════════════════════╗
║  recommend_safe_from_logs()  ← /recommend/safe-foods endpoint       ║
║  FULLY LOGIC-BASED — zero Claude API calls, zero additional cost.   ║
║                                                                      ║
║  Algorithm:                                                          ║
║   1. Temporal window matching: foods eaten before a symptom within  ║
║      its digestion window are SUSPECT.                              ║
║   2. History-safe foods: foods that never fell in any symptom        ║
║      window are SAFE and ranked first.                              ║
║   3. Nutrient ranking: safe history foods sorted by low nutrient     ║
║      risk vs the user's reported symptoms.                          ║
║   4. Curated fallback: when < n safe history foods exist, fill       ║
║      remaining slots from a curated gut-friendly food list,          ║
║      filtered against the user's danger nutrients.                  ║
╚══════════════════════════════════════════════════════════════════════╝

  recommend_safe_foods()  ← internal / other endpoints — unchanged
  (still uses Claude for personalised recommendations with user memory)

Public functions:
    init(claude_client, usda_client)
    recommend_safe_foods(eaten_usda_ids, user_symptoms, user_memory, n)
        -> list[SafeFoodRecommendation]
    recommend_safe_from_logs(food_logs_named, symptom_logs, n)
        -> list[str]
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
    log.info("food_recommender: ready (recommend_safe_from_logs is logic-based).")


# ─────────────────────────────────────────────────────────────────────────────
# Shared imports from food_symptom_predictor
# ─────────────────────────────────────────────────────────────────────────────

from food_symptom_predictor import (
    group_composite_meals,
    _DIGESTION_WINDOWS_H,
    _DEFAULT_WINDOW_H,
    _parse_dt as _pdt,
    _build_timeline_prompt,
    SYMPTOM_NUTRIENT_RISK,
    _NUTRIENT_RISK_THRESHOLDS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Symptom → danger nutrients  (for Claude-based recommend_safe_foods)
# ─────────────────────────────────────────────────────────────────────────────

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

# Internal nutrient key mapping  (matches USDA client keys)
_SYMPTOM_DANGER_KEYS: dict[str, list[str]] = {
    "Heartburn":      ["total_fat", "sat_fat", "sodium", "sugar"],
    "Acid Reflux":    ["total_fat", "sat_fat", "sodium", "sugar"],
    "Bloating":       ["carbs", "sugar", "sodium"],
    "Gas":            ["carbs", "sugar"],
    "Nausea":         ["total_fat", "sat_fat", "cholesterol"],
    "Cramps":         ["total_fat", "sat_fat", "sodium", "sugar"],
    "Abdominal Pain": ["total_fat", "sodium", "sugar", "cholesterol"],
    "Diarrhea":       ["sugar", "carbs", "total_fat", "sodium"],
    "Constipation":   ["total_fat", "sat_fat", "sodium"],
    "Fatigue":        ["sugar", "carbs", "calories"],
}


# ─────────────────────────────────────────────────────────────────────────────
# SafeFoodRecommendation data class  (unchanged)
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
# Curated gut-friendly fallback list
# Used when not enough safe history foods can be identified.
# Each entry: (food_name, gut_friendly_for_symptoms_set)
# None means good for all symptoms.
# ─────────────────────────────────────────────────────────────────────────────

_CURATED_SAFE_FOODS: list[tuple[str, Optional[set[str]]]] = [
    # universally gentle
    ("Banana",                     None),
    ("White rice, cooked",         None),
    ("Oatmeal",                    None),
    ("Sweet potato, baked",        None),
    ("Ginger tea",                 None),
    ("Boiled chicken breast",      None),
    ("Plain Greek yogurt",         None),
    ("Steamed carrots",            None),
    ("Avocado",                    None),
    ("Scrambled eggs",             None),
    ("Blueberries",                None),
    ("Watermelon",                 None),
    ("Cucumber",                   None),
    ("Boiled potatoes",            None),
    ("Toasted sourdough bread",    None),
    ("Baked salmon",               None),
    ("Steamed broccoli",           {"Bloating", "Gas"}),     # exclude for Bloating/Gas
    ("Lentil soup",                {"Gas", "Bloating"}),
    ("Apple, raw",                 None),
    ("Turkey breast, roasted",     None),
    ("Brown rice, cooked",         {"Diarrhea"}),            # high fibre can worsen diarrhea
    ("Plain rice cakes",           None),
    ("Peppermint tea",             None),
    ("Chamomile tea",              None),
    ("Cooked spinach",             None),
    ("Baked cod",                  None),
    ("Plain crackers",             None),
    ("Almond butter",              None),
    ("Mango, fresh",               None),
    ("Papaya, fresh",              None),                    # contains digestive enzymes
]

# Symptom → nutrients the safe food should be LOW in (for ranking curated foods)
_SAFE_FOOD_LOW_NUTRIENTS: dict[str, list[str]] = {
    sym: keys for sym, keys in _SYMPTOM_DANGER_KEYS.items()
}


def _curated_safe_list(
    symptom_names: list[str],
    exclude_names:  set[str],
    n:              int,
) -> list[str]:
    """
    Return up to n curated gut-friendly food names, excluding:
      • foods already in exclude_names (suspect foods from history)
      • foods marked as inappropriate for any of the user's symptoms
    """
    result: list[str] = []
    symptom_set = set(symptom_names)

    for food_name, bad_symptoms in _CURATED_SAFE_FOODS:
        if len(result) >= n:
            break
        if food_name in exclude_names:
            continue
        if bad_symptoms and bad_symptoms & symptom_set:
            continue
        result.append(food_name)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Nutrient-based safety ranker
# ─────────────────────────────────────────────────────────────────────────────

def _nutrient_danger_score(usda_id: int, symptom_names: list[str]) -> float:
    """
    Return a 0-1 danger score for a food given the user's symptoms.
    Lower score = safer food.  Requires USDA client.
    Returns 0.5 (neutral) when no data is available.
    """
    if not _usda_client or not usda_id:
        return 0.5
    data = _usda_client.get_nutrients(usda_id)
    if not data:
        return 0.5

    total_danger = 0.0
    n_symptoms   = 0

    for symptom in symptom_names:
        danger_keys = _SYMPTOM_DANGER_KEYS.get(symptom, [])
        weights     = SYMPTOM_NUTRIENT_RISK.get(symptom, {})
        total_w     = sum(weights.get(k, 0) for k in danger_keys) or 1.0
        sym_danger  = 0.0

        for key in danger_keys:
            val = data.get(key)
            if val is None:
                continue
            threshold = _NUTRIENT_RISK_THRESHOLDS.get(key, 1.0)
            ratio     = min(float(val) / max(threshold, 0.001), 2.0)
            sym_danger += ratio * weights.get(key, 0.1)

        total_danger += sym_danger / total_w
        n_symptoms += 1

    if n_symptoms == 0:
        return 0.5
    return round(min(total_danger / n_symptoms, 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Core rule-based logic
# ─────────────────────────────────────────────────────────────────────────────

def _find_suspect_foods(
    grouped_meals: list[dict],
    symptom_logs:  list[dict],
) -> set[str]:
    """
    Return the set of food names that were eaten within at least one
    symptom's digestion window (i.e. potentially caused a symptom).
    """
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
                # Mark all individual components as suspect
                for comp in m.get("components", [m["food_name"]]):
                    suspect.add(comp)
    return suspect


def _safe_history_foods_ranked(
    grouped_meals:  list[dict],
    suspect_foods:  set[str],
    symptom_names:  list[str],
    food_logs_named: list[dict],
) -> list[str]:
    """
    Return unique safe food names from history, ranked by:
      1. Foods that appear most frequently in the log (eaten often = trusted)
      2. Foods with lowest nutrient danger score for the user's symptoms

    A food is safe if none of its individual components are suspect.
    """
    # Collect all non-suspect unique food names with their usda_ids
    safe_name_counts: dict[str, int]  = {}
    name_to_usda:     dict[str, int]  = {}

    for entry in food_logs_named:
        name  = entry.get("food_name", "")
        uid   = entry.get("usda_id", 0)
        if not name or name in suspect_foods:
            continue
        safe_name_counts[name] = safe_name_counts.get(name, 0) + 1
        if uid:
            name_to_usda[name] = uid

    if not safe_name_counts:
        return []

    # Also check composite meals: only include if ALL components are safe
    for meal in grouped_meals:
        if meal["is_composite"]:
            comps = meal.get("components", [])
            if all(c not in suspect_foods for c in comps):
                meal_name = meal["food_name"]
                if meal_name not in safe_name_counts:
                    safe_name_counts[meal_name] = 1

    # Score each safe food: lower danger score + higher frequency = better
    def _rank_key(name: str) -> float:
        uid       = name_to_usda.get(name, 0)
        danger    = _nutrient_danger_score(uid, symptom_names)
        frequency = safe_name_counts.get(name, 1)
        # Lower danger = higher rank, higher frequency = higher rank
        return danger - 0.05 * frequency

    ranked = sorted(safe_name_counts.keys(), key=_rank_key)
    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# Public: recommend_safe_from_logs  (logic-based, no Claude)
# ─────────────────────────────────────────────────────────────────────────────

def recommend_safe_from_logs(
    food_logs_named: list[dict],
    symptom_logs:    list[dict],
    n:               int = 5,
) -> list[str]:
    """
    Identify n safe foods for this user — no Claude API, no additional cost.

    Parameters
    ----------
    food_logs_named : list[dict]  — {food_name, weight_g, logged_at[, usda_id]}
    symptom_logs    : list[dict]  — {symptom, intensity, logged_at}
    n               : int         — number of recommendations to return

    Returns
    -------
    list[str]
        Up to n food names.  Safe history foods are listed first (ranked by
        safety); new gut-friendly suggestions fill any remaining slots.

    Logic
    ─────
    Step 1 — Identify SUSPECT foods (eaten within any symptom's digestion window)
    Step 2 — SAFE history foods = logged foods never in any symptom window
    Step 3 — Rank safe history foods by nutrient danger score (low = safe)
    Step 4 — Fill remaining slots with curated gut-friendly foods (symptom-aware)
    """
    if not food_logs_named:
        log.info("recommend_safe_from_logs: no food logs — returning curated list only.")
        symptom_names = [s.get("symptom", "") for s in symptom_logs if s.get("symptom")]
        return _curated_safe_list(symptom_names, set(), n)

    grouped       = group_composite_meals(food_logs_named)
    symptom_names = list(dict.fromkeys(
        s.get("symptom", "") for s in symptom_logs if s.get("symptom")
    ))

    # ── Step 1: find suspect foods ────────────────────────────────────────────
    suspect = _find_suspect_foods(grouped, symptom_logs)

    # ── Step 2 & 3: rank safe history foods ───────────────────────────────────
    safe_history = _safe_history_foods_ranked(
        grouped_meals   = grouped,
        suspect_foods   = suspect,
        symptom_names   = symptom_names,
        food_logs_named = food_logs_named,
    )

    # ── Step 4: fill remaining slots from curated list ────────────────────────
    result: list[str] = safe_history[:n]

    if len(result) < n:
        all_seen = set(safe_history) | suspect
        extras = _curated_safe_list(
            symptom_names = symptom_names,
            exclude_names = all_seen,
            n             = n - len(result),
        )
        result.extend(extras)

    log.info(
        f"recommend_safe_from_logs (logic): "
        f"{len(safe_history[:n])} from history + "
        f"{max(0, len(result) - len(safe_history[:n]))} curated fillers = "
        f"{len(result)} total. "
        f"Suspect foods excluded: {len(suspect)}."
    )
    return result[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Public: recommend_safe_foods  (Claude-based, unchanged)
# Used by other endpoints with user_memory personalisation.
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
    eaten_names:      list[str],
    user_symptoms:    list[str],
    danger_nutrients: list[str],
    n:                int,
) -> str:
    symptom_str = ", ".join(user_symptoms) if user_symptoms else "no specific symptoms"
    danger_str  = ", ".join(danger_nutrients) if danger_nutrients else "none identified"
    eaten_str   = (", ".join(eaten_names[:15]) if eaten_names else "nothing logged yet")
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
    **_kwargs,
) -> list[SafeFoodRecommendation]:
    """
    Personalised food recommendations via Claude AI.
    Used by other endpoints that include user_memory.
    """
    if _claude_client is None:
        raise RuntimeError("food_recommender not initialised.")

    eaten_names: list[str] = []
    if _usda_client is not None:
        for uid in eaten_usda_ids[:20]:
            desc = _usda_client.get_description(uid)
            if desc and "USDA ID" not in desc:
                eaten_names.append(desc)

    danger: set[str] = set()
    for sym in user_symptoms:
        danger.update(_SYMPTOM_DANGER_NUTRIENTS.get(sym, []))

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
            max_tokens = 140 * n + 120,
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