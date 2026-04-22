"""
food_text_to_usda.py
────────────────────
Natural Text → USDA Food ID pipeline.

Claude API handles:
  • Food entity extraction with composite-dish decomposition
  • USDA-style name normalisation
  • Weight estimation (g) — per food, distributed for shared totals,
    or estimated from standard serving sizes
  • Meal type detection

USDA FoodData Central REST API handles ID lookup.

USDA search strategy (multi-attempt, never silently drops a food):
  1. Full normalised name
  2. Raw food phrase the user wrote
  3. Individual meaningful keywords from the normalised name (longest first)
  4. Last resort: main ingredient only

Public API
──────────
    text_to_usda(text) -> list[dict]

Each dict:
    raw_food, normalised_name, usda_id, usda_description,
    meal_type, weight_g, logged_at
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from text_context_parser import utc_now_iso

log = logging.getLogger(__name__)

_claude_client = None
_usda_client   = None


def init(claude_client, usda_client) -> None:
    global _claude_client, _usda_client
    _claude_client = claude_client
    _usda_client   = usda_client
    log.info("food_text_to_usda: Claude + USDA client ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Claude: extract + decompose + normalise + weight in one call
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are a clinical nutrition assistant for a USDA FoodData Central gut health app.
Parse meal text. Return ONLY valid JSON, no markdown:
{"meal_type":"Breakfast"|"Lunch"|"Dinner"|"Snack"|null,"foods":[{"word":"<user phrase>","normalised":"<USDA name>","weight_g":<positive number>}]}

NORMALISED: Use USDA naming (e.g. "Rice, white, long-grain, cooked" not "white rice").

COMPOSITE DISHES (biryani, fried rice, curry, pasta, soup, burger, sandwich, stew):
Return ONLY the 2-3 MAJOR ingredients — the protein source and the carb/starch base.
SKIP all minor ingredients: oils, spices, herbs, onions, garlic, condiments, sauces.
Rule: only include an ingredient if its estimated weight is ≥ 30g.

Example — "chicken biryani 400g" → return only:
{"word":"chicken biryani","normalised":"Chicken, broiler, breast, cooked, roasted","weight_g":160}
{"word":"chicken biryani","normalised":"Rice, white, long-grain, cooked","weight_g":240}

Example — "egg fried rice 350g" → return only:
{"word":"egg fried rice","normalised":"Rice, white, long-grain, cooked","weight_g":270}
{"word":"egg fried rice","normalised":"Egg, whole, cooked, fried","weight_g":80}

Simple whole foods (apple, oatmeal, grilled chicken): return directly.

WEIGHT:
a) User states weight for a food → use exactly.
b) User states total weight for a dish → split across major components only; must sum to stated total.
c) No weight → estimate realistic adult single-serving per major component.

weight_g must always be a positive number. No food found → {"meal_type":null,"foods":[]}
"""


def _extract_entities(text: str) -> tuple[Optional[str], list[dict]]:
    """
    Single Claude call → (meal_type, [{word, normalised, weight_g}, ...])

    For composite dishes Claude decomposes them into individual USDA-searchable
    components, so the foods list may be longer than the number of items the
    user explicitly mentioned.
    """
    if _claude_client is None:
        log.error("Claude client not initialised.")
        return None, []

    try:
        msg = _claude_client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 300,
            system     = _EXTRACT_SYSTEM,
            messages   = [{"role": "user", "content": text}],
        )
        raw = msg.content[0].text.strip()

        # Strip markdown fences if Claude adds them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data      = json.loads(raw)
        meal_type = data.get("meal_type")
        foods     = [
            f for f in data.get("foods", [])
            if isinstance(f, dict) and f.get("word") and f.get("weight_g")
        ]
        return meal_type, foods

    except Exception as exc:
        # Re-raise auth errors so the caller can surface them as 503 instead
        # of silently returning an empty list (which produces a confusing 422).
        err_str = str(exc).lower()
        if any(k in err_str for k in ("authentication", "api_key", "invalid x-api-key", "401")):
            log.error(f"Claude API authentication failed: {exc}")
            raise RuntimeError(
                "Claude API key is invalid or missing. "
                "Check Claude_API_key / CLAUDE_API_KEY in your .env file."
            ) from exc
        log.warning(f"Claude entity extraction failed: {exc}")
        return None, []


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — USDA: multi-attempt search, never silently drops a food
# ─────────────────────────────────────────────────────────────────────────────

# Stop-words that are not useful as standalone USDA search terms
_SEARCH_STOPWORDS = {
    "cooked", "raw", "whole", "with", "and", "or", "the", "a", "an",
    "by", "for", "in", "on", "of", "to", "from", "dried", "fresh",
    "prepared", "added", "without", "salt", "heat", "dry",
}


def _usda_search_attempts(normalised: str, raw_food: str) -> list[str]:
    """
    Build a ranked list of search query candidates to try against USDA.
    Returns queries ordered from most specific → least specific.
    """
    candidates: list[str] = []

    # 1. Full normalised USDA name
    candidates.append(normalised)

    # 2. Raw phrase the user typed (if different)
    if raw_food.lower() != normalised.lower():
        candidates.append(raw_food)

    # 3. Meaningful individual words from the normalised name (longest first,
    #    skip stop-words and very short tokens)
    words = [
        w.strip(",.") for w in normalised.replace(",", " ").split()
        if len(w.strip(",.")) > 3 and w.strip(",.").lower() not in _SEARCH_STOPWORDS
    ]
    words.sort(key=len, reverse=True)
    for w in words[:4]:           # try up to 4 individual keywords
        if w not in candidates:
            candidates.append(w)

    # 4. First two meaningful words combined (e.g. "Chicken breast")
    if len(words) >= 2:
        combo = f"{words[0]} {words[1]}"
        if combo not in candidates:
            candidates.append(combo)

    return candidates


def _find_usda_match(normalised: str, raw_food: str) -> Optional[dict]:
    """
    Try multiple USDA search queries in order.
    Returns the first successful match dict, or None when all attempts fail.
    """
    for query in _usda_search_attempts(normalised, raw_food):
        matches = _usda_client.search(query, top_k=1)
        if matches:
            log.debug(
                f"USDA matched '{raw_food}' via query='{query}' "
                f"→ {matches[0]['usda_description']}"
            )
            return matches[0]

    log.warning(
        f"USDA: no match found for '{raw_food}' / '{normalised}' after all attempts."
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public function
# ─────────────────────────────────────────────────────────────────────────────

def text_to_usda(text: str) -> list[dict]:
    """
    Natural meal text → list of USDA food records with weight_g.

    Composite/ethnic dishes are automatically decomposed into individual
    USDA-searchable components by Claude. Each component gets a USDA ID
    via multi-attempt search.

    Returns list of dicts:
        raw_food, normalised_name, usda_id, usda_description,
        meal_type, weight_g, logged_at
    """
    if _claude_client is None or _usda_client is None:
        raise RuntimeError("food_text_to_usda not initialised. Call init() first.")

    meal_type, entities = _extract_entities(text)

    if not entities:
        log.info("No food entities detected in text.")
        return []

    logged_at = utc_now_iso()
    results: list[dict] = []

    for ent in entities:
        raw_food   = ent.get("word", "").strip()
        normalised = ent.get("normalised", raw_food)
        weight_g   = float(ent.get("weight_g", 0))

        if not raw_food or weight_g <= 0:
            continue

        best = _find_usda_match(normalised, raw_food)

        if best is None:
            # Still include the food — log a warning but never silently drop it.
            # usda_id=0 signals an unmatched food to the caller.
            log.warning(f"Including '{raw_food}' with usda_id=0 (no USDA match).")
            results.append({
                "raw_food":         raw_food,
                "normalised_name":  normalised,
                "usda_id":          0,
                "usda_description": normalised,
                "meal_type":        meal_type,
                "weight_g":         round(weight_g, 1),
                "logged_at":        logged_at,
            })
            continue

        results.append({
            "raw_food":         raw_food,
            "normalised_name":  normalised,
            "usda_id":          best["usda_id"],
            "usda_description": best["usda_description"],
            "meal_type":        meal_type,
            "weight_g":         round(weight_g, 1),
            "logged_at":        logged_at,
        })

    return results