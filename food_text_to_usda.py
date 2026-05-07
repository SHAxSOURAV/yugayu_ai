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

import hashlib
import json
import logging
import re
from typing import Optional

from text_context_parser import utc_now_iso

log = logging.getLogger(__name__)

_claude_client = None
_usda_client   = None
_mongo_db      = None   # set by init() — optional, cache disabled if None


def init(claude_client, usda_client, mongo_db=None) -> None:
    global _claude_client, _usda_client, _mongo_db
    _claude_client = claude_client
    _usda_client   = usda_client
    _mongo_db      = mongo_db
    status = "MongoDB cache enabled" if mongo_db is not None else "no cache"
    log.info(f"food_text_to_usda: Claude + USDA client ready ({status}).")


# ─────────────────────────────────────────────────────────────────────────────
# Cache key helper
# ─────────────────────────────────────────────────────────────────────────────

def _light_normalise_text(text: str) -> str:
    norm = text.lower().strip()
    norm = re.sub(r"[^\w\s]", " ", norm)
    norm = re.sub(r"\b(had\s+eaten|had\s+eat|ate)\b", "eat", norm)
    norm = re.sub(r"\bfried\b", "fry", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def _cache_key(text: str) -> str:
    """
    SHA-256 of the lowercased, whitespace-normalised input text.
    Ensures 'Chicken Biryani 400g' and 'chicken biryani 400g' share one cache entry.
    """
    normalised = _light_normalise_text(text)
    return hashlib.sha256(normalised.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Claude: extract + decompose + normalise + weight in one call
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are a clinical nutrition assistant that parses meal descriptions for a \
gut health tracking app. The app stores foods using USDA FoodData Central IDs, \
so every food you return must be searchable in the USDA database.

Given a user's meal text, return a JSON object with exactly two keys:

1. "meal_type": one of "Breakfast", "Lunch", "Dinner", "Snack", or null.
   Detect from context words like "breakfast", "lunch", "dinner", "morning",
   "noon", "tonight", "snack", etc.

2. "foods": a JSON array. Each element:
   {
     "word":       <the food phrase as written by the user>,
     "normalised": <USDA-searchable food name — see rules below>,
     "weight_g":   <weight in grams as a positive number>
   }

NORMALISATION RULES:
- Use USDA FoodData Central naming conventions:
    Good: "Rice, white, long-grain, cooked"
    Good: "Chicken, broiler, breast, cooked, roasted"
    Bad:  "chicken biryani"  (USDA does not index ethnic dish names)
- For COMPOSITE or ETHNIC dishes (biryani, fried rice, pasta bake, curry,
  stew, soup, sandwich, burger, pilaf, etc.) ALWAYS decompose into individual
  USDA-searchable ingredient foods. Do NOT return the dish name as-is.
  Example — "chicken biriyani 400g":
    { "word": "chicken biriyani", "normalised": "Chicken, broiler or fryer, breast, meat only, cooked, roasted", "weight_g": 160 }
    { "word": "chicken biriyani", "normalised": "Rice, white, long-grain, cooked",                                "weight_g": 200 }
    { "word": "chicken biriyani", "normalised": "Oil, vegetable",                                                 "weight_g": 20  }
    { "word": "chicken biriyani", "normalised": "Onions, raw",                                                    "weight_g": 20  }
  Example — "egg fried rice":
    { "word": "egg fried rice", "normalised": "Rice, white, long-grain, cooked", "weight_g": 200 }
    { "word": "egg fried rice", "normalised": "Egg, whole, cooked, fried",       "weight_g": 60  }
    { "word": "egg fried rice", "normalised": "Oil, vegetable",                  "weight_g": 10  }
- For simple whole foods (apple, oatmeal, grilled chicken) return directly.

WEIGHT RULES (apply in order):
  a. User states a weight for a specific food — use it exactly.
  b. User states a TOTAL weight for a dish — decompose, then distribute the
     stated total across components in realistic proportions.
     Component weights MUST sum to the stated total.
  c. No weight mentioned — estimate a realistic adult single-serving weight
     per component based on standard dietary guidelines.

OUTPUT RULES:
  - weight_g must always be a positive number (never 0 or null).
  - Return ONLY the JSON object — no markdown, no extra text.
  - If no food is found, return: {"meal_type": null, "foods": []}
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
            model      = "claude-haiku-4-5-20251001",
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

    Cache strategy (MongoDB):
      - Cache key: SHA-256 of lowercased, whitespace-normalised input text.
      - On HIT:  skip Claude call and all USDA API calls entirely.
                 Attach a fresh logged_at and return immediately (~5 ms).
      - On MISS: run Claude extraction + USDA ID lookup as normal,
                 then store the result (without logged_at) for future hits.

    logged_at is always generated fresh so each returned record reflects
    the actual wall-clock time of the request, never the time of first parse.

    Returns list of dicts:
        raw_food, normalised_name, usda_id, usda_description,
        meal_type, weight_g, logged_at
    """
    if _claude_client is None or _usda_client is None:
        raise RuntimeError("food_text_to_usda not initialised. Call init() first.")

    key = _cache_key(text)

    # ── Cache lookup ──────────────────────────────────────────────────────────
    if _mongo_db is not None:
        try:
            cached = _mongo_db.get_food_parse(key)
            if cached:
                logged_at = utc_now_iso()
                log.info(f"food_parse cache HIT for key={key[:12]}… ({len(cached)} foods)")
                return [{**item, "logged_at": logged_at} for item in cached]
        except Exception as exc:
            # Cache failure must never block the real request
            log.warning(f"food_parse cache read failed: {exc}")

    # ── Cache miss: Claude + USDA ─────────────────────────────────────────────
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

    # ── Drop minor components (< 10 % of total meal weight) ──────────────────
    # Any food whose portion is under 10 % of the total is nutritionally
    # negligible and adds latency (extra USDA calls) with no scoring benefit.
    # Applied before cache write so cached entries are already filtered.
    if results:
        total_weight = sum(r["weight_g"] for r in results)
        threshold    = total_weight * 0.10
        before       = len(results)
        results      = [r for r in results if r["weight_g"] >= threshold]
        dropped      = before - len(results)
        if dropped:
            log.info(
                f"Dropped {dropped} minor component(s) below 10 % threshold "
                f"({threshold:.1f}g of {total_weight:.1f}g total)."
            )

    # ── Store in cache (strip logged_at — it must always be fresh) ────────────
    if _mongo_db is not None and results:
        try:
            to_store = [{k: v for k, v in item.items() if k != "logged_at"}
                        for item in results]
            _mongo_db.set_food_parse(key, to_store)
            log.info(f"food_parse cache SET for key={key[:12]}… ({len(to_store)} foods)")
        except Exception as exc:
            log.warning(f"food_parse cache write failed: {exc}")

    return results