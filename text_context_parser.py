"""
text_context_parser.py
──────────────────────
Pure-stdlib, zero-ML utility that enriches food parse results with three
pieces of context extracted from the original user text:

    1. meal_type  — which meal the user is describing
    2. quantity   — numeric amount for each detected food
    3. unit       — measurement unit that accompanies the quantity
    4. logged_at  — UTC ISO-8601 timestamp of when the parse was called

Public API
──────────
    parse_meal_type(text)                     -> str | None
    parse_quantity_for_entity(text, start, end) -> tuple[float | None, str | None]
    utc_now_iso()                             -> str

No external dependencies.  Import and call from food_text_to_usda.text_to_usda().
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional


# ── 1. Meal-type detection ────────────────────────────────────────────────────

# Each entry: (canonical_label, [keyword_patterns])
# Patterns are matched case-insensitively against the full user text.
_MEAL_RULES: list[tuple[str, list[str]]] = [
    (
        "Breakfast",
        [
            r"\bbreakfast\b",
            r"\bmorning\b",
            r"\bwoke\s+up\b",
            r"\bbrunch\b",
        ],
    ),
    (
        "Lunch",
        [
            r"\blunch\b",
            r"\bnoon\b",
            r"\bmidday\b",
            r"\bmid[-\s]?day\b",
        ],
    ),
    (
        "Dinner",
        [
            r"\bdinner\b",
            r"\bsupper\b",
            r"\bevening\s+meal\b",
            r"\btonight\b",
            r"\blast\s+night\b",
        ],
    ),
    (
        "Snack",
        [
            r"\bsnack\b",
            r"\bbite\b",
            r"\bmunch\b",
            r"\bnibble\b",
            r"\bbetween\s+meals\b",
        ],
    ),
]

# Pre-compile all patterns once at import time
_COMPILED_MEAL_RULES: list[tuple[str, list[re.Pattern[str]]]] = [
    (label, [re.compile(p, re.IGNORECASE) for p in patterns])
    for label, patterns in _MEAL_RULES
]


def parse_meal_type(text: str) -> Optional[str]:
    """
    Detect the meal type mentioned in *text*.

    Returns one of: "Breakfast" | "Lunch" | "Dinner" | "Snack" | None

    Evaluation order follows _MEAL_RULES (Breakfast first, Snack last).
    Returns the first matching label; None if no keyword matches.

    Examples
    --------
    >>> parse_meal_type("I had oatmeal for breakfast")
    'Breakfast'
    >>> parse_meal_type("just a quick snack before bed")
    'Snack'
    >>> parse_meal_type("ate some chicken")
    None
    """
    for label, compiled_patterns in _COMPILED_MEAL_RULES:
        for pattern in compiled_patterns:
            if pattern.search(text):
                return label
    return None


# ── 2. Quantity + unit detection ─────────────────────────────────────────────

# Canonical unit labels (what we return regardless of input variant)
_UNIT_CANONICAL: dict[str, str] = {
    # weight
    "g":        "g",
    "gram":     "g",
    "grams":    "g",
    "kg":       "kg",
    "kilogram": "kg",
    "kilograms":"kg",
    "oz":       "oz",
    "ounce":    "oz",
    "ounces":   "oz",
    "lb":       "lb",
    "lbs":      "lb",
    "pound":    "lb",
    "pounds":   "lb",
    # volume
    "ml":       "ml",
    "milliliter":"ml",
    "millilitre":"ml",
    "milliliters":"ml",
    "millilitres":"ml",
    "l":        "l",
    "liter":    "l",
    "litre":    "l",
    "liters":   "l",
    "litres":   "l",
    # culinary
    "cup":      "cup",
    "cups":     "cup",
    "tbsp":     "tbsp",
    "tablespoon":"tbsp",
    "tablespoons":"tbsp",
    "tsp":      "tsp",
    "teaspoon": "tsp",
    "teaspoons":"tsp",
    # count
    "piece":    "piece",
    "pieces":   "piece",
    "slice":    "piece",
    "slices":   "piece",
    "serving":  "serving",
    "servings": "serving",
    "portion":  "serving",
    "portions": "serving",
}

# Build the alternation string for the regex (longest first avoids partial matches)
_UNIT_ALTS = "|".join(
    sorted(_UNIT_CANONICAL.keys(), key=len, reverse=True)
)

# Matches patterns like:
#   "200g"  "200 g"  "1.5 cups"  "2 pieces"  "half a serving"
# Groups: (1) number part  (2) unit string
_QTY_RE = re.compile(
    rf"(\d+(?:[.,]\d+)?)\s*({_UNIT_ALTS})\b",
    re.IGNORECASE,
)

# How many characters to the left / right of the entity span to search
_CONTEXT_WINDOW = 60  # chars


def parse_quantity_for_entity(
    text: str,
    entity_start: int,
    entity_end: int,
) -> tuple[Optional[float], Optional[str]]:
    """
    Find a quantity + unit near a food entity whose character span in *text*
    is [entity_start, entity_end).

    Search strategy
    ───────────────
    1. Look in a ±CONTEXT_WINDOW char window around the entity.
    2. Return the match *closest* to the entity boundary (left wins over right
       when equidistant).
    3. If no unit-bearing quantity is found, try a bare integer/decimal
       immediately adjacent to the entity (e.g. "chicken 2" → quantity=2, unit=None).

    Returns
    -------
    (quantity, unit)  — (None, None) when nothing is found.

    Examples
    --------
    >>> parse_quantity_for_entity("I ate 200g chicken breast", 11, 25)
    (200.0, 'g')
    >>> parse_quantity_for_entity("have 1.5 cups of oatmeal", 17, 24)
    (1.5, 'cup')
    >>> parse_quantity_for_entity("ate chicken 2 pieces with rice", 4, 11)
    (2.0, 'piece')
    """
    window_start = max(0, entity_start - _CONTEXT_WINDOW)
    window_end   = min(len(text), entity_end + _CONTEXT_WINDOW)
    window_text  = text[window_start:window_end]

    best_match: Optional[re.Match[str]] = None
    best_dist   = float("inf")

    for m in _QTY_RE.finditer(window_text):
        # Distance = gap between match boundary and entity boundary in window coords
        entity_start_w = entity_start - window_start
        entity_end_w   = entity_end   - window_start
        dist = min(
            abs(m.end()   - entity_start_w),   # match ends before entity
            abs(m.start() - entity_end_w),      # match starts after entity
        )
        if dist < best_dist:
            best_dist  = dist
            best_match = m

    if best_match:
        raw_qty  = best_match.group(1).replace(",", ".")
        raw_unit = best_match.group(2).lower()
        return float(raw_qty), _UNIT_CANONICAL.get(raw_unit, raw_unit)

    # ── Fallback: bare number immediately adjacent (no unit) ──────────────────
    # e.g. "ate 3 bananas"  →  quantity=3, unit=None
    bare_re = re.compile(r"(\d+(?:[.,]\d+)?)")
    for m in bare_re.finditer(window_text):
        entity_start_w = entity_start - window_start
        entity_end_w   = entity_end   - window_start
        gap = min(
            abs(m.end()   - entity_start_w),
            abs(m.start() - entity_end_w),
        )
        if gap <= 3:                     # within 3 chars → likely the count
            raw_qty = m.group(1).replace(",", ".")
            return float(raw_qty), None

    return None, None


# ── 3. UTC timestamp ──────────────────────────────────────────────────────────

def utc_now_iso() -> str:
    """
    Return the current UTC time as an ISO-8601 string with 'Z' suffix.

    Example: "2025-07-14T08:23:11Z"
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # meal_type
    for sentence, expected in [
        ("I had oatmeal for breakfast",                  "Breakfast"),
        ("Lunch was grilled chicken breast",             "Lunch"),
        ("Tonight I had spaghetti with tomato sauce",    "Dinner"),
        ("Just a quick snack — some yogurt",             "Snack"),
        ("Ate some fruit",                               None),
        ("Morning bowl of oatmeal with banana",          "Breakfast"),
    ]:
        result = parse_meal_type(sentence)
        status = "✓" if result == expected else "✗"
        print(f"{status}  meal_type='{result}' (expected='{expected}')  |  {sentence!r}")

    print()

    # quantity + unit
    cases = [
        ("I ate 200g chicken breast",       11, 25,  200.0,  "g"),
        ("have 1.5 cups of oatmeal",         17, 24,  1.5,    "cup"),
        ("250 ml orange juice",              7,  19,  250.0,  "ml"),
        ("2 pieces of salmon",               2,  16,  2.0,    "piece"),
        ("a tbsp of peanut butter",           7,  20,  None,   None),     # "a" is not a digit → no match
        ("ate chicken with 3 tbsp sauce",    4,  11,  3.0,    "tbsp"),
    ]
    for text, s, e, eq, eu in cases:
        q, u = parse_quantity_for_entity(text, s, e)
        status = "✓" if q == eq and u == eu else "✗"
        print(f"{status}  qty={q} unit={u!r} (expected {eq} {eu!r})  |  {text!r}")

    print()
    print("utc_now_iso():", utc_now_iso())