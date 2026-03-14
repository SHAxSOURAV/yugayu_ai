"""
food_recommender.py
───────────────────
Personalised safe food recommendation engine.

━━━ WHAT IT DOES ━━━
Given a user's food log history and Bayesian symptom memory, finds 5 foods
that are:
  1. Nutritionally similar to what the user already enjoys (same macro class)
  2. Safer — lower in the nutrients that trigger their specific symptoms
  3. NOT foods the user has already eaten (genuine novelty)
  4. Described in natural language via Flan-T5 (no USDA ID jargon)

━━━ PIPELINE ━━━

Stage 1 — Build user nutrient profile
  Average the nutrient vectors of all foods the user has logged.
  This gives their "taste fingerprint" — what macro/calorie range they eat in.

Stage 2 — Identify personal risk nutrients
  From UserMemory: which (food, symptom) pairs have high causation probability.
  From those foods: extract which nutrients are elevated → these are the
  user's personal "danger nutrients" to avoid in recommendations.

Stage 3 — Candidate search from USDA table
  Filter USDA table to foods that:
    • Match the user's calorie/protein/carb range (similar macro class, ±40%)
    • Have significantly LOWER values of the user's danger nutrients
    • Are NOT in the user's already-eaten food set
  Score each candidate on a composite safety + similarity score.
  Take top 20 candidates.

Stage 4 — Flan-T5 natural description generation
  For each of the top 5 candidates, feed the USDA description + nutrients
  to Flan-T5 with a prompt:
    "Rewrite this USDA food entry as a friendly, natural food recommendation
     in one sentence. Include why it is gentle on digestion."
  This converts "CHICKEN,BROILERS OR FRYERS,BREAST,MEAT ONLY,CKD,ROASTED"
  into "Grilled chicken breast — a lean, easily digestible protein source
        that's low in saturated fat, ideal for sensitive stomachs."

━━━ HuggingFace model ━━━
  google/flan-t5-base — already loaded in food_text_to_usda.py.
  We reuse the same T5 instance rather than loading a second copy.
  If not available, falls back to a clean rule-based description formatter.

━━━ SIMILARITY METRIC ━━━
  Cosine similarity on a 5-dim nutrient vector:
    [calories_norm, protein_norm, carbs_norm, fat_norm, fiber_norm]
  Normalised to [0,1] using USDA dataset percentiles.
  Foods close in this space eat similarly but we filter by danger nutrients.

Public function:
  recommend_safe_foods()  — called from POST /recommend/safe-foods
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Flan-T5 description generator
# Reuses the T5 already loaded in food_text_to_usda.py via shared state.
# Falls back to rule-based formatter if T5 not available.
# ─────────────────────────────────────────────────────────────────────────────

_t5_tokenizer = None
_t5_model     = None

def init_t5(tokenizer, model) -> None:
    """Call once from lifespan with the already-loaded T5 objects."""
    global _t5_tokenizer, _t5_model
    _t5_tokenizer = tokenizer
    _t5_model     = model
    log.info("Flan-T5 injected into food_recommender.")


def _generate_with_t5(prompt: str, max_new_tokens: int = 80) -> str:
    """Run Flan-T5 generation. Returns empty string on failure."""
    if _t5_tokenizer is None or _t5_model is None:
        return ""
    try:
        inputs  = _t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = _t5_model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            num_beams      = 4,
            early_stopping = True,
            no_repeat_ngram_size = 3,
        )
        return _t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as exc:
        log.warning(f"T5 generation failed: {exc}")
        return ""


def _natural_name(raw_desc: str) -> str:
    """
    Convert USDA all-caps description to a readable food name.
    E.g. "CHICKEN,BROILERS OR FRYERS,BREAST,MEAT ONLY,CKD,ROASTED"
         → "Chicken breast, roasted"
    """
    # Title case, take first 2–3 meaningful parts
    parts = [p.strip().title() for p in raw_desc.split(",")]
    # Remove abbreviations and junk
    _junk = {"Nfs", "Upc", "Nos", "W/", "Wo/", "Cnd", "Drnd", "Liq", "Ckd"}
    parts = [p for p in parts if p not in _junk and len(p) > 1]
    return ", ".join(parts[:3]) if parts else raw_desc.title()


def _rule_based_description(
    food_name:     str,
    nutrients:     dict,
    safe_reasons:  list[str],
    user_symptoms: list[str],
) -> str:
    """
    Generate a natural recommendation sentence without T5.
    Used as fallback or when T5 output is too short.
    """
    highlights = []
    cal  = nutrients.get("calories")
    prot = nutrients.get("protein")
    fat  = nutrients.get("total_fat")
    fib  = nutrients.get("fiber")
    sod  = nutrients.get("sodium")

    if cal  is not None and cal < 120:  highlights.append("low in calories")
    if prot is not None and prot > 10:  highlights.append("rich in protein")
    if fat  is not None and fat < 5:    highlights.append("low in fat")
    if fib  is not None and fib > 3:    highlights.append("high in fibre")
    if sod  is not None and sod < 100:  highlights.append("low in sodium")

    symptom_str = " and ".join(user_symptoms[:2]) if user_symptoms else "digestive discomfort"
    nutrition_str = ", ".join(highlights[:3]) if highlights else "nutritionally balanced"
    reason_str    = safe_reasons[0].lower() if safe_reasons else "gentle on the gut"

    return (
        f"{food_name} — {nutrition_str}. "
        f"A gut-friendly choice for people prone to {symptom_str}. "
        f"Recommended because it is {reason_str}."
    )


def _t5_recommendation(
    raw_usda_desc: str,
    nutrients:     dict,
    safe_reasons:  list[str],
    user_symptoms: list[str],
) -> str:
    """
    Use Flan-T5 to generate a natural food recommendation sentence.
    Falls back to rule-based if T5 output is poor.
    """
    nutrient_parts = []
    for key, label in [
        ("calories",   "kcal"), ("protein", "g protein"),
        ("total_fat",  "g fat"), ("carbs",   "g carbs"),
        ("sodium",     "mg sodium"), ("sugar", "g sugar"),
    ]:
        v = nutrients.get(key)
        if v is not None:
            nutrient_parts.append(f"{v:.1f}{label}")

    nutrient_str  = ", ".join(nutrient_parts) if nutrient_parts else "balanced nutrition"
    symptom_str   = " and ".join(user_symptoms[:2]) if user_symptoms else "digestive sensitivity"
    reason_str    = "; ".join(safe_reasons[:2]) if safe_reasons else "low in gut-irritating nutrients"
    clean_name    = _natural_name(raw_usda_desc)

    prompt = (
        f"You are a gut health dietitian. Recommend this food in one friendly sentence.\n"
        f"Food: {clean_name}\n"
        f"Nutrition per 100g: {nutrient_str}\n"
        f"Why it is safe: {reason_str}\n"
        f"User's digestive concern: {symptom_str}\n"
        f"Write a warm, natural recommendation sentence (no USDA codes, no percentages, no lists):\n"
        f"Recommendation:"
    )

    t5_output = _generate_with_t5(prompt, max_new_tokens=80)

    # Quality check: T5 output must be >20 chars and not echo the prompt
    if (
        len(t5_output) < 25
        or t5_output.lower().startswith("recommendation:")
        or "usda" in t5_output.lower()
        or t5_output.count(",") > 6   # list-like, not a sentence
    ):
        return _rule_based_description(clean_name, nutrients, safe_reasons, user_symptoms)

    # Capitalise and clean
    t5_output = t5_output.strip().strip('"').strip("'")
    if t5_output and not t5_output[0].isupper():
        t5_output = t5_output[0].upper() + t5_output[1:]
    if t5_output and not t5_output.endswith("."):
        t5_output += "."

    return t5_output


# ─────────────────────────────────────────────────────────────────────────────
# USDA column resolver (mirrors nutrition_scorer)
# ─────────────────────────────────────────────────────────────────────────────

_COL_ALIASES: dict[str, list[str]] = {
    "id":          ["ID","id","NDB_No","fdc_id","FDC_ID"],
    "description": ["Description","description","DESCRIPTION","Long_Desc","NAME"],
    "calories":    ["Calories","calories","Energ_Kcal","ENERG_KCAL","Energy"],
    "protein":     ["Protein","protein","Protein_g"],
    "total_fat":   ["TotalFat","Total_Fat","total_fat","Lipid_Tot","Fat"],
    "carbs":       ["Carbohydrate","carbohydrate","Carbohydrt","Carbs"],
    "sodium":      ["Sodium","sodium","Sodium_mg"],
    "sat_fat":     ["SaturatedFat","Saturated_Fat","saturated_fat","FA_Sat"],
    "cholesterol": ["Cholesterol","cholesterol","Cholestrl"],
    "sugar":       ["Sugar","sugar","Sugar_Tot","Sugars"],
    "fiber":       ["Fiber_TD","FIBER_TD","fiber_td","Fiber","dietary_fiber"],
    "potassium":   ["Potassium","potassium","Potassium_mg"],
    "calcium":     ["Calcium","calcium","Calcium_mg"],
    "vitamin_c":   ["VitaminC","Vitamin_C","vitamin_c","Vit_C"],
}

def _col(df: pd.DataFrame, key: str) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for alias in _COL_ALIASES.get(key, []):
        if alias in df.columns:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None

def _gv(row, df: pd.DataFrame, key: str) -> Optional[float]:
    c = _col(df, key)
    if c is None:
        return None
    try:
        v = float(row[c])
        return v if not math.isnan(v) else None
    except (TypeError, ValueError, KeyError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Nutrient vector for cosine similarity
# ─────────────────────────────────────────────────────────────────────────────

_VECTOR_KEYS = ["calories", "protein", "total_fat", "carbs", "fiber", "sugar", "sodium"]

def _nutrient_vector(row, df: pd.DataFrame, norms: dict) -> np.ndarray:
    """
    Build a normalised nutrient vector for cosine similarity.
    Each nutrient is divided by its 95th-percentile value in the dataset
    → all dimensions are in [0, ~1] range.
    """
    vec = []
    for k in _VECTOR_KEYS:
        v    = _gv(row, df, k)
        norm = norms.get(k, 1.0)
        vec.append((v or 0.0) / norm if norm > 0 else 0.0)
    arr = np.array(vec, dtype=float)
    # Clip to [0, 2] to avoid outliers dominating
    return np.clip(arr, 0.0, 2.0)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────────────────────────────────────
# Symptom → danger nutrients mapping
# Which nutrients to penalise when recommending for users with this symptom
# ─────────────────────────────────────────────────────────────────────────────

_SYMPTOM_DANGER_NUTRIENTS: dict[str, list[str]] = {
    "Heartburn":      ["total_fat", "sat_fat", "sodium", "sugar"],
    "Acid Reflux":    ["total_fat", "sat_fat", "sodium", "sugar"],
    "Bloating":       ["carbs", "sugar", "sodium", "fiber"],
    "Gas":            ["carbs", "sugar", "fiber"],
    "Nausea":         ["total_fat", "sat_fat", "cholesterol"],
    "Cramps":         ["total_fat", "sat_fat", "sodium", "sugar"],
    "Abdominal Pain": ["total_fat", "sodium", "sugar", "cholesterol"],
    "Diarrhea":       ["sugar", "carbs", "total_fat", "sodium"],
    "Constipation":   ["total_fat", "sat_fat", "sodium"],   # low fiber is also a factor
    "Fatigue":        ["sugar", "carbs", "calories"],
}

# Per-nutrient "safe" thresholds per 100g
# Foods BELOW these values get a safety bonus
_SAFE_THRESHOLDS: dict[str, float] = {
    "total_fat":   8.0,
    "sat_fat":     3.0,
    "sodium":      200.0,
    "sugar":       5.0,
    "carbs":       25.0,
    "cholesterol": 50.0,
    "calories":    200.0,
    "fiber":       2.0,    # fiber is a GOOD thing — bonus if present
}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SafeFoodRecommendation:
    rank:            int
    food_name:       str           # clean natural name (not USDA raw)
    recommendation:  str           # Flan-T5 generated friendly description
    safety_score:    float         # 0–1, higher = safer for this user
    similarity_score:float         # 0–1, how similar to user's eating pattern
    combined_score:  float         # weighted final rank score
    key_nutrients:   dict          # per-100g snapshot shown to user
    safe_reasons:    list[str]     # why this food is recommended
    avoid_reasons:   list[str]     # any mild caveats


# ─────────────────────────────────────────────────────────────────────────────
# Main recommendation function
# ─────────────────────────────────────────────────────────────────────────────

def recommend_safe_foods(
    eaten_usda_ids:  list[int],        # foods the user has already eaten
    user_symptoms:   list[str],        # symptoms the user has logged
    user_memory,                       # UserMemory | None
    usda_df:         pd.DataFrame,
    id_col:          str,
    desc_col:        str,
    n:               int = 5,
) -> list[SafeFoodRecommendation]:
    """
    Recommend n safe, nutritionally similar foods for this user.

    Parameters
    ----------
    eaten_usda_ids : USDA IDs the user has previously eaten (exclude these)
    user_symptoms  : list of symptoms this user has reported
    user_memory    : UserMemory for personalised causation avoidance
    usda_df        : full USDA DataFrame
    id_col         : USDA ID column name
    desc_col       : USDA description column name
    n              : number of recommendations (default 5)
    """

    # ── Resolve column names ──────────────────────────────────────────────────
    id_col_r   = _col(usda_df, "id")   or id_col
    desc_col_r = _col(usda_df, "description") or desc_col

    # ── Stage 1: Build user nutrient profile (average of eaten foods) ─────────
    eaten_set = set(str(i) for i in eaten_usda_ids)

    eaten_rows = usda_df[usda_df[id_col_r].astype(str).isin(eaten_set)]

    if eaten_rows.empty:
        # No history — use a generic healthy profile as anchor
        user_profile = {
            "calories": 200.0, "protein": 15.0, "total_fat": 8.0,
            "carbs": 25.0, "fiber": 3.0, "sugar": 5.0, "sodium": 200.0,
        }
        log.info("No eaten foods found — using generic healthy anchor profile.")
    else:
        user_profile = {}
        for k in _VECTOR_KEYS:
            c = _col(usda_df, k)
            if c and c in eaten_rows.columns:
                vals = pd.to_numeric(eaten_rows[c], errors="coerce").dropna()
                if not vals.empty:
                    user_profile[k] = float(vals.mean())
        log.info(f"User nutrient profile built from {len(eaten_rows)} eaten foods.")

    # ── Stage 2: Compute dataset-wide 95th percentile norms for normalisation ──
    norms: dict[str, float] = {}
    for k in _VECTOR_KEYS:
        c = _col(usda_df, k)
        if c and c in usda_df.columns:
            vals = pd.to_numeric(usda_df[c], errors="coerce").dropna()
            if not vals.empty:
                norms[k] = float(np.percentile(vals, 95))
            else:
                norms[k] = 1.0
        else:
            norms[k] = 1.0

    # ── Stage 3: Build user vector ───────────────────────────────────────────
    user_vec_raw = np.array([
        (user_profile.get(k, 0.0)) / norms.get(k, 1.0)
        for k in _VECTOR_KEYS
    ], dtype=float)
    user_vec = np.clip(user_vec_raw, 0.0, 2.0)

    # ── Stage 4: Identify danger nutrients from user symptoms + memory ────────
    danger_nutrients: set[str] = set()
    for sym in user_symptoms:
        danger_nutrients.update(_SYMPTOM_DANGER_NUTRIENTS.get(sym, []))

    # From memory: high-causation foods contribute their elevated nutrients
    memory_danger: dict[str, float] = {}   # nutrient → extra penalty weight
    if user_memory is not None:
        for key, prior in user_memory.priors.items():
            if prior.posterior_mean < 0.55:
                continue   # only care about high-causation pairs
            try:
                uid_str, sym = key.split(":", 1)
                uid = int(uid_str)
            except ValueError:
                continue
            row_mask = usda_df[id_col_r].astype(str) == uid_str
            if not row_mask.any():
                continue
            row = usda_df[row_mask].iloc[0]
            for dn in _SYMPTOM_DANGER_NUTRIENTS.get(sym, []):
                v  = _gv(row, usda_df, dn)
                th = _SAFE_THRESHOLDS.get(dn, 1.0)
                if v is not None and v > th:
                    w = memory_danger.get(dn, 0.0)
                    memory_danger[dn] = max(w, prior.posterior_mean)

    all_danger = danger_nutrients | set(memory_danger.keys())

    # ── Stage 5: Score every non-eaten USDA food ─────────────────────────────
    log.info(f"Scoring USDA candidates (danger nutrients: {all_danger}) ...")

    candidates: list[dict] = []

    # Pre-filter to reduce computation: foods within ±60% of user calorie range
    user_cal  = user_profile.get("calories", 200.0)
    cal_col   = _col(usda_df, "calories")
    if cal_col:
        df_filtered = usda_df[
            (pd.to_numeric(usda_df[cal_col], errors="coerce").fillna(0) >= user_cal * 0.30) &
            (pd.to_numeric(usda_df[cal_col], errors="coerce").fillna(9999) <= user_cal * 1.80)
        ]
    else:
        df_filtered = usda_df

    # Exclude already-eaten foods
    df_filtered = df_filtered[~df_filtered[id_col_r].astype(str).isin(eaten_set)]

    # Subsample if too large for speed (score top 3000 by calorie proximity)
    if len(df_filtered) > 3000:
        if cal_col:
            df_filtered["_cal_dist"] = (
                pd.to_numeric(df_filtered[cal_col], errors="coerce").fillna(9999) - user_cal
            ).abs()
            df_filtered = df_filtered.nsmallest(3000, "_cal_dist").drop(columns=["_cal_dist"])
        else:
            df_filtered = df_filtered.sample(3000, random_state=42)

    for _, row in df_filtered.iterrows():
        desc = str(row.get(desc_col_r, ""))
        if not desc or len(desc) < 3:
            continue

        # Skip raw technical/industrial entries
        if any(junk in desc.upper() for junk in [
            "INFANT", "BABY FOOD", "FORMULA", "NFS", "NOS",
            "IMITATION", "SUBSTITUTE", "FLAVORING", "EXTRACT",
        ]):
            continue

        # Nutrient snapshot
        nuts: dict[str, Optional[float]] = {k: _gv(row, usda_df, k) for k in _VECTOR_KEYS + ["sat_fat", "cholesterol"]}

        # Similarity to user eating pattern
        cand_vec  = _nutrient_vector(row, usda_df, norms)
        sim_score = _cosine(user_vec, cand_vec)

        # Safety score: reward low values of danger nutrients
        safety   = 0.0
        n_scored = 0
        safe_reasons: list[str] = []
        avoid_reasons: list[str] = []

        for dn in all_danger:
            v  = nuts.get(dn)
            th = _SAFE_THRESHOLDS.get(dn, 1.0)
            if v is None:
                continue
            n_scored += 1
            if v <= th * 0.5:
                safety += 1.0
                unit = "mg" if dn in ("sodium","cholesterol") else ("kcal" if dn == "calories" else "g")
                safe_reasons.append(f"Very low {dn.replace('_',' ')} ({v:.1f}{unit}/100g)")
            elif v <= th:
                safety += 0.6
                safe_reasons.append(f"Moderate {dn.replace('_',' ')} — within safe range")
            else:
                safety += 0.0
                unit = "mg" if dn in ("sodium","cholesterol") else ("kcal" if dn == "calories" else "g")
                avoid_reasons.append(f"Elevated {dn.replace('_',' ')} ({v:.1f}{unit}/100g)")

        # Bonus for high fiber (gut-protective)
        fib = nuts.get("fiber")
        if fib is not None and fib >= _SAFE_THRESHOLDS["fiber"]:
            safety += 0.5
            safe_reasons.append(f"Good fibre content ({fib:.1f}g/100g) — supports gut motility")
            n_scored += 1

        safety_score = (safety / n_scored) if n_scored > 0 else 0.5

        # Personal memory penalty: if user's memory says this specific food
        # (or nutritionally identical food) is high-causation, penalise it
        uid_str = str(row.get(id_col_r, ""))
        memory_penalty = 0.0
        if user_memory is not None:
            for sym in user_symptoms:
                k   = f"{uid_str}:{sym}"
                pr  = user_memory.priors.get(k)
                if pr and pr.posterior_mean > 0.55:
                    memory_penalty = max(memory_penalty, pr.posterior_mean - 0.55)

        safety_score = max(0.0, safety_score - memory_penalty * 0.8)

        # Combined: 55% safety, 35% similarity, 10% variety (random jitter for diversity)
        combined = 0.55 * safety_score + 0.35 * sim_score + 0.10 * np.random.uniform(0, 0.3)

        candidates.append({
            "desc":          desc,
            "nuts":          nuts,
            "sim_score":     round(sim_score, 4),
            "safety_score":  round(safety_score, 4),
            "combined":      round(combined, 4),
            "safe_reasons":  safe_reasons[:3],
            "avoid_reasons": avoid_reasons[:2],
        })

    if not candidates:
        log.warning("No candidates found — returning empty recommendations.")
        return []

    # Sort and take top N (with diversity: max 2 per calorie bin)
    candidates.sort(key=lambda x: x["combined"], reverse=True)

    # Deduplicate: avoid recommending very similar foods (cosine > 0.97)
    diverse: list[dict] = []
    seen_vecs: list[np.ndarray] = []

    for c in candidates:
        c_row = df_filtered[df_filtered[desc_col_r] == c["desc"]]
        if c_row.empty:
            continue
        c_vec = _nutrient_vector(c_row.iloc[0], usda_df, norms)
        too_similar = any(_cosine(c_vec, sv) > 0.97 for sv in seen_vecs)
        if not too_similar:
            diverse.append(c)
            seen_vecs.append(c_vec)
        if len(diverse) >= n:
            break

    top = diverse[:n]

    # ── Stage 6: Flan-T5 natural description ─────────────────────────────────
    log.info(f"Generating natural descriptions for {len(top)} recommendations via Flan-T5 ...")

    results: list[SafeFoodRecommendation] = []

    for rank, cand in enumerate(top, 1):
        clean_name = _natural_name(cand["desc"])
        nuts       = {k: v for k, v in cand["nuts"].items() if v is not None}

        recommendation_text = _t5_recommendation(
            raw_usda_desc = cand["desc"],
            nutrients     = nuts,
            safe_reasons  = cand["safe_reasons"],
            user_symptoms = [s for s in user_symptoms if s],
        )

        # Build key_nutrients dict for API response (clean labels)
        key_nutrients = {}
        label_map = {
            "calories":   "Calories (kcal/100g)",
            "protein":    "Protein (g/100g)",
            "total_fat":  "Total Fat (g/100g)",
            "carbs":      "Carbohydrates (g/100g)",
            "fiber":      "Fibre (g/100g)",
            "sugar":      "Sugar (g/100g)",
            "sodium":     "Sodium (mg/100g)",
            "sat_fat":    "Saturated Fat (g/100g)",
            "cholesterol":"Cholesterol (mg/100g)",
        }
        for k, label in label_map.items():
            v = cand["nuts"].get(k)
            if v is not None:
                key_nutrients[label] = round(v, 2)

        results.append(SafeFoodRecommendation(
            rank             = rank,
            food_name        = clean_name,
            recommendation   = recommendation_text,
            safety_score     = cand["safety_score"],
            similarity_score = cand["sim_score"],
            combined_score   = cand["combined"],
            key_nutrients    = key_nutrients,
            safe_reasons     = cand["safe_reasons"],
            avoid_reasons    = cand["avoid_reasons"],
        ))

    log.info(f"Generated {len(results)} food recommendations.")
    return results