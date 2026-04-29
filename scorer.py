"""
digestion_score/scorer.py
─────────────────────────
Pure-Python digestive health scoring engine.
No ML model, no API key, no internet required.
Works offline, handles unlimited requests, instant response.

Usage (standalone):
    from scorer import calculate_score, DigestiveInput
    result = calculate_score(DigestiveInput(gender="Female", age=32, sleep_hours=6.0,
                                            weight_kg=65, foods=["Spicy","Dairy"],
                                            symptoms=["Bloating","Heartburn"]))
    print(result.model_dump_json(indent=2))
"""

from __future__ import annotations
from enum import Enum
from typing import List
from pydantic import BaseModel, Field, field_validator

_BASELINE_HEALTH_SCORE = 90


# ─────────────────────────────────────────────────────────────────────────────
# Enums  (validated input values — prevents typos / bad data)
# ─────────────────────────────────────────────────────────────────────────────

class Gender(str, Enum):
    male   = "Male"
    female = "Female"
    other  = "Other"


class FoodTrigger(str, Enum):
    dairy          = "Dairy"
    gluten         = "Gluten"
    spicy          = "Spicy"
    fried          = "Fried"
    sugar          = "Sugar"
    caffeine       = "Caffeine"
    processed_food = "Processed Food"
    other          = "Other"


class Symptom(str, Enum):
    bloating        = "Bloating"
    gas             = "Gas"
    abdominal_pain  = "Abdominal Pain"
    nausea          = "Nausea"
    heartburn       = "Heartburn"
    diarrhea        = "Diarrhea"
    constipation    = "Constipation"
    fatigue         = "Fatigue"


# ─────────────────────────────────────────────────────────────────────────────
# Scoring tables
# ─────────────────────────────────────────────────────────────────────────────

_FOOD_WEIGHTS: dict[str, dict] = {
    "Dairy":          {"deduct": 5,  "risk": "medium"},
    "Gluten":         {"deduct": 6,  "risk": "medium"},
    "Spicy":          {"deduct": 7,  "risk": "high"},
    "Fried":          {"deduct": 8,  "risk": "high"},
    "Sugar":          {"deduct": 5,  "risk": "medium"},
    "Caffeine":       {"deduct": 5,  "risk": "medium"},
    "Processed Food": {"deduct": 9,  "risk": "high"},
    "Other":          {"deduct": 3,  "risk": "low"},
}

_SYMPTOM_WEIGHTS: dict[str, dict] = {
    "Bloating":       {"deduct": 7,  "severity": "moderate"},
    "Gas":            {"deduct": 5,  "severity": "mild"},
    "Abdominal Pain": {"deduct": 12, "severity": "severe"},
    "Nausea":         {"deduct": 9,  "severity": "moderate"},
    "Heartburn":      {"deduct": 10, "severity": "moderate"},
    "Diarrhea":       {"deduct": 12, "severity": "severe"},
    "Constipation":   {"deduct": 10, "severity": "moderate"},
    "Fatigue":        {"deduct": 6,  "severity": "mild"},
}

_FOOD_CONCERN: dict[str, str] = {
    "Dairy":          "Dairy is a common trigger for bloating & gas due to lactose malabsorption.",
    "Gluten":         "Gluten can cause gut inflammation and permeability issues in sensitive individuals.",
    "Spicy":          "Spicy food irritates the gastric lining and worsens heartburn & IBS symptoms.",
    "Fried":          "Fried food slows gastric emptying and significantly increases acid reflux risk.",
    "Sugar":          "High sugar intake feeds harmful gut bacteria and disrupts the microbiome balance.",
    "Caffeine":       "Caffeine over-stimulates the bowel and can worsen diarrhea and acid reflux.",
    "Processed Food": "Processed food is low in fibre and high in additives, impairing gut motility.",
    "Other":          "Unspecified dietary habits may still contribute to overall gut load.",
}

_SYMPTOM_CONCERN: dict[str, str] = {
    "Bloating":       "Bloating suggests fermentation issues, gut dysbiosis, or motility problems.",
    "Gas":            "Excess gas is often linked to microbiome imbalance or food intolerance.",
    "Abdominal Pain": "Abdominal pain is a key IBS/IBD indicator and warrants clinical evaluation.",
    "Nausea":         "Nausea may signal gastritis, H. pylori infection, or delayed gastric emptying.",
    "Heartburn":      "Heartburn indicates acid reflux or GERD risk — dietary changes can help significantly.",
    "Diarrhea":       "Diarrhea disrupts nutrient absorption, causes dehydration and electrolyte loss.",
    "Constipation":   "Constipation usually reflects low dietary fibre or inadequate hydration.",
    "Fatigue":        "Digestive fatigue often points to gut-brain axis disruption or nutrient malabsorption.",
}

_FOOD_REC: dict[str, str] = {
    "Dairy":          "Try lactase supplements or switch to plant-based milks (oat, almond, rice).",
    "Gluten":         "Run a 2-week gluten-free trial. Substitute with rice, quinoa, or buckwheat.",
    "Spicy":          "Limit spicy meals to ≤2x/week. Pair with cooling foods like plain yogurt.",
    "Fried":          "Replace deep-fried items with air-fried, baked, or steamed alternatives.",
    "Sugar":          "Cut added sugars. Increase prebiotic fibre — oats, bananas, garlic, leeks.",
    "Caffeine":       "Cap caffeine at 1 cup/day. Green or herbal tea is gentler on the gut.",
    "Processed Food": "Aim for 25–30g dietary fibre/day from whole vegetables and legumes.",
    "Other":          "Keep a detailed food diary for 2 weeks to identify personal dietary triggers.",
}

_SYMPTOM_REC: dict[str, str] = {
    "Bloating":       "Eat slowly, avoid carbonated drinks, and try peppermint tea after meals.",
    "Gas":            "Temporarily reduce high-FODMAP foods. Try a Lactobacillus probiotic course.",
    "Abdominal Pain": "Track pain patterns in a food-symptom diary. Consult a gastroenterologist.",
    "Nausea":         "Eat smaller, more frequent meals. Avoid lying down within 2 hours of eating.",
    "Heartburn":      "Elevate your head while sleeping. Avoid eating within 3 hours of bedtime.",
    "Diarrhea":       "Stay hydrated with oral electrolytes. Consider a short probiotic course.",
    "Constipation":   "Drink ≥2L water/day. Add psyllium husk, prunes, or ground flaxseed.",
    "Fatigue":        "Check serum iron, B12, and Vitamin D — gut dysfunction impairs their absorption.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic I/O models
# ─────────────────────────────────────────────────────────────────────────────

class DigestiveInput(BaseModel):
    """All fields sent by the client."""
    gender:       Gender            = Field(...,  description="Biological sex / gender identity")
    age:          int               = Field(...,  ge=1, le=120, description="Age in years")
    sleep_hours:  float             = Field(...,  ge=0, le=24,  description="Average nightly sleep in hours")
    weight_kg:    float             = Field(...,  ge=1,         description="Body weight in kilograms")
    foods:        List[FoodTrigger] = Field(default_factory=list, description="Foods regularly consumed")
    symptoms:     List[Symptom]     = Field(default_factory=list, description="Current digestive symptoms")

    @field_validator("foods", "symptoms", mode="before")
    @classmethod
    def deduplicate(cls, v):
        seen, out = set(), []
        for item in v:
            if item not in seen:
                seen.add(item); out.append(item)
        return out


class Concern(BaseModel):
    category:    str = Field(..., description="'food', 'symptom', 'lifestyle', or 'compound'")
    description: str


class Recommendation(BaseModel):
    priority: int  = Field(..., description="1 = highest priority")
    advice:   str


class ScoreBreakdown(BaseModel):
    base_score:      int = _BASELINE_HEALTH_SCORE
    age_penalty:     int
    sleep_penalty:   int
    weight_penalty:  int
    gender_penalty:  int
    food_penalty:    int
    symptom_penalty: int
    compound_penalty:int
    final_score:     int


class DigestiveResult(BaseModel):
    """Full response returned by calculate_score()."""
    score:           int   = Field(..., ge=0, le=100)
    grade:           str   = Field(..., description="Excellent | Good | Fair | Poor | Critical")
    tagline:         str
    breakdown:       ScoreBreakdown
    concerns:        List[Concern]
    recommendations: List[Recommendation]


# ─────────────────────────────────────────────────────────────────────────────
# Core engine
# ─────────────────────────────────────────────────────────────────────────────

def calculate_score(data: DigestiveInput) -> DigestiveResult:
    """
    Deterministic rule-based digestive health scorer.
    Returns a DigestiveResult — no I/O, no randomness, fully testable.
    """
    score = _BASELINE_HEALTH_SCORE
    concerns:       list[Concern]        = []
    recommendations:list[Recommendation] = []
    rec_seen:       set[str]             = set()
    priority_counter = 1

    def add_concern(category: str, desc: str):
        concerns.append(Concern(category=category, description=desc))

    def add_rec(advice: str):
        nonlocal priority_counter
        if advice not in rec_seen:
            rec_seen.add(advice)
            recommendations.append(Recommendation(priority=priority_counter, advice=advice))
            priority_counter += 1

    # ── Age penalty ──────────────────────────────────────────────────────────
    age_pen = 0
    if data.age > 60:
        age_pen = 8
        add_concern("lifestyle", "Age >60: digestive enzyme production declines and gut motility slows with age.")
    elif data.age > 45:
        age_pen = 4
        add_concern("lifestyle", "Age >45: gradual reduction in digestive efficiency is common.")
    score -= age_pen

    # ── Sleep penalty ─────────────────────────────────────────────────────────
    sleep_pen = 0
    if data.sleep_hours < 5:
        sleep_pen = 10
        add_concern("lifestyle", f"Severe sleep deprivation ({data.sleep_hours}h) heavily disrupts the gut-brain axis and bowel motility.")
        add_rec("Prioritise 7–8h sleep: poor sleep increases intestinal permeability and worsens IBS symptoms.")
    elif data.sleep_hours < 6.5:
        sleep_pen = 6
        add_concern("lifestyle", f"Poor sleep ({data.sleep_hours}h) is linked to increased gut permeability and systemic inflammation.")
        add_rec("Aim for at least 7h of sleep. Sleep deprivation directly worsens gut barrier function.")
    elif data.sleep_hours > 9.5:
        sleep_pen = 4
        add_concern("lifestyle", f"Excessive sleep ({data.sleep_hours}h) may reflect fatigue-driven disorders that affect digestion.")
    score -= sleep_pen

    # ── Weight penalty ───────────────────────────────────────────────────────
    weight_pen = 0
    if data.weight_kg > 100:
        weight_pen = 5
        add_concern("lifestyle", "Elevated body weight increases intra-abdominal pressure, raising GERD and reflux risk.")
    elif data.weight_kg < 45:
        weight_pen = 4
        add_concern("lifestyle", "Very low body weight may indicate nutritional deficiency affecting the gut mucosal lining.")
    score -= weight_pen

    # ── Gender penalty ───────────────────────────────────────────────────────
    gender_pen = 0
    if data.gender == Gender.female:
        gender_pen = 3
        add_concern("lifestyle", "Women have statistically higher IBS prevalence and hormonal fluctuations that affect gut motility.")
        add_rec("Track symptoms alongside your menstrual cycle — oestrogen and progesterone significantly influence bowel behaviour.")
    score -= gender_pen

    # ── Food penalties ───────────────────────────────────────────────────────
    food_pen = 0
    high_risk_foods: list[str] = []
    for food in data.foods:
        w = _FOOD_WEIGHTS[food.value]
        food_pen += w["deduct"]
        if w["risk"] == "high":
            high_risk_foods.append(food.value)
        add_concern("food", _FOOD_CONCERN[food.value])
        add_rec(_FOOD_REC[food.value])
    score -= food_pen

    # ── Symptom penalties ────────────────────────────────────────────────────
    symptom_pen = 0
    severe_symptoms: list[str] = []
    for sym in data.symptoms:
        w = _SYMPTOM_WEIGHTS[sym.value]
        symptom_pen += w["deduct"]
        if w["severity"] == "severe":
            severe_symptoms.append(sym.value)
        add_concern("symptom", _SYMPTOM_CONCERN[sym.value])
        add_rec(_SYMPTOM_REC[sym.value])
    score -= symptom_pen

    # ── Compound rules ───────────────────────────────────────────────────────
    compound_pen = 0
    symptom_vals = {s.value for s in data.symptoms}
    food_vals    = {f.value for f in data.foods}

    if len(high_risk_foods) >= 2:
        compound_pen += 5
        add_concern("compound",
            f"Multiple high-risk foods ({', '.join(high_risk_foods)}) are combining to place heavy stress on your digestive system.")

    if len(severe_symptoms) >= 2:
        compound_pen += 6
        add_concern("compound",
            "Multiple severe symptoms co-occurring — clinical evaluation by a gastroenterologist is strongly recommended.")
        add_rec("Keep a detailed food-symptom diary for 2 weeks and share it with your GP before your appointment.")

    if "Diarrhea" in symptom_vals and "Constipation" in symptom_vals:
        compound_pen += 8
        add_concern("compound",
            "Alternating diarrhoea and constipation is the hallmark pattern of IBS-Mixed type (IBS-M).")
        add_rec("Alternating bowel pattern detected — ask your doctor about IBS-M and a low-FODMAP dietary protocol.")

    if "Heartburn" in symptom_vals and food_vals & {"Spicy", "Fried", "Caffeine"}:
        compound_pen += 5
        add_concern("compound",
            "Your current diet directly aggravates your heartburn — dietary modification alone may provide significant relief.")

    if "Bloating" in symptom_vals and food_vals & {"Dairy", "Gluten"}:
        compound_pen += 4
        add_concern("compound",
            "Bloating combined with Dairy/Gluten consumption strongly suggests a food intolerance (lactose or coeliac).")
        add_rec("Consider a supervised elimination diet to confirm Dairy or Gluten intolerance before pursuing lab tests.")

    score -= compound_pen

    # ── Universal recommendation ─────────────────────────────────────────────
    add_rec("Stay hydrated: 2–2.5L of water/day is the single most impactful habit for overall digestive health.")

    # ── Final clamp & grade ──────────────────────────────────────────────────
    final = max(0, min(100, score))

    if   final >= 85: grade, tagline = "Excellent", "Gut health is in great shape — maintain your current habits."
    elif final >= 70: grade, tagline = "Good",      "Minor improvements in diet or lifestyle could make a real difference."
    elif final >= 50: grade, tagline = "Fair",      "Several areas need attention — targeted changes are recommended."
    elif final >= 30: grade, tagline = "Poor",      "Digestive system is under significant stress — act now."
    else:             grade, tagline = "Critical",  "Urgent attention recommended — please consult a healthcare professional."

    breakdown = ScoreBreakdown(
        base_score       = _BASELINE_HEALTH_SCORE,
        age_penalty      = age_pen,
        sleep_penalty    = sleep_pen,
        weight_penalty   = weight_pen,
        gender_penalty   = gender_pen,
        food_penalty     = food_pen,
        symptom_penalty  = symptom_pen,
        compound_penalty = compound_pen,
        final_score      = final,
    )

    return DigestiveResult(
        score           = final,
        grade           = grade,
        tagline         = tagline,
        breakdown       = breakdown,
        concerns        = concerns,
        recommendations = recommendations,
    )
