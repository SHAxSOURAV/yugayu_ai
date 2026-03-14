"""
demo_seed.py
────────────
Seeds MongoDB with 3 realistic users and their 7-day food + symptom history.

Run:
    python demo_seed.py

Prerequisites:
    pip install pymongo
    MongoDB running on localhost:27017 (or set MONGO_URI env var)

What it creates:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  User 1 — Anika Rahman (28F)                                        │
  │    Profile : Low sleep, dairy sensitivity, heartburn pattern        │
  │    7 days  : 3 meals/day + evening snacks                          │
  │    Symptoms: Heartburn (recurring after fatty dinners)              │
  │    Pattern : Fried food at dinner → heartburn 1-2h later           │
  ├─────────────────────────────────────────────────────────────────────┤
  │  User 2 — Rafiul Islam (35M)                                        │
  │    Profile : IBS pattern, gluten + spicy sensitivity                │
  │    7 days  : 3 meals/day, larger portions                          │
  │    Symptoms: Bloating + Cramps (after lunch, spicy food pattern)   │
  │    Pattern : Spicy lunch → bloating + cramps 2-4h later            │
  ├─────────────────────────────────────────────────────────────────────┤
  │  User 3 — Sadia Hossain (45F)                                       │
  │    Profile : Constipation + fatigue, low fibre diet                │
  │    7 days  : 2-3 meals/day, snacks                                 │
  │    Symptoms: Constipation (next morning), Fatigue after meals      │
  │    Pattern : Low-fibre dinner → constipation 12-24h later          │
  └─────────────────────────────────────────────────────────────────────┘

After seeding, the script runs verification queries to prove data is stored,
and prints a full dashboard summary for each user.
"""

from __future__ import annotations

import sys
import logging
from datetime import datetime, timezone, timedelta
from pprint import pprint

# ── Setup logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    stream = sys.stdout,
)
log = logging.getLogger(__name__)


# ── Connect to MongoDB ────────────────────────────────────────────────────────
from database import db, MongoUserMemoryStore
from user_symptom_memory import UserMemory, auto_update_from_logs
from food_symptom_predictor import FoodLogEntry, SymptomLogEntry

db.connect()

# Use MongoDB-backed memory store for all seed operations
memory_store = MongoUserMemoryStore()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def dt(days_ago: float, hour: int, minute: int = 0) -> datetime:
    """Build a timezone-aware datetime: N days ago at HH:MM UTC."""
    base = datetime.now(timezone.utc).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    return base - timedelta(days=days_ago)


def drop_all() -> None:
    """Clear all collections for a clean seed."""
    for col in ["users", "food_logs", "symptom_logs", "user_memories", "score_history"]:
        db._col(col).drop()
    log.info("All collections dropped — fresh seed.")


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  USER 1 — Anika Rahman
#  28F · heartburn + acid reflux · fried food & caffeine triggers
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def seed_anika() -> None:
    log.info("━━━ Seeding User 1: Anika Rahman ━━━")

    # ── Profile ───────────────────────────────────────────────────────────────
    db.upsert_user({
        "user_id":        "user_anika",
        "name":           "Anika Rahman",
        "age":            28,
        "gender":         "Female",
        "sleep_hours":    5.5,
        "weight_kg":      58.0,
        "foods":          ["Fried", "Caffeine", "Dairy"],
        "symptoms":       ["Heartburn", "Acid Reflux"],
        "onboarding_score": 52,
        "current_score":  52,
        "grade":          "Fair",
    })

    # ── 7-day food logs (USDA IDs from the dataset) ───────────────────────────
    # USDA IDs used (realistic IDs from demomaster dataset):
    #   5064  = Fried chicken breast          (high fat, high sodium)
    #   1079  = Whole milk                    (dairy, moderate fat)
    #  11124  = Raw carrots                   (safe, low everything)
    #   9040  = Banana, raw                   (safe, moderate carbs)
    #   8121  = Oatmeal, cooked               (safe, high fibre)
    #   4058  = Olive oil                     (fat, but healthy)
    #  18033  = Croissant, butter             (high fat, high sat-fat)
    #   9206  = Orange juice                  (moderate sugar)
    #  16057  = Hummus                        (moderate, healthy)
    #   1017  = Cream cheese                  (high fat, dairy)

    food_logs_anika = [
        # ── Day 7 ago ─────────────────────────────────────────────────────────
        {"user_id":"user_anika","usda_id":8121, "usda_description":"Oatmeal, cooked",        "quantity_g":250,"unit":"g","meal_type":"Breakfast","logged_at":dt(7, 8)},
        {"user_id":"user_anika","usda_id":9040, "usda_description":"Banana, raw",            "quantity_g":120,"unit":"g","meal_type":"Breakfast","logged_at":dt(7, 8,30)},
        {"user_id":"user_anika","usda_id":11124,"usda_description":"Carrots, raw",           "quantity_g":100,"unit":"g","meal_type":"Lunch",    "logged_at":dt(7,13)},
        {"user_id":"user_anika","usda_id":5064, "usda_description":"Fried chicken breast",   "quantity_g":220,"unit":"g","meal_type":"Dinner",   "logged_at":dt(7,19)},
        {"user_id":"user_anika","usda_id":1079, "usda_description":"Whole milk",             "quantity_g":200,"unit":"ml","meal_type":"Snack",   "logged_at":dt(7,21)},

        # ── Day 6 ago ─────────────────────────────────────────────────────────
        {"user_id":"user_anika","usda_id":18033,"usda_description":"Croissant, butter",      "quantity_g":80, "unit":"g","meal_type":"Breakfast","logged_at":dt(6, 8)},
        {"user_id":"user_anika","usda_id":9206, "usda_description":"Orange juice",           "quantity_g":240,"unit":"ml","meal_type":"Breakfast","logged_at":dt(6, 8,15)},
        {"user_id":"user_anika","usda_id":16057,"usda_description":"Hummus",                 "quantity_g":80, "unit":"g","meal_type":"Lunch",    "logged_at":dt(6,13)},
        {"user_id":"user_anika","usda_id":11124,"usda_description":"Carrots, raw",           "quantity_g":100,"unit":"g","meal_type":"Lunch",    "logged_at":dt(6,13,15)},
        {"user_id":"user_anika","usda_id":5064, "usda_description":"Fried chicken breast",   "quantity_g":250,"unit":"g","meal_type":"Dinner",   "logged_at":dt(6,19,30)},
        {"user_id":"user_anika","usda_id":1017, "usda_description":"Cream cheese",           "quantity_g":30, "unit":"g","meal_type":"Snack",    "logged_at":dt(6,22)},

        # ── Day 5 ago ─────────────────────────────────────────────────────────
        {"user_id":"user_anika","usda_id":8121, "usda_description":"Oatmeal, cooked",        "quantity_g":250,"unit":"g","meal_type":"Breakfast","logged_at":dt(5, 7,30)},
        {"user_id":"user_anika","usda_id":9040, "usda_description":"Banana, raw",            "quantity_g":100,"unit":"g","meal_type":"Breakfast","logged_at":dt(5, 7,45)},
        {"user_id":"user_anika","usda_id":16057,"usda_description":"Hummus",                 "quantity_g":90, "unit":"g","meal_type":"Lunch",    "logged_at":dt(5,12,30)},
        {"user_id":"user_anika","usda_id":11124,"usda_description":"Carrots, raw",           "quantity_g":120,"unit":"g","meal_type":"Lunch",    "logged_at":dt(5,12,45)},
        {"user_id":"user_anika","usda_id":18033,"usda_description":"Croissant, butter",      "quantity_g":90, "unit":"g","meal_type":"Dinner",   "logged_at":dt(5,19)},
        {"user_id":"user_anika","usda_id":5064, "usda_description":"Fried chicken breast",   "quantity_g":200,"unit":"g","meal_type":"Dinner",   "logged_at":dt(5,19,30)},

        # ── Day 4 ago ─────────────────────────────────────────────────────────
        {"user_id":"user_anika","usda_id":9206, "usda_description":"Orange juice",           "quantity_g":200,"unit":"ml","meal_type":"Breakfast","logged_at":dt(4, 8)},
        {"user_id":"user_anika","usda_id":8121, "usda_description":"Oatmeal, cooked",        "quantity_g":200,"unit":"g","meal_type":"Breakfast","logged_at":dt(4, 8,20)},
        {"user_id":"user_anika","usda_id":11124,"usda_description":"Carrots, raw",           "quantity_g":80, "unit":"g","meal_type":"Lunch",    "logged_at":dt(4,13)},
        {"user_id":"user_anika","usda_id":4058, "usda_description":"Olive oil",              "quantity_g":15, "unit":"g","meal_type":"Dinner",   "logged_at":dt(4,19)},
        {"user_id":"user_anika","usda_id":5064, "usda_description":"Fried chicken breast",   "quantity_g":230,"unit":"g","meal_type":"Dinner",   "logged_at":dt(4,19,30)},
        {"user_id":"user_anika","usda_id":1079, "usda_description":"Whole milk",             "quantity_g":180,"unit":"ml","meal_type":"Snack",   "logged_at":dt(4,21,30)},

        # ── Day 3 ago ─────────────────────────────────────────────────────────
        {"user_id":"user_anika","usda_id":18033,"usda_description":"Croissant, butter",      "quantity_g":85, "unit":"g","meal_type":"Breakfast","logged_at":dt(3, 8)},
        {"user_id":"user_anika","usda_id":16057,"usda_description":"Hummus",                 "quantity_g":80, "unit":"g","meal_type":"Lunch",    "logged_at":dt(3,12,30)},
        {"user_id":"user_anika","usda_id":9040, "usda_description":"Banana, raw",            "quantity_g":120,"unit":"g","meal_type":"Snack",    "logged_at":dt(3,16)},
        {"user_id":"user_anika","usda_id":5064, "usda_description":"Fried chicken breast",   "quantity_g":260,"unit":"g","meal_type":"Dinner",   "logged_at":dt(3,20)},
        {"user_id":"user_anika","usda_id":1017, "usda_description":"Cream cheese",           "quantity_g":25, "unit":"g","meal_type":"Snack",    "logged_at":dt(3,22,30)},

        # ── Day 2 ago ─────────────────────────────────────────────────────────
        {"user_id":"user_anika","usda_id":8121, "usda_description":"Oatmeal, cooked",        "quantity_g":250,"unit":"g","meal_type":"Breakfast","logged_at":dt(2, 7,45)},
        {"user_id":"user_anika","usda_id":9206, "usda_description":"Orange juice",           "quantity_g":180,"unit":"ml","meal_type":"Breakfast","logged_at":dt(2, 8)},
        {"user_id":"user_anika","usda_id":11124,"usda_description":"Carrots, raw",           "quantity_g":100,"unit":"g","meal_type":"Lunch",    "logged_at":dt(2,13)},
        {"user_id":"user_anika","usda_id":5064, "usda_description":"Fried chicken breast",   "quantity_g":240,"unit":"g","meal_type":"Dinner",   "logged_at":dt(2,19,15)},
        {"user_id":"user_anika","usda_id":1079, "usda_description":"Whole milk",             "quantity_g":200,"unit":"ml","meal_type":"Snack",   "logged_at":dt(2,21)},

        # ── Day 1 ago ─────────────────────────────────────────────────────────
        {"user_id":"user_anika","usda_id":18033,"usda_description":"Croissant, butter",      "quantity_g":80, "unit":"g","meal_type":"Breakfast","logged_at":dt(1, 8,30)},
        {"user_id":"user_anika","usda_id":16057,"usda_description":"Hummus",                 "quantity_g":90, "unit":"g","meal_type":"Lunch",    "logged_at":dt(1,13)},
        {"user_id":"user_anika","usda_id":11124,"usda_description":"Carrots, raw",           "quantity_g":100,"unit":"g","meal_type":"Lunch",    "logged_at":dt(1,13,20)},
        {"user_id":"user_anika","usda_id":5064, "usda_description":"Fried chicken breast",   "quantity_g":230,"unit":"g","meal_type":"Dinner",   "logged_at":dt(1,19,45)},
        {"user_id":"user_anika","usda_id":1017, "usda_description":"Cream cheese",           "quantity_g":30, "unit":"g","meal_type":"Snack",    "logged_at":dt(1,22)},
    ]

    n_food = db.insert_food_logs_bulk(food_logs_anika)
    log.info(f"  Inserted {n_food} food logs for Anika")

    # ── 7-day symptom logs ────────────────────────────────────────────────────
    # Pattern: Fried chicken at dinner (7pm) → heartburn at ~9pm (2h later)
    #          Cream cheese late night → acid reflux around midnight

    symptom_logs_anika = [
        {"user_id":"user_anika","symptom":"Heartburn",   "severity":"Moderate","logged_at":dt(7,21,30),"notes":"After fried chicken dinner"},
        {"user_id":"user_anika","symptom":"Acid Reflux", "severity":"Mild",    "logged_at":dt(7,23,0), "notes":"Milk before bed"},
        {"user_id":"user_anika","symptom":"Heartburn",   "severity":"Severe",  "logged_at":dt(6,21,45),"notes":"Fried chicken + cream cheese"},
        {"user_id":"user_anika","symptom":"Heartburn",   "severity":"Moderate","logged_at":dt(5,21,15),"notes":"After heavy dinner"},
        {"user_id":"user_anika","symptom":"Nausea",      "severity":"Mild",    "logged_at":dt(5,22,0), "notes":"Post dinner discomfort"},
        {"user_id":"user_anika","symptom":"Heartburn",   "severity":"Severe",  "logged_at":dt(4,21,30),"notes":"Fried food again"},
        {"user_id":"user_anika","symptom":"Acid Reflux", "severity":"Moderate","logged_at":dt(4,23,30),"notes":"Woke up with reflux"},
        {"user_id":"user_anika","symptom":"Heartburn",   "severity":"Moderate","logged_at":dt(3,22,15),"notes":"After dinner"},
        {"user_id":"user_anika","symptom":"Acid Reflux", "severity":"Mild",    "logged_at":dt(3,23,45),"notes":"Cream cheese late night"},
        {"user_id":"user_anika","symptom":"Heartburn",   "severity":"Severe",  "logged_at":dt(2,21,0), "notes":"Worst episode this week"},
        {"user_id":"user_anika","symptom":"Nausea",      "severity":"Mild",    "logged_at":dt(2,21,45),"notes":"With heartburn"},
        {"user_id":"user_anika","symptom":"Heartburn",   "severity":"Moderate","logged_at":dt(1,21,30),"notes":"Fried dinner"},
        {"user_id":"user_anika","symptom":"Acid Reflux", "severity":"Mild",    "logged_at":dt(1,23,15),"notes":"Cream cheese snack"},
    ]

    n_sym = db.insert_symptom_logs_bulk(symptom_logs_anika)
    log.info(f"  Inserted {n_sym} symptom logs for Anika")

    # ── Build Bayesian memory from this 7-day history ─────────────────────────
    food_entries = [
        FoodLogEntry(
            user_id    = f["user_id"],
            usda_id    = f["usda_id"],
            logged_at  = f["logged_at"],
            quantity_g = f["quantity_g"],
        )
        for f in food_logs_anika
    ]
    sym_entries = [
        SymptomLogEntry(
            user_id   = s["user_id"],
            symptom   = s["symptom"],
            logged_at = s["logged_at"],
            intensity = s["severity"],
        )
        for s in symptom_logs_anika
    ]

    memory = auto_update_from_logs("user_anika", food_entries, sym_entries)
    memory_store.save(memory)
    log.info(
        f"  Bayesian memory saved — "
        f"personalisation_weight={memory.personalisation_weight}, "
        f"learned pairs={len([p for p in memory.priors.values() if p.observations > 0])}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  USER 2 — Rafiul Islam
#  35M · IBS pattern · spicy food + large carb meals → bloating + cramps
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def seed_rafiul() -> None:
    log.info("━━━ Seeding User 2: Rafiul Islam ━━━")

    db.upsert_user({
        "user_id":        "user_rafiul",
        "name":           "Rafiul Islam",
        "age":            35,
        "gender":         "Male",
        "sleep_hours":    7.0,
        "weight_kg":      82.0,
        "foods":          ["Spicy", "Processed Food", "Gluten"],
        "symptoms":       ["Bloating", "Cramps", "Gas"],
        "onboarding_score": 44,
        "current_score":  44,
        "grade":          "Fair",
    })

    # USDA IDs:
    #  20109 = White rice, cooked         (high carbs, moderate)
    #  16057 = Hummus                     (moderate, beans = fermentable)
    #  11156 = Peppers, hot chili, raw    (spicy trigger)
    #  15076 = Salmon, grilled            (safe, omega-3)
    #   9040 = Banana, raw                (safe)
    #   8121 = Oatmeal                    (safe, high fibre)
    #  18350 = White bread                (processed, high carbs)
    #  11209 = Eggplant, cooked           (moderate)
    #   1009 = Cheddar cheese             (dairy, moderate fat)
    #  11282 = Onions, raw                (fermentable — bloating trigger)

    food_logs_rafiul = [
        # Day 7
        {"user_id":"user_rafiul","usda_id":8121, "usda_description":"Oatmeal, cooked",        "quantity_g":300,"unit":"g","meal_type":"Breakfast","logged_at":dt(7, 7,30)},
        {"user_id":"user_rafiul","usda_id":20109,"usda_description":"White rice, cooked",      "quantity_g":350,"unit":"g","meal_type":"Lunch",    "logged_at":dt(7,12,30)},
        {"user_id":"user_rafiul","usda_id":11156,"usda_description":"Chili peppers, hot, raw", "quantity_g":30, "unit":"g","meal_type":"Lunch",    "logged_at":dt(7,12,45)},
        {"user_id":"user_rafiul","usda_id":11282,"usda_description":"Onions, raw",             "quantity_g":80, "unit":"g","meal_type":"Lunch",    "logged_at":dt(7,13)},
        {"user_id":"user_rafiul","usda_id":15076,"usda_description":"Salmon, cooked",          "quantity_g":180,"unit":"g","meal_type":"Dinner",   "logged_at":dt(7,19)},

        # Day 6
        {"user_id":"user_rafiul","usda_id":18350,"usda_description":"White bread",             "quantity_g":120,"unit":"g","meal_type":"Breakfast","logged_at":dt(6, 8)},
        {"user_id":"user_rafiul","usda_id":20109,"usda_description":"White rice, cooked",      "quantity_g":380,"unit":"g","meal_type":"Lunch",    "logged_at":dt(6,13)},
        {"user_id":"user_rafiul","usda_id":11156,"usda_description":"Chili peppers, hot, raw", "quantity_g":40, "unit":"g","meal_type":"Lunch",    "logged_at":dt(6,13,15)},
        {"user_id":"user_rafiul","usda_id":11282,"usda_description":"Onions, raw",             "quantity_g":100,"unit":"g","meal_type":"Lunch",    "logged_at":dt(6,13,30)},
        {"user_id":"user_rafiul","usda_id":1009, "usda_description":"Cheddar cheese",          "quantity_g":60, "unit":"g","meal_type":"Snack",    "logged_at":dt(6,16)},
        {"user_id":"user_rafiul","usda_id":15076,"usda_description":"Salmon, cooked",          "quantity_g":200,"unit":"g","meal_type":"Dinner",   "logged_at":dt(6,19,30)},

        # Day 5
        {"user_id":"user_rafiul","usda_id":8121, "usda_description":"Oatmeal, cooked",        "quantity_g":280,"unit":"g","meal_type":"Breakfast","logged_at":dt(5, 7,45)},
        {"user_id":"user_rafiul","usda_id":9040, "usda_description":"Banana, raw",            "quantity_g":130,"unit":"g","meal_type":"Snack",    "logged_at":dt(5,11)},
        {"user_id":"user_rafiul","usda_id":20109,"usda_description":"White rice, cooked",      "quantity_g":400,"unit":"g","meal_type":"Lunch",    "logged_at":dt(5,12,30)},
        {"user_id":"user_rafiul","usda_id":11156,"usda_description":"Chili peppers, hot, raw", "quantity_g":35, "unit":"g","meal_type":"Lunch",    "logged_at":dt(5,12,45)},
        {"user_id":"user_rafiul","usda_id":11209,"usda_description":"Eggplant, cooked",        "quantity_g":150,"unit":"g","meal_type":"Dinner",   "logged_at":dt(5,19)},

        # Day 4
        {"user_id":"user_rafiul","usda_id":18350,"usda_description":"White bread",             "quantity_g":140,"unit":"g","meal_type":"Breakfast","logged_at":dt(4, 8,30)},
        {"user_id":"user_rafiul","usda_id":16057,"usda_description":"Hummus",                  "quantity_g":120,"unit":"g","meal_type":"Lunch",    "logged_at":dt(4,13)},
        {"user_id":"user_rafiul","usda_id":11282,"usda_description":"Onions, raw",             "quantity_g":90, "unit":"g","meal_type":"Lunch",    "logged_at":dt(4,13,15)},
        {"user_id":"user_rafiul","usda_id":20109,"usda_description":"White rice, cooked",      "quantity_g":350,"unit":"g","meal_type":"Dinner",   "logged_at":dt(4,19)},
        {"user_id":"user_rafiul","usda_id":11156,"usda_description":"Chili peppers, hot, raw", "quantity_g":45, "unit":"g","meal_type":"Dinner",   "logged_at":dt(4,19,20)},

        # Day 3
        {"user_id":"user_rafiul","usda_id":8121, "usda_description":"Oatmeal, cooked",        "quantity_g":300,"unit":"g","meal_type":"Breakfast","logged_at":dt(3, 7,30)},
        {"user_id":"user_rafiul","usda_id":9040, "usda_description":"Banana, raw",            "quantity_g":120,"unit":"g","meal_type":"Snack",    "logged_at":dt(3,10,30)},
        {"user_id":"user_rafiul","usda_id":20109,"usda_description":"White rice, cooked",      "quantity_g":400,"unit":"g","meal_type":"Lunch",    "logged_at":dt(3,12,45)},
        {"user_id":"user_rafiul","usda_id":11282,"usda_description":"Onions, raw",             "quantity_g":100,"unit":"g","meal_type":"Lunch",    "logged_at":dt(3,13)},
        {"user_id":"user_rafiul","usda_id":11156,"usda_description":"Chili peppers, hot, raw", "quantity_g":50, "unit":"g","meal_type":"Lunch",    "logged_at":dt(3,13,10)},
        {"user_id":"user_rafiul","usda_id":15076,"usda_description":"Salmon, cooked",          "quantity_g":180,"unit":"g","meal_type":"Dinner",   "logged_at":dt(3,19,30)},

        # Day 2
        {"user_id":"user_rafiul","usda_id":18350,"usda_description":"White bread",             "quantity_g":130,"unit":"g","meal_type":"Breakfast","logged_at":dt(2, 8)},
        {"user_id":"user_rafiul","usda_id":16057,"usda_description":"Hummus",                  "quantity_g":100,"unit":"g","meal_type":"Lunch",    "logged_at":dt(2,13)},
        {"user_id":"user_rafiul","usda_id":11156,"usda_description":"Chili peppers, hot, raw", "quantity_g":40, "unit":"g","meal_type":"Lunch",    "logged_at":dt(2,13,20)},
        {"user_id":"user_rafiul","usda_id":20109,"usda_description":"White rice, cooked",      "quantity_g":380,"unit":"g","meal_type":"Dinner",   "logged_at":dt(2,19)},
        {"user_id":"user_rafiul","usda_id":1009, "usda_description":"Cheddar cheese",          "quantity_g":50, "unit":"g","meal_type":"Snack",    "logged_at":dt(2,21)},

        # Day 1
        {"user_id":"user_rafiul","usda_id":8121, "usda_description":"Oatmeal, cooked",        "quantity_g":280,"unit":"g","meal_type":"Breakfast","logged_at":dt(1, 7,45)},
        {"user_id":"user_rafiul","usda_id":20109,"usda_description":"White rice, cooked",      "quantity_g":360,"unit":"g","meal_type":"Lunch",    "logged_at":dt(1,12,30)},
        {"user_id":"user_rafiul","usda_id":11282,"usda_description":"Onions, raw",             "quantity_g":80, "unit":"g","meal_type":"Lunch",    "logged_at":dt(1,12,50)},
        {"user_id":"user_rafiul","usda_id":11156,"usda_description":"Chili peppers, hot, raw", "quantity_g":35, "unit":"g","meal_type":"Lunch",    "logged_at":dt(1,13)},
        {"user_id":"user_rafiul","usda_id":11209,"usda_description":"Eggplant, cooked",        "quantity_g":160,"unit":"g","meal_type":"Dinner",   "logged_at":dt(1,19,30)},
    ]

    n_food = db.insert_food_logs_bulk(food_logs_rafiul)
    log.info(f"  Inserted {n_food} food logs for Rafiul")

    # Pattern: Spicy + onions at lunch → bloating + cramps 2-4h later (3-5pm)
    symptom_logs_rafiul = [
        {"user_id":"user_rafiul","symptom":"Bloating","severity":"Moderate","logged_at":dt(7,16,0),"notes":"After spicy rice lunch"},
        {"user_id":"user_rafiul","symptom":"Gas",     "severity":"Mild",    "logged_at":dt(7,17,30),"notes":"Continued gas"},
        {"user_id":"user_rafiul","symptom":"Bloating","severity":"Severe",  "logged_at":dt(6,16,30),"notes":"Spicy chili + onions lunch"},
        {"user_id":"user_rafiul","symptom":"Cramps",  "severity":"Moderate","logged_at":dt(6,17,0), "notes":"After bloating started"},
        {"user_id":"user_rafiul","symptom":"Gas",     "severity":"Moderate","logged_at":dt(6,18,0), "notes":"Prolonged episode"},
        {"user_id":"user_rafiul","symptom":"Bloating","severity":"Moderate","logged_at":dt(5,15,30),"notes":"Spicy lunch again"},
        {"user_id":"user_rafiul","symptom":"Cramps",  "severity":"Mild",    "logged_at":dt(5,16,15),"notes":"Mild cramps"},
        {"user_id":"user_rafiul","symptom":"Bloating","severity":"Severe",  "logged_at":dt(4,16,0), "notes":"Worst episode — lots of chili"},
        {"user_id":"user_rafiul","symptom":"Cramps",  "severity":"Severe",  "logged_at":dt(4,16,45),"notes":"Severe cramping"},
        {"user_id":"user_rafiul","symptom":"Gas",     "severity":"Severe",  "logged_at":dt(4,17,30),"notes":"Very uncomfortable"},
        {"user_id":"user_rafiul","symptom":"Bloating","severity":"Moderate","logged_at":dt(3,16,30),"notes":"Spicy rice + onions"},
        {"user_id":"user_rafiul","symptom":"Cramps",  "severity":"Moderate","logged_at":dt(3,17,15),"notes":"Post-lunch cramps"},
        {"user_id":"user_rafiul","symptom":"Bloating","severity":"Moderate","logged_at":dt(2,15,45),"notes":"Spicy hummus lunch"},
        {"user_id":"user_rafiul","symptom":"Gas",     "severity":"Mild",    "logged_at":dt(2,17,0), "notes":"Ongoing gas"},
        {"user_id":"user_rafiul","symptom":"Bloating","severity":"Severe",  "logged_at":dt(1,15,30),"notes":"Daily spicy pattern continues"},
        {"user_id":"user_rafiul","symptom":"Cramps",  "severity":"Moderate","logged_at":dt(1,16,30),"notes":"Cramping with bloating"},
    ]

    n_sym = db.insert_symptom_logs_bulk(symptom_logs_rafiul)
    log.info(f"  Inserted {n_sym} symptom logs for Rafiul")

    food_entries = [FoodLogEntry(user_id=f["user_id"], usda_id=f["usda_id"], logged_at=f["logged_at"], quantity_g=f["quantity_g"]) for f in food_logs_rafiul]
    sym_entries  = [SymptomLogEntry(user_id=s["user_id"], symptom=s["symptom"], logged_at=s["logged_at"], intensity=s["severity"]) for s in symptom_logs_rafiul]
    memory = auto_update_from_logs("user_rafiul", food_entries, sym_entries)
    memory_store.save(memory)
    log.info(f"  Memory saved — weight={memory.personalisation_weight}, pairs={len([p for p in memory.priors.values() if p.observations > 0])}")


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  USER 3 — Sadia Hossain
#  45F · constipation + fatigue · low-fibre diet, sugary snacks
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def seed_sadia() -> None:
    log.info("━━━ Seeding User 3: Sadia Hossain ━━━")

    db.upsert_user({
        "user_id":        "user_sadia",
        "name":           "Sadia Hossain",
        "age":            45,
        "gender":         "Female",
        "sleep_hours":    6.0,
        "weight_kg":      68.0,
        "foods":          ["Sugar", "Processed Food"],
        "symptoms":       ["Constipation", "Fatigue"],
        "onboarding_score": 38,
        "current_score":  38,
        "grade":          "Poor",
    })

    # USDA IDs:
    #  18350 = White bread               (low fibre, high carbs)
    #   9040 = Banana, raw               (moderate sugar, some fibre)
    #  19335 = Sugar, white granulated   (pure sugar)
    #  18133 = Cake, chocolate           (high sugar, high fat)
    #  14209 = Coffee, brewed            (caffeine, dehydrating)
    #  20081 = Whole wheat bread         (high fibre — used occasionally)
    #  11124 = Carrots, raw              (fibre, vitamins)
    #   9316 = Raisins                   (high sugar, some fibre)
    #   1089 = Butter                    (high sat-fat)
    #  15076 = Salmon, cooked            (protein, omega-3)

    food_logs_sadia = [
        # Day 7
        {"user_id":"user_sadia","usda_id":18350,"usda_description":"White bread",              "quantity_g":120,"unit":"g","meal_type":"Breakfast","logged_at":dt(7, 9)},
        {"user_id":"user_sadia","usda_id":1089, "usda_description":"Butter",                   "quantity_g":20, "unit":"g","meal_type":"Breakfast","logged_at":dt(7, 9,10)},
        {"user_id":"user_sadia","usda_id":14209,"usda_description":"Coffee, brewed",            "quantity_g":240,"unit":"ml","meal_type":"Breakfast","logged_at":dt(7, 9,15)},
        {"user_id":"user_sadia","usda_id":18350,"usda_description":"White bread",              "quantity_g":100,"unit":"g","meal_type":"Lunch",    "logged_at":dt(7,13,30)},
        {"user_id":"user_sadia","usda_id":18133,"usda_description":"Chocolate cake",           "quantity_g":90, "unit":"g","meal_type":"Snack",    "logged_at":dt(7,16)},
        {"user_id":"user_sadia","usda_id":9316, "usda_description":"Raisins",                  "quantity_g":60, "unit":"g","meal_type":"Snack",    "logged_at":dt(7,17)},
        {"user_id":"user_sadia","usda_id":18350,"usda_description":"White bread",              "quantity_g":110,"unit":"g","meal_type":"Dinner",   "logged_at":dt(7,20)},
        {"user_id":"user_sadia","usda_id":1089, "usda_description":"Butter",                   "quantity_g":15, "unit":"g","meal_type":"Dinner",   "logged_at":dt(7,20,5)},

        # Day 6
        {"user_id":"user_sadia","usda_id":14209,"usda_description":"Coffee, brewed",            "quantity_g":240,"unit":"ml","meal_type":"Breakfast","logged_at":dt(6, 8,30)},
        {"user_id":"user_sadia","usda_id":18350,"usda_description":"White bread",              "quantity_g":130,"unit":"g","meal_type":"Breakfast","logged_at":dt(6, 8,45)},
        {"user_id":"user_sadia","usda_id":9040, "usda_description":"Banana, raw",             "quantity_g":120,"unit":"g","meal_type":"Snack",    "logged_at":dt(6,11)},
        {"user_id":"user_sadia","usda_id":11124,"usda_description":"Carrots, raw",            "quantity_g":80, "unit":"g","meal_type":"Lunch",    "logged_at":dt(6,13)},
        {"user_id":"user_sadia","usda_id":18350,"usda_description":"White bread",              "quantity_g":100,"unit":"g","meal_type":"Lunch",    "logged_at":dt(6,13,15)},
        {"user_id":"user_sadia","usda_id":18133,"usda_description":"Chocolate cake",           "quantity_g":100,"unit":"g","meal_type":"Snack",    "logged_at":dt(6,16,30)},
        {"user_id":"user_sadia","usda_id":15076,"usda_description":"Salmon, cooked",           "quantity_g":160,"unit":"g","meal_type":"Dinner",   "logged_at":dt(6,19,30)},
        {"user_id":"user_sadia","usda_id":18350,"usda_description":"White bread",              "quantity_g":90, "unit":"g","meal_type":"Dinner",   "logged_at":dt(6,19,45)},

        # Day 5
        {"user_id":"user_sadia","usda_id":18350,"usda_description":"White bread",              "quantity_g":120,"unit":"g","meal_type":"Breakfast","logged_at":dt(5, 9,15)},
        {"user_id":"user_sadia","usda_id":1089, "usda_description":"Butter",                   "quantity_g":20, "unit":"g","meal_type":"Breakfast","logged_at":dt(5, 9,20)},
        {"user_id":"user_sadia","usda_id":14209,"usda_description":"Coffee, brewed",            "quantity_g":240,"unit":"ml","meal_type":"Breakfast","logged_at":dt(5, 9,30)},
        {"user_id":"user_sadia","usda_id":9316, "usda_description":"Raisins",                  "quantity_g":70, "unit":"g","meal_type":"Snack",    "logged_at":dt(5,11,30)},
        {"user_id":"user_sadia","usda_id":18350,"usda_description":"White bread",              "quantity_g":110,"unit":"g","meal_type":"Lunch",    "logged_at":dt(5,13,30)},
        {"user_id":"user_sadia","usda_id":18133,"usda_description":"Chocolate cake",           "quantity_g":80, "unit":"g","meal_type":"Snack",    "logged_at":dt(5,16)},
        {"user_id":"user_sadia","usda_id":18350,"usda_description":"White bread",              "quantity_g":120,"unit":"g","meal_type":"Dinner",   "logged_at":dt(5,20,30)},
        {"user_id":"user_sadia","usda_id":1089, "usda_description":"Butter",                   "quantity_g":15, "unit":"g","meal_type":"Dinner",   "logged_at":dt(5,20,35)},

        # Day 4
        {"user_id":"user_sadia","usda_id":20081,"usda_description":"Whole wheat bread",         "quantity_g":100,"unit":"g","meal_type":"Breakfast","logged_at":dt(4, 8,45)},  # rare healthy day
        {"user_id":"user_sadia","usda_id":14209,"usda_description":"Coffee, brewed",            "quantity_g":240,"unit":"ml","meal_type":"Breakfast","logged_at":dt(4, 9)},
        {"user_id":"user_sadia","usda_id":11124,"usda_description":"Carrots, raw",            "quantity_g":100,"unit":"g","meal_type":"Lunch",    "logged_at":dt(4,13)},
        {"user_id":"user_sadia","usda_id":15076,"usda_description":"Salmon, cooked",           "quantity_g":180,"unit":"g","meal_type":"Lunch",    "logged_at":dt(4,13,20)},
        {"user_id":"user_sadia","usda_id":18133,"usda_description":"Chocolate cake",           "quantity_g":110,"unit":"g","meal_type":"Snack",    "logged_at":dt(4,16,30)},
        {"user_id":"user_sadia","usda_id":18350,"usda_description":"White bread",              "quantity_g":130,"unit":"g","meal_type":"Dinner",   "logged_at":dt(4,20)},
        {"user_id":"user_sadia","usda_id":1089, "usda_description":"Butter",                   "quantity_g":25, "unit":"g","meal_type":"Dinner",   "logged_at":dt(4,20,5)},

        # Day 3
        {"user_id":"user_sadia","usda_id":18350,"usda_description":"White bread",              "quantity_g":120,"unit":"g","meal_type":"Breakfast","logged_at":dt(3, 9)},
        {"user_id":"user_sadia","usda_id":1089, "usda_description":"Butter",                   "quantity_g":20, "unit":"g","meal_type":"Breakfast","logged_at":dt(3, 9,5)},
        {"user_id":"user_sadia","usda_id":9040, "usda_description":"Banana, raw",             "quantity_g":130,"unit":"g","meal_type":"Snack",    "logged_at":dt(3,11)},
        {"user_id":"user_sadia","usda_id":9316, "usda_description":"Raisins",                  "quantity_g":65, "unit":"g","meal_type":"Snack",    "logged_at":dt(3,16)},
        {"user_id":"user_sadia","usda_id":18350,"usda_description":"White bread",              "quantity_g":100,"unit":"g","meal_type":"Dinner",   "logged_at":dt(3,20)},
        {"user_id":"user_sadia","usda_id":18133,"usda_description":"Chocolate cake",           "quantity_g":95, "unit":"g","meal_type":"Snack",    "logged_at":dt(3,21,30)},

        # Day 2
        {"user_id":"user_sadia","usda_id":14209,"usda_description":"Coffee, brewed",            "quantity_g":240,"unit":"ml","meal_type":"Breakfast","logged_at":dt(2, 8,30)},
        {"user_id":"user_sadia","usda_id":18350,"usda_description":"White bread",              "quantity_g":140,"unit":"g","meal_type":"Breakfast","logged_at":dt(2, 8,45)},
        {"user_id":"user_sadia","usda_id":1089, "usda_description":"Butter",                   "quantity_g":25, "unit":"g","meal_type":"Breakfast","logged_at":dt(2, 8,50)},
        {"user_id":"user_sadia","usda_id":18133,"usda_description":"Chocolate cake",           "quantity_g":100,"unit":"g","meal_type":"Snack",    "logged_at":dt(2,16)},
        {"user_id":"user_sadia","usda_id":18350,"usda_description":"White bread",              "quantity_g":120,"unit":"g","meal_type":"Dinner",   "logged_at":dt(2,19,30)},
        {"user_id":"user_sadia","usda_id":9316, "usda_description":"Raisins",                  "quantity_g":60, "unit":"g","meal_type":"Dinner",   "logged_at":dt(2,19,45)},

        # Day 1
        {"user_id":"user_sadia","usda_id":18350,"usda_description":"White bread",              "quantity_g":130,"unit":"g","meal_type":"Breakfast","logged_at":dt(1, 9,30)},
        {"user_id":"user_sadia","usda_id":1089, "usda_description":"Butter",                   "quantity_g":20, "unit":"g","meal_type":"Breakfast","logged_at":dt(1, 9,35)},
        {"user_id":"user_sadia","usda_id":14209,"usda_description":"Coffee, brewed",            "quantity_g":240,"unit":"ml","meal_type":"Breakfast","logged_at":dt(1, 9,45)},
        {"user_id":"user_sadia","usda_id":9040, "usda_description":"Banana, raw",             "quantity_g":120,"unit":"g","meal_type":"Snack",    "logged_at":dt(1,11)},
        {"user_id":"user_sadia","usda_id":18350,"usda_description":"White bread",              "quantity_g":110,"unit":"g","meal_type":"Lunch",    "logged_at":dt(1,13)},
        {"user_id":"user_sadia","usda_id":18133,"usda_description":"Chocolate cake",           "quantity_g":90, "unit":"g","meal_type":"Snack",    "logged_at":dt(1,16,30)},
        {"user_id":"user_sadia","usda_id":18350,"usda_description":"White bread",              "quantity_g":120,"unit":"g","meal_type":"Dinner",   "logged_at":dt(1,20,30)},
        {"user_id":"user_sadia","usda_id":1089, "usda_description":"Butter",                   "quantity_g":20, "unit":"g","meal_type":"Dinner",   "logged_at":dt(1,20,35)},
    ]

    n_food = db.insert_food_logs_bulk(food_logs_sadia)
    log.info(f"  Inserted {n_food} food logs for Sadia")

    # Pattern: Low-fibre dinner → constipation next morning (12-18h later)
    #          Sugar snacks + coffee → fatigue 1-3h later
    symptom_logs_sadia = [
        {"user_id":"user_sadia","symptom":"Constipation","severity":"Moderate","logged_at":dt(6, 8, 0),"notes":"No movement since yesterday"},
        {"user_id":"user_sadia","symptom":"Fatigue",     "severity":"Mild",    "logged_at":dt(7,18, 0),"notes":"Tired after sugar snacks"},
        {"user_id":"user_sadia","symptom":"Constipation","severity":"Severe",  "logged_at":dt(5, 7,30),"notes":"Very difficult morning"},
        {"user_id":"user_sadia","symptom":"Fatigue",     "severity":"Moderate","logged_at":dt(5,19, 0),"notes":"Post-dinner fatigue, heavy and sluggish"},
        {"user_id":"user_sadia","symptom":"Constipation","severity":"Moderate","logged_at":dt(4, 8, 0),"notes":"Missed yesterday completely"},
        {"user_id":"user_sadia","symptom":"Fatigue",     "severity":"Mild",    "logged_at":dt(4,18,30),"notes":"After cake snack"},
        {"user_id":"user_sadia","symptom":"Constipation","severity":"Mild",    "logged_at":dt(3, 7, 0),"notes":"Better day — had carrots"},
        {"user_id":"user_sadia","symptom":"Fatigue",     "severity":"Moderate","logged_at":dt(3,19, 0),"notes":"Evening fatigue"},
        {"user_id":"user_sadia","symptom":"Constipation","severity":"Severe",  "logged_at":dt(2, 7,30),"notes":"Three days of white bread catching up"},
        {"user_id":"user_sadia","symptom":"Fatigue",     "severity":"Severe",  "logged_at":dt(2,17, 0),"notes":"Exhausted after sugar + coffee day"},
        {"user_id":"user_sadia","symptom":"Constipation","severity":"Moderate","logged_at":dt(1, 8, 0),"notes":"Ongoing constipation"},
        {"user_id":"user_sadia","symptom":"Fatigue",     "severity":"Moderate","logged_at":dt(1,18,30),"notes":"Post-cake fatigue"},
    ]

    n_sym = db.insert_symptom_logs_bulk(symptom_logs_sadia)
    log.info(f"  Inserted {n_sym} symptom logs for Sadia")

    food_entries = [FoodLogEntry(user_id=f["user_id"], usda_id=f["usda_id"], logged_at=f["logged_at"], quantity_g=f["quantity_g"]) for f in food_logs_sadia]
    sym_entries  = [SymptomLogEntry(user_id=s["user_id"], symptom=s["symptom"], logged_at=s["logged_at"], intensity=s["severity"]) for s in symptom_logs_sadia]
    memory = auto_update_from_logs("user_sadia", food_entries, sym_entries)
    memory_store.save(memory)
    log.info(f"  Memory saved — weight={memory.personalisation_weight}, pairs={len([p for p in memory.priors.values() if p.observations > 0])}")


# ─────────────────────────────────────────────────────────────────────────────
# Verification — prove data is in MongoDB
# ─────────────────────────────────────────────────────────────────────────────

def verify_and_print() -> None:
    print("\n" + "═"*70)
    print("  VERIFICATION — DATA IN MONGODB")
    print("═"*70)

    users = db.list_users()
    print(f"\n  ✅ Users seeded: {len(users)}")
    for u in users:
        food_count = db.count_food_logs(u["user_id"])
        sym_count  = db.count_symptom_logs(u["user_id"])
        mem        = db.load_memory(u["user_id"])
        learned    = len([p for p in mem.priors.values() if p.observations > 0])
        print(f"\n  ┌─ {u['name']} ({u['user_id']})")
        print(f"  │  Age: {u['age']}  Gender: {u['gender']}  Score: {u['onboarding_score']} ({u['grade']})")
        print(f"  │  Food logs : {food_count}")
        print(f"  │  Sym logs  : {sym_count}")
        print(f"  │  Personalisation weight : {mem.personalisation_weight}")
        print(f"  │  Learned food→symptom pairs : {learned}")
        if learned > 0:
            top = sorted(mem.priors.items(), key=lambda x: x[1].posterior_mean, reverse=True)[:3]
            for k, v in top:
                print(f"  │    {k:40s}  p={v.posterior_mean:.3f}  obs={v.observations}")

    print("\n" + "─"*70)
    print("  DASHBOARD SUMMARY — LAST 7 DAYS")
    print("─"*70)

    for user_id in ["user_anika", "user_rafiul", "user_sadia"]:
        dash = db.user_dashboard(user_id)
        print(f"\n  [{dash['name']}]")
        print(f"  Current score : {dash['current_score']}  Grade: {dash['grade']}")
        print(f"  Food logs     : {dash['food_logs_this_week']}  |  Symptom logs: {dash['symptom_logs_this_week']}")
        print(f"  Personalisation weight : {dash['personalisation_weight']}")
        print(f"  Symptom frequency this week:")
        for sym, count in sorted(dash["symptom_frequency"].items(), key=lambda x: -x[1]):
            print(f"    {sym:20s}: {count}x")
        print(f"  Top sensitivities:")
        for s in dash["top_sensitivities"]:
            print(f"    {s['pair']:40s}  p={s['probability']:.3f}  obs={s['observations']}")

    print("\n" + "═"*70)
    print("  ✅  Seed complete. MongoDB is ready.")
    print("  Run:  uvicorn main:app --reload")
    print("═"*70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═"*70)
    print("  GUT HEALTH API — MongoDB Demo Seed")
    print("═"*70)

    drop_all()
    seed_anika()
    seed_rafiul()
    seed_sadia()
    verify_and_print()

    db.disconnect()