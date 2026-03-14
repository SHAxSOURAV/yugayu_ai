"""
tests/test_api.py
─────────────────
Smoke-test suite for the Gut Health API.

All HuggingFace models, MongoDB, and the USDA pipeline are mocked in
conftest.py — which runs before pytest collects this file.

Run from the project root:
    pytest tests/ -v

With coverage:
    pytest tests/ -v --cov=. --cov-report=term-missing
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _dt(days_ago: float, hour: int, minute: int = 0) -> str:
    t = datetime.now(timezone.utc).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    return (t - timedelta(days=days_ago)).isoformat()


@pytest.fixture(scope="session")
def client() -> Generator:
    from main import app, _state
    from user_symptom_memory import UserMemoryStore
    from tests.conftest import FAKE_USDA_DF

    _state.usda_df    = FAKE_USDA_DF
    _state.id_col     = "ID"
    _state.desc_col   = "Description"
    _state.usda_ready = True

    def _mock_text_to_usda(text: str, top_k: int = 3):
        return [{
            "raw_food": "oatmeal", "ner_confidence": 0.98,
            "normalised_name": "Oatmeal, cooked", "usda_id": 8121,
            "usda_description": "OATMEAL,INST,FORT,PLAIN,PREP W/WATER",
            "usda_similarity": 0.92,
            "usda_top_matches": [{"rank": 1, "usda_id": 8121,
                "usda_description": "OATMEAL,INST,FORT,PLAIN,PREP W/WATER",
                "similarity": 0.92}],
        }]
    _state.text_to_usda = _mock_text_to_usda

    # analyse_food_log_batch returns a plain tuple: (total_modifier: int, list[FoodLogScore])
    # analyse_symptom_log returns a SymptomLogScore dataclass
    # — use the REAL class names from nutrition_scorer.py
    from nutrition_scorer import FoodLogScore, SymptomLogScore, NutrientProfile

    def _mock_analyse_food(foods, meal_type, usda_df, id_col, desc_col):
        # Real return type is tuple(int, list[FoodLogScore])
        fake_food = FoodLogScore(
            usda_id           = 8121,
            food_description  = "OATMEAL,INST,FORT,PLAIN,PREP W/WATER",
            portion_grams     = 250.0,
            meal_type         = meal_type,
            score_modifier    = 3,
            nutrient_modifier = 2,
            hf_modifier       = 1,
            meal_type_note    = f"{meal_type} multiplier applied",
            digestibility     = "easy to digest and gut-friendly",
            hf_confidence     = 0.82,
            hf_note           = "Low fat, high fibre food",
            nutrient_profile  = NutrientProfile(
                calories=71.0, protein=2.5, total_fat=1.5, carbs=12.0,
                sodium=218.0, sat_fat=0.3, cholesterol=0.0, sugar=0.3,
                calcium=100.0, iron=4.2, potassium=95.0,
                vitamin_c=0.0, vitamin_e=0.1, vitamin_d=0.0,
                portion_grams=250.0,
            ),
            nutrient_impacts  = [],
            nutrient_notes    = ["Good source of iron", "Low in saturated fat"],
        )
        return 3, [fake_food]   # (total_modifier, list[FoodLogScore])

    def _mock_analyse_symptom(symptom, severity, logged_at):
        return SymptomLogScore(
            symptom       = symptom,
            severity      = severity,
            logged_at     = logged_at.isoformat() if hasattr(logged_at, "isoformat") else str(logged_at),
            hour          = logged_at.hour if hasattr(logged_at, "hour") else 12,
            score_penalty = -7,
            severity_note = "Moderate severity",
            time_note     = "Daytime symptom",
            clinical_note = "Bloating is commonly triggered by fermentable carbohydrates.",
        )

    _state.analyse_food_log_batch = _mock_analyse_food
    _state.analyse_symptom_log    = _mock_analyse_symptom
    _state.nutrition_ready        = True
    _state.predict_food_symptom_causes = lambda *a, **k: []
    _state.predictor_ready             = True
    _state.memory_store          = UserMemoryStore()
    _state.auto_update_from_logs = lambda *a, **k: None

    from food_recommender import SafeFoodRecommendation

    def _mock_recommend(eaten_usda_ids, user_symptoms, user_memory,
                        usda_df, id_col, desc_col, n=5):
        return [
            SafeFoodRecommendation(
                rank=i+1, food_name=f"Recommended Food {i+1}",
                recommendation=f"A gut-friendly option packed with nutrients. Choice {i+1}.",
                safety_score=round(0.90-i*0.05, 2), similarity_score=0.75,
                combined_score=round(0.85-i*0.04, 2),
                key_nutrients={"Calories (kcal/100g)": 100.0, "Protein (g/100g)": 8.0},
                safe_reasons=["Low sodium", "Low saturated fat"], avoid_reasons=[],
            )
            for i in range(min(n, 5))
        ]

    _state.recommend_safe_foods = _mock_recommend
    _state.recommender_ready    = True

    from tests.conftest import _mock_db_instance
    _state._mongo_db = _mock_db_instance

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestFoodParse:
    def test_happy_path(self, client):
        r = client.post("/food/parse", json={"text": "I had oatmeal for breakfast"})
        assert r.status_code == 200
        assert "foods_detected" in r.json()

    def test_text_too_short(self, client):
        assert client.post("/food/parse", json={"text": "ab"}).status_code == 422

    def test_missing_text_field(self, client):
        assert client.post("/food/parse", json={"top_k": 2}).status_code == 422

    def test_top_k_out_of_range(self, client):
        assert client.post("/food/parse", json={"text": "chicken and rice", "top_k": 99}).status_code == 422

    def test_response_schema(self, client):
        r = client.post("/food/parse", json={"text": "oatmeal with banana"})
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["foods_detected"], int)
        if body["results"]:
            for key in ["usda_id", "ner_confidence", "usda_top_matches"]:
                assert key in body["results"][0]


class TestFoodTextToId:
    def test_happy_path(self, client):
        r = client.post("/food/text-to-id", json={"text": "grilled salmon with broccoli"})
        assert r.status_code == 200
        assert "foods" in r.json()

    def test_short_text_rejected(self, client):
        assert client.post("/food/text-to-id", json={"text": "ok"}).status_code == 422

    def test_returns_usda_id(self, client):
        r = client.post("/food/text-to-id", json={"text": "I ate oatmeal"})
        assert r.status_code == 200
        if r.json()["foods"]:
            assert isinstance(r.json()["foods"][0]["usda_id"], int)


class TestFoodLookup:
    def test_known_id(self, client):
        r = client.get("/food/lookup/8121")
        assert r.status_code == 200
        assert r.json()["usda_id"] == 8121
        assert r.json()["per"] == "100g"

    def test_unknown_id_returns_404(self, client):
        assert client.get("/food/lookup/9999999").status_code == 404

    def test_core_macros_present(self, client):
        body = client.get("/food/lookup/8121").json()
        for f in ["calories", "protein", "carbs", "fat"]:
            assert f in body

    def test_invalid_id_type(self, client):
        assert client.get("/food/lookup/not-a-number").status_code == 422


class TestOnboardingScore:
    VALID = {"gender": "Female", "age": 28, "sleep_hours": 5.5, "weight_kg": 58.0,
             "foods": ["Fried", "Dairy", "Caffeine"], "symptoms": ["Heartburn", "Bloating"]}

    def test_happy_path(self, client):
        r = client.post("/score", json=self.VALID)
        assert r.status_code == 200
        assert all(k in r.json() for k in ["score", "grade", "tagline"])

    def test_score_in_range(self, client):
        score = client.post("/score", json=self.VALID).json()["score"]
        assert 0 <= score <= 100

    def test_grade_matches_score(self, client):
        body = client.post("/score", json=self.VALID).json()
        score, grade = body["score"], body["grade"]
        if   score >= 85: assert grade == "Excellent"
        elif score >= 70: assert grade == "Good"
        elif score >= 50: assert grade == "Fair"
        elif score >= 30: assert grade == "Poor"
        else:             assert grade == "Critical"

    def test_missing_field_rejected(self, client):
        p = {k: v for k, v in self.VALID.items() if k != "gender"}
        assert client.post("/score", json=p).status_code == 422

    def test_healthy_scores_high(self, client):
        r = client.post("/score", json={"gender": "Male", "age": 25, "sleep_hours": 8.0,
                                         "weight_kg": 70.0, "foods": [], "symptoms": []})
        assert r.json()["score"] >= 70

    def test_worst_case_scores_low(self, client):
        r = client.post("/score", json={
            "gender": "Female", "age": 65, "sleep_hours": 4.0, "weight_kg": 110.0,
            "foods": ["Fried", "Processed Food", "Sugar", "Caffeine", "Gluten", "Dairy"],
            "symptoms": ["Diarrhea", "Abdominal Pain", "Heartburn", "Nausea", "Bloating"],
        })
        assert r.json()["score"] < 50


class TestLogFood:
    VALID = {"current_score": 65, "meal_type": "Breakfast",
             "foods": [{"usda_id": 8121, "quantity": 250, "unit": "g"}]}

    def test_happy_path(self, client):
        r = client.post("/log/food", json=self.VALID)
        assert r.status_code == 200
        assert all(k in r.json() for k in ["updated_score", "previous_score", "score_modifier"])

    def test_previous_score_echoed(self, client):
        assert client.post("/log/food", json=self.VALID).json()["previous_score"] == 65

    def test_score_clamped_max(self, client):
        assert client.post("/log/food", json={**self.VALID, "current_score": 99}).json()["updated_score"] <= 100

    def test_score_clamped_min(self, client):
        assert client.post("/log/food", json={**self.VALID, "current_score": 0}).json()["updated_score"] >= 0

    def test_invalid_meal_type(self, client):
        assert client.post("/log/food", json={**self.VALID, "meal_type": "Brunch"}).status_code == 422

    def test_invalid_unit(self, client):
        assert client.post("/log/food", json={"current_score": 65, "meal_type": "Lunch",
            "foods": [{"usda_id": 8121, "quantity": 250, "unit": "tablespoons"}]}).status_code == 422

    def test_empty_foods_rejected(self, client):
        assert client.post("/log/food", json={"current_score": 65, "meal_type": "Lunch",
                                               "foods": []}).status_code == 422

    def test_all_meal_types(self, client):
        for mt in ["Breakfast", "Lunch", "Dinner", "Snack"]:
            r = client.post("/log/food", json={"current_score": 70, "meal_type": mt,
                "foods": [{"usda_id": 8121, "quantity": 200, "unit": "g"}]})
            assert r.status_code == 200, f"Meal type {mt!r} rejected"

    def test_all_valid_units(self, client):
        for unit in ["g", "gram", "ml", "oz", "lb", "piece", "cup", "tbsp", "tsp", "serving"]:
            r = client.post("/log/food", json={"current_score": 70, "meal_type": "Snack",
                "foods": [{"usda_id": 9040, "quantity": 1, "unit": unit}]})
            assert r.status_code == 200, f"Unit {unit!r} rejected"


class TestLogSymptom:
    VALID = {"current_score": 70, "symptom": "Bloating", "severity": "Moderate"}

    def test_happy_path(self, client):
        r = client.post("/log/symptom", json=self.VALID)
        assert r.status_code == 200
        assert r.json()["score_penalty"] <= 0

    def test_score_decreases(self, client):
        body = client.post("/log/symptom", json=self.VALID).json()
        assert body["updated_score"] <= body["previous_score"]

    def test_score_clamped_min(self, client):
        assert client.post("/log/symptom", json={**self.VALID, "current_score": 2}).json()["updated_score"] >= 0

    def test_invalid_symptom(self, client):
        assert client.post("/log/symptom", json={**self.VALID, "symptom": "Migraine"}).status_code == 422

    def test_invalid_severity(self, client):
        assert client.post("/log/symptom", json={**self.VALID, "severity": "Extreme"}).status_code == 422

    def test_all_symptoms(self, client):
        for sym in ["Bloating","Abdominal Pain","Nausea","Constipation",
                    "Heartburn","Gas","Fatigue","Acid Reflux","Cramps","Diarrhea"]:
            assert client.post("/log/symptom", json={"current_score": 70,
                "symptom": sym, "severity": "Mild"}).status_code == 200, f"{sym!r} rejected"

    def test_all_severities(self, client):
        for sev in ["Mild", "Moderate", "Severe"]:
            assert client.post("/log/symptom", json={"current_score": 70,
                "symptom": "Gas", "severity": sev}).status_code == 200


class TestPredictFoodSymptom:
    VALID = {
        "food_logs": [{"user_id":"u","usda_id":5064,"logged_at":_dt(0,19),"quantity_g":250}],
        "symptom_logs": [{"user_id":"u","symptom":"Heartburn","logged_at":_dt(0,21),"intensity":"Moderate"}],
    }

    def test_happy_path(self, client):
        r = client.post("/predict/food-symptom", json=self.VALID)
        assert r.status_code == 200
        assert "predictions" in r.json()

    def test_bad_symptom_rejected(self, client):
        p = {**self.VALID, "symptom_logs": [{"user_id":"u","symptom":"BadOne",
             "logged_at":_dt(0,21),"intensity":"Mild"}]}
        assert client.post("/predict/food-symptom", json=p).status_code == 422

    def test_bad_intensity_rejected(self, client):
        p = {**self.VALID, "symptom_logs": [{"user_id":"u","symptom":"Heartburn",
             "logged_at":_dt(0,21),"intensity":"Catastrophic"}]}
        assert client.post("/predict/food-symptom", json=p).status_code == 422

    def test_empty_food_logs_rejected(self, client):
        assert client.post("/predict/food-symptom", json={
            "food_logs": [], "symptom_logs": self.VALID["symptom_logs"]}).status_code == 422

    def test_empty_symptom_logs_rejected(self, client):
        assert client.post("/predict/food-symptom", json={
            "food_logs": self.VALID["food_logs"], "symptom_logs": []}).status_code == 422


class TestPredictFeedback:
    def test_confirmed(self, client):
        r = client.post("/predict/feedback", json={"user_id":"u1","usda_id":5064,
                                                    "symptom":"Heartburn","confirmed":True})
        assert r.status_code == 200
        assert 0.0 <= r.json()["updated_prior"] <= 1.0

    def test_denied(self, client):
        r = client.post("/predict/feedback", json={"user_id":"u1","usda_id":5064,
                                                    "symptom":"Heartburn","confirmed":False})
        assert r.status_code == 200
        assert r.json()["confirmed"] is False

    def test_prior_rises_after_repeat_confirmations(self, client):
        uid = "u_repeat_confirm"
        f = client.post("/predict/feedback", json={"user_id":uid,"usda_id":5064,
                        "symptom":"Bloating","confirmed":True}).json()["updated_prior"]
        for _ in range(5):
            client.post("/predict/feedback", json={"user_id":uid,"usda_id":5064,
                        "symptom":"Bloating","confirmed":True})
        final = client.post("/predict/feedback", json={"user_id":uid,"usda_id":5064,
                        "symptom":"Bloating","confirmed":True}).json()["updated_prior"]
        assert final >= f

    def test_missing_fields_rejected(self, client):
        assert client.post("/predict/feedback", json={"user_id":"u","usda_id":5064}).status_code == 422


class TestLearningSummary:
    def test_new_user(self, client):
        r = client.get("/user/brand_new_xyz/learning-summary")
        assert r.status_code == 200
        body = r.json()
        assert body["total_food_logs"] == 0
        assert body["personalisation_weight"] == 0.20
        assert "New user" in body["personalisation_stage"]

    def test_schema(self, client):
        body = client.get("/user/schema_check/learning-summary").json()
        for f in ["user_id","total_food_logs","total_symptom_logs","personalisation_weight",
                  "model_weight","personalisation_stage","learned_pairs",
                  "top_sensitivities","learning_message"]:
            assert f in body, f"Missing: {f}"


class TestRecommendSafeFoods:
    VALID = {"user_id":"u_rec","eaten_usda_ids":[8121,5064],
             "user_symptoms":["Heartburn","Bloating"],"n":5}

    def test_happy_path(self, client):
        r = client.post("/recommend/safe-foods", json=self.VALID)
        assert r.status_code == 200
        assert len(r.json()["recommendations"]) <= 5

    def test_natural_language(self, client):
        for rec in client.post("/recommend/safe-foods", json=self.VALID).json()["recommendations"]:
            assert len(rec["recommendation"]) > 10
            assert not rec["recommendation"].isupper()

    def test_bad_symptom_rejected(self, client):
        assert client.post("/recommend/safe-foods", json={**self.VALID,
            "user_symptoms":["Migraine"]}).status_code == 422

    def test_n_too_large_rejected(self, client):
        assert client.post("/recommend/safe-foods", json={**self.VALID, "n":50}).status_code == 422

    def test_safety_score_range(self, client):
        for rec in client.post("/recommend/safe-foods", json=self.VALID).json()["recommendations"]:
            assert 0.0 <= rec["safety_score"] <= 1.0

    def test_rank_order(self, client):
        ranks = [r["rank"] for r in client.post("/recommend/safe-foods",
                 json=self.VALID).json()["recommendations"]]
        assert ranks == sorted(ranks)

    def test_no_symptoms_ok(self, client):
        assert client.post("/recommend/safe-foods", json={**self.VALID,
            "user_symptoms":[]}).status_code == 200


class TestDashboard:
    def test_happy_path(self, client):
        r = client.get("/user/test_user/dashboard")
        assert r.status_code == 200
        assert r.json()["user_id"] == "test_user"

    def test_unknown_user_404(self, client):
        from tests.conftest import _mock_db_instance
        _mock_db_instance.get_user.return_value = None
        assert client.get("/user/nobody_xyz/dashboard").status_code == 404
        _mock_db_instance.get_user.return_value = {"user_id":"test_user","name":"Test User",
                                                     "current_score":65,"grade":"Good"}

    def test_all_fields(self, client):
        body = client.get("/user/test_user/dashboard").json()
        for f in ["user_id","name","current_score","grade","food_logs_this_week",
                  "symptom_logs_this_week","unique_foods_eaten","symptom_frequency",
                  "personalisation_weight","top_sensitivities"]:
            assert f in body, f"Missing: {f}"


class TestDbHealth:
    def test_returns_200(self, client):
        assert client.get("/db/health").status_code == 200

    def test_has_status(self, client):
        assert "status" in client.get("/db/health").json()


class TestScorerLogic:
    def test_score_max_100(self):
        from scorer import calculate_score, DigestiveInput
        assert calculate_score(DigestiveInput(gender="Male", age=20, sleep_hours=9.0,
                               weight_kg=65.0, foods=[], symptoms=[])).score <= 100

    def test_score_min_0(self):
        from scorer import calculate_score, DigestiveInput
        assert calculate_score(DigestiveInput(
            gender="Female", age=70, sleep_hours=3.0, weight_kg=115.0,
            foods=["Fried","Processed Food","Sugar","Caffeine","Gluten","Dairy","Other"],
            symptoms=["Diarrhea","Abdominal Pain","Heartburn","Nausea",
                      "Bloating","Constipation","Fatigue","Gas"],
        )).score >= 0

    def test_empty_profile_high(self):
        from scorer import calculate_score, DigestiveInput
        assert calculate_score(DigestiveInput(gender="Male", age=25, sleep_hours=8.0,
                               weight_kg=70.0, foods=[], symptoms=[])).score >= 80

    def test_grade_boundaries(self):
        from main import _grade
        cases = [(100,"Excellent"),(85,"Excellent"),(84,"Good"),(70,"Good"),
                 (69,"Fair"),(50,"Fair"),(49,"Poor"),(30,"Poor"),(29,"Critical"),(0,"Critical")]
        for score, expected in cases:
            assert _grade(score) == expected, f"_grade({score}) != {expected!r}"


class TestBayesianMemory:
    def test_neutral_prior(self):
        from user_symptom_memory import UserMemory
        assert abs(UserMemory.new("t").get_prior(5064,"Heartburn").posterior_mean - 0.5) < 0.01

    def test_confirmations_push_up(self):
        from user_symptom_memory import UserMemory
        m = UserMemory.new("t_up")
        for _ in range(5): m.record_cooccurrence(5064,"Heartburn",occurred=True)
        assert m.get_prior(5064,"Heartburn").posterior_mean > 0.5

    def test_denials_push_down(self):
        from user_symptom_memory import UserMemory
        m = UserMemory.new("t_dn")
        for _ in range(5): m.record_cooccurrence(5064,"Heartburn",occurred=False)
        assert m.get_prior(5064,"Heartburn").posterior_mean < 0.5

    def test_weight_schedule(self):
        from user_symptom_memory import UserMemory
        m = UserMemory.new("tw")
        assert m.personalisation_weight == 0.20
        m.total_food_logs=3; m.total_symptom_logs=2
        assert m.personalisation_weight == 0.40
        m.total_food_logs=10; m.total_symptom_logs=8
        assert m.personalisation_weight == 0.60
        m.total_food_logs=20; m.total_symptom_logs=15
        assert m.personalisation_weight == 0.75

    def test_roundtrip(self):
        from user_symptom_memory import UserMemory
        m = UserMemory.new("tr")
        m.record_cooccurrence(5064,"Heartburn",occurred=True); m.total_food_logs=10
        r = UserMemory.from_dict(m.to_dict())
        assert r.total_food_logs == 10
        assert abs(m.get_prior(5064,"Heartburn").posterior_mean -
                   r.get_prior(5064,"Heartburn").posterior_mean) < 0.001

    def test_explicit_stronger_than_auto(self):
        from user_symptom_memory import UserMemory
        a = UserMemory.new("ta"); e = UserMemory.new("te")
        a.record_cooccurrence(5064,"Heartburn",occurred=True,strength=1.0)
        e.apply_explicit_feedback(5064,"Heartburn",confirmed=True)
        assert (e.get_prior(5064,"Heartburn").posterior_mean >
                a.get_prior(5064,"Heartburn").posterior_mean)