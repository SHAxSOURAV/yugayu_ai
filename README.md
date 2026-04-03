# Gut Health API

A personalised digestive health scoring system using Claude AI, USDA FoodData Central, and MongoDB.

---

## Architecture

```
User Request
     │
     ▼
FastAPI (main.py)
     │
     ├── food_text_to_usda.py     ← Claude API (entity extract + normalise) + USDA FDC API
     ├── nutrition_scorer.py      ← Rule-based nutrient math + Claude API (digestibility)
     ├── food_symptom_predictor.py← Temporal filter + nutrient math + Claude API (NLI)
     ├── food_recommender.py      ← Claude API (personalised safe food descriptions)
     ├── meal_symptom_forecast.py ← Reuses food_symptom_predictor Claude NLI calls
     ├── food_tag_classifier.py   ← Claude API (zero-shot tags) + nutrient heuristics
     ├── diet_symptom_risk.py     ← Claude API (diet → symptom probability)
     ├── culprit_food_finder.py   ← Temporal filter + Claude API (batched NLI)
     ├── user_symptom_memory.py   ← Bayesian Beta prior per (user, food, symptom) — pure Python
     ├── scorer.py                ← Rule-based onboarding score engine — pure Python
     ├── usda_client.py           ← USDA FoodData Central REST API wrapper + MongoDB cache
     └── database.py              ← MongoDB CRUD layer
```

**Zero ML models loaded at startup. Zero RAM for model weights. Startup time: < 2 seconds.**

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | |
| MongoDB | 6.0+ | Local or Atlas |
| RAM | 512 MB | No model weights |
| Claude API key | Required | console.anthropic.com |
| USDA API key | Required (free) | api.data.gov/signup |

---

## Installation

```bash
git clone <repo-url>
cd gut-health-api

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and fill in your keys
```

---

## Configuration

`.env` variables:

| Variable | Description |
|---|---|
| `Claude_API_key` | Your Anthropic API key |
| `USDA_API_KEY` | USDA FoodData Central key (free at api.data.gov) |
| `MONGO_URI` | MongoDB connection string (default: `mongodb://localhost:27017`) |
| `MONGO_DB` | Database name (default: `gut_health`) |
| `ENV` | Set to `production` to block demo_seed.py |

---

## Running

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production (multiple workers safe — no shared model state)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

> **Note:** Multiple workers are now safe. The old single-worker restriction was
> caused by HuggingFace models held in RAM. Claude API calls are stateless HTTP.

Swagger UI: `http://localhost:8000/docs`

---

## Seeding Demo Data

```bash
python demo_seed.py
```

Creates 3 demo users with 7-day food + symptom history. Blocked in `ENV=production`.

| User ID | Name | Pattern |
|---|---|---|
| `user_anika` | Anika Rahman, 28F | Fried chicken → Heartburn |
| `user_rafiul` | Rafiul Islam, 35M | Spicy ramen → Bloating + Cramps |
| `user_sadia` | Sadia Hossain, 45F | White bread → Constipation |

---

## API Endpoints

### Food Parsing
| Method | Path | Description |
|---|---|---|
| `POST` | `/food/parse` | Natural meal text → top-k USDA food IDs |
| `POST` | `/food/text-to-id` | Natural food text → best USDA ID per food |
| `GET` | `/food/lookup/{usda_id}` | USDA ID → food name + 16 nutrients |
| `GET` | `/food/tags/{usda_id}` | USDA ID → meal category tags |

### Scoring
| Method | Path | Description |
|---|---|---|
| `POST` | `/score` | Onboarding questionnaire → baseline score |
| `POST` | `/log/food` | Log a meal → updated score |
| `POST` | `/log/symptom` | Log symptoms → updated score |

### Prediction
| Method | Path | Description |
|---|---|---|
| `POST` | `/predict/food-symptom` | Food + symptom logs → causation prediction |
| `POST` | `/predict/feedback` | User confirms/denies prediction → updates Bayesian prior |
| `POST` | `/predict/meal-symptom-forecast` | Uneaten meal → symptom risk forecast |
| `POST` | `/culprit-foods` | Food + symptom logs → top culprit foods |

### Recommendations & User
| Method | Path | Description |
|---|---|---|
| `POST` | `/recommend/safe-foods` | 5 personalised gut-friendly food recommendations |
| `GET` | `/user/{user_id}/dashboard` | 7-day summary from MongoDB |
| `GET` | `/user/{user_id}/learning-summary` | Bayesian personalisation summary |
| `GET` | `/db/health` | MongoDB + USDA cache status |
| `POST` | `/scan/barcode` | Barcode → product name (OpenFoodFacts) |

---

## Claude API Usage

All AI calls use `claude-sonnet-4-6`. Each call is purpose-built and token-efficient:

| Module | What Claude does | Max tokens |
|---|---|---|
| `food_text_to_usda` | Extract + normalise food entities from text | 400 |
| `nutrition_scorer` | Classify food digestibility (easy/difficult) | 120 |
| `food_symptom_predictor` | Batch NLI: food→symptom entailment scores | 80 × n pairs |
| `food_recommender` | Generate 5 personalised food recommendations | 200 × n |
| `food_tag_classifier` | Classify food into 7 meal category tags | 150 |
| `diet_symptom_risk` | Predict 10 symptom probabilities from diet | 300 |
| `culprit_food_finder` | Batch NLI: culprit food scoring | 80 × n pairs |

NLI calls (`food_symptom_predictor`, `culprit_food_finder`) batch all food×symptom pairs
into **one Claude call** per request — not one call per pair.

---

## USDA Data

Food data is fetched from the **USDA FoodData Central REST API** and cached in MongoDB
(`usda_cache` collection). Each food ID is fetched from the network only once.

USDA API key: free, instant at https://api.data.gov/signup

---

## MongoDB Collections

| Collection | Contents |
|---|---|
| `users` | User profiles and onboarding scores |
| `food_logs` | Every food entry logged by a user |
| `symptom_logs` | Every symptom entry logged by a user |
| `user_memories` | Serialised Bayesian priors per user |
| `score_history` | Score change events |
| `usda_cache` | Cached USDA nutrient profiles (permanent cache) |

---

## File Structure

```
gut-health-api/
├── main.py                    ← FastAPI app — all endpoints
├── usda_client.py             ← USDA FoodData Central API wrapper + cache  [NEW]
├── food_text_to_usda.py       ← Claude entity extract + USDA search
├── nutrition_scorer.py        ← Rule-based nutrient math + Claude digestibility
├── food_symptom_predictor.py  ← Temporal filter + Claude batched NLI
├── food_recommender.py        ← Claude personalised safe food recommendations
├── meal_symptom_forecast.py   ← Pre-meal symptom risk forecast
├── food_tag_classifier.py     ← Claude zero-shot tags + nutrient heuristics
├── diet_symptom_risk.py       ← Claude diet → symptom probability
├── culprit_food_finder.py     ← Temporal filter + Claude batched NLI
├── user_symptom_memory.py     ← Bayesian Beta prior learning engine  [unchanged]
├── scorer.py                  ← Rule-based onboarding score engine    [unchanged]
├── database.py                ← MongoDB CRUD layer                    [unchanged]
├── text_context_parser.py     ← Meal type + quantity parser           [unchanged]
├── symptom_note_analyser.py   ← Keyword-based note analyser           [unchanged]
├── scanner.py                 ← OpenFoodFacts barcode lookup           [unchanged]
├── demo_seed.py               ← Seeds 3 demo users (ENV guard added)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Production Checklist

- [ ] Add JWT authentication (`user_id` must come from decoded token, not request body)
- [ ] Restrict CORS `allow_origins` to your frontend domain
- [ ] Remove or auth-gate `/db/health` endpoint
- [ ] Add rate limiting (`slowapi`) on `/food/parse` and `/predict/food-symptom`
- [ ] Set `ENV=production` in your deployment environment
- [ ] Set `USDA_API_KEY` (your own key, not DEMO_KEY)
- [ ] Set `MONGO_URI` to your Atlas or production MongoDB URI
