# Gut Health API

A personalised digestive health scoring system that combines USDA nutritional data, HuggingFace NLP models, Bayesian learning, and MongoDB persistence.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the API](#running-the-api)
6. [Seeding Demo Data](#seeding-demo-data)
7. [API Endpoints Reference](#api-endpoints-reference)
8. [File Structure](#file-structure)
9. [HuggingFace Models](#huggingface-models)
10. [MongoDB Collections](#mongodb-collections)
11. [Startup Time](#startup-time)
12. [Known Issues & Handover Notes for Backend Developer](#known-issues--handover-notes-for-backend-developer)

---

## Architecture Overview

```
User Request
     │
     ▼
FastAPI (main.py)
     │
     ├── food_text_to_usda.py    ← NER + T5 + SentenceTransformer + USDA dataset
     ├── nutrition_scorer.py     ← DeBERTa-v3 zero-shot classifier
     ├── food_symptom_predictor.py ← cross-encoder NLI causation model
     ├── food_recommender.py     ← Flan-T5 + cosine similarity over USDA
     ├── user_symptom_memory.py  ← Bayesian Beta prior per (user, food, symptom)
     ├── scorer.py               ← Rule-based onboarding score engine
     └── database.py             ← MongoDB CRUD layer (pymongo)
```

**All ML models are loaded once at startup and kept in memory. No model is loaded per-request.**

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | f-string syntax used throughout |
| MongoDB | 6.0+ | Running locally or via Atlas |
| RAM | 8 GB minimum | 4 HuggingFace models in memory simultaneously |
| Disk | 5 GB | Models cached by HuggingFace + Kaggle dataset |
| Kaggle account | Required | For USDA dataset download (one-time) |

---

## Installation

```bash
# 1. Clone the repo and enter the directory
git clone <repo-url>
cd gut-health-api

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up Kaggle credentials (one-time, for USDA dataset download)
#    Go to: https://www.kaggle.com/settings → API → "Create New Token"
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

---

## Configuration

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB` | `gut_health` | Database name |
| `ENV` | `development` | Set to `production` to disable demo seed |

**Never commit `.env` to version control.**

---

## Running the API

```bash
# Development (auto-reload on file changes)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Important:** Use `--workers 1` only. Multiple workers each load all 4 ML models into RAM.
> For horizontal scaling, run separate single-worker containers behind a load balancer.

**First startup takes 3–8 minutes** — HuggingFace downloads and caches all 4 models (~900 MB total) and Kaggle downloads the USDA dataset (~50 MB). Subsequent starts take 30–60 seconds.

### Swagger UI

```
http://localhost:8000/docs
```

All 12 endpoints are fully documented with example request/response bodies in the Swagger UI.

---

## Seeding Demo Data

Seeds MongoDB with 3 users and their full 7-day food + symptom history for testing:

```bash
# WARNING: This drops and recreates all collections
python demo_seed.py
```

**Demo users created:**

| User ID | Name | Age | Condition | Pattern |
|---|---|---|---|---|
| `user_anika` | Anika Rahman | 28F | Heartburn + Acid Reflux | Fried chicken → heartburn 2h later |
| `user_rafiul` | Rafiul Islam | 35M | Bloating + Cramps | Spicy food → bloating 3h later |
| `user_sadia` | Sadia Hossain | 45F | Constipation + Fatigue | White bread → constipation next morning |

After seeding, verify with:
```
GET http://localhost:8000/db/health
GET http://localhost:8000/user/user_anika/dashboard
```

---

## API Endpoints Reference

### Food Parsing

| Method | Path | Description |
|---|---|---|
| `POST` | `/food/parse` | Natural meal text → top-k USDA food IDs with candidates |
| `POST` | `/food/text-to-id` | Natural food text → single best USDA ID per food |
| `GET` | `/food/lookup/{usda_id}` | USDA ID → food name + all 16 nutrients (per 100g) |

**Example — parse meal text:**
```json
POST /food/parse
{ "text": "I had oatmeal with banana for breakfast", "top_k": 3 }
```

---

### Scoring

| Method | Path | Description |
|---|---|---|
| `POST` | `/score` | Onboarding questionnaire → baseline digestion score (call once) |
| `POST` | `/log/food` | Log a meal → updated digestion score |
| `POST` | `/log/symptom` | Log a symptom → updated digestion score |

**Example — log a meal:**
```json
POST /log/food
{
  "current_score": 65,
  "meal_type": "Dinner",
  "foods": [
    { "usda_id": 5064, "quantity": 250, "unit": "g" },
    { "usda_id": 11124, "quantity": 100, "unit": "g" }
  ]
}
```

**Allowed meal types:** `Breakfast` · `Lunch` · `Dinner` · `Snack`

**Allowed units:** `g` · `gram` · `kg` · `ml` · `l` · `oz` · `lb` · `piece` · `pieces` · `cup` · `tbsp` · `tsp` · `serving`

**Allowed symptoms:** `Bloating` · `Abdominal Pain` · `Nausea` · `Constipation` · `Heartburn` · `Gas` · `Fatigue` · `Acid Reflux` · `Cramps` · `Diarrhea`

**Allowed severity:** `Mild` · `Moderate` · `Severe`

---

### Prediction & Learning

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict/food-symptom` | Food logs + symptom logs → which food caused which symptom |
| `POST` | `/predict/feedback` | User confirms/denies a prediction → updates Bayesian prior |
| `GET` | `/user/{user_id}/learning-summary` | View what the model has learned for this user |

**How the causation prediction works:**

```
Clinical window filter → foods within digestion window per symptom type
         ↓
NLI scoring → cross-encoder/nli-deberta-v3-small rates each food premise
              against symptom hypothesis (entailment probability)
         ↓
Combined score = 35% NLI + 35% nutrient risk + 20% temporal fit + 10% quantity
         ↓
Bayesian personalisation → user's learned prior blended in (grows with log history)
```

**Digestion windows per symptom:**

| Symptom | Window |
|---|---|
| Heartburn / Acid Reflux | 0.25h – 3h before |
| Bloating / Gas | 0.5h – 8h before |
| Nausea | 0.5h – 4h before |
| Cramps / Abdominal Pain | 0.5h – 8h before |
| Diarrhea | 1h – 16h before |
| Constipation | 12h – 48h before |
| Fatigue | 1h – 12h before |

---

### Recommendations

| Method | Path | Description |
|---|---|---|
| `POST` | `/recommend/safe-foods` | Returns 5 personalised safe food recommendations in natural language |

Recommendations are:
- Nutritionally similar to what the user already eats (cosine similarity on macro vector)
- Lower in the nutrients that trigger their reported symptoms
- Excluded from already-eaten foods
- Described in natural language via Flan-T5 (not raw USDA codes)

---

### User & Database

| Method | Path | Description |
|---|---|---|
| `GET` | `/user/{user_id}/dashboard` | Full 7-day summary from MongoDB |
| `GET` | `/db/health` | MongoDB connection status + collection counts |

---

## File Structure

```
gut-health-api/
│
├── main.py                    ← FastAPI app — all 12 endpoints
├── scorer.py                  ← Rule-based onboarding score engine
├── nutrition_scorer.py        ← DeBERTa-v3 zero-shot nutrition classifier
├── food_text_to_usda.py       ← NER + T5 + SentenceTransformer pipeline
├── food_symptom_predictor.py  ← cross-encoder NLI causation predictor
├── food_recommender.py        ← Flan-T5 + USDA cosine similarity recommender
├── user_symptom_memory.py     ← Bayesian Beta prior learning engine
├── database.py                ← MongoDB CRUD layer
├── demo_seed.py               ← Seeds 3 demo users with 7-day history
│
├── requirements.txt
├── .env.example
├── .gitignore
└── tests/
    └── test_api.py            ← pytest smoke tests (no ML — mocked)
```

---

## HuggingFace Models

All models are downloaded automatically on first startup and cached by the HuggingFace library.

| Model | Size | Used for |
|---|---|---|
| `Dizex/InstaFoodRoBERTa-NER` | ~125 MB | Food entity extraction from text |
| `google/flan-t5-base` | ~250 MB | Food name normalisation + recommendations |
| `sentence-transformers/all-MiniLM-L6-v2` | ~90 MB | Semantic USDA search |
| `MoritzLaurer/deberta-v3-base-mnli-fever-anli` | ~180 MB | Nutrition digestibility scoring |
| `cross-encoder/nli-deberta-v3-small` | ~170 MB | Food→symptom causation NLI |

**Total: ~815 MB download on first run.**

Cache location: `~/.cache/huggingface/hub/`

---

## MongoDB Collections

| Collection | Schema | Indexes |
|---|---|---|
| `users` | `user_id, name, age, gender, sleep_hours, weight_kg, current_score, grade` | `user_id` (unique) |
| `food_logs` | `user_id, usda_id, usda_description, quantity_g, unit, meal_type, logged_at` | `(user_id, logged_at DESC)` |
| `symptom_logs` | `user_id, symptom, severity, logged_at, notes` | `(user_id, logged_at DESC)` |
| `user_memories` | Serialised Bayesian priors — full JSON blob per user | `user_id` (unique) |
| `score_history` | `user_id, event_type, previous_score, new_score, modifier, timestamp` | `(user_id, timestamp ASC)` |

---

## Startup Time

| Stage | Time (CPU) | Notes |
|---|---|---|
| USDA dataset download (first run) | 1–3 min | Kaggle API, cached after |
| NER model load | ~5s | |
| Flan-T5 load | ~8s | |
| SentenceTransformer load | ~3s | |
| USDA embedding (8k+ foods) | ~30s | Pre-computed at startup |
| DeBERTa zero-shot load | ~6s | |
| cross-encoder NLI load | ~5s | |
| MongoDB connect | <1s | |
| **Total cold start** | **~60s** | After first download |

---

## Known Issues & Handover Notes for Backend Developer

> Read this section carefully before integrating.

### 🔴 Critical

**1. No authentication**
All endpoints are open. `user_id` comes from the request body — any client can impersonate any user.
Add JWT middleware (e.g. `fastapi-jwt-auth` or `python-jose`) before any production use.
The `user_id` in all requests should come from the decoded JWT token, not the body.

**2. Blocking ML calls — concurrency issue**
All endpoint functions are `def` (synchronous). Heavy ML inference (NLI, T5, NER) blocks the event loop.
Wrap the ML-heavy endpoints (`/food/parse`, `/food/text-to-id`, `/predict/food-symptom`, `/recommend/safe-foods`) with `asyncio.run_in_executor`:
```python
import asyncio
result = await asyncio.get_event_loop().run_in_executor(None, ml_function, args)
```
Or run them as Celery background tasks and poll for results.

**3. `demo_seed.py` drops all data — guard it**
Add an environment check before running in any shared environment:
```python
if os.getenv("ENV") == "production":
    sys.exit("ERROR: demo_seed.py must not run in production.")
```

### 🟠 Important

**4. CORS — add before connecting any frontend**
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-app.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**5. `/db/health` exposes internal data — auth-gate it or remove in production**
Currently returns user names, scores, and collection counts to any caller.

**6. MongoDB errors surface as raw 500s**
`database.py` has minimal try/except. Add error handling to every DB function and return clean HTTP errors.

**7. `POST /predict/food-symptom` is not atomic**
If the server crashes after inference but before memory save, learning is lost. Wrap in try/finally.

### 🟡 Recommended Improvements

**8. Split `main.py` into routers** — it is 1,784 lines. Use `APIRouter` per domain (food, scoring, predict, recommend, user).

**9. Rate limiting** — `/food/parse` and `/predict/food-symptom` are expensive. Add `slowapi` or a Redis-backed rate limiter.

**10. Add `python-dotenv`** — load `.env` at startup:
```python
from dotenv import load_dotenv
load_dotenv()
```

**11. Recommendation diversity jitter uses `np.random`** — results differ on each call. Set `np.random.seed(hash(user_id) % 2**31)` before the scoring loop in `food_recommender.py` for reproducible results per user.