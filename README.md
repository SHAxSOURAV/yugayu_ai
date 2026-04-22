# Gut Health API

A personalised digestive health scoring system built with FastAPI, Claude AI (Anthropic), USDA FoodData Central, and MongoDB.

**Zero ML model weights. Startup time under 2 seconds. 512 MB RAM sufficient.**

---

## Architecture

```
User Request
     │
     ▼
FastAPI (main.py)  ←  slowapi rate limiter (per-IP)
     │
     ├── food_text_to_usda.py      ← Claude: extract + normalise food entities
     ├── nutrition_scorer.py       ← Claude: digestibility + meal score + symptom penalty
     ├── food_symptom_predictor.py ← Pure logic: nutrient-risk + temporal windows (no Claude)
     ├── food_recommender.py       ← Logic: safe-food ranking  |  Claude: personalised recs
     ├── meal_symptom_forecast.py  ← Reuses food_symptom_predictor logic (no Claude)
     ├── food_tag_classifier.py    ← Claude: zero-shot food tags + nutrient heuristics
     ├── diet_symptom_risk.py      ← Claude: diet → symptom probability
     ├── culprit_food_finder.py    ← Pure logic: temporal filter + heuristic NLI scoring
     ├── user_symptom_memory.py    ← Bayesian Beta prior per (user, food, symptom) — pure Python
     ├── scorer.py                 ← Rule-based onboarding score — pure Python
     ├── usda_client.py            ← USDA FoodData Central REST API + MongoDB cache
     └── database.py               ← MongoDB CRUD layer
```

> **Note:** `food_symptom_predictor.py` uses **zero Claude API calls**. Symptom causation is
> scored entirely through deterministic nutrient-risk thresholds, keyword heuristics, and
> temporal window matching. The module exports `batch_nli_score` and `_nli_score` for API
> compatibility, but these functions use the same pure-Python scoring internally.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | |
| MongoDB | 6.0+ | Local or Atlas |
| RAM | 512 MB | No model weights loaded |
| Claude API key | Required | [console.anthropic.com](https://console.anthropic.com) |
| USDA API key | Required (free) | [api.data.gov/signup](https://api.data.gov/signup) |

---

## Installation

```bash
git clone <repo-url>
cd gut-health-api

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your keys
```

---

## Configuration

`.env` variables:

| Variable | Required | Description |
|---|---|---|
| `CLAUDE_API_KEY` | Yes | Anthropic API key |
| `USDA_API_KEY` | Yes | USDA FoodData Central key (free at api.data.gov) |
| `MONGO_URI` | Yes | MongoDB connection string (default: `mongodb://localhost:27017`) |
| `MONGO_DB` | No | Database name (default: `gut_health`) |
| `ENV` | No | Set to `production` to block demo_seed.py |
| `MOCK_MODE` | No | Set to `true` to return fake data — no Claude, USDA, or MongoDB calls made |

> **Environment variable names are case-sensitive on Linux.** Use `CLAUDE_API_KEY` exactly.
> The app also accepts the legacy form `Claude_API_key` as a fallback.

---

## Running

```bash
# Development (single worker, auto-reload)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production (Procfile — 2 workers, proxy headers for correct rate-limit IP detection)
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 2 --proxy-headers --forwarded-allow-ips='*'
```

Swagger UI: `http://localhost:8000/docs`

> **Workers:** Each worker is an independent process with its own Claude/USDA clients
> (stateless HTTP — safe to duplicate). The in-memory `UserMemoryStore` fallback is
> per-process; run MongoDB so all workers share persisted state. Increase to `--workers 4`
> on hosts with ≥ 1 GB RAM available.

> **Mock mode:** `MOCK_MODE=true` skips all Claude, USDA, and MongoDB calls. Every endpoint
> returns realistic stub data. Use this for frontend development and CI pipelines.

---

## Seeding Demo Data

```bash
python demo_seed.py
```

Creates 3 demo users with 7-day food + symptom history. Blocked when `ENV=production`.

| User ID | Name | Demo Pattern |
|---|---|---|
| `user_anika` | Anika Rahman, 28F | Fried chicken → Heartburn |
| `user_rafiul` | Rafiul Islam, 35M | Spicy ramen → Bloating + Cramps |
| `user_sadia` | Sadia Hossain, 45F | White bread → Constipation |

---

## API Endpoints

### Food Parsing

| Method | Path | Rate limit | Description |
|---|---|---|---|
| `POST` | `/food/parse` | 20/min | Natural meal text → USDA IDs + optional gut score |
| `POST` | `/food/lookup` | 60/min | List of USDA IDs + weights → macros per food and totals |
| `POST` | `/food/tags` | 20/min | List of USDA IDs → top USDA categories with AI gut-health insight |

### Scoring

| Method | Path | Rate limit | Description |
|---|---|---|---|
| `POST` | `/score` | 60/min | Onboarding questionnaire → baseline gut score |
| `POST` | `/log/food` | 20/min | Log a meal in plain text → updated score |
| `POST` | `/log/symptom` | 20/min | Log symptoms → updated score |

### Prediction

| Method | Path | Rate limit | Description |
|---|---|---|---|
| `POST` | `/predict/meal-symptom-forecast` | 20/min | Uneaten meal → symptom risk forecast |
| `POST` | `/predict/feedback` | 60/min | User confirms/denies prediction → updates Bayesian prior |
| `POST` | `/culprit-foods` | 30/min | Food + symptom logs → top culprit foods (no `user_id` needed) |

### Recommendations

| Method | Path | Rate limit | Description |
|---|---|---|---|
| `POST` | `/recommend/safe_food` | 30/min | n personalised gut-friendly food recommendations |
| `POST` | `/recommend/risky_food` | 30/min | Which foods triggered which symptoms |
| `POST` | `/recommend/triggers_food` | 20/min | 1-sentence AI insight about foods that trigger a symptom |

### User & System

| Method | Path | Rate limit | Description |
|---|---|---|---|
| `GET` | `/user/{user_id}/dashboard` | 60/min | 7-day summary from MongoDB |
| `GET` | `/user/{user_id}/learning-summary` | 60/min | Bayesian personalisation summary |
| `GET` | `/db/health` | 60/min | MongoDB + USDA cache status |
| `POST` | `/scan/barcode` | 30/min | Barcode → product name (OpenFoodFacts) |

---

## Rate Limiting

Rate limits are enforced per client IP using [slowapi](https://github.com/laurentS/slowapi).

When running behind a reverse proxy (Heroku, nginx), the API reads `X-Forwarded-For` to
identify the real client IP so limits apply per end-user, not per load-balancer address.
The `Procfile` starts uvicorn with `--proxy-headers` to enable this.

On limit breach the API returns `HTTP 429 Too Many Requests`.

Limit tiers:
- **20/min** — endpoints that trigger Claude API calls (food parsing, logging, AI insights)
- **30/min** — endpoints that call external APIs or run heavy logic (barcode, culprit, recommendations)
- **60/min** — lightweight read/write endpoints (lookup, score, dashboard, feedback)

---

## Claude API Usage

All AI calls use `claude-sonnet-4-6`. Every call is a single-turn request — no streaming,
no conversation history maintained across calls.

| Module | What Claude does | max_tokens |
|---|---|---|
| `food_text_to_usda` | Extract + normalise food entities; decompose composite dishes | 600 |
| `nutrition_scorer` | Classify food digestibility (easy / difficult) | 80 |
| `nutrition_scorer` | Score meal gut impact (−5 to +5) + 10-12 word note | 120 |
| `nutrition_scorer` | Score symptom log penalty (0–40) + 7-10 word note | 80 |
| `food_recommender` | Generate n personalised food recommendations | 140n + 120 |
| `food_tag_classifier` | Classify food into 7 meal category tags (0–1 scores) | 120 |
| `diet_symptom_risk` | Predict 10 symptom probabilities (0–1) from diet summary | 180 |
| `main` (`/food/tags`) | 5-7 word gut insight + severity per USDA food category | 35n + 60 |
| `main` (`/recommend/triggers_food`) | 1-sentence clinical insight about trigger foods | 60 |

**Token savings vs original baseline:**

| Call site | Tokens saved per call |
|---|---|
| `food_text_to_usda` system prompt | ~420 input tokens |
| `nutrition_scorer` meal score system prompt | ~120 input tokens |
| `nutrition_scorer` symptom score system prompt | ~80 input tokens |
| `food_recommender` max_tokens (n=5) | ~350 output tokens |
| `diet_symptom_risk` max_tokens | ~120 output tokens |
| `/food/tags` max_tokens (5 categories) | ~120 output tokens |

---

## Scoring Logic

### Food Logging (`/log/food`)
1. **Claude** parses the plain-text meal into USDA-searchable components and decomposes composite dishes.
2. **USDA FoodData Central** resolves each component to a food ID.
3. **Claude** scores the assembled meal (−5 to +5) considering meal timing, nutritional quality, and gut-irritant signals.
4. A neutralising multiplier drifts the score toward the 80–90 healthy band — positive foods have more impact when the score is low; negative foods have more impact when the score is high.

### Symptom Logging (`/log/symptom`)
1. Symptoms and severity are validated against a fixed taxonomy (10 symptoms × 3 severity levels).
2. **Claude** scores the combined symptom picture (penalty 0–40) in a single call accounting for symptom count, severity, and the optional free-text note together.
3. The penalty is subtracted from the current score.

### Symptom Causation (`/recommend/risky_food`, `/predict/meal-symptom-forecast`, `/culprit-foods`)
Pure-logic engine — **zero Claude calls:**
- **Temporal window matching** — clinical digestion windows per symptom (e.g. Heartburn: 15 min–3 h; Constipation: 12–48 h)
- **Nutrient-risk scoring** — USDA nutrients vs per-symptom risk thresholds
- **Keyword heuristics** — food name keywords (fried, spicy, dairy, sugar, etc.) mapped to symptom risk scores

### Bayesian Personalisation (`UserMemory`)
Each `(usda_id, symptom)` pair starts with a neutral Beta(1, 1) prior.
Every confirmed co-occurrence increments α; every non-occurrence increments β.
Posterior mean = α / (α + β). Weight given to personal data grows with log count:

| Log count | Personal weight | Model weight |
|---|---|---|
| 0–4 | 20% | 80% |
| 5–14 | 40% | 60% |
| 15–29 | 60% | 40% |
| 30+ | 75% | 25% |

---

## MongoDB Collections

| Collection | Contents |
|---|---|
| `users` | User profiles and onboarding scores |
| `food_logs` | Every food entry logged by a user |
| `symptom_logs` | Every symptom entry logged by a user |
| `user_memories` | Serialised Bayesian priors per user |
| `score_history` | Score change events |
| `usda_cache` | Cached USDA nutrient profiles (permanent) |

Indexes are created automatically on first startup via `database._ensure_indexes()`.

---

## File Structure

```
gut-health-api/
├── main.py                    ← FastAPI app — all 16 endpoints + rate limiting
├── usda_client.py             ← USDA FoodData Central wrapper + MongoDB cache
├── food_text_to_usda.py       ← Claude: food entity extraction + USDA search
├── nutrition_scorer.py        ← Claude: digestibility, meal score, symptom penalty
├── food_symptom_predictor.py  ← Logic-only: nutrient risk + temporal causation (no Claude)
├── food_recommender.py        ← Logic: safe ranking  |  Claude: personalised recs
├── meal_symptom_forecast.py   ← Pre-meal symptom risk (reuses predictor logic, no Claude)
├── food_tag_classifier.py     ← Claude: zero-shot tags + nutrient heuristics
├── diet_symptom_risk.py       ← Claude: diet → symptom probability
├── culprit_food_finder.py     ← Logic: temporal filter + heuristic NLI scoring (no Claude)
├── user_symptom_memory.py     ← Bayesian Beta prior personalisation engine
├── scorer.py                  ← Rule-based onboarding score engine
├── database.py                ← MongoDB CRUD layer
├── text_context_parser.py     ← Meal type + quantity parser (pure stdlib, no Claude)
├── symptom_note_analyser.py   ← Keyword-based note analyser (no Claude)
├── scanner.py                 ← OpenFoodFacts barcode lookup
├── requirements.txt
├── Procfile
└── .env.example
```

---

## Production Checklist

Items marked ✅ are implemented in this codebase. Items marked ☐ are the backend developer's scope.

- ✅ Per-IP rate limiting (slowapi, 20–60 req/min per endpoint)
- ✅ Proxy-aware IP detection (X-Forwarded-For via `--proxy-headers`)
- ✅ Multi-worker Procfile (`--workers 2 --proxy-headers --forwarded-allow-ips='*'`)
- ✅ MOCK_MODE for CI/frontend development
- ✅ Claude API calls token-optimised across all modules
- ✅ MongoDB indexes on all queried fields
- ✅ Graceful degradation when MongoDB is unavailable (falls back to in-memory store)
- ✅ Input validation on all endpoints (Pydantic field constraints)
- ✅ Env var loaded with both `CLAUDE_API_KEY` and legacy `Claude_API_key` fallback
- ☐ JWT authentication (`user_id` must come from a decoded token, not the request body)
- ☐ Persist food and symptom logs in MongoDB from `/log/food` and `/log/symptom`
- ☐ Restrict CORS `allow_origins` to your frontend domain
- ☐ Auth-gate or remove `/db/health` endpoint
- ☐ Add index on `score_history` collection for `(user_id, timestamp)`
- ☐ Set `ENV=production` in deployment environment
- ☐ Confirm `MOCK_MODE` is not set in production
