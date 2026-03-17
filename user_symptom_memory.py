"""
user_symptom_memory.py
──────────────────────
Per-user personalised learning engine for food → symptom causation.

━━━ WHY THE STATIC MODEL DOESN'T LEARN ━━━
The cross-encoder NLI model (cross-encoder/nli-deberta-v3-small) is frozen.
It was trained on generic text and has no memory of what any specific user ate.
It gives identical scores to two users who eat the same food, even if User A
always gets heartburn from it and User B never does.

━━━ SOLUTION: BAYESIAN PERSONALISATION ━━━
We maintain a per-user history of (usda_id, symptom) co-occurrences.
Each pair starts with a neutral Beta(1, 1) prior — no knowledge, no bias.

Every time the user logs food + later reports a symptom:
  → If the food was in the digestion window → co-occurrence confirmed → α += 1
  → If the food was in the window but symptom did NOT occur → β += 1

Bayesian posterior mean = α / (α + β)

This posterior is a learned probability specific to this user that:
  • Starts at 0.5 (no knowledge)
  • Moves toward 1.0 as food→symptom co-occurrences accumulate
  • Moves toward 0.0 if food is repeatedly eaten without that symptom
  • Converges to the true personal sensitivity as log count grows

━━━ PERSONALISATION WEIGHT SCHEDULE ━━━
The user's personal prior gains weight as their data grows:
  • 0–4 logs:   20% personal prior, 80% NLI model
  • 5–14 logs:  40% personal prior, 60% NLI model
  • 15–29 logs: 60% personal prior, 40% NLI model
  • 30+ logs:   75% personal prior, 25% NLI model

This prevents over-fitting to small samples while allowing strong
personalisation as the user builds a history.

━━━ STORAGE ━━━
UserMemory is a plain Python dataclass — fully JSON-serialisable.
Serialise with .to_dict() and store in any database (Postgres, MongoDB,
Redis, SQLite). Load with UserMemory.from_dict().

━━━ DAILY LOG FLOW ━━━
Normal flow every day:
  1. User logs food at meal time
     → call record_food_window_exposure() for each food × each recent symptom
  2. User logs symptom
     → call record_symptom_occurrence() for each symptom × each recent food
  3. POST /predict/food-symptom
     → NLI score is blended with user's personal prior
     → prediction improves daily

━━━ FEEDBACK FLOW ━━━
Optional explicit feedback (user confirms/denies a prediction):
  POST /predict/feedback
  → call apply_explicit_feedback()
  → strongest signal for learning — fast convergence
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Beta distribution helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BetaPrior:
    """
    Beta(alpha, beta) conjugate prior for a Bernoulli probability.
    Posterior mean = alpha / (alpha + beta).
    Starts at Beta(1,1) = uniform (0.5 mean, maximum uncertainty).
    """
    alpha:       float = 1.0   # pseudo-counts of "food caused symptom"
    beta:        float = 1.0   # pseudo-counts of "food did NOT cause symptom"
    total_windows: int = 0     # total times food was in digestion window
    last_updated:  str = ""

    @property
    def posterior_mean(self) -> float:
        """Expected probability that this food causes this symptom for this user."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence(self) -> float:
        """
        How confident we are in the estimate.
        Based on effective sample size: n = alpha + beta - 2 (subtract priors).
        Sigmoid-scaled: 0 at n=0, 0.5 at n=5, 0.9 at n=20.
        """
        n = max(0, self.alpha + self.beta - 2.0)
        return round(1.0 - 1.0 / (1.0 + n / 8.0), 4)

    @property
    def observations(self) -> int:
        """Real observations (excludes the initial Beta(1,1) pseudo-counts)."""
        return max(0, round(self.alpha + self.beta - 2))

    @property
    def confirmations(self) -> int:
        """Number of times this food was confirmed to cause this symptom."""
        return max(0, round(self.alpha - 1))

    def to_dict(self) -> dict:
        return {
            "alpha":         self.alpha,
            "beta":          self.beta,
            "total_windows": self.total_windows,
            "last_updated":  self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BetaPrior":
        return cls(
            alpha         = float(d.get("alpha", 1.0)),
            beta          = float(d.get("beta", 1.0)),
            total_windows = int(d.get("total_windows", 0)),
            last_updated  = str(d.get("last_updated", "")),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-user memory
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserMemory:
    """
    All learned data for one user.

    priors: dict keyed by "{usda_id}:{symptom}" → BetaPrior
    total_food_logs:    lifetime food log count for this user
    total_symptom_logs: lifetime symptom log count
    created_at:         ISO timestamp
    """
    user_id:            str
    priors:             dict[str, BetaPrior] = field(default_factory=dict)
    total_food_logs:    int = 0
    total_symptom_logs: int = 0
    created_at:         str = ""

    # ── personalisation weight based on data volume ──────────────────────────

    @property
    def personalisation_weight(self) -> float:
        """
        How much weight to give the user's personal prior vs the NLI model.
        Grows with log volume. Caps at 0.75 (never fully ignore the model).

          0–4 logs  → 0.20
          5–14      → 0.40
          15–29     → 0.60
          30+       → 0.75
        """
        n = self.total_food_logs + self.total_symptom_logs
        if   n >= 30: return 0.75
        elif n >= 15: return 0.60
        elif n >= 5:  return 0.40
        else:         return 0.20

    @property
    def model_weight(self) -> float:
        return round(1.0 - self.personalisation_weight, 2)

    def _key(self, usda_id: int, symptom: str) -> str:
        return f"{usda_id}:{symptom}"

    def get_prior(self, usda_id: int, symptom: str) -> BetaPrior:
        return self.priors.get(self._key(usda_id, symptom), BetaPrior())

    def _ensure_prior(self, usda_id: int, symptom: str) -> BetaPrior:
        k = self._key(usda_id, symptom)
        if k not in self.priors:
            self.priors[k] = BetaPrior()
        return self.priors[k]

    # ── update methods ────────────────────────────────────────────────────────

    def record_cooccurrence(
        self,
        usda_id:  int,
        symptom:  str,
        occurred: bool,
        strength: float = 1.0,
    ) -> None:
        """
        Update the prior for (usda_id, symptom).

        occurred=True  → food was in window when symptom occurred → α += strength
        occurred=False → food was in window, no symptom → β += strength
        strength allows soft updates (e.g. 0.5 for uncertain temporal fit)
        """
        prior = self._ensure_prior(usda_id, symptom)
        if occurred:
            prior.alpha += strength
        else:
            prior.beta  += strength
        prior.total_windows += 1
        prior.last_updated   = datetime.utcnow().isoformat()
        log.debug(
            f"Updated prior [{usda_id}:{symptom}] "
            f"α={prior.alpha:.2f} β={prior.beta:.2f} "
            f"mean={prior.posterior_mean:.3f}"
        )

    def apply_explicit_feedback(
        self,
        usda_id:   int,
        symptom:   str,
        confirmed: bool,
    ) -> None:
        """
        Explicit user feedback — stronger update (weight=2.0).
        confirmed=True:  user says "yes, that food caused my symptom"
        confirmed=False: user says "no, that food didn't cause it"
        """
        self.record_cooccurrence(usda_id, symptom, confirmed, strength=2.0)
        log.info(
            f"Explicit feedback [{usda_id}:{symptom}] confirmed={confirmed} "
            f"→ prior mean={self.get_prior(usda_id, symptom).posterior_mean:.3f}"
        )

    def increment_food_logs(self, count: int = 1) -> None:
        self.total_food_logs += count

    def increment_symptom_logs(self, count: int = 1) -> None:
        self.total_symptom_logs += count

    # ── personalised score blending ───────────────────────────────────────────

    def blend_score(
        self,
        usda_id:   int,
        symptom:   str,
        nli_score: float,
    ) -> tuple[float, dict]:
        """
        Blend the static NLI model score with the user's personal prior.

        Returns (blended_score, breakdown_dict)

        breakdown_dict contains the components for API transparency:
          personal_prior, confidence, personalisation_weight, model_weight, blended_score
        """
        prior         = self.get_prior(usda_id, symptom)
        personal_mean = prior.posterior_mean
        confidence    = prior.confidence
        p_weight      = self.personalisation_weight
        m_weight      = self.model_weight

        # Blend: personal prior weighted by how confident we are in it
        effective_personal = personal_mean * confidence
        effective_nli      = nli_score     * (1.0 - confidence * p_weight)

        blended = (
            p_weight * effective_personal +
            m_weight * effective_nli
        )
        # Also add a straight weighted blend for smoother behaviour at low confidence
        straight_blend = p_weight * personal_mean + m_weight * nli_score
        # Average the two blend strategies
        final = round((blended + straight_blend) / 2.0, 4)
        final = min(1.0, max(0.0, final))

        return final, {
            "personal_prior":         round(personal_mean, 4),
            "prior_confidence":       round(confidence, 4),
            "prior_observations":     prior.observations,
            "prior_confirmations":    prior.confirmations,
            "personalisation_weight": round(p_weight, 2),
            "model_weight":           round(m_weight, 2),
            "nli_score_input":        round(nli_score, 4),
            "blended_score":          final,
        }

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "user_id":            self.user_id,
            "total_food_logs":    self.total_food_logs,
            "total_symptom_logs": self.total_symptom_logs,
            "created_at":         self.created_at,
            "priors": {
                k: v.to_dict() for k, v in self.priors.items()
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "UserMemory":
        obj = cls(
            user_id            = str(d.get("user_id", "")),
            total_food_logs    = int(d.get("total_food_logs", 0)),
            total_symptom_logs = int(d.get("total_symptom_logs", 0)),
            created_at         = str(d.get("created_at", datetime.utcnow().isoformat())),
        )
        for k, v in d.get("priors", {}).items():
            obj.priors[k] = BetaPrior.from_dict(v)
        return obj

    @classmethod
    def from_json(cls, s: str) -> "UserMemory":
        return cls.from_dict(json.loads(s))

    @classmethod
    def new(cls, user_id: str) -> "UserMemory":
        return cls(user_id=user_id, created_at=datetime.utcnow().isoformat())


# ─────────────────────────────────────────────────────────────────────────────
# In-memory user store
# (replace with DB-backed store in production — see to_dict/from_dict)
# ─────────────────────────────────────────────────────────────────────────────

class UserMemoryStore:
    """
    In-memory store for UserMemory objects.

    Production usage:
      Override load() and save() to read/write from your database.
      The UserMemory.to_dict() / from_dict() methods handle serialisation.

    Example for Postgres:
        store.load(user_id) → SELECT memory_json FROM user_memories WHERE user_id = %s
        store.save(memory)  → INSERT ... ON CONFLICT DO UPDATE SET memory_json = %s
    """

    def __init__(self):
        self._store: dict[str, UserMemory] = {}

    def load(self, user_id: str) -> UserMemory:
        """Load or create a UserMemory for this user."""
        if user_id not in self._store:
            log.info(f"Creating new UserMemory for user: {user_id}")
            self._store[user_id] = UserMemory.new(user_id)
        return self._store[user_id]

    def save(self, memory: UserMemory) -> None:
        """Persist the updated memory."""
        self._store[memory.user_id] = memory

    def exists(self, user_id: str) -> bool:
        return user_id in self._store

    def summary(self, user_id: str) -> dict:
        """Return a human-readable learning summary for this user."""
        if not self.exists(user_id):
            return {"user_id": user_id, "status": "no data yet"}
        m = self.load(user_id)
        learned_pairs = [
            {
                "food_symptom_pair": k,
                "causation_probability": round(v.posterior_mean, 3),
                "observations": v.observations,
                "confirmations": v.confirmations,
                "confidence": round(v.confidence, 3),
                "last_updated": v.last_updated,
            }
            for k, v in m.priors.items()
            if v.observations > 0
        ]
        learned_pairs.sort(key=lambda x: x["causation_probability"], reverse=True)
        return {
            "user_id":                m.user_id,
            "total_food_logs":        m.total_food_logs,
            "total_symptom_logs":     m.total_symptom_logs,
            "personalisation_weight": m.personalisation_weight,
            "model_weight":           m.model_weight,
            "learned_pairs":          len(learned_pairs),
            "top_sensitivities":      learned_pairs[:5],
        }


# Singleton — shared across all requests
user_memory_store = UserMemoryStore()


# ─────────────────────────────────────────────────────────────────────────────
# Daily auto-update helper
# Called inside POST /predict/food-symptom after predictions are made
# ─────────────────────────────────────────────────────────────────────────────

from food_symptom_predictor import SYMPTOM_WINDOWS

def auto_update_from_logs(
    user_id:      str,
    food_logs:    list,              # list[FoodLogEntry]
    symptom_logs: list,              # list[SymptomLogEntry]
    store:        "UserMemoryStore | None" = None,  # inject the active store
) -> UserMemory:
    """
    Automatically update the user's Bayesian priors from today's logs.

    For each symptom:
      - Foods in digestion window → record_cooccurrence(occurred=True, strength=temporal_weight)
      - Foods NOT in window but logged that day → record_cooccurrence(occurred=False, strength=0.3)

    This runs every time POST /predict/food-symptom is called — no extra action needed.

    Args:
        store: Pass the active UserMemoryStore (e.g. MongoUserMemoryStore from _state).
               Defaults to the module-level in-memory singleton when None — useful for
               tests and standalone scripts, but callers with a persistent store MUST
               pass it explicitly so updates survive a server restart.
    """
    _store = store if store is not None else user_memory_store
    memory = _store.load(user_id)
    memory.increment_food_logs(len(food_logs))
    memory.increment_symptom_logs(len(symptom_logs))

    for sym_entry in symptom_logs:
        symptom  = sym_entry.symptom
        sym_time = sym_entry.logged_at
        win_min, win_max = SYMPTOM_WINDOWS.get(symptom, (0.5, 8.0))

        for food in food_logs:
            if food.logged_at >= sym_time:
                continue
            hours_before = (sym_time - food.logged_at).total_seconds() / 3600.0

            if win_min <= hours_before <= win_max:
                # Temporal strength: foods at window centre get strength=1.0,
                # edges get strength=0.5 — soft update for uncertain timing
                window_span   = win_max - win_min
                window_centre = win_min + window_span / 2.0
                distance      = abs(hours_before - window_centre)
                max_dist      = window_span / 2.0
                strength      = 0.5 + 0.5 * (1.0 - distance / max_dist)
                memory.record_cooccurrence(food.usda_id, symptom, occurred=True, strength=round(strength, 3))
            else:
                # Food was eaten that day but outside window — weak negative evidence
                memory.record_cooccurrence(food.usda_id, symptom, occurred=False, strength=0.3)

    _store.save(memory)
    log.info(
        f"Auto-updated memory for user {user_id}: "
        f"{memory.total_food_logs} food logs, "
        f"{memory.total_symptom_logs} symptom logs, "
        f"personalisation_weight={memory.personalisation_weight}"
    )
    return memory