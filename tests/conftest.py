# ── Path fix — must be the very first two lines of code ──────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────────────────────────────────────

"""
tests/conftest.py
─────────────────
Intercepts all heavy imports (kagglehub, torch, transformers,
sentence_transformers) at the sys.modules level BEFORE pytest
collects any test files.

Why this must be in conftest.py at module level (not inside a fixture):
  food_text_to_usda.py, nutrition_scorer.py, and food_symptom_predictor.py
  all run model-loading code at the TOP LEVEL on import — not inside
  functions. By the time any fixture runs, these modules have already
  been imported. conftest.py runs first, so we intercept here.

Strategy:
  1. Create a real temp directory with a real fake USDA CSV file.
  2. Make kagglehub.dataset_download() return that real temp dir path.
  3. Mock all HuggingFace and torch objects with correct return types
     so the top-level module code completes without crashing.
  4. After all modules import cleanly, the client fixture in test_api.py
     replaces _state with pre-configured test doubles.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Create a real temp directory with a real CSV file
# kagglehub returns a real path string; food_text_to_usda.py then calls
# glob.glob() and os.path.getsize() on it, so the path MUST actually exist.
# ─────────────────────────────────────────────────────────────────────────────

_TEMP_DIR = tempfile.mkdtemp(prefix="gut_health_test_")

_FAKE_CSV_PATH = os.path.join(_TEMP_DIR, "usda_nutrients.csv")

_FAKE_DF = pd.DataFrame({
    "ID":           [8121,  5064,  11124,  9040,  20109],
    "Description":  [
        "OATMEAL,INST,FORT,PLAIN,PREP W/WATER",
        "CHICKEN,BROILERS OR FRYERS,BREAST,FRIED",
        "CARROTS,RAW",
        "BANANAS,RAW",
        "RICE,WHITE,COOKED",
    ],
    "Calories":     [71.0,  219.0,  41.0,  89.0,  130.0],
    "Protein":      [2.5,   24.7,   0.9,   1.1,   2.7],
    "TotalFat":     [1.5,   8.8,    0.2,   0.3,   0.3],
    "Carbohydrate": [12.0,  7.8,    9.6,   23.0,  28.0],
    "Sodium":       [218.0, 338.0,  69.0,  1.0,   1.0],
    "SaturatedFat": [0.3,   2.3,    0.0,   0.1,   0.1],
    "Cholesterol":  [0.0,   73.0,   0.0,   0.0,   0.0],
    "Sugar":        [0.3,   0.0,    4.7,   12.2,  0.1],
    "Calcium":      [100.0, 10.0,   33.0,  5.0,   10.0],
    "Iron":         [4.2,   0.9,    0.3,   0.3,   0.2],
    "Potassium":    [95.0,  220.0,  320.0, 358.0, 35.0],
    "VitaminC":     [0.0,   0.0,    5.9,   8.7,   0.0],
    "VitaminE":     [0.1,   0.3,    0.4,   0.1,   0.0],
    "VitaminD":     [0.0,   0.0,    0.0,   0.0,   0.0],
})

# Write the real CSV so os.path.getsize() and pd.read_csv() both work
_FAKE_DF.to_csv(_FAKE_CSV_PATH, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Build fake kagglehub that returns the real temp dir path
# ─────────────────────────────────────────────────────────────────────────────

_mock_kagglehub = MagicMock()
_mock_kagglehub.dataset_download.return_value = _TEMP_DIR
sys.modules["kagglehub"] = _mock_kagglehub


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Build fake torch
# ─────────────────────────────────────────────────────────────────────────────

_mock_torch = MagicMock()

# torch.no_grad() must work as a context manager
_no_grad_ctx = MagicMock()
_no_grad_ctx.__enter__ = MagicMock(return_value=None)
_no_grad_ctx.__exit__ = MagicMock(return_value=False)
_mock_torch.no_grad.return_value = _no_grad_ctx

# torch.softmax must return a real tensor-like object with .tolist()
_fake_logits_row = MagicMock()
_fake_logits_row.tolist.return_value = [0.1, 0.7, 0.2]   # contradiction, entailment, neutral

_fake_softmax = MagicMock()
_fake_softmax.__getitem__ = MagicMock(return_value=_fake_logits_row)
_mock_torch.softmax.return_value = _fake_softmax

sys.modules["torch"] = _mock_torch


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Build fake transformers
# All .from_pretrained() calls return mocks that behave correctly
# ─────────────────────────────────────────────────────────────────────────────

def _make_tokenizer_mock():
    tok = MagicMock()
    # When called as tok(text, ...) return a dict-like that forwards **kwargs
    tok_output = MagicMock()
    tok_output.__iter__ = MagicMock(return_value=iter([]))
    # from_pretrained returns the same mock
    tok.from_pretrained.return_value = tok
    return tok


def _make_model_mock():
    model = MagicMock()
    model.from_pretrained.return_value = model
    model.eval.return_value = model

    # model(**inputs) → output with .logits
    logits_mock = MagicMock()
    logits_mock.__getitem__ = MagicMock(return_value=_fake_logits_row)
    output_mock = MagicMock()
    output_mock.logits = logits_mock
    model.return_value = output_mock

    # model.generate() → tensor ids, decode → string
    model.generate.return_value = [[0, 1, 2]]
    return model


_mock_transformers = MagicMock()

# AutoTokenizer
_auto_tokenizer        = _make_tokenizer_mock()
_mock_transformers.AutoTokenizer = _auto_tokenizer

# AutoModelForTokenClassification (NER)
_mock_ner_model        = _make_model_mock()
_mock_transformers.AutoModelForTokenClassification = _mock_ner_model

# AutoModelForSequenceClassification (cross-encoder NLI)
_mock_seq_model        = _make_model_mock()
_mock_transformers.AutoModelForSequenceClassification = _mock_seq_model

# T5Tokenizer — decode must return a real string
_t5_tokenizer          = _make_tokenizer_mock()
_t5_tokenizer.decode   = MagicMock(return_value="Oatmeal, cooked, no added fat")
_mock_transformers.T5Tokenizer = _t5_tokenizer

# T5ForConditionalGeneration
_t5_model              = _make_model_mock()
_t5_model.generate     = MagicMock(return_value=[[0]])
_mock_transformers.T5ForConditionalGeneration = _t5_model

# pipeline() — used for NER in food_text_to_usda.py
# Must return a callable that returns a list of entity dicts
def _fake_ner_pipeline(*args, **kwargs):
    def _call(text):
        return [{"entity_group": "FOOD", "word": "oatmeal",
                 "start": 0, "end": 7, "score": 0.98}]
    return _call

_mock_transformers.pipeline = _fake_ner_pipeline

sys.modules["transformers"]          = _mock_transformers
sys.modules["transformers.pipelines"] = MagicMock()


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Build fake sentence_transformers
# SentenceTransformer.encode() must return a real numpy array
# The @ operator (matrix multiply) is used for similarity: embeddings @ query
# ─────────────────────────────────────────────────────────────────────────────

_N_ROWS = len(_FAKE_DF)   # 5 rows in our fake dataset

class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, batch_size=32, show_progress_bar=False,
               normalize_embeddings=False, convert_to_tensor=False):
        # Return a real numpy array with correct shape
        if isinstance(texts, str):
            return np.random.rand(64).astype(np.float32)
        n = len(texts) if hasattr(texts, "__len__") else 5
        arr = np.random.rand(n, 64).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / (norms + 1e-9)
        return arr

_mock_st = MagicMock()
_mock_st.SentenceTransformer = _FakeSentenceTransformer

sys.modules["sentence_transformers"] = _mock_st


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Mock sentencepiece and accelerate (optional deps that may not exist)
# ─────────────────────────────────────────────────────────────────────────────

sys.modules.setdefault("sentencepiece", MagicMock())
sys.modules.setdefault("accelerate",    MagicMock())
sys.modules.setdefault("pymongo",       MagicMock())
sys.modules.setdefault("pymongo.errors",MagicMock())


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Mock database.py so it never connects to MongoDB
# ─────────────────────────────────────────────────────────────────────────────

_mock_db_module        = MagicMock()
_mock_db_instance      = MagicMock()
_mock_db_instance.connect.return_value = None

# list_users returns empty list by default (overridden per test in fixture)
_mock_db_instance.list_users.return_value = []
_mock_db_instance.get_user.return_value = {
    "user_id": "test_user", "name": "Test User",
    "current_score": 65, "grade": "Good",
}
_mock_db_instance.user_dashboard.return_value = {
    "user_id": "test_user", "name": "Test User",
    "current_score": 65, "grade": "Good",
    "food_logs_this_week": 10,
    "symptom_logs_this_week": 3,
    "unique_foods_eaten": ["Oatmeal", "Banana"],
    "symptom_frequency": {"Bloating": 2, "Heartburn": 1},
    "score_history": [],
    "personalisation_weight": 0.20,
    "top_sensitivities": [],
}
_mock_db_instance._col.return_value.count_documents.return_value = 5
_mock_db_instance._db.return_value.name = "gut_health"
_mock_db_instance._client = MagicMock()
_mock_db_instance._client.admin.command.return_value = {"ok": 1}

_mock_db_module.db = _mock_db_instance

# MongoUserMemoryStore — use real in-memory store instead
from user_symptom_memory import UserMemoryStore
_mock_db_module.MongoUserMemoryStore = UserMemoryStore

sys.modules["database"] = _mock_db_module


# ─────────────────────────────────────────────────────────────────────────────
# Expose the fake df and temp path so test_api.py can reference them
# ─────────────────────────────────────────────────────────────────────────────

FAKE_USDA_DF   = _FAKE_DF
FAKE_CSV_PATH  = _FAKE_CSV_PATH
FAKE_TEMP_DIR  = _TEMP_DIR