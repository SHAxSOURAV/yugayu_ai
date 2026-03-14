# ══════════════════════════════════════════════════════════════════════════════
# Natural Text  →  USDA Food Name + USDA Food ID  Pipeline
#
# Model 1 — Dizex/InstaFoodRoBERTa-NER
#   Extracts raw food words/phrases from any natural sentence.
#
# Model 2 — google/flan-t5-base
#   Normalises the raw extracted food name into a clean USDA-style name.
#
# Model 3 — sentence-transformers/all-MiniLM-L6-v2
#   Embeds the normalised name + all USDA descriptions to find the
#   closest matching row → returns the real USDA ID.
#
# Dataset — demomaster/usda-national-nutrient-database  (via KaggleHub)
#   Downloaded automatically on first run, cached locally after that.
#
# Install:
#   pip install kagglehub transformers torch sentencepiece accelerate
#   pip install sentence-transformers pandas numpy
#
# Kaggle credentials (one-time setup, choose either option):
#   Option A — kaggle.json  (recommended)
#     1. Go to https://www.kaggle.com/settings → API → "Create New Token"
#     2. Move the downloaded file:  mv kaggle.json ~/.kaggle/kaggle.json
#     3. Lock permissions:          chmod 600 ~/.kaggle/kaggle.json
#
#   Option B — environment variables
#     export KAGGLE_USERNAME="your_kaggle_username"
#     export KAGGLE_KEY="your_kaggle_api_key"
# ══════════════════════════════════════════════════════════════════════════════

import os
import glob

import kagglehub
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    T5Tokenizer,
    T5ForConditionalGeneration,
    pipeline,
)
from sentence_transformers import SentenceTransformer


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Download USDA dataset from Kaggle via KaggleHub
#
#   kagglehub.dataset_download() behaviour:
#     • First run  → downloads dataset and caches it locally
#     • Next runs  → returns the cached path instantly (no re-download)
#     • New Kaggle version released → automatically re-downloads
#
#   Returns: local folder path containing the dataset files
# ══════════════════════════════════════════════════════════════════════════════

KAGGLE_DATASET = "demomaster/usda-national-nutrient-database"

print(f"Step 1 | Downloading dataset: '{KAGGLE_DATASET}' ...")
dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
print(f"        Dataset path: {dataset_path}")

# ── Auto-discover the CSV inside the downloaded folder ───────────────────────
# kagglehub returns the root folder; we search recursively for CSV files.
csv_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)

if not csv_files:
    raise FileNotFoundError(
        f"No CSV file found inside: {dataset_path}\n"
        f"Files present: {os.listdir(dataset_path)}"
    )

# If multiple CSVs exist, pick the largest one (most likely the main table)
usda_csv_path = max(csv_files, key=os.path.getsize)
print(f"        Using CSV : {os.path.basename(usda_csv_path)}")
print(f"        Full path : {usda_csv_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Load the CSV and pre-compute USDA embeddings
#
#   We embed all USDA food descriptions once at startup so every
#   query lookup is just a fast dot-product, not a full re-embed.
# ══════════════════════════════════════════════════════════════════════════════

print("Step 2 | Loading USDA CSV ...")
usda_df = pd.read_csv(usda_csv_path)
usda_df.columns = [c.strip() for c in usda_df.columns]   # strip whitespace

# Resolve ID column (handles different naming conventions across datasets)
_id_col = next(
    (c for c in usda_df.columns if c.upper() in ("ID", "FDC_ID", "FOOD_ID", "NDB_NO")),
    usda_df.columns[0],   # fallback: first column
)

# Resolve Description column
_desc_col = next(
    (c for c in usda_df.columns if c.upper() in ("DESCRIPTION", "DESC", "FOOD_NAME", "LONG_DESC", "NAME")),
    usda_df.columns[1],   # fallback: second column
)

print(f"        Rows     : {len(usda_df):,}")
print(f"        ID col   : '{_id_col}'")
print(f"        Desc col : '{_desc_col}'")
print(f"        Columns  : {list(usda_df.columns)}\n")

print("Step 2 | Embedding USDA descriptions with SentenceTransformer ...")
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

usda_descriptions = usda_df[_desc_col].fillna("").tolist()
usda_embeddings   = st_model.encode(
    usda_descriptions,
    batch_size           = 256,
    show_progress_bar    = True,
    normalize_embeddings = True,   # unit vectors → dot product = cosine sim
    convert_to_tensor    = True,
)
print("        USDA embeddings ready.\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Load InstaFoodRoBERTa-NER
#
#   Fine-tuned RoBERTa that detects FOOD entity spans in informal text
#   (social media captions, casual meal descriptions, etc.)
# ══════════════════════════════════════════════════════════════════════════════

print("Step 3 | Loading InstaFoodRoBERTa-NER ...")

ner_tokenizer = AutoTokenizer.from_pretrained("Dizex/InstaFoodRoBERTa-NER")
ner_model     = AutoModelForTokenClassification.from_pretrained("Dizex/InstaFoodRoBERTa-NER")

# aggregation_strategy="simple" → merges sub-word tokens back into full words
# e.g. ["oat", "##meal"] → "oatmeal"
ner_pipe = pipeline(
    "ner",
    model                = ner_model,
    tokenizer            = ner_tokenizer,
    aggregation_strategy = "simple",
)
print("        NER model ready.\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Load Flan-T5-base
#
#   Instruction-finetuned T5 used zero-shot to normalise informal food
#   names into clean USDA-style names via a simple text prompt.
# ══════════════════════════════════════════════════════════════════════════════

print("Step 4 | Loading Flan-T5-base ...")

t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
t5_model     = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

print("        Flan-T5 ready.\n")


# ══════════════════════════════════════════════════════════════════════════════
# Helper A — Extract food entities from raw text  (NER stage)
# ══════════════════════════════════════════════════════════════════════════════

def extract_food_entities(text: str) -> list[dict]:
    """
    Run InstaFoodRoBERTa-NER on text.
    Returns list of { word, start, end, score }.

    Adjacent FOOD spans (within 1 char) are merged so that multi-word
    foods like "grilled chicken breast" come back as one entity.
    """
    raw = ner_pipe(text)

    merged: list[dict] = []
    for ent in raw:
        if ent["entity_group"] != "FOOD":
            continue
        if merged and ent["start"] - merged[-1]["end"] <= 1:
            merged[-1]["end"]   = ent["end"]
            merged[-1]["word"]  = text[merged[-1]["start"]: ent["end"]]
            merged[-1]["score"] = (merged[-1]["score"] + ent["score"]) / 2
        else:
            merged.append({
                "word":  ent["word"],
                "start": ent["start"],
                "end":   ent["end"],
                "score": ent["score"],
            })

    return [e for e in merged if e["word"].strip()]


# ══════════════════════════════════════════════════════════════════════════════
# Helper B — Normalise informal name → USDA-style name  (Flan-T5 stage)
# ══════════════════════════════════════════════════════════════════════════════

def normalise_to_usda_name(raw_food: str) -> str:
    """
    Zero-shot prompt asking Flan-T5 to rewrite an informal food name
    in USDA database format.

    "oatmeal"         → "Oatmeal, cooked, with no added fat"
    "grilled chicken" → "Chicken, broilers or fryers, breast, cooked, roasted"
    "banana"          → "Bananas, raw"
    """
    prompt = (
        "Convert the informal food name below into a standard USDA food "
        "database name. Return only the standardised name, nothing else.\n\n"
        f"Informal food name: {raw_food}\n"
        "Standard USDA name:"
    )

    inputs  = t5_tokenizer(prompt, return_tensors="pt")
    outputs = t5_model.generate(
        **inputs,
        max_new_tokens = 40,
        num_beams      = 4,
        early_stopping = True,
    )
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Helper C — Match normalised name → USDA row  (SentenceTransformer stage)
# ══════════════════════════════════════════════════════════════════════════════

def match_usda(normalised_name: str, top_k: int = 3) -> list[dict]:
    """
    Cosine-similarity search over pre-computed USDA embeddings.
    Returns top_k matches: { rank, usda_id, usda_description, similarity }
    """
    q_emb = st_model.encode(
        normalised_name,
        normalize_embeddings = True,
        convert_to_tensor    = True,
    )

    sims    = (usda_embeddings @ q_emb).cpu().numpy()
    top_idx = np.argsort(sims)[::-1][:top_k]

    return [
        {
            "rank":             rank,
            "usda_id":          int(usda_df.iloc[idx][_id_col]),
            "usda_description": str(usda_df.iloc[idx][_desc_col]),
            "similarity":       round(float(sims[idx]), 4),
        }
        for rank, idx in enumerate(top_idx, 1)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Full end-to-end pipeline
# ══════════════════════════════════════════════════════════════════════════════

def text_to_usda(text: str, top_k: int = 3) -> list[dict]:
    """
    Natural text → USDA food ID + description.

    Input : any meal description string
    Output: list of dicts (one per detected food):

        {
          "raw_food"        : "oatmeal",
          "ner_confidence"  : 0.9871,
          "normalised_name" : "Oatmeal, cooked, with no added fat",
          "usda_id"         : 8121,            ← real USDA ID from dataset
          "usda_description": "OATMEAL,INST,FORT,PLAIN,PREP W/WATER",
          "usda_similarity" : 0.8934,
          "usda_top_matches": [                ← top_k candidates
              {"rank":1, "usda_id":8121, "usda_description":"...", "similarity":0.89},
              {"rank":2, "usda_id":8126, "usda_description":"...", "similarity":0.85},
              {"rank":3, "usda_id":8003, "usda_description":"...", "similarity":0.82},
          ]
        }
    """
    entities = extract_food_entities(text)
    if not entities:
        print("  No food entities detected.")
        return []

    results = []
    for ent in entities:
        raw_food   = ent["word"].strip()
        normalised = normalise_to_usda_name(raw_food)   # Flan-T5
        matches    = match_usda(normalised, top_k)       # SentenceTransformer
        best       = matches[0]

        results.append({
            "raw_food":         raw_food,
            "ner_confidence":   round(ent["score"], 4),
            "normalised_name":  normalised,
            "usda_id":          best["usda_id"],
            "usda_description": best["usda_description"],
            "usda_similarity":  best["similarity"],
            "usda_top_matches": matches,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Run examples
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    test_inputs = [
        "I had a big bowl of oatmeal with banana for breakfast",
        "Lunch was grilled chicken breast with brown rice and broccoli",
        "Just ate some Greek yogurt and blueberries as a snack",
        "Tonight I had spaghetti with tomato sauce and garlic bread",
        "200g salmon fillet with mashed potatoes and green beans",
    ]

    for text in test_inputs:
        print(f"\n{'=' * 70}")
        print(f"Input : {text}")
        print('=' * 70)

        results = text_to_usda(text, top_k=3)

        for r in results:
            print(
                f"\n  Raw food        : {r['raw_food']}\n"
                f"  NER confidence  : {r['ner_confidence']:.2%}\n"
                f"  Normalised name : {r['normalised_name']}\n"
                f"  ── Best USDA match ──────────────────────────────\n"
                f"  USDA ID         : {r['usda_id']}\n"
                f"  USDA Description: {r['usda_description']}\n"
                f"  USDA Similarity : {r['usda_similarity']:.2%}\n"
                f"  ── Other candidates ─────────────────────────────"
            )
            for m in r["usda_top_matches"][1:]:
                print(
                    f"    #{m['rank']}  "
                    f"ID {m['usda_id']:>6}  "
                    f"({m['similarity']:.2%})  "
                    f"{m['usda_description']}"
                )