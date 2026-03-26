"""
symptom_note_analyser.py
────────────────────────
Analyses a free-text symptom note submitted by the user alongside a
POST /log/symptom request.

Two jobs:
    1. Detect known gut symptoms mentioned in the note that the user
       did NOT already list in `symptom`.  These are "note-detected"
       symptoms — penalised at 50% weight because they are inferred,
       not explicitly confirmed.

    2. Detect non-gut symptoms (e.g. "headache", "dizziness", "fever")
       that are real physical complaints but fall outside the 10-label
       gut taxonomy. These get a small flat penalty and a hint.

Detection approach
──────────────────
Uses keyword/phrase matching — NOT zero-shot NLI.

Why not NLI for notes?
    DeBERTa NLI multi_label=True is designed for long premise-hypothesis
    pairs. On a short 10-word user note it distributes high confidence
    broadly across all labels. "Stomach feels tight" scores 0.99 for
    Abdominal Pain AND 0.97 for Cramps AND 0.95 for Heartburn AND 0.83
    for Fatigue — all wrong except the first two. Threshold tuning cannot
    fix this; the model simply lacks discriminative power on micro-text.

    Keyword matching is deterministic, fast, and controllable:
    - "headache" -> headache detected (not gut)
    - "stomach feels tight" -> Cramps / Abdominal Pain detected
    - "tired", "exhausted" -> Fatigue detected
    - NO false positives for symptoms with zero keyword match

No ML dependency — no classifier injection needed.

Public API
──────────
    analyse_symptom_note(note_text,
                         already_logged,
                         severity)        -> NoteAnalysisResult

    NoteAnalysisResult fields:
        detected_symptoms       : list[DetectedSymptom]
        total_additional_penalty: int
        analysis_summary        : str
        novel_symptom_flag      : bool
        novel_symptom_hint      : str
        raw_scores              : dict[str, float]   # match count per label
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Gut symptom keyword map ───────────────────────────────────────────────────
# Each entry: symptom -> list of word/phrase patterns (case-insensitive).
# A note must contain at least one pattern to trigger that symptom.
# Patterns are matched as whole words (word-boundary anchored).

_GUT_KEYWORD_MAP: dict[str, list[str]] = {
    "Bloating": [
        "bloat", "bloating", "bloated", "distended", "distension",
        "swollen belly", "swollen stomach", "full stomach", "stomach is full",
        "puffiness", "puffy belly", "puffy stomach",
    ],
    "Gas": [
        r"\bgas\b", "gassy", "flatulence", "farting", "fart", "burping",
        "belching", "belch", "burp", "passing gas", "passing wind",
    ],
    "Constipation": [
        "constipat", "cannot poop", "can't poop", "no bowel", "hard stool",
        "hard stools", "difficult to pass", "straining", "no movement",
        "no bowel movement", "haven't gone", "not going to bathroom",
    ],
    "Diarrhea": [
        "diarrhea", "diarrhoea", "loose stool", "loose stools", "watery stool",
        "watery stools", "runny stool", "frequent stool", "liquid stool",
        "urgent stool", "urgent bowel",
    ],
    "Nausea": [
        "nausea", "nauseous", "nauseated", "feel like vomiting", "want to vomit",
        "queasy", "sick to my stomach", "sick stomach", "going to be sick",
        "urge to vomit",
    ],
    "Heartburn": [
        "heartburn", "heart burn", "burning in chest", "burning chest",
        "chest burning", "burning sensation in chest", "burning behind sternum",
        "burning after eating",
    ],
    "Abdominal Pain": [
        "abdominal pain", "stomach pain", "stomach ache", "stomachache",
        "belly pain", "belly ache", "bellyache", "pain in stomach",
        "pain in belly", "pain in abdomen", "abdominal ache",
        "stomach hurts", "belly hurts", "abdomen hurts",
        "stomach is hurting", "tummy ache", "tummy pain",
    ],
    "Cramps": [
        "cramp", "cramping", "crampy", "spasm", "spasms",
        "tight stomach", "tight belly", "tight abdomen",
        "stomach is tight", "stomach feels tight", "stomach very tight",
        "stomach feels very tight", "stomach is very tight",
        "tightness in stomach", "tightness in belly", "stomach tightness",
        "gut tightness", "stomach is cramping", "intestinal cramp",
        "belly is tight", "belly feels tight",
    ],
    "Fatigue": [
        "fatigue", "fatigued", "exhausted", "exhaustion", "tired", "tiredness",
        "no energy", "low energy", "feel weak", "weakness", "lethargy",
        "lethargic", "sluggish", "drained", "worn out",
    ],
    "Acid Reflux": [
        "acid reflux", "reflux", "regurgitation", "regurgitat",
        "acid coming up", "acid in throat", "sour taste", "bitter taste",
        "food coming back up", "food coming up", "stomach acid",
    ],
}

# ── Non-gut symptom keyword map ───────────────────────────────────────────────
# Maps a display label -> keyword patterns
# Non-gut symptoms get a flat -3 penalty each and set novel_symptom_flag.

_NON_GUT_KEYWORD_MAP: dict[str, list[str]] = {
    "headache": [
        "headache", "head ache", "head hurts", "head is pounding",
        "pounding head", "throbbing head", "head pain",
    ],
    "migraine": [
        "migraine", "migrain",
    ],
    "dizziness": [
        "dizzy", "dizziness", "lightheaded", "light-headed", "light headed",
        "vertigo", "room is spinning", "spinning",
    ],
    "fever": [
        "fever", "feverish", "chills", "shivering", "hot and cold",
        "temperature", "high temperature",
    ],
    "chest pain": [
        "chest pain", "chest ache", "chest pressure", "pressure in chest",
        "tight chest", "chest tightness",
    ],
    "back pain": [
        "back pain", "back ache", "backache", "lower back", "upper back",
        "spine pain",
    ],
    "joint or muscle pain": [
        "joint pain", "muscle pain", "muscle ache", "muscle aches",
        "body aches", "body pain", "sore muscles", "aching muscles",
        "aching joints", "joints ache", "joints hurt",
    ],
    "shortness of breath": [
        "short of breath", "shortness of breath", "can't breathe",
        "cannot breathe", "difficulty breathing", "hard to breathe",
        "breathless", "breathlessness",
    ],
    "skin rash": [
        "rash", "itching", "itchy skin", "hives", "skin irritation",
        "skin redness",
    ],
    "anxiety or palpitations": [
        "palpitations", "heart racing", "heart pounding", "racing heart",
        "anxiety", "anxious", "panic",
    ],
}

# ── Penalties ─────────────────────────────────────────────────────────────────

_BASE_PENALTY: dict[str, int] = {
    "Bloating":       -6,
    "Gas":            -5,
    "Constipation":   -8,
    "Diarrhea":       -12,
    "Nausea":         -9,
    "Heartburn":      -10,
    "Abdominal Pain": -11,
    "Cramps":         -10,
    "Fatigue":        -7,
    "Acid Reflux":    -10,
}

_SEVERITY_MULTIPLIER: dict[str, float] = {
    "Mild":     0.5,
    "Moderate": 1.0,
    "Severe":   1.6,
}

_NOTE_CONFIDENCE_WEIGHT = 0.50   # note-detected symptoms penalised at 50%
_NOVEL_SYMPTOM_PENALTY  = -3     # flat penalty per non-gut symptom found


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class DetectedSymptom:
    symptom:         str
    confidence:      float        # 1.0 for keyword match (exact), 0.0 otherwise
    penalty:         int
    note:            str
    matched_phrases: list[str]    # which keyword(s) triggered the match


@dataclass
class NoteAnalysisResult:
    detected_symptoms:        list[DetectedSymptom] = field(default_factory=list)
    total_additional_penalty: int                    = 0
    analysis_summary:         str                    = ""
    novel_symptom_flag:       bool                   = False
    novel_symptom_hint:       str                    = ""
    raw_scores:               dict[str, float]       = field(default_factory=dict)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _find_matches(text: str, patterns: list[str]) -> list[str]:
    """
    Return which patterns from the list appear in text.
    Patterns that start with \\b are used as-is (regex).
    All others are matched as whole-word substrings (word-boundary wrapped).
    """
    found: list[str] = []
    for pat in patterns:
        if pat.startswith(r"\b"):
            # Already a regex pattern — use directly
            if re.search(pat, text, re.IGNORECASE):
                found.append(pat.strip(r"\b"))
        else:
            # Escape for regex, wrap in word boundaries
            escaped = re.escape(pat)
            if re.search(rf"\b{escaped}\b", text, re.IGNORECASE):
                found.append(pat)
    return found


def _compute_gut_penalty(symptom: str, severity: str) -> int:
    base = _BASE_PENALTY.get(symptom, -6)
    mult = _SEVERITY_MULTIPLIER.get(severity, 1.0)
    raw  = base * mult * _NOTE_CONFIDENCE_WEIGHT
    return max(int(round(raw)), -15)


# ── Main public function ──────────────────────────────────────────────────────

def analyse_symptom_note(
    note_text:      str,
    already_logged: list[str],
    severity:       str = "Moderate",
    classifier:     Any = None,   # kept for API compatibility — not used
) -> NoteAnalysisResult:
    """
    Analyse a free-text symptom note using keyword matching.

    Parameters
    ----------
    note_text       : user's free-text description
    already_logged  : symptoms already submitted in the request — these are skipped
    severity        : severity from the SymptomLogRequest (used for penalty scaling)
    classifier      : ignored — kept for API compatibility with main.py

    Returns
    -------
    NoteAnalysisResult
    """
    note_text = note_text.strip()
    if not note_text:
        return NoteAnalysisResult(analysis_summary="Empty note.")

    normalised    = _normalise(note_text)
    already_set   = {s.lower() for s in already_logged}

    # ── Stage 1: Gut symptom keyword detection ────────────────────────────────
    detected:   list[DetectedSymptom] = []
    gut_penalty = 0
    raw_scores: dict[str, float] = {}

    for symptom, patterns in _GUT_KEYWORD_MAP.items():
        matches = _find_matches(normalised, patterns)
        raw_scores[symptom] = float(len(matches))   # match count as "score"

        if not matches:
            continue
        if symptom.lower() in already_set:
            continue   # already explicitly logged — skip

        penalty = _compute_gut_penalty(symptom, severity)
        gut_penalty += penalty
        detected.append(DetectedSymptom(
            symptom         = symptom,
            confidence      = 1.0,   # keyword match is deterministic
            penalty         = penalty,
            note            = (
                f"Keyword match in note: {', '.join(repr(m) for m in matches)}. "
                f"Penalty reduced to 50% ({penalty} pts) because inferred, not confirmed."
            ),
            matched_phrases = matches,
        ))

    # Sort by base penalty severity (worst first)
    detected.sort(key=lambda d: d.penalty)

    # ── Stage 2: Non-gut (novel) symptom keyword detection ────────────────────
    novel_found: list[tuple[str, list[str]]] = []

    for label, patterns in _NON_GUT_KEYWORD_MAP.items():
        matches = _find_matches(normalised, patterns)
        if matches:
            novel_found.append((label, matches))

    novel_flag    = bool(novel_found)
    novel_penalty = _NOVEL_SYMPTOM_PENALTY * len(novel_found) if novel_flag else 0
    novel_hint    = ""

    if novel_flag:
        label_list = ", ".join(label for label, _ in novel_found)
        top_label, top_matches = novel_found[0]
        novel_hint = (
            f"Note mentions non-gut symptom(s): {label_list}. "
            f"Matched keyword(s) for '{top_label}': {', '.join(repr(m) for m in top_matches)}. "
            f"These are outside the gut health tracking scope — "
            f"consider consulting a healthcare provider if they persist."
        )

    # ── Totals and summary ────────────────────────────────────────────────────
    total_penalty = gut_penalty + novel_penalty

    parts: list[str] = []

    if detected:
        names = ", ".join(d.symptom for d in detected)
        parts.append(
            f"Note analysis detected {len(detected)} additional gut symptom(s) "
            f"not explicitly listed: {names}."
        )

    if novel_flag:
        parts.append(novel_hint)

    if not detected and not novel_flag:
        parts.append(
            "Note received and analysed — no additional symptoms detected "
            "beyond what was already logged."
        )

    if total_penalty < 0:
        parts.append(f"Additional score adjustment from note: {total_penalty} pts.")

    return NoteAnalysisResult(
        detected_symptoms        = detected,
        total_additional_penalty = total_penalty,
        analysis_summary         = " ".join(parts),
        novel_symptom_flag       = novel_flag,
        novel_symptom_hint       = novel_hint,
        raw_scores               = raw_scores,
    )


# ── init_classifier kept for API compatibility — does nothing ─────────────────

def init_classifier(classifier: Any) -> None:  # noqa: ARG001
    """
    No-op. Keyword matching requires no ML model.
    Kept so main.py lifespan block does not need to change.
    """
    pass