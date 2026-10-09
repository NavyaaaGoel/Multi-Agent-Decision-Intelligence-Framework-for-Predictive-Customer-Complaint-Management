"""Light-weight, dependency-free text signals: sentiment, legal threats, money amounts."""
from __future__ import annotations

import re

NEG_PHRASES = [
    "This is completely unacceptable.", "I am extremely frustrated with this service.",
    "Worst experience with any bank.", "I feel cheated and harassed.",
    "Very disappointed and angry.", "This is pathetic.",
]
THREAT_PHRASES = [
    "I will approach the banking ombudsman.", "I will file a complaint with RBI.",
    "I will take this to consumer court.", "I am consulting my lawyer.",
    "I will post this on social media.", "I will report this to the police.",
]
POLITE_PHRASES = [
    "Kindly look into this.", "Please help me resolve this.",
    "I would appreciate a quick resolution.", "Thank you for your support.",
]

_NEG = {"unacceptable", "frustrated", "worst", "cheated", "harassed", "disappointed",
        "angry", "pathetic", "terrible", "horrible"}
_POS = {"kindly", "please", "appreciate", "thank", "thanks", "help"}
_THREAT_RX = re.compile(r"ombudsman|\brbi\b|consumer court|lawyer|social media|police|legal action", re.I)
_AMOUNT_RX = re.compile(r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)", re.I)
_WORD_RX = re.compile(r"[a-z]+")


def sentiment_score(text: str) -> float:
    """Lexicon sentiment in [-1, 1]; negative = angry / distressed."""
    words = _WORD_RX.findall(text.lower())
    neg = sum(w in _NEG for w in words)
    pos = sum(w in _POS for w in words)
    return round((pos - neg) / (pos + neg + 1.0), 4)


def threat_count(text: str) -> int:
    return len(_THREAT_RX.findall(text))


def extract_amount(text: str) -> float:
    """Largest rupee amount mentioned in the text (0.0 if none)."""
    vals = []
    for m in _AMOUNT_RX.finditer(text):
        try:
            vals.append(float(m.group(1).replace(",", "")))
        except ValueError:
            continue
    return max(vals) if vals else 0.0
