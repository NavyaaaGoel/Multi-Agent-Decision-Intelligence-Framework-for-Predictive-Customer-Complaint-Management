"""Severity scoring for the 161-issue taxonomy (ordered keyword rules + context bumps).

The hackathon severity table only covered 8 of the current issues, so this module
derives a base score from rules and then adjusts it with financial impact and sentiment.
"""
from __future__ import annotations

import re

_RULES = [
    (10, r"fraud|unauthori[sz]ed|phishing|skimming|stolen|harassment"),
    (9, r"card lost|pension not credited|pm kisan|cash not dispensed|cash deposit not reflected|staff misbehavior"),
    (8, r"amount debited|beneficiary not credited|freeze|frozen|disbursement|maturity amount incorrect|"
        r"credit bureau|claim rejected|wrong upi id|swift|international fund transfer|shares not credited|"
        r"merchant settlement|chargeback|insufficient balance error"),
    (7, r"transfer failed|withdrawal failed|payment failed|order failed|refund not received|"
        r"incorrect (interest|currency)|wrong charges|claim|locker access denied|emi paid but showing due|"
        r"auto[- ]?debit failed|bulk payment|premature .* failed|stop payment"),
    (6, r"delay|pending|rejected|retained|not received|not credited|not debited|not updated|aml"),
    (4, r"failed|not working|not opening|crashing|not responding|out of service|queue|not reachable|poor customer"),
    (3, r"request|update|forgot|incorrect upi pin|closure"),
]
_COMPILED = [(score, re.compile(p, re.I)) for score, p in _RULES]
DEFAULT_BASE = 5


def base_severity(issue: str) -> int:
    for score, rx in _COMPILED:
        if rx.search(issue):
            return score
    return DEFAULT_BASE


def label_for(score: float) -> str:
    if score >= 9:
        return "Critical"
    if score >= 7:
        return "High"
    if score >= 5:
        return "Medium"
    return "Low"


def compute(base: float, amount: float = 0.0, sentiment: float = 0.0) -> tuple[float, str]:
    """Adjust the issue's base severity with money at stake and customer sentiment."""
    score = float(base)
    if amount >= 100_000:
        score += 1.0
    elif amount >= 25_000:
        score += 0.5
    if sentiment <= -0.5:
        score += 0.5
    score = min(10.0, score)
    return score, label_for(score)
