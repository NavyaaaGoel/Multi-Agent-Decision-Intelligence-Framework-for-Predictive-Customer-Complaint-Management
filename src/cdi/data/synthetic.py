"""Synthetic complaint generator + structural 'world model'.

Why synthetic: the hackathon project had no labelled history of resolution times / SLA
outcomes. We therefore simulate a bank's complaint desk with a documented causal process:

  resolution_hours = route_median x issue_complexity x severity_speed x (1 + 0.9*team_load)
                     x fast_track x reroute x weekend x amount x repeat x channel x tier x lognormal noise

The same function (`resolution_hours`) is later used by the policy simulator to answer
"what *would* have happened under a different intervention", with common random numbers.
Models never see the latent noise or the issue complexity - they must learn them from data.
"""
from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from cdi import config, nlp, severity, taxonomy

SIGMA = 0.45
ROUTE_MEDIAN = {"auto_resolve": 3.0, "bank_escalation": 48.0, "partner_escalation": 84.0}
SEV_SPEED = {"Critical": 0.35, "High": 0.6, "Medium": 1.0, "Low": 1.4}
TIER_SPEED = {"Standard": 1.0, "Premium": 0.95, "Priority": 0.9}
CHANNEL_SPEED = {"Web": 1.0, "Email": 1.05, "Chatbot": 0.9, "Voice": 0.95, "Branch": 1.1, "OCR": 1.1}
LOAD_EFFECT = 0.9
FAST_TRACK_FACTOR = 0.65
REROUTE_FACTOR = 1.08

SYNONYMS = {
    "UPI": ["UPI", "UPI payment", "Google Pay", "PhonePe", "UPI transaction"],
    "Credit Card": ["credit card", "credit card account", "card statement"],
    "Debit Card": ["debit card", "ATM card", "debit card"],
    "ATM": ["ATM", "ATM machine", "cash withdrawal at ATM"],
    "Internet Banking": ["net banking", "internet banking", "online banking portal"],
    "Mobile Banking": ["mobile banking app", "banking app", "mobile app"],
    "Loans": ["loan", "loan account", "EMI", "home loan", "personal loan"],
    "Savings Account": ["savings account", "bank account", "account"],
    "Current Account": ["current account", "business account"],
    "Cheque Services": ["cheque", "cheque book", "cheque leaf"],
    "Fixed Deposit": ["fixed deposit", "FD", "term deposit"],
    "Recurring Deposit": ["recurring deposit", "RD account"],
    "Insurance": ["insurance policy", "insurance", "policy"],
    "Demat / Trading Account": ["demat account", "trading account", "shares"],
    "Forex / International Transactions": ["forex", "international transfer", "foreign remittance"],
    "Wallet / Prepaid Card": ["wallet", "prepaid card", "bank wallet"],
    "Customer Service": ["customer care", "customer service", "helpline"],
    "Branch Services": ["branch", "branch office", "bank branch"],
    "KYC / AML / Compliance": ["KYC", "KYC documents", "compliance team"],
    "Pension / Government Schemes": ["pension", "government scheme", "pension credit"],
    "Lockers": ["locker", "bank locker", "safe deposit locker"],
    "Merchant Services / POS": ["POS machine", "merchant account", "merchant services"],
    "Agriculture / Priority Sector Loans": ["agriculture loan", "kisan credit card", "crop loan"],
    "General Service Requests": ["service request", "request", "bank request"],
}
FRAMES = [
    "I am facing a problem with my {prod}. {issue}. {detail}",
    "{issue} on my {prod}. {detail}",
    "Writing to complain about {prod}: {issue}. {detail}",
    "This is regarding my {prod} - {issue}. {detail}",
    "Issue with {prod}. {detail} {issue}.",
]


def issue_complexity(product: str, issue: str) -> float:
    """Deterministic hidden per-issue difficulty in [0.7, 1.5]."""
    h = int(hashlib.md5(f"{product}|{issue}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return 0.7 + 0.8 * h


# ---------------------------------------------------------------- team load
class TeamLoadModel:
    """Hourly team utilisation (0.1-1.6): weekly cycle + slow AR(1) drift. Deterministic."""

    def __init__(self, start: datetime, days: int, seed: int = config.SEED + 1):
        rng = np.random.default_rng(seed)
        self.start, self.hours = start, days * 24
        t = np.arange(self.hours)
        self._series = {}
        for team in config.ALL_TEAMS:
            if team == config.AUTOMATION_TEAM:
                self._series[team] = np.full(self.hours, 0.15)
                continue
            ar = np.zeros(self.hours)
            eps = rng.normal(0, 0.05, self.hours)
            for i in range(1, self.hours):
                ar[i] = 0.98 * ar[i - 1] + eps[i]
            base = rng.uniform(0.5, 0.9)
            phase = rng.uniform(0, 2 * np.pi)
            self._series[team] = np.clip(base + 0.2 * np.sin(2 * np.pi * t / 168 + phase) + ar, 0.1, 1.6)

    def load(self, team: str, ts: datetime) -> float:
        idx = int((pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None) - self.start).total_seconds() // 3600)
        idx = min(max(idx, 0), self.hours - 1)
        return float(self._series[team][idx])


# ---------------------------------------------------------------- world model
def _m(values, mapping):
    return np.array([mapping[v] for v in values], dtype=float)


def resolution_hours(df: pd.DataFrame, fast_tracked=None, rerouted=None, team_load=None,
                     noise=None) -> np.ndarray:
    """Structural resolution time (hours). Optional overrides enable counterfactuals."""
    ft = df["fast_tracked"].to_numpy(float) if fast_tracked is None else np.asarray(fast_tracked, float)
    rr = df["rerouted"].to_numpy(float) if rerouted is None else np.asarray(rerouted, float)
    ld = df["team_load"].to_numpy(float) if team_load is None else np.asarray(team_load, float)
    eps = df["noise"].to_numpy(float) if noise is None else np.asarray(noise, float)
    cx = np.array([issue_complexity(p, i) for p, i in zip(df["product"], df["issue"])])
    sub = pd.to_datetime(df["submitted_at"])
    weekend = (sub.dt.dayofweek >= 5).to_numpy(float)
    h = (_m(df["route"], ROUTE_MEDIAN) * cx * _m(df["severity_label"], SEV_SPEED)
         * (1 + LOAD_EFFECT * ld)
         * np.where(ft > 0, FAST_TRACK_FACTOR, 1.0) * np.where(rr > 0, REROUTE_FACTOR, 1.0)
         * (1 + 0.15 * weekend)
         * (1 + 0.06 * np.log1p(df["amount"].to_numpy(float) / 1000.0))
         * (1 + 0.08 * df["repeat_count"].to_numpy(float))
         * _m(df["channel"], CHANNEL_SPEED) * _m(df["tier"], TIER_SPEED)
         * np.exp(SIGMA * eps))
    return np.maximum(0.25, h)


# ---------------------------------------------------------------- text
def _narrative(rng, spec, anger, amount, repeat, when) -> str:
    syn = SYNONYMS[spec.product]
    prod = rng.choice(syn)
    if rng.random() < 0.10:                                   # ambiguity / mislabel noise
        prod = rng.choice(SYNONYMS[rng.choice(list(SYNONYMS))])
    words = spec.issue.lower().split()
    if len(words) > 2 and rng.random() < 0.35:                # paraphrase-by-dropout
        keep = max(2, int(round(len(words) * rng.uniform(0.5, 0.9))))
        idx = sorted(rng.sample(range(len(words)), keep))
        words = [words[i] for i in idx]
    detail = [f"This happened on {when:%d %b}."]
    if amount > 0:
        detail.append(rng.choice([f"The amount involved is Rs. {int(amount):,}.",
                                  f"Amount: ₹{int(amount)}.", f"INR {int(amount):,} is affected."]))
    if repeat > 0 or rng.random() < 0.3:
        detail.append(f"I have already contacted customer care {max(1, repeat + rng.randint(0, 2))} times.")
    detail.append(f"Reference {rng.randint(100000, 999999)}.")
    extra = []
    n_neg = np.random.default_rng(rng.randint(0, 10**9)).poisson(anger * 3)
    extra += rng.sample(nlp.NEG_PHRASES, min(n_neg, len(nlp.NEG_PHRASES)))
    if rng.random() < anger ** 2 * 0.9:
        extra.append(rng.choice(nlp.THREAT_PHRASES))
    n_pol = np.random.default_rng(rng.randint(0, 10**9)).poisson((1 - anger) * 1.5)
    extra += rng.sample(nlp.POLITE_PHRASES, min(n_pol, len(nlp.POLITE_PHRASES)))
    rng.shuffle(extra)
    body = rng.choice(FRAMES).format(prod=prod, issue=" ".join(words).capitalize(), detail=" ".join(detail))
    return (body + " " + " ".join(extra)).strip()


# ---------------------------------------------------------------- generator
def generate(n: int = 9000, seed: int = config.SEED, start: datetime = datetime(2026, 1, 1),
             days: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    load_model = TeamLoadModel(start, days + 10, seed + 1)

    n_cust = 2500
    cust_tier = rng.choice(config.TIERS, size=n_cust, p=[0.70, 0.22, 0.08])
    cust_w = rng.pareto(2.5, n_cust) + 1.0
    cust_w /= cust_w.sum()

    specs = list(taxonomy.TAXONOMY.values())
    w = np.array([np.exp(-0.12 * s.base_severity) for s in specs]) * rng.lognormal(0, 0.6, len(specs))
    w /= w.sum()

    hour_w = np.array([1] * 8 + [4] * 10 + [3] * 3 + [1.5] * 3, float)
    hour_w /= hour_w.sum()
    day_idx = rng.integers(0, days, n)
    hr = rng.choice(24, size=n, p=hour_w)
    mn = rng.integers(0, 60, n)
    stamps = sorted(start + timedelta(days=int(d), hours=int(h), minutes=int(m))
                    for d, h, m in zip(day_idx, hr, mn))

    history: dict = {}
    rows = []
    for i, ts in enumerate(stamps):
        ci = int(rng.choice(n_cust, p=cust_w))
        tier = str(cust_tier[ci])
        spec = specs[int(rng.choice(len(specs), p=w))]
        prev = [t for t in history.get(ci, []) if (ts - t).days < 30]
        repeat = len(prev)
        history.setdefault(ci, []).append(ts)

        mention = 0.0 if spec.product == "General Service Requests" else 0.65
        amount = float(round(rng.lognormal(np.log(4000), 1.2), -1)) if rng.random() < mention else 0.0
        anger = float(np.clip(rng.beta(2, 4) + 0.07 * min(repeat, 4) + 0.02 * (spec.base_severity - 5), 0, 1))
        text = _narrative(prng, spec, anger, amount, repeat, ts - timedelta(days=prng.randint(0, 3)))
        # the pipeline re-derives these from the text, so derive them the same way here
        amount_x = nlp.extract_amount(text)
        sent = nlp.sentiment_score(text)
        thr = nlp.threat_count(text)
        sev_score, sev_label = severity.compute(spec.base_severity, amount_x, sent)

        team, load = spec.team, load_model.load(spec.team, ts)
        p_ft = {"Critical": 0.6, "High": 0.3, "Medium": 0.08, "Low": 0.02}[sev_label] + (0.1 if tier != "Standard" else 0)
        fast = int(spec.route != "auto_resolve" and rng.random() < p_ft)
        rerouted = 0
        if spec.route == "bank_escalation" and spec.backup_team:
            if rng.random() < (0.05 + (0.25 if load > 1.0 else 0.0)):
                rerouted, team = 1, spec.backup_team
                load = load_model.load(team, ts)

        rows.append(dict(
            complaint_id=f"C{i:06d}", customer_id=f"CU{ci:05d}", tier=tier, submitted_at=ts,
            channel=str(rng.choice(config.CHANNELS, p=[0.28, 0.2, 0.17, 0.15, 0.12, 0.08])),
            product=spec.product, issue=spec.issue, route=spec.route, team=team,
            backup_team=spec.backup_team, narrative=text, amount=amount_x, repeat_count=repeat,
            sentiment=sent, threat_count=thr, severity_score=sev_score, severity_label=sev_label,
            sla_hours=config.SLA_HOURS[sev_label], team_load=round(load, 4), fast_tracked=fast,
            rerouted=rerouted, noise=float(rng.normal()), _anger=anger))

    df = pd.DataFrame(rows)
    df["resolution_hours"] = np.round(resolution_hours(df), 2)
    df["breached"] = (df["resolution_hours"] > df["sla_hours"]).astype(int)
    logit = (-3.4 + 2.0 * df["_anger"] + 0.25 * df["repeat_count"] + 1.0 * df["breached"]
             + 0.12 * (df["severity_score"] - 5) + 0.5 * df["threat_count"]
             + 0.3 * (df["tier"] != "Standard"))
    df["escalated"] = (rng.random(len(df)) < 1 / (1 + np.exp(-logit))).astype(int)
    return df.drop(columns=["_anger"])


def chronological_split(df: pd.DataFrame, train=0.70, val=0.15):
    df = df.sort_values("submitted_at").reset_index(drop=True)
    a, b = int(len(df) * train), int(len(df) * (train + val))
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()
