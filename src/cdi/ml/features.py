"""Feature engineering shared by training, simulation and the live agents."""
from __future__ import annotations

import numpy as np
import pandas as pd

CAT_COLS = ["issue_key", "product", "route", "severity_label", "channel", "tier", "team"]
NUM_COLS = ["severity_score", "log_amount", "sentiment", "threat_count", "repeat_count",
            "team_load", "hour", "dow", "is_weekend", "text_len", "fast_tracked", "rerouted", "sla_hours"]
BASE_COLS = ["submitted_at", "product", "issue", "route", "severity_score", "severity_label",
             "sla_hours", "channel", "tier", "team", "amount", "narrative", "sentiment",
             "threat_count", "repeat_count", "team_load", "fast_tracked", "rerouted"]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Everything here is known at intake time (no leakage of outcomes)."""
    out = df.copy()
    sub = pd.to_datetime(out["submitted_at"])
    out["issue_key"] = out["product"] + "::" + out["issue"]
    out["log_amount"] = np.log1p(out["amount"].astype(float))
    out["hour"] = sub.dt.hour
    out["dow"] = sub.dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["text_len"] = out["narrative"].str.split().str.len()
    for c in ("fast_tracked", "rerouted", "threat_count", "repeat_count"):
        out[c] = out[c].astype(int)
    return out


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    d = add_derived(df)
    return d[CAT_COLS + NUM_COLS]
