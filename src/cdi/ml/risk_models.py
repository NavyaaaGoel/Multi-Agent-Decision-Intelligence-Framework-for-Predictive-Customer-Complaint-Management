"""Predictive risk models.

  * breach probability  : equal-weight ensemble of three diverse members
        - gradient boosting classifier (isotonic-calibrated)
        - logistic regression
        - distributional: P(T > SLA) from the log-normal resolution-time model
    (ensemble chosen on the VALIDATION split; all members are reported separately)
  * escalation classifier: P(customer escalates)      same
  * resolution-time regressor: median hours (log-target HistGradientBoosting); residual sigma on
    a validation split gives a log-normal predictive distribution -> P90 and survival curves.
"""
from __future__ import annotations

import math

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from cdi import config
from cdi.ml.features import CAT_COLS, NUM_COLS, model_matrix

Z90 = 1.2816


def _pre() -> ColumnTransformer:
    return ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
         ("num", "passthrough", NUM_COLS)], sparse_threshold=0)


def _mask():
    return [True] * len(CAT_COLS) + [False] * len(NUM_COLS)


def _clf(seed):
    return Pipeline([("pre", _pre()), ("m", HistGradientBoostingClassifier(
        categorical_features=_mask(), max_iter=250, learning_rate=0.06, max_leaf_nodes=24,
        l2_regularization=1.0, random_state=seed))])


def _reg(seed):
    return Pipeline([("pre", _pre()), ("m", HistGradientBoostingRegressor(
        categorical_features=_mask(), max_iter=300, learning_rate=0.06, max_leaf_nodes=24,
        l2_regularization=1.0, random_state=seed))])


def lr_pipeline(seed=42):
    pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
                             ("n", StandardScaler(), NUM_COLS)])
    return Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=600, C=1.0, random_state=seed))])


def lognormal_sf(x: float, mu: float, sigma: float) -> float:
    """P(T > x) for T ~ LogNormal(mu, sigma)."""
    if x <= 0:
        return 1.0
    return 0.5 * math.erfc((math.log(x) - mu) / (sigma * math.sqrt(2)))


class RiskModels:
    def __init__(self, seed: int = config.SEED):
        self.seed = seed
        self.breach = self.lr = self.escalation = self.time_model = None
        self.sigma = 0.5
        self.breach_threshold = 0.5
        self.meta: dict = {}

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "RiskModels":
        Xtr, Xva = model_matrix(train), model_matrix(val)
        self.breach = CalibratedClassifierCV(_clf(self.seed), method="isotonic", cv=3).fit(Xtr, train["breached"])
        self.lr = lr_pipeline(self.seed).fit(Xtr, train["breached"])
        self.escalation = CalibratedClassifierCV(_clf(self.seed), method="isotonic", cv=3).fit(Xtr, train["escalated"])
        self.time_model = _reg(self.seed).fit(Xtr, np.log(train["resolution_hours"]))
        resid = np.log(val["resolution_hours"]) - self.time_model.predict(Xva)
        self.sigma = float(np.std(resid))
        p = self._breach_members(Xva, val["sla_hours"].to_numpy(), self.time_model.predict(Xva))["breach_prob"]
        pr, rc, th = precision_recall_curve(val["breached"], p)
        f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
        self.breach_threshold = float(th[int(np.nanargmax(f1[:-1]))])
        self.meta = {"n_train": len(train), "n_val": len(val), "sigma": self.sigma,
                     "breach_threshold": self.breach_threshold}
        return self

    def _breach_members(self, X, sla_hours, mu) -> dict:
        p_gb = self.breach.predict_proba(X)[:, 1]
        p_lr = self.lr.predict_proba(X)[:, 1]
        p_dist = 1 - norm.cdf((np.log(sla_hours) - mu) / self.sigma)
        return {"p_gb": p_gb, "p_lr": p_lr, "p_dist": p_dist, "breach_prob": (p_gb + p_lr + p_dist) / 3}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = model_matrix(df)
        mu = self.time_model.predict(X)
        mem = self._breach_members(X, df["sla_hours"].to_numpy(float), mu)
        return pd.DataFrame({
            **mem,
            "escalation_prob": self.escalation.predict_proba(X)[:, 1],
            "median_hours": np.exp(mu),
            "p90_hours": np.exp(mu + Z90 * self.sigma),
            "log_mu": mu}, index=df.index)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "RiskModels":
        return joblib.load(path)
