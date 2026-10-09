"""Held-out evaluation of every ML component against explicit baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, mean_absolute_error,
                             median_absolute_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cdi.baselines.keyword_classifier import ClassifierAgent
from cdi.ml.features import CAT_COLS, NUM_COLS, model_matrix

R = lambda x, n=4: round(float(x), n)
ROUTE_RANK = {"auto_resolve": 0, "bank_escalation": 1, "partner_escalation": 2}


def _lr(seed=42):
    pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
                             ("n", StandardScaler(), NUM_COLS)])
    return Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=600, C=1.0, random_state=seed))])


def _cls_block(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"roc_auc": R(roc_auc_score(y, p)), "pr_auc": R(average_precision_score(y, p)),
            "brier": R(brier_score_loss(y, p)), "precision": R(precision_score(y, pred, zero_division=0)),
            "recall": R(recall_score(y, pred)), "f1": R(f1_score(y, pred))}


def _ensemble_breach(risk, X):
    mu = risk.time_model.predict(X)
    return risk._breach_members(X, X["sla_hours"].to_numpy(float), mu)["breach_prob"]


def permutation_importance_auc(risk, X, y, repeats=3, seed=0):
    """AUC drop when one feature column is shuffled (ensemble breach model)."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, _ensemble_breach(risk, X))
    out = []
    for col in X.columns:
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base - roc_auc_score(y, _ensemble_breach(risk, Xp)))
        out.append((col, float(np.mean(drops))))
    return sorted(out, key=lambda t: -t[1])


def evaluate_models(risk, intake, train, val, test, fast: bool = False) -> dict:
    out = {"splits": {"train": len(train), "val": len(val), "test": len(test),
                      "breach_rate_test": R(test["breached"].mean()),
                      "escalation_rate_test": R(test["escalated"].mean())}}
    Xtr, Xte = model_matrix(train), model_matrix(test)
    pred = risk.predict(test)

    # --- 1. SLA breach
    y = test["breached"].to_numpy()
    rule_score = test["route"].map(ROUTE_RANK) + test["severity_score"] / 10
    rule_pred = ((test["severity_label"].isin(["Critical", "High"])) & (test["route"] != "auto_resolve")).astype(int)
    thr = risk.breach_threshold
    top20 = np.argsort(-pred["breach_prob"].to_numpy())[: int(0.2 * len(test))]
    out["breach"] = {
        "ensemble_used_in_pipeline": _cls_block(y, pred["breach_prob"].to_numpy(), thr),
        "gradient_boosting": _cls_block(y, pred["p_gb"].to_numpy(), 0.5),
        "logistic_regression": _cls_block(y, pred["p_lr"].to_numpy(), 0.5),
        "distributional_lognormal": _cls_block(y, pred["p_dist"].to_numpy(), 0.5),
        "static_rule_baseline": {"roc_auc": R(roc_auc_score(y, rule_score)),
                                 "precision": R(precision_score(y, rule_pred, zero_division=0)),
                                 "recall": R(recall_score(y, rule_pred)), "f1": R(f1_score(y, rule_pred))},
        "threshold_from_validation": R(thr),
        "recall_in_top20pct_flagged": R(y[top20].sum() / max(1, y.sum())),
        "lift_top20pct": R(y[top20].mean() / y.mean()),
    }
    # calibration deciles
    bins = pd.qcut(pred["breach_prob"], 10, duplicates="drop")
    cal = pd.DataFrame({"p": pred["breach_prob"], "y": y}).groupby(bins, observed=True).mean()
    out["breach"]["calibration_deciles"] = [{"mean_pred": R(a), "observed": R(b)} for a, b in zip(cal["p"], cal["y"])]

    # --- 2. Resolution time
    grp = train.groupby(["route", "severity_label"])["resolution_hours"].median()
    base_med = test.apply(lambda r: grp.get((r["route"], r["severity_label"]), train["resolution_hours"].median()), axis=1)
    out["resolution_time"] = {
        "gradient_boosting": {"mae_hours": R(mean_absolute_error(test["resolution_hours"], pred["median_hours"]), 2),
                              "median_abs_err_hours": R(median_absolute_error(test["resolution_hours"], pred["median_hours"]), 2),
                              "r2_log_scale": R(r2_score(np.log(test["resolution_hours"]), pred["log_mu"]))},
        "route_x_severity_median_baseline": {"mae_hours": R(mean_absolute_error(test["resolution_hours"], base_med), 2),
                                             "r2_log_scale": R(r2_score(np.log(test["resolution_hours"]), np.log(base_med)))},
        "p90_coverage": R((test["resolution_hours"] <= pred["p90_hours"]).mean()),
        "residual_sigma_log": R(risk.sigma),
    }

    # --- 3. Escalation
    ye = test["escalated"].to_numpy()
    lre = _lr().fit(Xtr, train["escalated"])
    out["escalation"] = {"gradient_boosting": _cls_block(ye, pred["escalation_prob"].to_numpy(), 0.3),
                         "logistic_regression": _cls_block(ye, lre.predict_proba(Xte)[:, 1], 0.3)}

    # --- 4. Intake
    res = intake.predict_batch(list(test["narrative"]))
    kw = ClassifierAgent()
    kres = [kw.classify(t) for t in test["narrative"]]
    pa = np.mean([r["product"] == p for r, p in zip(res, test["product"])])
    ia = np.mean([(r["product"], r["issue"]) == (p, i) for r, p, i in zip(res, test["product"], test["issue"])])
    kpa = np.mean([r["product"] == p for r, p in zip(kres, test["product"])])
    kia = np.mean([(r["product"], r["issue_subtype"]) == (p, i) for r, p, i in zip(kres, test["product"], test["issue"])])
    conf = np.array([r["confidence"] for r in res])
    ok = np.array([(r["product"], r["issue"]) == (p, i) for r, p, i in zip(res, test["product"], test["issue"])])
    out["intake"] = {"tfidf_logreg": {"product_acc": R(pa), "issue_acc": R(ia)},
                     "keyword_baseline_hackathon": {"product_acc": R(kpa), "issue_acc": R(kia)},
                     "accuracy_when_confidence_ge_0.4": R(ok[conf >= 0.4].mean()) if (conf >= 0.4).any() else None,
                     "share_routed_to_human_review": R((conf < 0.4).mean())}

    # --- 5. Permutation importance (breach model)
    if not fast:
        sub = Xte.sample(min(1200, len(Xte)), random_state=0)
        ys = test.loc[sub.index, "breached"]
        imp = permutation_importance_auc(risk, sub, ys)[:10]
        out["breach_feature_importance_auc_drop"] = [{"feature": f, "auc_drop": R(v)} for f, v in imp]
    return out
