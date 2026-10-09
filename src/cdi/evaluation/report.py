"""Figures + auto-generated RESULTS.md from the metrics / simulation JSON."""
from __future__ import annotations

import json
from pathlib import Path


def plots(risk, test, metrics, sim, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    out.mkdir(parents=True, exist_ok=True)
    files = []
    cal = metrics["breach"]["calibration_deciles"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot([c["mean_pred"] for c in cal], [c["observed"] for c in cal], "o-", label="breach model")
    ax.set_xlabel("predicted P(breach)"); ax.set_ylabel("observed breach rate"); ax.set_title("Calibration (test)")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=130); plt.close(fig)
    files.append("calibration.png")

    pol = sim["policies"]
    names = ["no_intervention", "static_rule_fast_track_high_critical", "predictive_multi_agent"]
    labels = ["No intervention", "Static rule\n(fast-track High/Crit)", "Predictive\nmulti-agent"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, key, title in zip(axs, ["breach_rate", "net_loss", "intervention_cost"],
                              ["SLA breach rate", "Net loss (penalty + cost)", "Intervention cost"]):
        vals = [pol[n][key] for n in names]
        ax.bar(labels, vals, color=["#9aa5b1", "#d9a441", "#2f7d6d"])
        ax.set_title(title); ax.tick_params(axis="x", labelsize=8)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "policy_comparison.png", dpi=130); plt.close(fig)
    files.append("policy_comparison.png")
    return files


def results_md(metrics: dict, sim: dict | None) -> str:
    b, t, e, i = metrics["breach"], metrics["resolution_time"], metrics["escalation"], metrics["intake"]
    L = ["# Results (auto-generated)", "",
         "> Synthetic data, chronological split (train 70% / val 15% / test 15%). "
         "See README for what this does and does not prove.", "",
         f"Test set: {metrics['splits']['test']} complaints, breach rate {metrics['splits']['breach_rate_test']:.1%}.", "",
         "## 1. SLA breach prediction", "", "| Model | ROC-AUC | PR-AUC | Brier | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|"]
    for k, lab in (("ensemble_used_in_pipeline", "**Ensemble (used in pipeline)**"), ("gradient_boosting", "Gradient boosting (calibrated)"),
                   ("logistic_regression", "Logistic regression"), ("distributional_lognormal", "Distributional (log-normal time model)")):
        m = b[k]; L.append(f"| {lab} | {m['roc_auc']} | {m['pr_auc']} | {m['brier']} | {m['precision']} | {m['recall']} | {m['f1']} |")
    m = b["static_rule_baseline"]
    L.append(f"| Static rule (High/Critical & escalated) | {m['roc_auc']} | - | - | {m['precision']} | {m['recall']} | {m['f1']} |")
    L += ["", f"Flagging the top 20% riskiest complaints captures **{b['recall_in_top20pct_flagged']:.0%}** of all breaches "
          f"(lift {b['lift_top20pct']}x).", "",
          "## 2. Resolution-time forecast", "", "| Model | MAE (h) | R2 (log) |", "|---|---|---|",
          f"| Gradient boosting | {t['gradient_boosting']['mae_hours']} | {t['gradient_boosting']['r2_log_scale']} |",
          f"| Route x severity median baseline | {t['route_x_severity_median_baseline']['mae_hours']} | {t['route_x_severity_median_baseline']['r2_log_scale']} |",
          "", f"P90 interval coverage: **{t['p90_coverage']:.1%}** (nominal 90%).", "",
          "## 3. Customer-escalation risk", "", "| Model | ROC-AUC | PR-AUC | Brier |", "|---|---|---|---|"]
    for k in ("gradient_boosting", "logistic_regression"):
        L.append(f"| {k} | {e[k]['roc_auc']} | {e[k]['pr_auc']} | {e[k]['brier']} |")
    L += ["", "## 4. Intake classification", "", "| Model | Product acc. | Issue acc. |", "|---|---|---|",
          f"| TF-IDF + LogReg (new) | {i['tfidf_logreg']['product_acc']} | {i['tfidf_logreg']['issue_acc']} |",
          f"| Keyword matcher (hackathon) | {i['keyword_baseline_hackathon']['product_acc']} | {i['keyword_baseline_hackathon']['issue_acc']} |",
          "", f"Share routed to human review (confidence < 0.4): {i['share_routed_to_human_review']:.1%}.", ""]
    if "breach_feature_importance_auc_drop" in metrics:
        L += ["## 5. Top drivers of breach risk (permutation importance)", "", "| Feature | AUC drop |", "|---|---|"]
        L += [f"| {x['feature']} | {x['auc_drop']} |" for x in metrics["breach_feature_importance_auc_drop"]] + [""]
    if sim:
        p, c, x = sim["policies"], sim["comparisons"], sim["pipeline"]
        L += ["## 6. Policy simulation (held-out period, common random numbers)", "",
              "| Policy | Breach rate | Crit/High breach rate | Mean hrs | Fast-tracks | Reroutes | Intervention cost | Net loss |",
              "|---|---|---|---|---|---|---|---|"]
        for k, v in p.items():
            L.append(f"| {k} | {v['breach_rate']:.1%} | {v['breach_rate_critical_high']:.1%} | {v['mean_resolution_hours']} | "
                     f"{v['fast_tracks']} | {v['reroutes']} | {v['intervention_cost']} | {v['net_loss']} |")
        for k, v in c.items():
            L.append(f"\n**{k}**: net-loss reduction {v['net_loss_reduction']:.1%} (95% CI {v['ci95'][0]:.1%} to {v['ci95'][1]:.1%}); "
                     f"breaches avoided {v['breaches_avoided']} (95% CI {v['breaches_avoided_ci95'][0]:.0f} to {v['breaches_avoided_ci95'][1]:.0f}).")
        L += ["", f"Pipeline: {x['n_complaints']} complaints, verification pass rate {x['verification_pass_rate']:.1%}, "
              f"human-review share {x['human_review_share']:.1%}, mean latency {x['mean_pipeline_ms']} ms, "
              f"end-to-end intake issue accuracy {x['intake_issue_accuracy_in_pipeline']:.1%}.",
              f"Interventions chosen: {x['interventions_chosen']}. Escalation levels: {x['escalation_levels']}.", ""]
    return "\n".join(L)
