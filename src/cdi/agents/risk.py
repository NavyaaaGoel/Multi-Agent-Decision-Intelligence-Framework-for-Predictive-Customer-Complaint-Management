from __future__ import annotations

import pandas as pd

from cdi.agents.base import Agent, feature_row
from cdi.data.synthetic import ROUTE_MEDIAN, SEV_SPEED


def rule_based_risk(state) -> dict:
    """Degraded-mode fallback if the ML models are unavailable."""
    p = {"Critical": 0.45, "High": 0.30, "Medium": 0.20, "Low": 0.08}[state.severity_label]
    p = min(0.95, p * (1.6 if state.route == "partner_escalation" else 1.0)) if state.route != "auto_resolve" else 0.01
    med = ROUTE_MEDIAN[state.route] * SEV_SPEED[state.severity_label]
    return {"breach_prob": p, "escalation_prob": 0.15, "median_hours": med, "p90_hours": med * 1.8,
            "log_mu": 0.0, "source": "rule_based_fallback"}


class RiskPredictorAgent(Agent):
    """ML forecasts: P(SLA breach), P(customer escalation), median & P90 resolution time."""
    name = "risk_predictor"

    def run(self, state, ctx):
        state.team_load = round(ctx.load.load(state.team, state.submitted_at), 4)
        try:
            row = pd.DataFrame([feature_row(state)])
            r = ctx.risk.predict(row).iloc[0].to_dict()
            r["source"] = "ml_models"
        except Exception as exc:  # graceful degradation
            state.degraded = True
            state.errors.append(f"risk_predictor fell back to rules: {exc!r}")
            r = rule_based_risk(state)
        state.baseline_risk = {k: (round(float(v), 4) if k != "source" else v) for k, v in r.items()}
        return {"team_load": state.team_load, **{k: state.baseline_risk[k] for k in
                ("breach_prob", "escalation_prob", "median_hours", "p90_hours", "source")}}
