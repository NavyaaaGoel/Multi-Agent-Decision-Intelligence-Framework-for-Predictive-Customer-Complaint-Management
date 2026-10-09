from __future__ import annotations

from datetime import timedelta

from cdi.agents.base import Agent


def risk_bucket(p: float) -> str:
    return "Low" if p < 0.2 else "Elevated" if p < 0.5 else "High" if p < 0.75 else "Severe"


class SLAPlannerAgent(Agent):
    """Turns the forecast into an SLA plan: deadline, slack, risk bucket, watch checkpoints."""
    name = "sla_planner"

    def run(self, state, ctx):
        b = state.baseline_risk
        state.deadline = state.submitted_at + timedelta(hours=state.sla_hours)
        state.expected_completion = state.submitted_at + timedelta(hours=b["median_hours"])
        state.risk_bucket = risk_bucket(b["breach_prob"])
        state.sla_status = "At Risk" if b["breach_prob"] >= 0.5 else "On Track"
        state.checkpoints = [(state.submitted_at + timedelta(hours=f * state.sla_hours)).isoformat()
                             for f in (0.5, 0.8)]
        slack = state.sla_hours - b["p90_hours"]
        return {"deadline": state.deadline.isoformat(), "sla_hours": state.sla_hours,
                "risk_bucket": state.risk_bucket, "sla_status": state.sla_status,
                "slack_hours_vs_p90": round(slack, 1)}
