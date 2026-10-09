from __future__ import annotations

from cdi import config, nlp, severity, taxonomy
from cdi.agents.base import Agent


class IntakeAgent(Agent):
    """Text -> product/issue (ML), severity, sentiment, money at stake, repeat history."""
    name = "intake"

    def run(self, state, ctx):
        pred = ctx.intake.predict_one(state.text)
        spec = taxonomy.lookup(pred["product"], pred["issue"])
        state.product, state.issue = pred["product"], pred["issue"]
        state.intake_confidence = round(pred["confidence"], 4)
        state.top_issues = [(i, round(p, 3)) for i, p in pred["top_issues"]]
        state.route, state.base_actions = spec.route, list(spec.actions)
        state.team, state.backup_team, state.partner = spec.team, spec.backup_team, spec.partner
        state.amount = nlp.extract_amount(state.text)
        state.sentiment = nlp.sentiment_score(state.text)
        state.threat_count = nlp.threat_count(state.text)
        state.severity_score, state.severity_label = severity.compute(
            spec.base_severity, state.amount, state.sentiment)
        state.sla_hours = config.SLA_HOURS[state.severity_label]
        state.needs_human_review = state.intake_confidence < ctx.min_intake_confidence
        return {"product": state.product, "issue": state.issue, "confidence": state.intake_confidence,
                "route": state.route, "severity": f"{state.severity_label} ({state.severity_score})",
                "sentiment": state.sentiment, "amount": state.amount,
                "repeat_count": state.repeat_count, "needs_human_review": state.needs_human_review}
