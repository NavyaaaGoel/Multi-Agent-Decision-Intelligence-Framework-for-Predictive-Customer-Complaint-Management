from __future__ import annotations

from cdi import config
from cdi.agents.base import Agent
from cdi.agents.decision import LEVELS
from cdi.agents.action import STATUS_BY_ROUTE


class VerificationAgent(Agent):
    """Independent invariant checks on the finished plan (also guards against model misbehaviour)."""
    name = "verification"

    def run(self, state, ctx):
        c = []
        add = lambda n, ok, d="": c.append({"check": n, "ok": bool(ok), "detail": d})
        fb, fa = state.baseline_risk, state.final_risk
        held = state.status.startswith("Held")
        add("route_valid", state.route in config.ROUTES, state.route)
        add("probabilities_in_range", all(0 <= x <= 1 for x in
            (fb["breach_prob"], fb["escalation_prob"], fa["breach_prob"], fa["escalation_prob"])))
        add("deadline_consistent", state.deadline is not None and state.deadline > state.submitted_at,
            str(state.deadline))
        add("team_assigned", bool(state.assigned_team), state.assigned_team)
        add("intervention_not_harmful", state.intervention["breach_prob_after"]
            <= state.intervention["breach_prob_before"] + 1e-9)
        add("escalation_level_valid", state.escalation_level in LEVELS, state.escalation_level)
        add("critical_risk_escalated", not (state.severity_label == "Critical" and fa["breach_prob"] >= 0.5
            and state.route != "auto_resolve" and LEVELS.index(state.escalation_level) < 2),
            "Critical + high breach risk needs Regional/MD escalation")
        if held:
            add("kill_switch_no_actions", not state.executed_actions)
        else:
            exp = STATUS_BY_ROUTE[state.route]
            add("status_matches_route", state.status == exp or (state.needs_human_review and
                state.status == "Pending Human Review"), state.status)
            expected_n = 1 if state.needs_human_review else len(state.base_actions) + len(state.extra_actions)
            add("all_actions_executed", len(state.executed_actions) == expected_n)
            add("no_auto_action_on_low_confidence", not state.needs_human_review or
                [a["action"] for a in state.executed_actions] == ["route_to_human_triage"])
            add("audit_trail_persisted", ctx.store.audit_count(state.complaint_id) >= len(state.executed_actions))
        failed = [x["check"] for x in c if not x["ok"]]
        state.verification = {"passed": not failed, "failed": failed, "checks": c}
        return {"passed": not failed, "failed": failed, "n_checks": len(c)}
