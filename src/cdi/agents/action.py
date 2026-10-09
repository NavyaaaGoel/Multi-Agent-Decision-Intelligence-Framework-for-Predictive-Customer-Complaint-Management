from __future__ import annotations

from cdi.agents.base import Agent

STATUS_BY_ROUTE = {"auto_resolve": "Resolved", "bank_escalation": "Escalated to Bank",
                   "partner_escalation": "Forwarded to Partner"}


class ActionAgent(Agent):
    """Executes the taxonomy actions + decision-agent interventions; writes the audit trail."""
    name = "action"

    def run(self, state, ctx):
        # Safety rule: a low-confidence classification must never trigger real back-office actions.
        actions = (["route_to_human_triage"] if state.needs_human_review
                   else list(state.base_actions) + list(state.extra_actions))
        if ctx.kill_switch:
            state.status = "Held (kill switch)"
            ctx.store.log_action(state.complaint_id, state.submitted_at, self.name, "kill_switch_hold",
                                 {"skipped": actions})
            return {"executed": 0, "skipped": len(actions), "reason": "kill switch active"}
        state.executed_actions = []
        for a in actions:
            res = ctx.executor.execute(a, state)
            state.executed_actions.append(res)
            ctx.store.log_action(state.complaint_id, state.submitted_at, self.name, a, res)
        state.status = ("Pending Human Review" if state.needs_human_review
                        else STATUS_BY_ROUTE[state.route])
        ctx.store.log_action(state.complaint_id, state.submitted_at, self.name, "status_set",
                             {"status": state.status})
        return {"executed": len(actions), "status": state.status}
