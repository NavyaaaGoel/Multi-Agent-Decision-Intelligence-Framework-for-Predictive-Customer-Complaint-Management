"""Orchestrator: runs the agent graph on a shared state, persists, verifies, re-plans."""
from __future__ import annotations

import itertools
from datetime import datetime

from cdi.agents.action import ActionAgent
from cdi.agents.base import Context
from cdi.agents.decision import DecisionAgent
from cdi.agents.intake import IntakeAgent
from cdi.agents.risk import RiskPredictorAgent
from cdi.agents.router import ResourceRouterAgent
from cdi.agents.sla import SLAPlannerAgent
from cdi.agents.verifier import VerificationAgent
from cdi.orchestrator.state import ComplaintInput, ComplaintState

_counter = itertools.count(1)


class Orchestrator:
    def __init__(self, ctx: Context, max_replans: int = 1):
        self.ctx, self.max_replans = ctx, max_replans
        self.pipeline = [IntakeAgent(), RiskPredictorAgent(), SLAPlannerAgent(), DecisionAgent(),
                         ResourceRouterAgent(), ActionAgent()]
        self.verifier = VerificationAgent()

    def process(self, inp: ComplaintInput) -> ComplaintState:
        ctx = self.ctx
        cid = inp.complaint_id or f"CMP-{inp.submitted_at:%Y%m%d}-{next(_counter):05d}"
        st = ComplaintState(complaint_id=cid, customer_id=inp.customer_id, text=inp.text,
                            channel=inp.channel, tier=inp.tier, submitted_at=inp.submitted_at)
        st.repeat_count = (inp.repeat_count if inp.repeat_count is not None
                           else ctx.store.recent_count(inp.customer_id, inp.submitted_at))
        ctx.store.log_action(cid, inp.submitted_at, "orchestrator", "received", {"channel": inp.channel})
        try:
            for agent in self.pipeline:
                agent(st, ctx)
                if agent.name == "intake" and inp.amount is not None:   # channel-supplied amount wins
                    st.amount = float(inp.amount)
            self._verify_loop(st)
        except Exception as exc:  # any hard failure -> safe human hand-off, never a silent drop
            st.errors.append(f"pipeline aborted: {exc!r}")
            st.status, st.needs_human_review = "Pending Human Review", True
            st.verification = st.verification or {"passed": False, "failed": ["pipeline_exception"], "checks": []}
            ctx.store.log_action(cid, inp.submitted_at, "orchestrator", "pipeline_failure", {"error": repr(exc)})
            st.team = st.team or "Branch & Service Ops"
            st.assigned_team = st.assigned_team or st.team
        d = st.to_dict()
        ctx.store.upsert(d)
        return st

    def _verify_loop(self, st: ComplaintState):
        ctx = self.ctx
        self.verifier(st, ctx)
        while not st.verification["passed"] and st.replans < self.max_replans:
            st.replans += 1
            ctx.store.log_action(st.complaint_id, st.submitted_at, "orchestrator", "replan",
                                 {"failed": st.verification["failed"]})
            # recovery: re-run decision with human-in-the-loop flag, then re-verify
            st.needs_human_review = True
            st.executed_actions, st.status = [], "Received"
            for agent in (DecisionAgent(), ResourceRouterAgent(), ActionAgent()):
                agent(st, ctx)
            self.verifier(st, ctx)
