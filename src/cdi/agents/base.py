"""Agent protocol, shared context, load providers and the action executor."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from cdi import config
from cdi.ml.features import BASE_COLS  # noqa: F401  (re-exported for agents)


class LiveLoadProvider:
    """Team utilisation from the store: open cases / capacity (matches the training scale)."""

    def __init__(self, store, capacity: int = config.TEAM_CAPACITY_CASES):
        self.store, self.capacity = store, capacity

    def load(self, team: str, ts=None) -> float:
        if team == config.AUTOMATION_TEAM:
            return 0.15
        return min(1.6, max(0.1, self.store.team_open_count(team) / self.capacity))


class ActionExecutor:
    """Simulated back-office connectors. Every call is audited; nothing leaves the sandbox."""

    DESCRIPTIONS = {
        "set_priority_queue_fast_track": "Moved case to the fast-track queue",
        "notify_customer_delay_risk": "Sent proactive delay-risk notice to customer",
        "route_to_human_triage": "Queued for human triage (low-confidence classification, no automated action)",
        "schedule_sla_watch": "Registered mid-point and 80% SLA checkpoints",
        "assign_relationship_manager_followup": "Alerted relationship manager for follow-up",
    }

    def execute(self, action: str, state) -> dict:
        desc = self.DESCRIPTIONS.get(action)
        if desc is None:
            if action.startswith("forward_to_"):
                desc = f"Forwarded case to {state.partner or 'partner'} (simulated API call)"
            elif action.startswith("reassign_to_team:"):
                desc = f"Reassigned case to {action.split(':', 1)[1]}"
            else:
                desc = f"Executed '{action}' (simulated back-office call)"
        return {"action": action, "status": "simulated_success", "description": desc}


@dataclass
class Context:
    intake: object                 # IntakeClassifier
    risk: object                   # RiskModels
    store: object                  # Store
    load: object                   # object with .load(team, ts)
    executor: ActionExecutor = field(default_factory=ActionExecutor)
    kill_switch: bool = False      # when True, no actions are executed (safety stop)
    min_intake_confidence: float = config.MIN_INTAKE_CONFIDENCE


class Agent:
    name = "agent"

    def run(self, state, ctx) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, state, ctx) -> dict:
        t0 = time.perf_counter()
        out = self.run(state, ctx) or {}
        state.trace.append({"agent": self.name, "ms": round((time.perf_counter() - t0) * 1000, 2),
                            "output": out})
        return out


def feature_row(state, *, team=None, team_load=None, fast=0, rerouted=0) -> dict:
    """One model-input row for the complaint under a hypothetical plan."""
    return dict(submitted_at=state.submitted_at, product=state.product, issue=state.issue,
                route=state.route, severity_score=state.severity_score, severity_label=state.severity_label,
                sla_hours=state.sla_hours, channel=state.channel, tier=state.tier,
                team=team or state.team, amount=state.amount, narrative=state.text,
                sentiment=state.sentiment, threat_count=state.threat_count,
                repeat_count=state.repeat_count,
                team_load=state.team_load if team_load is None else team_load,
                fast_tracked=int(fast), rerouted=int(rerouted))
