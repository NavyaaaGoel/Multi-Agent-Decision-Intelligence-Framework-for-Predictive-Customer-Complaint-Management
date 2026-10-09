"""Shared blackboard passed between agents."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np


@dataclass
class ComplaintInput:
    text: str
    customer_id: str = "ANON"
    channel: str = "Web"
    tier: str = "Standard"
    submitted_at: datetime = field(default_factory=datetime.now)
    complaint_id: Optional[str] = None
    amount: Optional[float] = None          # if the channel already captured it
    repeat_count: Optional[int] = None      # if known; else derived from the store


@dataclass
class ComplaintState:
    complaint_id: str
    customer_id: str
    text: str
    channel: str
    tier: str
    submitted_at: datetime
    # intake
    product: str = ""
    issue: str = ""
    intake_confidence: float = 0.0
    top_issues: list = field(default_factory=list)
    route: str = ""
    base_actions: list = field(default_factory=list)
    partner: str = ""
    severity_score: float = 0.0
    severity_label: str = ""
    sentiment: float = 0.0
    threat_count: int = 0
    amount: float = 0.0
    repeat_count: int = 0
    needs_human_review: bool = False
    # taxonomy / team
    team: str = ""
    backup_team: str = ""
    team_load: float = 0.0
    # prediction + SLA
    baseline_risk: dict = field(default_factory=dict)
    sla_hours: int = 0
    deadline: Optional[datetime] = None
    sla_status: str = ""
    risk_bucket: str = ""
    checkpoints: list = field(default_factory=list)
    expected_completion: Optional[datetime] = None
    # decision
    candidates: list = field(default_factory=list)
    intervention: dict = field(default_factory=dict)
    final_risk: dict = field(default_factory=dict)
    priority_score: float = 0.0
    escalation_level: str = "None"
    extra_actions: list = field(default_factory=list)
    # routing / action / verification
    assigned_team: str = ""
    executed_actions: list = field(default_factory=list)
    status: str = "Received"
    verification: dict = field(default_factory=dict)
    replans: int = 0
    degraded: bool = False
    errors: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    monitor_events: list = field(default_factory=list)
    live_breach_prob: Optional[float] = None

    def to_dict(self) -> dict:
        return jsonable(dataclasses.asdict(self))


def jsonable(o: Any):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, np.generic):
        return o.item()
    return o
