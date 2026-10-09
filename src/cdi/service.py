"""Framework-agnostic service layer (the FastAPI app is a thin wrapper around this)."""
from __future__ import annotations

import json
from datetime import datetime

from cdi import config
from cdi.agents.base import Context, LiveLoadProvider
from cdi.agents.monitor import SLAMonitorAgent
from cdi.ml.intake_model import IntakeClassifier
from cdi.ml.risk_models import RiskModels
from cdi.orchestrator.pipeline import Orchestrator
from cdi.orchestrator.state import ComplaintInput
from cdi.store import Store


class ComplaintService:
    def __init__(self, intake, risk, store=None, load_provider=None):
        self.store = store or Store(":memory:")
        self.ctx = Context(intake, risk, self.store, load_provider or LiveLoadProvider(self.store))
        self.orch = Orchestrator(self.ctx)
        self.monitor = SLAMonitorAgent()

    @classmethod
    def from_artifacts(cls, db_path: str = ":memory:"):
        intake = IntakeClassifier.load(config.MODEL_DIR / "intake_model.joblib")
        risk = RiskModels.load(config.MODEL_DIR / "risk_models.joblib")
        return cls(intake, risk, Store(db_path))

    def submit(self, payload: dict) -> dict:
        ts = payload.get("submitted_at")
        inp = ComplaintInput(text=payload["text"], customer_id=payload.get("customer_id", "ANON"),
                             channel=payload.get("channel", "Web"), tier=payload.get("tier", "Standard"),
                             submitted_at=datetime.fromisoformat(ts) if ts else datetime.now(),
                             amount=payload.get("amount"))
        if inp.channel not in config.CHANNELS or inp.tier not in config.TIERS:
            raise ValueError(f"channel must be in {config.CHANNELS} and tier in {config.TIERS}")
        if not inp.text or not inp.text.strip():
            raise ValueError("text is required")
        return self.orch.process(inp).to_dict()

    def get(self, complaint_id: str):
        return self.store.get(complaint_id)

    def trace(self, complaint_id: str):
        d = self.store.get(complaint_id)
        return None if d is None else {"complaint_id": complaint_id, "trace": d["trace"],
                                       "verification": d["verification"], "audit": self.store.audit(complaint_id)}

    def scan_sla(self, now: str | None = None) -> list:
        return self.monitor.scan(self.ctx, datetime.fromisoformat(now) if now else datetime.now())

    def team_load(self) -> dict:
        return {t: round(self.ctx.load.load(t), 3) for t in config.ALL_TEAMS}

    def metrics(self):
        p = config.REPORT_DIR / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else None
