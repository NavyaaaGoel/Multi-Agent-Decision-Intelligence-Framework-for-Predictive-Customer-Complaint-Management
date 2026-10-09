"""Shared, cached small-scale world so the whole suite trains models only once (~15 s)."""
from datetime import datetime

from cdi.agents.base import Context
from cdi.data import synthetic
from cdi.ml.intake_model import IntakeClassifier
from cdi.ml.risk_models import RiskModels
from cdi.orchestrator.pipeline import Orchestrator
from cdi.store import Store

_W = {}


def world():
    if not _W:
        df = synthetic.generate(3500, seed=11, days=60)
        tr, va, te = synthetic.chronological_split(df)
        risk = RiskModels().fit(tr, va)
        intake = IntakeClassifier().fit(tr["narrative"], tr["product"], tr["product"] + "::" + tr["issue"])
        _W.update(df=df, tr=tr, va=va, te=te, risk=risk, intake=intake,
                  load=synthetic.TeamLoadModel(datetime(2026, 1, 1), 80, 12))
    return _W


def make_orchestrator(**ctx_kw):
    w = world()
    ctx = Context(w["intake"], w["risk"], Store(":memory:"), w["load"], **ctx_kw)
    return Orchestrator(ctx), ctx


UPI_TEXT = ("Rs. 45,000 was debited from my UPI account but the beneficiary was not credited. "
            "This is unacceptable, I will approach the RBI ombudsman.")
