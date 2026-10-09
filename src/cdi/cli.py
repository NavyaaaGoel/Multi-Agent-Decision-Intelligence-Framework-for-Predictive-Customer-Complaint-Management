"""Command line: python -m cdi {generate|train|simulate|demo|all|serve}"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta

import pandas as pd

from cdi import config
from cdi.data import synthetic


def _dirs():
    for d in (config.DATA_DIR, config.MODEL_DIR, config.REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def cmd_generate(a):
    _dirs()
    t = time.time()
    df = synthetic.generate(a.n, a.seed)
    df.to_csv(config.DATA_DIR / "complaints_synthetic.csv", index=False)
    print(f"generated {len(df)} complaints in {time.time() - t:.1f}s | breach rate {df.breached.mean():.1%} | "
          f"escalation rate {df.escalated.mean():.1%} -> data/complaints_synthetic.csv")


def _load_data():
    df = pd.read_csv(config.DATA_DIR / "complaints_synthetic.csv", parse_dates=["submitted_at"])
    df["backup_team"] = df["backup_team"].fillna("")
    return synthetic.chronological_split(df)


def cmd_train(a):
    from cdi.evaluation import metrics as M
    from cdi.evaluation.report import results_md
    from cdi.ml.intake_model import IntakeClassifier
    from cdi.ml.risk_models import RiskModels
    _dirs()
    tr, va, te = _load_data()
    t = time.time()
    risk = RiskModels().fit(tr, va)
    risk.save(config.MODEL_DIR / "risk_models.joblib")
    ic = IntakeClassifier().fit(tr["narrative"], tr["product"], tr["product"] + "::" + tr["issue"])
    ic.save(config.MODEL_DIR / "intake_model.joblib")
    print(f"trained in {time.time() - t:.1f}s; evaluating on {len(te)} held-out complaints ...")
    m = M.evaluate_models(risk, ic, tr, va, te)
    (config.REPORT_DIR / "metrics.json").write_text(json.dumps(m, indent=2))
    (config.REPORT_DIR / "RESULTS.md").write_text(results_md(m, None))
    b = m["breach"]["ensemble_used_in_pipeline"]
    print(f"breach AUC {b['roc_auc']} (GB {m['breach']['gradient_boosting']['roc_auc']}, LR {m['breach']['logistic_regression']['roc_auc']}, dist {m['breach']['distributional_lognormal']['roc_auc']}, rule {m['breach']['static_rule_baseline']['roc_auc']}) | "
          f"time MAE {m['resolution_time']['gradient_boosting']['mae_hours']}h | P90 coverage {m['resolution_time']['p90_coverage']} | "
          f"intake issue acc {m['intake']['tfidf_logreg']['issue_acc']} (keyword {m['intake']['keyword_baseline_hackathon']['issue_acc']})")


def cmd_simulate(a):
    from cdi.evaluation.report import plots, results_md
    from cdi.evaluation.simulate import run_simulation
    from cdi.ml.intake_model import IntakeClassifier
    from cdi.ml.risk_models import RiskModels
    tr, va, te = _load_data()
    risk = RiskModels.load(config.MODEL_DIR / "risk_models.joblib")
    ic = IntakeClassifier.load(config.MODEL_DIR / "intake_model.joblib")
    full = pd.read_csv(config.DATA_DIR / "complaints_synthetic.csv", parse_dates=["submitted_at"])
    load_model = synthetic.TeamLoadModel(datetime(2026, 1, 1), 160, config.SEED + 1)
    print(f"simulating {len(te) if not a.max_n else a.max_n} held-out complaints through the agent pipeline ...")
    sim, states = run_simulation(te, risk, ic, load_model, a.max_n)
    m = json.loads((config.REPORT_DIR / "metrics.json").read_text())
    (config.REPORT_DIR / "simulation.json").write_text(json.dumps(sim, indent=2))
    (config.REPORT_DIR / "RESULTS.md").write_text(results_md(m, sim))
    # sample traces: biggest wins, a human-review case, an auto-resolve case
    by_gain = sorted([s for s in states if not s.needs_human_review], key=lambda s: s.intervention["loss_before"] - s.intervention["loss_after"], reverse=True)
    pick = by_gain[:3] + [s for s in states if s.needs_human_review][:1] + [s for s in states if s.route == "auto_resolve"][:1]
    (config.REPORT_DIR / "sample_traces.json").write_text(json.dumps([s.to_dict() for s in pick], indent=2))
    figs = plots(risk, te, m, sim, config.REPORT_DIR / "figures")
    p = sim["policies"]
    for k, v in p.items():
        print(f"{k:42s} breach {v['breach_rate']:.1%}  net_loss {v['net_loss']:>8}  fast {v['fast_tracks']:>4} reroute {v['reroutes']:>3}")
    print(json.dumps(sim["comparisons"], indent=1))
    print("figures:", figs)


def cmd_demo(a):
    from cdi.agents.base import Context
    from cdi.ml.intake_model import IntakeClassifier
    from cdi.ml.risk_models import RiskModels
    from cdi.agents.monitor import SLAMonitorAgent
    from cdi.orchestrator.pipeline import Orchestrator
    from cdi.orchestrator.state import ComplaintInput
    from cdi.store import Store
    ic = IntakeClassifier.load(config.MODEL_DIR / "intake_model.joblib")
    risk = RiskModels.load(config.MODEL_DIR / "risk_models.joblib")
    store = Store(":memory:")
    # busy-day load profile from the simulated load model (stands in for the live queue)
    ctx = Context(ic, risk, store, synthetic.TeamLoadModel(datetime(2026, 1, 1), 160, config.SEED + 1))
    orch, t0 = Orchestrator(ctx), datetime(2026, 5, 20, 11, 0)
    cases = [
        ("Rs. 45,000 was debited from my UPI account but the beneficiary was not credited. This is unacceptable, I will approach the RBI ombudsman.", "Priority"),
        ("Unauthorized transaction of ₹82,000 on my debit card. I am extremely frustrated, please help.", "Premium"),
        ("I lost my credit card, please block it.", "Standard"),
        ("My home loan EMI paid but showing due and the recovery agent keeps calling me. Rs. 18,500", "Standard"),
        ("Not happy with everything, nobody helps.", "Standard"),
    ]
    for k, (txt, tier) in enumerate(cases):
        st = orch.process(ComplaintInput(txt, customer_id=f"DEMO{k}", tier=tier, submitted_at=t0))
        print("=" * 100)
        print(f"[{st.complaint_id}] {txt}")
        for ev in st.trace:
            print(f"  - {ev['agent']:<15} {ev['ms']:>7.1f} ms  {json.dumps(ev['output'], default=str)[:230]}")
        v = st.verification
        print(f"  => status={st.status} | team={st.assigned_team} | level={st.escalation_level} | "
              f"P(breach) {st.baseline_risk['breach_prob']:.0%} -> {st.final_risk['breach_prob']:.0%} | verified={v['passed']}")
    print("\n" + "=" * 100 + "\nSLA monitor sweeps (simulated clock):")
    mon = SLAMonitorAgent()
    for h in (12, 20, 30):
        evs = mon.scan(ctx, t0 + timedelta(hours=h))
        print(f"  t+{h:>2}h: {len(evs)} status change(s)")
        for e in evs:
            print("     ", json.dumps(e))
    print(f"\naudit rows written: {sum(store.audit_count(s['complaint_id']) for s in store.list())}")


def cmd_serve(a):
    import uvicorn
    uvicorn.run("cdi.api.app:app", host="0.0.0.0", port=a.port)


def cmd_all(a):
    a.n, a.seed, a.max_n = 9000, config.SEED, None
    cmd_generate(a); cmd_train(a); cmd_simulate(a)


def main():
    ap = argparse.ArgumentParser(prog="cdi")
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("generate"); g.add_argument("--n", type=int, default=9000); g.add_argument("--seed", type=int, default=config.SEED); g.set_defaults(f=cmd_generate)
    sp.add_parser("train").set_defaults(f=cmd_train)
    s = sp.add_parser("simulate"); s.add_argument("--max-n", type=int, default=None); s.set_defaults(f=cmd_simulate)
    sp.add_parser("demo").set_defaults(f=cmd_demo)
    sv = sp.add_parser("serve"); sv.add_argument("--port", type=int, default=8000); sv.set_defaults(f=cmd_serve)
    sp.add_parser("all").set_defaults(f=cmd_all)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
