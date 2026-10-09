"""Counterfactual policy simulation on the held-out period.

Every policy is applied to the SAME complaints with the SAME latent noise (common random
numbers), and outcomes are produced by the structural world model in data/synthetic.py.
The predictive policy runs the *real* multi-agent pipeline (incl. ML intake errors).

IMPORTANT: results are offline, on synthetic data. They show the framework works as designed;
they are not a claim about real-bank performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cdi import config, taxonomy
from cdi.agents.base import Context
from cdi.data.synthetic import resolution_hours
from cdi.orchestrator.pipeline import Orchestrator
from cdi.orchestrator.state import ComplaintInput
from cdi.store import Store


def _world(base: pd.DataFrame, fast, reroute, prim_load, backup_load, can_reroute):
    rr = np.asarray(reroute, bool) & can_reroute
    load = np.where(rr, backup_load, prim_load)
    hours = resolution_hours(base, fast_tracked=np.asarray(fast, float), rerouted=rr.astype(float), team_load=load)
    breach = hours > base["sla_hours"].to_numpy()
    ft_cost = config.FAST_TRACK_BASE_COST * np.maximum(0.5, load)
    cost = np.where(np.asarray(fast, bool), ft_cost, 0.0) + np.where(rr, config.REROUTE_COST, 0.0)
    pen = base["severity_label"].map(config.BREACH_PENALTY).to_numpy() * base["tier"].map(config.TIER_MULTIPLIER).to_numpy()
    return {"hours": hours, "breach": breach, "cost": cost, "penalty": breach * pen, "fast": np.asarray(fast, bool),
            "reroute": rr}


def _summ(w, base):
    hi = base["severity_label"].isin(["Critical", "High"]).to_numpy()
    net = w["penalty"] + w["cost"]
    return {"breaches": int(w["breach"].sum()), "breach_rate": round(float(w["breach"].mean()), 4),
            "breach_rate_critical_high": round(float(w["breach"][hi].mean()), 4),
            "mean_resolution_hours": round(float(w["hours"].mean()), 2),
            "p90_resolution_hours": round(float(np.quantile(w["hours"], 0.9)), 2),
            "fast_tracks": int(w["fast"].sum()), "reroutes": int(w["reroute"].sum()),
            "intervention_cost": round(float(w["cost"].sum()), 1),
            "breach_penalty": round(float(w["penalty"].sum()), 1),
            "net_loss": round(float(net.sum()), 1)}


def _boot(a_net, b_net, a_br, b_br, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a_net), (n, len(a_net)))
    d_net = 1 - a_net[idx].sum(1) / b_net[idx].sum(1)          # relative net-loss reduction of A vs B
    d_br = b_br[idx].sum(1) - a_br[idx].sum(1)                  # breaches avoided
    q = lambda v: [round(float(np.quantile(v, .025)), 4), round(float(np.quantile(v, .975)), 4)]
    return {"net_loss_reduction": round(float(1 - a_net.sum() / b_net.sum()), 4), "ci95": q(d_net),
            "breaches_avoided": int(b_br.sum() - a_br.sum()), "breaches_avoided_ci95": q(d_br)}


def run_simulation(test: pd.DataFrame, risk, intake, load_model, max_n: int | None = None, verbose=True):
    test = test.sort_values("submitted_at").reset_index(drop=True)
    if max_n:
        test = test.head(max_n)
    specs = [taxonomy.lookup(p, i) for p, i in zip(test["product"], test["issue"])]
    prim = np.array([load_model.load(s.team, t) for s, t in zip(specs, test["submitted_at"])])
    bkp = np.array([load_model.load(s.backup_team, t) if s.backup_team else 0.0
                    for s, t in zip(specs, test["submitted_at"])])
    can_rr = np.array([s.route == "bank_escalation" and bool(s.backup_team) for s in specs])
    base = test.copy()
    base["team_load"], base["fast_tracked"], base["rerouted"] = prim, 0, 0
    n = len(base)
    zeros = np.zeros(n, bool)

    w_none = _world(base, zeros, zeros, prim, bkp, can_rr)
    static_fast = (base["severity_label"].isin(["Critical", "High"]) & (base["route"] != "auto_resolve")).to_numpy()
    w_rule = _world(base, static_fast, zeros, prim, bkp, can_rr)

    # ---- predictive: run the real pipeline
    ctx = Context(intake, risk, Store(":memory:"), load_model)
    orch = Orchestrator(ctx)
    fast, rr, states = np.zeros(n, bool), np.zeros(n, bool), []
    for k, r in enumerate(base.itertuples(index=False)):
        st = orch.process(ComplaintInput(text=r.narrative, customer_id=r.customer_id, channel=r.channel,
                                         tier=r.tier, submitted_at=pd.Timestamp(r.submitted_at).to_pydatetime(),
                                         complaint_id=r.complaint_id, repeat_count=int(r.repeat_count)))
        held = st.needs_human_review or st.status.startswith("Held")   # nothing is executed for these
        fast[k] = st.intervention.get("fast_track", False) and not held
        rr[k] = (st.intervention.get("reroute_to") is not None) and not held
        states.append(st)
        if verbose and (k + 1) % 300 == 0:
            print(f"  simulated {k + 1}/{n}")
    w_pred = _world(base, fast, rr, prim, bkp, can_rr)

    pol = {"no_intervention": _summ(w_none, base), "static_rule_fast_track_high_critical": _summ(w_rule, base),
           "predictive_multi_agent": _summ(w_pred, base)}
    nl = lambda w: w["penalty"] + w["cost"]
    comp = {"predictive_vs_no_intervention": _boot(nl(w_pred), nl(w_none), w_pred["breach"], w_none["breach"]),
            "predictive_vs_static_rule": _boot(nl(w_pred), nl(w_rule), w_pred["breach"], w_rule["breach"])}
    ver = np.mean([s.verification["passed"] for s in states])
    extra = {"n_complaints": n, "verification_pass_rate": round(float(ver), 4),
             "human_review_share": round(float(np.mean([s.needs_human_review for s in states])), 4),
             "mean_pipeline_ms": round(float(np.mean([sum(t["ms"] for t in s.trace) for s in states])), 1),
             "intake_issue_accuracy_in_pipeline": round(float(np.mean(
                 [(s.product, s.issue) == (p, i) for s, p, i in zip(states, base["product"], base["issue"])])), 4),
             "interventions_chosen": pd.Series([s.intervention["name"] for s in states]).value_counts().to_dict(),
             "escalation_levels": pd.Series([s.escalation_level for s in states]).value_counts().to_dict()}
    return {"policies": pol, "comparisons": comp, "pipeline": extra}, states
