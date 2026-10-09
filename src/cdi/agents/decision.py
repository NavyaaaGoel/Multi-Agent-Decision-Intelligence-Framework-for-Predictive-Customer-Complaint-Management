"""Decision agent: risk-aware, cost-aware intervention selection via model counterfactuals.

For each feasible plan (standard / fast-track / reroute / both) the *trained models* are
re-queried with the plan's features changed. The plan minimising

    expected_loss = P(breach | plan) x severity_penalty x tier_multiplier + plan_cost

is chosen. This is explicit decision-theory on top of calibrated probabilities, so every
choice is explainable and tunable (change the penalties/costs in config.py).
"""
from __future__ import annotations

import pandas as pd

from cdi import config
from cdi.agents.base import Agent, feature_row

LEVELS = ["None", "Branch", "Regional Office", "MD/CEO Desk"]


def level_at_least(cur: str, floor: str) -> str:
    return LEVELS[max(LEVELS.index(cur), LEVELS.index(floor))]


class DecisionAgent(Agent):
    name = "decision"

    def _plans(self, state, ctx):
        ts, ld = state.submitted_at, state.team_load
        plans = [dict(name="standard", fast=0, reroute=0, team=state.team, load=ld, cost=0.0)]
        ft_cost = lambda load: config.FAST_TRACK_BASE_COST * max(0.5, load)
        if state.route != "auto_resolve":
            plans.append(dict(name="fast_track", fast=1, reroute=0, team=state.team, load=ld, cost=ft_cost(ld)))
        if state.route == "bank_escalation" and state.backup_team:
            bl = ctx.load.load(state.backup_team, ts)
            plans.append(dict(name="reroute", fast=0, reroute=1, team=state.backup_team, load=bl,
                              cost=config.REROUTE_COST))
            plans.append(dict(name="fast_track+reroute", fast=1, reroute=1, team=state.backup_team, load=bl,
                              cost=config.REROUTE_COST + ft_cost(bl)))
        return plans

    def run(self, state, ctx):
        plans = self._plans(state, ctx)
        pen = config.BREACH_PENALTY[state.severity_label] * config.TIER_MULTIPLIER.get(state.tier, 1.0)
        frame = pd.DataFrame([feature_row(state, team=p["team"], team_load=p["load"],
                                          fast=p["fast"], rerouted=p["reroute"]) for p in plans])
        pred = ctx.risk.predict(frame) if state.baseline_risk.get("source") == "ml_models" else None
        for i, p in enumerate(plans):
            if pred is not None:
                p.update(breach_prob=float(pred.iloc[i]["breach_prob"]), esc_prob=float(pred.iloc[i]["escalation_prob"]),
                         median_hours=float(pred.iloc[i]["median_hours"]), p90_hours=float(pred.iloc[i]["p90_hours"]),
                         log_mu=float(pred.iloc[i]["log_mu"]))
            else:  # degraded mode: only the baseline is known, interventions get a fixed heuristic discount
                b = state.baseline_risk
                disc = {"standard": 1.0, "fast_track": 0.65, "reroute": 0.9, "fast_track+reroute": 0.6}[p["name"]]
                p.update(breach_prob=b["breach_prob"] * disc, esc_prob=b["escalation_prob"],
                         median_hours=b["median_hours"] * disc, p90_hours=b["p90_hours"] * disc,
                         log_mu=b.get("log_mu", 0.0))
            p["loss"] = p["breach_prob"] * pen + p["cost"]
        best = min(plans, key=lambda p: (p["loss"], p["cost"]))
        base = plans[0]
        state.candidates = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()} for p in plans]
        state.intervention = {
            "name": best["name"], "fast_track": bool(best["fast"]),
            "reroute_to": best["team"] if best["reroute"] else None, "cost": round(best["cost"], 3),
            "breach_prob_before": round(base["breach_prob"], 4), "breach_prob_after": round(best["breach_prob"], 4),
            "loss_before": round(base["loss"], 3), "loss_after": round(best["loss"], 3),
            "rationale": (f"{best['name']}: P(breach) {base['breach_prob']:.0%} -> {best['breach_prob']:.0%}, "
                          f"expected loss {base['loss']:.2f} -> {best['loss']:.2f} (cost {best['cost']:.2f}, penalty {pen:.1f})"
                          if best["name"] != "standard" else
                          f"standard: no feasible intervention lowers expected loss ({base['loss']:.2f})")}
        state.final_risk = {k: round(best[k], 4) for k in
                            ("breach_prob", "esc_prob", "median_hours", "p90_hours", "log_mu")}
        state.final_risk["escalation_prob"] = state.final_risk.pop("esc_prob")

        # ---- priority score & escalation level
        fr = state.final_risk
        state.priority_score = round(100 * (
            0.30 * state.severity_score / 10 + 0.30 * fr["breach_prob"] + 0.15 * fr["escalation_prob"]
            + 0.10 * min(1.0, state.amount / 100_000)
            + 0.10 * (config.TIER_MULTIPLIER.get(state.tier, 1.0) - 1) / 0.6
            + 0.05 * max(0.0, -state.sentiment)), 1)
        if state.route == "auto_resolve":
            level = "None"
        elif state.priority_score >= 60 or (state.severity_label == "Critical" and fr["breach_prob"] >= 0.5):
            level = "MD/CEO Desk"
        elif state.priority_score >= 45 or fr["breach_prob"] >= 0.5 or fr["escalation_prob"] >= 0.6:
            level = "Regional Office"
        elif state.priority_score >= 30:
            level = "Branch"
        else:
            level = "None"
        state.escalation_level = level

        # ---- extra (non-taxonomy) actions
        extra = []
        if best["fast"]:
            extra.append("set_priority_queue_fast_track")
        if best["reroute"]:
            extra.append(f"reassign_to_team:{best['team']}")
        if fr["breach_prob"] >= 0.5 and state.route != "auto_resolve":
            extra += ["notify_customer_delay_risk", "schedule_sla_watch"]
        if fr["escalation_prob"] >= 0.5:
            extra.append("assign_relationship_manager_followup")
        state.extra_actions = extra
        if state.route != "auto_resolve" and fr["breach_prob"] >= 0.5:
            state.sla_status = "At Risk"
        return {"chosen": state.intervention["name"], "rationale": state.intervention["rationale"],
                "priority_score": state.priority_score, "escalation_level": level,
                "extra_actions": extra,
                "candidates": [{k: p[k] for k in ("name", "breach_prob", "cost", "loss")} for p in state.candidates]}
