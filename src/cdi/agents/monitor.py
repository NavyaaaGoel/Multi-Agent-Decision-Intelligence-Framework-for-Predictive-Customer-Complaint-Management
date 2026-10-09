"""SLA Monitor: re-scores in-flight cases as time passes (the 'SLA-aware' loop).

Conditional breach probability uses the predictive log-normal:  P(T > SLA | T > elapsed)
= S(SLA) / S(elapsed), so a case that is still open late in its window becomes riskier
even if the intake-time forecast looked safe.
"""
from __future__ import annotations

from datetime import datetime

from cdi.agents.decision import LEVELS, level_at_least
from cdi.ml.risk_models import lognormal_sf


class SLAMonitorAgent:
    name = "sla_monitor"

    def scan(self, ctx, now: datetime) -> list:
        events = []
        for p in ctx.store.open_complaints():
            if p.get("status", "").startswith("Held") or not p.get("deadline"):
                continue
            sub = datetime.fromisoformat(p["submitted_at"])
            elapsed = (now - sub).total_seconds() / 3600
            sla = p["sla_hours"]
            mu, sg = p["final_risk"].get("log_mu", 0.0), ctx.risk.sigma
            if elapsed >= sla:
                prob, status = 1.0, "Breached"
            else:
                s_e = lognormal_sf(max(elapsed, 1e-6), mu, sg)
                prob = 1.0 if s_e < 1e-9 else min(1.0, lognormal_sf(sla, mu, sg) / s_e)
                status = "At Risk" if (prob >= 0.5 or elapsed / sla >= 0.8) else "On Track"
            old_status, old_level = p.get("sla_status"), p.get("escalation_level", "None")
            level = old_level
            if status == "Breached":
                level = level_at_least(level, "MD/CEO Desk" if p["severity_label"] in ("Critical", "High") else "Regional Office")
            elif status == "At Risk":
                level = level_at_least(level, "Regional Office")
            if status != old_status or level != old_level:
                ev = {"complaint_id": p["complaint_id"], "elapsed_hours": round(elapsed, 1),
                      "live_breach_prob": round(prob, 3), "sla_status": f"{old_status} -> {status}",
                      "escalation": f"{old_level} -> {level}"}
                p["sla_status"], p["escalation_level"], p["live_breach_prob"] = status, level, round(prob, 4)
                p.setdefault("monitor_events", []).append({**ev, "at": now.isoformat()})
                ctx.store.upsert(p)
                ctx.store.log_action(p["complaint_id"], now, self.name, "sla_status_change", ev)
                events.append(ev)
            else:  # no state change, but keep the live risk score fresh for API consumers
                p["live_breach_prob"] = round(prob, 4)
                ctx.store.upsert(p)
        return events
