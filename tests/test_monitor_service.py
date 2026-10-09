import unittest
from datetime import datetime, timedelta

from cdi.agents.base import Context
from cdi.agents.monitor import SLAMonitorAgent
from cdi.service import ComplaintService
from fixtures import UPI_TEXT, make_orchestrator, world

T0 = datetime(2026, 2, 10, 11, 0)


class TestMonitor(unittest.TestCase):
    def setUp(self):
        from cdi.orchestrator.state import ComplaintInput
        self.orch, self.ctx = make_orchestrator()
        self.st = self.orch.process(ComplaintInput(UPI_TEXT, customer_id="M1", tier="Priority", submitted_at=T0))
        self.mon = SLAMonitorAgent()

    def test_no_change_right_after_intake(self):
        self.assertEqual(self.mon.scan(self.ctx, T0 + timedelta(minutes=1)), [])

    def test_breach_detected_and_escalated(self):
        ev = self.mon.scan(self.ctx, T0 + timedelta(hours=self.st.sla_hours + 1))
        self.assertEqual(len(ev), 1)
        self.assertTrue(ev[0]["sla_status"].endswith("Breached"))
        saved = self.ctx.store.get(self.st.complaint_id)
        self.assertEqual(saved["sla_status"], "Breached")
        self.assertIn(saved["escalation_level"], ("Regional Office", "MD/CEO Desk"))
        self.assertIn("sla_status_change", [a["action"] for a in self.ctx.store.audit(self.st.complaint_id)])

    def test_conditional_risk_rises_over_time(self):
        probs = []
        for h in (1, 0.5 * self.st.sla_hours, 0.9 * self.st.sla_hours):
            self.mon.scan(self.ctx, T0 + timedelta(hours=h))
            probs.append(self.ctx.store.get(self.st.complaint_id).get("live_breach_prob"))
        self.assertTrue(all(p is not None for p in probs[1:]))
        self.assertLessEqual(probs[1], probs[2] + 1e-9)

    def test_resolved_cases_are_ignored(self):
        from cdi.orchestrator.state import ComplaintInput
        st = self.orch.process(ComplaintInput("I lost my credit card please block it", customer_id="M2", submitted_at=T0))
        self.assertEqual(st.status, "Resolved")
        ev = self.mon.scan(self.ctx, T0 + timedelta(days=30))
        self.assertNotIn(st.complaint_id, [e["complaint_id"] for e in ev])


class TestService(unittest.TestCase):
    def setUp(self):
        w = world()
        self.svc = ComplaintService(w["intake"], w["risk"])

    def test_submit_get_trace(self):
        out = self.svc.submit({"text": UPI_TEXT, "customer_id": "S1", "tier": "Premium",
                               "submitted_at": T0.isoformat()})
        self.assertEqual(self.svc.get(out["complaint_id"])["complaint_id"], out["complaint_id"])
        tr = self.svc.trace(out["complaint_id"])
        self.assertGreaterEqual(len(tr["trace"]), 6)
        self.assertGreater(len(tr["audit"]), 0)
        self.assertIsNone(self.svc.get("nope"))

    def test_validation(self):
        with self.assertRaises(ValueError):
            self.svc.submit({"text": "   "})
        with self.assertRaises(ValueError):
            self.svc.submit({"text": "hello there", "tier": "Gold"})

    def test_scan_and_live_load(self):
        self.svc.submit({"text": UPI_TEXT, "customer_id": "S2", "submitted_at": T0.isoformat()})
        events = self.svc.scan_sla((T0 + timedelta(days=20)).isoformat())
        self.assertTrue(len(events) >= 1)
        load = self.svc.team_load()
        self.assertTrue(all(0.1 <= v <= 1.6 for v in load.values()))


if __name__ == "__main__":
    unittest.main()
