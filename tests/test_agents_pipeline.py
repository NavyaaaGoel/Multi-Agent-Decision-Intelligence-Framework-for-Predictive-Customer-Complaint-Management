import unittest
from datetime import datetime

from cdi import config
from cdi.agents.verifier import VerificationAgent
from cdi.orchestrator.state import ComplaintInput
from fixtures import UPI_TEXT, make_orchestrator, world

T0 = datetime(2026, 2, 10, 11, 0)


def run(orch, text=UPI_TEXT, tier="Priority", cid="U1", t=T0):
    return orch.process(ComplaintInput(text, customer_id=cid, tier=tier, submitted_at=t))


class TestPipeline(unittest.TestCase):
    def test_end_to_end_state_is_complete(self):
        orch, ctx = make_orchestrator()
        st = run(orch)
        self.assertEqual([e["agent"] for e in st.trace],
                         ["intake", "risk_predictor", "sla_planner", "decision", "resource_router", "action", "verification"])
        self.assertTrue(st.verification["passed"], st.verification["failed"])
        self.assertIn(st.route, config.ROUTES)
        self.assertGreater(st.sla_hours, 0)
        self.assertTrue(st.deadline > st.submitted_at)
        self.assertTrue(st.assigned_team)
        self.assertGreater(ctx.store.audit_count(st.complaint_id), len(st.executed_actions))
        self.assertEqual(ctx.store.get(st.complaint_id)["status"], st.status)

    def test_amount_and_sentiment_extracted(self):
        st = run(make_orchestrator()[0])
        self.assertEqual(st.amount, 45000)
        self.assertLess(st.sentiment, 0)
        self.assertGreaterEqual(st.threat_count, 1)

    def test_decision_minimises_expected_loss_and_never_hurts(self):
        orch, _ = make_orchestrator()
        te = world()["te"].head(25)
        for k, r in enumerate(te.itertuples()):
            st = orch.process(ComplaintInput(r.narrative, customer_id=f"c{k}", tier=r.tier,
                                             submitted_at=r.submitted_at.to_pydatetime(), repeat_count=int(r.repeat_count)))
            losses = {c["name"]: c["loss"] for c in st.candidates}
            self.assertAlmostEqual(min(losses.values()), losses[st.intervention["name"]], places=3)
            self.assertLessEqual(st.intervention["breach_prob_after"], st.intervention["breach_prob_before"] + 1e-9)
            if st.route == "auto_resolve":
                self.assertEqual([c["name"] for c in st.candidates], ["standard"])
                self.assertEqual(st.status, "Resolved")
            if st.route == "partner_escalation":
                self.assertNotIn("reroute", [c["name"] for c in st.candidates])
            self.assertTrue(st.verification["passed"])

    def test_low_confidence_goes_to_human_review(self):
        st = run(make_orchestrator()[0], text="not happy with everything", tier="Standard")
        self.assertTrue(st.needs_human_review)
        self.assertEqual(st.status, "Pending Human Review")
        # safety: a guess must never trigger real route actions
        self.assertEqual([a["action"] for a in st.executed_actions], ["route_to_human_triage"])
        self.assertTrue(st.verification["passed"])

    def test_kill_switch_blocks_actions(self):
        orch, ctx = make_orchestrator(kill_switch=True)
        st = run(orch)
        self.assertEqual(st.executed_actions, [])
        self.assertTrue(st.status.startswith("Held"))
        self.assertTrue(st.verification["passed"])
        self.assertIn("kill_switch_hold", [a["action"] for a in ctx.store.audit(st.complaint_id)])

    def test_repeat_count_comes_from_store(self):
        orch, _ = make_orchestrator()
        a = run(orch, cid="REP", t=datetime(2026, 2, 10, 9, 0))
        b = run(orch, cid="REP", t=datetime(2026, 2, 12, 9, 0))
        self.assertEqual((a.repeat_count, b.repeat_count), (0, 1))

    def test_graceful_degradation_when_ml_fails(self):
        orch, ctx = make_orchestrator()
        original = ctx.risk.predict
        ctx.risk.predict = lambda df: (_ for _ in ()).throw(RuntimeError("model server down"))
        try:
            st = run(orch)
        finally:
            ctx.risk.predict = original
        self.assertTrue(st.degraded)
        self.assertEqual(st.baseline_risk["source"], "rule_based_fallback")
        self.assertTrue(st.verification["passed"])
        self.assertTrue(any("fell back" in e for e in st.errors))

    def test_hard_failure_hands_off_to_human(self):
        orch, ctx = make_orchestrator()
        ctx.intake = None  # breaks the intake agent
        st = run(orch)
        self.assertEqual(st.status, "Pending Human Review")
        self.assertTrue(any("aborted" in e for e in st.errors))
        self.assertIsNotNone(ctx.store.get(st.complaint_id))


class TestVerificationAndReplan(unittest.TestCase):
    def test_verifier_catches_corrupted_state(self):
        orch, ctx = make_orchestrator()
        st = run(orch)
        st.deadline = st.submitted_at          # invalid deadline
        st.final_risk["breach_prob"] = 1.7     # invalid probability
        st.intervention["breach_prob_after"] = 0.9
        st.intervention["breach_prob_before"] = 0.1  # intervention made it worse
        VerificationAgent()(st, ctx)
        self.assertFalse(st.verification["passed"])
        for c in ("deadline_consistent", "probabilities_in_range", "intervention_not_harmful"):
            self.assertIn(c, st.verification["failed"])

    def test_replan_loop_recovers_with_human_review(self):
        orch, ctx = make_orchestrator()
        calls = {"n": 0}
        real = orch.verifier

        class FlakyVerifier:
            name = "verification"

            def __call__(self, st, c):
                calls["n"] += 1
                if calls["n"] == 1:
                    st.verification = {"passed": False, "failed": ["forced"], "checks": []}
                    return {}
                return real(st, c)

        orch.verifier = FlakyVerifier()
        st = run(orch)
        self.assertEqual(st.replans, 1)
        self.assertTrue(st.needs_human_review)
        self.assertEqual(st.status, "Pending Human Review")
        self.assertIn("replan", [a["action"] for a in ctx.store.audit(st.complaint_id)])


if __name__ == "__main__":
    unittest.main()
