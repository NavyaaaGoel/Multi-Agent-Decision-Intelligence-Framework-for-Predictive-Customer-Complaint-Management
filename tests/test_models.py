import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from cdi.ml.intake_model import IntakeClassifier
from cdi.ml.risk_models import RiskModels, lognormal_sf
from fixtures import world


class TestRiskModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = world()
        cls.pred = cls.w["risk"].predict(cls.w["te"])

    def test_breach_beats_chance_and_rule(self):
        te = self.w["te"]
        auc = roc_auc_score(te["breached"], self.pred["breach_prob"])
        self.assertGreater(auc, 0.80)

    def test_probabilities_valid(self):
        for c in ("breach_prob", "escalation_prob", "p_gb", "p_lr", "p_dist"):
            self.assertTrue(self.pred[c].between(0, 1).all(), c)

    def test_time_forecast_and_interval(self):
        te = self.w["te"]
        self.assertTrue((self.pred["p90_hours"] > self.pred["median_hours"]).all())
        cover = (te["resolution_hours"] <= self.pred["p90_hours"]).mean()
        self.assertTrue(0.80 < cover < 0.97, cover)
        self.assertGreater(np.corrcoef(np.log(te["resolution_hours"]), self.pred["log_mu"])[0, 1], 0.8)

    def test_models_respond_to_interventions_in_the_right_direction(self):
        te = self.w["te"]
        sub = te[(te["route"] == "bank_escalation") & (te["severity_label"].isin(["Critical", "High"]))].head(100).copy()
        base = sub.assign(fast_tracked=0, rerouted=0)
        fast = sub.assign(fast_tracked=1, rerouted=0)
        r = self.w["risk"]
        self.assertGreater(r.predict(base)["median_hours"].mean(), r.predict(fast)["median_hours"].mean())
        self.assertGreaterEqual(r.predict(base)["breach_prob"].mean(), r.predict(fast)["breach_prob"].mean())

    def test_unknown_category_does_not_crash(self):
        row = self.w["te"].head(2).copy()
        row["issue"] = "A brand new issue"
        row["channel"] = "Telepathy"
        self.assertEqual(len(self.w["risk"].predict(row)), 2)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.joblib"
            self.w["risk"].save(p)
            again = RiskModels.load(p).predict(self.w["te"].head(20))
            np.testing.assert_allclose(again["breach_prob"], self.pred["breach_prob"].head(20))

    def test_lognormal_sf(self):
        self.assertAlmostEqual(lognormal_sf(np.exp(2.0), 2.0, 0.5), 0.5, places=6)
        self.assertGreater(lognormal_sf(5, 2, 0.5), lognormal_sf(10, 2, 0.5))
        self.assertEqual(lognormal_sf(0, 2, 0.5), 1.0)


class TestIntake(unittest.TestCase):
    def test_accuracy_and_confidence(self):
        w = world()
        res = w["intake"].predict_batch(list(w["te"]["narrative"]))
        acc = np.mean([r["product"] == p for r, p in zip(res, w["te"]["product"])])
        self.assertGreater(acc, 0.80)
        self.assertTrue(all(0 <= r["confidence"] <= 1 for r in res))
        self.assertEqual(len(res[0]["top_issues"]), 3)

    def test_issue_belongs_to_predicted_product(self):
        from cdi import taxonomy
        r = world()["intake"].predict_one("my upi payment failed and money was debited")
        self.assertIsNotNone(taxonomy.lookup(r["product"], r["issue"]))


if __name__ == "__main__":
    unittest.main()
