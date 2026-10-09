import unittest

import numpy as np

from cdi.data import synthetic as s
from fixtures import world


class TestGenerator(unittest.TestCase):
    def test_deterministic(self):
        a, b = s.generate(300, seed=3, days=30), s.generate(300, seed=3, days=30)
        self.assertTrue(a.drop(columns=[]).equals(b))

    def test_schema_and_ranges(self):
        df = world()["df"]
        self.assertFalse(df.isna().any().any())
        self.assertTrue(df["submitted_at"].is_monotonic_increasing)
        self.assertTrue(0.10 < df["breached"].mean() < 0.35)
        self.assertTrue(0.05 < df["escalated"].mean() < 0.30)
        self.assertTrue(((df["resolution_hours"] > df["sla_hours"]).astype(int) == df["breached"]).all())
        self.assertEqual(df.loc[df.route == "auto_resolve", "breached"].sum(), 0)
        self.assertTrue(df["team_load"].between(0.1, 1.6).all())

    def test_world_counterfactuals_are_directional(self):
        df = world()["te"].head(200)
        base = s.resolution_hours(df, fast_tracked=np.zeros(len(df)), rerouted=np.zeros(len(df)))
        fast = s.resolution_hours(df, fast_tracked=np.ones(len(df)), rerouted=np.zeros(len(df)))
        busy = s.resolution_hours(df, fast_tracked=np.zeros(len(df)), rerouted=np.zeros(len(df)),
                                  team_load=np.full(len(df), 1.5))
        calm = s.resolution_hours(df, fast_tracked=np.zeros(len(df)), rerouted=np.zeros(len(df)),
                                  team_load=np.full(len(df), 0.2))
        self.assertTrue((fast < base).all())
        self.assertTrue((busy > calm).all())

    def test_split_is_chronological(self):
        w = world()
        self.assertLessEqual(w["tr"]["submitted_at"].max(), w["va"]["submitted_at"].min())
        self.assertLessEqual(w["va"]["submitted_at"].max(), w["te"]["submitted_at"].min())

    def test_load_model_bounds(self):
        from datetime import datetime, timedelta
        lm = s.TeamLoadModel(datetime(2026, 1, 1), 10, 1)
        for team in ("Lending Ops", "Fraud & Risk"):
            vals = [lm.load(team, datetime(2026, 1, 1) + timedelta(hours=h)) for h in range(0, 200, 7)]
            self.assertTrue(all(0.1 <= v <= 1.6 for v in vals))
        self.assertEqual(lm.load("Automation Engine", datetime(2026, 1, 3)), 0.15)


if __name__ == "__main__":
    unittest.main()
