import unittest

from cdi import config, nlp, severity, taxonomy


class TestNLP(unittest.TestCase):
    def test_sentiment_direction(self):
        self.assertLess(nlp.sentiment_score("This is unacceptable and I am frustrated"), 0)
        self.assertGreater(nlp.sentiment_score("Please help, I would appreciate it. Thank you"), 0)
        self.assertEqual(nlp.sentiment_score("my card"), 0)

    def test_threats(self):
        self.assertEqual(nlp.threat_count("I will go to the ombudsman and consumer court"), 2)
        self.assertEqual(nlp.threat_count("hello"), 0)

    def test_amount_formats(self):
        self.assertEqual(nlp.extract_amount("The amount is Rs. 12,500."), 12500)
        self.assertEqual(nlp.extract_amount("Amount: ₹830"), 830)
        self.assertEqual(nlp.extract_amount("INR 1,830 affected and Rs 40"), 1830)
        self.assertEqual(nlp.extract_amount("no money here"), 0.0)


class TestSeverity(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(severity.base_severity("UPI Fraud Complaint"), 10)
        self.assertEqual(severity.base_severity("Statement Request"), 3)
        self.assertEqual(severity.base_severity("Something unusual"), severity.DEFAULT_BASE)

    def test_labels_and_bumps(self):
        self.assertEqual(severity.label_for(9), "Critical")
        self.assertEqual(severity.label_for(4.9), "Low")
        score, label = severity.compute(8, amount=150_000, sentiment=-0.8)
        self.assertEqual((score, label), (9.5, "Critical"))
        self.assertEqual(severity.compute(10, 1e6, -1)[0], 10.0)  # capped


class TestTaxonomy(unittest.TestCase):
    def test_size_and_integrity(self):
        self.assertEqual(len(taxonomy.TAXONOMY), 161)
        self.assertEqual(len(taxonomy.PRODUCTS), 24)
        for s in taxonomy.TAXONOMY.values():
            self.assertIn(s.route, config.ROUTES)
            self.assertTrue(s.team)
            self.assertTrue(s.actions)
            if s.route == "partner_escalation":
                self.assertTrue(s.partner)
            if s.route != "bank_escalation":
                self.assertEqual(s.backup_team, "")

    def test_lookup(self):
        s = taxonomy.lookup("UPI", "UPI Fraud Complaint")
        self.assertEqual(s.team, "Fraud & Risk")
        self.assertIsNone(taxonomy.lookup("UPI", "nonexistent"))


if __name__ == "__main__":
    unittest.main()
