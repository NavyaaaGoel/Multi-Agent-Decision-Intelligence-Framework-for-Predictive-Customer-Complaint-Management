"""ML intake classifier: complaint text -> product -> issue (hierarchical decoding)."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from cdi import taxonomy


class IntakeClassifier:
    def __init__(self, seed: int = 42):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=30000)
        self.prod = LogisticRegression(C=20, max_iter=400, random_state=seed)
        self.issue = LogisticRegression(C=20, max_iter=400, random_state=seed)
        self._issue_product = None

    def fit(self, texts, products, issue_keys) -> "IntakeClassifier":
        X = self.vec.fit_transform(texts)
        self.prod.fit(X, products)
        self.issue.fit(X, issue_keys)
        self._issue_product = np.array([k.split("::", 1)[0] for k in self.issue.classes_])
        return self

    def predict_batch(self, texts) -> list:
        X = self.vec.transform(texts)
        pp, ip = self.prod.predict_proba(X), self.issue.predict_proba(X)
        out = []
        for i in range(X.shape[0]):
            j = int(np.argmax(pp[i]))
            product, p_prod = self.prod.classes_[j], float(pp[i, j])
            mask = self._issue_product == product
            cand = np.where(mask, ip[i], 0.0)
            tot = cand.sum()
            cond = cand / tot if tot > 0 else cand
            order = np.argsort(-cond)[:3]
            top = [(self.issue.classes_[k].split("::", 1)[1], float(cond[k])) for k in order]
            out.append({"product": product, "issue": top[0][0], "product_confidence": p_prod,
                        "issue_confidence": top[0][1], "confidence": p_prod * top[0][1],
                        "top_issues": top})
        return out

    def predict_one(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "IntakeClassifier":
        return joblib.load(path)
