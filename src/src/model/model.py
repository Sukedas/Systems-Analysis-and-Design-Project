"""Simple model loader and wrapper.

This module provides a minimal ModelWrapper that can load a scikit-learn
estimator saved with joblib and run predictions. It also includes a tiny
train_dummy function to create a placeholder model for testing.
"""
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


class ModelWrapper:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.model = None

    def load(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Model file not found: {self.path}")
        self.model = joblib.load(self.path)
        return self.model

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model.predict(X)


def train_dummy(save_path: str | Path):
    """Train a tiny random forest on synthetic data and save it.

    This is for template/testing purposes only.
    """
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, save_path)
    return save_path


if __name__ == "__main__":
    # Quick local test when run as a script
    out = train_dummy("models/model.joblib")
    print("Saved dummy model to", out)
