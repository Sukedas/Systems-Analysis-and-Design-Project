"""Simulation harness to test model integration under varied data sizes."""
import time
import numpy as np
import pandas as pd
from src.model.model import ModelWrapper
from pathlib import Path


def run_simulation(model_path, n_samples_list=(10, 100, 1000)):
    mw = ModelWrapper(model_path)
    mw.load()
    results = []
    for n in n_samples_list:
        X = np.random.randn(n, 10)
        start = time.time()
        preds = mw.predict(X)
        elapsed = time.time() - start
        results.append({"n": n, "time_s": elapsed})
    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.joblib")
    args = parser.parse_args()
    df = run_simulation(args.model)
    print(df)
