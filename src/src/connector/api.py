"""Simple CLI to connect input CSV data to the model for prediction.

Usage:
    python -m src.connector.api --model models/model.joblib --input data/test.csv --output submissions/pred.csv

The input CSV is expected to contain only feature columns (no target). The
script will load the model and write a CSV with a `prediction` column.
"""
import click
import pandas as pd
from pathlib import Path
from src.model.model import ModelWrapper


@click.command()
@click.option("--model", required=True, help="Path to saved model (joblib)")
@click.option("--input", "input_csv", required=True, help="Input CSV with features")
@click.option("--output", required=True, help="Output CSV path for predictions")
def predict_from_csv(model, input_csv, output):
    model_path = Path(model)
    in_path = Path(input_csv)
    out_path = Path(output)

    if not in_path.exists():
        raise SystemExit(f"Input CSV not found: {in_path}")

    mw = ModelWrapper(model_path)
    mw.load()

    df = pd.read_csv(in_path)
    preds = mw.predict(df.values)
    df_out = df.copy()
    df_out["prediction"] = preds
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    predict_from_csv()
