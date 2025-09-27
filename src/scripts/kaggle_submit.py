"""Prepare a Kaggle submission CSV from model predictions."""
import pandas as pd
from pathlib import Path


def format_submission(pred_df: pd.DataFrame, id_col: str | None = None) -> pd.DataFrame:
    df = pred_df.copy()
    if id_col and id_col in df.columns:
        df_out = df[[id_col, 'prediction']].rename(columns={id_col: 'Id', 'prediction': 'Prediction'})
    elif 'id' in df.columns:
        df_out = df[['id', 'prediction']].rename(columns={'id': 'Id', 'prediction': 'Prediction'})
    else:
        df_out = df.reset_index().rename(columns={'index': 'Id', 'prediction': 'Prediction'})
    return df_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True, help="CSV with predictions (from connector)")
    parser.add_argument("--out", default="submissions/submission.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.pred_csv)
    sub = format_submission(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print("Prepared Kaggle submission:", args.out)
