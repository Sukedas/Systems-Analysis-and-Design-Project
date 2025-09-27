"""Script to train or create a model artifact for testing."""
from src.model.model import train_dummy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="models/model.joblib")
args = parser.parse_args()

if __name__ == "__main__":
    out = train_dummy(args.output)
    print("Model saved to", out)
