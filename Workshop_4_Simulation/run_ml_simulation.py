import argparse
import yaml
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.ingestion import load_data, validate_schema
from src.preprocessing import preprocess_data
from src.features import create_features
from src.models import train_model, evaluate_model, save_model
from src.experiments import simulate_drift_and_retrain
from src.monitoring import setup_logging, SimulationLogger

def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['output_dir'], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    sim_logger = SimulationLogger(output_dir)
    
    # 1. Ingestion
    df = load_data(config['data_path'])
    validate_schema(df)
    
    # 2. Preprocessing & Features
    df = preprocess_data(df, config)
    df = create_features(df)
    
    # Prepare X, y (Assuming 'microbusiness_density' is target)
    target_col = 'microbusiness_density'
    if target_col not in df.columns:
        # Fallback for demo if column missing (e.g. if scaling removed it or name mismatch)
        target_col = df.columns[-1]
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config['random_seed'])
    
    # 3. Train Initial Model
    model = train_model(X_train, y_train, config, model_type='random_forest')
    metrics, preds = evaluate_model(model, X_test, y_test)
    
    sim_logger.log_metric("initial_rmse", metrics['rmse'])
    save_model(model, os.path.join(output_dir, "rf_model.joblib"))
    
    # Plot Residuals
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=preds)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title("Actual vs Predicted")
    plt.savefig(os.path.join(output_dir, "residuals.png"))
    plt.close()
    
    # 4. Drift Simulation & Retraining
    simulate_drift_and_retrain(model, X_test, y_test, config, sim_logger)
    
    print(f"ML Simulation completed. Results in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)
