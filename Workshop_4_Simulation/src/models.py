import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import logging
import os

logger = logging.getLogger(__name__)

def train_model(X_train, y_train, config: dict, model_type='random_forest'):
    """
    Trains a model based on config.
    """
    if model_type == 'random_forest':
        params = config.get('models', {}).get('random_forest', {})
        model = RandomForestRegressor(**params)
    elif model_type == 'mlp':
        params = config.get('models', {}).get('mlp', {})
        model = MLPRegressor(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    logger.info(f"Trained {model_type} model.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model.
    """
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    metrics = {
        'rmse': rmse,
        'mae': mae
    }
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics, predictions

def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")

def load_model(path: str):
    return joblib.load(path)
