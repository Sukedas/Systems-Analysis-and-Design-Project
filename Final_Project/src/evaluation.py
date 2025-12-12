import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .utils import setup_logger

logger = setup_logger("evaluation")

def calculate_metrics(y_true, y_pred):
    """
    Calculates RMSE and MAE.
    SMAPE is also often used in Kaggle.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # SMAPE calculation
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    smape = np.mean(diff)
    
    metrics = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "SMAPE": float(smape)
    }
    
    logger.info(f"Evaluation Metrics: {metrics}")
    return metrics
