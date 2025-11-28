import numpy as np
from scipy.stats import ks_2samp
import logging
from .models import train_model, evaluate_model

logger = logging.getLogger(__name__)

def simulate_drift_and_retrain(model, X_test, y_test, config, sim_logger):
    """
    Simulates data drift, detects it, and triggers retraining.
    """
    logger.info("Starting drift simulation...")
    
    # 1. Perturb data (Drift)
    noise_level = config.get('drift_simulation', {}).get('noise_level', 0.1)
    X_drifted = X_test + np.random.normal(0, noise_level, X_test.shape)
    
    # 2. Detect Drift (KS Test on first feature as proxy)
    # Compare original test distribution vs drifted
    stat, p_value = ks_2samp(X_test.iloc[:, 0], X_drifted.iloc[:, 0])
    
    drift_threshold_p = config.get('drift_simulation', {}).get('drift_threshold_pvalue', 0.01)
    drift_detected = p_value < drift_threshold_p
    
    sim_logger.log_metric("drift_p_value", p_value, f"Drift detected: {drift_detected}")
    
    if drift_detected:
        logger.warning(f"Drift detected! (p={p_value:.4f}). Triggering retraining...")
        
        # 3. Retrain (Simulated by training on drifted data + original)
        # In reality, we'd label new data. Here we assume y stays valid or we have new labels.
        # For simulation, we'll just retrain on the original train set (or a mix) to show the workflow.
        # Let's just re-run training on X_test (treating it as new batch) for demonstration
        
        new_model = train_model(X_drifted, y_test, config) # Using drifted X and original y (assuming concept drift didn't break labels completely)
        
        # Evaluate new model
        metrics, _ = evaluate_model(new_model, X_drifted, y_test)
        sim_logger.log_metric("retrained_rmse", metrics['rmse'])
        
        return new_model, metrics
    else:
        logger.info("No significant drift detected.")
        return model, None
