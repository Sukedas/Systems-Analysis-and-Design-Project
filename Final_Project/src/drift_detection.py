import numpy as np
from scipy.stats import ks_2samp
from .utils import setup_logger

logger = setup_logger("drift_detection")

def detect_drift(train_dist, new_dist, alpha=0.05):
    """
    Performs Kolmogorov-Smirnov test to detect drift between training distribution and new data.
    """
    logger.info("Performing KS-Test for drift detection...")
    stat, p_value = ks_2samp(train_dist, new_dist)
    
    drift_detected = p_value < alpha
    
    result = {
        "statistic": float(stat),
        "p_value": float(p_value),
        "drift_detected": bool(drift_detected)
    }
    
    if drift_detected:
        logger.warning(f"Drift DETECTED! p-value: {p_value:.5f} < alpha: {alpha}")
    else:
        logger.info(f"No drift detected. p-value: {p_value:.5f}")
        
    return result
