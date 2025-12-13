import numpy as np
import pandas as pd
from .utils import setup_logger

logger = setup_logger("event_simulation")

def apply_shock(series, shock_magnitude=-0.2, duration=3):
    """
    Applies a temporary shock to the time series.
    """
    series = series.copy()
    shock_indices = np.random.choice(len(series) - duration, 1)
    start_idx = shock_indices[0]
    
    logger.info(f"Applying shock of magnitude {shock_magnitude} at index {start_idx} for {duration} steps.")
    
    for i in range(duration):
        series[start_idx + i] *= (1 + shock_magnitude)
        
    return series

def simulate_future_scenario(model, current_data, steps=12, shock_prob=0.1):
    """
    Simulates future trajectory with potential random shocks.
    Returns the predicted values.
    """
    predictions = []
    # If we have a model and data we would predict, here we might return a dummy trajectory for demo
    logger.info("Simulating future scenario...")
    
    # Create a dummy trajectory starting from last value or 1.0
    last_val = 1.0
    if isinstance(current_data, pd.DataFrame) and 'microbusiness_density' in current_data.columns:
        if not current_data.empty:
            last_val = current_data['microbusiness_density'].iloc[-1]
            
    for i in range(steps):
        # Random drift
        change = np.random.normal(0, 0.05)
        next_val = last_val * (1 + change)
        
        # Random shock
        if np.random.rand() < shock_prob:
             next_val = next_val * (1 - 0.2) # Negative shock
             logger.info(f"Shock applied at step {i}")
             
        predictions.append(next_val)
        last_val = next_val
        
    return predictions
