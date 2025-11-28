import pytest
import pandas as pd
import numpy as np
import os
from src.preprocessing import preprocess_data

def test_preprocess_data():
    # Create dummy data
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, 6, 7, 8]
    })
    
    config = {'preprocessing': {'imputation_strategy': 'median'}}
    
    # Run preprocessing
    df_processed = preprocess_data(df, config)
    
    # Assertions
    assert df_processed.shape == df.shape
    assert not df_processed.isnull().any().any()
    # Median of 1, 2, 4 is 2. So missing value should be filled with 2.
    # Standard scaler will then transform it.
    
    # Check if scaler file is saved if path provided
    save_path = "test_scaler.joblib"
    preprocess_data(df, config, save_path=save_path)
    assert os.path.exists(save_path)
    os.remove(save_path)
