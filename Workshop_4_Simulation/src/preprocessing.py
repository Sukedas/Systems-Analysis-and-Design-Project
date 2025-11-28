import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame, config: dict, save_path: str = None) -> pd.DataFrame:
    """
    Preprocesses the data: imputation and scaling.
    """
    # Select numeric columns for simplicity in this workshop
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols].copy()

    # Imputation
    strategy = config.get('preprocessing', {}).get('imputation_strategy', 'median')
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_cols)
    
    # Scaling
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=numeric_cols)
    
    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        joblib.dump(scaler, save_path)
        logger.info(f"Scaler saved to {save_path}")

    return df_scaled

def load_scaler(path: str):
    return joblib.load(path)
