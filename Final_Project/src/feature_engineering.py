import pandas as pd
import numpy as np
from .utils import setup_logger

logger = setup_logger("feature_engineering")

def create_fips_features(census_df):
    """
    Standardize census data to be merged by cfips.
    """
    # Assuming census_df has 'cfips' and various features.
    # We might want to select specific columns or feature engineer on them.
    # For now, return as is or minimal selection.
    return census_df

def create_lag_features(df, lags=[1, 2, 3, 6, 12]):
    """
    Creates lag features for the target variable 'microbusiness_density'.
    Must be applied per 'cfips'.
    """
    df = df.copy()
    for lag in lags:
        df[f'mbd_lag_{lag}'] = df.groupby('cfips')['microbusiness_density'].shift(lag)
    return df

def create_rolling_features(df, window=3):
    """
    Creates rolling mean features.
    """
    df = df.copy()
    df[f'mbd_roll_mean_{window}'] = df.groupby('cfips')['microbusiness_density'].transform(lambda x: x.shift(1).rolling(window).mean())
    return df

def feature_engineering_pipeline(train_df, test_df, census_df):
    """
    Applies feature engineering to both train and test sets.
    Merges census data.
    """
    logger.info("Starting feature engineering...")
    
    # Combine for consistent feature generation (handle time continuity)
    # Typically we might concat train and test to generate lags safely if test is sequential
    # For Kaggle time series, test set often follows train set.
    
    # Add a flag to distinguish
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    
    # Ensure test has 'microbusiness_density' column for concatenation (NaN usually)
    if 'microbusiness_density' not in test_df.columns:
        test_df['microbusiness_density'] = np.nan
        
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.sort_values(by=['cfips', 'first_day_of_month'])
    
    # 1. Merge Census Data
    full_df = full_df.merge(census_df, on='cfips', how='left')
    
    # 2. Lag Features
    full_df = create_lag_features(full_df)
    
    # 3. Rolling Features
    full_df = create_rolling_features(full_df)
    
    # 4. Extract Date Features
    full_df['year'] = full_df['first_day_of_month'].dt.year
    full_df['month'] = full_df['first_day_of_month'].dt.month
    
    # Split back
    train_fe = full_df[full_df['is_train'] == 1].drop(columns=['is_train'])
    test_fe = full_df[full_df['is_train'] == 0].drop(columns=['is_train'])
    
    # For training, we need to drop rows where lags generated NaNs (start of series)
    train_fe = train_fe.dropna(subset=['mbd_lag_1']) # Basic drop check
    
    logger.info("Feature engineering completed.")
    return train_fe, test_fe
