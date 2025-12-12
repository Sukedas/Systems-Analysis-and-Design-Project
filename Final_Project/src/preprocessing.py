import pandas as pd
import numpy as np
import os
from .utils import setup_logger

logger = setup_logger("preprocessing")

def load_data(data_path: str):
    """
    Loads raw datasets: train.csv, test.csv, census_starter.csv.
    """
    try:
        train = pd.read_csv(os.path.join(data_path, 'train.csv'))
        test = pd.read_csv(os.path.join(data_path, 'test.csv'))
        census = pd.read_csv(os.path.join(data_path, 'census_starter.csv'))
        logger.info("Datasets loaded successfully.")
        return train, test, census
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning pipeline:
    1. Convert dates
    2. Sort values
    3. Handle missing values (if any in critical columns)
    """
    df = df.copy()
    
    # Convert date to datetime
    if 'first_day_of_month' in df.columns:
        df['first_day_of_month'] = pd.to_datetime(df['first_day_of_month'])
        df = df.sort_values(by=['cfips', 'first_day_of_month']).reset_index(drop=True)
    
    # Imputation example (Microbusiness density shouldn't be null in train, but just in case)
    if 'microbusiness_density' in df.columns:
         df['microbusiness_density'] = df['microbusiness_density'].fillna(method='ffill')
    
    logger.info("Data cleaning completed.")
    return df

def preprocess_pipeline(data_path: str):
    """
    Orchestrates the loading and cleaning process.
    """
    logger.info("Starting preprocessing pipeline...")
    train, test, census = load_data(data_path)
    
    train_clean = clean_data(train)
    # Test data might not strictly need cleaning of target, but structure alignment
    test_clean = clean_data(test) 
    
    # Census might need cleaning? 
    # Usually it's static per cfips, check for simple structure
    
    return train_clean, test_clean, census
