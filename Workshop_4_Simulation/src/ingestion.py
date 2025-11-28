import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def load_data(path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
    
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded data from {path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def validate_schema(df: pd.DataFrame, required_columns: list = None) -> bool:
    """
    Validates that the DataFrame contains the required columns.
    """
    if required_columns is None:
        # Default columns expected in census_starter.csv based on description
        # Adjust based on actual file content if needed
        required_columns = ['pct_bb_2017', 'pct_bb_2018', 'pct_bb_2019', 'pct_bb_2020', 'pct_bb_2021',
                            'cfips', 'microbusiness_density'] 
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        # In a real scenario, we might return False or raise an error.
        # For this workshop, we'll log and return False but let the caller decide.
        return False
    
    logger.info("Schema validation passed.")
    return True
