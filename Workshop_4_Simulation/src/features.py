import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived features.
    """
    df_features = df.copy()
    
    # Example derived features (placeholders if columns don't exist)
    # Assuming we might have 'pop' (population) and 'area' (land area) in a real dataset
    # For census_starter.csv, we'll create dummy derived features or use existing ones
    
    # Moving average example (if we had time series columns)
    # For this workshop, let's create a simple interaction feature if columns exist
    if 'pct_bb_2020' in df.columns and 'pct_bb_2021' in df.columns:
        df_features['growth_20_21'] = df_features['pct_bb_2021'] - df_features['pct_bb_2020']
    
    # Placeholder for population density if not present
    if 'population_density' not in df.columns:
        # Create a random synthetic feature for demonstration if real data is missing
        # In a real app, we would join with another table
        pass

    return df_features

def save_features(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    logger.info(f"Features saved to {path}")

def load_features(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
