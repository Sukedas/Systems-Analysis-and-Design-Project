import pandas as pd
import os
from .utils import setup_logger

logger = setup_logger("submission_generator")

def generate_submission_file(predictions, row_ids, output_path):
    """
    Generates the Kaggle submission file.
    """
    df = pd.DataFrame({
        'row_id': row_ids,
        'microbusiness_density': predictions
    })
    
    df.to_csv(output_path, index=False)
    logger.info(f"Submission file generated at: {output_path}")
    return df
