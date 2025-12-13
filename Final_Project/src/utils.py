import os
import random
import numpy as np
import logging
import pandas as pd

def setup_logger(name: str = "project_logger", log_file: str = "project.log") -> logging.Logger:
    """
    Sets up a logger with console and file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console Handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File Handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def set_seed(seed: int = 42):
    """
    Sets the seed for reproducibility across random, numpy, and other libraries if needed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # If using torch/tensorflow, add their seed setting here
    
def load_config():
    """
    Placeholder for loading configuration (yaml/json) if needed.
    Returns a dictionary of config values.
    """
    return {
        "DATA_PATH": "data",
        "OUTPUT_PATH": "outputs",
        "SEED": 42
    }
