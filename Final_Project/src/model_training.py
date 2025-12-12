import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from .utils import setup_logger, set_seed

logger = setup_logger("model_training")

class ModelTrainer:
    def __init__(self, model_type='rf', params=None):
        self.model_type = model_type
        self.params = params if params else {}
        self.model = None
        set_seed(42)  # Ensure reproducibility per run attempt

    def train(self, X_train, y_train):
        """
        Trains the selected model.
        """
        logger.info(f"Training {self.model_type} model...")
        
        if self.model_type == 'rf':
            # Default params for RF if not provided
            default_params = {'n_estimators': 100, 'max_depth': 10, 'n_jobs': -1, 'random_state': 42}
            final_params = {**default_params, **self.params}
            self.model = RandomForestRegressor(**final_params)
            self.model.fit(X_train, y_train)
            
        elif self.model_type == 'xgb':
            # Default params for XGBoost
            default_params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42}
            final_params = {**default_params, **self.params}
            self.model = xgb.XGBRegressor(**final_params)
            self.model.fit(X_train, y_train)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        logger.info("Training completed.")

    def predict(self, X):
        if not self.model:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def get_feature_importance(self, feature_names):
        if self.model_type == 'rf':
            importances = self.model.feature_importances_
        elif self.model_type == 'xgb':
            importances = self.model.feature_importances_
        else:
            return None
        
        return pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
