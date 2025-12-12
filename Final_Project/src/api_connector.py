import pandas as pd
import os
from .utils import setup_logger, load_config
from .preprocessing import preprocess_pipeline
from .feature_engineering import feature_engineering_pipeline
from .model_training import ModelTrainer
from .evaluation import calculate_metrics
from .drift_detection import detect_drift
from .submission_generator import generate_submission_file

logger = setup_logger("api_connector")

class Pipeline:
    def __init__(self, config=None):
        self.config = config if config else load_config()
        self.data_path = self.config['DATA_PATH']
        self.output_path = self.config['OUTPUT_PATH']
        
    def run(self, model_type='rf'):
        logger.info("Initializing End-to-End Pipeline...")
        
        # 1. Preprocessing
        train_clean, test_clean, census = preprocess_pipeline(self.data_path)
        
        # 2. Feature Engineering
        train_fe, test_fe = feature_engineering_pipeline(train_clean, test_clean, census)
        
        # Prepare Data for Modeling
        # Target: microbusiness_density
        # Drop columns not needed for training
        drop_cols = ['row_id', 'cfips', 'county', 'state', 'first_day_of_month', 'active', 'microbusiness_density']
        feature_cols = [c for c in train_fe.columns if c not in drop_cols]
        feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_fe[c])]
        
        X_train = train_fe[feature_cols]
        y_train = train_fe['microbusiness_density']
        X_test = test_fe[feature_cols]
        
        # 3. Model Training
        trainer = ModelTrainer(model_type=model_type)
        trainer.train(X_train, y_train)
        
        # 4. Evaluation (Training Set - Just for check, ideally use validation set)
        train_preds = trainer.predict(X_train)
        metrics = calculate_metrics(y_train, train_preds)
        
        # 5. Drift Detection (Concept Drift)
        # Check distribution of predictions or key features
        # Here checking target variable distribution vs prediction distribution
        drift_result = detect_drift(y_train, train_preds)
        
        # 6. Generate Submission
        test_preds = trainer.predict(X_test)
        
        # Create output dir if not exists
        os.makedirs(self.output_path, exist_ok=True)
        
        submission_path = os.path.join(self.output_path, 'submission.csv')
        # We need row_id from test_fe
        submission_df = pd.DataFrame({
            'row_id': test_fe['row_id'],
            'microbusiness_density': test_preds
        })
        
        submission_df.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")
        
        return metrics, drift_result

if __name__ == "__main__":
    # Example usage
    pipeline = Pipeline()
    # pipeline.run() # Uncomment to run if data is present
