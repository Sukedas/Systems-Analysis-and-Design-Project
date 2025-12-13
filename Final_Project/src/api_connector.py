import pandas as pd
import os
from .utils import setup_logger, load_config
from .preprocessing import preprocess_pipeline
from .feature_engineering import feature_engineering_pipeline
from .model_training import ModelTrainer
from .evaluation import calculate_metrics
from .drift_detection import detect_drift
from .submission_generator import generate_submission_file

from .cellular_automata import MicroEnterpriseCA
from .event_simulation import simulate_future_scenario, apply_shock

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
        if train_clean is not None:
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
            
            # 7. Run Simulation & Event Sim (Optional based on flag or always)
            logger.info("Running Simulations...")
            self.run_simulations(census)

            return metrics, drift_result
        else:
            logger.warning("Training data not available. Skipping Feature Engineering, Training, and Submission.")
            
            # Limited Mode: Run Simulation demo if census data exists
            if census is not None:
                logger.info(f"Census data available. entries: {len(census)}")
                logger.info("Running Limited Mode: Cellular Automata & Event Simulation")
                
                self.run_simulations(census)
            
            return None, None

    def run_simulations(self, census_data):
        # Ensure output directory exists because it might not be created in Limited Mode
        os.makedirs(self.output_path, exist_ok=True)

        # 1. Cellular Automata
        logger.info("Running Cellular Automata Simulation...")
        ca = MicroEnterpriseCA(grid_size=50)
        # Initialize with random or census density proxy (e.g. mean of normalized population)
        density = 0.1
        if census_data is not None and 'pct_bb_2021' in census_data.columns:
             # Example proxy: use mean broadband access as proxy for initial density * 0.01
             density = census_data['pct_bb_2021'].mean() / 1000.0 
             density = max(0.05, min(0.3, density)) # Clip
        
        ca.initialize_random(density=density)
        ca.run_simulation(steps=30)
        
        ca_output_path = os.path.join(self.output_path, 'ca_simulation_final.png')
        ca.visualize_step(step_idx=30, output_path=ca_output_path)
        
        # 2. Event Simulation
        logger.info("Running Event Simulation...")
        # Simulating a hypothetical county trajectory
        future_traj = simulate_future_scenario(model=None, current_data=None, steps=24)
        
        # Save event sim plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(future_traj, marker='o', linestyle='-')
        plt.title("Event Simulation: Future Density Projection with Shocks")
        plt.xlabel("Months")
        plt.ylabel("Projected Density")
        plt.grid(True)
        event_output_path = os.path.join(self.output_path, 'event_simulation_trajectory.png')
        plt.savefig(event_output_path)
        plt.close()
        logger.info(f"Event simulation plot saved to {event_output_path}")

if __name__ == "__main__":
    # Example usage
    pipeline = Pipeline()
    # pipeline.run() # Uncomment to run if data is present
