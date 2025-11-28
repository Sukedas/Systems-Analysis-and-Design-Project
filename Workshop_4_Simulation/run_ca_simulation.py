import argparse
import yaml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.ca_sim import CellularAutomata
from src.monitoring import setup_logging

def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    
    # Initialize CA
    grid_shape = (config['grid']['height'], config['grid']['width'])
    ca = CellularAutomata(grid_shape, config)
    
    # Initialize from random data (simulating data slice)
    # In real app, load_data() and pass slice
    dummy_data = pd.DataFrame(np.random.rand(100, 1), columns=['val'])
    ca.initialize_from_data(dummy_data)
    
    # Run
    steps = config['simulation']['steps']
    history = ca.run(steps)
    
    # Visualize
    # 1. Final State
    plt.figure(figsize=(8, 8))
    sns.heatmap(history[-1], cmap="viridis", vmin=0, vmax=1)
    plt.title(f"CA State at Step {steps}")
    plt.savefig(os.path.join(output_dir, "ca_final_state.png"))
    plt.close()
    
    # 2. Time Series of Total Activity
    activity = [np.sum(grid) for grid in history]
    plt.figure(figsize=(10, 5))
    plt.plot(activity)
    plt.title("Total Microenterprise Activity Over Time")
    plt.xlabel("Step")
    plt.ylabel("Total Activity")
    plt.savefig(os.path.join(output_dir, "ca_activity_series.png"))
    plt.close()
    
    print(f"CA Simulation completed. Results in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)
