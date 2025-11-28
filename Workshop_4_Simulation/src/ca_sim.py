import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class CellularAutomata:
    def __init__(self, grid_shape, config):
        self.grid_shape = grid_shape
        self.config = config
        self.grid = np.zeros(grid_shape)
        self.history = []

    def initialize_from_data(self, df_slice):
        """
        Initializes the grid from a data slice.
        Maps values to a normalized 0-1 range for the grid.
        """
        # Simple mapping: take first N values and fill grid
        flat_grid = self.grid.flatten()
        values = df_slice.values[:len(flat_grid)]
        
        # Normalize
        if len(values) > 0:
            norm_values = (values - values.min()) / (values.max() - values.min() + 1e-9)
            flat_grid[:len(values)] = norm_values.flatten()
        
        self.grid = flat_grid.reshape(self.grid_shape)
        self.history.append(self.grid.copy())
        logger.info("CA Grid initialized.")

    def get_neighbors_sum(self):
        # Simple convolution for neighbor sum (Moore neighborhood)
        padded = np.pad(self.grid, 1, mode='wrap')
        neighbor_sum = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2]                     + padded[1:-1, 2:] +
            padded[2:, :-2]  + padded[2:, 1:-1]  + padded[2:, 2:]
        )
        return neighbor_sum

    def step(self):
        """
        Applies CA rules.
        """
        neighbor_sum = self.get_neighbors_sum()
        
        # Rule: Growth based on neighbors and random perturbation
        growth_threshold = self.config.get('simulation', {}).get('growth_threshold', 0.6)
        decay_prob = self.config.get('simulation', {}).get('decay_probability', 0.02)
        perturbation_sigma = self.config.get('simulation', {}).get('perturbation_sigma', 0.05)
        
        # Stochastic update
        noise = np.random.normal(0, perturbation_sigma, self.grid.shape)
        
        # Logic: If neighbors are strong, grow. If random decay, die.
        # This is a continuous CA (values 0-1)
        new_grid = self.grid + (neighbor_sum / 8.0) * 0.1 + noise
        
        # Decay
        decay_mask = np.random.random(self.grid.shape) < decay_prob
        new_grid[decay_mask] *= 0.5
        
        # Clip
        new_grid = np.clip(new_grid, 0, 1)
        
        self.grid = new_grid
        self.history.append(self.grid.copy())

    def run(self, steps):
        for _ in range(steps):
            self.step()
        logger.info(f"CA Simulation completed for {steps} steps.")
        return np.array(self.history)
