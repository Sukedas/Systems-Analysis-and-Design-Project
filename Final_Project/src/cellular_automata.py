import numpy as np
import matplotlib.pyplot as plt
from .utils import setup_logger, set_seed

logger = setup_logger("cellular_automata")

class MicroEnterpriseCA:
    def __init__(self, grid_size=50, p_growth=0.05, p_decay=0.01):
        """
        Initializes the Cellular Automata grid.
        States: 0 (Empty/Dead), 1 (Active Microenterprise)
        """
        self.grid_size = grid_size
        self.p_growth = p_growth
        self.p_decay = p_decay
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        set_seed(42)

    def initialize_random(self, density=0.1):
        """
        Randomly populates the grid.
        """
        self.grid = (np.random.rand(self.grid_size, self.grid_size) < density).astype(int)
        logger.info(f"CA initialized with density {density}")

    def step(self):
        """
        Evolves grid by one step.
        Rules:
        - If Empty (0): Grows to 1 with prob p_growth * neighbors
        - If Active (1): Decays to 0 with prob p_decay (or shock)
        """
        new_grid = self.grid.copy()
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Count neighbors
                neighbors = np.sum(self.grid[max(0, i-1):min(self.grid_size, i+2), 
                                             max(0, j-1):min(self.grid_size, j+2)]) - self.grid[i, j]
                
                if self.grid[i, j] == 0:
                    # Growth rule: chance increases with active neighbors
                    chance = self.p_growth * (neighbors + 0.1) # Base chance even without neighbors
                    if np.random.rand() < chance:
                        new_grid[i, j] = 1
                else:
                    # Decay rule
                    if np.random.rand() < self.p_decay:
                        new_grid[i, j] = 0
                        
        self.grid = new_grid
        return self.grid

    def run_simulation(self, steps=50):
        history = []
        for _ in range(steps):
            history.append(self.grid.copy())
            self.step()
        return history

    def visualize_step(self, step_idx=None):
        plt.imshow(self.grid, cmap='Greens')
        plt.title(f"CA State - Step {step_idx}")
        plt.show()
