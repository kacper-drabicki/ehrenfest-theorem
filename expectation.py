import numpy as np
from utils import simpson_integration

class ExpectationCalculator:
    def __init__(self, x, dx):
        self.x = np.array(x, dtype=float)
        self.dx = float(dx)

    def position(self, psi):
        # psi can be complex
        density_times_x = (np.abs(psi)**2) * self.x
        return simpson_integration(density_times_x, self.dx)
