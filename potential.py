import numpy as np

class Potential:
    def __init__(self, L, f, M, wall_value=1e12):
        self.L = float(L)
        self.f = float(f)
        self.M = int(M)
        self.wall_value = wall_value
        self.dx = self.L / self.M
        self.x = np.linspace(0.0, self.L, self.M + 1)

    def generate(self):
        """Linear potential with high (large finite) walls at x=0 and x=L."""
        V = np.zeros_like(self.x, dtype=float)
        for i, xi in enumerate(self.x):
            if xi <= 0.0 or xi >= self.L:
                V[i] = self.wall_value
            else:
                V[i] = self.f * xi
        return V
