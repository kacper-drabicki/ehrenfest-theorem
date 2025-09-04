import numpy as np
from utils import normalize

class WaveFunction:
    def __init__(self, x, x0, sigma):
        self.x = x
        self.x0 = float(x0)
        self.sigma = float(sigma)

    def gaussian_packet(self):
        prefac = (1.0 / (np.pi * self.sigma**2)) ** 0.25
        psi = prefac * np.exp(-0.5 * ((self.x - self.x0) / self.sigma) ** 2)
        # ensure complex dtype
        psi = psi.astype(np.complex128)
        return normalize(psi, self.x[1] - self.x[0])
