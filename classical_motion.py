import numpy as np

class ClassicalMotion:
    def __init__(self, x0, f):
        self.x0 = float(x0)
        # mass m = 1/2, so a = F/m = (-f)/(1/2) = -2f
        self.a = -2.0 * float(f)

    def compute_T(self):
        if self.a == 0:
            return np.inf
        # time until reaches x=0 from x0 with initial velocity 0: x0 + 0.5*a*T^2 = 0 -> T = sqrt(-2*x0/a)
        return np.sqrt(2.0 * self.x0 / abs(self.a))

    def trajectory(self, t_values):
        t = np.array(t_values, dtype=float)
        return self.x0 + 0.5 * self.a * t**2
