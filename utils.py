import numpy as np

def simpson_integration(y, dx):
    """Composite Simpson's rule for equally spaced points.
    If number of points is even, apply Simpson on first n-1 points and trapezoid on last interval.
    Works for real or complex y.
    """
    n = len(y)
    if n < 2:
        return 0.0
    # If odd number of points -> n is odd -> Simpson directly
    if n % 2 == 1:
        return dx / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))
    else:
        # apply Simpson on first n-1 points (which is odd), then trapezoid on last interval
        simpson_part = simpson_integration(y[:-1], dx)
        trap_part = 0.5 * (y[-2] + y[-1]) * dx
        return simpson_part + trap_part

def normalize(psi, dx):
    """Normalize wavefunction psi (can be complex)."""
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm == 0:
        return psi
    return psi / norm
