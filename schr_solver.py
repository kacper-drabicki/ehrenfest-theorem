import numpy as np

class SchrodingerSolver:
    def __init__(self, V, dx, dt, M):
        self.V = np.array(V, dtype=float)
        self.dx = float(dx)
        self.dt = float(dt)
        self.M = int(M)

    def build_matrices(self):
        # use complex dtype everywhere
        alpha = 1j * self.dt / (2.0 * self.dx**2)
        beta = 1j * self.dt / 2.0 * self.V  # V is float array
        # Build interior (1...M-1) matrices (exclude boundary points)
        n = self.M - 1  # number of interior points = M-1 (since points are 0..M)
        if n <= 0:
            raise ValueError("M must be >= 2")
        diag_G = (1.0 + 2.0 * alpha + beta)[1:-0]  # safe slice; we'll take interior below
        # but to be explicit take interior:
        diag_G = (1.0 + 2.0 * alpha + beta)[1:self.M]
        off = -alpha * np.ones(n - 1, dtype=np.complex128)

        G = np.diag(diag_G.astype(np.complex128))
        if n - 1 > 0:
            G += np.diag(off, 1) + np.diag(off, -1)

        diag_H = (1.0 - 2.0 * alpha - beta)[1:self.M].astype(np.complex128)
        H = np.diag(diag_H)
        if n - 1 > 0:
            H += np.diag(-off, 1) + np.diag(-off, -1)

        return G, H

    def evolve(self, psi0, steps):
        from numpy.linalg import solve
        G, H = self.build_matrices()
        # ensure psi is complex and has length M+1
        psi = psi0.astype(np.complex128).copy()
        results = [psi.copy()]
        n = self.M - 1  # interior count
        for _ in range(steps):
            b = H @ psi[1:-1]
            psi_interior = solve(G, b)
            psi[1:-1] = psi_interior
            # enforce boundary conditions explicitly
            psi[0] = 0.0 + 0.0j
            psi[-1] = 0.0 + 0.0j
            results.append(psi.copy())
        return results
