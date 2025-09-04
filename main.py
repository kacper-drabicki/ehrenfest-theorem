import argparse
import numpy as np
from potential import Potential
from wavefunction import WaveFunction
from schr_solver import SchrodingerSolver
from expectation import ExpectationCalculator
from classical_motion import ClassicalMotion
from plotter import Plotter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Symulacja twierdzenia Ehrenfesta")
    parser.add_argument("--K", type=int, required=True, help="Liczba kroków czasowych")
    parser.add_argument("--M", type=int, required=True, help="Liczba punktów przestrzennych (parzyste)")
    parser.add_argument("--L", type=float, required=True, help="Szerokość studni")
    parser.add_argument("--f", type=float, required=True, help="Nachylenie potencjału")
    parser.add_argument("--x0", type=float, required=True, help="Początkowe położenie paczki")
    parser.add_argument("--sigma", type=float, required=True, help="Szerokość paczki Gaussa")

    args = parser.parse_args()

    if args.M % 2 != 0:
        raise ValueError("M must be even so that M+1 is odd (Simpson's rule requirement).")

    potential = Potential(args.L, args.f, args.M)
    V = potential.generate()

    motion = ClassicalMotion(args.x0, args.f)
    T = motion.compute_T()
    if not np.isfinite(T):
        raise ValueError("Computed T is infinite or NaN. Check parameters.")

    dt = T / args.K
    dx = args.L / args.M
    t_values = np.linspace(0.0, T, args.K + 1)

    wf = WaveFunction(potential.x, args.x0, args.sigma)
    psi0 = wf.gaussian_packet()

    solver = SchrodingerSolver(V, dx, dt, args.M)
    psi_all = solver.evolve(psi0, args.K)

    exp_calc = ExpectationCalculator(potential.x, dx)
    quantum_positions = [exp_calc.position(psi) for psi in psi_all]

    classical_positions = motion.trajectory(t_values)

    # compute simple convergence metrics
    quantum = np.array(quantum_positions, dtype=float)
    classical = np.array(classical_positions, dtype=float)

    plotter = Plotter()
    plotter.plot_all(t_values, quantum, classical)
