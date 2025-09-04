import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def plot_all(self, t, quantum, classical):
        t = np.array(t)
        quantum = np.array(quantum, dtype=float)
        classical = np.array(classical, dtype=float)
        diff = quantum - classical

        plt.figure(figsize=(10, 9))

        # Calculate common y-axis limits for both quantum and classical plots
        y_min = min(np.min(quantum), np.min(classical))
        y_max = max(np.max(quantum), np.max(classical))
        y_margin = (y_max - y_min) * 0.05  # 5% margin
        y_limits = [y_min - y_margin, y_max + y_margin]

        plt.subplot(3, 1, 1)
        plt.plot(t, quantum, label="Kwantowa ⟨x⟩")
        plt.xlabel("t")
        plt.ylabel("⟨x⟩")
        plt.ylim(y_limits)
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(t, classical, label="Klasyczna x(t)")
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.ylim(y_limits)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(t, diff, label="Różnica", )
        plt.xlabel("t")
        plt.ylabel("⟨x⟩ - x(t)")
        plt.legend()

        plt.tight_layout()
        plt.savefig("results/plot.png")
