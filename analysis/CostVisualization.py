import numpy as np
import matplotlib.pyplot as plt


class CostVisualization:
    def __init__(self, c_m, c_h):
        """Initialize with cost parameters.

        Args:
            c_m (float): Machine cost (normalized)
            c_h (float): Human cost (normalized)
        """
        self.c_m = c_m
        self.c_h = c_h
        self.q_threshold = c_m / c_h

    def _calculate_costs(self, Q1, Q2):
        """Calculate costs for different arrangements."""
        cost_human = 2 * self.c_h
        cost_machine = (self.c_m / Q1) + (self.c_m / Q2)
        cost_m1h2 = (self.c_m / Q1) + self.c_h
        cost_h1m2 = self.c_h + (self.c_m / Q2)
        cost_chained = self.c_m / (Q1 * Q2)

        return np.stack(
            [
                cost_human * np.ones_like(Q1),
                cost_machine,
                cost_m1h2,
                cost_h1m2,
                cost_chained,
            ]
        )

    def _add_label_with_background(self, x, y, text, fontsize=12):
        """Add text label with white background."""
        plt.text(
            x,
            y,
            text,
            fontsize=fontsize,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=3),
            horizontalalignment="center",
            verticalalignment="center",
        )

    def create_pdf(self, filepath, resolution=300, figsize=(10, 10)):
        """Create and save the visualization as a PDF.

        Args:
            filepath (str): Path where to save the PDF
            resolution (int): Number of points for q1 and q2 grid
            figsize (tuple): Figure size in inches (width, height)
        """
        # Create grid
        q1 = np.linspace(0.001, 1, resolution)
        q2 = np.linspace(0.001, 1, resolution)
        Q1, Q2 = np.meshgrid(q1, q2)

        # Calculate costs and find minimum
        costs = self._calculate_costs(Q1, Q2)
        min_cost = np.min(costs, axis=0)

        # Set up figure
        plt.figure(figsize=figsize)

        # Create contour plot
        levels = np.linspace(min_cost.min(), min_cost.max(), 20)
        plt.contourf(Q1, Q2, min_cost, levels=levels, cmap="Blues", alpha=0.5)
        plt.contour(Q1, Q2, min_cost, levels=levels, colors="k", alpha=0.2)

        # Add boundary lines
        start = self.q_threshold

        # Machine-Chained boundary
        q1_line = np.linspace(start, 1 - start, 200)
        q2_mc_chained = 1 - q1_line
        valid = (q2_mc_chained >= 0) & (q2_mc_chained <= 1)
        plt.plot(q1_line[valid], q2_mc_chained[valid], "white", lw=2)

        # M1H2-Chained boundary
        q1_line = np.linspace(1 - start, 1, 200)
        q2_m1h2_chained = self.c_m / (self.c_m + self.c_h * q1_line)
        valid = (q2_m1h2_chained > 0) & (q2_m1h2_chained <= 1)
        plt.plot(q1_line[valid], q2_m1h2_chained[valid], "white", lw=2)

        # H1M2-Chained boundary
        q1_line = np.linspace(0.001, start, 200)
        q2_h1m2_chained = self.c_m * (1 - q1_line) / (self.c_h * q1_line)
        valid = (q2_h1m2_chained > 0) & (q2_h1m2_chained <= 1)
        plt.plot(q1_line[valid], q2_h1m2_chained[valid], "white", lw=2)

        # Threshold lines
        plt.axhline(
            y=self.q_threshold,
            color="white",
            linestyle="-",
            alpha=1,
            lw=2,
            xmin=0,
            xmax=1 - start,
        )
        plt.axvline(
            x=self.q_threshold,
            color="white",
            linestyle="-",
            alpha=1,
            lw=2,
            ymin=0,
            ymax=1 - start,
        )

        # Add threshold labels
        plt.text(self.q_threshold + 0.01, 0.02, r"$q_1 = c_m/c_h$", rotation=90)
        plt.text(0.01, self.q_threshold + 0.01, r"$q_2 = c_m/c_h$")

        # Add region labels
        self._add_label_with_background(
            self.q_threshold / 2, self.q_threshold / 2, "(1)(2)"
        )
        self._add_label_with_background(
            self.q_threshold / 2, (1 + self.q_threshold) / 2, "(1)<2>"
        )
        self._add_label_with_background(
            (1 + self.q_threshold) / 2, self.q_threshold / 2, "<1>(2)"
        )
        self._add_label_with_background(0.4, 0.4, "<1><2>")
        self._add_label_with_background(1 - start, 1 - start, "<1|2>")

        # Final plot settings
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("$q_1$")
        plt.ylabel("$q_2$")
        plt.title("Figure 2: Cost minimization with two tasks")
        plt.tight_layout()

        # Save to PDF
        plt.savefig(filepath)
        plt.close()
