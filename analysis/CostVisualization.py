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
        self.current_fig = None

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

    def build_cost_table(self, m):
        """Build discretized cost table for dynamic programming."""
        cost_table = np.zeros((m + 1, m + 1))
        for i in range(m + 1):
            for j in range(m + 1):
                if i == 0 or j == 0:
                    cost_table[i, j] = 2 * self.c_h
                else:
                    q1 = i / m
                    q2 = j / m
                    # Calculate costs for single point
                    cost_human = 2 * self.c_h
                    cost_machine = (self.c_m / q1) + (self.c_m / q2)
                    cost_m1h2 = (self.c_m / q1) + self.c_h
                    cost_h1m2 = self.c_h + (self.c_m / q2)
                    cost_chained = self.c_m / (q1 * q2)

                    cost_table[i, j] = min(
                        cost_human, cost_machine, cost_m1h2, cost_h1m2, cost_chained
                    )
        return cost_table

    def find_optimal_path(self, q1_start, q2_start, T, m=100):
        """Find optimal path from starting point using dynamic programming.

        Args:
            q1_start (float): Starting q1 value between 0 and 1
            q2_start (float): Starting q2 value between 0 and 1
            T (int): Number of time periods
            m (int): Grid resolution for discretization

        Returns:
            tuple: (minimal cost, list of (q1, q2) coordinates for optimal path)
        """
        # Convert continuous coordinates to grid coordinates
        i0 = int(round(q1_start * m))
        j0 = int(round(q2_start * m))

        # Build cost table
        cost_table = self.build_cost_table(m)

        # Initialize DP arrays
        V = np.zeros((T + 1, m + 1, m + 1))
        action = [
            [["NONE" for _ in range(m + 1)] for _ in range(m + 1)] for _ in range(T + 1)
        ]

        # Backward fill
        for t in reversed(range(T)):
            for i in range(m + 1):
                for j in range(m + 1):
                    immediate_cost = cost_table[i, j]
                    candidates = []

                    if i < m:
                        candidates.append(("Q1", V[t + 1, i + 1, j]))
                    if j < m:
                        candidates.append(("Q2", V[t + 1, i, j + 1]))

                    if len(candidates) == 0:
                        V[t, i, j] = immediate_cost
                        action[t][i][j] = "NONE"
                    else:
                        best_act, best_val = min(candidates, key=lambda x: x[1])
                        V[t, i, j] = immediate_cost + best_val
                        action[t][i][j] = best_act

        # Reconstruct path
        path = [(i0 / m, j0 / m)]
        i_curr, j_curr = i0, j0

        for t in range(T):
            a = action[t][i_curr][j_curr]
            if a == "Q1" and i_curr < m:
                i_curr += 1
            elif a == "Q2" and j_curr < m:
                j_curr += 1
            path.append((i_curr / m, j_curr / m))

        return V[0, i0, j0], path

    def create_visualization(self, resolution=300, figsize=(10, 10)):
        """Create the base visualization."""
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
        min_val = min_cost.min()
        max_val = min_cost.max()

        # Handle case where all values are the same
        if np.isclose(min_val, max_val):
            levels = np.linspace(min_val - 0.1, min_val + 0.1, 20)
        else:
            levels = np.linspace(min_val, max_val + 1e-10, 20)

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
        plt.tight_layout(pad=1.5)

        # Store the current figure
        self.current_fig = plt.gcf()

    def plot_optimal_paths(
        self,
        starting_points,
        time_horizons,
        m=100,
        marker_size=100,
        colors=None,
        show_labels=True,
    ):
        """Plot multiple optimal paths from different starting points with different time horizons.

        Args:
            starting_points (list): List of (q1, q2) tuples for starting points
            time_horizons (list): List of T values (time horizons) for each starting point
            m (int): Grid resolution for discretization
            marker_size (float): Size of the markers
            colors (list, optional): List of colors for each path. If None, uses default color cycle
            show_labels (bool): Whether to show index labels at starting points
        """
        if self.current_fig is None:
            self.create_visualization()

        if len(starting_points) != len(time_horizons):
            raise ValueError(
                "Number of starting points must match number of time horizons"
            )

        if colors is None:
            # Use a colorblind-friendly palette
            colors = [
                "#E69F00",
                "#56B4E9",
                "#009E73",
                "#F0E442",
                "#0072B2",
                "#D55E00",
                "#CC79A7",
                "#000000",
            ]

        # Make sure we have enough colors
        while len(colors) < len(starting_points):
            colors.extend(colors)

        for idx, ((q1_start, q2_start), T, color) in enumerate(
            zip(starting_points, time_horizons, colors)
        ):
            # Find optimal path
            cost, path = self.find_optimal_path(q1_start, q2_start, T, m)

            # Extract coordinates
            q1_coords, q2_coords = zip(*path)

            # Plot path with transparency
            plt.plot(
                q1_coords,
                q2_coords,
                color=color,
                linestyle="-",
                alpha=0.6,
                linewidth=2,
                zorder=5,
                label=f"Path {idx+1} (T={T})",
            )

            # Add transparent markers along the path
            plt.scatter(
                q1_coords,
                q2_coords,
                color=color,
                alpha=0.4,
                s=marker_size / 2,
                zorder=5,
            )

            if show_labels:
                # Add numbered label at starting point with white background
                plt.scatter(
                    [q1_start],
                    [q2_start],
                    color=color,
                    alpha=0.8,
                    s=marker_size,
                    zorder=6,
                )
                plt.text(
                    q1_start,
                    q2_start,
                    str(idx + 1),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white",
                    fontweight="bold",
                    zorder=7,
                )

        if show_labels:
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    def save_figure(self, filepath):
        """Save the current figure to a file."""
        if self.current_fig is None:
            self.create_visualization()
        plt.savefig(filepath)
        plt.close()
        self.current_fig = None


if __name__ == "__main__":
    # Create visualization instance
    viz = CostVisualization(c_m=1.0, c_h=3.0)

    # Example usage:
    # Option 1: Just the base visualization
    viz.create_visualization()
    viz.save_figure("cost_visualization.pdf")

    # Option 2: With optimal paths
    viz = CostVisualization(c_m=1.0, c_h=3.0)
    starting_points = [(0.1, 0.1), (0.3, 0.2), (0.2, 0.4)]
    time_horizons = [50, 100, 150]
    viz.plot_optimal_paths(starting_points, time_horizons)
    viz.save_figure("cost_visualization_with_paths.pdf")
