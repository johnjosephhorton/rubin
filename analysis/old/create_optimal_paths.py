from CostVisualization import CostVisualization

m = 300

# Create visualization instance
viz = CostVisualization(t_m=1.0, t_h=3)

starting_points = [
    (0.1, 0.9),
    (0.1, 0.1),  # Point 1
    (0.3, 0.2),  # Point 2
    (0.2, 0.4),  # Point 3
    (0.4, 0.3),  # Point 4
    (0.7, 0.6),
    (0.7, 0.2),
]

time_horizons = [int(m / 100) * x for x in [100] * len(starting_points)]

# Plot all paths with different time horizons
viz.plot_optimal_paths(starting_points, time_horizons, m=m)

# Plot all paths
# viz.plot_optimal_paths(starting_points, T=150, m=100)

# Save the result
viz.save_figure("../writeup/plots/optimal_paths.pdf")
