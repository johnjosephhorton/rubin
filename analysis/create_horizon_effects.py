from CostVisualization import CostVisualization

# Create visualization instance
viz = CostVisualization(t_m=1.0, t_h=3)

starting_points = [(0.4, 0.2), (0.4, 0.19), (0.4, 0.18), (0.4, 0.17)]

time_horizons = [2 * x for x in [20, 40, 50, 80]]

# Plot all paths with different time horizons
viz.plot_optimal_paths(starting_points, time_horizons, m=200)

# Plot all paths
# viz.plot_optimal_paths(starting_points, T=150, m=100)

# Save the result
viz.save_figure("../writeup/plots/horizon_effects.pdf")
