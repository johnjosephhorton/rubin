import matplotlib.pyplot as plt
import numpy as np


def create_title_with_worker_assignments(W):
    """
    Create a title string showing task sequence grouped by workers
    Example: Job Design [(1)(2)][(3)(4)(5)] for 5 tasks split between 2 workers
    """
    # Initialize variables
    current_worker = W[0]
    groups = []
    current_group = []

    # Group tasks by worker
    for task_idx, worker in enumerate(W, 1):
        if worker != current_worker:
            groups.append(current_group)
            current_group = []
            current_worker = worker
        current_group.append(str(task_idx))

    # Add the last group
    groups.append(current_group)

    # Create the formatted string
    worker_sections = [
        "[" + "".join(f"({task})" for task in group) + "]" for group in groups
    ]
    return "Job Design " + "".join(worker_sections)


def draw_rect_square_unit(ax, x, y, t, c, h, task_idx):
    """
    Draw a single rectangle-square unit at position (x,y)
    Parameters:
    - ax: matplotlib axis
    - x, y: starting coordinates
    - t: length of rectangle
    - c: height of rectangle
    - h: area of square to drop down
    - task_idx: task index to display in rectangle center
    Returns:
    - next_x, next_y: coordinates for the next unit
    - coords: list of important coordinates for boundary calculation
    """
    # Draw rectangle
    rect = plt.Rectangle((x, y), t, c, fill=False, color="blue", linewidth=2)
    ax.add_patch(rect)

    # Add task index at center of rectangle
    center_x = x + t / 2
    center_y = y + c / 2
    ax.text(
        center_x,
        center_y,
        str(task_idx),
        horizontalalignment="center",
        verticalalignment="center",
    )

    # Calculate and draw square
    square_side = np.sqrt(h)
    square_x = x + t
    square = plt.Rectangle(
        (x + t, y + c),
        square_side,
        -square_side,
        fill=False,
        color="#FF9999",  # Lighter red
        linestyle="--",
        linewidth=1,
    )
    ax.add_patch(square)

    # Return next starting point and coordinates for bounds
    return (x + t, y + c), [
        (x + t, y + c),  # Upper right of rectangle
        (square_x + square_side, y - square_side),  # Lower right of square
    ]


def draw_rect_square_sequence(T, C, H, W):
    """
    Draw a sequence of rectangle-square units where:
    - T: lengths of rectangles
    - C: heights of rectangles
    - H: areas of squares
    - W: worker assignments
    """
    if not (len(T) == len(C) == len(H) == len(W)):
        raise ValueError("All input vectors must have the same length")

    fig, ax = plt.subplots(figsize=(10, 8))
    current_pos = (0, 0)
    all_coords = [(0, 0)]

    # Keep track of all x positions and corresponding workers for labeling
    x_positions = [0]  # Include start position
    worker_sections = []  # Store (start_x, start_y, end_x, worker_id) tuples
    current_section_start_x = 0
    current_section_start_y = 0
    current_worker = W[0]
    current_worker_T = []  # Track T values for current worker
    current_worker_C = []  # Track C values for current worker
    transition_squares = []  # Store (x, y, h) for transition squares

    # Draw each unit in sequence
    for i, (t, c, h, w) in enumerate(zip(T, C, H, W)):
        next_pos, new_coords = draw_rect_square_unit(
            ax,
            current_pos[0],
            current_pos[1],
            t,
            c,
            h,
            i + 1,  # Pass task index (1-based)
        )

        # Accumulate T and C values for current worker
        current_worker_T.append(t)
        current_worker_C.append(c)

        # If worker changes or this is the last task
        if (i < len(T) - 1 and w != W[i + 1]) or i == len(T) - 1:
            # Store transition square info if this is a worker change
            if i < len(T) - 1 and w != W[i + 1]:
                transition_squares.append((current_pos[0] + t, current_pos[1] + c, h))

            # Store section info including T and C sums
            worker_sections.append(
                (
                    current_section_start_x,
                    current_section_start_y,
                    next_pos[0],
                    current_worker,
                    sum(current_worker_T),
                    sum(current_worker_C),
                )
            )
            x_positions.append(next_pos[0])
            if i < len(T) - 1:  # If not the last task
                current_section_start_x = next_pos[0]
                current_section_start_y = next_pos[1]
                current_worker = W[i + 1]
                current_worker_T = []  # Reset for next worker
                current_worker_C = []  # Reset for next worker

        current_pos = next_pos
        all_coords.extend(new_coords)

    # Calculate plot bounds with padding
    all_coords = np.array(all_coords)
    x_min, y_min = all_coords.min(axis=0)
    x_max, y_max = all_coords.max(axis=0)

    # Draw vertical lines at all worker transitions
    for x in x_positions:
        ax.axvline(x=x, color="gray", linestyle="--", alpha=0.7)

    # Draw worker bounding boxes and labels
    max_y = y_max + (y_max - y_min) * 0.1  # Position for labels
    for start_x, start_y, end_x, worker_id, total_t, total_c in worker_sections:
        # Draw worker bounding box
        worker_box = plt.Rectangle(
            (start_x, start_y),  # Start at worker's first task position
            total_t,  # Width is sum of T values
            total_c,  # Height is sum of C values
            fill=True,
            facecolor="lightblue",
            edgecolor="blue",
            linestyle="--",
            linewidth=1,
            alpha=0.2,
        )
        ax.add_patch(worker_box)

        # Add worker label
        center_x = (start_x + end_x) / 2
        ax.text(
            center_x,
            max_y,
            f"Worker {worker_id}",
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    # Draw transition squares with pink fill
    for x, y, h in transition_squares:
        square_side = np.sqrt(h)
        handoff_square = plt.Rectangle(
            (x, y),
            square_side,
            -square_side,
            fill=True,
            facecolor="pink",
            edgecolor="#FF9999",  # Lighter red
            linestyle="--",
            linewidth=1,
            alpha=0.3,
        )
        ax.add_patch(handoff_square)

    padding = max(x_max - x_min, y_max - y_min) * 0.1
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, max_y + padding)

    ax.set_aspect("equal")

    # Set the new title using the worker assignment function
    ax.set_title(create_title_with_worker_assignments(W))

    # Remove grid but keep the legend
    ax.grid(False)
    ax.legend()

    return fig, ax


# Example usage
T = np.array([2, 4, 2, 6, 7, 1])  # Rectangle lengths
C = np.array([2, 5, 1, 3, 4, 2])  # Rectangle heights
H = np.array([1, 4, 4, 1, 1, 0.5])  # Square areas
W = np.array([1, 1, 1, 2, 2, 3])  # Worker assignments

fig, ax = draw_rect_square_sequence(T, C, H, W)
# plt.show()

plt.savefig("../writeup/plots/job_design.png")
