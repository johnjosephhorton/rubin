import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
from pathlib import Path

# Create output path if doesn't exist
path = Path("../writeup/plots/job_design")
path.mkdir(parents=True, exist_ok=True)


def create_title_with_worker_assignments(W):
    """
    Create a title string showing task sequence grouped by workers
    Example: Job Design [(1)(2)][(3)(4)][(5)(6)] for 6 tasks split between 3 workers
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
    Draw a single rectangle unit with an attached rectangle (formerly a square)
    at position (x,y). The main rectangle has width t and height c. The attached
    rectangle is drawn as a rectangle with width h and pre-determined hand-off height.
    
    Parameters:
    - ax: matplotlib axis
    - x, y: starting coordinates
    - t: length of main rectangle
    - c: height of main rectangle
    - h: width of the attached rectangle (height fixed at hand-off height)
    - task_idx: task index to display in main rectangle center
    
    Returns:
    - next_x, next_y: coordinates for the next unit
    - coords: list of important coordinates for boundary calculation
    """
    # Draw main rectangle (task)
    rect = plt.Rectangle((x, y), t, c, fill=False, color="blue", linewidth=1)
    ax.add_patch(rect)

    # Add task index at center of main rectangle
    center_x = x + t / 2
    center_y = y + c / 2
    ax.text(
        center_x,
        center_y,
        str(task_idx),
        horizontalalignment="center",
        verticalalignment="center",
    )

    # Draw attached rectangle (formerly square) with fixed height and width h.
    # We position it so that its top edge aligns with the main rectangle's top edge.
    attached_x = x + t
    attached_top_y = y + c
    attached_rect = plt.Rectangle(
        (attached_x, attached_top_y - handoff_height),  # lower left corner (drop down by handoff_height)
        h,  # width given directly by h
        handoff_height,  # fixed height of handoff_height
        fill=False,
        color="#FF9999",  # Lighter red
        linestyle="--",
        linewidth=1,
    )
    ax.add_patch(attached_rect)

    # Return next starting point and coordinates for bounds.
    # The new "end" is at the top right of the attached rectangle.
    next_pos = (attached_x + h, attached_top_y)
    coords = [
        (attached_x, attached_top_y),             # junction of main and attached rect.
        (attached_x + h, attached_top_y - handoff_height),       # lower right of attached rect.
    ]
    return next_pos, coords


def draw_rect_square_sequence(T, C, H, W):
    """
    Draw a sequence of rectangle units where each task is represented as:
      - A main rectangle (dimensions from T and C)
      - An attached rectangle with handoff_height and width from H.
    The tasks are grouped by worker assignments (W). Additionally, whenever the worker
    doesn't change from one task to the next, the starting x position for the next task
    is shifted left by the current task's h.
    """
    if not (len(T) == len(C) == len(H) == len(W)):
        raise ValueError("All input vectors must have the same length")

    fig, ax = plt.subplots(figsize=(10, 8))
    current_pos = (0, 0)
    all_coords = [(0, 0)]

    # Keep track of all x positions and corresponding workers for labeling
    x_positions = [0]  # Include start position
    worker_sections = []  # Store (start_x, start_y, end_x, worker_id, total_T, total_C) tuples
    current_section_start_x = 0
    current_section_start_y = 0
    current_worker = W[0]
    current_worker_T = []  # Track T values for current worker
    current_worker_C = []  # Track C values for current worker
    transition_rects = []  # Store (x, y, h, total_c) for transition attached rectangles

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

        # If the worker doesn't change, shift next_pos to the left by h.
        if i < len(T) - 1 and w == W[i + 1]:
            next_pos = (next_pos[0] - h, next_pos[1])

        # Accumulate T and C values for current worker (for bounding box dimensions)
        current_worker_T.append(t)
        current_worker_C.append(c)

        # If worker changes or this is the last task, record section info.
        if (i < len(T) - 1 and w != W[i + 1]) or i == len(T) - 1:
            # Store transition rectangle info if this is a worker change.
            if i < len(T) - 1 and w != W[i + 1]:
                transition_rects.append((current_pos[0] + t, current_pos[1] + c, h, sum(current_worker_C), i + 1))

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
            if i < len(T) - 1:  # If not the last task, reset for next worker.
                current_section_start_x = next_pos[0]
                current_section_start_y = next_pos[1]
                current_worker = W[i + 1]
                current_worker_T = []
                current_worker_C = []

        current_pos = next_pos
        all_coords.extend(new_coords)

    # Calculate plot bounds with padding.
    all_coords = np.array(all_coords)
    x_min, y_min = all_coords.min(axis=0)
    x_max, y_max = all_coords.max(axis=0)

    # Draw worker bounding boxes.
    max_y = y_max + (y_max - y_min) * 0.1  # Position for labels
    for start_x, start_y, end_x, worker_id, total_t, total_c in worker_sections:
        worker_box = plt.Rectangle(
            (start_x, start_y),
            total_t,  # Total width of tasks for the worker (excluding attached rectangles)
            total_c,  # Total height of tasks for the worker
            fill=True,
            facecolor="blue",
            edgecolor="blue",
            linestyle="--",
            linewidth=1,
            alpha=0.2,
        )
        ax.add_patch(worker_box)
        # Optionally, add worker labels (currently commented out)
        # center_x = (start_x + end_x) / 2
        # ax.text(center_x, max_y, f"Worker {worker_id}", horizontalalignment="center", verticalalignment="bottom")

    # Draw transition attached rectangles with pink fill.
    for x, y, h, total_c, task_idx in transition_rects:
        transition_rect = plt.Rectangle(
            (x, y - total_c),  # lower left corner (positioned relative to worker section)
            h,  # width from h
            total_c,  # height as accumulated for the worker's section
            fill=True,
            facecolor="pink",
            edgecolor="#FF9999",
            linestyle="--",
            linewidth=1,
            alpha=0.4,
        )
        ax.add_patch(transition_rect)

        # Add hand-off index at center of hand-off rectangle
        center_x = x + h / 2
        center_y = y - total_c / 2
        ax.text(
            center_x,
            center_y,
            f"$h_{{{task_idx}}}$",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=10
        )

    padding = max(x_max - x_min, y_max - y_min) * 0.1
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, max_y + padding)
    ax.set_aspect("equal")

    # Set the title using the worker assignment function.
    ax.set_title(create_title_with_worker_assignments(W))

    # Remove grid and ticks.
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def generate_worker_assignments(n):
    def generate_recursive(pos, prev_assignment):
        if pos == n:
            yield prev_assignment[:]
            return

        # Use same worker as previous position
        prev_assignment[pos] = prev_assignment[pos - 1] if pos > 0 else 1
        yield from generate_recursive(pos + 1, prev_assignment)

        # Use new worker
        prev_assignment[pos] = max(prev_assignment[:pos], default=0) + 1
        yield from generate_recursive(pos + 1, prev_assignment)

    yield from generate_recursive(0, [0] * n)


# Example usage
handoff_height = 0.15  # Fixed height for attached rectangles
T = np.array([2, 4, 5, 2, 7, 4])  # Main rectangle lengths
C = np.array([2, 5, 6, 3, 4, 1])  # Main rectangle heights
H = np.array([5, 2, 8, 2, 6, 0])  # Attached rectangle widths (height fixed at 0.1)


image_files = []
for index, assignment in enumerate(generate_worker_assignments(6)):
    W = np.array(assignment)
    # print(W)
    fig, ax = draw_rect_square_sequence(T, C, H, W)
    filename = f"../writeup/plots/job_design/job_design_{index}.png"
    image_files.append(filename)
    plt.savefig(filename, dpi=100)


# Combine PNGs into a grid
grid_size = math.ceil(math.sqrt(len(image_files)))
images = [Image.open(img) for img in image_files]
img_width, img_height = images[0].size
canvas_width = img_width * grid_size
canvas_height = img_height * grid_size
grid_image = Image.new("RGB", (canvas_width, canvas_height), "white")

for index, img in enumerate(images):
    x = (index % grid_size) * img_width
    y = (index // grid_size) * img_height
    grid_image.paste(img, (x, y))

grid_image.save("../writeup/plots/combined_grid.png")
