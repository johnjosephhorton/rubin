import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math


# Define functions for creating titles, drawing sequences, etc.
def create_title_with_worker_assignments(W):
    """
    Create a title string showing task sequence grouped by workers
    Example: Job Design [(1)(2)][(3)(4)(5)] for 5 tasks split between 2 workers
    """
    current_worker = W[0]
    groups = []
    current_group = []
    for task_idx, worker in enumerate(W, 1):
        if worker != current_worker:
            groups.append(current_group)
            current_group = []
            current_worker = worker
        current_group.append(str(task_idx))
    groups.append(current_group)
    worker_sections = [
        "[" + "".join(f"({task})" for task in group) + "]" for group in groups
    ]
    return "Job Design " + "".join(worker_sections)


def draw_rect_square_unit(ax, x, y, t, c, h, task_idx):
    rect = plt.Rectangle((x, y), t, c, fill=False, color="blue", linewidth=1)
    ax.add_patch(rect)
    center_x = x + t / 2
    center_y = y + c / 2
    ax.text(
        center_x,
        center_y,
        str(task_idx),
        horizontalalignment="center",
        verticalalignment="center",
    )
    square_side = np.sqrt(h)
    square_x = x + t
    square = plt.Rectangle(
        (x + t, y + c),
        square_side,
        -square_side,
        fill=False,
        color="#FF9999",
        linestyle="--",
        linewidth=1,
    )
    ax.add_patch(square)
    return (x + t, y + c), [(x + t, y + c), (square_x + square_side, y - square_side)]


def draw_rect_square_sequence(T, C, H, W):
    if not (len(T) == len(C) == len(H) == len(W)):
        raise ValueError("All input vectors must have the same length")
    fig, ax = plt.subplots(figsize=(10, 8))
    current_pos = (0, 0)
    all_coords = [(0, 0)]
    x_positions = [0]
    worker_sections = []
    current_section_start_x = 0
    current_section_start_y = 0
    current_worker = W[0]
    current_worker_T = []
    current_worker_C = []
    transition_squares = []
    for i, (t, c, h, w) in enumerate(zip(T, C, H, W)):
        next_pos, new_coords = draw_rect_square_unit(
            ax, current_pos[0], current_pos[1], t, c, h, i + 1
        )
        current_worker_T.append(t)
        current_worker_C.append(c)
        if (i < len(T) - 1 and w != W[i + 1]) or i == len(T) - 1:
            if i < len(T) - 1 and w != W[i + 1]:
                transition_squares.append((current_pos[0] + t, current_pos[1] + c, h))
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
            if i < len(T) - 1:
                current_section_start_x = next_pos[0]
                current_section_start_y = next_pos[1]
                current_worker = W[i + 1]
                current_worker_T = []
                current_worker_C = []
        current_pos = next_pos
        all_coords.extend(new_coords)
    all_coords = np.array(all_coords)
    x_min, y_min = all_coords.min(axis=0)
    x_max, y_max = all_coords.max(axis=0)
    max_y = y_max + (y_max - y_min) * 0.1
    for start_x, start_y, end_x, worker_id, total_t, total_c in worker_sections:
        worker_box = plt.Rectangle(
            (start_x, start_y),
            total_t,
            total_c,
            fill=True,
            facecolor="blue",
            edgecolor="blue",
            linestyle="--",
            linewidth=1,
            alpha=0.2,
        )
        ax.add_patch(worker_box)
    for x, y, h in transition_squares:
        square_side = np.sqrt(h)
        handoff_square = plt.Rectangle(
            (x, y),
            square_side,
            -square_side,
            fill=True,
            facecolor="pink",
            edgecolor="#FF9999",
            linestyle="--",
            linewidth=1,
            alpha=0.4,
        )
        ax.add_patch(handoff_square)
    padding = max(x_max - x_min, y_max - y_min) * 0.1
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, max_y + padding)
    ax.set_aspect("equal")
    ax.set_title(create_title_with_worker_assignments(W))
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
T = np.array([2, 4, 2, 6, 7, 1])
C = np.array([2, 5, 1, 3, 4, 2])
H = np.array([1, 4, 4, 1, 1, 0.5])

image_files = []
for index, assignment in enumerate(generate_worker_assignments(6)):
    W = np.array(assignment)
    # print(W)
    fig, ax = draw_rect_square_sequence(T, C, H, W)
    filename = f"job_design_{index}.png"
    image_files.append(filename)
    plt.savefig(filename)

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
