#!/usr/bin/env python
# coding: utf-8

# #### Task-sequence visualizations under frequency pruning
# #### By: Peyman Shahidi
# #### Created: Jul 20, 2026
#
# Replicates the per-occupation task-sequence plots of
# `onet_taskSequence_vs_antrhopicIndex.ipynb` (saved in
# writeup/plots/taskSequence_vs_anthropicIndex/all_occupation_task_sequences)
# for the frequency-pruned workflows used in the frequency-robustness appendix.
#
# Cuts: threshold = 50% of incumbents, three filtering logics
#   Daily+        (FT 5-7: Daily, Several times daily, Hourly or more)
#   SeveralDaily+ (FT 6-7: Several times daily, Hourly or more)
#   Hourly+       (FT 7:   Hourly or more)
#
# Sample rule matches the appendix's shared definition: an occupation is kept
# if its frequency-pruned workflow retains >= MIN_TASKS_PER_OCC (5) tasks.
# Surviving tasks keep their original Task Position numbers, so gaps in the
# left-hand numbering show exactly which steps were pruned.

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import warnings
warnings.filterwarnings('ignore')

# ====================== PARAMETERS ======================
FREQ_THRESHOLD = 50.0
MIN_TASKS_PER_OCC = 5

FAMILIES = [
    ('Daily+',        'daily',    ['FT_Daily', 'FT_Several times daily', 'FT_Hourly or more']),
    ('SeveralDaily+', 'sevdaily', ['FT_Several times daily', 'FT_Hourly or more']),
    ('Hourly+',       'hourly',   ['FT_Hourly or more']),
]

main_folder_path = ".."
input_file_path = f"{main_folder_path}/data/computed_objects/ONET_Eloundou_Anthropic_GPT/ONET_Eloundou_Anthropic_GPT.csv"
output_plot_path = f"{main_folder_path}/writeup/plots/taskSequence_vs_anthropicIndex"


def plot_task_sequence(occ_data, occ_code, cut_note, title_max_length=120):
    """Plot task sequence for one occupation with color-coded rectangles.
    Same style as the all-tasks version; occ_data is already pruned/sorted."""
    color_map = {
        'Manual': 'lightgray',
        'Augmentation': 'orange',
        'Automation': 'green'
    }

    fig, ax = plt.subplots(figsize=(12, max(6, len(occ_data) * 0.45)))

    for i, (idx, row) in enumerate(occ_data.iterrows()):
        y_pos = len(occ_data) - i - 1  # Start from top

        rect = Rectangle((0, y_pos), 5, 0.8,
                         facecolor=color_map.get(row['label'], 'lightgray'),
                         edgecolor='black',
                         linewidth=1)
        ax.add_patch(rect)

        task_title = str(row['Task Title'])
        if len(task_title) > title_max_length:
            task_title = task_title[:title_max_length] + "..."

        # Original workflow position: gaps reveal the pruned steps
        ax.text(-0.1, y_pos + 0.4, f"{int(row['Task Position'])}",
                ha='right', va='center', fontweight='bold', fontsize=10)

        ax.text(0.05, y_pos + 0.4, task_title,
                ha='left', va='center', fontsize=9, wrap=True)

        ax.text(5.1, y_pos + 0.4, row['label'],
                ha='left', va='center', fontweight='bold', fontsize=9,
                color=color_map.get(row['label'], 'black'))

    ax.set_xlim(-1, 6)
    ax.set_ylim(-0.5, len(occ_data) - 0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ['top', 'right', 'bottom', 'left']:
        ax.spines[side].set_visible(False)

    occupation_title = occ_data['Occupation Title'].iloc[0]
    plt.title(f"Task Sequence for {occupation_title}\n({occ_code} | {cut_note})",
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


merged_data = pd.read_csv(input_file_path)
merged_data['Task Position'] = pd.to_numeric(merged_data['Task Position'], errors='coerce')

for family_label, family_tag, family_cols in FAMILIES:
    pruned = merged_data[merged_data[family_cols].sum(axis=1) >= FREQ_THRESHOLD].copy()
    pruned = pruned[pruned['Task Position'].notna()]

    # Shared sample rule: keep occupations with >= MIN_TASKS_PER_OCC surviving tasks
    counts = pruned.groupby('O*NET-SOC Code')['Task ID'].nunique()
    valid = set(counts[counts >= MIN_TASKS_PER_OCC].index)
    pruned = pruned[pruned['O*NET-SOC Code'].isin(valid)].reset_index(drop=True)

    # Total (unpruned) task counts, for the "kept X of Y" note
    total_counts = merged_data[merged_data['O*NET-SOC Code'].isin(valid)] \
        .groupby('O*NET-SOC Code')['Task ID'].nunique().to_dict()

    output_folder = os.path.join(output_plot_path,
                                 f"all_occupation_task_sequences_{family_tag}{int(FREQ_THRESHOLD)}")
    os.makedirs(output_folder, exist_ok=True)

    occupations = sorted(pruned['O*NET-SOC Code'].unique())
    print(f"[{family_label} >={int(FREQ_THRESHOLD)}%] {len(occupations)} occupations -> {output_folder}",
          flush=True)

    for i, occ_code in enumerate(occupations):
        occ_data = pruned[pruned['O*NET-SOC Code'] == occ_code].sort_values('Task Position')
        n_kept = occ_data['Task ID'].nunique()
        n_total = total_counts.get(occ_code, n_kept)
        cut_note = f"{family_label} >={int(FREQ_THRESHOLD)}% tasks: kept {n_kept} of {n_total}"

        fig = plot_task_sequence(occ_data, occ_code, cut_note)

        occ_title = occ_data['Occupation Title'].iloc[0]
        safe_title = ''.join(c if (c.isalnum() or c in (' ', '_', '-')) else '_' for c in occ_title) \
            .replace(' ', '_')[:120]
        filename = os.path.join(output_folder, f"task_sequence_{occ_code}_{safe_title}.png")
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(occupations)}", flush=True)

    print(f"[{family_label} >={int(FREQ_THRESHOLD)}%] done: {len(occupations)} plots saved", flush=True)

print("All cuts complete.", flush=True)
