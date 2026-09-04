"""Regenerate the Figure SA.D.1 split panels.

Faithful to cell 12 of analysis/GPT_task_sequences_overlap.ipynb: same BINS,
same styling, same normalisation, dpi=300. Two changes carried in from the
notebook edits: the EFI legend no longer carries "(Definition 1)", and the
exposure split is the E1-or-E2 measure, whose panel therefore takes the
human_aiExposure_fraction filename the notebook now emits.
"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

REPO = sys.argv[1]
OUT  = sys.argv[2]
BINS = np.linspace(-1, 1, 21)

kend = pd.read_csv(f"{REPO}/data/computed_objects/tasks_sequences_robustness/data/computed_objects/"
                   f"GPT_task_sequences_overlap_analysis/GPT_task_sequences_kendall_results.csv")
occ  = pd.read_csv(f"{REPO}/data/computed_objects/fragmentationIndex/"
                   f"occupation_analysis_with_fragmentationIndex_def1.csv")
for v in ['fragmentation_index', 'ai_fraction', 'human_E1_fraction', 'human_aiExposure_fraction']:
    occ[f'{v}_above_median'] = (occ[v] >= occ[v].median()).astype(int)

summary = (kend.groupby("Occupation Title")["kendall_tau"].mean().reset_index(name="mean"))
summary = summary.merge(
    occ[['Occupation Title', 'fragmentation_index_above_median', 'ai_fraction_above_median',
         'human_E1_fraction_above_median', 'human_aiExposure_fraction_above_median']]
       .drop_duplicates(subset=['Occupation Title']),
    on="Occupation Title", how="left")

VARS = {
    "fragmentation_index_above_median": "EFI",
    "human_aiExposure_fraction_above_median": "AI Exposure",
    "ai_fraction_above_median": "AI Execution",
}

def plot(var, label, as_share):
    sc = summary.dropna(subset=[var])
    below, above = sc[sc[var] == 0]['mean'], sc[sc[var] == 1]['mean']
    mb, ma = below.mean(), above.mean()
    wt = (lambda s: np.ones(len(s)) / len(s)) if as_share else (lambda s: None)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(below, bins=BINS, weights=wt(below), alpha=0.6, color='steelblue',
            edgecolor='black', label=f'Below Median {label}\n(n={len(below)}, mean={mb:.2f})')
    ax.hist(above, bins=BINS, weights=wt(above), alpha=0.6, color='red',
            edgecolor='black', label=f'Above Median {label}\n(n={len(above)}, mean={ma:.2f})')
    ax.axvline(mb, color='steelblue', linestyle='--', linewidth=2)
    ax.axvline(ma, color='red', linestyle='--', linewidth=2)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("Mean Kendall τ (within Occupation)", fontsize=16)
    ax.set_ylabel("Share of Occupations" if as_share else "Number of Occupations", fontsize=16)
    if as_share:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(-1.02, 1.02)
    ax.legend(loc="upper left", fontsize=15)
    plt.tight_layout()
    suffix = "" if as_share else "_count"
    path = f"{OUT}/GPT_task_sequence_robustness_by_{var[:-13]}_distribution{suffix}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  wrote {path.split('/')[-1]}  n={len(below)}/{len(above)}  mean={mb:.3f}/{ma:.3f}")

for var, label in VARS.items():
    for as_share in (True, False):
        plot(var, label, as_share)
