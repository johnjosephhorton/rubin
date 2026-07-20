"""
Paper figure for Appendix J: EFI Definition 1 (preferred exposure-based measure, E1|E2) fragmentation heatmap across the
frequency logic x threshold grid, with the three FE specifications (No FE /
SOC Major group / SOC Minor group) shown side by side.

Reads the sweep produced by onet_fragmentationIndex_weeklyTasks.ipynb
(frag_logic_threshold_sweep.csv) and reuses that notebook's heatmap styling.
Run after the notebook so the sweep CSV is up to date:
    python3 analysis/make_frag_def1_heatmap.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(BASE, "data/computed_objects/fragmentationIndex_weeklyTasks/frag_logic_threshold_sweep.csv")
OUT = os.path.join(BASE, "writeup/plots/fragmentationIndex_weeklyTasks/frag_logic_threshold_heatmap_def1.png")

sweep = pd.read_csv(SWEEP)
SWEEP_THRESHOLDS = [20, 35, 50, 65]
fam_order = ['daily', 'sevdaily', 'hourly']
fam_label = {'daily': 'Daily+', 'sevdaily': 'SeveralDaily+', 'hourly': 'Hourly+'}
row_order = ['all'] + fam_order
row_label = {'all': 'All tasks', **fam_label}
FE_SPECS = [('none', 'No Fixed Effects'), ('Major', 'SOC Major Group FE'), ('Minor', 'SOC Minor Group FE')]


def star(p):
    return '***' if (pd.notna(p) and p < .01) else '**' if (pd.notna(p) and p < .05) else '*' if (pd.notna(p) and p < .1) else ''


def _cell(sub, fam, t):
    x = sub[sub['family'] == 'all'] if fam == 'all' else sub[(sub['family'] == fam) & (sub['threshold'] == t)]
    if len(x) and pd.notna(x['coef'].iloc[0]):
        return x['coef'].iloc[0], x['pval'].iloc[0], int(x['N_occ'].iloc[0])
    return None


def heatmap(ax, fe, title):
    sub = sweep[(sweep['FI_def'] == 'v1') & (sweep['FE'] == fe)]  # v1 = paper Definition 1 (exposure E1|E2) after the July 2026 renumbering; rerun the sweep notebook before this script
    M = np.full((len(row_order), len(SWEEP_THRESHOLDS)), np.nan)
    ann = [['—' for _ in SWEEP_THRESHOLDS] for _ in row_order]
    for i, fam in enumerate(row_order):
        for j, t in enumerate(SWEEP_THRESHOLDS):
            c = _cell(sub, fam, t)
            if c:
                M[i, j] = c[0]
                ann[i][j] = f"{c[0]:.2f}{star(c[1])}\nN={c[2]}"
    vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    ax.imshow(M, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect='auto')
    ax.set_xticks(range(len(SWEEP_THRESHOLDS)))
    ax.set_xticklabels([f"≥{t}%" for t in SWEEP_THRESHOLDS])
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels([row_label[f] for f in row_order])
    ax.axhline(0.5, color='black', lw=2)  # separate the all-tasks baseline row
    for i in range(len(row_order)):
        for j in range(len(SWEEP_THRESHOLDS)):
            ax.text(j, i, ann[i][j], ha='center', va='center', fontsize=8,
                    color='black' if (np.isnan(M[i, j]) or abs(M[i, j]) < 0.6 * vmax) else 'white')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Threshold')


fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (fe_code, fe_name) in zip(axes, FE_SPECS):
    heatmap(ax, fe_code, fe_name)
fig.suptitle('Empirical fragmentation index (EFI Definition 1) by frequency logic × threshold',
             fontweight='bold', fontsize=12.5)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches='tight')
print("Saved", OUT)
