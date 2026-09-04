"""Re-render Figure SA.E.3 (neighbor AME heatmaps across frequency cuts) under a chosen
exposure mask.

Plotting code copied VERBATIM out of analysis/onet_neighborAI_weeklyTasks.ipynb cells 17-18
(functions `_cell`, `heatmap`, the NB_SPECS / SPEC_TITLE_HM / TERM_TAG loop, figsize, cmap,
TwoSlopeNorm, annotation font sizes, axhline, tight_layout, bbox_inches).

The ONLY thing that differs between the two renders is the sweep table fed in, which itself
differs only by  human_labels.isin(['E1'])  ->  human_labels.isin(['E1','E2']).

Sweep inputs:
  E1    : data/computed_objects/.../neighbor_logic_threshold_sweep.csv   (the published file)
  E1|E2 : the audited rebuild sae3_sweep_E1_vs_E1E2.csv (mask == 'E1E2'), which reproduces
          the published E1 side to 1.9e-16 on all 144 cells with identical nobs.

Writes ONLY under writeup/_e1e2_preview/.
Usage: python3 render_sae3_heatmaps.py
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

REPO = "/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin"
STAGE = f"{REPO}/writeup/_e1e2_preview"
M6 = ("/private/tmp/claude-502/-Users-peymansh-MIT-Dropbox-Peyman-Shahidi-GitHub-rubin/"
      "5af7d286-51e7-470a-b57e-4e1373499eea/scratchpad/m6")
PUB_SWEEP = f"{REPO}/data/computed_objects/execTypeVaryingDWA_anthropicIndex_weeklyTasks/neighbor_logic_threshold_sweep.csv"

# ---- notebook cell 2 / 11 constants ----
SWEEP_THRESHOLDS = [20, 35, 50, 65]
star = lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''

# ---- notebook cell 17 ----
fam_order = ['daily', 'sevdaily', 'hourly']
fam_label = {'daily': 'Daily+', 'sevdaily': 'SeveralDaily+', 'hourly': 'Hourly+'}
row_order = ['all'] + fam_order
row_label = {'all': 'All tasks', **fam_label}

def _cell(sub, fam, t):
    x = sub[sub['family'] == 'all'] if fam == 'all' else sub[(sub['family'] == fam) & (sub['threshold'] == t)]
    if len(x):
        return x['ame'].iloc[0], x['pval'].iloc[0], int(x['nobs'].iloc[0])
    return None

NB_SPECS = [('no_fe_no_dwa', 'Baseline (no FE)', 'baseline'),
            ('major_fe_no_dwa', 'Major SOC group FE', 'majorFE'),
            ('minor_fe_no_dwa', 'Minor SOC group FE', 'minorFE')]

# ---- notebook cell 18 ----
def heatmap(ax, sweep, spec, term, title):
    sub = sweep[(sweep['spec'] == spec) & (sweep['term'] == term)]
    M = np.full((len(row_order), len(SWEEP_THRESHOLDS)), np.nan)
    ann = [['—' for _ in SWEEP_THRESHOLDS] for _ in row_order]
    for i, fam in enumerate(row_order):
        for j, t in enumerate(SWEEP_THRESHOLDS):
            c = _cell(sub, fam, t)
            if c:
                M[i, j] = c[0]; ann[i][j] = f"{c[0]:.3f}{star(c[1])}\nN={c[2]}"
    vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    ax.imshow(M, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect='auto')
    ax.set_xticks(range(len(SWEEP_THRESHOLDS))); ax.set_xticklabels([f"≥{t}%" for t in SWEEP_THRESHOLDS])
    ax.set_yticks(range(len(row_order))); ax.set_yticklabels([row_label[f] for f in row_order])
    ax.axhline(0.5, color='black', lw=2)
    for i in range(len(row_order)):
        for j in range(len(SWEEP_THRESHOLDS)):
            ax.text(j, i, ann[i][j], ha='center', va='center', fontsize=7.5,
                    color='black' if (np.isnan(M[i, j]) or abs(M[i, j]) < 0.6 * vmax) else 'white')
    ax.set_title(title, fontsize=10.5); ax.set_xlabel('Threshold')

SPEC_TITLE_HM = {'baseline': 'No Fixed Effects', 'majorFE': 'SOC Major Group FE', 'minorFE': 'SOC Minor Group FE'}
TERM_TAG = {'prev_is_ai': 'prev', 'next_is_ai': 'next'}   # only these two are \includegraphics-ed


def render(sweep, outdir, dpi):
    os.makedirs(outdir, exist_ok=True)
    made = []
    for term, term_tag in TERM_TAG.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for ax, (spec_code, spec_name, spec_tag) in zip(axes, NB_SPECS):
            heatmap(ax, sweep, spec_code, term, SPEC_TITLE_HM[spec_tag])
        fig.tight_layout()
        f = f"{outdir}/neighbor_logic_threshold_heatmap_{term_tag}_bySpec.png"
        fig.savefig(f, dpi=dpi, bbox_inches='tight'); plt.close(fig)
        made.append(f)
    return made


if __name__ == '__main__':
    pub = pd.read_csv(PUB_SWEEP)
    reb = pd.read_csv(f"{M6}/sae3_sweep_E1_vs_E1E2.csv")
    e1 = reb[reb['mask'] == 'E1'].drop(columns=['mask']).reset_index(drop=True)
    e12 = reb[reb['mask'] == 'E1E2'].drop(columns=['mask']).reset_index(drop=True)

    # fixture: the audited E1 rebuild must equal the published sweep table
    m = pub.merge(e1, on=['cut', 'family', 'threshold', 'spec', 'term'], suffixes=('_p', '_r'))
    assert len(m) == len(pub) == 144, (len(m), len(pub))
    print(f"[fixture] E1 rebuild vs published sweep: max|dAME|={np.abs(m.ame_p - m.ame_r).max():.2e}  "
          f"max|dp|={np.abs(m.pval_p - m.pval_r).max():.2e}  nobs identical={bool((m.nobs_p == m.nobs_r).all())}")

    # (1) published-side re-render from the published CSV, at the notebook's own dpi=200
    render(pub, f"{STAGE}/plots_fixture/E1_from_published_csv_dpi200", dpi=200)
    # (2) E1|E2 at the notebook's own dpi=200 (pixel-comparable to the published PNG)
    render(e12, f"{STAGE}/plots_fixture/E1E2_dpi200", dpi=200)
    # (3) the deliverable: E1|E2 at dpi=300
    out = render(e12, f"{STAGE}/plots_new", dpi=300)
    for f in out:
        print("wrote", f)
