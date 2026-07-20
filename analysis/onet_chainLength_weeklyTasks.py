#!/usr/bin/env python3
"""
PREDICTION #1 (AI CHAIN LENGTH) robustness across frequency cuts.

The frequent-task counterpart of the main-text AI-chain-length analysis (analysis/onet_chainLength.ipynb).
An AI-EXECUTED step is any task labeled Augmentation OR Automation (we treat the two identically). An AI chain is
a maximal contiguous run of AI-executed steps in the occupation's workflow, and the statistic is the average chain
length, pooled across chains (chain_lengths['chain_length'].mean()), exactly as in the main exercise.

Deliverables, in the same manner as the fragmentation / neighbor robustness analyses:
  (1) HEATMAP of the observed average AI chain length across the logic x threshold grid,
  (2) FOREST comparing the observed average AI chain length, per cut, to its within-occupation position-reshuffle null.
Same cut grid, same >=5-frequent-step shared sample, same reshuffle seeds (42+i) as the other robustness pieces.
Run from the analysis/ directory.
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

# ====================== PARAMETERS ======================
N_RESHUFFLES      = int(os.environ.get('N_RESHUFFLES', 1000))   # 0 -> observed + heatmap only (skip null/forest)
SHUFFLE_SEED_BASE = 42
MIN_TASKS_PER_OCC = 5
group_cols = ['O*NET-SOC Code', 'Occupation Title']

FT_DAILY        = ['FT_Daily', 'FT_Several times daily', 'FT_Hourly or more']   # FT 5-7
FT_SEVERALDAILY = ['FT_Several times daily', 'FT_Hourly or more']               # FT 6-7
FT_HOURLY       = ['FT_Hourly or more']                                         # FT 7
FAMILIES = [('Daily+', 'daily', FT_DAILY), ('SeveralDaily+', 'sevdaily', FT_SEVERALDAILY), ('Hourly+', 'hourly', FT_HOURLY)]
SWEEP_THRESHOLDS = [20, 35, 50, 65]
CUTS = [('All tasks', 'all', 0, None)]
for lab, tag, cols in FAMILIES:
    for t in SWEEP_THRESHOLDS:
        CUTS.append((f"{lab} >={t}%", tag, t, cols))

# ====================== PATHS ======================
main_folder_path = ".."
input_data_path  = f"{main_folder_path}/data"
input_file_path  = f"{input_data_path}/computed_objects/ONET_Eloundou_Anthropic_GPT/ONET_Eloundou_Anthropic_GPT.csv"
output_data_path = f"{input_data_path}/computed_objects/fragmentationIndex_weeklyTasks"
output_plot_path = f"{main_folder_path}/writeup/plots/fragmentationIndex_weeklyTasks"
os.makedirs(output_data_path, exist_ok=True)
os.makedirs(output_plot_path, exist_ok=True)

POOL = pd.read_csv(input_file_path)[['O*NET-SOC Code', 'Occupation Title', 'Task ID', 'Task Position',
        'label'] + FT_DAILY].copy()
POOL['Task Position'] = pd.to_numeric(POOL['Task Position'], errors='coerce')

# ====================== CHAIN-LENGTH MEASURE (main-text "any contiguous AI-executed run") ======================
def mean_chain_length(df):
    """Pooled mean AI-chain length. AI-executed = label in {Augmentation, Automation}; a chain is a maximal
    contiguous run of AI-executed steps in current Task-Position order. NaN if the cut has no AI-executed steps."""
    f = df.sort_values(group_cols + ['Task Position']).copy()
    f['is_ai'] = f['label'].isin(['Augmentation', 'Automation']).astype(int)
    f['prev_is_ai'] = f.groupby(group_cols)['is_ai'].shift(1).fillna(0).astype(int)
    f['start_chain'] = ((f['is_ai'] == 1) & (f['prev_is_ai'] == 0)).astype(int)
    f['chain_id'] = f.groupby(group_cols)['start_chain'].cumsum()
    cl = f[f['is_ai'] == 1].groupby(group_cols + ['chain_id']).size()
    return cl.mean() if len(cl) else np.nan

# ---- validate against the main-text full-sample value (~1.45) ----
if N_RESHUFFLES == 0:
    full = pd.read_csv(input_file_path)[['O*NET-SOC Code', 'Occupation Title', 'Task Position', 'label']]
    print(f"VALIDATION (full unfiltered sample): mean AI chain length = {mean_chain_length(full):.3f}  (main ~1.45)")

# ====================== SAMPLE RULE + CUTS ======================
_VOS_CACHE = {}
def valid_occ_set(cols, thr):
    key = (None if cols is None else tuple(cols), thr)
    if key not in _VOS_CACHE:
        d = POOL if cols is None else POOL[POOL[cols].sum(axis=1) >= thr]
        cnt = d.groupby('O*NET-SOC Code')['Task ID'].nunique()
        _VOS_CACHE[key] = set(cnt[cnt >= MIN_TASKS_PER_OCC].index)
    return _VOS_CACHE[key]

def filter_cut(pool, cols, thr):
    d = pool if cols is None else pool[pool[cols].sum(axis=1) >= thr]
    return d[d['O*NET-SOC Code'].isin(valid_occ_set(cols, thr))]

def shuffle_pool(pool, seed):
    d = pool.copy()
    d['Task Position'] = d.groupby('O*NET-SOC Code')['Task Position'].transform(
        lambda x: x.sample(frac=1, random_state=seed).values)
    return d

# ====================== OBSERVED SWEEP (heatmap data) ======================
observed, n_occ, sweep_rows = {}, {}, []
for label, fam, thr, cols in CUTS:
    d = filter_cut(POOL, cols, thr)
    n_occ[label] = d['O*NET-SOC Code'].nunique()
    observed[label] = mean_chain_length(d)
    sweep_rows.append({'cut': label, 'family': fam, 'threshold': thr,
                       'mean_chain_length': observed[label], 'N_occ': n_occ[label]})
sweep = pd.DataFrame(sweep_rows)
sweep.to_csv(f"{output_data_path}/chainLength_logic_threshold_sweep.csv", index=False)
print("Saved chainLength_logic_threshold_sweep.csv")
print("Observed average AI chain length (AI-executed = Augmentation or Automation):")
for label, fam, thr, cols in CUTS:
    print(f"  {label:20} mean={observed[label]:.2f}  N_occ={n_occ[label]}")

# ====================== HEATMAP ======================
fam_order = ['all', 'daily', 'sevdaily', 'hourly']
fam_label = {'all': 'All tasks', 'daily': 'Daily+', 'sevdaily': 'SeveralDaily+', 'hourly': 'Hourly+'}
M = np.full((len(fam_order), len(SWEEP_THRESHOLDS)), np.nan)
ann = [['' for _ in SWEEP_THRESHOLDS] for _ in fam_order]
for i, fam in enumerate(fam_order):
    for j, t in enumerate(SWEEP_THRESHOLDS):
        sub = sweep[(sweep['family'] == 'all') if fam == 'all' else
                    ((sweep['family'] == fam) & (sweep['threshold'] == t))]
        if len(sub) and pd.notna(sub['mean_chain_length'].iloc[0]):
            v = sub['mean_chain_length'].iloc[0]; M[i, j] = v
            ann[i][j] = f"{v:.2f}\nN={int(sub['N_occ'].iloc[0])}"
vmin, vmax = np.nanmin(M), np.nanmax(M)
fig, ax = plt.subplots(figsize=(7.2, 5.2))
im = ax.imshow(M, cmap='Purples', norm=Normalize(vmin=vmin, vmax=vmax), aspect='auto')  # sequential palette distinct from the diverging RdBu used for EFI/neighbor heatmaps
ax.set_xticks(range(len(SWEEP_THRESHOLDS))); ax.set_xticklabels([f"≥{t}%" for t in SWEEP_THRESHOLDS])
ax.set_yticks(range(len(fam_order))); ax.set_yticklabels([fam_label[f] for f in fam_order])
ax.axhline(0.5, color='black', lw=2)
for i in range(len(fam_order)):
    for j in range(len(SWEEP_THRESHOLDS)):
        if not np.isnan(M[i, j]):
            shade = (M[i, j] - vmin) / (vmax - vmin + 1e-9)
            ax.text(j, i, ann[i][j], ha='center', va='center', fontsize=9,
                    color='white' if shade > 0.6 else 'black')
ax.set_xlabel('Threshold')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Average AI chain length')
# figure titles live in the writeup, not the PNG
fig.tight_layout()
fig.savefig(f"{output_plot_path}/chainLength_logic_threshold_heatmap.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved chainLength_logic_threshold_heatmap.png")

# ====================== PLACEBO NULL ======================
if N_RESHUFFLES > 0:
    null_path = f"{output_data_path}/chainLength_placebo_null_draws.csv"
    if os.path.exists(null_path):     # reuse cached draws (delete the file to force a fresh run)
        null_df = pd.read_csv(null_path)
        print(f"Loaded cached null ({len(null_df):,} rows) from chainLength_placebo_null_draws.csv")
    else:
        null_rows = []
        t0 = time.time()
        PRINT_EVERY = max(1, N_RESHUFFLES // 20)
        for i in range(N_RESHUFFLES):
            sp = shuffle_pool(POOL, SHUFFLE_SEED_BASE + i)
            for label, fam, thr, cols in CUTS:
                d = sp if cols is None else sp[sp[cols].sum(axis=1) >= thr]
                d = d[d['O*NET-SOC Code'].isin(valid_occ_set(cols, thr))]
                null_rows.append({'cut': label, 'draw': i, 'mean_chain_length': mean_chain_length(d)})
            if (i + 1) % PRINT_EVERY == 0 or i == 0:
                el = time.time() - t0
                print(f"  draw {i+1}/{N_RESHUFFLES} | {el/60:4.1f}m | {el/(i+1):4.2f}s/draw", flush=True)
        null_df = pd.DataFrame(null_rows)
        null_df.to_csv(null_path, index=False)
        print(f"DONE null: {N_RESHUFFLES} draws in {(time.time()-t0)/60:.1f} min; {len(null_df):,} rows.")

    # ====================== FOREST (rows = cuts) ======================
    def _ord(n):
        n = int(round(n)); suf = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10 if n % 100 not in (11, 12, 13) else 0, 'th')
        return f"{n}{suf}"

    order = [c[0] for c in CUTS]
    ys = list(range(len(order)))[::-1]
    fig, ax = plt.subplots(figsize=(10, 7.2))
    for label, y in zip(order, ys):
        v = null_df[null_df['cut'] == label]['mean_chain_length'].dropna().values
        obs = observed[label]
        if not len(v) or np.isnan(obs):
            continue
        lo, hi = np.percentile(v, 10), np.percentile(v, 90)
        pc = 100.0 * np.mean(v < obs)
        ax.plot([lo, hi], [y, y], color='0.55', lw=5, alpha=0.5, solid_capstyle='round', zorder=1)
        ax.plot(v.mean(), y, marker='|', ms=14, mew=2, color='0.35', zorder=2)
        sig = obs < lo or obs > hi
        ax.plot(obs, y, 'o', ms=9, color=('crimson' if sig else 'steelblue'), markeredgecolor='k', mew=0.6, zorder=3)
        ax.text(1.012, y, f"{obs:.2f}  ({_ord(pc)})", transform=ax.get_yaxis_transform(), clip_on=False,
                va='center', ha='left', fontsize=8, color='dimgray')
    ax.axhline(len(order) - 1.5, color='black', lw=1)
    ax.set_yticks(ys); ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel('Average AI chain length (steps per contiguous AI-executed run)', fontsize=9.5)
    ax.grid(axis='x', ls=':', alpha=0.5); ax.margins(x=0.16)
    # figure titles live in the writeup, not the PNG
    leg = [Line2D([0], [0], color='0.55', lw=5, alpha=0.5, label='Placebo null 10–90%'),
           Line2D([0], [0], marker='|', ls='', ms=12, mew=2, color='0.35', label='Placebo mean'),
           Line2D([0], [0], marker='o', ls='', ms=9, color='crimson', mec='k', label='Observed (outside 10–90% null)'),
           Line2D([0], [0], marker='o', ls='', ms=9, color='steelblue', mec='k', label='Observed (within 10–90% null)')]
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.legend(handles=leg, loc='lower center', ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.savefig(f"{output_plot_path}/chainLength_placebo_forest.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved chainLength_placebo_forest.png")
