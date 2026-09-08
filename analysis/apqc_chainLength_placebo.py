"""Prediction #1 on APQC's documented sequences, drawn as the O*NET placebo histograms are.

Mirrors the two nulls of `apqc_pooled_predictions.py` exactly, but keeps every draw rather than
only its summary, so the placebo distributions can be plotted the way
`onet_chainLength.ipynb` plots them for the O*NET sample.

Consumes the step-to-task match file written by `apqc_industry_leaf_matching.py`.
Steps below the similarity floor stay in place, coded as neither AI-exposed nor AI-executed, so
the documented ordering is preserved and an unverifiable label is treated as an absent one.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = f"{MAIN}/data/computed_objects/apqc_pred3_industry/industry_leaf_matches.csv"
OBJ  = f"{MAIN}/data/computed_objects/apqc_chainLength_placebo"
FIG  = f"{MAIN}/writeup/plots/apqc_chainLength_placebo"
os.makedirs(OBJ, exist_ok=True); os.makedirs(FIG, exist_ok=True)

SIM_FLOOR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.71
MIN_STEPS = 5
N_DRAWS   = 1000

L = pd.read_csv(SRC, dtype={'hid': str})
L['sk'] = L['hid'].map(lambda h: tuple(int(x) for x in h.split('.')))
L = L.sort_values(['uid', 'sk']).reset_index(drop=True)
L['category'] = L['hid'].str.split('.').str[0]

carried = L['similarity'] >= SIM_FLOOR
L['executed'] = (carried & L['label'].isin(['Augmentation', 'Automation'])).astype(int)
L['exposed']  = (carried & L['human_labels'].isin(['E1', 'E2'])).astype(int)

L = L.groupby('uid').filter(lambda g: len(g) >= MIN_STEPS)
print(f"floor {SIM_FLOOR} | {L.uid.nunique():,} groups | {len(L):,} steps")
print(f"  AI-exposed {L.exposed.mean()*100:.1f}%   AI-executed {L.executed.mean()*100:.1f}%")

seqs   = {u: g['executed'].to_numpy() for u, g in L.groupby('uid', sort=False)}
cat_of = L.groupby('uid')['category'].first().to_dict()
units  = list(seqs)
arrays = [seqs[u] for u in units]


def mean_chain(arrs):
    """Mean length of the maximal runs of AI-executed steps."""
    runs, c = [], 0
    for a in arrs:
        c = 0
        for v in a:
            if v:
                c += 1
            elif c:
                runs.append(c); c = 0
        if c:
            runs.append(c)
    return float(np.mean(runs)) if runs else np.nan


observed = mean_chain(arrays)

# Null A: reshuffle step order within each process group. Composition fixed, arrangement moves.
rng = np.random.default_rng(42)
reshuffle = np.array([mean_chain([rng.permutation(a) for a in arrays]) for _ in range(N_DRAWS)])

# Null B: reassign steps across groups within a PCF Category, preserving each group's size.
rng = np.random.default_rng(42)
by_cat = {}
for u, a in zip(units, arrays):
    by_cat.setdefault(cat_of[u], []).append(a)
reassign = []
for _ in range(N_DRAWS):
    out = []
    for arrs in by_cat.values():
        pool, i = rng.permutation(np.concatenate(arrs)), 0
        for a in arrs:
            out.append(pool[i:i + len(a)]); i += len(a)
    reassign.append(mean_chain(out))
reassign = np.array(reassign)

summary = []
for nm, null in [('within-group reshuffle', reshuffle), ('within-category reassignment', reassign)]:
    z, pct = (observed - null.mean()) / null.std(ddof=1), (null < observed).mean() * 100
    summary.append({'null': nm, 'observed': observed, 'null_mean': null.mean(),
                    'null_sd': null.std(ddof=1), 'z': z, 'percentile': pct})
    print(f"  {nm:<30} observed {observed:.3f} vs {null.mean():.3f} "
          f"(sd {null.std(ddof=1):.3f}) | z {z:+.2f} | {pct:.0f}th pct")
pd.DataFrame(summary).to_csv(f"{OBJ}/chain_length_placebo_summary.csv", index=False)
pd.DataFrame({'reshuffle': reshuffle, 'reassign': reassign}).to_csv(
    f"{OBJ}/chain_length_placebo_draws.csv", index=False)

# ---------------- the two panels, styled as the O*NET pair in onet_chainLength.ipynb ----------------
panels = [
    (reshuffle, 'Shuffled Step Positions',        'aiChains_chainLength_stepPositionReshuffle_apqc.png'),
    (reassign,  'Shuffled Step Execution Labels', 'aiChains_chainLength_stepAssignmentReshuffle_apqc.png'),
]
lo = min(reshuffle.min(), reassign.min())
pad = (observed - lo) * 0.12
for vals, lab, fname in panels:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.hist(vals, bins=30, color='orange', edgecolor='black', label=lab)
    ax.axvline(observed, color='red', linestyle='dashed', linewidth=2,
               label=f'Observed = {observed:.2f}')
    ax.set_xlabel('Average AI Chain Length', fontsize=19)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=14, loc='upper left')   # the observed line sits at the right edge
    ax.set_xlim(lo - pad, observed + pad)          # common scale across the two panels
    fig.tight_layout()
    fig.savefig(f"{FIG}/{fname}", dpi=300)
    plt.close()
print(f"\nwrote 2 figures to {FIG}")
