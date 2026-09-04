"""Predictions #1 and #2 on APQC's documented sequences, pooling the cross-industry and 17 industry PCFs.

Consumes the step-to-task match file written by `apqc_industry_leaf_matching.py`.

Label transfer. A step inherits its matched O*NET task's exposure and execution labels only when the match
reaches SIM_FLOOR cosine similarity. Steps below the floor are NOT dropped: dropping them would shorten the
very sequences the exercise is about, and non-randomly so, since weak matches concentrate in the operational
and industry-specific parts of the frameworks. They stay in place, coded as neither AI-exposed nor
AI-executed, so the documented ordering is preserved and an unverifiable label is treated as an absent one.

Specification mirrors `onet_fragmentationIndex.ipynb`: EFI Definition 1 is built from E1|E2, the exposure
regressor is the E1 share, and the count of E1|E2 steps enters as a control. All four are standardized.
"""
import os, sys, warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

MAIN = "/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin"
SRC  = f"{MAIN}/data/computed_objects/apqc_pred3_industry/industry_leaf_matches.csv"
OUT  = f"{MAIN}/data/computed_objects/apqc_pred3_industry"
TAB  = f"{MAIN}/writeup/tables/apqc_fragmentation_index_regression.tex"

SIM_FLOOR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.73
MIN_STEPS = 5            # a group needs >= 5 steps, as in the matching script
N_DRAWS   = 1000

L = pd.read_csv(SRC, dtype={'hid': str})
L['sk'] = L['hid'].map(lambda h: tuple(int(x) for x in h.split('.')))
L = L.sort_values(['uid', 'sk']).reset_index(drop=True)
L['category'] = L['hid'].str.split('.').str[0]

carried = L['similarity'] >= SIM_FLOOR
L['exposed']  = (carried & L['human_labels'].isin(['E1', 'E2'])).astype(int)      # EFI Definition 1
L['e1']       = (carried & (L['human_labels'] == 'E1')).astype(int)               # exposure regressor
L['executed'] = (carried & L['label'].isin(['Augmentation', 'Automation'])).astype(int)

L = L.groupby('uid').filter(lambda g: len(g) >= MIN_STEPS)
print(f"floor {SIM_FLOOR} | {L.uid.nunique():,} groups | {len(L):,} steps | "
      f"{len(L) / L.uid.nunique():.1f} per group")
print(f"  labels carried on {carried.sum():,} steps ({carried.mean() * 100:.1f}%), "
      f"mean cosine among those {L.loc[L['similarity'] >= SIM_FLOOR, 'similarity'].mean():.3f}")
print(f"  AI-exposed {L.exposed.mean() * 100:.1f}%   E1 {L.e1.mean() * 100:.1f}%   "
      f"AI-executed {L.executed.mean() * 100:.1f}%")

seqs_exec = {u: g['executed'].to_numpy() for u, g in L.groupby('uid', sort=False)}
seqs_exp  = {u: g['exposed'].to_numpy()  for u, g in L.groupby('uid', sort=False)}
seqs_e1   = {u: g['e1'].to_numpy()       for u, g in L.groupby('uid', sort=False)}
cat_of    = L.groupby('uid')['category'].first().to_dict()
fw_of     = L.groupby('uid')['framework'].first().to_dict()
units     = list(seqs_exec)

# ---------------- Prediction #1: mean length of contiguous AI-executed runs ----------------
def mean_chain(arrays):
    runs, c = [], 0
    for a in arrays:
        c = 0
        for v in a:
            if v:
                c += 1
            elif c:
                runs.append(c); c = 0
        if c:
            runs.append(c)
    return float(np.mean(runs)) if runs else np.nan

arrays = [seqs_exec[u] for u in units]
observed = mean_chain(arrays)

# Null A: reshuffle step order within each group. Composition is untouched; only arrangement moves.
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

print("\nPrediction #1")
p1 = []
for nm, null in [('within-group reshuffle', reshuffle), ('within-category reassignment', reassign)]:
    z, pct = (observed - null.mean()) / null.std(ddof=1), (null < observed).mean() * 100
    p1.append({'null': nm, 'observed': observed, 'null_mean': null.mean(),
               'null_sd': null.std(ddof=1), 'z': z, 'percentile': pct})
    print(f"  {nm:<30} observed {observed:.3f} vs {null.mean():.3f} "
          f"(sd {null.std(ddof=1):.3f}) | z {z:+.2f} | {pct:.0f}th pct")
pd.DataFrame(p1).to_csv(f"{OUT}/chain_length_placebo.csv", index=False)

# ---------------- Prediction #2: the fragmentation regression ----------------
def efi_of(seq):
    """Blocks per step: a position counts as a switch unless it and its successor are both AI-able."""
    sw = np.ones(len(seq))
    sw[:-1][(seq[:-1] == 1) & (seq[1:] == 1)] = 0
    return sw.mean()

panel = pd.DataFrame([{
    'unit': u, 'num_steps': len(seqs_exec[u]),
    'ai_fraction': seqs_exec[u].mean(),
    'ai_exposure': seqs_e1[u].mean(),
    'num_E1E2_tasks': float(seqs_exp[u].sum()),
    'fragmentation_index': efi_of(seqs_exp[u]),
    'category': str(cat_of[u]), 'framework': str(fw_of[u]),
} for u in units])
panel.to_csv(f"{OUT}/process_group_panel.csv", index=False)

# plain object dtype so patsy can build the design matrix on pandas >= 3.0
for c in ('category', 'framework'):
    panel[c] = panel[c].astype('object')

for c in ['ai_fraction', 'ai_exposure', 'fragmentation_index', 'num_E1E2_tasks']:
    panel[c] = (panel[c] - panel[c].mean()) / panel[c].std()

base = 'ai_fraction ~ fragmentation_index + ai_exposure + num_E1E2_tasks'
mods = [smf.ols(f, panel).fit(cov_type='HC1')
        for f in (base, base + ' + C(category)', base + ' + C(framework)')]

print("\nPrediction #2")
for nm, m in zip(['No FE', 'PCF Category FE', 'Framework FE'], mods):
    print(f"  {nm:<16} EFI {m.params['fragmentation_index']:+.3f} (p={m.pvalues['fragmentation_index']:.3f})"
          f"   Exposure {m.params['ai_exposure']:+.3f} (p={m.pvalues['ai_exposure']:.3f})"
          f"   R2 {m.rsquared:.2f}  n={int(m.nobs)}")

star = lambda p: '***' if p < .01 else '**' if p < .05 else '*' if p < .1 else ''
row = lambda lab, v: [f"{lab} & " + " & ".join(f"{m.params[v]:.2f}{star(m.pvalues[v])}" for m in mods) + r" \\",
                      " & " + " & ".join(f"({m.bse[v]:.2f})" for m in mods) + r" \\"]
tex = [r"\setlength{\tabcolsep}{12pt}", r"\begin{tabular}{lccc}", r"\toprule",
       r" & \multicolumn{3}{c}{Share of AI-executed Tasks} \\", r"\cmidrule(lr){2-4}",
       r" & (1) & (2) & (3) \\", r"\midrule", r"\addlinespace"]
tex += row("Share of AI-exposed Tasks", 'ai_exposure') + [r"\addlinespace"]
tex += row("Empirical Fragmentation Index (Definition 1)", 'fragmentation_index')
tex += [r"\hline\\[-1.25em]",
        r"Fixed Effect & & PCF Category & Framework \\",
        r"Number of AI-able Steps Control & \checkmark & \checkmark & \checkmark \\",
        "R-squared & " + " & ".join(f"{m.rsquared:.2f}" for m in mods) + r" \\",
        "Observations & " + " & ".join(f"{int(m.nobs)}" for m in mods) + r" \\",
        r"\bottomrule",
        r"\multicolumn{4}{l}{\footnotesize Standardized coefficients. Robust standard errors in parentheses. "
        r"*** p$<$0.01, ** p$<$0.05, * p$<$0.1} \\",
        r"\end{tabular}"]
open(TAB, 'w').write("\n".join(tex) + "\n")
print(f"\nwrote {TAB}")

# ---------------- Prediction #3: why it is not estimated ----------------
dwa = pd.read_csv(f"{MAIN}/data/computed_objects/DWA_analysis/repetitiveDWA_long.csv")
per_task = dwa.groupby('Task ID')['DWA ID'].nunique()
single = set(per_task[per_task == 1].index)
t2d = dwa.drop_duplicates('Task ID').set_index('Task ID')['DWA ID'].to_dict()

S = L[(L['similarity'] >= SIM_FLOOR) & (L['match_task_id'].isin(single))].copy()
S['dwa'] = S['match_task_id'].map(t2d)
recurring = S.groupby('dwa')['uid'].nunique()
S = S[S['dwa'].isin(recurring[recurring > 1].index)]
varies = S.groupby('dwa')['executed'].nunique()
print(f"\nPrediction #3 power: {len(S):,} steps across {S['dwa'].nunique()} recurring DWAs; "
      f"{(varies > 1).sum()} DWAs carry both an executed and a non-executed step")
print("RUN COMPLETE")
