"""Read-only re-estimation of the SA.F APQC external-validation fragmentation results.

Reimplements, without touching the repo:
  (A) analysis/apqc_pooled_predictions.py  -> writeup/tables/apqc_fragmentation_index_regression.tex
      pooled Cross-Industry + 17 industry PCFs (18 framework labels in all; the paper says 16, which is a miscount), SIM_FLOOR 0.73, MIN_STEPS 5, num_E1E2_tasks control
  (B) analysis/apqc_pcf_fragmentationIndex.ipynb section 6 -> exposure_definition_grid.csv
      cross-industry PCF only, NO similarity floor, MIN_STEPS 3, NO step-count control

Output goes to data/computed_objects/efi_matched_exposure/. No published exhibit is touched.
"""
import os, sys, warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats as sps

import os as _os
# Repo root and output dir are derived from this file's own location, so the folder
# can move and no home directory is baked in. Generated output follows the repo
# convention: data/computed_objects/<name of this analysis folder>/.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
OUT = _os.path.join(REPO, "data", "computed_objects", "efi_matched_exposure")
_os.makedirs(OUT, exist_ok=True)


warnings.filterwarnings('ignore')
pd.set_option('display.width', 240)

MAIN = REPO
POOLED = f"{MAIN}/data/computed_objects/apqc_pred3_industry/industry_leaf_matches.csv"
PCFXI  = f"{MAIN}/data/computed_objects/apqc_pcf_fragmentation/pcf_leaf_matches.csv"
ONETP  = f"{MAIN}/data/computed_objects/ONET_Eloundou_Anthropic_GPT/ONET_Eloundou_Anthropic_GPT.csv"
ONETOCC= f"{MAIN}/data/computed_objects/fragmentationIndex/occupation_analysis_with_fragmentationIndex_def1.csv"
SIM_FLOOR, MIN_STEPS_POOLED, N_DRAWS = 0.73, 5, 1000

star = lambda p: '***' if p < .01 else '**' if p < .05 else '*' if p < .1 else ''


def efi_of(seq):
    """Blocks per step, exactly as the repo: a position is a switch unless it and its successor are both AI-able."""
    sw = np.ones(len(seq))
    sw[:-1][(seq[:-1] == 1) & (seq[1:] == 1)] = 0
    return sw.mean()


# =====================================================================================
# 1. POOLED SAMPLE (the one behind the PUBLISHED SA.F table)
# =====================================================================================
L = pd.read_csv(POOLED, dtype={'hid': str})
L['sk'] = L['hid'].map(lambda h: tuple(int(x) for x in h.split('.')))
L = L.sort_values(['uid', 'sk']).reset_index(drop=True)
L['category'] = L['hid'].str.split('.').str[0]

carried = L['similarity'] >= SIM_FLOOR
L['exposed']  = (carried & L['human_labels'].isin(['E1', 'E2'])).astype(int)
L['e1']       = (carried & (L['human_labels'] == 'E1')).astype(int)
L['executed'] = (carried & L['label'].isin(['Augmentation', 'Automation'])).astype(int)
L = L.groupby('uid').filter(lambda g: len(g) >= MIN_STEPS_POOLED)

print("=" * 100)
print("POOLED APQC SAMPLE (SA.F headline)")
print(f"  floor {SIM_FLOOR} | {L.uid.nunique():,} groups | {len(L):,} steps | {len(L)/L.uid.nunique():.1f} per group")
print(f"  labels carried on {carried.sum():,} steps of full corpus; within sample {L.exposed.mean()*100:.1f}% AI-exposed, "
      f"{L.e1.mean()*100:.1f}% E1, {L.executed.mean()*100:.1f}% AI-executed")

seqs_exec = {u: g['executed'].to_numpy() for u, g in L.groupby('uid', sort=False)}
seqs_exp  = {u: g['exposed'].to_numpy()  for u, g in L.groupby('uid', sort=False)}
seqs_e1   = {u: g['e1'].to_numpy()       for u, g in L.groupby('uid', sort=False)}
cat_of = L.groupby('uid')['category'].first().to_dict()
fw_of  = L.groupby('uid')['framework'].first().to_dict()
units  = list(seqs_exec)

panel = pd.DataFrame([{
    'unit': u, 'num_steps': len(seqs_exec[u]),
    'ai_fraction': seqs_exec[u].mean(),
    'ai_exposure_E1': seqs_e1[u].mean(),
    'ai_exposure_E1E2': seqs_exp[u].mean(),
    'num_E1E2_tasks': float(seqs_exp[u].sum()),
    'fragmentation_index': efi_of(seqs_exp[u]),
    'category': str(cat_of[u]), 'framework': str(fw_of[u]),
} for u in units])

RAW_POOLED = panel.copy()   # keep the unstandardized panel for level-scale arithmetic

Z = panel.copy()
for c in ('category', 'framework'):
    Z[c] = Z[c].astype('object')
for c in ['ai_fraction', 'ai_exposure_E1', 'ai_exposure_E1E2', 'fragmentation_index', 'num_E1E2_tasks']:
    Z[c] = (Z[c] - Z[c].mean()) / Z[c].std()


def fit_pooled(expo, control=True):
    rhs = 'fragmentation_index + ' + expo + (' + num_E1E2_tasks' if control else '')
    return [smf.ols(f'ai_fraction ~ {rhs}' + fe, Z).fit(cov_type='HC1')
            for fe in ('', ' + C(category)', ' + C(framework)')]


def report(models, expo, tag):
    print(f"\n--- {tag} ---")
    print(f"{'column':<18}{'EFI':>10}{'se':>9}{'p':>9}{'95% CI':>22}{'exposure':>11}{'se':>9}{'p':>9}{'R2':>7}{'N':>6}")
    rows = []
    for nm, m in zip(['(1) no FE', '(2) PCF Category FE', '(3) Framework FE'], models):
        ci = m.conf_int().loc['fragmentation_index']
        cie = m.conf_int().loc[expo]
        rows.append(dict(col=nm, b=m.params['fragmentation_index'], se=m.bse['fragmentation_index'],
                         p=m.pvalues['fragmentation_index'], lo=ci[0], hi=ci[1],
                         be=m.params[expo], see=m.bse[expo], pe=m.pvalues[expo],
                         eLo=cie[0], eHi=cie[1], r2=m.rsquared, n=int(m.nobs)))
        print(f"{nm:<18}{m.params['fragmentation_index']:>+9.3f}{star(m.pvalues['fragmentation_index']):<4}"
              f"{m.bse['fragmentation_index']:>7.3f}{m.pvalues['fragmentation_index']:>9.3f}"
              f"  [{ci[0]:+.3f},{ci[1]:+.3f}]"
              f"{m.params[expo]:>+9.3f}{star(m.pvalues[expo]):<4}{m.bse[expo]:>7.3f}{m.pvalues[expo]:>9.3f}"
              f"{m.rsquared:>7.3f}{int(m.nobs):>6}")
    return pd.DataFrame(rows)


pub_pooled = fit_pooled('ai_exposure_E1', True)
mat_pooled = fit_pooled('ai_exposure_E1E2', True)
mat_pooled_nc = fit_pooled('ai_exposure_E1E2', False)
pub_pooled_nc = fit_pooled('ai_exposure_E1', False)

R_pub = report(pub_pooled, 'ai_exposure_E1', "PUBLISHED pooled: exposure = E1 share, control = num_E1E2_tasks")
R_mat = report(mat_pooled, 'ai_exposure_E1E2', "MATCHED pooled: exposure = E1|E2 share, control = num_E1E2_tasks (HEADLINE)")
R_matnc = report(mat_pooled_nc, 'ai_exposure_E1E2', "MATCHED pooled, NO step-count control (diagnostic only)")
R_pubnc = report(pub_pooled_nc, 'ai_exposure_E1', "PUBLISHED pooled, NO step-count control (diagnostic only)")

print("\nCollinearity in the pooled sample:")
u = RAW_POOLED
print(f"  corr(EFI, E1|E2 share)  = {u['fragmentation_index'].corr(u['ai_exposure_E1E2']):+.4f}")
print(f"  corr(EFI, E1 share)     = {u['fragmentation_index'].corr(u['ai_exposure_E1']):+.4f}")
r2 = smf.ols('fragmentation_index ~ ai_exposure_E1E2', RAW_POOLED).fit().rsquared
print(f"  R2(EFI on E1|E2)        = {r2:.4f}   VIF = {1/(1-r2):.1f}")
r2c = smf.ols('fragmentation_index ~ ai_exposure_E1E2 + num_E1E2_tasks', RAW_POOLED).fit().rsquared
print(f"  R2(EFI on E1|E2 + count)= {r2c:.4f}   VIF = {1/(1-r2c):.1f}")

# =====================================================================================
# 2. PLACEBO: within-group order reshuffle, pooled sample, matched spec
# =====================================================================================
arrays = [seqs_exp[u] for u in units]


def placebo(spec_expo, control, n_draws=N_DRAWS, seed=42):
    rng = np.random.default_rng(seed)
    base = Z.copy()
    rhs = 'fragmentation_index + ' + spec_expo + (' + num_E1E2_tasks' if control else '')
    f = f'ai_fraction ~ {rhs}'
    obs = smf.ols(f, base).fit(cov_type='HC1').params['fragmentation_index']
    draws = np.empty(n_draws)
    for i in range(n_draws):
        e = np.array([efi_of(rng.permutation(a)) for a in arrays])
        d = base.copy()
        d['fragmentation_index'] = (e - e.mean()) / e.std()
        draws[i] = smf.ols(f, d).fit().params['fragmentation_index']
    return obs, draws


print("\n" + "=" * 100)
print("PLACEBO, pooled n=%d, within-group step-order reshuffle, %d draws" % (len(panel), N_DRAWS))
plac = {}
for tag, expo, ctl in [('published (E1)', 'ai_exposure_E1', True),
                       ('matched (E1|E2)', 'ai_exposure_E1E2', True),
                       ('matched, no control', 'ai_exposure_E1E2', False)]:
    o, v = placebo(expo, ctl)
    z = (o - v.mean()) / v.std(ddof=1)
    pct = (v < o).mean() * 100
    pl = (1 + int((v <= o).sum())) / (1 + N_DRAWS)
    plac[tag] = (o, v)
    print(f"  {tag:<22} observed {o:+.4f} | null mean {v.mean():+.4f} (sd {v.std(ddof=1):.4f}) | "
          f"z {z:+.2f} | pctile {pct:.1f} | one-sided p {pl:.4f}")

# =====================================================================================
# 3. CROSS-INDUSTRY-ONLY NOTEBOOK SAMPLE (n=73), the grid's construction
# =====================================================================================
print("\n" + "=" * 100)
print("CROSS-INDUSTRY PCF ONLY (the exposure_definition_grid.csv construction)")
lv = pd.read_csv(PCFXI, dtype={'hierarchy_id': str})
lv['sort_key'] = lv['hierarchy_id'].map(lambda h: tuple(int(x) for x in h.split('.')))
lv = lv.sort_values('sort_key').reset_index(drop=True)
lv['unit'] = lv['hierarchy_id'].map(lambda h: '.'.join(h.split('.')[:2]))
lv['cat'] = lv['hierarchy_id'].str.split('.').str[0].astype(int)

rows, seq73 = [], {}
for uu, g in lv.groupby('unit', sort=False):
    if len(g) < 3:
        continue
    isai = g['human_labels'].isin(['E1', 'E2']).astype(int).to_numpy()
    seq73[uu] = isai
    rows.append({'unit': uu, 'n': len(g),
                 'ai': g['label'].isin(['Augmentation', 'Automation']).mean(),
                 'e1': (g['human_labels'] == 'E1').mean(),
                 'e12': isai.mean(),
                 'nE1E2': float(isai.sum()),
                 'fe': g['cat'].iloc[0]})
P73 = pd.DataFrame(rows)
P73['efi'] = P73['unit'].map(lambda x: 1.0 - float((seq73[x][:-1] & seq73[x][1:]).sum()) / len(seq73[x]))
RAW73 = P73.copy()
print(f"  n={len(P73)} groups | mean steps {P73['n'].mean():.1f} | EFI mean {P73['efi'].mean():.4f} sd {P73['efi'].std():.4f}")


def fit73(formula, fe=False, data=None):
    z = (P73 if data is None else data).copy()
    for c in ['ai', 'e1', 'efi', 'e12', 'nE1E2']:
        z[c] = (z[c] - z[c].mean()) / z[c].std()
    z['fe'] = z['fe'].astype('object')
    return smf.ols(formula + (' + C(fe)' if fe else ''), z).fit(cov_type='HC1')


print(f"\n{'spec':<40}{'FE':<7}{'EFI':>12}{'se':>8}{'p':>9}{'exposure':>11}{'se':>8}{'p':>9}{'R2':>8}{'N':>5}")
grid_mine = []
for tag, f, ev in [('A. paper (E1), no control', 'ai ~ efi + e1', 'e1'),
                   ('B. matched (E1|E2), no control', 'ai ~ efi + e12', 'e12'),
                   ('A+ paper (E1) + count control', 'ai ~ efi + e1 + nE1E2', 'e1'),
                   ('B+ matched (E1|E2) + count control', 'ai ~ efi + e12 + nE1E2', 'e12')]:
    for fe in (False, True):
        m = fit73(f, fe)
        ci = m.conf_int().loc['efi']
        grid_mine.append(dict(spec=tag, fe='FE' if fe else 'No FE', b=m.params['efi'], se=m.bse['efi'],
                              p=m.pvalues['efi'], lo=ci[0], hi=ci[1], be=m.params[ev], see=m.bse[ev],
                              pe=m.pvalues[ev], r2=m.rsquared, n=int(m.nobs)))
        print(f"{tag:<40}{'FE' if fe else 'No FE':<7}{m.params['efi']:>+10.3f}{star(m.pvalues['efi']):<3}"
              f"{m.bse['efi']:>7.3f}{m.pvalues['efi']:>9.3f}{m.params[ev]:>+9.3f}{star(m.pvalues[ev]):<3}"
              f"{m.bse[ev]:>7.3f}{m.pvalues[ev]:>9.3f}{m.rsquared:>8.3f}{int(m.nobs):>5}")

G = pd.read_csv(f"{MAIN}/data/computed_objects/apqc_pcf_fragmentation/exposure_definition_grid.csv")
print("\nrepo grid rows for APQC, for comparison:")
print(G[G['sample'].str.startswith('APQC')][['spec', 'fe', 'n', 'b_efi', 'se_efi', 'p_efi', 'b_e1', 'b_e12', 'r2']].to_string(index=False))

# ---- placebo on the n=73 sample, matched spec, with and without the count control ----
arr73 = [seq73[x] for x in P73['unit']]


def placebo73(f, control_note, seed=42, n_draws=N_DRAWS):
    rng = np.random.default_rng(seed)
    obs = fit73(f).params['efi']
    dr = np.empty(n_draws)
    for i in range(n_draws):
        d = P73.copy()
        d['efi'] = np.array([1.0 - float((p[:-1] & p[1:]).sum()) / len(p)
                             for p in (rng.permutation(a) for a in arr73)])
        dr[i] = fit73(f, data=d).params['efi']
    z = (obs - dr.mean()) / dr.std(ddof=1)
    print(f"  {control_note:<38} observed {obs:+.4f} | null {dr.mean():+.4f} (sd {dr.std(ddof=1):.4f}) | "
          f"z {z:+.2f} | pctile {(dr < obs).mean()*100:.1f} | p {(1+int((dr<=obs).sum()))/(1+n_draws):.4f}")
    return obs, dr


print(f"\nPLACEBO, cross-industry n=73, {N_DRAWS} draws:")
placebo73('ai ~ efi + e1', 'A. paper (E1), no control')
placebo73('ai ~ efi + e12', 'B. matched (E1|E2), no control')
placebo73('ai ~ efi + e12 + nE1E2', 'B+ matched (E1|E2) + count control')

# =====================================================================================
# 4. O*NET main sample, for the reconciliation of the grid's O*NET rows
# =====================================================================================
print("\n" + "=" * 100)
print("O*NET reconciliation")
occ = pd.read_csv(ONETOCC)
print(f"  published occupation panel: n={len(occ)} | EFI sd {occ['fragmentation_index'].std():.4f} | "
      f"ai_fraction sd {occ['ai_fraction'].std():.4f} | mean AI exec {occ['ai_fraction'].mean():.4f}")

onet = pd.read_csv(ONETP).dropna(subset=['human_labels']).copy()
onet['Task Position'] = pd.to_numeric(onet['Task Position'], errors='coerce')
onet = onet.dropna(subset=['Task Position'])
onet = onet.sort_values(['O*NET-SOC Code', 'Task Position'])
rws, so = [], {}
for uu, g in onet.groupby('O*NET-SOC Code', sort=False):
    if len(g) < 3:
        continue
    isai = g['human_labels'].isin(['E1', 'E2']).astype(int).to_numpy()
    so[uu] = isai
    rws.append({'unit': uu, 'n': len(g), 'ai': g['label'].isin(['Augmentation', 'Automation']).mean(),
                'e1': (g['human_labels'] == 'E1').mean(), 'e12': isai.mean(),
                'nE1E2': float(isai.sum()), 'fe': g['Major_Group_Code'].iloc[0]})
PO = pd.DataFrame(rws)
PO['efi'] = PO['unit'].map(lambda x: 1.0 - float((so[x][:-1] & so[x][1:]).sum()) / len(so[x]))
print(f"  notebook-style O*NET panel: n={len(PO)} | EFI sd {PO['efi'].std():.4f} | ai sd {PO['ai'].std():.4f}")


def fitO(f, fe=False):
    z = PO.copy()
    for c in ['ai', 'e1', 'efi', 'e12', 'nE1E2']:
        z[c] = (z[c] - z[c].mean()) / z[c].std()
    z['fe'] = z['fe'].astype('object')
    return smf.ols(f + (' + C(fe)' if fe else ''), z).fit(cov_type='HC1')


print(f"\n{'spec':<40}{'FE':<7}{'EFI':>12}{'se':>8}{'p':>9}{'N':>6}")
for tag, f in [('B. matched (E1|E2), no control', 'ai ~ efi + e12'),
               ('B+ matched (E1|E2) + count control', 'ai ~ efi + e12 + nE1E2')]:
    for fe in (False, True):
        m = fitO(f, fe)
        print(f"{tag:<40}{'FE' if fe else 'No FE':<7}{m.params['efi']:>+10.3f}{star(m.pvalues['efi']):<3}"
              f"{m.bse['efi']:>7.3f}{m.pvalues['efi']:>9.3f}{int(m.nobs):>6}")

# =====================================================================================
# 5. MDE and level-scale arithmetic
# =====================================================================================
print("\n" + "=" * 100)
print("MINIMUM DETECTABLE EFFECT (80% power, 5% two-sided): MDE = 2.802 * SE, in sd units")
for nm, R in [('pooled MATCHED n=525', R_mat), ('pooled PUBLISHED n=525', R_pub)]:
    for _, r in R.iterrows():
        print(f"  {nm:<24}{r['col']:<22} SE {r['se']:.3f} -> MDE {2.802*r['se']:+.3f} sd   (|b| = {abs(r['b']):.3f})")
gm = pd.DataFrame(grid_mine)
for _, r in gm[gm['spec'].str.startswith(('B.', 'B+'))].iterrows():
    print(f"  {'cross-industry n=73':<24}{r['spec'][:20]:<20} {r['fe']:<6} SE {r['se']:.3f} -> MDE {2.802*r['se']:+.3f} sd")

print("\nLEVEL SCALE")
sd_efi_p, sd_ai_p = RAW_POOLED['fragmentation_index'].std(), RAW_POOLED['ai_fraction'].std()
sd_efi_o, sd_ai_o = occ['fragmentation_index'].std(), occ['ai_fraction'].std()
print(f"  pooled APQC : sd(EFI) {sd_efi_p:.4f} | sd(AI exec share) {sd_ai_p:.4f} | mean AI exec {RAW_POOLED['ai_fraction'].mean():.4f}")
print(f"  cross-ind 73: sd(EFI) {RAW73['efi'].std():.4f} | sd(AI exec share) {RAW73['ai'].std():.4f}")
print(f"  O*NET 872   : sd(EFI) {sd_efi_o:.4f} | sd(AI exec share) {sd_ai_o:.4f}")
print("  effect of a 1-sd rise in EFI on the AI-execution share, in percentage points:")
for lab, R, sd_ai, sd_efi in [('APQC published', R_pub, sd_ai_p, sd_efi_p), ('APQC matched', R_mat, sd_ai_p, sd_efi_p)]:
    s = "  ".join(f"{r['col'][:3]} {100*r['b']*sd_ai:+.2f}pp" for _, r in R.iterrows())
    print(f"    {lab:<16} {s}")
    s2 = "  ".join(f"{r['col'][:3]} {r['b']*sd_ai/sd_efi:+.3f}" for _, r in R.iterrows())
    print(f"    {'':16} per unit of EFI (share per index unit): {s2}")
for lab, betas in [('O*NET published', [-0.261, -0.380, -0.283]), ('O*NET matched', [-0.007, -0.086, -0.040])]:
    print(f"    {lab:<16} " + "  ".join(f"{100*b*sd_ai_o:+.2f}pp" for b in betas)
          + "   per unit EFI: " + "  ".join(f"{b*sd_ai_o/sd_efi_o:+.3f}" for b in betas))

# =====================================================================================
# 6. LaTeX table in the published form, matched spec
# =====================================================================================
mods = mat_pooled
row = lambda lab, v: [f"{lab} & " + " & ".join(f"{m.params[v]:.2f}{star(m.pvalues[v])}" for m in mods) + r" \\",
                      " & " + " & ".join(f"({m.bse[v]:.2f})" for m in mods) + r" \\"]
tex = [r"\setlength{\tabcolsep}{12pt}", r"\begin{tabular}{lccc}", r"\toprule",
       r" & \multicolumn{3}{c}{Share of AI-executed Tasks} \\", r"\cmidrule(lr){2-4}",
       r" & (1) & (2) & (3) \\", r"\midrule", r"\addlinespace"]
tex += row("Share of AI-exposed Tasks", 'ai_exposure_E1E2') + [r"\addlinespace"]
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
print("\n" + "=" * 100)
print("NEW TABLE (matched spec), published LaTeX form:\n")
print("\n".join(tex))
open(f"{OUT}/apqc_fragmentation_index_regression_MATCHED.tex", 'w').write("\n".join(tex) + "\n")
print("\nDONE")
