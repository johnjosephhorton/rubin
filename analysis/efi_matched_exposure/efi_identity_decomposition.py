"""
Establish the EFI decomposition identity

    EFI = (m - k + r) / m = 1 - k/m + r/m

empirically on every estimation sample the paper uses.

m = number of workflow steps in the unit
k = number of AI-able steps (Definition 1: human_labels in {E1,E2};
                             Definition 2: label in {Augmentation,Automation})
r = number of MAXIMAL RUNS (blocks) of consecutive AI-able steps

The EFI is reconstructed exactly the way the repo notebooks build it
(is_ai -> next_is_ai via groupby.shift(-1).fillna(0) -> num_switches = 1 except 0
when current and next are both AI-able -> unit mean), and then compared, to
machine precision, against (m - k + r)/m computed by a completely independent
route (run counting with shift(+1)).

Source notebooks read (not executed, not modified):
  analysis/onet_fragmentationIndex.ipynb            (main O*NET sample)
  analysis/onet_fragmentationIndex_robustness.ipynb (10 alternative-prompt orderings)
  analysis/onet_fragmentationIndex_weeklyTasks.ipynb + analysis/make_frag_def1_heatmap.py
                                                    (frequency-pruned samples)
  analysis/apqc_pcf_fragmentationIndex.ipynb        (APQC PCF process groups)

Writes NOTHING into the repo.  All output goes next to this file.
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

import os as _os
# Repo root and output dir are derived from this file's own location, so the folder
# can move and no home directory is baked in. Generated output follows the repo
# convention: data/computed_objects/<name of this analysis folder>/.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
OUT = _os.path.join(REPO, "data", "computed_objects", "efi_matched_exposure")
_os.makedirs(OUT, exist_ok=True)


warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)

DATA = os.path.join(REPO, "data/computed_objects")
MAIN = os.path.join(DATA, "ONET_Eloundou_Anthropic_GPT/ONET_Eloundou_Anthropic_GPT.csv")

OCC = 'O*NET-SOC Code'
TITLE = 'Occupation Title'


# ------------------------------------------------------------------ core
def efi_notebook_way(df, group_cols, ai_col, ai_values):
    """EFI reconstructed EXACTLY as the notebooks do it."""
    f = df.copy()
    f['is_ai'] = f[ai_col].isin(ai_values).astype(int)
    f['next_is_ai'] = f.groupby(group_cols)['is_ai'].shift(-1).fillna(0).astype(int)
    f['num_switches'] = 1
    f.loc[(f['is_ai'] == 1) & (f['next_is_ai'] == 1), 'num_switches'] = 0
    out = (f.groupby(group_cols)['num_switches'].mean()
             .reset_index().rename(columns={'num_switches': 'EFI_nb'}))
    return out, f


def mkr_independent_way(df, group_cols, ai_col, ai_values):
    """m, k, r computed by an INDEPENDENT route: run starts via shift(+1)."""
    f = df.copy()
    f['is_ai'] = f[ai_col].isin(ai_values).astype(int)
    f['prev_is_ai'] = f.groupby(group_cols)['is_ai'].shift(1).fillna(0).astype(int)
    f['run_start'] = ((f['is_ai'] == 1) & (f['prev_is_ai'] == 0)).astype(int)
    g = f.groupby(group_cols)
    out = pd.DataFrame({
        'm': g.size(),
        'k': g['is_ai'].sum(),
        'r': g['run_start'].sum(),
    }).reset_index()
    return out


def identity_check(df, group_cols, ai_col, ai_values, exposure_share_col=None):
    """Returns a per-unit frame with EFI_nb, m, k, r, EFI_id and the max abs deviation."""
    nb, _ = efi_notebook_way(df, group_cols, ai_col, ai_values)
    mkr = mkr_independent_way(df, group_cols, ai_col, ai_values)
    u = nb.merge(mkr, on=group_cols, how='outer', validate='one_to_one')
    u['EFI_id'] = (u['m'] - u['k'] + u['r']) / u['m']
    u['k_over_m'] = u['k'] / u['m']
    u['r_over_m'] = u['r'] / u['m']
    u['dev'] = (u['EFI_nb'] - u['EFI_id']).abs()
    return u


def r2_of(y, X):
    """OLS R-squared of y on a constant plus the columns of X (a DataFrame)."""
    Xc = sm.add_constant(np.asarray(X, dtype=float), has_constant='add')
    res = sm.OLS(np.asarray(y, dtype=float), Xc).fit()
    return res


def banner(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


# =====================================================================
# PART 1 + 2 + 3 + 5 : MAIN O*NET SAMPLE
# =====================================================================
banner("PART 1.  MAIN O*NET SAMPLE (872 occupations) -- identity check, EFI Definition 1 (E1|E2)")

raw = pd.read_csv(MAIN)
# main-notebook sample filter: occupations with >= 3 distinct Task IDs
cnt = raw.groupby(OCC)['Task ID'].nunique()
valid = cnt[cnt >= 3].index
main = raw[raw[OCC].isin(valid)].reset_index(drop=True)
print(f"rows {len(main):,} | occupations {main[OCC].nunique():,} "
      f"(filter >=3 distinct Task IDs dropped {raw[OCC].nunique() - main[OCC].nunique()} occupations)")

u1 = identity_check(main, [OCC, TITLE], 'human_labels', ['E1', 'E2'])
print(f"\nunits: {len(u1):,}")
print(f"max |EFI_notebook - (m-k+r)/m|  = {u1['dev'].max():.3e}")
print(f"mean |dev|                      = {u1['dev'].mean():.3e}")
print(f"number of units with dev > 0    = {int((u1['dev'] > 0).sum())}")
print(f"number of units with dev > 1e-12= {int((u1['dev'] > 1e-12).sum())}")
print(f"exact bitwise equality (==)     = {int((u1['EFI_nb'] == u1['EFI_id']).sum())} / {len(u1)}")

# cross-check k/m against the notebook's own human_aiExposure_fraction
occ_share = (main.assign(_e12=main['human_labels'].isin(['E1', 'E2']).astype(int),
                         _e1=(main['human_labels'] == 'E1').astype(int))
                 .groupby([OCC, TITLE])
                 .agg(n=('_e12', 'size'), e12=('_e12', 'sum'), e1=('_e1', 'sum'))
                 .reset_index())
occ_share['human_aiExposure_fraction'] = occ_share['e12'] / occ_share['n']
occ_share['human_E1_fraction'] = occ_share['e1'] / occ_share['n']
u1 = u1.merge(occ_share[[OCC, TITLE, 'human_aiExposure_fraction', 'human_E1_fraction']],
              on=[OCC, TITLE])
print(f"max |k/m - human_aiExposure_fraction| = "
      f"{(u1['k_over_m'] - u1['human_aiExposure_fraction']).abs().max():.3e}   "
      f"(k is exactly the notebook's num_E1E2_tasks)")

banner("PART 2.  MAIN SAMPLE -- how much of the EFI is the exposure level term")

c_e12 = np.corrcoef(u1['EFI_nb'], u1['human_aiExposure_fraction'])[0, 1]
c_e1 = np.corrcoef(u1['EFI_nb'], u1['human_E1_fraction'])[0, 1]
print(f"corr(EFI_def1, E1|E2 share)  = {c_e12:+.4f}")
print(f"corr(EFI_def1, E1 share)     = {c_e1:+.4f}")

m_a = r2_of(u1['EFI_nb'], u1[['human_aiExposure_fraction']])
print(f"\nR2 of EFI on the E1|E2 share alone           = {m_a.rsquared:.4f}   "
      f"(slope {m_a.params[1]:+.4f}, se {m_a.bse[1]:.4f})")
m_b = r2_of(u1['EFI_nb'], u1[['human_aiExposure_fraction', 'r_over_m']])
print(f"R2 of EFI on the E1|E2 share PLUS r/m        = {m_b.rsquared:.10f}   "
      f"(slopes: k/m {m_b.params[1]:+.10f}, r/m {m_b.params[2]:+.10f}, const {m_b.params[0]:+.3e})")
m_c = r2_of(u1['EFI_nb'], u1[['human_E1_fraction']])
print(f"R2 of EFI on the E1 share alone (published)  = {m_c.rsquared:.4f}   "
      f"(slope {m_c.params[1]:+.4f})")

# variance accounting of the identity
var_efi = u1['EFI_nb'].var(ddof=1)
var_k = u1['k_over_m'].var(ddof=1)
var_r = u1['r_over_m'].var(ddof=1)
cov_kr = np.cov(u1['k_over_m'], u1['r_over_m'], ddof=1)[0, 1]
print(f"\nvariance accounting  Var(EFI) = Var(k/m) + Var(r/m) - 2Cov(k/m, r/m)")
print(f"  Var(EFI)   = {var_efi:.6f}")
print(f"  Var(k/m)   = {var_k:.6f}")
print(f"  Var(r/m)   = {var_r:.6f}")
print(f"  Cov(k/m,r/m) = {cov_kr:.6f}")
print(f"  RHS        = {var_k + var_r - 2 * cov_kr:.6f}   (residual {abs(var_efi - (var_k + var_r - 2*cov_kr)):.3e})")
print(f"  corr(k/m, r/m) = {cov_kr / np.sqrt(var_k * var_r):+.4f}")

banner("PART 3.  MAIN SAMPLE -- EXECUTION-BASED EFI (Definition 2, Augmentation|Automation)")

u2 = identity_check(main, [OCC, TITLE], 'label', ['Augmentation', 'Automation'])
print(f"units: {len(u2):,}")
print(f"max |EFI_notebook - (m-k2+r2)/m| = {u2['dev'].max():.3e}")
print(f"exact bitwise equality (==)      = {int((u2['EFI_nb'] == u2['EFI_id']).sum())} / {len(u2)}")

# the outcome variable of the SA.B table is ai_fraction = k2/m
ai_frac = (main.assign(_ai=main['label'].isin(['Augmentation', 'Automation']).astype(int))
               .groupby([OCC, TITLE])['_ai'].mean().reset_index()
               .rename(columns={'_ai': 'ai_fraction'}))
u2 = u2.merge(ai_frac, on=[OCC, TITLE]).merge(
    occ_share[[OCC, TITLE, 'human_aiExposure_fraction', 'human_E1_fraction']], on=[OCC, TITLE])
print(f"max |k2/m - ai_fraction (the SA.B OUTCOME)| = "
      f"{(u2['k_over_m'] - u2['ai_fraction']).abs().max():.3e}")
print(f"\ncorr(EFI_def2, ai_fraction = AI-execution share) = "
      f"{np.corrcoef(u2['EFI_nb'], u2['ai_fraction'])[0,1]:+.4f}")
print(f"corr(EFI_def2, E1|E2 share)                     = "
      f"{np.corrcoef(u2['EFI_nb'], u2['human_aiExposure_fraction'])[0,1]:+.4f}")
print(f"corr(EFI_def2, E1 share)                        = "
      f"{np.corrcoef(u2['EFI_nb'], u2['human_E1_fraction'])[0,1]:+.4f}")
m_d = r2_of(u2['EFI_nb'], u2[['ai_fraction']])
print(f"\nR2 of EFI_def2 on the AI-EXECUTION share alone = {m_d.rsquared:.4f}  "
      f"(slope {m_d.params[1]:+.4f})")
m_e = r2_of(u2['EFI_nb'], u2[['ai_fraction', 'r_over_m']])
print(f"R2 of EFI_def2 on execution share PLUS r2/m    = {m_e.rsquared:.10f}  "
      f"(slopes: k2/m {m_e.params[1]:+.10f}, r2/m {m_e.params[2]:+.10f})")
m_f = r2_of(u2['EFI_nb'], u2[['human_E1_fraction']])
print(f"R2 of EFI_def2 on the E1 share (SA.B control)  = {m_f.rsquared:.4f}")
m_g = r2_of(u2['EFI_nb'], u2[['human_aiExposure_fraction']])
print(f"R2 of EFI_def2 on the E1|E2 share              = {m_g.rsquared:.4f}")
print("\nk2/m distribution (= the SA.B outcome): "
      f"mean {u2['k_over_m'].mean():.4f} sd {u2['k_over_m'].std():.4f} "
      f"min {u2['k_over_m'].min():.4f} max {u2['k_over_m'].max():.4f}")
print("r2/m distribution: "
      f"mean {u2['r_over_m'].mean():.4f} sd {u2['r_over_m'].std():.4f} "
      f"min {u2['r_over_m'].min():.4f} max {u2['r_over_m'].max():.4f}")
print(f"units with k2 == 0 (no AI-executed step, so EFI_def2 == 1 exactly): "
      f"{int((u2['k'] == 0).sum())} / {len(u2)}")

# Illustrative: regress ai_fraction on EFI_def2 alone, z-scored (what SA.B column 1 is close to)
zz = u2.copy()
for c in ['ai_fraction', 'EFI_nb', 'human_E1_fraction']:
    zz['z_' + c] = (zz[c] - zz[c].mean()) / zz[c].std()
m_h = r2_of(zz['z_ai_fraction'], zz[['z_EFI_nb']])
print(f"\nz-scored bivariate: ai_fraction on EFI_def2 -> slope {m_h.params[1]:+.4f}, R2 {m_h.rsquared:.4f}")

banner("PART 5.  MAIN SAMPLE -- distributions of m, k, r and the leftover variation in r/m")


def describe(s, name):
    return (f"{name:<10} mean {s.mean():>8.3f}  median {s.median():>7.2f}  sd {s.std():>8.3f}  "
            f"min {s.min():>7.3f}  max {s.max():>8.3f}")


print("EFI Definition 1 (E1|E2):")
for nm, col in [('m', 'm'), ('k', 'k'), ('r', 'r'),
                ('k/m', 'k_over_m'), ('r/m', 'r_over_m'), ('EFI', 'EFI_nb')]:
    print("  " + describe(u1[col], nm))
print("\nEFI Definition 2 (execution):")
for nm, col in [('m', 'm'), ('k2', 'k'), ('r2', 'r'),
                ('k2/m', 'k_over_m'), ('r2/m', 'r_over_m'), ('EFI2', 'EFI_nb')]:
    print("  " + describe(u2[col], nm))

print("\n--- independent variation left in r/m (Definition 1) ---")
d = u1.copy()
d['km_x_m'] = d['k_over_m'] * d['m']
specs = [
    ('r/m on k/m', ['k_over_m']),
    ('r/m on k/m + m', ['k_over_m', 'm']),
    ('r/m on k/m + m + k/m*m', ['k_over_m', 'm', 'km_x_m']),
]
print(f"unconditional sd(r/m) = {d['r_over_m'].std(ddof=1):.5f}")
for nm, cols in specs:
    res = r2_of(d['r_over_m'], d[cols])
    resid_sd = np.std(res.resid, ddof=len(cols) + 1)
    print(f"  {nm:<26} R2 = {res.rsquared:.4f}   residual sd = {resid_sd:.5f}   "
          f"({resid_sd / d['r_over_m'].std(ddof=1) * 100:.1f}% of unconditional sd)")

print(f"\nunconditional sd(EFI) = {d['EFI_nb'].std(ddof=1):.5f}")
for nm, cols in [('EFI on k/m', ['k_over_m']),
                 ('EFI on k/m + m', ['k_over_m', 'm']),
                 ('EFI on k/m + m + k/m*m', ['k_over_m', 'm', 'km_x_m'])]:
    res = r2_of(d['EFI_nb'], d[cols])
    resid_sd = np.std(res.resid, ddof=len(cols) + 1)
    print(f"  {nm:<26} R2 = {res.rsquared:.4f}   residual sd = {resid_sd:.5f}   "
          f"({resid_sd / d['EFI_nb'].std(ddof=1) * 100:.1f}% of unconditional sd)")

# residual of EFI on k/m must equal residual of r/m on k/m, by the identity
ra = r2_of(d['EFI_nb'], d[['k_over_m']]).resid
rb = r2_of(d['r_over_m'], d[['k_over_m']]).resid
print(f"\nmax |resid(EFI ~ k/m) - resid(r/m ~ k/m)| = {np.abs(ra - rb).max():.3e}  "
      "(they are the same object, by the identity)")

# with the num_E1E2_tasks control that the headline keeps: k itself
d['kk'] = d['k']
for nm, cols in [('EFI on k/m + k', ['k_over_m', 'kk']),
                 ('EFI on k/m + k + m', ['k_over_m', 'kk', 'm'])]:
    res = r2_of(d['EFI_nb'], d[cols])
    resid_sd = np.std(res.resid, ddof=len(cols) + 1)
    print(f"  {nm:<26} R2 = {res.rsquared:.4f}   residual sd = {resid_sd:.5f}   "
          f"({resid_sd / d['EFI_nb'].std(ddof=1) * 100:.1f}% of unconditional sd)")

u1.to_csv(os.path.join(OUT, "efi_identity_main_def1.csv"), index=False)
u2.to_csv(os.path.join(OUT, "efi_identity_main_def2.csv"), index=False)


# =====================================================================
# PART 4 : EVERY OTHER ESTIMATION SAMPLE
# =====================================================================
banner("PART 4a.  ALTERNATIVE-PROMPT ORDERINGS (analysis/onet_fragmentationIndex_robustness.ipynb)")
print("The robustness notebook does NOT apply the >=3-task filter (it is commented out),")
print("so the samples below are the raw merged files, as the notebook reads them.\n")
print(f"{'file':<38}{'occs':>6}{'rows':>8}   {'max|dev| d1':>13} {'corr(EFI,E1E2)':>16} "
      f"{'R2 on E1E2':>11}   {'max|dev| d2':>13} {'corr(EFI2,exec)':>16}")

rows_out = []
prompt_files = [("ONET_Eloundou_Anthropic_GPT.csv", "baseline (prompt 0)")] + \
               [(f"ONET_Eloundou_Anthropic_GPT_{x}.csv", f"prompt {x}") for x in range(1, 11)]
for fn, lab in prompt_files:
    p = os.path.join(DATA, "ONET_Eloundou_Anthropic_GPT", fn)
    if not os.path.exists(p):
        print(f"{fn:<38}  MISSING")
        continue
    df = pd.read_csv(p)
    a = identity_check(df, [OCC, TITLE], 'human_labels', ['E1', 'E2'])
    b = identity_check(df, [OCC, TITLE], 'label', ['Augmentation', 'Automation'])
    sh = (df.assign(_e12=df['human_labels'].isin(['E1', 'E2']).astype(int))
            .groupby([OCC, TITLE])['_e12'].mean().reset_index().rename(columns={'_e12': 'e12'}))
    aa = a.merge(sh, on=[OCC, TITLE])
    exec_sh = (df.assign(_x=df['label'].isin(['Augmentation', 'Automation']).astype(int))
                 .groupby([OCC, TITLE])['_x'].mean().reset_index().rename(columns={'_x': 'x'}))
    bb = b.merge(exec_sh, on=[OCC, TITLE])
    c1 = np.corrcoef(aa['EFI_nb'], aa['e12'])[0, 1]
    R1 = r2_of(aa['EFI_nb'], aa[['e12']]).rsquared
    c2 = np.corrcoef(bb['EFI_nb'], bb['x'])[0, 1]
    print(f"{fn:<38}{df[OCC].nunique():>6}{len(df):>8}   {a['dev'].max():>13.3e} "
          f"{c1:>+16.4f} {R1:>11.4f}   {b['dev'].max():>13.3e} {c2:>+16.4f}")
    rows_out.append(dict(sample=lab, file=fn, units=int(df[OCC].nunique()), rows=int(len(df)),
                         max_dev_def1=a['dev'].max(), corr_def1=c1, r2_def1=R1,
                         max_dev_def2=b['dev'].max(), corr_def2=c2))

banner("PART 4b.  FREQUENCY-PRUNED SAMPLES (onet_fragmentationIndex_weeklyTasks.ipynb / make_frag_def1_heatmap.py)")
MIN_TASKS_PER_OCC = 5
WEEKLY_PLUS_COLS = ['FT_More than weekly', 'FT_Daily', 'FT_Several times daily', 'FT_Hourly or more']
FAMILIES = [
    ('Weekly+', WEEKLY_PLUS_COLS),
    ('Daily+', ['FT_Daily', 'FT_Several times daily', 'FT_Hourly or more']),
    ('SeveralDaily+', ['FT_Several times daily', 'FT_Hourly or more']),
    ('Hourly+', ['FT_Hourly or more']),
]
CUTS = [('All tasks', None, None)]
CUTS.append(('Weekly+ >=50%', WEEKLY_PLUS_COLS, 50))     # the SA.E headline pruning
for lab, cols in FAMILIES[1:]:
    for t in [20, 35, 50, 65]:
        CUTS.append((f"{lab} >={t}%", cols, t))

print(f"{'cut':<22}{'occs':>6}{'rows':>8}   {'max|dev| d1':>13} {'corr(EFI,E1E2)':>16} "
      f"{'R2 on E1E2':>11} {'mean k/m':>10} {'mean r/m':>10} {'sd(r/m|k/m)':>12}")
for lab, cols, thr in CUTS:
    df = raw.copy()
    if cols is not None:
        df = df[df[cols].sum(axis=1) >= thr].reset_index(drop=True)
    c = df.groupby(OCC)['Task ID'].nunique()
    keep = c[c >= MIN_TASKS_PER_OCC].index
    df = df[df[OCC].isin(keep)].reset_index(drop=True)
    if df.empty or df[OCC].nunique() < 5:
        print(f"{lab:<22}{df[OCC].nunique():>6}{len(df):>8}   (too small)")
        continue
    a = identity_check(df, [OCC, TITLE], 'human_labels', ['E1', 'E2'])
    sh = (df.assign(_e12=df['human_labels'].isin(['E1', 'E2']).astype(int))
            .groupby([OCC, TITLE])['_e12'].mean().reset_index().rename(columns={'_e12': 'e12'}))
    aa = a.merge(sh, on=[OCC, TITLE])
    cc = np.corrcoef(aa['EFI_nb'], aa['e12'])[0, 1]
    RR = r2_of(aa['EFI_nb'], aa[['e12']]).rsquared
    resid = r2_of(aa['r_over_m'], aa[['k_over_m']]).resid
    print(f"{lab:<22}{df[OCC].nunique():>6}{len(df):>8}   {a['dev'].max():>13.3e} "
          f"{cc:>+16.4f} {RR:>11.4f} {aa['k_over_m'].mean():>10.4f} {aa['r_over_m'].mean():>10.4f} "
          f"{np.std(resid, ddof=2):>12.5f}")
    rows_out.append(dict(sample=lab, file='frequency cut', units=int(df[OCC].nunique()),
                         rows=int(len(df)), max_dev_def1=a['dev'].max(), corr_def1=cc, r2_def1=RR,
                         max_dev_def2=np.nan, corr_def2=np.nan))

banner("PART 4c.  APQC PCF PROCESS GROUPS (analysis/apqc_pcf_fragmentationIndex.ipynb)")
leaf_p = os.path.join(DATA, "apqc_pcf_fragmentation/pcf_leaf_matches.csv")
leaves = pd.read_csv(leaf_p, dtype={'hierarchy_id': str, 'parent_id': str})
leaves['sort_key'] = leaves['hierarchy_id'].apply(lambda h: tuple(int(x) for x in h.split('.')))
leaves = leaves.sort_values('sort_key').reset_index(drop=True)
print(f"PCF leaf elements read from {leaf_p}: {len(leaves):,}")


def unit_id(h, level):
    parts = h.split('.')
    if len(parts) < level:
        return None
    return f"{parts[0]}.0" if level == 1 else '.'.join(parts[:level])


for level, nm in [(1, 'Level 1 (Category)'), (2, 'Level 2 (Process Group)'), (3, 'Level 3 (Process)')]:
    d = leaves.copy()
    d['unit'] = d['hierarchy_id'].map(lambda h: unit_id(h, level))
    d = d.dropna(subset=['unit']).sort_values('sort_key').reset_index(drop=True)
    a = identity_check(d, ['unit'], 'human_labels', ['E1', 'E2'])
    b = identity_check(d, ['unit'], 'label', ['Augmentation', 'Automation'])
    keep = a['m'] >= 3
    a3 = a[keep].copy()
    b3 = b[b['m'] >= 3].copy()
    sh = (d.assign(_e12=d['human_labels'].isin(['E1', 'E2']).astype(int),
                   _e1=(d['human_labels'] == 'E1').astype(int),
                   _x=d['label'].isin(['Augmentation', 'Automation']).astype(int))
            .groupby('unit').agg(e12=('_e12', 'mean'), e1=('_e1', 'mean'), x=('_x', 'mean')).reset_index())
    a3 = a3.merge(sh, on='unit')
    b3 = b3.merge(sh, on='unit')
    print(f"\n--- {nm} (>= 3 steps): {len(a3)} units, {int(a3['m'].sum())} steps ---")
    print(f"  max |EFI_notebook - (m-k+r)/m|, Definition 1 = {a3['dev'].max():.3e}   "
          f"(all units incl. <3 steps: {a['dev'].max():.3e})")
    print(f"  max |EFI_notebook - (m-k+r)/m|, Definition 2 = {b3['dev'].max():.3e}")
    print(f"  corr(EFI_def1, E1|E2 share) = {np.corrcoef(a3['EFI_nb'], a3['e12'])[0,1]:+.4f}   "
          f"R2 = {r2_of(a3['EFI_nb'], a3[['e12']]).rsquared:.4f}")
    print(f"  corr(EFI_def1, E1 share)    = {np.corrcoef(a3['EFI_nb'], a3['e1'])[0,1]:+.4f}   "
          f"R2 = {r2_of(a3['EFI_nb'], a3[['e1']]).rsquared:.4f}")
    print(f"  corr(EFI_def2, exec share)  = {np.corrcoef(b3['EFI_nb'], b3['x'])[0,1]:+.4f}   "
          f"R2 = {r2_of(b3['EFI_nb'], b3[['x']]).rsquared:.4f}")
    print(f"  m: mean {a3['m'].mean():.1f} median {a3['m'].median():.0f} min {a3['m'].min()} max {a3['m'].max()}")
    print(f"  k/m: mean {a3['k_over_m'].mean():.4f} sd {a3['k_over_m'].std():.4f} "
          f"min {a3['k_over_m'].min():.4f} max {a3['k_over_m'].max():.4f}")
    print(f"  r/m: mean {a3['r_over_m'].mean():.4f} sd {a3['r_over_m'].std():.4f}")
    resid = r2_of(a3['r_over_m'], a3[['k_over_m']]).resid
    print(f"  sd(r/m | k/m) = {np.std(resid, ddof=2):.5f}  "
          f"({np.std(resid, ddof=2)/a3['r_over_m'].std(ddof=1)*100:.1f}% of unconditional)")
    rows_out.append(dict(sample=f"APQC {nm}", file='pcf_leaf_matches.csv', units=int(len(a3)),
                         rows=int(a3['m'].sum()), max_dev_def1=a3['dev'].max(),
                         corr_def1=np.corrcoef(a3['EFI_nb'], a3['e12'])[0, 1],
                         r2_def1=r2_of(a3['EFI_nb'], a3[['e12']]).rsquared,
                         max_dev_def2=b3['dev'].max(),
                         corr_def2=np.corrcoef(b3['EFI_nb'], b3['x'])[0, 1]))

# cross-check my rebuilt Level 2 against the repo's saved units_level2.csv
saved = pd.read_csv(os.path.join(DATA, "apqc_pcf_fragmentation/units_level2.csv"),
                    dtype={'unit': str})
d2 = leaves.copy()
d2['unit'] = d2['hierarchy_id'].map(lambda h: unit_id(h, 2))
d2 = d2.dropna(subset=['unit']).sort_values('sort_key').reset_index(drop=True)
mine = identity_check(d2, ['unit'], 'human_labels', ['E1', 'E2'])
mine = mine[mine['m'] >= 3]
chk = saved.merge(mine, on='unit', how='inner')
print(f"\ncross-check against repo file units_level2.csv: {len(chk)} of {len(saved)} units matched; "
      f"max |EFI_saved - EFI_mine| = {(chk['fragmentation_index'] - chk['EFI_nb']).abs().max():.3e}; "
      f"max |num_steps_saved - m| = {(chk['num_steps'] - chk['m']).abs().max()}")

pd.DataFrame(rows_out).to_csv(os.path.join(OUT, "efi_identity_all_samples.csv"), index=False)


# =====================================================================
# PART 6 : what the identity does to the standard errors
# =====================================================================
banner("PART 6.  MAIN SAMPLE -- collinearity the identity forces on the headline specification")

reg = u1[[OCC, TITLE, 'EFI_nb', 'k_over_m', 'human_E1_fraction', 'k', 'm']].copy()
reg = reg.rename(columns={'EFI_nb': 'EFI', 'k_over_m': 'e12', 'human_E1_fraction': 'e1',
                          'k': 'numE1E2'})
soc = (main[[OCC, 'Major_Group_Code', 'Minor_Group_Code']]
       .drop_duplicates(subset=[OCC]))
reg = reg.merge(soc, on=OCC, how='left')


def partial_r2(y, Xcols, fe=None):
    X = reg[Xcols].astype(float).copy()
    if fe is not None:
        D = pd.get_dummies(reg[fe].astype(str), drop_first=True).astype(float)
        X = pd.concat([X.reset_index(drop=True), D.reset_index(drop=True)], axis=1)
    res = r2_of(reg[y], X)
    return res


print("R2 of the EFI on everything ELSE in the regression (this is what the SE pays for):")
print(f"{'specification':<52}{'R2':>9}{'VIF':>9}{'SE inflation':>15}")
grid = [
    ("PUBLISHED  no FE : EFI ~ E1 share + numE1E2", ['e1', 'numE1E2'], None),
    ("PUBLISHED  MajFE : EFI ~ E1 share + numE1E2 + MajorFE", ['e1', 'numE1E2'], 'Major_Group_Code'),
    ("PUBLISHED  MinFE : EFI ~ E1 share + numE1E2 + MinorFE", ['e1', 'numE1E2'], 'Minor_Group_Code'),
    ("MATCHED    no FE : EFI ~ E1|E2 share + numE1E2", ['e12', 'numE1E2'], None),
    ("MATCHED    MajFE : EFI ~ E1|E2 share + numE1E2 + MajorFE", ['e12', 'numE1E2'], 'Major_Group_Code'),
    ("MATCHED    MinFE : EFI ~ E1|E2 share + numE1E2 + MinorFE", ['e12', 'numE1E2'], 'Minor_Group_Code'),
    ("MATCHED no ctrl  : EFI ~ E1|E2 share", ['e12'], None),
]
for lab, cols, fe in grid:
    res = partial_r2('EFI', cols, fe)
    R2 = res.rsquared
    vif = 1.0 / (1.0 - R2)
    print(f"{lab:<52}{R2:>9.4f}{vif:>9.2f}{np.sqrt(vif):>14.2f}x")

print("\nSame thing expressed as leftover identifying variation in the EFI:")
for lab, cols, fe in grid:
    res = partial_r2('EFI', cols, fe)
    sd_resid = np.std(res.resid, ddof=1)
    print(f"  {lab:<52} sd(EFI | others) = {sd_resid:.5f}  "
          f"= {sd_resid / reg['EFI'].std(ddof=1) * 100:5.1f}% of sd(EFI)")

banner("DONE")
print("wrote", os.path.join(OUT, "efi_identity_all_samples.csv"))
print("wrote", os.path.join(OUT, "efi_identity_main_def1.csv"))
print("wrote", os.path.join(OUT, "efi_identity_main_def2.csv"))
