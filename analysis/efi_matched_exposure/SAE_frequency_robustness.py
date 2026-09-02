"""
Re-estimation of the SA.E frequency-pruning fragmentation sweep.

Reimplements, from scratch, the construction in
  analysis/onet_fragmentationIndex_weeklyTasks.ipynb  (cells 6, 7, 17)
  analysis/make_frag_def1_heatmap.py
under BOTH the published spec (exposure = human_E1_fraction) and the
MATCHED spec (exposure = human_aiExposure_fraction = E1|E2 share).

Writes its output to data/computed_objects/efi_matched_exposure/. Does not touch any
published exhibit under writeup/tables/ or the paper's own computed objects.
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
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

INFILE = os.path.join(REPO, "data/computed_objects/ONET_Eloundou_Anthropic_GPT/ONET_Eloundou_Anthropic_GPT.csv")
CLEANED = os.path.join(REPO, "data/computed_objects/ONET_cleaned_tasks.csv")

OCC = 'O*NET-SOC Code'
TITLE = 'Occupation Title'
MIN_TASKS_PER_OCC = 5           # SA.E floor, applied to the PRUNED workflow

# ---------------------------------------------------------------- sample defs
FAMILIES = [
    ('Daily+',        'daily',    ['FT_Daily', 'FT_Several times daily', 'FT_Hourly or more']),
    ('SeveralDaily+', 'sevdaily', ['FT_Several times daily', 'FT_Hourly or more']),
    ('Hourly+',       'hourly',   ['FT_Hourly or more']),
]
SWEEP_THRESHOLDS = [20, 35, 50, 65]
CUTS = [('All tasks', 'all', None, None)]
for lab, tag, cols in FAMILIES:
    for t in SWEEP_THRESHOLDS:
        CUTS.append((f"{lab} >={t}%", tag, cols, t))

RAW = pd.read_csv(INFILE)
ONETC = pd.read_csv(CLEANED)
SOC_mappings = ONETC[[OCC, TITLE, 'Major_Group_Code', 'Major_Group_Title',
                      'Minor_Group_Code', 'Minor_Group_Title',
                      'Broad_Occupation_Code', 'Broad_Occupation_Title',
                      'Detailed_Occupation_Code', 'Detailed_Occupation_Title']].copy()
SOC_mappings = SOC_mappings.drop_duplicates(subset=[OCC])

_VOS_RAW = RAW[[OCC, 'Task ID', 'FT_Daily', 'FT_Several times daily', 'FT_Hourly or more']].copy()
_VOS_CACHE = {}


def valid_occ_set(family_cols, threshold):
    key = (None if family_cols is None else tuple(family_cols), threshold)
    if key not in _VOS_CACHE:
        d = _VOS_RAW if family_cols is None else _VOS_RAW[_VOS_RAW[family_cols].sum(axis=1) >= threshold]
        cnt = d.groupby(OCC)['Task ID'].nunique()
        _VOS_CACHE[key] = set(cnt[cnt >= MIN_TASKS_PER_OCC].index)
    return _VOS_CACHE[key]


def prepare_cut(family_cols, threshold):
    df = RAW.copy()                       # fresh copy each time; original CSV row order preserved
    if family_cols is not None:
        df = df[df[family_cols].sum(axis=1) >= threshold].reset_index(drop=True)
    df = df[df[OCC].isin(valid_occ_set(family_cols, threshold))].reset_index(drop=True)
    return df


# ------------------------------------------------------------- pipeline parts
def create_occupation_analysis(df):
    rows = []
    for (soc, title), g in df.groupby([OCC, TITLE]):
        total = len(g)
        aug = (g['label'] == 'Augmentation').sum() / total
        aut = (g['label'] == 'Automation').sum() / total
        h1 = (g['human_labels'] == 'E1').sum() / total
        h2 = (g['human_labels'] == 'E2').sum() / total
        rows.append({OCC: soc, TITLE: title,
                     'num_tasks': g['Task ID'].nunique(),
                     'ai_fraction': aug + aut,
                     'human_E1_fraction': h1,
                     'human_E2_fraction': h2,
                     'human_aiExposure_fraction': h1 + h2,
                     'num_E1E2_tasks': int(g['human_labels'].isin(['E1', 'E2']).sum())})
    return pd.DataFrame(rows)


def construct_fragmentation_index(df):
    """EFI Definition 1: a step is AI-able iff human_labels in {E1,E2}; consecutive
    AI-able steps merge.  num_switches = 1 unless (is_ai & next_is_ai) -> 0; EFI = mean."""
    f = df.copy()
    f['is_ai'] = f['human_labels'].isin(['E1', 'E2']).astype(int)
    f['next_is_ai'] = f.groupby([OCC, TITLE])['is_ai'].shift(-1).fillna(0).astype(int)
    f['num_switches'] = 1
    f.loc[(f['is_ai'] == 1) & (f['next_is_ai'] == 1), 'num_switches'] = 0
    return (f.groupby([OCC, TITLE])['num_switches'].mean().reset_index()
             .rename(columns={'num_switches': 'fragmentation_index'}))


def build_agg(df, exposure_var):
    occ = create_occupation_analysis(df)
    fi = construct_fragmentation_index(df)
    occ = occ.merge(fi, on=[OCC, TITLE], how='left')
    occ = occ.merge(SOC_mappings, on=[OCC, TITLE], how='left')
    agg = occ.groupby([OCC, TITLE]).agg({
        'fragmentation_index': 'mean', exposure_var: 'mean', 'ai_fraction': 'mean',
        'num_tasks': 'mean', 'num_E1E2_tasks': 'mean'}).reset_index()
    agg = agg.merge(SOC_mappings.drop_duplicates(subset=[OCC]), on=OCC, how='left', suffixes=('', '_drop'))
    agg = agg.loc[:, ~agg.columns.str.endswith('_drop')]
    agg = agg.rename(columns={exposure_var: 'ai_exposure'})
    for c in ('Major_Group_Code', 'Minor_Group_Code', OCC):
        agg[c] = agg[c].astype('object')
    for z in ['ai_fraction', 'ai_exposure', 'fragmentation_index', 'num_E1E2_tasks']:
        s = agg[z]
        sd = s.std()
        agg[z] = (s - s.mean()) / sd if (sd and not np.isnan(sd)) else np.nan
    return agg


def run_cell(df, exposure_var):
    agg = build_agg(df, exposure_var)
    n_occ = int(len(agg))
    clu = dict(cov_type="cluster", cov_kwds={"groups": agg[OCC], "use_correction": True, "df_correction": True})
    out = {}
    for fe_name, fe_term in [('none', ''), ('Major', ' + C(Major_Group_Code)'), ('Minor', ' + C(Minor_Group_Code)')]:
        try:
            if n_occ < 10 or agg['fragmentation_index'].nunique() < 2:
                raise ValueError('too small')
            m = smf.ols(f'ai_fraction ~ fragmentation_index + ai_exposure + num_E1E2_tasks{fe_term}', data=agg).fit(**clu)
            out[fe_name] = dict(coef=m.params['fragmentation_index'], se=m.bse['fragmentation_index'],
                                pval=m.pvalues['fragmentation_index'],
                                exp_coef=m.params['ai_exposure'], exp_se=m.bse['ai_exposure'],
                                exp_pval=m.pvalues['ai_exposure'],
                                nctl_coef=m.params['num_E1E2_tasks'], nctl_pval=m.pvalues['num_E1E2_tasks'],
                                r2=m.rsquared, N_occ=n_occ)
        except Exception as e:
            out[fe_name] = dict(coef=np.nan, se=np.nan, pval=np.nan, exp_coef=np.nan, exp_se=np.nan,
                                exp_pval=np.nan, nctl_coef=np.nan, nctl_pval=np.nan, r2=np.nan,
                                N_occ=n_occ, err=repr(e))
    return out


# --------------------------------------------------------------------- sweep
rows = []
for label, fam, cols, thr in CUTS:
    d = prepare_cut(cols, thr)
    for spec, evar in [('OLD', 'human_E1_fraction'), ('MATCHED', 'human_aiExposure_fraction')]:
        res = run_cell(d, evar)
        for fe, r in res.items():
            rows.append(dict(cut=label, family=fam, threshold=(thr if thr is not None else 0),
                             spec=spec, FE=fe, **r))
    print(f"done {label}", file=sys.stderr)

sweep = pd.DataFrame(rows)
sweep.to_csv(os.path.join(OUT, "sae_frequency_sweep_old_vs_matched.csv"), index=False)


def star(p):
    return '***' if (pd.notna(p) and p < .01) else '**' if (pd.notna(p) and p < .05) else '*' if (pd.notna(p) and p < .1) else ''


fam_order = ['all', 'daily', 'sevdaily', 'hourly']
fam_label = {'all': 'All tasks', 'daily': 'Daily+', 'sevdaily': 'SeveralDaily+', 'hourly': 'Hourly+'}

print("\n" + "=" * 118)
print("EFI (Definition 1, E1|E2) COEFFICIENT  -- OLD (exposure=E1 share)  vs  MATCHED (exposure=E1|E2 share)")
print("=" * 118)
for fe in ['none', 'Major', 'Minor']:
    print(f"\n--- FE = {fe} ---")
    print(f"{'cut':<22}{'N':>6}   {'OLD coef(se)':>22} {'p':>9}   {'MATCHED coef(se)':>22} {'p':>9}")
    for fam in fam_order:
        ts = [0] if fam == 'all' else SWEEP_THRESHOLDS
        for t in ts:
            o = sweep[(sweep.family == fam) & (sweep.threshold == t) & (sweep.spec == 'OLD') & (sweep.FE == fe)]
            n = sweep[(sweep.family == fam) & (sweep.threshold == t) & (sweep.spec == 'MATCHED') & (sweep.FE == fe)]
            if not len(o):
                continue
            o = o.iloc[0]; n = n.iloc[0]
            lab = fam_label[fam] if fam == 'all' else f"{fam_label[fam]} >={t}%"
            print(f"{lab:<22}{int(o.N_occ):>6}   "
                  f"{o.coef:>+8.3f} ({o.se:.3f}){star(o.pval):<3} {o.pval:>9.4f}   "
                  f"{n.coef:>+8.3f} ({n.se:.3f}){star(n.pval):<3} {n.pval:>9.4f}")

print("\n" + "=" * 118)
print("EXPOSURE COEFFICIENT  -- OLD (E1 share)  vs  MATCHED (E1|E2 share)")
print("=" * 118)
for fe in ['none', 'Major', 'Minor']:
    print(f"\n--- FE = {fe} ---")
    print(f"{'cut':<22}{'N':>6}   {'OLD exp(se)':>22} {'p':>9}   {'MATCHED exp(se)':>22} {'p':>9}")
    for fam in fam_order:
        ts = [0] if fam == 'all' else SWEEP_THRESHOLDS
        for t in ts:
            o = sweep[(sweep.family == fam) & (sweep.threshold == t) & (sweep.spec == 'OLD') & (sweep.FE == fe)]
            n = sweep[(sweep.family == fam) & (sweep.threshold == t) & (sweep.spec == 'MATCHED') & (sweep.FE == fe)]
            if not len(o):
                continue
            o = o.iloc[0]; n = n.iloc[0]
            lab = fam_label[fam] if fam == 'all' else f"{fam_label[fam]} >={t}%"
            print(f"{lab:<22}{int(o.N_occ):>6}   "
                  f"{o.exp_coef:>+8.3f} ({o.exp_se:.3f}){star(o.exp_pval):<3} {o.exp_pval:>9.4f}   "
                  f"{n.exp_coef:>+8.3f} ({n.exp_se:.3f}){star(n.exp_pval):<3} {n.exp_pval:>9.4f}")

# --------------------------------------------------- significance tallies
print("\n" + "=" * 90)
print("SIGNIFICANT-CELL TALLIES over the 13 cuts x 3 FE grid (39 cells per spec)")
print("=" * 90)
for spec in ['OLD', 'MATCHED']:
    print(f"\n{spec}:")
    for fe in ['none', 'Major', 'Minor']:
        s = sweep[(sweep.spec == spec) & (sweep.FE == fe)]
        neg5 = ((s.coef < 0) & (s.pval < .05)).sum()
        neg10 = ((s.coef < 0) & (s.pval < .10)).sum()
        neg1 = ((s.coef < 0) & (s.pval < .01)).sum()
        pos5 = ((s.coef > 0) & (s.pval < .05)).sum()
        negsign = (s.coef < 0).sum()
        print(f"  FE={fe:<6} n_cells={len(s):>3}  neg sign={negsign:>3}  "
              f"sig-neg @1%={neg1:>2}  @5%={neg5:>2}  @10%={neg10:>2}   sig-pos @5%={pos5:>2}")
    s = sweep[sweep.spec == spec]
    print(f"  ALL FE   n_cells={len(s):>3}  neg sign={(s.coef<0).sum():>3}  "
          f"sig-neg @5%={((s.coef<0)&(s.pval<.05)).sum():>2}  sig-pos @5%={((s.coef>0)&(s.pval<.05)).sum():>2}")
    print(f"  coef range: [{s.coef.min():+.3f}, {s.coef.max():+.3f}]  "
          f"excl. Hourly+>=65%: [{s[~((s.family=='hourly')&(s.threshold==65))].coef.min():+.3f}, "
          f"{s[~((s.family=='hourly')&(s.threshold==65))].coef.max():+.3f}]")
    e = sweep[sweep.spec == spec]
    print(f"  exposure: sig-pos @5% = {((e.exp_coef>0)&(e.exp_pval<.05)).sum():>2}/{len(e)}  "
          f"@1% = {((e.exp_coef>0)&(e.exp_pval<.01)).sum():>2}/{len(e)}  "
          f"range [{e.exp_coef.min():+.3f}, {e.exp_coef.max():+.3f}]")

print("\nWrote", os.path.join(OUT, "sae_frequency_sweep_old_vs_matched.csv"))
