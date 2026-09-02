"""
Addendum: what the Definition-2 (execution-based) EFI identity means for
the SA.B table (writeup/tables/fragmentation_index_regression_execution.tex).

Definition 2 sets a step AI-able iff label in {Augmentation, Automation}.  Write
    y = k2/m = share of steps executed by AI  (this is the DEPENDENT VARIABLE of SA.B)
    a = r2/m = AI-execution blocks per step   (the arrangement term)
Then, exactly,
    EFI_def2 = 1 - y + a.
So SA.B regresses y on (1 - y + a) plus controls.  This script (i) replicates SA.B's
published numbers, (ii) shows that swapping EFI_def2 for (a - y) reproduces them to
machine precision, and (iii) shows what is left when the level term is removed.

Writes its output to data/computed_objects/efi_matched_exposure/. Does not touch any
published exhibit under writeup/tables/ or the paper's own computed objects.
"""
import os
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

MAIN = os.path.join(REPO, "data/computed_objects/ONET_Eloundou_Anthropic_GPT/ONET_Eloundou_Anthropic_GPT.csv")
CLEANED = os.path.join(REPO, "data/computed_objects/ONET_cleaned_tasks.csv")
OCC, TITLE = 'O*NET-SOC Code', 'Occupation Title'

raw = pd.read_csv(MAIN)
cnt = raw.groupby(OCC)['Task ID'].nunique()
raw = raw[raw[OCC].isin(cnt[cnt >= 3].index)].reset_index(drop=True)

f = raw.copy()
f['is_ai'] = f['label'].isin(['Augmentation', 'Automation']).astype(int)
f['next_is_ai'] = f.groupby([OCC, TITLE])['is_ai'].shift(-1).fillna(0).astype(int)
f['prev_is_ai'] = f.groupby([OCC, TITLE])['is_ai'].shift(1).fillna(0).astype(int)
f['num_switches'] = 1
f.loc[(f['is_ai'] == 1) & (f['next_is_ai'] == 1), 'num_switches'] = 0
f['run_start'] = ((f['is_ai'] == 1) & (f['prev_is_ai'] == 0)).astype(int)
f['e1'] = (f['human_labels'] == 'E1').astype(int)
f['e12'] = f['human_labels'].isin(['E1', 'E2']).astype(int)

g = f.groupby([OCC, TITLE])
d = pd.DataFrame({
    'm': g.size(),
    'k2': g['is_ai'].sum(),
    'r2': g['run_start'].sum(),
    'EFI2': g['num_switches'].mean(),
    'ai_exposure_E1': g['e1'].mean(),
    'ai_exposure_E1E2': g['e12'].mean(),
    'numE1E2': g['e12'].sum(),
}).reset_index()
d['ai_fraction'] = d['k2'] / d['m']       # the SA.B outcome
d['arrangement'] = d['r2'] / d['m']       # a
d['minus_level_plus_arr'] = d['arrangement'] - d['ai_fraction']   # = EFI2 - 1

print(f"max |EFI2 - (1 - ai_fraction + arrangement)| = "
      f"{(d['EFI2'] - (1 - d['ai_fraction'] + d['arrangement'])).abs().max():.3e}   (n={len(d)})")

soc = pd.read_csv(CLEANED)[[OCC, 'Major_Group_Code', 'Minor_Group_Code']].drop_duplicates(subset=[OCC])
d = d.merge(soc, on=OCC, how='left')
for c in ('Major_Group_Code', 'Minor_Group_Code', OCC):
    d[c] = d[c].astype('object')

z = d.copy()
for c in ['ai_fraction', 'ai_exposure_E1', 'ai_exposure_E1E2', 'EFI2', 'numE1E2',
          'arrangement', 'minus_level_plus_arr']:
    z['z_' + c] = (z[c] - z[c].mean()) / z[c].std()

clu = dict(cov_type='cluster', cov_kwds={'groups': z[OCC], 'use_correction': True, 'df_correction': True})
FE = [('no FE', ''), ('Major FE', ' + C(Major_Group_Code)'), ('Minor FE', ' + C(Minor_Group_Code)')]


def show(title, rhs, keys):
    print(f"\n{title}")
    for nm, fe in FE:
        m = smf.ols(f'z_ai_fraction ~ {rhs}{fe}', data=z).fit(**clu)
        parts = []
        for kk in keys:
            st = '***' if m.pvalues[kk] < .01 else '**' if m.pvalues[kk] < .05 else '*' if m.pvalues[kk] < .1 else ''
            parts.append(f"{kk} {m.params[kk]:+.3f} ({m.bse[kk]:.3f}){st:<3}")
        print(f"  {nm:<9} " + "  ".join(parts) + f"   R2 {m.rsquared:.3f}  N {int(m.nobs)}")


print("\n" + "=" * 96)
print("A. PUBLISHED SA.B spec: y = AI-execution share on EFI_def2 + E1 share + numE1E2")
print("   (paper prints -0.78 / -0.70 / -0.68 with R2 0.84 / 0.86 / 0.89)")
print("=" * 96)
show("", 'z_EFI2 + z_ai_exposure_E1 + z_numE1E2', ['z_EFI2', 'z_ai_exposure_E1'])

print("\n" + "=" * 96)
print("B. IDENTICAL regression with EFI_def2 replaced by (arrangement - outcome), which")
print("   differs from EFI_def2 only by the constant 1 -- the coefficients must coincide")
print("=" * 96)
show("", 'z_minus_level_plus_arr + z_ai_exposure_E1 + z_numE1E2',
     ['z_minus_level_plus_arr', 'z_ai_exposure_E1'])

print("\n" + "=" * 96)
print("C. ARRANGEMENT TERM ONLY: y on r2/m + E1 share + numE1E2")
print("   (the level term -y is removed; this is what 'fragmentation' would mean")
print("    if the index did not contain the dependent variable)")
print("=" * 96)
show("", 'z_arrangement + z_ai_exposure_E1 + z_numE1E2', ['z_arrangement', 'z_ai_exposure_E1'])

print("\n" + "=" * 96)
print("D. Variance decomposition of EFI_def2 = 1 - y + a")
print("=" * 96)
vy, va = d['ai_fraction'].var(ddof=1), d['arrangement'].var(ddof=1)
cov_ya = np.cov(d['ai_fraction'], d['arrangement'], ddof=1)[0, 1]
print(f"  Var(EFI2)     = {d['EFI2'].var(ddof=1):.6f}")
print(f"  Var(y)        = {vy:.6f}   sd {np.sqrt(vy):.4f}")
print(f"  Var(a)        = {va:.6f}   sd {np.sqrt(va):.4f}")
print(f"  Cov(y,a)      = {cov_ya:.6f}   corr(y,a) = {cov_ya/np.sqrt(vy*va):+.4f}")
print(f"  Var(y)+Var(a)-2Cov = {vy + va - 2*cov_ya:.6f}")
print(f"  corr(EFI2, y) = {np.corrcoef(d['EFI2'], d['ai_fraction'])[0,1]:+.4f}")
print(f"  corr(EFI2, a) = {np.corrcoef(d['EFI2'], d['arrangement'])[0,1]:+.4f}")
print(f"  occupations with k2 = 0 (EFI2 exactly 1, y exactly 0): "
      f"{int((d['k2']==0).sum())} of {len(d)} = {(d['k2']==0).mean()*100:.1f}%")
