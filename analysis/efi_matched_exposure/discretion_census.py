#!/usr/bin/env python
"""(a) the mechanism behind the 872 vs 871 gap; (b) full discretion census."""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import arrangement_statistics as A

import os as _os
# Repo root and output dir are derived from this file's own location, so the folder
# can move and no home directory is baked in. Generated output follows the repo
# convention: data/computed_objects/<name of this analysis folder>/.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
OUT = _os.path.join(REPO, "data", "computed_objects", "efi_matched_exposure")
_os.makedirs(OUT, exist_ok=True)


OCC, TIT = A.OCC, A.TIT


def add_stats(occ):
    kk = occ['k_E1E2'].to_numpy(); mm = occ['m'].to_numpy(); rr = occ['r_E1E2'].to_numpy()
    occ = occ.copy()
    occ['C1_r_over_m'] = rr / mm
    occ['C2_r_over_k'] = np.where(kk >= 1, rr / np.maximum(kk, 1), np.nan)
    occ['C3_adj_share'] = np.where(kk >= 1, (kk - rr) / np.maximum(kk - 1, 1), np.nan)
    occ['C6_longest_over_k'] = np.where(kk >= 1, occ['lr_E1E2'] / np.maximum(kk, 1), np.nan)
    occ['level_term'] = -kk / mm
    occ['arr_term'] = rr / mm
    ER = kk * (mm - kk + 1) / mm
    VR = kk * (kk - 1) * (mm - kk) * (mm - kk + 1) / (mm.astype(float) ** 2 * (mm - 1))
    SD = np.sqrt(np.maximum(VR, 0))
    occ['C5_z'] = np.where(SD > 0, (rr - ER) / SD, np.nan)
    occ['C5_z0'] = occ['C5_z'].fillna(0.0)
    occ['logm'] = np.log(mm)
    return occ


occ872 = add_stats(A.build_panel(A.load_merged(False)))
occ871 = add_stats(A.build_panel(A.load_merged(True)))

print("=" * 126)
print("A. MECHANISM: what does an unlabelled task do to the EFI?")
print("=" * 126)
j = occ872.merge(occ871, on=OCC, suffixes=('_872', '_871'))
print(f"  occupations in both: {len(j)}")
d_m = j['m_872'] - j['m_871']
d_r = j['r_E1E2_872'] - j['r_E1E2_871']
d_k = j['k_E1E2_872'] - j['k_E1E2_871']
print(f"  k identical in both samples (unlabelled rows are never E1|E2): max |dk| = {d_k.abs().max()}")
print(f"  m falls by:  mean {d_m.mean():.3f}, >0 for {(d_m>0).sum()} occupations, max {d_m.max()}")
print(f"  r falls by:  mean {d_r.mean():.3f}, >0 for {(d_r>0).sum()} occupations, max {d_r.max()}")
print(f"  i.e. {int(d_r.sum())} maximal runs across the panel exist ONLY because an unlabelled")
print(f"  task sits inside a block of E1|E2 tasks and the notebook codes it as 'not AI-able',")
print(f"  splitting the block. Total runs in the 872 panel: {int(occ872['r_E1E2'].sum())}; "
      f"in the 871 panel: {int(occ871['r_E1E2'].sum())}"
      f"  ({d_r.sum()/occ872['r_E1E2'].sum():.1%} of all runs are such artefacts).")
print(f"  EFI change: mean {(j['EFI_E1E2_872']-j['EFI_E1E2_871']).mean():+.4f}, "
      f"corr(EFI_872, EFI_871) = {j['EFI_E1E2_872'].corr(j['EFI_E1E2_871']):.4f}")
print(f"  r/m change: corr(r/m_872, r/m_871) = {j['C1_r_over_m_872'].corr(j['C1_r_over_m_871']):.4f}")
print(f"  perm z:     corr(z_872, z_871) = "
      f"{j['C5_z0_872'].corr(j['C5_z0_871']):.4f}")
print(f"  the 364 occupations with unlabelled tasks: mean ai_fraction "
      f"{j.loc[d_m>0,'ai_fraction_872'].mean():.4f} vs {j.loc[d_m==0,'ai_fraction_872'].mean():.4f} "
      f"for the rest")

print("\n" + "=" * 126)
print("B. DISCRETION CENSUS: over every defensible matched cell, how many are")
print("   significant at 5% with the theory-predicted sign?")
print("=" * 126)
stats = [('EFI (E1|E2)', 'EFI_E1E2', None, '-'),
         ('C1 r/m', 'C1_r_over_m', None, '-'),
         ('C2 r/k (k>=1)', 'C2_r_over_k', 'k1', '-'),
         ('C3 adj share (k>=1)', 'C3_adj_share', 'k1', '+'),
         ('C4 arrangement term', 'arr_term', None, '-'),
         ('C5 perm z (non-deg)', 'C5_z', 'z', '-'),
         ('C5b perm z (deg=0)', 'C5_z0', None, '-'),
         ('C6 longest/k (k>=1)', 'C6_longest_over_k', 'k1', '+')]
ctrls = [('k', ['k_E1E2']), ('none', []), ('logm', ['logm']), ('k+logm', ['k_E1E2', 'logm'])]
rows = []
for sname, base in [('872 (notebook, unlabelled kept)', occ872),
                    ('871 (unlabelled rows dropped)', occ871)]:
    for nm, col, msk, sign in stats:
        if msk == 'k1':
            sub0 = base[base['k_E1E2'] >= 1].reset_index(drop=True)
        elif msk == 'z':
            sub0 = base[base['C5_z'].notna()].reset_index(drop=True)
        else:
            sub0 = base
        for cn, cl in ctrls:
            for fe in ('none', 'major', 'minor'):
                extra = (['share_E1E2'] + cl) if col != 'arr_term' else (['level_term'] + cl)
                m = A.fit(sub0, col, extra, fe)
                p = A.pack(m, col)
                ok = (p['p'] < 0.05) and ((p['b'] < 0) if sign == '-' else (p['b'] > 0))
                rows.append(dict(sample=sname, stat=nm, sign=sign, ctrl=cn, fe=fe,
                                 b=p['b'], se=p['se'], p=p['p'], n=p['n'], mde=p['mde'],
                                 hit=ok))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, 'arrangement_census.csv'), index=False)
print(f"  total cells: {len(R)}   significant at 5% with the predicted sign: {R.hit.sum()}")
print(f"  by sample:\n{R.groupby('sample')['hit'].agg(['sum','count']).to_string()}")
print(f"\n  ALL hits:")
if R.hit.sum():
    print(R[R.hit][['sample', 'stat', 'ctrl', 'fe', 'b', 'se', 'p', 'n']].to_string(index=False))
print(f"\n  significant at 5% with the WRONG sign: {(~R.hit & (R.p<0.05)).sum()}")
w = R[(~R.hit) & (R.p < 0.05)]
if len(w):
    print(w[['sample', 'stat', 'sign', 'ctrl', 'fe', 'b', 'se', 'p', 'n']].to_string(index=False))

print("\n  headline-control-only slice (ctrl=k), which is what the author has fixed:")
h = R[R.ctrl == 'k']
print(f"    cells {len(h)}, hits {h.hit.sum()}")
print(h[['sample', 'stat', 'fe', 'b', 'se', 'p', 'n', 'hit']].to_string(index=False))

print("\n" + "=" * 126)
print("C. MDE COMPARISON (headline control k, 872 sample) -- 2.802 * SE, z units")
print("=" * 126)
print(f"  {'statistic':<24}{'FE':<8}{'b':>10}{'SE':>9}{'p':>9}{'MDE':>9}"
      f"{'MDE rel. to EFI':>18}")
for fe in ('none', 'major', 'minor'):
    ref = A.fit(occ872, 'EFI_E1E2', ['share_E1E2', 'k_E1E2'], fe)
    refmde = 2.802 * ref.bse['EFI_E1E2']
    for nm, col, msk, sign in stats:
        if msk == 'k1':
            sub0 = occ872[occ872['k_E1E2'] >= 1].reset_index(drop=True)
        elif msk == 'z':
            sub0 = occ872[occ872['C5_z'].notna()].reset_index(drop=True)
        else:
            sub0 = occ872
        extra = (['share_E1E2', 'k_E1E2']) if col != 'arr_term' else (['level_term', 'k_E1E2'])
        m = A.fit(sub0, col, extra, fe)
        p = A.pack(m, col)
        print(f"  {nm:<24}{fe:<8}{p['b']:>+10.4f}{p['se']:>9.4f}{p['p']:>9.4f}"
              f"{p['mde']:>9.4f}{p['mde']/refmde:>18.3f}")
