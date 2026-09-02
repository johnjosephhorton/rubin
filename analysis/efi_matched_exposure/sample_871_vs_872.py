#!/usr/bin/env python
"""
follow-up.

(1) Decompose the 872 -> 871 gap: is it the dropped occupation, or the re-computed
    denominators for the other occupations with unlabelled tasks?
(2) Influence diagnostics on the matched EFI coefficient (DFBETA), 872 sample.
(3) Re-run every alternative arrangement statistic on the 871 (labelled-rows-only) sample.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
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
SCRATCH = os.path.dirname(os.path.abspath(__file__))


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


md872 = A.load_merged(False)
occ872 = add_stats(A.build_panel(md872))
md871 = A.load_merged(True)
occ871 = add_stats(A.build_panel(md871))

DROPPED = '33-3051.00'
occ871a = occ872[occ872[OCC] != DROPPED].reset_index(drop=True)

print("=" * 128)
print("1. DECOMPOSING THE 872 -> 871 GAP, matched spec (exposure = E1|E2 share, EFI on E1|E2)")
print("=" * 128)
g = md872[md872[OCC] == DROPPED]
row = occ872[occ872[OCC] == DROPPED].iloc[0]
print(f"the occupation the 871 run drops: {DROPPED} '{row[TIT]}'")
print(f"   m={row['m']}, unlabelled rows={g['human_labels'].isna().sum()}, "
      f"k_E1E2={row['k_E1E2']}, r_E1E2={row['r_E1E2']}, "
      f"EFI={row['EFI_E1E2']:.4f} (the maximum, 1.0), share_E1E2={row['share_E1E2']:.4f}, "
      f"ai_fraction={row['ai_fraction']:.4f}")
print(f"   sample mean ai_fraction = {occ872['ai_fraction'].mean():.4f} "
      f"(sd {occ872['ai_fraction'].std():.4f})  -> this occupation is "
      f"{(row['ai_fraction']-occ872['ai_fraction'].mean())/occ872['ai_fraction'].std():+.2f} SD")
nun = md872.groupby(OCC)['human_labels'].apply(lambda s: s.isna().sum())
print(f"   occupations with >=1 unlabelled task: {(nun>0).sum()} "
      f"({nun[nun>0].sum()} unlabelled rows in total, {int(nun[DROPPED])} of them in {DROPPED})")

samples = [("S1  872  (notebook sample, unlabelled kept in m)", occ872),
           ("S2  871a (S1 minus 33-3051.00 ONLY, m unchanged)", occ871a),
           ("S3  871b (unlabelled ROWS dropped, m recomputed)", occ871)]
for ctrl_name, ctrls in [("with k  [HEADLINE]", ['k_E1E2']), ("no ctrl [VARIANT] ", [])]:
    print(f"\n  --- {ctrl_name} ---")
    for nm, d in samples:
        for fe in ('none', 'major', 'minor'):
            m = A.fit(d, 'EFI_E1E2', ['share_E1E2'] + ctrls, fe)
            p = A.pack(m, 'EFI_E1E2', 'share_E1E2')
            print(f"   {nm:<50} FE={fe:<6} b {p['b']:+.4f} ({p['se']:.4f}){A.st(p['p']):<3} "
                  f"p {p['p']:.4f} CI[{p['lo']:+.3f},{p['hi']:+.3f}] N {p['n']} R2 {p['r2']:.3f}"
                  f"  | expo {p['b_exp']:+.3f}({p['se_exp']:.3f}){A.st(p['p_exp'])}")

print("\n  same decomposition for the PUBLISHED spec (exposure = E1 share), for contrast:")
for nm, d in samples:
    for fe in ('none', 'major'):
        m = A.fit(d, 'EFI_E1E2', ['share_E1', 'k_E1E2'], fe)
        p = A.pack(m, 'EFI_E1E2', 'share_E1')
        print(f"   {nm:<50} FE={fe:<6} b {p['b']:+.4f} ({p['se']:.4f}){A.st(p['p']):<3} "
              f"p {p['p']:.4f} N {p['n']}")

# =========================================================================================
print("\n" + "=" * 128)
print("2. INFLUENCE ON THE MATCHED EFI COEFFICIENT (872 sample, no FE, headline control)")
print("=" * 128)
d = occ872.copy()
for c in ['ai_fraction', 'EFI_E1E2', 'share_E1E2', 'k_E1E2']:
    d[c] = (d[c] - d[c].mean()) / d[c].std()
X = sm.add_constant(d[['EFI_E1E2', 'share_E1E2', 'k_E1E2']].astype(float).to_numpy())
y = d['ai_fraction'].astype(float).to_numpy()
res = sm.OLS(y, X).fit()
infl = res.get_influence()
dfb = infl.dfbetas[:, 1]          # column 1 = EFI
h = infl.hat_matrix_diag
b_full = res.params[1]
print(f"   full-sample OLS EFI coef (z units, homoskedastic fit) = {b_full:+.5f}")
ord_ = np.argsort(-np.abs(dfb))[:12]
print(f"   {'rank':<5}{'SOC':<12}{'title':<46}{'k':>4}{'r':>4}{'m':>4}"
      f"{'EFI':>8}{'share':>8}{'aifrac':>9}{'hat':>8}{'dfbeta':>9}")
for i, ix in enumerate(ord_):
    r0 = occ872.iloc[ix]
    print(f"   {i+1:<5}{r0[OCC]:<12}{str(r0[TIT])[:44]:<46}{r0['k_E1E2']:>4}{r0['r_E1E2']:>4}"
          f"{r0['m']:>4}{r0['EFI_E1E2']:>8.3f}{r0['share_E1E2']:>8.3f}{r0['ai_fraction']:>9.3f}"
          f"{h[ix]:>8.4f}{dfb[ix]:>9.4f}")
print(f"   rank of {DROPPED}: "
      f"{int(np.where(occ872[OCC].to_numpy()[np.argsort(-np.abs(dfb))]==DROPPED)[0][0])+1} "
      f"of {len(dfb)}")

# leave-one-out sweep: how many single occupations move p across 0.05?
print("\n   leave-one-out sweep over all 872 occupations (matched, no FE, headline control,")
print("   clustered SE, as in the headline):")
ps, bs = [], []
for i in range(len(occ872)):
    sub = occ872.drop(index=i)
    m = A.fit(sub, 'EFI_E1E2', ['share_E1E2', 'k_E1E2'], 'none')
    ps.append(m.pvalues['EFI_E1E2']); bs.append(m.params['EFI_E1E2'])
ps = np.array(ps); bs = np.array(bs)
print(f"     leave-one-out b: min {bs.min():+.4f}  max {bs.max():+.4f}  "
      f"(full sample {-0.0069:+.4f})")
print(f"     leave-one-out p: min {ps.min():.4f}  max {ps.max():.4f}")
print(f"     number of single-occupation deletions that yield p<0.05: {(ps<0.05).sum()}")
if (ps < 0.05).sum():
    for i in np.where(ps < 0.05)[0]:
        r0 = occ872.iloc[i]
        print(f"       dropping {r0[OCC]} '{str(r0[TIT])[:40]}' -> b {bs[i]:+.4f} p {ps[i]:.4f}")

# =========================================================================================
print("\n" + "=" * 128)
print("3. EVERY ALTERNATIVE ARRANGEMENT STATISTIC ON THE 871 (labelled-rows-only) SAMPLE")
print("   matched exposure = E1|E2 share, headline control k IN")
print("=" * 128)
specs = [('EFI (E1|E2)  [= C1 = C4 up to scale]', 'EFI_E1E2', None, '-'),
         ('C1 r/m', 'C1_r_over_m', None, '-'),
         ('C2 r/k (k>=1)', 'C2_r_over_k', 'k1', '-'),
         ('C2b r/k (k>=2)', 'C2_r_over_k', 'k2', '-'),
         ('C3 adj-pair share (k>=1)', 'C3_adj_share', 'k1', '+'),
         ('C3b adj-pair share (k>=2)', 'C3_adj_share', 'k2', '+'),
         ('C5 permutation z (non-degenerate)', 'C5_z', 'z', '-'),
         ('C5b permutation z (deg=0)', 'C5_z0', None, '-'),
         ('C6 longest/k (k>=1)', 'C6_longest_over_k', 'k1', '+'),
         ('C6b longest/k (k>=2)', 'C6_longest_over_k', 'k2', '+')]
for nm, col, msk, sign in specs:
    if msk == 'k1':
        sub = occ871[occ871['k_E1E2'] >= 1].reset_index(drop=True)
    elif msk == 'k2':
        sub = occ871[occ871['k_E1E2'] >= 2].reset_index(drop=True)
    elif msk == 'z':
        sub = occ871[occ871['C5_z'].notna()].reset_index(drop=True)
    else:
        sub = occ871
    for fe in ('none', 'major', 'minor'):
        m = A.fit(sub, col, ['share_E1E2', 'k_E1E2'], fe)
        p = A.pack(m, col, 'share_E1E2')
        print(f"   {nm:<38}[pred {sign}] FE={fe:<6} b {p['b']:+.4f} ({p['se']:.4f}){A.st(p['p']):<3}"
              f" p {p['p']:.4f} CI[{p['lo']:+.3f},{p['hi']:+.3f}] N {p['n']:>4} MDE {p['mde']:.3f}")

# also: C4 decomposition on 871
print("\n   C4 (level -k/m and arrangement r/m separately) on the 871 sample, k control in:")
for fe in ('none', 'major', 'minor'):
    m = A.fit(occ871, 'arr_term', ['level_term', 'k_E1E2'], fe)
    p = A.pack(m, 'arr_term')
    print(f"     FE={fe:<6} arrangement {p['b']:+.4f} ({p['se']:.4f}){A.st(p['p']):<3} p {p['p']:.4f}"
          f"  | level {m.params['level_term']:+.4f} ({m.bse['level_term']:.4f})"
          f"{A.st(m.pvalues['level_term'])} p {m.pvalues['level_term']:.4f}  N {p['n']}")

# permutation z descriptives on 871
ok = occ871['C5_z'].notna()
print(f"\n   871 sample permutation z: n non-degenerate {ok.sum()}, mean "
      f"{occ871.loc[ok,'C5_z'].mean():+.4f}, corr with share "
      f"{occ871.loc[ok,'C5_z'].corr(occ871.loc[ok,'share_E1E2']):+.4f}")

# =========================================================================================
print("\n" + "=" * 128)
print("4. WHY THE ARRANGEMENT SIGNAL IS SO WEAK: how much residual variation is left")
print("=" * 128)
for nm, d, label in [("872", occ872, "notebook sample"), ("871", occ871, "labelled rows only")]:
    z = d[['EFI_E1E2', 'share_E1E2', 'k_E1E2', 'C5_z0', 'C1_r_over_m']].copy()
    for c in z.columns:
        z[c] = (z[c] - z[c].mean()) / z[c].std()
    Xc = sm.add_constant(z[['share_E1E2', 'k_E1E2']].to_numpy())
    for c in ['EFI_E1E2', 'C1_r_over_m', 'C5_z0']:
        e = z[c].to_numpy() - Xc @ np.linalg.lstsq(Xc, z[c].to_numpy(), rcond=None)[0]
        print(f"   {nm} ({label}): residual SD of {c:<12} after partialling out "
              f"(exposure, k) = {e.std(ddof=1):.4f}  (R2 of that auxiliary reg "
              f"{1-e.var(ddof=1):.4f})")
