#!/usr/bin/env python
"""
Is the MATCHED null an artefact of the EFI being a poor arrangement measure?

Rebuilds the main O*NET panel exactly as analysis/onet_fragmentationIndex.ipynb
builds it, then runs the definition grid (A), the control variants (B), six alternative
arrangement statistics (C), and the MDE calculation (D).

Output goes to data/computed_objects/efi_matched_exposure/. No published exhibit is touched.
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

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 400)

DATA = os.path.join(REPO, "data")
SCRATCH = OUT

OCC = 'O*NET-SOC Code'
TIT = 'Occupation Title'

MERGED = os.path.join(DATA, 'computed_objects', 'ONET_Eloundou_Anthropic_GPT',
                      'ONET_Eloundou_Anthropic_GPT.csv')
CLEAN = os.path.join(DATA, 'computed_objects', 'ONET_cleaned_tasks.csv')


# =========================================================================================
# 0. PANEL
# =========================================================================================
def load_merged(drop_unlabelled=False):
    md = pd.read_csv(MERGED)
    if drop_unlabelled:
        md = md[md['human_labels'].notna()].reset_index(drop=True)
    cnt = md.groupby(OCC)['Task ID'].nunique()
    keep = cnt[cnt >= 3].index
    md = md[md[OCC].isin(keep)].reset_index(drop=True)
    return md


def seq_stats(v):
    """k, r, longest run, for a 0/1 numpy vector in workflow order."""
    v = np.asarray(v, dtype=np.int64)
    m = len(v)
    k = int(v.sum())
    if k == 0:
        return 0, 0, 0
    starts = int(v[0]) + int(((v[1:] == 1) & (v[:-1] == 0)).sum())
    # longest run
    best = cur = 0
    for x in v:
        cur = cur + 1 if x == 1 else 0
        best = max(best, cur)
    return k, starts, best


def build_panel(md):
    """One row per occupation. m = rows in the occupation (as the notebook does:
    denominators are len(group), unlabelled rows included and treated as not AI-able)."""
    rows = []
    for (soc, tit), g in md.groupby([OCC, TIT], sort=False):
        m = len(g)
        hl = g['human_labels']
        lab = g['label']
        d = {OCC: soc, TIT: tit, 'm': m,
             'num_tasks': g['Task ID'].nunique(),
             'ai_fraction': lab.isin(['Augmentation', 'Automation']).sum() / m}
        for name, ind in (('E1', (hl == 'E1')),
                          ('E2', (hl == 'E2')),
                          ('E1E2', hl.isin(['E1', 'E2'])),
                          ('exec', lab.isin(['Augmentation', 'Automation']))):
            v = ind.to_numpy().astype(int)
            k, r, lr = seq_stats(v)
            d[f'k_{name}'] = k
            d[f'r_{name}'] = r
            d[f'lr_{name}'] = lr
            d[f'share_{name}'] = k / m
            d[f'EFI_{name}'] = (m - k + r) / m
        rows.append(d)
    occ = pd.DataFrame(rows)

    ON = pd.read_csv(CLEAN)
    SOC = ON[[OCC, 'Major_Group_Code', 'Minor_Group_Code']].drop_duplicates(subset=[OCC])
    occ = occ.merge(SOC, on=OCC, how='left')
    for c in ('Major_Group_Code', 'Minor_Group_Code', OCC):
        occ[c] = occ[c].astype('object')
    return occ


# =========================================================================================
# regression helper
# =========================================================================================
FE_FORMULA = {'none': '', 'major': ' + C(Major_Group_Code)', 'minor': ' + C(Minor_Group_Code)'}


def fit(df, focal, extra, fe, zcols=None, y='ai_fraction'):
    """OLS of y on focal + extra (+FE), all listed columns z-scored within the sample,
    clustered on O*NET-SOC Code exactly as the notebook does."""
    d = df.copy()
    zc = zcols if zcols is not None else ([y, focal] + list(extra))
    for c in zc:
        s = d[c].astype(float)
        sd = s.std()
        d[c] = (s - s.mean()) / sd if sd > 0 else 0.0
    rhs = ' + '.join([focal] + list(extra))
    f = f'{y} ~ {rhs}{FE_FORMULA[fe]}'
    m = smf.ols(f, data=d).fit(cov_type='cluster',
                               cov_kwds={'groups': d[OCC], 'use_correction': True,
                                         'df_correction': True})
    return m


def pack(m, focal, expo=None):
    ci = m.conf_int().loc[focal]
    out = dict(b=m.params[focal], se=m.bse[focal], p=m.pvalues[focal],
               lo=ci[0], hi=ci[1], n=int(m.nobs), r2=m.rsquared,
               mde=2.802 * m.bse[focal])
    if expo is not None and expo in m.params.index:
        out.update(b_exp=m.params[expo], se_exp=m.bse[expo], p_exp=m.pvalues[expo])
    return out


def st(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else '  '


def line(tag, d, w=52):
    s = f"{tag:<{w}} b {d['b']:+.4f} ({d['se']:.4f}){st(d['p']):<3} p {d['p']:.4f}  " \
        f"CI[{d['lo']:+.3f},{d['hi']:+.3f}] N {d['n']:>4} R2 {d['r2']:.3f} MDE {d['mde']:.3f}"
    if 'b_exp' in d:
        s += f"  | expo {d['b_exp']:+.3f}({d['se_exp']:.3f}){st(d['p_exp'])}"
    return s


RESULTS = []


def rec(block, tag, d, **kw):
    row = dict(block=block, cell=tag)
    row.update(d)
    row.update(kw)
    RESULTS.append(row)


# =========================================================================================
def main():
    md = load_merged()
    occ = build_panel(md)
    print("=" * 130)
    print("PANEL")
    print("=" * 130)
    print(f"merged rows kept: {len(md)}   occupations: {occ.shape[0]}")
    print(f"rows with missing human_labels in the kept sample: {md['human_labels'].isna().sum()}")

    # identity check EFI = 1 - k/m + r/m
    dev = (occ['EFI_E1E2'] - (1 - occ['k_E1E2'] / occ['m'] + occ['r_E1E2'] / occ['m'])).abs().max()
    print(f"identity |EFI - (1 - k/m + r/m)| max dev: {dev:.2e}")
    print(f"corr(EFI_E1E2, share_E1E2) = {np.corrcoef(occ['EFI_E1E2'], occ['share_E1E2'])[0,1]:.4f}"
          f"   R2 = {np.corrcoef(occ['EFI_E1E2'], occ['share_E1E2'])[0,1]**2:.4f}")
    print(f"m: mean {occ['m'].mean():.2f} sd {occ['m'].std():.2f} min {occ['m'].min()} max {occ['m'].max()}")
    print(f"k_E1E2: mean {occ['k_E1E2'].mean():.2f}  k=0 occs {(occ['k_E1E2']==0).sum()}  "
          f"k=1 occs {(occ['k_E1E2']==1).sum()}  k=m occs {(occ['k_E1E2']==occ['m']).sum()}")
    print(f"r_E1E2: mean {occ['r_E1E2'].mean():.2f} sd {occ['r_E1E2'].std():.2f}")

    # anchor check
    print("\nANCHOR CHECK (should match the two verified specs)")
    for tag, expo, efi, ctrl in [("PUBLISHED", 'share_E1', 'EFI_E1E2', 'k_E1E2'),
                                 ("MATCHED  ", 'share_E1E2', 'EFI_E1E2', 'k_E1E2')]:
        for fe in ('none', 'major', 'minor'):
            m = fit(occ, efi, [expo, ctrl], fe)
            d = pack(m, efi, expo)
            print("  " + line(f"{tag} FE={fe}", d))

    # =====================================================================================
    print("\n" + "=" * 130)
    print("A. FULL DEFINITION GRID  (control = count of steps of the SAME label kind the EFI "
          "is built on; that is what both anchor specs use)")
    print("=" * 130)
    print(f"{'EFI built on':<14}{'exposure':<14}{'FE':<8}"
          f"{'EFI coef':>34}   {'exposure coef':>22}   {'N':>5} {'R2':>6}  mark")
    for efik in ('E1', 'E2', 'E1E2'):
        for expok in ('E1', 'E2', 'E1E2'):
            for fe in ('none', 'major', 'minor'):
                efi = f'EFI_{efik}'
                expo = f'share_{expok}'
                ctrl = f'k_{efik}'
                m = fit(occ, efi, [expo, ctrl], fe)
                d = pack(m, efi, expo)
                mark = ''
                if efik == 'E1E2' and expok == 'E1':
                    mark = '<< PUBLISHED'
                if efik == 'E1E2' and expok == 'E1E2':
                    mark = '<< MATCHED HEADLINE'
                print(f"{efik:<14}{expok:<14}{fe:<8}"
                      f"{d['b']:+9.4f} ({d['se']:.4f}){st(d['p']):<3} p={d['p']:<9.4f}"
                      f"   {d['b_exp']:+7.3f}({d['se_exp']:.3f}){st(d['p_exp']):<3} p={d['p_exp']:<8.4f}"
                      f"  {d['n']:>5} {d['r2']:>6.3f}  {mark}")
                rec('A', f'EFI={efik} expo={expok} FE={fe}', d,
                    efi=efik, exposure=expok, fe=fe, ctrl=f'k_{efik}')

    # =====================================================================================
    print("\n" + "=" * 130)
    print("B. THE STEP-COUNT CONTROL (variants; (i) is the headline)")
    print("=" * 130)
    occ['logm'] = np.log(occ['m'])
    variants = [('(i)   k only  [HEADLINE]', ['k_E1E2']),
                ('(ii)  no control [VARIANT]', []),
                ('(iii) log(m) only [VARIANT]', ['logm']),
                ('(iv)  k and log(m) [VARIANT]', ['k_E1E2', 'logm'])]
    for tag, expo in [("PUBLISHED (expo=E1)", 'share_E1'), ("MATCHED (expo=E1|E2)", 'share_E1E2')]:
        print(f"\n  --- {tag}, EFI built on E1|E2, n=872 full sample ---")
        for vt, ctrls in variants:
            for fe in ('none', 'major', 'minor'):
                m = fit(occ, 'EFI_E1E2', [expo] + ctrls, fe)
                d = pack(m, 'EFI_E1E2', expo)
                print("   " + line(f"{vt:<28} FE={fe:<6}", d, w=42))
                rec('B', f'{tag} {vt} FE={fe}', d, sample='872')

    # ---- 871 reconciliation -------------------------------------------------------------
    print("\n  --- reconciling the pre-existing 871-occupation run "
          "(data/computed_objects/apqc_pcf_fragmentation/exposure_definition_grid.csv) ---")
    md871 = load_merged(drop_unlabelled=True)
    occ871 = build_panel(md871)
    lost = set(occ[OCC]) - set(occ871[OCC])
    print(f"    dropping rows with missing human_labels: rows {len(md)} -> {len(md871)}, "
          f"occupations {len(occ)} -> {len(occ871)}")
    print(f"    occupation(s) lost: {sorted(lost)}")
    for s in lost:
        g = md[md[OCC] == s]
        print(f"      {s} {g[TIT].iloc[0]}: {len(g)} rows, "
              f"{g['human_labels'].isna().sum()} unlabelled, "
              f"{g['Task ID'].nunique()} unique Task IDs")
    print("    published spec (expo=E1) / matched spec (expo=E1|E2), NO step-count control, n=871:")
    for tag, expo in [("A. paper (E1)   ", 'share_E1'), ("B. matched (E1|E2)", 'share_E1E2')]:
        for fe in ('none', 'major'):
            m = fit(occ871, 'EFI_E1E2', [expo], fe)
            d = pack(m, 'EFI_E1E2', expo)
            print("      " + line(f"{tag} FE={fe:<6}", d, w=30))
            rec('B871', f'{tag} no-ctrl FE={fe}', d, sample='871')
    print("    same 871 sample WITH the headline control:")
    for tag, expo in [("A. paper (E1)   ", 'share_E1'), ("B. matched (E1|E2)", 'share_E1E2')]:
        for fe in ('none', 'major'):
            m = fit(occ871, 'EFI_E1E2', [expo, 'k_E1E2'], fe)
            d = pack(m, 'EFI_E1E2', expo)
            print("      " + line(f"{tag} FE={fe:<6}", d, w=30))
            rec('B871c', f'{tag} with-ctrl FE={fe}', d, sample='871')

    # ---- what does the control absorb? --------------------------------------------------
    print("\n  --- mechanically, what variation does k absorb? (872 sample, no FE) ---")
    z = occ[['ai_fraction', 'EFI_E1E2', 'share_E1E2', 'k_E1E2', 'm', 'r_E1E2', 'logm']].copy()
    for c in z.columns:
        z[c] = (z[c] - z[c].mean()) / z[c].std()
    print("    correlation matrix (z-scored):")
    print(z[['ai_fraction', 'EFI_E1E2', 'share_E1E2', 'k_E1E2', 'm', 'r_E1E2']].corr().round(3).to_string())
    # FWL: residualise EFI on exposure only vs on exposure+k
    import statsmodels.api as sm
    def resid(y, X):
        X = sm.add_constant(np.asarray(X, dtype=float))
        return np.asarray(y, dtype=float) - X @ np.linalg.lstsq(X, np.asarray(y, dtype=float), rcond=None)[0]
    e_noc = resid(z['EFI_E1E2'], z[['share_E1E2']])
    e_wc = resid(z['EFI_E1E2'], z[['share_E1E2', 'k_E1E2']])
    print(f"    Var(EFI | exposure)            = {e_noc.var(ddof=1):.4f}  "
          f"(share of total EFI variance {e_noc.var(ddof=1)/z['EFI_E1E2'].var(ddof=1):.4f})")
    print(f"    Var(EFI | exposure, k)         = {e_wc.var(ddof=1):.4f}  "
          f"(share of total EFI variance {e_wc.var(ddof=1)/z['EFI_E1E2'].var(ddof=1):.4f})")
    print(f"    fraction of the residual EFI variation removed by adding k: "
          f"{1 - e_wc.var(ddof=1)/e_noc.var(ddof=1):.4f}")
    print(f"    corr(EFI resid on exposure, k) = {np.corrcoef(e_noc, z['k_E1E2'])[0,1]:+.4f}")
    print(f"    corr(EFI resid on exposure, m) = {np.corrcoef(e_noc, z['m'])[0,1]:+.4f}")
    print(f"    corr(EFI resid on exposure, r) = {np.corrcoef(e_noc, z['r_E1E2'])[0,1]:+.4f}")
    # what is EFI | exposure?  EFI = 1 - k/m + r/m ; exposure = k/m ; so EFI+exposure-1 = r/m
    print(f"    check: EFI + exposure - 1 == r/m  max dev "
          f"{np.abs(occ['EFI_E1E2'] + occ['share_E1E2'] - 1 - occ['r_E1E2']/occ['m']).max():.2e}")
    # regression of the outcome on m directly
    print("\n  --- D29: does m belong in the equation? (872, matched exposure, headline k) ---")
    for fe in ('none', 'major', 'minor'):
        m1 = fit(occ, 'EFI_E1E2', ['share_E1E2', 'k_E1E2', 'm'], fe)
        d = pack(m1, 'EFI_E1E2', 'share_E1E2')
        mm = m1.params['m'] / m1.bse['m']
        print("    " + line(f"EFI + expo + k + m  FE={fe:<6}", d, w=34) +
              f"  | m coef {m1.params['m']:+.3f}({m1.bse['m']:.3f}) t={mm:+.2f} p={m1.pvalues['m']:.4f}")
        rec('B_m', f'with m FE={fe}', d)
    for fe in ('none', 'major', 'minor'):
        m1 = fit(occ, 'EFI_E1E2', ['share_E1E2', 'k_E1E2', 'logm'], fe)
        d = pack(m1, 'EFI_E1E2', 'share_E1E2')
        print("    " + line(f"EFI + expo + k + log(m) FE={fe:<6}", d, w=34) +
              f"  | logm coef {m1.params['logm']:+.3f}({m1.bse['logm']:.3f}) p={m1.pvalues['logm']:.4f}")
        rec('B_logm', f'with logm FE={fe}', d)

    # =====================================================================================
    print("\n" + "=" * 130)
    print("C. ALTERNATIVE ARRANGEMENT STATISTICS "
          "(matched exposure = E1|E2 share, headline control = k, EFI replaced)")
    print("=" * 130)

    occ['C1_r_over_m'] = occ['r_E1E2'] / occ['m']
    kk = occ['k_E1E2'].to_numpy()
    mm_ = occ['m'].to_numpy()
    rr = occ['r_E1E2'].to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        occ['C2_r_over_k'] = np.where(kk >= 1, rr / np.maximum(kk, 1), np.nan)
        occ['C3_adj_share'] = np.where(kk >= 1, (kk - rr) / np.maximum(kk - 1, 1), np.nan)
        occ['C6_longest_over_k'] = np.where(kk >= 1, occ['lr_E1E2'] / np.maximum(kk, 1), np.nan)
    occ['level_term'] = -kk / mm_
    occ['arr_term'] = rr / mm_

    # ---- permutation moments -----------------------------------------------------------
    # For a uniformly random arrangement of k ones in m slots:
    #   E[R]   = k(m-k+1)/m
    #   Var[R] = k(k-1)(m-k)(m-k+1) / (m^2 (m-1))
    ER = kk * (mm_ - kk + 1) / mm_
    VR = kk * (kk - 1) * (mm_ - kk) * (mm_ - kk + 1) / (mm_.astype(float) ** 2 * (mm_ - 1))
    SD = np.sqrt(np.maximum(VR, 0))
    occ['perm_ER'] = ER
    occ['perm_SD'] = SD
    with np.errstate(divide='ignore', invalid='ignore'):
        occ['C5_z'] = np.where(SD > 0, (rr - ER) / SD, np.nan)
    occ['C5_z0'] = occ['C5_z'].fillna(0.0)

    # Monte-Carlo validation of the analytic moments on a handful of (m,k) cells
    rng = np.random.default_rng(20260901)
    print("\n  Monte-Carlo validation of the analytic permutation moments (200,000 draws each):")
    for (mv, kv) in [(10, 3), (20, 7), (33, 12), (15, 1), (8, 8)]:
        v = np.zeros((200000, mv), dtype=np.int8)
        idx = np.argsort(rng.random((200000, mv)), axis=1)[:, :kv]
        np.put_along_axis(v, idx, 1, axis=1)
        R = v[:, 0] + ((v[:, 1:] == 1) & (v[:, :-1] == 0)).sum(1)
        ea = kv * (mv - kv + 1) / mv
        va = kv * (kv - 1) * (mv - kv) * (mv - kv + 1) / (mv ** 2 * (mv - 1))
        print(f"    m={mv:>3} k={kv:>3}  MC mean {R.mean():7.4f} vs analytic {ea:7.4f} | "
              f"MC sd {R.std(ddof=1):7.4f} vs analytic {np.sqrt(max(va,0)):7.4f}")

    ndeg = occ['C5_z'].isna().sum()
    print(f"\n  permutation z: degenerate (k<2 or k=m, so Var[R]=0) occupations: {ndeg} of {len(occ)}")
    print(f"    of which k=0: {(occ['k_E1E2']==0).sum()}, k=1: {(occ['k_E1E2']==1).sum()}, "
          f"k=m: {(occ['k_E1E2']==occ['m']).sum()}")
    ok = occ['C5_z'].notna()
    print(f"    z on the non-degenerate sample: mean {occ.loc[ok,'C5_z'].mean():+.4f} "
          f"sd {occ.loc[ok,'C5_z'].std():.4f} min {occ.loc[ok,'C5_z'].min():+.3f} "
          f"max {occ.loc[ok,'C5_z'].max():+.3f}")
    print(f"    corr(z, E1|E2 share) = {occ.loc[ok,'C5_z'].corr(occ.loc[ok,'share_E1E2']):+.4f}   "
          f"corr(z, k) = {occ.loc[ok,'C5_z'].corr(occ.loc[ok,'k_E1E2']):+.4f}   "
          f"corr(z, m) = {occ.loc[ok,'C5_z'].corr(occ.loc[ok,'m']):+.4f}   "
          f"corr(z, EFI) = {occ.loc[ok,'C5_z'].corr(occ.loc[ok,'EFI_E1E2']):+.4f}")
    print(f"    corr(EFI, E1|E2 share) for reference = "
          f"{occ['EFI_E1E2'].corr(occ['share_E1E2']):+.4f}")

    # descriptive correlations of every candidate with the exposure level
    print("\n  how scale-free is each candidate? corr with the E1|E2 share, with k, with m:")
    cand_desc = [('EFI (E1|E2)', 'EFI_E1E2'), ('C1 r/m', 'C1_r_over_m'), ('C2 r/k', 'C2_r_over_k'),
                 ('C3 (k-r)/(k-1)', 'C3_adj_share'), ('C4 arr term r/m', 'arr_term'),
                 ('C5 perm z', 'C5_z'), ('C6 longest/k', 'C6_longest_over_k')]
    for nm, c in cand_desc:
        s = occ[c]
        g = occ[s.notna()]
        print(f"    {nm:<18} corr(.,share)={g[c].corr(g['share_E1E2']):+.4f}  "
              f"corr(.,k)={g[c].corr(g['k_E1E2']):+.4f}  corr(.,m)={g[c].corr(g['m']):+.4f}  "
              f"n={len(g)}")

    # ---- run each candidate ------------------------------------------------------------
    specs = [
        ('C1 r/m (runs per step)', 'C1_r_over_m', None, '-'),
        ('C2 r/k (runs per exposed step)', 'C2_r_over_k', occ['k_E1E2'] >= 1, '-'),
        ('C2b r/k, k>=2 only', 'C2_r_over_k', occ['k_E1E2'] >= 2, '-'),
        ('C3 adj-pair share (k-r)/max(k-1,1)', 'C3_adj_share', occ['k_E1E2'] >= 1, '+'),
        ('C3b adj-pair share, k>=2 only', 'C3_adj_share', occ['k_E1E2'] >= 2, '+'),
        ('C5 permutation z (non-degenerate)', 'C5_z', occ['C5_z'].notna(), '-'),
        ('C5b permutation z (deg. set to 0)', 'C5_z0', None, '-'),
        ('C6 longest run / k', 'C6_longest_over_k', occ['k_E1E2'] >= 1, '+'),
        ('C6b longest run / k, k>=2 only', 'C6_longest_over_k', occ['k_E1E2'] >= 2, '+'),
    ]
    print("\n  headline control (k) IN, matched exposure (E1|E2 share) IN:")
    for nm, col, mask, sign in specs:
        sub = occ if mask is None else occ[mask].reset_index(drop=True)
        print(f"\n   {nm}   [predicted sign {sign}]")
        for fe in ('none', 'major', 'minor'):
            m = fit(sub, col, ['share_E1E2', 'k_E1E2'], fe)
            d = pack(m, col, 'share_E1E2')
            print("     " + line(f"FE={fe:<6}", d, w=14))
            rec('C', f'{nm} FE={fe}', d, pred=sign, stat=col)

    print("\n  same, control DROPPED (labelled variant, not the headline):")
    for nm, col, mask, sign in specs:
        sub = occ if mask is None else occ[mask].reset_index(drop=True)
        for fe in ('none', 'major', 'minor'):
            m = fit(sub, col, ['share_E1E2'], fe)
            d = pack(m, col, 'share_E1E2')
            print("     " + line(f"{nm:<36} FE={fe:<6}", d, w=46))
            rec('C_noctrl', f'{nm} FE={fe} no-ctrl', d, pred=sign, stat=col)

    # ---- C4: level and arrangement as separate regressors --------------------------------
    print("\n  C4. level term (-k/m) and arrangement term (r/m) as SEPARATE regressors.")
    print("      Note: -k/m is (minus) the matched exposure regressor, so the exposure term "
          "cannot also be included; it IS the level term.")
    for fe in ('none', 'major', 'minor'):
        m = fit(occ, 'arr_term', ['level_term', 'k_E1E2'], fe)
        d = pack(m, 'arr_term')
        print("     " + line(f"C4 with k     FE={fe:<6}", d, w=26) +
              f"  | level {m.params['level_term']:+.3f}({m.bse['level_term']:.3f})"
              f"{st(m.pvalues['level_term'])} p={m.pvalues['level_term']:.4f}")
        rec('C4', f'arr_term with k FE={fe}', d, pred='-')
    for fe in ('none', 'major', 'minor'):
        m = fit(occ, 'arr_term', ['level_term'], fe)
        d = pack(m, 'arr_term')
        print("     " + line(f"C4 no k [VAR] FE={fe:<6}", d, w=26) +
              f"  | level {m.params['level_term']:+.3f}({m.bse['level_term']:.3f})"
              f"{st(m.pvalues['level_term'])} p={m.pvalues['level_term']:.4f}")
        rec('C4_noctrl', f'arr_term no k FE={fe}', d, pred='-')

    # ---- algebraic-equivalence check ----------------------------------------------------
    print("\n  ALGEBRAIC EQUIVALENCE CHECK (raw, un-z-scored units, no FE, control k in):")
    raw = occ.copy()
    import statsmodels.api as sm
    def raw_fit(cols):
        X = sm.add_constant(raw[cols].astype(float))
        return sm.OLS(raw['ai_fraction'].astype(float), X).fit(
            cov_type='cluster', cov_kwds={'groups': raw[OCC], 'use_correction': True,
                                          'df_correction': True})
    a = raw_fit(['EFI_E1E2', 'share_E1E2', 'k_E1E2'])
    b = raw_fit(['C1_r_over_m', 'share_E1E2', 'k_E1E2'])
    c = raw_fit(['arr_term', 'level_term', 'k_E1E2'])
    print(f"    matched EFI coef (raw)      = {a.params['EFI_E1E2']:+.6f} "
          f"se {a.bse['EFI_E1E2']:.6f} t {a.tvalues['EFI_E1E2']:+.4f} p {a.pvalues['EFI_E1E2']:.4f}")
    print(f"    C1 r/m coef (raw)           = {b.params['C1_r_over_m']:+.6f} "
          f"se {b.bse['C1_r_over_m']:.6f} t {b.tvalues['C1_r_over_m']:+.4f} p {b.pvalues['C1_r_over_m']:.4f}")
    print(f"    C4 arrangement coef (raw)   = {c.params['arr_term']:+.6f} "
          f"se {c.bse['arr_term']:.6f} t {c.tvalues['arr_term']:+.4f} p {c.pvalues['arr_term']:.4f}")
    print(f"    R2:  matched {a.rsquared:.6f}   C1 {b.rsquared:.6f}   C4 {c.rsquared:.6f}")

    out = pd.DataFrame(RESULTS)
    out.to_csv(os.path.join(OUT, "arrangement_measures_results.csv"), index=False)
    print(f"\nwrote {os.path.join(SCRATCH, 'arrangement_measures_results.csv')} "
          f"({len(out)} rows)")

    # D: MDE table printed from the saved frame in the report step
    print("\n" + "=" * 130)
    print("D. MDE (2.802 * SE) for every candidate, matched exposure + headline control")
    print("=" * 130)
    d_rows = out[out.block.isin(['C', 'C4'])]
    print(f"{'statistic / FE':<50}{'b':>10}{'SE':>10}{'p':>10}{'MDE':>10}{'|b|/MDE':>10}{'N':>7}")
    for _, r in d_rows.iterrows():
        print(f"{r['cell']:<50}{r['b']:>+10.4f}{r['se']:>10.4f}{r['p']:>10.4f}"
              f"{r['mde']:>10.4f}{abs(r['b'])/r['mde']:>10.3f}{r['n']:>7.0f}")
    # EFI reference
    for fe in ('none', 'major', 'minor'):
        m = fit(occ, 'EFI_E1E2', ['share_E1E2', 'k_E1E2'], fe)
        d = pack(m, 'EFI_E1E2')
        print(f"{'REFERENCE matched EFI FE='+fe:<50}{d['b']:>+10.4f}{d['se']:>10.4f}"
              f"{d['p']:>10.4f}{d['mde']:>10.4f}{abs(d['b'])/d['mde']:>10.3f}{d['n']:>7.0f}")


if __name__ == '__main__':
    main()
