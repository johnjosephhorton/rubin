#!/usr/bin/env python
"""
What the matched headline specification can and cannot detect.

Reimplements analysis/onet_fragmentationIndex.ipynb (Definition 1, exposure-based
E1|E2 fragmentation index) on the 872-occupation O*NET sample, then:

  (1) coefficient / clustered SE / t / p / 95% CI on the EFI
  (2) MDE at 80% power, 5% two-sided = 2.802 * SE
  (3) published point estimate inside or outside the matched 95% CI
  (4) horse race: EFI + E1 share + E1|E2 share together
  (5) residual-bootstrap power simulation on the matched design
  (6) translation of every coefficient and CI bound into percentage points

Output goes to data/computed_objects/efi_matched_exposure/. No published exhibit is touched.
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import os as _os
# Repo root and output dir are derived from this file's own location, so the folder
# can move and no home directory is baked in. Generated output follows the repo
# convention: data/computed_objects/<name of this analysis folder>/.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
OUT = _os.path.join(REPO, "data", "computed_objects", "efi_matched_exposure")
_os.makedirs(OUT, exist_ok=True)
REPO_ROOT = REPO  # alias used below


DATA = os.path.join(REPO_ROOT, "data")

CODE_VAR = "O*NET-SOC Code"
TITLE_VAR = "Occupation Title"

ZCRIT = stats.norm.ppf(0.975)      # 1.959964
ZPOW = stats.norm.ppf(0.80)        # 0.841621
MDE_MULT = ZCRIT + ZPOW            # 2.801583

PUBLISHED_POINT = {"no FE": -0.261, "major FE": -0.380, "minor FE": -0.283}

# ----------------------------------------------------------------- data build
merged = pd.read_csv(os.path.join(
    DATA, "computed_objects", "ONET_Eloundou_Anthropic_GPT",
    "ONET_Eloundou_Anthropic_GPT.csv"))

# notebook filter: keep occupations with >= 3 distinct Task IDs
counts = merged.groupby(CODE_VAR)["Task ID"].nunique()
valid = counts[counts >= 3].index
merged = merged[merged[CODE_VAR].isin(valid)].reset_index(drop=True)

ONET = pd.read_csv(os.path.join(DATA, "computed_objects", "ONET_cleaned_tasks.csv"))
SOC = ONET[[CODE_VAR, TITLE_VAR, "Major_Group_Code", "Major_Group_Title",
            "Minor_Group_Code", "Minor_Group_Title",
            "Broad_Occupation_Code", "Broad_Occupation_Title",
            "Detailed_Occupation_Code", "Detailed_Occupation_Title"]].copy()
SOC = SOC.drop_duplicates(subset=[CODE_VAR])


def occupation_analysis(df):
    rows = []
    for (soc, title), g in df.groupby([CODE_VAR, TITLE_VAR]):
        n = len(g)
        aug = (g["label"] == "Augmentation").sum() / n
        aut = (g["label"] == "Automation").sum() / n
        e1 = (g["human_labels"] == "E1").sum() / n
        e2 = (g["human_labels"] == "E2").sum() / n
        rows.append({
            CODE_VAR: soc, TITLE_VAR: title,
            "num_tasks": g["Task ID"].nunique(),
            "ai_fraction": aug + aut,
            "human_E1_fraction": e1,
            "human_E2_fraction": e2,
            "human_aiExposure_fraction": e1 + e2,
            "num_E1E2_tasks": int(g["human_labels"].isin(["E1", "E2"]).sum()),
        })
    return pd.DataFrame(rows)


def fragmentation_index(df):
    """Definition 1: consecutive E1|E2 steps merged into runs; EFI = (m-k+r)/m."""
    f = df.copy()
    f["is_ai"] = f["human_labels"].isin(["E1", "E2"]).astype(int)
    f["next_is_ai"] = f.groupby([CODE_VAR, TITLE_VAR])["is_ai"].shift(-1).fillna(0).astype(int)
    f["num_switches"] = 1
    f.loc[(f["is_ai"] == 1) & (f["next_is_ai"] == 1), "num_switches"] = 0
    f = f.groupby([CODE_VAR, TITLE_VAR])["num_switches"].mean().reset_index()
    return f.rename(columns={"num_switches": "fragmentation_index"})


occ = occupation_analysis(merged)
occ = occ.merge(fragmentation_index(merged), on=[CODE_VAR, TITLE_VAR], how="left")
occ = occ.merge(SOC.drop(columns=[TITLE_VAR]), on=[CODE_VAR], how="left")
for c in ("Major_Group_Code", "Minor_Group_Code", CODE_VAR):
    occ[c] = occ[c].astype("object")

RAW = occ.copy()
ZCOLS = ["ai_fraction", "fragmentation_index", "num_E1E2_tasks",
         "human_E1_fraction", "human_E2_fraction", "human_aiExposure_fraction"]
SD = {c: RAW[c].std() for c in ZCOLS}
MU = {c: RAW[c].mean() for c in ZCOLS}
for c in ZCOLS:
    occ[c] = (RAW[c] - MU[c]) / SD[c]

SD_Y = SD["ai_fraction"]
SD_EFI = SD["fragmentation_index"]

print("=" * 104)
print("SAMPLE AND SCALING")
print("=" * 104)
print(f"  rows / unique O*NET-SOC codes : {len(occ)} / {occ[CODE_VAR].nunique()}")
print(f"  major groups / minor groups   : {occ['Major_Group_Code'].nunique()} / "
      f"{occ['Minor_Group_Code'].nunique()}")
print(f"  sd(ai_fraction)   [outcome]   : {SD_Y:.4f}   mean {MU['ai_fraction']:.4f}")
print(f"  sd(fragmentation_index)       : {SD_EFI:.4f}   mean {MU['fragmentation_index']:.4f}")
print(f"  sd(E1 share)                  : {SD['human_E1_fraction']:.4f}")
print(f"  sd(E1|E2 share)               : {SD['human_aiExposure_fraction']:.4f}")
print(f"  corr(EFI, E1|E2 share)        : {RAW['fragmentation_index'].corr(RAW['human_aiExposure_fraction']):.4f}"
      f"   R2 = {RAW['fragmentation_index'].corr(RAW['human_aiExposure_fraction'])**2:.4f}")
print(f"  corr(EFI, E1 share)           : {RAW['fragmentation_index'].corr(RAW['human_E1_fraction']):.4f}"
      f"   R2 = {RAW['fragmentation_index'].corr(RAW['human_E1_fraction'])**2:.4f}")
print(f"  corr(E1 share, E1|E2 share)   : {RAW['human_E1_fraction'].corr(RAW['human_aiExposure_fraction']):.4f}")
print()

FE = {"no FE": "", "major FE": " + C(Major_Group_Code)", "minor FE": " + C(Minor_Group_Code)"}


def fit(rhs, fe):
    return smf.ols(f"ai_fraction ~ {rhs}{FE[fe]}", data=occ).fit(
        cov_type="cluster",
        cov_kwds={"groups": occ[CODE_VAR], "use_correction": True, "df_correction": True})


SPECS = {
    "PUBLISHED": ("fragmentation_index + human_E1_fraction + num_E1E2_tasks",
                  "human_E1_fraction"),
    "MATCHED":   ("fragmentation_index + human_aiExposure_fraction + num_E1E2_tasks",
                  "human_aiExposure_fraction"),
}

models = {}
print("=" * 104)
print("(1) + (2)  EFI COEFFICIENT, CLUSTERED SE, t, p, 95% CI, AND MDE AT 80% POWER")
print("=" * 104)
for sname, (rhs, expvar) in SPECS.items():
    print(f"\n--- {sname}   (exposure regressor = {expvar}) ---")
    print(f'{"column":9s} {"EFI b":>8s} {"SE":>8s} {"t":>8s} {"p":>9s} '
          f'{"CI low":>8s} {"CI high":>8s} {"MDE80":>8s} | '
          f'{"expos b":>8s} {"SE":>7s} {"p":>8s} | {"ctrl b":>7s} {"p":>7s} | {"R2":>6s} {"N":>5s}')
    for fe in FE:
        m = fit(rhs, fe)
        models[(sname, fe)] = m
        b, se = m.params["fragmentation_index"], m.bse["fragmentation_index"]
        t, p = m.tvalues["fragmentation_index"], m.pvalues["fragmentation_index"]
        lo, hi = m.conf_int().loc["fragmentation_index"]
        print(f'{fe:9s} {b:8.3f} {se:8.3f} {t:8.3f} {p:9.3f} '
              f'{lo:8.3f} {hi:8.3f} {MDE_MULT*se:8.3f} | '
              f'{m.params[expvar]:8.3f} {m.bse[expvar]:7.3f} {m.pvalues[expvar]:8.3f} | '
              f'{m.params["num_E1E2_tasks"]:7.3f} {m.pvalues["num_E1E2_tasks"]:7.3f} | '
              f'{m.rsquared:6.3f} {int(m.nobs):5d}')

m0 = models[("MATCHED", "no FE")]
print(f"\n  inference: use_t = {m0.use_t}, df used for the t critical value = {m0.df_resid:.0f}, "
      f"clusters = {occ[CODE_VAR].nunique()} (all singletons)")
print(f"  MDE multiplier z_.975 + z_.80 = {ZCRIT:.6f} + {ZPOW:.6f} = {MDE_MULT:.6f}")

# ---------------------------------------------- (3) published point vs matched CI
print()
print("=" * 104)
print("(3) DOES THE PUBLISHED POINT ESTIMATE LIE INSIDE THE MATCHED 95% CI?")
print("=" * 104)
print(f'{"column":9s} {"published":>10s} {"matched b":>10s} {"matched SE":>11s} '
      f'{"CI low":>9s} {"CI high":>9s} {"verdict":>9s} {"gap to CI":>10s} '
      f'{"(pub-b)/SE":>11s} {"p":>8s}')
verdicts = {}
for fe in FE:
    m = models[("MATCHED", fe)]
    b, se = m.params["fragmentation_index"], m.bse["fragmentation_index"]
    lo, hi = m.conf_int().loc["fragmentation_index"]
    pub = PUBLISHED_POINT[fe]
    inside = (lo <= pub <= hi)
    gap = 0.0 if inside else (lo - pub if pub < lo else pub - hi)
    z = (pub - b) / se
    pv = 2 * stats.t.sf(abs(z), m.df_resid)
    verdicts[fe] = (pub, b, se, lo, hi, inside, gap, z, pv)
    print(f'{fe:9s} {pub:10.3f} {b:10.3f} {se:11.3f} {lo:9.3f} {hi:9.3f} '
          f'{"INSIDE" if inside else "OUTSIDE":>9s} {gap:10.3f} {z:11.3f} {pv:8.3f}')

# ---------------------------------------------- (4) horse race
print()
print("=" * 104)
print("(4) HORSE RACE: EFI + E1 SHARE + E1|E2 SHARE + num_E1E2_tasks, ALL TOGETHER")
print("=" * 104)
race_rhs = ("fragmentation_index + human_E1_fraction + human_aiExposure_fraction "
            "+ num_E1E2_tasks")
print(f'{"column":9s} | {"EFI b":>8s} {"SE":>7s} {"p":>7s} | {"E1 b":>8s} {"SE":>7s} {"p":>7s} '
      f'| {"E1|E2 b":>8s} {"SE":>7s} {"p":>7s} | {"ctrl b":>7s} {"p":>7s} | {"R2":>6s}')
race = {}
for fe in FE:
    m = fit(race_rhs, fe)
    race[fe] = m
    def g(v):
        return m.params[v], m.bse[v], m.pvalues[v]
    fb, fs, fp = g("fragmentation_index")
    ab, asd, ap = g("human_E1_fraction")
    eb, es, ep = g("human_aiExposure_fraction")
    cb, _, cp = g("num_E1E2_tasks")
    print(f'{fe:9s} | {fb:8.3f} {fs:7.3f} {fp:7.3f} | {ab:8.3f} {asd:7.3f} {ap:7.3f} '
          f'| {eb:8.3f} {es:7.3f} {ep:7.3f} | {cb:7.3f} {cp:7.3f} | {m.rsquared:6.3f}')

print()
print("  Orthogonal reparameterisation (E1 share and E2 share entered separately;")
print("  spans the same column space as {E1 share, E1|E2 share}):")
alt_rhs = ("fragmentation_index + human_E1_fraction + human_E2_fraction + num_E1E2_tasks")
print(f'  {"column":9s} | {"EFI b":>8s} {"SE":>7s} {"p":>7s} | {"E1 b":>8s} {"p":>7s} '
      f'| {"E2 b":>8s} {"p":>7s} | {"R2":>6s}')
for fe in FE:
    m = fit(alt_rhs, fe)
    print(f'  {fe:9s} | {m.params["fragmentation_index"]:8.3f} {m.bse["fragmentation_index"]:7.3f} '
          f'{m.pvalues["fragmentation_index"]:7.3f} | {m.params["human_E1_fraction"]:8.3f} '
          f'{m.pvalues["human_E1_fraction"]:7.3f} | {m.params["human_E2_fraction"]:8.3f} '
          f'{m.pvalues["human_E2_fraction"]:7.3f} | {m.rsquared:6.3f}')

print()
print("  Wald tests inside the horse race (cluster-robust chi2, 1 df):")
for fe in FE:
    m = race[fe]
    w1 = m.wald_test("human_E1_fraction = 0", use_f=False, scalar=True)
    w2 = m.wald_test("human_aiExposure_fraction = 0", use_f=False, scalar=True)
    w3 = m.wald_test("fragmentation_index = 0", use_f=False, scalar=True)
    w4 = m.wald_test("fragmentation_index = 0, human_E1_fraction = 0", use_f=False)
    print(f'  {fe:9s} EFI chi2={float(w3.statistic):7.3f} p={float(w3.pvalue):.3f} | '
          f'E1 chi2={float(w1.statistic):7.3f} p={float(w1.pvalue):.3f} | '
          f'E1|E2 chi2={float(w2.statistic):7.3f} p={float(w2.pvalue):.3f} | '
          f'joint(EFI,E1) chi2={float(np.squeeze(w4.statistic)):7.3f} p={float(np.squeeze(w4.pvalue)):.3f}')

print()
print("  Auxiliary: what does the published EFI coefficient stand in for?")
print("  Regress the PUBLISHED-spec EFI coefficient's omitted variable (E1|E2 share) on the")
print("  published RHS; OVB = delta * beta_[E1|E2 in the matched spec].")
for fe in FE:
    aux = smf.ols(f"human_aiExposure_fraction ~ fragmentation_index + human_E1_fraction "
                  f"+ num_E1E2_tasks{FE[fe]}", data=occ).fit()
    delta = aux.params["fragmentation_index"]
    bmatch = models[("MATCHED", fe)].params["human_aiExposure_fraction"]
    bpub = models[("PUBLISHED", fe)].params["fragmentation_index"]
    bmat_efi = models[("MATCHED", fe)].params["fragmentation_index"]
    print(f'  {fe:9s} delta(EFI -> E1|E2) = {delta:7.3f}  x  beta_E1E2 = {bmatch:6.3f}  '
          f'=> OVB {delta*bmatch:7.3f};  matched EFI {bmat_efi:7.3f} + OVB = '
          f'{bmat_efi + delta*bmatch:7.3f}  vs published EFI {bpub:7.3f}')

# ---------------------------------------------- (5) simulation
print()
print("=" * 104)
print("(5) RESIDUAL-BOOTSTRAP POWER SIMULATION ON THE MATCHED DESIGN")
print("=" * 104)

rng = np.random.default_rng(20260901)
NDRAW = 4000
BETAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def cluster_V_singleton(X, u, XtX_inv, n, k):
    """statsmodels cluster sandwich when every cluster is a single observation,
    with use_correction=True and df_correction=True."""
    Xu = X * u[:, None]
    V = XtX_inv @ (Xu.T @ Xu) @ XtX_inv
    G = n
    return V * (G / (G - 1.0)) * ((n - 1.0) / (n - k))


print("  fast-path verification against statsmodels:")
for fe in FE:
    m = models[("MATCHED", fe)]
    X = np.asarray(m.model.exog, float)
    y = np.asarray(m.model.endog, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    V = cluster_V_singleton(X, y - X @ b, XtX_inv, n, k)
    j = list(m.model.exog_names).index("fragmentation_index")
    print(f'    {fe:9s} b {b[j]: .6f} vs {m.params["fragmentation_index"]: .6f}   '
          f'se {np.sqrt(V[j,j]):.6f} vs {m.bse["fragmentation_index"]:.6f}   '
          f'(n={n}, k={k}, rank={np.linalg.matrix_rank(X)})')

print()
print(f"  {NDRAW} draws per cell. DGP: y* = X_(-EFI) gamma_hat + beta * EFI + e*, with EFI")
print("  and all other regressors held at their observed z-scored values. Two residual")
print("  bootstraps: wild Rademacher (e*_i = u_i v_i, v in {-1,+1}) and iid resample of u.")
print("  Rejection = |t| > t_crit(0.975, df) using the same clustered SE as the headline.")
print()

sim_rows = []
for fe in FE:
    m = models[("MATCHED", fe)]
    X = np.asarray(m.model.exog, float)
    y = np.asarray(m.model.endog, float)
    n, k = X.shape
    names = list(m.model.exog_names)
    j = names.index("fragmentation_index")
    XtX_inv = np.linalg.pinv(X.T @ X)
    H = XtX_inv @ X.T                 # k x n, b = H y
    a = X @ XtX_inv[:, j]             # n-vector for the (j,j) sandwich entry
    b_hat = H @ y
    u_hat = y - X @ b_hat
    b_null = b_hat.copy(); b_null[j] = 0.0
    fit_null = X @ b_null
    frag = X[:, j]
    corr = (n / (n - 1.0)) * ((n - 1.0) / (n - k))
    tcrit = stats.t.ppf(0.975, m.df_resid)
    se_obs = m.bse["fragmentation_index"]

    def rejrate(base, mode):
        if mode == "wild":
            e = u_hat[None, :] * rng.choice([-1.0, 1.0], size=(NDRAW, n))
        else:
            e = u_hat[rng.integers(0, n, size=(NDRAW, n))]
        Y = base[None, :] + e
        B = Y @ H.T
        U = Y - B @ X.T
        var = corr * ((a[None, :] * U) ** 2).sum(axis=1)
        return (np.abs(B[:, j] / np.sqrt(var)) > tcrit).mean()

    print(f"  --- {fe}: observed SE = {se_obs:.3f}, analytic MDE = {MDE_MULT*se_obs:.3f}, "
          f"t_crit = {tcrit:.3f} ---")
    print(f'  {"beta":>6s} {"wild rej":>10s} {"iid rej":>9s} {"analytic power":>15s} '
          f'{"pp of AI-exec":>14s}')
    xs, ys = [], []
    for beta in [0.0] + BETAS:
        base = fit_null + beta * frag
        rw = rejrate(base, "wild")
        ri = rejrate(base, "iid")
        ap = (stats.norm.sf(ZCRIT - beta / se_obs)
              + stats.norm.cdf(-ZCRIT - beta / se_obs))
        tag = "  (size check)" if beta == 0.0 else ""
        print(f'  {beta:6.2f} {rw:10.3f} {ri:9.3f} {ap:15.3f} '
              f'{beta*100*SD_Y:14.3f}{tag}')
        xs.append(beta); ys.append(rw)
        sim_rows.append({"fe": fe, "beta": beta, "wild": rw, "iid": ri, "analytic": ap})
    ys_a = np.array(ys); xs_a = np.array(xs)
    if ys_a.max() >= 0.80:
        b80 = np.interp(0.80, ys_a, xs_a)
        print(f'  simulated beta at 80% power (linear interp on wild): {b80:.3f}  '
              f'vs analytic MDE {MDE_MULT*se_obs:.3f}  '
              f'(ratio {b80/(MDE_MULT*se_obs):.3f})')
    else:
        print(f'  wild rejection never reaches 0.80 over the grid (max {ys_a.max():.3f})')
    print()

pd.DataFrame(sim_rows).to_csv(os.path.join(OUT, "mde_headline_simulation.csv"), index=False)

# ---------------------------------------------- (6) translation to percentage points
print("=" * 104)
print("(6) TRANSLATION INTO PERCENTAGE POINTS OF THE AI-EXECUTION SHARE")
print("=" * 104)
print(f"  A z-unit coefficient beta means: a 1-sd rise in the EFI (= {SD_EFI:.4f} index points,")
print(f"  i.e. {SD_EFI*100:.2f} pp more of the workflow spent switching) moves the AI-execution")
print(f"  share by beta * sd(y) = beta * {SD_Y:.4f} = beta * {SD_Y*100:.2f} pp.")
print(f"  Per raw index point: beta * sd(y)/sd(EFI) = beta * {SD_Y/SD_EFI:.4f}.")
print()
print(f'{"spec":10s} {"column":9s} {"b (z)":>8s} {"pp per sd":>10s} '
      f'{"CI low pp":>10s} {"CI high pp":>11s} {"MDE pp":>8s} {"pp per +0.10 EFI":>17s}')
for sname in SPECS:
    for fe in FE:
        m = models[(sname, fe)]
        b, se = m.params["fragmentation_index"], m.bse["fragmentation_index"]
        lo, hi = m.conf_int().loc["fragmentation_index"]
        per_raw10 = b * (SD_Y / SD_EFI) * 0.10 * 100
        print(f'{sname:10s} {fe:9s} {b:8.3f} {b*100*SD_Y:10.3f} {lo*100*SD_Y:10.3f} '
              f'{hi*100*SD_Y:11.3f} {MDE_MULT*se*100*SD_Y:8.3f} {per_raw10:17.3f}')

print()
print("  Published point estimates in the same units, for comparison:")
for fe in FE:
    pub = PUBLISHED_POINT[fe]
    print(f'    {fe:9s} {pub:7.3f} z  =  {pub*100*SD_Y:7.3f} pp per sd of EFI  '
          f'=  {pub*(SD_Y/SD_EFI)*0.10*100:7.3f} pp per +0.10 raw EFI')

print()
print("  Matched exposure coefficients in pp (for context on what DID move):")
for sname, (rhs, expvar) in SPECS.items():
    for fe in FE:
        m = models[(sname, fe)]
        b = m.params[expvar]
        print(f'    {sname:10s} {fe:9s} {expvar:26s} {b:7.3f} z = {b*100*SD_Y:7.3f} pp per sd')

print()
print("=" * 104)
print("DONE")
print("=" * 104)
