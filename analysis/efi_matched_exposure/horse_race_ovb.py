#!/usr/bin/env python
"""Addendum: the EXACT omitted-variable identity linking the published EFI coefficient
to the horse-race coefficients, plus equivalence-test framing of the matched CI.

Read-only with respect to the repo.
"""
import os
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
CODE_VAR, TITLE_VAR = "O*NET-SOC Code", "Occupation Title"

merged = pd.read_csv(os.path.join(DATA, "computed_objects",
                                  "ONET_Eloundou_Anthropic_GPT",
                                  "ONET_Eloundou_Anthropic_GPT.csv"))
c = merged.groupby(CODE_VAR)["Task ID"].nunique()
merged = merged[merged[CODE_VAR].isin(c[c >= 3].index)].reset_index(drop=True)
ONET = pd.read_csv(os.path.join(DATA, "computed_objects", "ONET_cleaned_tasks.csv"))
SOC = ONET[[CODE_VAR, "Major_Group_Code", "Minor_Group_Code"]].drop_duplicates(subset=[CODE_VAR])

rows = []
for (soc, title), g in merged.groupby([CODE_VAR, TITLE_VAR]):
    n = len(g)
    rows.append({CODE_VAR: soc, TITLE_VAR: title,
                 "ai_fraction": ((g["label"] == "Augmentation").sum()
                                 + (g["label"] == "Automation").sum()) / n,
                 "human_E1_fraction": (g["human_labels"] == "E1").sum() / n,
                 "human_aiExposure_fraction": g["human_labels"].isin(["E1", "E2"]).sum() / n,
                 "num_E1E2_tasks": int(g["human_labels"].isin(["E1", "E2"]).sum())})
occ = pd.DataFrame(rows)

f = merged.copy()
f["is_ai"] = f["human_labels"].isin(["E1", "E2"]).astype(int)
f["next_is_ai"] = f.groupby([CODE_VAR, TITLE_VAR])["is_ai"].shift(-1).fillna(0).astype(int)
f["num_switches"] = 1
f.loc[(f["is_ai"] == 1) & (f["next_is_ai"] == 1), "num_switches"] = 0
fi = (f.groupby([CODE_VAR, TITLE_VAR])["num_switches"].mean().reset_index()
      .rename(columns={"num_switches": "fragmentation_index"}))
occ = occ.merge(fi, on=[CODE_VAR, TITLE_VAR]).merge(SOC, on=[CODE_VAR], how="left")
for col in ("Major_Group_Code", "Minor_Group_Code", CODE_VAR):
    occ[col] = occ[col].astype("object")

RAW = occ.copy()
SD_Y = RAW["ai_fraction"].std()
SD_EFI = RAW["fragmentation_index"].std()
for col in ["ai_fraction", "fragmentation_index", "num_E1E2_tasks",
            "human_E1_fraction", "human_aiExposure_fraction"]:
    occ[col] = (RAW[col] - RAW[col].mean()) / RAW[col].std()

FE = {"no FE": "", "major FE": " + C(Major_Group_Code)", "minor FE": " + C(Minor_Group_Code)"}
PUB = {"no FE": -0.261, "major FE": -0.380, "minor FE": -0.283}


def fit(rhs, fe, y="ai_fraction", cluster=True):
    mod = smf.ols(f"{y} ~ {rhs}{FE[fe]}", data=occ)
    if not cluster:
        return mod.fit()
    return mod.fit(cov_type="cluster",
                   cov_kwds={"groups": occ[CODE_VAR], "use_correction": True,
                             "df_correction": True})


PUB_RHS = "fragmentation_index + human_E1_fraction + num_E1E2_tasks"
MAT_RHS = "fragmentation_index + human_aiExposure_fraction + num_E1E2_tasks"
RACE_RHS = ("fragmentation_index + human_E1_fraction + human_aiExposure_fraction "
            "+ num_E1E2_tasks")

print("=" * 100)
print("EXACT OMITTED-VARIABLE IDENTITY")
print("  long  = horse race  y ~ EFI + E1 + E1|E2 + ctrl (+FE)")
print("  short = published   y ~ EFI + E1 + ctrl (+FE)     [omits the E1|E2 share]")
print("  b_short(EFI) = b_long(EFI) + b_long(E1|E2) * delta,")
print("  delta = coefficient on EFI when the E1|E2 share is regressed on the published RHS")
print("=" * 100)
print(f'{"column":9s} {"b_long(EFI)":>12s} {"b_long(E1|E2)":>14s} {"delta":>8s} '
      f'{"OVB":>8s} {"implied":>9s} {"actual pub":>11s} {"resid":>8s}')
for fe in FE:
    long_m = fit(RACE_RHS, fe)
    short_m = fit(PUB_RHS, fe)
    aux = fit(PUB_RHS, fe, y="human_aiExposure_fraction", cluster=False)
    bl = long_m.params["fragmentation_index"]
    bz = long_m.params["human_aiExposure_fraction"]
    d = aux.params["fragmentation_index"]
    imp = bl + bz * d
    act = short_m.params["fragmentation_index"]
    print(f'{fe:9s} {bl:12.3f} {bz:14.3f} {d:8.3f} {bz*d:8.3f} {imp:9.3f} '
          f'{act:11.3f} {imp-act:8.2e}   (paper prints {PUB[fe]:.3f})')

print()
print("  Share of the published EFI coefficient that is pure omitted-E1|E2 bias:")
for fe in FE:
    long_m = fit(RACE_RHS, fe)
    short_m = fit(PUB_RHS, fe)
    aux = fit(PUB_RHS, fe, y="human_aiExposure_fraction", cluster=False)
    ovb = long_m.params["human_aiExposure_fraction"] * aux.params["fragmentation_index"]
    act = short_m.params["fragmentation_index"]
    print(f'    {fe:9s} OVB {ovb:7.3f} / published {act:7.3f} = {100*ovb/act:6.1f}%')

print()
print("  Mirror identity: what the MATCHED EFI coefficient omits (the E1 share)")
print(f'  {"column":9s} {"b_long(EFI)":>12s} {"b_long(E1)":>11s} {"delta2":>8s} '
      f'{"OVB":>8s} {"implied":>9s} {"actual matched":>15s}')
for fe in FE:
    long_m = fit(RACE_RHS, fe)
    mat_m = fit(MAT_RHS, fe)
    aux2 = fit(MAT_RHS, fe, y="human_E1_fraction", cluster=False)
    bl = long_m.params["fragmentation_index"]
    be1 = long_m.params["human_E1_fraction"]
    d2 = aux2.params["fragmentation_index"]
    print(f'  {fe:9s} {bl:12.3f} {be1:11.3f} {d2:8.3f} {be1*d2:8.3f} '
          f'{bl + be1*d2:9.3f} {mat_m.params["fragmentation_index"]:15.3f}')

# ----------------------------------------------------------- equivalence framing
print()
print("=" * 100)
print("EQUIVALENCE / EXCLUSION FRAMING OF THE MATCHED CI")
print("=" * 100)
print("  TOST-style: the largest |effect| the matched design rules out at 5% one-sided")
print("  is the CI bound; the smallest it rules out is nothing (CI contains 0).")
print(f'{"column":9s} {"b":>8s} {"SE":>7s} {"CI":>20s} {"most neg. not excl.":>21s} '
      f'{"= pp":>8s} {"pub excluded?":>14s}')
for fe in FE:
    m = fit(MAT_RHS, fe)
    b, se = m.params["fragmentation_index"], m.bse["fragmentation_index"]
    lo, hi = m.conf_int().loc["fragmentation_index"]
    print(f'{fe:9s} {b:8.3f} {se:7.3f} {f"[{lo:.3f}, {hi:.3f}]":>20s} {lo:21.3f} '
          f'{lo*100*SD_Y:8.3f} {"YES" if PUB[fe] < lo else "no":>14s}')

print()
print("  One-sided test of H0: beta_EFI <= published value, against beta_EFI > published:")
for fe in FE:
    m = fit(MAT_RHS, fe)
    b, se = m.params["fragmentation_index"], m.bse["fragmentation_index"]
    z = (b - PUB[fe]) / se
    p1 = stats.t.sf(z, m.df_resid)
    print(f'    {fe:9s} t = ({b:.3f} - ({PUB[fe]:.3f})) / {se:.3f} = {z:6.3f}, '
          f'one-sided p = {p1:.4f}, two-sided p = {2*p1:.4f}')

print()
print("  How large would the true |beta| have to be for the matched design to reject")
print("  it 80% of the time, expressed as pp of the AI-execution share:")
MDE_MULT = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
for fe in FE:
    m = fit(MAT_RHS, fe)
    se = m.bse["fragmentation_index"]
    print(f'    {fe:9s} MDE = {MDE_MULT*se:.3f} z = {MDE_MULT*se*100*SD_Y:.3f} pp per sd of EFI '
          f'= {MDE_MULT*se*(SD_Y/SD_EFI)*0.10*100:.3f} pp per +0.10 raw EFI')

print()
print("  Published-spec SE vs matched-spec SE (precision cost of the fix):")
for fe in FE:
    sp = fit(PUB_RHS, fe).bse["fragmentation_index"]
    sm = fit(MAT_RHS, fe).bse["fragmentation_index"]
    print(f'    {fe:9s} published SE {sp:.3f} -> matched SE {sm:.3f}  '
          f'(ratio {sm/sp:.3f}, variance inflation {(sm/sp)**2:.3f})')

# VIF-style diagnostic
print()
print("  Partial R2 of the EFI given the other regressors (why the SE inflates):")
for fe in FE:
    for lab, rhs_o in (("published", "human_E1_fraction + num_E1E2_tasks"),
                       ("matched", "human_aiExposure_fraction + num_E1E2_tasks"),
                       ("horse race", "human_E1_fraction + human_aiExposure_fraction "
                                      "+ num_E1E2_tasks")):
        a = fit(rhs_o, fe, y="fragmentation_index", cluster=False)
        print(f'    {fe:9s} {lab:11s} R2(EFI on others) = {a.rsquared:.4f}  '
              f'VIF = {1/(1-a.rsquared):8.3f}  sd of residual EFI = '
              f'{np.std(a.resid, ddof=1):.4f}')
print()
print("=" * 100)
