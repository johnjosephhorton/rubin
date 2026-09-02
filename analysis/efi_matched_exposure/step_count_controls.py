"""
Does the fragmentation result depend on which step-count control is in the regression?

Two different objects get called "the step-count control" in the draft, and they are not
the same variable:

    k = num_E1E2_tasks   the COUNT of AI-able (E1 or E2) steps.  This is what Equation (11)
                         actually controls for, printed as "Number of AI-able Steps Control".
    m = num_tasks        the TOTAL number of steps in the workflow.  This is NOT in the
                         regression. Review issue D29 argues it should be, because
                         EFI = 1 - k/m + r/m still moves with m once k is held fixed.

This script runs the full grid so both questions are answered separately:

    sample   x  exposure regressor  x  count control          x  fixed effects
    872/871     E1 / E1|E2             none / k / m / log m / k+m    none / major / minor

Reports the EFI coefficient with clustered SE and p in every cell, plus the coefficients on
the controls themselves, the variance inflation on the EFI, and how much of the index's
variation survives each conditioning set. Writes nothing outside the analysis output folder.
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

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

MERGED = os.path.join(REPO, "data/computed_objects/ONET_Eloundou_Anthropic_GPT/ONET_Eloundou_Anthropic_GPT.csv")
CLEAN = os.path.join(REPO, "data/computed_objects/ONET_cleaned_tasks.csv")

OCC, TITLE = "O*NET-SOC Code", "Occupation Title"


def build_panel(drop_unlabelled):
    """Occupation panel, exactly as analysis/onet_fragmentationIndex.ipynb builds it.

    drop_unlabelled=False reproduces the paper's 872-occupation headline sample: rows with a
    missing Eloundou label stay in the workflow and count toward m.
    drop_unlabelled=True reproduces the 871-occupation sample used by the repo's own
    exposure_definition_grid.csv, which removes those rows before computing m and r.
    """
    df = pd.read_csv(MERGED)
    cnt = df.groupby(OCC)["Task ID"].nunique()
    df = df[df[OCC].isin(cnt[cnt >= 3].index)].reset_index(drop=True)
    if drop_unlabelled:
        df = df.dropna(subset=["human_labels"]).reset_index(drop=True)

    d = df.copy()
    d["is_ai"] = d["human_labels"].isin(["E1", "E2"]).astype(int)
    d["nxt"] = d.groupby([OCC, TITLE])["is_ai"].shift(-1).fillna(0).astype(int)
    d["prv"] = d.groupby([OCC, TITLE])["is_ai"].shift(1).fillna(0).astype(int)
    d["sw"] = 1
    d.loc[(d["is_ai"] == 1) & (d["nxt"] == 1), "sw"] = 0
    d["run_start"] = ((d["is_ai"] == 1) & (d["prv"] == 0)).astype(int)

    g = d.groupby([OCC, TITLE])
    p = pd.DataFrame({
        "EFI": g["sw"].mean(),
        "m": g["sw"].size(),
        "k": g["is_ai"].sum(),
        "r": g["run_start"].sum(),
        "ai_fraction": g["label"].apply(lambda s: s.isin(["Augmentation", "Automation"]).mean()),
        "E1": g["human_labels"].apply(lambda s: (s == "E1").mean()),
        "E1E2": g["human_labels"].apply(lambda s: s.isin(["E1", "E2"]).mean()),
    }).reset_index()

    soc = pd.read_csv(CLEAN)[[OCC, "Major_Group_Code", "Minor_Group_Code"]].drop_duplicates(subset=[OCC])
    p = p.merge(soc, on=OCC, how="left")
    # Coerce the SOC string columns to plain object dtype so patsy can build the design
    # matrix on pandas >= 3.0, as analysis/onet_fragmentationIndex.ipynb also has to do.
    for c in ("Major_Group_Code", "Minor_Group_Code", OCC):
        p[c] = p[c].astype("object")
    p["logm"] = np.log(p["m"])
    assert np.abs(p["EFI"] - (1 - p["k"] / p["m"] + p["r"] / p["m"])).max() < 1e-12
    return p


def z(s):
    return (s - s.mean()) / s.std()


CONTROLS = {
    "none":   [],
    "k":      ["k"],            # the paper's control: count of AI-able steps
    "m":      ["m"],            # workflow length, the D29 proposal
    "log m":  ["logm"],
    "k + m":  ["k", "m"],
}
FE = {"none": "", "major": " + C(Major_Group_Code)", "minor": " + C(Minor_Group_Code)"}


def fit(panel, expo_col, ctrl_key, fe_key):
    d = panel.copy()
    d["y"] = z(d["ai_fraction"])
    d["efi"] = z(d["EFI"])
    d["expo"] = z(d[expo_col])
    terms = ["efi", "expo"]
    for c in CONTROLS[ctrl_key]:
        d["c_" + c] = z(d[c])
        terms.append("c_" + c)
    f = "y ~ " + " + ".join(terms) + FE[fe_key]
    m = smf.ols(f, data=d).fit(cov_type="cluster",
                               cov_kwds={"groups": d[OCC], "use_correction": True, "df_correction": True})
    # variance inflation on the EFI: 1/(1-R2) from regressing efi on everything else
    aux_terms = [t for t in terms if t != "efi"]
    aux = smf.ols("efi ~ " + (" + ".join(aux_terms) if aux_terms else "1") + FE[fe_key], data=d).fit()
    return dict(
        b=m.params["efi"], se=m.bse["efi"], p=m.pvalues["efi"],
        lo=m.conf_int().loc["efi", 0], hi=m.conf_int().loc["efi", 1],
        b_expo=m.params["expo"], p_expo=m.pvalues["expo"],
        b_ctrl="; ".join(f"{c}={m.params['c_' + c]:+.3f}(p{m.pvalues['c_' + c]:.2f})"
                         for c in CONTROLS[ctrl_key]) or "-",
        r2=m.rsquared, n=int(m.nobs),
        vif=1.0 / max(1e-12, 1 - aux.rsquared),
        efi_resid_sd=np.sqrt(max(0.0, 1 - aux.rsquared)),  # sd of EFI left, in z units
    )


def star(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


rows = []
for sample, drop in [("872 (paper headline)", False), ("871 (drops unlabelled rows)", True)]:
    panel = build_panel(drop)
    for expo_lab, expo_col in [("E1 (published)", "E1"), ("E1|E2 (matched)", "E1E2")]:
        for ck in CONTROLS:
            for fk in FE:
                rows.append(dict(sample=sample, exposure=expo_lab, control=ck, fe=fk,
                                 **fit(panel, expo_col, ck, fk)))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "step_count_controls.csv"), index=False)

for sample in R["sample"].unique():
    for expo in R["exposure"].unique():
        print("\n" + "=" * 104)
        print(f"SAMPLE: {sample}    EXPOSURE REGRESSOR: {expo}")
        print("=" * 104)
        print(f"{'control':<8} | " + " | ".join(f"{f:^28}" for f in FE))
        print(f"{'':8} | " + " | ".join(f"{'EFI coef (SE)      p':^28}" for _ in FE))
        print("-" * 104)
        for ck in CONTROLS:
            cells = []
            for fk in FE:
                x = R[(R["sample"] == sample) & (R["exposure"] == expo) &
                      (R["control"] == ck) & (R["fe"] == fk)].iloc[0]
                cells.append(f"{x.b:+.3f} ({x.se:.3f}){star(x.p):<3} {x.p:5.3f}")
            print(f"{ck:<8} | " + " | ".join(f"{c:^28}" for c in cells))

print("\n" + "=" * 104)
print("WHAT EACH CONTROL DOES TO THE EFI'S IDENTIFYING VARIATION (matched exposure, 872 sample)")
print("=" * 104)
print(f"{'control':<8} {'fe':<6} {'VIF':>7} {'sd of EFI left':>16} {'EFI coef':>11} {'SE':>8}")
sub = R[(R["sample"].str.startswith("872")) & (R["exposure"].str.startswith("E1|E2"))]
for _, x in sub.iterrows():
    print(f"{x.control:<8} {x.fe:<6} {x.vif:7.2f} {x.efi_resid_sd:15.3f}  {x.b:+11.4f} {x.se:8.4f}")

print("\n" + "=" * 104)
print("CONTROL COEFFICIENTS THEMSELVES (matched exposure, 872 sample)")
print("=" * 104)
for _, x in sub.iterrows():
    print(f"  {x.control:<8} {x.fe:<6} {x.b_ctrl}")

print("\nwrote", os.path.join(OUT, "step_count_controls.csv"))
