"""
Re-estimation of main-text Table 2 (EFI Definition 1, exposure-based)
and its SA.B counterpart (EFI Definition 2, execution-based) under the MATCHED
specification.

Reimplements analysis/onet_fragmentationIndex.ipynb.
Writes its output to data/computed_objects/efi_matched_exposure/. Does not touch any
published exhibit under writeup/tables/ or the paper's own computed objects.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import os as _os
# Repo root and output dir are derived from this file's own location, so the folder
# can move and no home directory is baked in. Generated output follows the repo
# convention: data/computed_objects/<name of this analysis folder>/.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
OUT = _os.path.join(REPO, "data", "computed_objects", "efi_matched_exposure")
_os.makedirs(OUT, exist_ok=True)


CODE = "O*NET-SOC Code"
TITLE = "Occupation Title"

# ----------------------------------------------------------------------------- load
merged = pd.read_csv(f"{REPO}/data/computed_objects/ONET_Eloundou_Anthropic_GPT/"
                     "ONET_Eloundou_Anthropic_GPT.csv")
counts = merged.groupby(CODE)["Task ID"].nunique()
valid = counts[counts >= 3].index
merged = merged[merged[CODE].isin(valid)].reset_index(drop=True)

ONET = pd.read_csv(f"{REPO}/data/computed_objects/ONET_cleaned_tasks.csv")
SOC = ONET[[CODE, TITLE, "Major_Group_Code", "Major_Group_Title",
            "Minor_Group_Code", "Minor_Group_Title",
            "Broad_Occupation_Code", "Broad_Occupation_Title",
            "Detailed_Occupation_Code", "Detailed_Occupation_Title"]].copy()
SOC = SOC.drop_duplicates(subset=[CODE, CODE])


# ------------------------------------------------------------------ occupation stats
def occupation_analysis(df):
    rows = []
    for (soc, occ), g in df.groupby([CODE, TITLE]):
        total = len(g)
        aug = (g["label"] == "Augmentation").sum() / total
        aut = (g["label"] == "Automation").sum() / total
        hE1 = (g["human_labels"] == "E1").sum() / total
        hE2 = (g["human_labels"] == "E2").sum() / total
        rows.append({
            CODE: soc, TITLE: occ,
            "num_tasks": g["Task ID"].nunique(),
            "n_rows": total,
            "ai_fraction": aug + aut,
            "human_E1_fraction": hE1,
            "human_E2_fraction": hE2,
            "human_aiExposure_fraction": hE1 + hE2,
            "num_E1E2_tasks": int(g["human_labels"].isin(["E1", "E2"]).sum()),
            "num_exec_tasks": int(g["label"].isin(["Augmentation", "Automation"]).sum()),
        })
    return pd.DataFrame(rows)


def fragmentation_index(df, definition):
    fi = df.copy()
    if definition == 1:
        fi["is_ai"] = fi["human_labels"].isin(["E1", "E2"]).astype(int)
    elif definition == 2:
        fi["is_ai"] = fi["label"].isin(["Augmentation", "Automation"]).astype(int)
    fi["next_is_ai"] = fi.groupby([CODE, TITLE])["is_ai"].shift(-1).fillna(0).astype(int)
    fi["num_switches"] = 1
    fi.loc[(fi["is_ai"] == 1) & (fi["next_is_ai"] == 1), "num_switches"] = 0
    # r = number of maximal runs of AI-able steps
    fi["run_start"] = ((fi["is_ai"] == 1) &
                       (fi.groupby([CODE, TITLE])["is_ai"].shift(1).fillna(0).astype(int) == 0)
                       ).astype(int)
    agg = fi.groupby([CODE, TITLE]).agg(
        fragmentation_index=("num_switches", "mean"),
        k=("is_ai", "sum"),
        r=("run_start", "sum"),
        m=("is_ai", "size"),
    ).reset_index()
    return agg


def build(definition, exposure_var):
    occ = occupation_analysis(merged)
    fi = fragmentation_index(merged, definition)
    occ = occ.merge(fi, on=[CODE, TITLE], how="left")
    occ = occ.merge(SOC, on=[CODE, TITLE], how="left")
    d = occ.groupby([CODE, TITLE]).agg({
        "fragmentation_index": "mean",
        exposure_var: "mean",
        "ai_fraction": "mean",
        "num_tasks": "mean",
        "num_E1E2_tasks": "mean",
        "k": "mean", "r": "mean", "m": "mean",
    }).reset_index()
    d = d.merge(SOC.drop_duplicates(subset=[CODE]), on=CODE, how="left",
                suffixes=("", "_drop"))
    d = d.loc[:, ~d.columns.str.endswith("_drop")]
    d = d.rename(columns={exposure_var: "ai_exposure"})
    for c in ("Major_Group_Code", "Minor_Group_Code", CODE):
        d[c] = d[c].astype("object")
    return d


def zscore(d, cols=("ai_fraction", "ai_exposure", "fragmentation_index", "num_E1E2_tasks")):
    d = d.copy()
    sds = {}
    for c in cols:
        s = d[c]
        sds[c] = s.std()
        d[c] = (s - s.mean()) / s.std()
    return d, sds


def run(d, fe=None):
    f = "ai_fraction ~ fragmentation_index + ai_exposure + num_E1E2_tasks"
    if fe:
        f += f" + C({fe})"
    return smf.ols(formula=f, data=d).fit(
        cov_type="cluster",
        cov_kwds={"groups": d[CODE], "use_correction": True, "df_correction": True})


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def report(models, header):
    print("\n" + "=" * 100)
    print(header)
    print("=" * 100)
    names = ["(1) no FE", "(2) Major FE", "(3) Minor FE"]
    for nm, m in zip(names, models):
        print(f"\n--- {nm} ---")
        print(f"{'term':<22}{'coef':>9}{'clSE':>9}{'t':>9}{'p':>11}{'sig':>5}"
              f"{'CI low':>10}{'CI high':>10}")
        ci = m.conf_int(alpha=0.05)
        for t in ["fragmentation_index", "ai_exposure", "num_E1E2_tasks", "Intercept"]:
            b, se, tv, pv = m.params[t], m.bse[t], m.tvalues[t], m.pvalues[t]
            lo, hi = ci.loc[t, 0], ci.loc[t, 1]
            print(f"{t:<22}{b:>9.4f}{se:>9.4f}{tv:>9.3f}{pv:>11.3e}{stars(pv):>5}"
                  f"{lo:>10.4f}{hi:>10.4f}")
        print(f"R2 = {m.rsquared:.4f}   AdjR2 = {m.rsquared_adj:.4f}   N = {int(m.nobs)}")


# ============================================================== 1. REPLICATE PUBLISHED
print("#" * 100)
print("STEP 1: REPLICATION OF PUBLISHED SPECIFICATION (exposure = human_E1_fraction)")
print("#" * 100)

d1_pub_raw = build(1, "human_E1_fraction")
d1_pub, sds1 = zscore(d1_pub_raw)
pub1 = [run(d1_pub), run(d1_pub, "Major_Group_Code"), run(d1_pub, "Minor_Group_Code")]
report(pub1, "PUBLISHED  Definition 1 (exposure-based EFI), exposure = E1 share")

d2_pub_raw = build(2, "human_E1_fraction")
d2_pub, sds2 = zscore(d2_pub_raw)
pub2 = [run(d2_pub), run(d2_pub, "Major_Group_Code"), run(d2_pub, "Minor_Group_Code")]
report(pub2, "PUBLISHED  Definition 2 (execution-based EFI), exposure = E1 share")

# ============================================================== 2. MATCHED
print("\n" + "#" * 100)
print("STEP 2: MATCHED SPECIFICATION (exposure = human_aiExposure_fraction = E1|E2)")
print("#" * 100)

d1_new_raw = build(1, "human_aiExposure_fraction")
d1_new, sds1n = zscore(d1_new_raw)
new1 = [run(d1_new), run(d1_new, "Major_Group_Code"), run(d1_new, "Minor_Group_Code")]
report(new1, "MATCHED  Definition 1 (exposure-based EFI), exposure = E1|E2 share")

d2_new_raw = build(2, "human_aiExposure_fraction")
d2_new, sds2n = zscore(d2_new_raw)
new2 = [run(d2_new), run(d2_new, "Major_Group_Code"), run(d2_new, "Minor_Group_Code")]
report(new2, "MATCHED  Definition 2 (execution-based EFI), exposure = E1|E2 share")

# ============================================================== 3. IDENTITIES / SDs
print("\n" + "#" * 100)
print("STEP 3: IDENTITIES AND LEVEL SCALES")
print("#" * 100)

for tag, raw in [("Def 1 (E1|E2 runs)", d1_new_raw), ("Def 2 (exec runs)", d2_new_raw)]:
    idn = raw["fragmentation_index"] - (1 - raw["k"] / raw["m"] + raw["r"] / raw["m"])
    print(f"{tag}: max |EFI - (1 - k/m + r/m)| = {np.abs(idn).max():.3e}")

# Definition 2: the level term IS the dependent variable
lhs = d2_new_raw["ai_fraction"]
rhs_k = d2_new_raw["k"] / d2_new_raw["m"]
print(f"\nDef 2: max |ai_fraction - k/m| = {np.abs(lhs - rhs_k).max():.3e}")
resid = d2_new_raw["fragmentation_index"] - (1 - d2_new_raw["ai_fraction"]
                                             + d2_new_raw["r"] / d2_new_raw["m"])
print(f"Def 2: max |EFI2 - (1 - ai_fraction + r/m)| = {np.abs(resid).max():.3e}")
print(f"Def 2: corr(EFI2, ai_fraction) = "
      f"{np.corrcoef(d2_new_raw['fragmentation_index'], d2_new_raw['ai_fraction'])[0,1]:.4f}")

print("\nRaw SDs (ddof=1, pandas default), 872-occupation sample:")
print(f"  SD(ai_fraction, AI-execution share)      = {d1_new_raw['ai_fraction'].std():.4f}")
print(f"  SD(EFI Definition 1)                     = {d1_new_raw['fragmentation_index'].std():.4f}")
print(f"  SD(EFI Definition 2)                     = {d2_new_raw['fragmentation_index'].std():.4f}")
print(f"  SD(E1 share)                             = {d1_pub_raw['ai_exposure'].std():.4f}")
print(f"  SD(E1|E2 share)                          = {d1_new_raw['ai_exposure'].std():.4f}")
print(f"  SD(num_E1E2_tasks)                       = {d1_new_raw['num_E1E2_tasks'].std():.4f}")

print("\ncorr(EFI def1, E1|E2 share) = "
      f"{np.corrcoef(d1_new_raw['fragmentation_index'], d1_new_raw['ai_exposure'])[0,1]:.4f}")
print("corr(EFI def1, E1 share)    = "
      f"{np.corrcoef(d1_pub_raw['fragmentation_index'], d1_pub_raw['ai_exposure'])[0,1]:.4f}")

SD_Y = d1_new_raw["ai_fraction"].std()


def level_translate(models, label, sd_efi):
    print(f"\nLevel-scale translation, {label}")
    print(f"  (beta_std * SD(y) * 100 = pp of AI-execution share per 1 SD of EFI; "
          f"SD(y)={SD_Y:.4f}, SD(EFI)={sd_efi:.4f})")
    for nm, m in zip(["(1) no FE", "(2) Major FE", "(3) Minor FE"], models):
        for t in ["fragmentation_index", "ai_exposure"]:
            b = m.params[t]
            se = m.bse[t]
            ci = m.conf_int(alpha=0.05).loc[t]
            print(f"  {nm:<14}{t:<22} {b*SD_Y*100:+8.3f} pp  "
                  f"(SE {se*SD_Y*100:.3f} pp, 95% CI [{ci[0]*SD_Y*100:+.3f}, "
                  f"{ci[1]*SD_Y*100:+.3f}] pp)")


level_translate(pub1, "PUBLISHED Def 1", d1_new_raw["fragmentation_index"].std())
level_translate(new1, "MATCHED Def 1", d1_new_raw["fragmentation_index"].std())
level_translate(pub2, "PUBLISHED Def 2", d2_new_raw["fragmentation_index"].std())
level_translate(new2, "MATCHED Def 2", d2_new_raw["fragmentation_index"].std())

# ============================================================== 4. LATEX
def latex_table(models, defn):
    def cell(m, t):
        return f"{m.params[t]:.2f}{stars(m.pvalues[t])}"

    def secell(m, t):
        return f"({m.bse[t]:.2f})"

    L = []
    L.append(r"\setlength{\tabcolsep}{12pt} % roomier padding for the narrow three-column layout")
    L.append(r"\begin{tabular}{lccc}")
    L.append(r"\toprule")
    L.append(r" & \multicolumn{3}{c}{Share of AI-executed Tasks} \\")
    L.append(r" \cmidrule(lr){2-4}")
    L.append(r" & (1) & (2) & (3) \\")
    L.append(r"\midrule")
    L.append(r"\addlinespace")
    L.append("Share of AI-exposed Tasks & " +
             " & ".join(cell(m, "ai_exposure") for m in models) + r" \\")
    L.append(" & " + " & ".join(secell(m, "ai_exposure") for m in models) + r" \\")
    L.append(r"\addlinespace")
    L.append(f"Empirical Fragmentation Index (Definition {defn}) & " +
             " & ".join(cell(m, "fragmentation_index") for m in models) + r" \\")
    L.append(" & " + " & ".join(secell(m, "fragmentation_index") for m in models) + r" \\")
    L.append(r"\hline\\[-1.25em]")
    L.append(r"SOC Group Fixed Effect & & Major & Minor \\")
    L.append(r"Number of AI-able Steps Control & \checkmark & \checkmark & \checkmark \\")
    L.append("R-squared & " + " & ".join(f"{m.rsquared:.2f}" for m in models) + r" \\")
    L.append("Adj. R-squared & " + " & ".join(f"{m.rsquared_adj:.2f}" for m in models) + r" \\")
    L.append("Observations & " + " & ".join(f"{int(m.nobs)}" for m in models) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    return "\n".join(L)


print("\n" + "#" * 100)
print("STEP 4: LATEX TABLES")
print("#" * 100)
print("\n%%% fragmentation_index_regression_exposure.tex  (MATCHED, Definition 1)")
print(latex_table(new1, 1))
print("\n%%% fragmentation_index_regression_execution.tex  (MATCHED, Definition 2)")
print(latex_table(new2, 2))

print("\n%%% [check] PUBLISHED Definition 1 regenerated by this script")
print(latex_table(pub1, 1))
print("\n%%% [check] PUBLISHED Definition 2 regenerated by this script")
print(latex_table(pub2, 2))

# The matched tables are written to the analysis output folder, NOT into writeup/tables/.
# They are drop-in replacements for the published files if and when the draft adopts the
# matched specification; nothing in the paper reads them from here.
for _name, _models, _defn in [
    ("fragmentation_index_regression_exposure_MATCHED.tex", new1, 1),
    ("fragmentation_index_regression_execution_MATCHED.tex", new2, 2),
    ("fragmentation_index_regression_exposure_PUBLISHED_REPLICATED.tex", pub1, 1),
    ("fragmentation_index_regression_execution_PUBLISHED_REPLICATED.tex", pub2, 2),
]:
    with open(os.path.join(OUT, _name), "w", encoding="utf-8") as _f:
        _f.write(latex_table(_models, _defn) + "\n")
    print("wrote", os.path.join(OUT, _name))
