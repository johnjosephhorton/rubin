"""
INDEPENDENT SPOT-CHECK, SA.D (Robustness to Alternative GPT Prompts), Prediction #2 leg.

Written from scratch by the verification agent. Reads the repo only; output goes to data/computed_objects/efi_matched_exposure/. No published exhibit is touched.
Reimplements analysis/onet_fragmentationIndex_robustness.ipynb cells 5 and 7.

Construction being reimplemented (quoted logic, not copied code):
  occupation panel: for each (O*NET-SOC Code, Occupation Title) group of the merged
    prompt-specific file, total_tasks = number of ROWS in the group,
      ai_fraction               = share of rows with label in {Augmentation, Automation}
      human_E1_fraction         = share of rows with human_labels == 'E1'
      human_aiExposure_fraction = share of rows with human_labels in {'E1','E2'}
      num_E1E2_tasks            = COUNT of rows with human_labels in {'E1','E2'}
  EFI Definition 1: is_ai = human_labels in {E1,E2}; next_is_ai = within-occupation
      shift(-1) filled 0; num_switches = 1 except 0 when (is_ai & next_is_ai);
      EFI = mean(num_switches) over the occupation's rows = (m - k + r)/m
  z-score ai_fraction, ai_exposure, fragmentation_index, num_E1E2_tasks (sample sd, ddof=1)
  OLS with cluster-robust SE on O*NET-SOC Code, use_correction=True, df_correction=True
  Three models: no FE, C(Major_Group_Code), C(Minor_Group_Code)

OLD leg  : ai_exposure = human_E1_fraction          (what the published figure uses)
MATCHED  : ai_exposure = human_aiExposure_fraction  (E1|E2, matched to the EFI)

NOTE the notebook does NOT apply the >=3-task occupation filter (it is commented out);
all 872 occupations in each prompt file are used.
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
REPO = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
OUT = _os.path.join(REPO, "data", "computed_objects", "efi_matched_exposure")
_os.makedirs(OUT, exist_ok=True)


DATA = os.path.join(REPO, "data", "computed_objects", "ONET_Eloundou_Anthropic_GPT")

SOC = "O*NET-SOC Code"
TITLE = "Occupation Title"

USECOLS = [SOC, TITLE, "Task ID", "Task Position", "human_labels", "label",
           "Major_Group_Code", "Minor_Group_Code"]


def prompt_file(x):
    if x == 0:
        return os.path.join(DATA, "ONET_Eloundou_Anthropic_GPT.csv")
    return os.path.join(DATA, f"ONET_Eloundou_Anthropic_GPT_{x}.csv")


def build_panel(df):
    """Occupation-level panel: EFI(def 1), exposure shares, execution share, step count."""
    d = df.copy()
    # ---- EFI Definition 1, on the file's own row order (= Task Position order) ----
    is_ai = d["human_labels"].isin(["E1", "E2"]).astype(int)
    d["is_ai"] = is_ai
    d["next_is_ai"] = d.groupby([SOC, TITLE])["is_ai"].shift(-1).fillna(0).astype(int)
    d["num_switches"] = 1
    d.loc[(d["is_ai"] == 1) & (d["next_is_ai"] == 1), "num_switches"] = 0

    g = d.groupby([SOC, TITLE], sort=True)
    panel = pd.DataFrame({
        "fragmentation_index": g["num_switches"].mean(),
        "m_rows": g.size(),
        "num_E1E2_tasks": g["is_ai"].sum().astype(int),
        "ai_fraction": g["label"].apply(lambda s: s.isin(["Augmentation", "Automation"]).sum() / len(s)),
        "human_E1_fraction": g["human_labels"].apply(lambda s: (s == "E1").sum() / len(s)),
        "human_aiExposure_fraction": g["human_labels"].apply(lambda s: s.isin(["E1", "E2"]).sum() / len(s)),
        "Major_Group_Code": g["Major_Group_Code"].first(),
        "Minor_Group_Code": g["Minor_Group_Code"].first(),
    }).reset_index()
    return panel


def run_models(panel, exposure_col):
    p = panel.copy()
    p["ai_exposure"] = p[exposure_col]
    for c in ("Major_Group_Code", "Minor_Group_Code", SOC):
        p[c] = p[c].astype("object")
    for c in ["ai_fraction", "ai_exposure", "fragmentation_index", "num_E1E2_tasks"]:
        s = p[c].astype(float)
        p[c] = (s - s.mean()) / s.std()
    groups = p[SOC]
    specs = {
        "noFE": "ai_fraction ~ fragmentation_index + ai_exposure + num_E1E2_tasks",
        "majorFE": "ai_fraction ~ fragmentation_index + ai_exposure + num_E1E2_tasks + C(Major_Group_Code)",
        "minorFE": "ai_fraction ~ fragmentation_index + ai_exposure + num_E1E2_tasks + C(Minor_Group_Code)",
    }
    out = []
    for name, f in specs.items():
        res = smf.ols(f, data=p).fit(cov_type="cluster",
                                     cov_kwds={"groups": groups, "use_correction": True,
                                               "df_correction": True})
        for term in ["fragmentation_index", "ai_exposure", "num_E1E2_tasks"]:
            out.append({
                "model": name, "term": term,
                "coef": res.params[term], "se": res.bse[term],
                "t": res.tvalues[term], "p": res.pvalues[term],
                "ci_lo": res.conf_int().loc[term, 0], "ci_hi": res.conf_int().loc[term, 1],
                "nobs": int(res.nobs), "r2": float(res.rsquared),
            })
    return pd.DataFrame(out)


PROMPTS = list(range(0, 11))  # all eleven; the task asks for at least four

rows = []
for x in PROMPTS:
    f = prompt_file(x)
    df = pd.read_csv(f, usecols=USECOLS)
    panel = build_panel(df)
    # descriptive: EFI / exposure collinearity for this prompt
    corr = np.corrcoef(panel["fragmentation_index"], panel["human_aiExposure_fraction"])[0, 1]
    for leg, col in [("OLD_E1", "human_E1_fraction"),
                     ("MATCHED_E1E2", "human_aiExposure_fraction")]:
        r = run_models(panel, col)
        r["prompt"] = x
        r["leg"] = leg
        r["n_occ"] = len(panel)
        r["corr_EFI_E1E2share"] = corr
        rows.append(r)
    print(f"prompt {x}: n_occ={len(panel)} rows={len(df)} "
          f"corr(EFI,E1|E2 share)={corr:.4f} meanEFI={panel['fragmentation_index'].mean():.4f}",
          flush=True)

res = pd.concat(rows, ignore_index=True)
res.to_csv(os.path.join(OUT, "ind_sad_results.csv"), index=False)


def star(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "   "


for term in ["fragmentation_index", "ai_exposure"]:
    print("\n" + "=" * 100)
    print(f"SA.D  term = {term}   (EFI Definition 1, control num_E1E2_tasks kept)")
    print("=" * 100)
    for model in ["noFE", "majorFE", "minorFE"]:
        print(f"\n--- {model} ---")
        print(f"{'prompt':>7} {'N':>5} | {'OLD coef':>10} {'(se)':>9} {'p':>10} | "
              f"{'MATCHED coef':>13} {'(se)':>9} {'p':>10} | {'MATCHED 95% CI':>22}")
        for x in PROMPTS:
            o = res[(res.prompt == x) & (res.leg == "OLD_E1") & (res.model == model) & (res.term == term)].iloc[0]
            m = res[(res.prompt == x) & (res.leg == "MATCHED_E1E2") & (res.model == model) & (res.term == term)].iloc[0]
            print(f"{x:>7} {int(o.nobs):>5} | {o.coef:>+10.4f}{star(o.p)} ({o.se:.3f}) {o.p:>10.2e} | "
                  f"{m.coef:>+13.4f}{star(m.p)} ({m.se:.3f}) {m.p:>10.2e} | "
                  f"[{m.ci_lo:+.3f}, {m.ci_hi:+.3f}]")

# summary counts
print("\n\nSUMMARY over 11 prompts x 3 FE specs = 33 cells")
for leg in ["OLD_E1", "MATCHED_E1E2"]:
    sub = res[(res.leg == leg) & (res.term == "fragmentation_index")]
    print(f"{leg:>14}  EFI: neg {int((sub.coef<0).sum())}/33, "
          f"p<.05 {int((sub.p<.05).sum())}/33, p<.10 {int((sub.p<.10).sum())}/33, "
          f"mean {sub.coef.mean():+.4f}, median|b| {sub.coef.abs().median():.4f}, "
          f"range [{sub.coef.min():+.4f}, {sub.coef.max():+.4f}]")
    sub = res[(res.leg == leg) & (res.term == "ai_exposure")]
    print(f"{'':>14}  EXP: pos {int((sub.coef>0).sum())}/33, "
          f"p<.05 {int((sub.p<.05).sum())}/33, mean {sub.coef.mean():+.4f}, "
          f"range [{sub.coef.min():+.4f}, {sub.coef.max():+.4f}]")
print("\nwrote", os.path.join(OUT, "ind_sad_results.csv"))
