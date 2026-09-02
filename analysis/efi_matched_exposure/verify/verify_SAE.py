"""
INDEPENDENT SPOT-CHECK, SA.E (Robustness to Frequently-Executed Tasks), Prediction #2 leg.

Written from scratch by the verification agent. The repo is read only; output goes to data/computed_objects/efi_matched_exposure/. No published exhibit is touched.

PRUNING RULE, derived by reading analysis/onet_fragmentationIndex_weeklyTasks.ipynb cell 17:

  FAMILIES (cumulative O*NET Frequency-of-Task tails):
    Daily+        -> ['FT_Daily', 'FT_Several times daily', 'FT_Hourly or more']  (FT 5-7)
    SeveralDaily+ -> ['FT_Several times daily', 'FT_Hourly or more']              (FT 6-7)
    Hourly+       -> ['FT_Hourly or more']                                        (FT 7)
  SWEEP_THRESHOLDS = [20, 35, 50, 65]
  CUTS = [('All tasks', no filter)] + every (family, threshold) pair  -> 13 cuts

  prepare_cut(family_cols, threshold):
     1. read the merged workflow file
     2. if family_cols is not None:
            df = df[df[family_cols].sum(axis=1) >= threshold].reset_index(drop=True)
        i.e. the threshold is applied to the SUM of that family's FT percentage columns,
        task by task; failing rows are DROPPED and survivors keep their original CSV
        row order, so relative sequence order is preserved and nothing is re-sorted.
     3. restrict to valid_occ_set(family_cols, threshold): occupations whose PRUNED
        workflow retains >= MIN_TASKS_PER_OCC (=5) distinct Task IDs, counted on the
        full pruned workflow (not the DWA-eligible pool).
  NOTE: neighbor_occ_set() is defined in that cell but is NOT called by prepare_cut.

  Then EFI Definition 1, exposure share, ai_fraction and num_E1E2_tasks are ALL
  recomputed on the pruned rows (total_tasks = len(group) = pruned row count), all four
  are z-scored WITHIN the cut, and the regression is
     ai_fraction ~ fragmentation_index + ai_exposure + num_E1E2_tasks [+ C(Major)|C(Minor)]
  with cluster-robust SE on O*NET-SOC Code (use_correction, df_correction).
  A cell is left NaN when n_occ < 10 or EFI has no variation.

  Published exposure regressor: ai_exposure_var = 'human_E1_fraction'.
  Matched  exposure regressor: 'human_aiExposure_fraction'.
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


IN_FILE = os.path.join(REPO, "data", "computed_objects", "ONET_Eloundou_Anthropic_GPT",
                       "ONET_Eloundou_Anthropic_GPT.csv")
CLEANED = os.path.join(REPO, "data", "computed_objects", "ONET_cleaned_tasks.csv")

SOC, TITLE = "O*NET-SOC Code", "Occupation Title"
MIN_TASKS_PER_OCC = 5

FAMILIES = [
    ("Daily+", "daily", ["FT_Daily", "FT_Several times daily", "FT_Hourly or more"]),
    ("SeveralDaily+", "sevdaily", ["FT_Several times daily", "FT_Hourly or more"]),
    ("Hourly+", "hourly", ["FT_Hourly or more"]),
]
THRESHOLDS = [20, 35, 50, 65]
CUTS = [("All tasks", "all", None, None)]
for lab, tag, cols in FAMILIES:
    for t in THRESHOLDS:
        CUTS.append((f"{lab} >={t}%", tag, cols, t))

FTCOLS = ["FT_Daily", "FT_Several times daily", "FT_Hourly or more"]
USECOLS = [SOC, TITLE, "Task ID", "Task Position", "human_labels", "label"] + FTCOLS

RAW = pd.read_csv(IN_FILE, usecols=USECOLS)
CLEAN = pd.read_csv(CLEANED, usecols=[SOC, TITLE, "Major_Group_Code", "Minor_Group_Code"])
SOC_MAP = CLEAN.drop_duplicates(subset=[SOC, TITLE])


def prepare_cut(family_cols, threshold):
    df = RAW
    if family_cols is not None:
        df = df[df[family_cols].sum(axis=1) >= threshold].reset_index(drop=True)
    cnt = df.groupby(SOC)["Task ID"].nunique()
    valid = set(cnt[cnt >= MIN_TASKS_PER_OCC].index)
    df = df[df[SOC].isin(valid)].reset_index(drop=True)
    return df


def build_panel(d):
    d = d.copy()
    d["is_ai"] = d["human_labels"].isin(["E1", "E2"]).astype(int)
    d["next_is_ai"] = d.groupby([SOC, TITLE])["is_ai"].shift(-1).fillna(0).astype(int)
    d["num_switches"] = 1
    d.loc[(d["is_ai"] == 1) & (d["next_is_ai"] == 1), "num_switches"] = 0
    g = d.groupby([SOC, TITLE], sort=True)
    panel = pd.DataFrame({
        "fragmentation_index": g["num_switches"].mean(),
        "m_rows": g.size(),
        "num_E1E2_tasks": g["is_ai"].sum().astype(float),
        "ai_fraction": g["label"].apply(lambda s: s.isin(["Augmentation", "Automation"]).sum() / len(s)),
        "human_E1_fraction": g["human_labels"].apply(lambda s: (s == "E1").sum() / len(s)),
        "human_aiExposure_fraction": g["human_labels"].apply(lambda s: s.isin(["E1", "E2"]).sum() / len(s)),
    }).reset_index()
    panel = panel.merge(SOC_MAP, on=[SOC, TITLE], how="left")
    return panel


def run_cut(panel, exposure_col):
    p = panel.copy()
    p["ai_exposure"] = p[exposure_col]
    for c in ("Major_Group_Code", "Minor_Group_Code", SOC):
        p[c] = p[c].astype("object")
    for c in ["ai_fraction", "ai_exposure", "fragmentation_index", "num_E1E2_tasks"]:
        s = p[c].astype(float)
        sd = s.std()
        p[c] = (s - s.mean()) / sd if (sd and not np.isnan(sd)) else np.nan
    n_occ = len(p)
    clu = dict(cov_type="cluster", cov_kwds={"groups": p[SOC], "use_correction": True,
                                             "df_correction": True})
    out = {}
    for fe_name, fe_term in [("none", ""), ("Major", " + C(Major_Group_Code)"),
                             ("Minor", " + C(Minor_Group_Code)")]:
        try:
            if n_occ < 10 or p["fragmentation_index"].nunique() < 2:
                raise ValueError("too small")
            m = smf.ols(f"ai_fraction ~ fragmentation_index + ai_exposure + num_E1E2_tasks{fe_term}",
                        data=p).fit(**clu)
            ci = m.conf_int()
            out[fe_name] = dict(
                coef=m.params["fragmentation_index"], se=m.bse["fragmentation_index"],
                pval=m.pvalues["fragmentation_index"],
                ci_lo=ci.loc["fragmentation_index", 0], ci_hi=ci.loc["fragmentation_index", 1],
                exp_coef=m.params["ai_exposure"], exp_se=m.bse["ai_exposure"],
                exp_p=m.pvalues["ai_exposure"], N_occ=n_occ, r2=float(m.rsquared))
        except Exception as e:
            out[fe_name] = dict(coef=np.nan, se=np.nan, pval=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                                exp_coef=np.nan, exp_se=np.nan, exp_p=np.nan, N_occ=n_occ,
                                r2=np.nan, err=repr(e))
    return out


rows = []
for label, fam, cols, thr in CUTS:
    d = prepare_cut(cols, thr)
    panel = build_panel(d)
    for leg, col in [("OLD_E1", "human_E1_fraction"),
                     ("MATCHED_E1E2", "human_aiExposure_fraction")]:
        res = run_cut(panel, col)
        for fe, r in res.items():
            rows.append({"cut": label, "family": fam, "threshold": thr if thr is not None else 0,
                         "FE": fe, "leg": leg, **{k: v for k, v in r.items() if k != "err"}})
    print(f"{label:<18} rows={len(d):>6} occ={len(panel):>4} "
          f"meanEFI={panel['fragmentation_index'].mean():.4f} "
          f"meanE1E2share={panel['human_aiExposure_fraction'].mean():.4f}", flush=True)

sw = pd.DataFrame(rows)
sw.to_csv(os.path.join(OUT, "ind_sae_results.csv"), index=False)


def star(p):
    if pd.isna(p):
        return "   "
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "   "


order = [c[0] for c in CUTS]
for fe, nm in [("none", "No fixed effects"), ("Major", "SOC major-group FE"),
               ("Minor", "SOC minor-group FE")]:
    print("\n" + "=" * 108)
    print(f"SA.E  EFI coefficient  ===  {nm}")
    print("=" * 108)
    print(f"{'cut':<18} {'N':>5} | {'OLD coef':>11} {'(se)':>8} {'p':>9} | "
          f"{'MATCHED coef':>13} {'(se)':>8} {'p':>9} | {'MATCHED 95% CI':>20}")
    for c in order:
        o = sw[(sw.cut == c) & (sw.FE == fe) & (sw.leg == "OLD_E1")].iloc[0]
        m = sw[(sw.cut == c) & (sw.FE == fe) & (sw.leg == "MATCHED_E1E2")].iloc[0]
        print(f"{c:<18} {int(o.N_occ):>5} | {o.coef:>+11.4f}{star(o.pval)} ({o.se:.3f}) {o.pval:>9.4f} | "
              f"{m.coef:>+13.4f}{star(m.pval)} ({m.se:.3f}) {m.pval:>9.4f} | "
              f"[{m.ci_lo:+.3f}, {m.ci_hi:+.3f}]")

for fe, nm in [("none", "No fixed effects"), ("Major", "SOC major-group FE"),
               ("Minor", "SOC minor-group FE")]:
    print("\n" + "-" * 90)
    print(f"SA.E  EXPOSURE coefficient  ===  {nm}")
    print("-" * 90)
    for c in order:
        o = sw[(sw.cut == c) & (sw.FE == fe) & (sw.leg == "OLD_E1")].iloc[0]
        m = sw[(sw.cut == c) & (sw.FE == fe) & (sw.leg == "MATCHED_E1E2")].iloc[0]
        print(f"{c:<18} {int(o.N_occ):>5} | OLD {o.exp_coef:>+8.4f}{star(o.exp_p)} ({o.exp_se:.3f}) "
              f"| MATCHED {m.exp_coef:>+8.4f}{star(m.exp_p)} ({m.exp_se:.3f})")

print("\n\nSIGN / SIGNIFICANCE TALLIES (39 cells = 13 cuts x 3 FE)")
for leg in ["OLD_E1", "MATCHED_E1E2"]:
    s = sw[sw.leg == leg]
    pruned = s[s.family != "all"]
    print(f"{leg:>14} EFI: neg {int((s.coef<0).sum())}/39 (pruned {int((pruned.coef<0).sum())}/36), "
          f"p<.05 {int((s.pval<.05).sum())}/39, p<.10 {int((s.pval<.10).sum())}/39, "
          f"median|b| {s.coef.abs().median():.4f}")
    print(f"{'':>14} EXP: pos {int((s.exp_coef>0).sum())}/39, p<.05 {int((s.exp_p<.05).sum())}/39")
print("\nwrote", os.path.join(OUT, "ind_sae_results.csv"))
