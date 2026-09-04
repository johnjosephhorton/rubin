"""Fragmentation regression across corpora, exposure definitions, control sets and fixed effects.

The grid, 54 estimated cells in all:

  corpus            O*NET main prompt | O*NET 10 alternative prompts | APQC PCF process groups
  exposure / EFI    E1 only | E1 or E2          (both the regressor and the index use the SAME
                                                 label set, so no level term leaks into beta_2)
  controls          + number of AI-able steps   (the draft's specification)
                    + number of steps           (workflow length instead)
                    none
  fixed effects     none | SOC major | SOC minor      (O*NET)
                    none | PCF category | framework   (APQC)

Unlabelled O*NET tasks are kept in place and coded not-AI-able throughout, because they are part
of the sequence the orderings were generated over; dropping them would change the adjacency
structure the index is computed on.

Writes a tidy CSV of every cell plus a summary figure.

    python analysis/efi_exposure_grid/efi_exposure_grid.py
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin"
NAME = os.path.basename(_HERE)
OUT = os.path.join(r"/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin/writeup/_e1e2_preview/work", "grid_out")
FIG = os.path.join(r"/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin/writeup/_e1e2_preview/work", "grid_fig")
for d in (OUT, FIG):
    os.makedirs(d, exist_ok=True)

DATA = os.path.join(REPO, "data")
OCC = "O*NET-SOC Code"
SIM_FLOOR, MIN_STEPS_PCF = 0.71, 5

CONTROLS = [("num_ai_able", "+ # AI-able steps"),
            ("num_steps",   "+ # steps in workflow"),
            (None,          "no controls")]
EXPOSURES = [("E1", "E1 only"), ("E1E2", "E1 or E2")]

star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def efi(seq):
    """One minus the share of adjacent AI-able pairs: a position counts unless it and its
    successor are both AI-able. Identical to the notebook's num_switches mean."""
    sw = np.ones(len(seq))
    sw[:-1][(seq[:-1] == 1) & (seq[1:] == 1)] = 0
    return sw.mean()


def onet_panel(path, soc, which):
    """Occupation panel from one ordering file. `which` selects the AI-able label set."""
    d = pd.read_csv(path)
    labels = ["E1"] if which == "E1" else ["E1", "E2"]
    d["is_ai"] = d["human_labels"].isin(labels).astype(int)
    d["exec"] = d["label"].isin(["Augmentation", "Automation"]).astype(int)
    rows = []
    for code, g in d.groupby(OCC, sort=False):
        s = g["is_ai"].to_numpy()
        rows.append({OCC: code, "num_steps": float(len(g)), "ai_fraction": g["exec"].mean(),
                     "ai_exposure": s.mean(), "num_ai_able": float(s.sum()),
                     "fragmentation_index": efi(s)})
    p = pd.DataFrame(rows)
    p = p[p.num_steps >= 3].merge(soc, on=OCC, how="left")
    for c in ("Major_Group_Code", "Minor_Group_Code", OCC):
        p[c] = p[c].astype(object)
    return p


def pcf_panel(which):
    L = pd.read_csv(os.path.join(DATA, "computed_objects", "apqc_pred3_industry",
                                 "industry_leaf_matches.csv"), dtype={"hid": str})
    L["sk"] = L["hid"].map(lambda h: tuple(int(x) for x in h.split(".")))
    L = L.sort_values(["uid", "sk"]).reset_index(drop=True)
    L["category"] = L["hid"].str.split(".").str[0]
    carried = L["similarity"] >= SIM_FLOOR
    labels = ["E1"] if which == "E1" else ["E1", "E2"]
    L["is_ai"] = (carried & L["human_labels"].isin(labels)).astype(int)
    L["exec"] = (carried & L["label"].isin(["Augmentation", "Automation"])).astype(int)
    L = L.groupby("uid").filter(lambda g: len(g) >= MIN_STEPS_PCF)
    rows = []
    for u, g in L.groupby("uid", sort=False):
        s = g["is_ai"].to_numpy()
        rows.append({"unit": u, "num_steps": float(len(g)), "ai_fraction": g["exec"].mean(),
                     "ai_exposure": s.mean(), "num_ai_able": float(s.sum()),
                     "fragmentation_index": efi(s),
                     "category": str(g["category"].iloc[0]),
                     "framework": str(g["framework"].iloc[0])})
    p = pd.DataFrame(rows)
    for c in ("category", "framework"):
        p[c] = p[c].astype(object)
    return p


def fit(panel, control, fe_term, cluster_col):
    """Standardize within the estimation sample, then OLS. Clustered SEs on O*NET (as in the
    paper), heteroskedasticity-robust on APQC."""
    d = panel.copy()
    cols = ["ai_fraction", "ai_exposure", "fragmentation_index"] + ([control] if control else [])
    for c in cols:
        d[c] = (d[c] - d[c].mean()) / d[c].std()
    rhs = "fragmentation_index + ai_exposure" + (f" + {control}" if control else "") + fe_term
    if cluster_col:
        m = smf.ols(f"ai_fraction ~ {rhs}", d).fit(
            cov_type="cluster", cov_kwds={"groups": d[cluster_col],
                                          "use_correction": True, "df_correction": True})
    else:
        m = smf.ols(f"ai_fraction ~ {rhs}", d).fit(cov_type="HC1")
    return {"efi": m.params["fragmentation_index"], "efi_se": m.bse["fragmentation_index"],
            "efi_p": m.pvalues["fragmentation_index"],
            "efi_lo": m.conf_int().loc["fragmentation_index", 0],
            "efi_hi": m.conf_int().loc["fragmentation_index", 1],
            "exp": m.params["ai_exposure"], "exp_se": m.bse["ai_exposure"],
            "exp_p": m.pvalues["ai_exposure"], "n": int(m.nobs), "r2": float(m.rsquared), "adj_r2": float(m.rsquared_adj)}


def main():
    onet_dir = os.path.join(DATA, "computed_objects", "ONET_Eloundou_Anthropic_GPT")
    cleaned = pd.read_csv(os.path.join(DATA, "computed_objects", "ONET_cleaned_tasks.csv"))
    soc = cleaned[[OCC, "Major_Group_Code", "Minor_Group_Code"]].drop_duplicates(subset=[OCC])

    ONET_FE = [("", "no FE"), (" + C(Major_Group_Code)", "SOC major"),
               (" + C(Minor_Group_Code)", "SOC minor")]
    PCF_FE = [("", "no FE"), (" + C(category)", "PCF category"), (" + C(framework)", "framework")]

    recs = []
    for which, wlab in EXPOSURES:
        for prompt in range(11):
            suffix = "" if prompt == 0 else f"_{prompt}"
            panel = onet_panel(os.path.join(onet_dir, f"ONET_Eloundou_Anthropic_GPT{suffix}.csv"),
                               soc, which)
            corpus = "O*NET main prompt" if prompt == 0 else "O*NET alt prompts"
            for control, clab in CONTROLS:
                for fe_term, felab in ONET_FE:
                    recs.append(dict(corpus=corpus, prompt=prompt, exposure=wlab,
                                     controls=clab, fe=felab,
                                     **fit(panel, control, fe_term, OCC)))
            print(f"  O*NET {wlab:<9} prompt {prompt:>2} done", flush=True)
        panel = pcf_panel(which)
        for control, clab in CONTROLS:
            for fe_term, felab in PCF_FE:
                recs.append(dict(corpus="APQC PCF", prompt=-1, exposure=wlab,
                                 controls=clab, fe=felab,
                                 **fit(panel, control, fe_term, None)))
        print(f"  APQC  {wlab:<9} done", flush=True)

    R = pd.DataFrame(recs)
    R.to_csv(os.path.join(OUT, "efi_exposure_grid.csv"), index=False)
    print("\nwrote", os.path.join(OUT, "efi_exposure_grid.csv"))
    return R


if __name__ == "__main__":
    main()
