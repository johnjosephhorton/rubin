"""Sensitivity of the PCF results to the cosine-similarity floor used to carry labels over.

AI exposure and execution are measured on O*NET tasks and have no counterpart on PCF elements,
so each PCF step is matched to its nearest O*NET task by embedding similarity and a label is
carried across only when the match clears a floor. The paper sets that floor at 0.71. This script
sweeps it and reports what moves.

The 525 process groups are FIXED across thresholds: the five-step minimum is applied to the raw
step count, not to the labels, so raising the floor only relabels steps as unexposed and
unexecuted. That makes the sweep a clean sensitivity check on the labelling rather than on the
sample.

For each threshold it reports
  the density of the resulting labels,
  the average AI chain length with its within-group step-order reshuffle null (Prediction #1),
  the fragmentation coefficient under all three fixed-effect specifications (Prediction #3),
matching the specifications in the draft: exposure and the index both built on E1|E2, and the
number of AI-able steps as the count control.

    python analysis/apqc_similarity_threshold/apqc_similarity_threshold.py
"""
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
NAME = os.path.basename(_HERE)
OUT = os.path.join(REPO, "data", "computed_objects", NAME)
os.makedirs(OUT, exist_ok=True)

POOLED = os.path.join(REPO, "data", "computed_objects", "apqc_pred3_industry",
                      "industry_leaf_matches.csv")
THRESHOLDS = [round(0.65 + 0.01 * i, 2) for i in range(11)]   # 0.65 to 0.75
MIN_STEPS, N_DRAWS, SEED = 5, 1000, 20240902
CHOSEN = 0.71                                                 # the floor used in the paper
star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def efi(seq):
    """One minus the share of adjacent AI-able pairs."""
    sw = np.ones(len(seq))
    sw[:-1][(seq[:-1] == 1) & (seq[1:] == 1)] = 0
    return sw.mean()


def run_lengths(v):
    out, i = [], 0
    while i < len(v):
        if v[i] == 1:
            j = i
            while j + 1 < len(v) and v[j + 1] == 1:
                j += 1
            out.append(j - i + 1)
            i = j + 1
        else:
            i += 1
    return out


def load():
    L = pd.read_csv(POOLED, dtype={"hid": str})
    L["sk"] = L["hid"].map(lambda h: tuple(int(x) for x in h.split(".")))
    L = L.sort_values(["uid", "sk"]).reset_index(drop=True)
    L["category"] = L["hid"].str.split(".").str[0]
    return L.groupby("uid").filter(lambda g: len(g) >= MIN_STEPS)


def one_threshold(L, floor, rng):
    carried = L["similarity"] >= floor
    exposed = (carried & L["human_labels"].isin(["E1", "E2"])).astype(int).to_numpy()
    execed = (carried & L["label"].isin(["Augmentation", "Automation"])).astype(int).to_numpy()
    uid = L["uid"].to_numpy()
    _, starts = np.unique(uid, return_index=True)
    bounds = np.sort(starts)
    seq_exp = np.split(exposed, bounds[1:])
    seq_exe = np.split(execed, bounds[1:])
    order = uid[bounds]

    # ---- Prediction #1: mean AI chain length against a within-group order reshuffle
    obs = np.mean([l for s in seq_exe for l in run_lengths(s)]) if execed.sum() else np.nan
    null = np.empty(N_DRAWS)
    for b in range(N_DRAWS):
        lens = [l for s in seq_exe for l in run_lengths(rng.permutation(s))]
        null[b] = np.mean(lens) if lens else np.nan
    z = (obs - null.mean()) / null.std(ddof=1)

    # ---- Prediction #3: the fragmentation regression on the same panel
    meta = L.groupby("uid", sort=False).agg(category=("category", "first"),
                                            framework=("framework", "first"))
    P = pd.DataFrame({
        "unit": order,
        "ai_fraction": [s.mean() for s in seq_exe],
        "ai_exposure": [s.mean() for s in seq_exp],
        "num_ai_able": [float(s.sum()) for s in seq_exp],
        "fragmentation_index": [efi(s) for s in seq_exp],
    }).merge(meta.reset_index(), left_on="unit", right_on="uid")
    for c in ("category", "framework"):
        P[c] = P[c].astype(object)
    Z = P.copy()
    for c in ["ai_fraction", "ai_exposure", "fragmentation_index", "num_ai_able"]:
        Z[c] = (Z[c] - Z[c].mean()) / Z[c].std()

    row = dict(threshold=floor, groups=P.unit.nunique(), steps=len(L),
               cleared=int(carried.sum()), cleared_share=float(carried.mean()),
               exposed_share=float(exposed.mean()), executed_share=float(execed.mean()),
               chain=obs, chain_null=null.mean(), chain_null_sd=null.std(ddof=1), chain_z=z)
    for fe, tag in [("", "noFE"), (" + C(category)", "cat"), (" + C(framework)", "fw")]:
        m = smf.ols("ai_fraction ~ fragmentation_index + ai_exposure + num_ai_able" + fe,
                    Z).fit(cov_type="HC1")
        row[f"efi_{tag}"] = m.params["fragmentation_index"]
        row[f"efi_se_{tag}"] = m.bse["fragmentation_index"]
        row[f"efi_p_{tag}"] = m.pvalues["fragmentation_index"]
        row[f"exp_{tag}"] = m.params["ai_exposure"]
        row[f"exp_p_{tag}"] = m.pvalues["ai_exposure"]
    return row


def main():
    L = load()
    rng = np.random.default_rng(SEED)
    rows = [one_threshold(L, f, rng) for f in THRESHOLDS]
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "apqc_similarity_threshold.csv"), index=False)

    print("=" * 118)
    print("  PCF SAMPLE AS THE SIMILARITY FLOOR MOVES        (525 process groups, 13,482 steps, "
          "fixed across thresholds)")
    print("=" * 118)
    print(f"  {'floor':>7}{'labels carried':>16}{'% exposed':>11}{'% executed':>12}"
          f"{'chain':>8}{'null':>8}{'z':>8}")
    print("  " + "-" * 114)
    for r in rows:
        mark = "  <-- chosen" if abs(r["threshold"] - CHOSEN) < 1e-9 else ""
        print(f"  {r['threshold']:>7.2f}{r['cleared']:>10,} ({r['cleared_share']*100:>4.1f}%)"
              f"{r['exposed_share']*100:>10.1f}%{r['executed_share']*100:>11.1f}%"
              f"{r['chain']:>8.2f}{r['chain_null']:>8.2f}{r['chain_z']:>8.1f}{mark}")

    print()
    print("=" * 118)
    print("  FRAGMENTATION COEFFICIENT (standardized, HC1), by fixed effects")
    print("=" * 118)
    print(f"  {'floor':>7}{'no FE':>22}{'PCF category':>22}{'framework':>22}{'exposure, no FE':>20}")
    print("  " + "-" * 114)
    for r in rows:
        cells = "".join(f"{r[f'efi_{t}']:>+14.3f}{star(r[f'efi_p_{t}']):<8}"
                        for t in ("noFE", "cat", "fw"))
        mark = "  <-- chosen" if abs(r["threshold"] - CHOSEN) < 1e-9 else ""
        print(f"  {r['threshold']:>7.2f}{cells}{r['exp_noFE']:>+14.3f}"
              f"{star(r['exp_p_noFE']):<6}{mark}")
    print("\nwrote", os.path.join(OUT, "apqc_similarity_threshold.csv"))
    return R


if __name__ == "__main__":
    main()
