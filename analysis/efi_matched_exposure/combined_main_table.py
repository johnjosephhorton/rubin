"""Emit the main-text fragmentation table: O*NET columns (1)-(3) beside APQC PCF columns (4)-(6).

The specification, identical in both halves:

  dependent variable  share of steps AI-executed
  exposure regressor  share of E1- or E2-labelled steps
  index               empirical fragmentation index built on the SAME E1|E2 label set
  control             number of AI-able steps
  fixed effects       none / SOC major / SOC minor   (O*NET)
                      none / PCF category / framework (APQC)

Two design points are load-bearing. Exposure and the index are measured on the same label set,
because the index decomposes exactly as EFI = 1 - k/m + r/m and a narrower exposure regressor
would leave the level term -k/m inside the fragmentation coefficient (see
writeup/EFI_MATCHED_SPECIFICATION.md section 1). And the count control is the AI-able count k: the index
mechanically depends on how many AI-able steps a workflow has, since r <= k, so holding k fixed
makes beta_2 a comparison among workflows carrying the same amount of AI-able material.
analysis/efi_exposure_grid shows the choice is not load-bearing: the fragmentation coefficient
moves by at most 0.04 across the two candidate controls and dropping the control entirely.

Estimation is delegated to analysis/efi_exposure_grid/efi_exposure_grid.py so this file cannot
drift from the grid that validates it. Unlike the other scripts in this folder, this one DOES
write into writeup/tables/, because the paper reads the table it emits (Table 3, Subsection 7.3).

    python analysis/efi_matched_exposure/combined_main_table.py
"""
import importlib.util
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
DEST = os.path.join(REPO, "writeup", "tables",
                    "fragmentation_index_regression_combined.tex")
GRID = os.path.join(REPO, "analysis", "efi_exposure_grid", "efi_exposure_grid.py")

CONTROL = "num_ai_able"
EXPOSURE = "E1E2"

star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def _load_grid():
    spec = importlib.util.spec_from_file_location("efi_exposure_grid", GRID)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["efi_exposure_grid"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cells(g, panel, fes, cluster_col):
    out = {k: [] for k in ("expo", "expo_se", "efi", "efi_se", "r2", "adj", "n")}
    for fe_term in fes:
        r = g.fit(panel, CONTROL, fe_term, cluster_col)
        out["expo"].append(f"{r['exp']:.2f}{star(r['exp_p'])}")
        out["expo_se"].append(f"({r['exp_se']:.2f})")
        out["efi"].append(f"{r['efi']:.2f}{star(r['efi_p'])}")
        out["efi_se"].append(f"({r['efi_se']:.2f})")
        out["r2"].append(f"{r['r2']:.2f}")
        out["adj"].append(f"{r['adj_r2']:.2f}")
        out["n"].append(f"{r['n']}")
    return out


def main():
    g = _load_grid()
    data = os.path.join(REPO, "data")
    cleaned = pd.read_csv(os.path.join(data, "computed_objects", "ONET_cleaned_tasks.csv"))
    soc = cleaned[[g.OCC, "Major_Group_Code", "Minor_Group_Code"]].drop_duplicates(subset=[g.OCC])

    onet = _cells(g, g.onet_panel(os.path.join(
        data, "computed_objects", "ONET_Eloundou_Anthropic_GPT",
        "ONET_Eloundou_Anthropic_GPT.csv"), soc, EXPOSURE),
        ["", " + C(Major_Group_Code)", " + C(Minor_Group_Code)"], g.OCC)
    apqc = _cells(g, g.pcf_panel(EXPOSURE),
                  ["", " + C(category)", " + C(framework)"], None)

    row = lambda label, a, b: f"{label} & " + " & ".join(a + b) + r" \\"

    lines = [
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{6}{c}{Share of AI-executed Steps} \\",
        r"\cmidrule(lr){2-7}",
        r" & \multicolumn{3}{c}{O*NET Occupations} & \multicolumn{3}{c}{APQC Process Groups} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r" & (1) & (2) & (3) & (4) & (5) & (6) \\",
        r"\midrule",
        r"\addlinespace",
        row("Share of AI-exposed Steps", onet["expo"], apqc["expo"]),
        row("", onet["expo_se"], apqc["expo_se"]),
        r"\addlinespace",
        row("Empirical Fragmentation Index", onet["efi"], apqc["efi"]),
        row("", onet["efi_se"], apqc["efi_se"]),
        r"\hline\\[-1.25em]",
        r"Fixed Effect & & SOC Major & SOC Minor & & PCF Category & Framework \\",
        r"Number of AI-able Steps Control & \checkmark & \checkmark & \checkmark "
        r"& \checkmark & \checkmark & \checkmark \\",
        row("R-squared", onet["r2"], apqc["r2"]),
        row("Adj. R-squared", onet["adj"], apqc["adj"]),
        row("Observations", onet["n"], apqc["n"]),
        r"\bottomrule",
        r"\end{tabular}",
    ]

    with open(DEST, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote", DEST)


if __name__ == "__main__":
    main()
