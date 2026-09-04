"""Figure OA.A.1 panels (a)-(d): AME_full_is_ai_{spec}.png, published vs E1|E2.

Plotting code transcribed VERBATIM from cell 20 of
analysis/onet_antrhopicIndex_execTypeVaryingDWA.ipynb (plot_comparison_hist, the
"Save Individual Rows" branch that writes AME_full_is_ai_<spec>.png), together with
cell 19's assembly of resh_full / obs_dict_full / obs_se_full.

Two modes:
  published   observed AMEs and the 1,000 placebo draws read from the repo's own cached
              regression_ame_results_full_{i}.csv, i = 0..999 -- exactly what cell 19 does.
              Written to work/t2f1/repro_current/ and diffed against the committed PNGs
              as a control on the plotting code.
  e1e2        same code, same seeds, same draw indices; observed AMEs from the E1|E2
              rerun and placebo draws from the E1|E2 reshuffle (i = 1..999), with i = 0
              taken from the E1|E2 observed run exactly as the notebook takes i = 0 from
              the E1-only observed run (cell 19 reads full_0.csv, which cell 18 wrote
              from the UNSHUFFLED data -- so one of the notebook's 1,000 "draws" is the
              observed estimate; that quirk is reproduced rather than silently fixed).

Usage: python3 p2_figs.py <published|e1e2> <out_dir>
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

REPO = "/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin"
STAGE = f"{REPO}/writeup/_e1e2_preview"
WORK = f"{STAGE}/work/t2f1"
CACHE = (f"{REPO}/data/computed_objects/execTypeVaryingDWA_anthropicIndex"
         f"_noTasksWithRepetitiveDWAs/regression_summaries_is_ai")

dependent_var = "is_ai"
n_shuffles = 1000
TARGET_REGS = ["prev2_is_ai", "prev_is_ai", "next_is_ai", "next2_is_ai"]
SPECS = ["no_fe_no_dwa", "major_fe_no_dwa", "minor_fe_no_dwa", "no_fe_with_dwa",
         "no_fe_no_dwa_withTaskDWACount", "no_fe_with_dwa_withTaskDWACount"]
PLOT_TITLES = ["Task Before Previous Task", "Previous Task", "Next Task",
               "Task After Next Task"]
VAR_LABELS = {
    "prev2_is_ai": "Task ($k-2$) is AI",
    "prev_is_ai": "Task ($k-1$) is AI",
    "next_is_ai": "Task ($k+1$) is AI",
    "next2_is_ai": "Task ($k+2$) is AI",
}


# ---- cell 19, helper 1: results_to_dict (verbatim) -----------------------------------
def results_to_dict(df_results):
    coef_out = {spec: {term: np.nan for term in TARGET_REGS} for spec in SPECS}
    se_out = {spec: {term: np.nan for term in TARGET_REGS} for spec in SPECS}
    if df_results is None or (hasattr(df_results, "empty") and df_results.empty):
        return coef_out, se_out
    for _, row in df_results.iterrows():
        spec = row.get("model")
        term = row.get("term")
        if spec in coef_out and term in coef_out[spec]:
            if "ame_coef" in row and pd.notna(row["ame_coef"]):
                coef_out[spec][term] = row["ame_coef"]
            if "ame_se" in row and pd.notna(row["ame_se"]):
                se_out[spec][term] = row["ame_se"]
    return coef_out, se_out


# ---- cell 20: plot_comparison_hist, individual-row branch (verbatim) ------------------
def plot_comparison_hist(resh_dict, obs_dict, obs_se_dict, out_name, out_dir, bins=30):
    all_resh_vals = [v for inner in resh_dict.values() for lst in inner.values()
                     for v in lst if not np.isnan(v)]
    all_obs_vals = [v for inner in obs_dict.values() for v in inner.values()
                    if not np.isnan(v)]
    total_vals = all_resh_vals + all_obs_vals
    if not total_vals:
        raise RuntimeError("No valid data found to plot.")

    g_min, g_max = min(total_vals), max(total_vals)
    symmetric_bound = max(abs(g_min), abs(g_max))
    span = 2 * symmetric_bound
    if span == 0:
        span = 0.1
    x_limit_min = -symmetric_bound - (span * 0.01)
    x_limit_max = symmetric_bound + (span * 0.01)

    colors = [plt.cm.tab10(i % 10) for i in range(len(SPECS))]

    base_name = out_name.rsplit(".", 1)[0]
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for r, spec in enumerate(SPECS):
        fig_row, axs_row = plt.subplots(nrows=1, ncols=len(TARGET_REGS), figsize=(24, 5),
                                        sharey=False)
        if len(TARGET_REGS) == 1:
            axs_row = [axs_row]
        color_row = colors[r]
        for c, term in enumerate(TARGET_REGS):
            axr = axs_row[c]
            vals = np.array(resh_dict[spec][term], dtype=float)
            vals_clean = vals[~np.isnan(vals)]

            if len(vals_clean):
                axr.hist(vals_clean, bins=bins, color=color_row, alpha=0.7, edgecolor="k",
                         label="Task Position Reshuffled AMEs", zorder=2)
            else:
                axr.text(0.5, 0.5, "no estimates", ha="center", va="center")

            obs_val = obs_dict.get(spec, {}).get(term, np.nan)
            if not np.isnan(obs_val):
                obs_se = obs_se_dict.get(spec, {}).get(term, np.nan)
                if not np.isnan(obs_se):
                    se_band = 1.645 * obs_se
                    axr.axvspan(obs_val - se_band, obs_val + se_band, color="red",
                                alpha=0.08, zorder=1)
                    axr.axvline(obs_val - se_band, color="red", linestyle="--",
                                linewidth=1, alpha=0.9, zorder=3)
                    axr.axvline(obs_val + se_band, color="red", linestyle="--",
                                linewidth=1, alpha=0.9, zorder=3)
                axr.axvline(obs_val, color="red", linestyle="--", linewidth=3,
                            label=f"Observed = {obs_val:.3f}", zorder=4)

            axr.axvline(0.0, color="black", linestyle="-", linewidth=1.5, alpha=0.5,
                        zorder=4)
            clean_title = VAR_LABELS.get(term, term)
            axr.set_title(clean_title, fontsize=30)
            if c == 0:
                axr.set_ylabel("Count", fontsize=24)
            axr.grid(axis="y", linestyle=":", alpha=0.5)
            axr.set_xlim(x_limit_min, x_limit_max)
            axr.xaxis.set_major_locator(MultipleLocator(0.05))
            axr.set_xlabel("Average Marginal Effect", fontsize=28)
            axr.legend(loc="best", fontsize=14)
            axr.tick_params(labelsize=16)

        fig_row.tight_layout()
        out_path_row = f"{out_dir}/{base_name}_{spec}.png"
        fig_row.savefig(out_path_row, dpi=300, bbox_inches="tight")
        plt.close(fig_row)
        written.append(out_path_row)
        print("Saved row plot to", out_path_row, flush=True)
    return written, (x_limit_min, x_limit_max)


# ---- cell 19: assemble resh_full / obs_dict_full / obs_se_full ------------------------
def load_published():
    res_full = pd.read_csv(f"{CACHE}/regression_ame_results_full_0.csv")
    obs, obs_se = results_to_dict(res_full)
    resh = {spec: {t: [] for t in TARGET_REGS} for spec in SPECS}
    for i in range(n_shuffles):
        d = pd.read_csv(f"{CACHE}/regression_ame_results_full_{i}.csv")
        di, _ = results_to_dict(d)
        for spec in SPECS:
            for t in TARGET_REGS:
                resh[spec][t].append(di[spec][t])
    return resh, obs, obs_se


def load_e1e2():
    res_full = pd.read_csv(f"{WORK}/ame_full_0_E1E2.csv")
    obs, obs_se = results_to_dict(res_full)
    draws = pd.read_csv(f"{WORK}/resh_E1E2_all.csv")
    by_draw = {i: g for i, g in draws.groupby("draw")}
    resh = {spec: {t: [] for t in TARGET_REGS} for spec in SPECS}
    for i in range(n_shuffles):
        d = res_full if i == 0 else by_draw[i]      # cell 19 reads full_0.csv at i = 0
        di, _ = results_to_dict(d)
        for spec in SPECS:
            for t in TARGET_REGS:
                resh[spec][t].append(di[spec][t])
    missing = [i for i in range(1, n_shuffles) if i not in by_draw]
    if missing:
        raise RuntimeError(f"missing placebo draws: {missing[:20]} ({len(missing)} total)")
    return resh, obs, obs_se


def main():
    mode, out_dir = sys.argv[1], sys.argv[2]
    resh, obs, obs_se = load_published() if mode == "published" else load_e1e2()
    written, xlim = plot_comparison_hist(resh, obs, obs_se,
                                         f"AME_full_{dependent_var}.png", out_dir)
    print(f"[{mode}] xlim = ({xlim[0]:.6f}, {xlim[1]:.6f})")
    # dump the numbers behind the labels
    rows = []
    for spec in SPECS:
        for t in TARGET_REGS:
            v = np.array(resh[spec][t], dtype=float)
            v = v[~np.isnan(v)]
            rows.append({"mode": mode, "spec": spec, "term": t,
                         "observed": obs[spec][t], "observed_se": obs_se[spec][t],
                         "label": f"Observed = {obs[spec][t]:.3f}",
                         "n_draws": len(v), "null_mean": v.mean(), "null_sd": v.std(ddof=1),
                         "null_min": v.min(), "null_max": v.max(),
                         "share_null_ge_obs": float((v >= obs[spec][t]).mean())})
    pd.DataFrame(rows).to_csv(f"{WORK}/figOAA1_stats_{mode}.csv", index=False)
    print("wrote", f"{WORK}/figOAA1_stats_{mode}.csv")


if __name__ == "__main__":
    main()
