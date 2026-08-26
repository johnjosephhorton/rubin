"""Regenerate the neighbor-AI average-marginal-effect figures from cached results.

Why this exists
---------------
The AME figures are produced by `onet_antrhopicIndex_execTypeVaryingDWA.ipynb`,
but that notebook cannot execute under the pandas currently installed (3.x): it
uses `groupby(...).apply(f)` expecting the grouping column to be passed into `f`,
which pandas stopped doing. Patching the notebook to run again would touch the
analysis path and risk moving published numbers.

Nothing here recomputes anything. Every input is read from the regression
summaries the notebook already wrote to
`data/computed_objects/execTypeVaryingDWA_anthropicIndex*/regression_summaries_*/`,
which are exactly the values cell 20 of the notebook plots. Only the drawing is
redone, so the figures cannot change except in appearance.

The plotting below mirrors the "Save Individual Rows" block of that cell. If the
notebook's plotting changes, this file has to change with it.

Usage
-----
    python3 replot_AME_figures.py            # regenerate with current styling
    python3 replot_AME_figures.py --legacy   # regenerate with the pre-2026-08 styling
                                             # (for verifying this script reproduces
                                             #  the committed figures)
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

# --- configuration mirrored from the notebook -------------------------------

N_SHUFFLES = 1000
TARGET_REGS = ["prev2_is_ai", "prev_is_ai", "next_is_ai", "next2_is_ai"]
SPECS = [
    "no_fe_no_dwa",
    "major_fe_no_dwa",
    "minor_fe_no_dwa",
    "no_fe_with_dwa",
    "no_fe_no_dwa_withTaskDWACount",
    "no_fe_with_dwa_withTaskDWACount",
]
VAR_LABELS = {
    "prev2_is_ai": "Task ($k-2$) is AI",
    "prev_is_ai": "Task ($k-1$) is AI",
    "next_is_ai": "Task ($k+1$) is AI",
    "next2_is_ai": "Task ($k+2$) is AI",
}
LEGACY_VAR_LABELS = {
    "prev2_is_ai": "($t-2$) Task AI",
    "prev_is_ai": "($t-1$) Task AI",
    "next_is_ai": "($t+1$) Task AI",
    "next2_is_ai": "($t+2$) Task AI",
}

# Only font sizes differ from the legacy styling; figure dimensions, axis ranges,
# and legend placement are left exactly as the notebook produced them.
STYLE = dict(title=30, title_weight="normal", xlabel=28, ylabel=24, ticks=16, legend=14)
LEGACY_STYLE = dict(title=15, title_weight="normal", xlabel=15, ylabel=15, ticks=10, legend=10)

BINS = 30
# Ticks every 0.05 give seven labels per panel; 16pt is the largest size at
# which the three negative ones still clear each other.
XTICK_STEP = 0.05
REPO = Path(__file__).resolve().parent.parent

# (dependent_var, path_suffix) pairs the paper draws on
CONFIGS = [
    ("is_ai", "_noTasksWithRepetitiveDWAs"),
    ("is_automated", ""),
]


def results_to_dict(df_results):
    """Mirror of the notebook helper: (coef_dict, se_dict) keyed [spec][term]."""
    coef = {s: {t: np.nan for t in TARGET_REGS} for s in SPECS}
    se = {s: {t: np.nan for t in TARGET_REGS} for s in SPECS}
    if df_results is None or df_results.empty:
        return coef, se
    for _, row in df_results.iterrows():
        spec, term = row.get("model"), row.get("term")
        if spec in coef and term in coef[spec]:
            if pd.notna(row.get("ame_coef")):
                coef[spec][term] = row["ame_coef"]
            if pd.notna(row.get("ame_se")):
                se[spec][term] = row["ame_se"]
    return coef, se


def load_null(summary_dir, kind):
    """Collect the reshuffled AMEs for `kind` in {'full','filtered'}.

    The notebook's loop starts at i=0, and file 0 is the observed fit, so it
    enters the null as one draw. Reproduced here so the figures match.
    """
    resh = {s: {t: [] for t in TARGET_REGS} for s in SPECS}
    missing = 0
    for i in range(N_SHUFFLES):
        f = summary_dir / f"regression_ame_results_{kind}_{i}.csv"
        if not f.exists():
            missing += 1
            continue
        coef, _ = results_to_dict(pd.read_csv(f))
        for s in SPECS:
            for t in TARGET_REGS:
                resh[s][t].append(coef[s][t])
    return resh, missing


def x_bounds(resh, obs):
    vals = [v for d in resh.values() for lst in d.values() for v in lst if not np.isnan(v)]
    vals += [v for d in obs.values() for v in d.values() if not np.isnan(v)]
    if not vals:
        return -0.1, 0.1
    bound = max(abs(min(vals)), abs(max(vals)))
    span = 2 * bound or 0.1
    return -bound - span * 0.01, bound + span * 0.01


def draw(resh, obs, obs_se, out_dir, base_name, style, labels):
    lo_x, hi_x = x_bounds(resh, obs)
    colors = [plt.cm.tab10(i % 10) for i in range(len(SPECS))]
    written = []
    for r, spec in enumerate(SPECS):
        fig, axs = plt.subplots(nrows=1, ncols=len(TARGET_REGS), figsize=(24, 5), sharey=False)
        color_row = colors[r]
        for c, term in enumerate(TARGET_REGS):
            ax = axs[c]
            vals = np.array(resh[spec][term], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals):
                ax.hist(vals, bins=BINS, color=color_row, alpha=0.7, edgecolor="k",
                        label="Task Position Reshuffled AMEs", zorder=2)
            else:
                ax.text(0.5, 0.5, "no estimates", ha="center", va="center")

            obs_val = obs.get(spec, {}).get(term, np.nan)
            if not np.isnan(obs_val):
                se = obs_se.get(spec, {}).get(term, np.nan)
                if not np.isnan(se):
                    band = 1.645 * se
                    ax.axvspan(obs_val - band, obs_val + band, color="red", alpha=0.08, zorder=1)
                    ax.axvline(obs_val - band, color="red", linestyle="--", linewidth=1,
                               alpha=0.9, zorder=3)
                    ax.axvline(obs_val + band, color="red", linestyle="--", linewidth=1,
                               alpha=0.9, zorder=3)
                ax.axvline(obs_val, color="red", linestyle="--", linewidth=3,
                           label=f"Observed = {obs_val:.3f}", zorder=4)

            ax.axvline(0.0, color="black", linestyle="-", linewidth=1.5, alpha=0.5, zorder=4)
            ax.set_title(labels.get(term, term), fontsize=style["title"],
                         fontweight=style["title_weight"])
            if c == 0:
                ax.set_ylabel("Count", fontsize=style["ylabel"])
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            ax.set_xlim(lo_x, hi_x)
            ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP))
            ax.set_xlabel("Average Marginal Effect", fontsize=style["xlabel"])
            ax.tick_params(labelsize=style["ticks"])
            ax.legend(loc="best", fontsize=style["legend"])

        fig.tight_layout()
        out = out_dir / f"{base_name}_{spec}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", action="store_true",
                    help="use the pre-2026-08 labels and font sizes")
    args = ap.parse_args()
    style = LEGACY_STYLE if args.legacy else STYLE
    labels = LEGACY_VAR_LABELS if args.legacy else VAR_LABELS

    total = 0
    for dep, suffix in CONFIGS:
        summary_dir = (REPO / "data" / "computed_objects"
                       / f"execTypeVaryingDWA_anthropicIndex{suffix}"
                       / f"regression_summaries_{dep}")
        out_dir = REPO / "writeup" / "plots" / f"execTypeVaryingDWA{suffix}" / dep
        if not summary_dir.is_dir():
            print(f"  SKIP {dep}{suffix}: no cached summaries at {summary_dir}")
            continue
        os.makedirs(out_dir, exist_ok=True)

        for kind in ("full", "filtered"):
            obs_df = pd.read_csv(summary_dir / f"regression_ame_results_{kind}_0.csv")
            obs, obs_se = results_to_dict(obs_df)
            resh, missing = load_null(summary_dir, kind)
            if missing:
                print(f"  note: {missing} of {N_SHUFFLES} {kind} shuffle files absent")
            written = draw(resh, obs, obs_se, out_dir, f"AME_{kind}_{dep}", style, labels)
            total += len(written)
            print(f"  {dep}{suffix} [{kind}]: {len(written)} figures -> {out_dir}")
    print(f"done: {total} figures written")


if __name__ == "__main__":
    main()
