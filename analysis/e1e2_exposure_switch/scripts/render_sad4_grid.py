"""Re-render Figure SA.D.4 (Table-2 neighbor AMEs across the 11 GPT prompts) under a chosen
exposure mask.

Plotting code copied VERBATIM out of
analysis/onet_antrhopicIndex_execTypeVaryingDWA_robustness.ipynb cell 15
(`plot_ame_by_model_shaded` and everything above it: SPEC2COLOR from tab10 by index in the
six-element SPECS list, the 90% CI from ame_se * 1.645, x_full / x_ticks, the errorbar styling,
the purple prompt-0 highlight on the fourth spec, the dashed across-prompt mean with its
"Mean (across prompts) = {:.2f}" label, the hard-coded ax.set_ylim(-0.075, 0.175), figsize
(24,5), constrained_layout, dpi=300, bbox_inches='tight').

The ONLY thing that differs between the two renders is the table fed in, which itself differs
only by the mask line in sad4_estimate.py.

Only the four panels the writeup \\includegraphics-es are written:
  ame_no_fe_no_dwa_robustness.png, ame_major_fe_no_dwa_robustness.png,
  ame_minor_fe_no_dwa_robustness.png, ame_no_fe_with_dwa_robustness.png

Writes ONLY under writeup/_e1e2_preview/.
Usage: python3 render_sad4_grid.py
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = "/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin"
STAGE = f"{REPO}/writeup/_e1e2_preview"
PUB = (f"{REPO}/data/computed_objects/"
       "execTypeVaryingDWA_anthropicIndex_noTasksWithRepetitiveDWAs_robustness/allTasks_ai.csv")

# ---- notebook cell 5 ----
TARGET_REGS = ['prev2_is_ai', 'prev_is_ai', 'next_is_ai', 'next2_is_ai']
SPECS = ['no_fe_no_dwa', 'major_fe_no_dwa', 'minor_fe_no_dwa', 'no_fe_with_dwa',
         'no_fe_no_dwa_withTaskDWACount', 'no_fe_with_dwa_withTaskDWACount']
PLOT_TITLES = ['Task ($k-2$) is AI', 'Task ($k-1$) is AI', 'Task ($k+1$) is AI', 'Task ($k+2$) is AI']
WRITE_SPECS = ['no_fe_no_dwa', 'major_fe_no_dwa', 'minor_fe_no_dwa', 'no_fe_with_dwa']

# ---- notebook cell 15 ----
colors = [plt.cm.tab10(i % 10) for i in range(len(SPECS))]
SPEC2COLOR = dict(zip(SPECS, colors))
term_col = "term"


def render(master_df, out_dir, dpi=300):
    os.makedirs(out_dir, exist_ok=True)
    dfp = master_df[master_df["model"].isin(SPECS) & master_df[term_col].isin(TARGET_REGS)].copy()
    z = 1.645
    dfp["ci_low"] = dfp["ame_coef"].astype(float) - z * dfp["ame_se"].astype(float)
    dfp["ci_high"] = dfp["ame_coef"].astype(float) + z * dfp["ame_se"].astype(float)
    dfp["dataset_num"] = pd.to_numeric(dfp["dataset"], errors="coerce")
    dfp = dfp.dropna(subset=["dataset_num"]).copy()
    dfp["dataset_num"] = dfp["dataset_num"].astype(float)

    max_dataset = int(np.nanmax(dfp["dataset_num"]))
    x_full = np.arange(0, max_dataset + 1)
    if len(x_full) <= 25:
        x_ticks = x_full
    else:
        step = int(np.ceil(len(x_full) / 12)); x_ticks = np.arange(0, max_dataset + 1, step)

    figsize = (24, 5); avg_lw = 1.7; avg_alpha = 0.95
    original_label = "Main Prompt + 90% CI"
    fourth_spec = SPECS[3] if len(SPECS) >= 4 else None

    def _plot_panel(ax, sub, title, color, original_dot_color):
        sub = sub.dropna(subset=["dataset_num", "ame_coef", "ci_low", "ci_high"]).sort_values("dataset_num")
        ax.axhline(0, color="0.25", lw=1.2, ls="--", zorder=0)
        if sub.empty:
            ax.set_title(title, fontsize=14); ax.set_xlim(-0.5, max_dataset + 0.5)
            ax.set_xticks(x_ticks); ax.set_xlabel("GPT Prompt", fontsize=14); return
        x = sub["dataset_num"].astype(float).values
        y = sub["ame_coef"].astype(float).values
        lo = sub["ci_low"].astype(float).values
        hi = sub["ci_high"].astype(float).values
        yerr = np.vstack([y - lo, hi - y])
        mask0 = (x == 0)
        m = ~mask0
        if m.any():
            ax.errorbar(x[m], y[m], yerr=yerr[:, m], fmt="o", ms=6, color=color, ecolor=color,
                        elinewidth=1.6, capsize=3.5, capthick=1.6, alpha=0.95, zorder=3,
                        label="Robustness Prompts + 90% CI")
        if mask0.any():
            ax.errorbar(x[mask0], y[mask0], yerr=yerr[:, mask0], fmt="o", ms=7.5,
                        color=original_dot_color, ecolor=original_dot_color, elinewidth=1.8,
                        capsize=3.5, capthick=1.8, alpha=1.0, zorder=6, label=original_label)
        y_mean = float(np.nanmean(y))
        ax.axhline(y_mean, color=color, ls="--", lw=avg_lw, alpha=avg_alpha, zorder=3,
                   label=f"Mean (across prompts) = {y_mean:.2f}")
        ax.set_title(title, fontsize=14)
        ax.set_xlim(-0.5, max_dataset + 0.5); ax.set_xticks(x_ticks)
        ax.set_xlabel("GPT Prompt", fontsize=14)
        ax.set_ylim(-0.075, 0.175)          # hard-coded in the notebook; kept
        ax.legend(loc="best", fontsize=14)

    made = []
    for spec in SPECS:
        if spec not in WRITE_SPECS:
            continue
        spec_color = SPEC2COLOR[spec]
        original_dot_color = "purple" if (fourth_spec is not None and spec == fourth_spec) else "red"
        fig, axes = plt.subplots(1, 4, figsize=figsize, sharey="col", constrained_layout=True)
        for j, (reg, title) in enumerate(zip(TARGET_REGS, PLOT_TITLES)):
            ax = axes[j]
            sub = dfp[(dfp["model"] == spec) & (dfp[term_col] == reg)].copy()
            _plot_panel(ax, sub, title=title, color=spec_color, original_dot_color=original_dot_color)
            if j == 0:
                ax.set_ylabel("Average Marginal Effect (AME)", fontsize=14)
            else:
                ax.set_ylabel(""); ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        f = os.path.join(out_dir, f"ame_{spec}_robustness.png")
        fig.savefig(f, dpi=dpi, bbox_inches="tight"); plt.close()
        made.append(f)
    return made


if __name__ == '__main__':
    pub = pd.read_csv(PUB)
    e1 = pd.read_csv(f"{STAGE}/diffs/_sad4_allTasks_ai_E1.csv")
    e2 = pd.read_csv(f"{STAGE}/diffs/_sad4_allTasks_ai_E1E2.csv")

    K = ['dataset', 'model', 'term']
    p4 = pub[pub.model.isin(WRITE_SPECS)][K + ['ame_coef', 'ame_se', 'p_value', 'nobs']]
    fx = p4.merge(e1, on=K, suffixes=('_p', '_r'))
    print(f"[fixture] E1 rebuild vs published allTasks_ai.csv: n={len(fx)} "
          f"max|dcoef|={np.abs(fx.ame_coef_p - fx.ame_coef_r).max():.2e} "
          f"max|dse|={np.abs(fx.ame_se_p - fx.ame_se_r).max():.2e} "
          f"nobs identical={bool((fx.nobs_p == fx.nobs_r).all())}")

    render(pub, f"{STAGE}/plots_fixture/SAD4_E1_from_published_csv", dpi=300)
    render(e1, f"{STAGE}/plots_fixture/SAD4_E1_rebuild", dpi=300)
    for f in render(e2, f"{STAGE}/plots_new", dpi=300):
        print("wrote", f)

    # The PUBLISHED PNGs were rendered before the notebook's panel titles were renamed from
    # "($t-2$) Task AI" to "Task ($k-2$) is AI" (cell 5). plots_new follows the notebook as it
    # stands today, which is the pipeline; this extra pair reproduces the older label so the two
    # exhibits can also be compared with the title held fixed. Nothing but the title differs.
    PLOT_TITLES[:] = ['($t-2$) Task AI', '($t-1$) Task AI', '($t+1$) Task AI', '($t+2$) Task AI']
    render(pub, f"{STAGE}/plots_fixture/SAD4_E1_from_published_csv_oldTitles", dpi=300)
    render(e2, f"{STAGE}/plots_fixture/SAD4_E1E2_oldTitles", dpi=300)
    print("also wrote the old-panel-title variants under plots_fixture/")
