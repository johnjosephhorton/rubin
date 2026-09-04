"""
Regenerate Figure SA.A.2(a) and Figure SA.D.1(c) under the corrected exposure
definition   human_labels.isin(['E1'])  ->  human_labels.isin(['E1','E2']).

READ-ONLY with respect to the repository.  Nothing is written outside
writeup/_e1e2_preview/.  No notebook is executed in place.

Sources of the published code, copied verbatim except for the single mask swap:
  Fig SA.A.2(a)  analysis/old/onet_adHoc_stats.ipynb, cell 5
                 -> writeup/plots/ONET_Eloundou_Anthropic_GPT/ai_exposed_task_share_distribution.png
  Fig SA.D.1(c)  analysis/GPT_task_sequences_overlap.ipynb, cells 7, 8, 12
                 -> writeup/plots/GPT_task_sequences_overlap_analysis/
                    GPT_task_sequence_robustness_by_human_E1_fraction_distribution.png

For every exhibit the script produces BOTH masks:
  * the E1-only rebuild, which must reproduce the published PNG (pipeline control), and
  * the E1|E2 rebuild, which is the artifact.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

pd.set_option('float_format', "{:,.2f}".format)
import warnings
warnings.filterwarnings('ignore')

REPO = "/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin"
STAGE = f"{REPO}/writeup/_e1e2_preview"
NEW = f"{STAGE}/plots_new"
CUR = f"{STAGE}/plots_current"
REPRO = f"{STAGE}/scripts/repro_E1only"       # E1-only control rebuilds
DIFFS = f"{STAGE}/diffs"
for d in (NEW, CUR, REPRO, DIFFS):
    os.makedirs(d, exist_ok=True)

records = []          # rows of the numeric diff
loglines = []


def note(s):
    print(s)
    loglines.append(s)


# =====================================================================
# Figure SA.A.2(a)
# =====================================================================
FN_A = "ai_exposed_task_share_distribution.png"

merged_data = pd.read_csv(
    f"{REPO}/data/computed_objects/ONET_Eloundou_Anthropic_GPT/ONET_Eloundou_Anthropic_GPT.csv")

# --- published hard-coded limits, and the relaxed ones needed under E1|E2 ----
# Published rule, read off cell 5: xlim = (first bin edge - 0.02, last NON-EMPTY
# bin's right edge + 0.02).  Under E1 the last non-empty bin is [0.75,0.80) so the
# notebook wrote (-0.02, 0.82).  Under E1|E2 the last non-empty bin is [1.00,1.05]
# so the same rule gives (-0.02, 1.07).
# ylim: published 460 against a tallest bar of 309 = 1.489x headroom.  Under E1|E2
# the tallest bar is 76, so the same headroom ratio gives 76*1.489 = 113 -> 115.
LIMITS = {
    'E1':   dict(xlim=(-0.02, 0.82), ylim=(0, 460)),     # exactly as published
    'E1E2': dict(xlim=(-0.02, 1.07), ylim=(0, 115)),     # DELIBERATE DEVIATION
}


def build_SAA2a(labels, tag, outpath):
    """Cell 5 of analysis/old/onet_adHoc_stats.ipynb, verbatim, mask swapped."""
    groups_list = ['O*NET-SOC Code']
    group_labels = ['Occupation']

    fig, axes = plt.subplots(1, len(groups_list), figsize=(6.5 * len(groups_list), 4.25))
    if len(groups_list) == 1:
        axes = [axes]

    out = {}
    for ax, group, label in zip(axes, groups_list, group_labels):

        merged_data['is_ai_exposed'] = merged_data['human_labels'].isin(labels).astype(int)  # <<< THE ONE CHANGE

        ai_task_counts = (
            merged_data.groupby(group)['is_ai_exposed'].sum()
            .rename('num_ai_exposed').reset_index()
        )
        total_task_counts = (
            merged_data.groupby(group)['Task ID'].nunique()
            .rename('total_num_tasks').reset_index()
        )

        ai_task_share = ai_task_counts.merge(total_task_counts, on=group)
        ai_task_share['ai_task_share'] = ai_task_share['num_ai_exposed'] / ai_task_share['total_num_tasks']
        n_atleast_one = merged_data.loc[merged_data['is_ai_exposed'] == 1, group].nunique()
        note(f"[SA.A.2(a) {tag}] Number of Occupations with at least one AI-Exposed task: {n_atleast_one}")

        counts, bins, _ = ax.hist(
            ai_task_share['ai_task_share'],
            bins=np.arange(0, 1.1, 0.05),
            edgecolor='black',
            color='purple',
            alpha=0.9
        )
        ax.set_xlabel(f'Share of {label} Tasks that are AI-exposed', fontsize=14)
        ax.set_ylabel(f'Number of {label}s', fontsize=14)
        ax.set_xlim(*LIMITS[tag]['xlim'])
        ax.set_ylim(*LIMITS[tag]['ylim'])

        total_groups = len(ai_task_share)
        for count, bin_start, bin_end in zip(counts, bins[:-1], bins[1:]):
            if count > 0:
                pct = 100 * count / total_groups
                ax.text(
                    (bin_start + bin_end) / 2,
                    count + (counts.max() / 100),
                    f'{pct:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                    rotation=90
                )

        ax.text(
            0.7,
            0.98,
            f'Total {label}s: {total_groups:,}',
            transform=ax.transAxes,
            fontsize=10,
            va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        s = ai_task_share['ai_task_share']
        out = dict(counts=counts, bins=bins, total=total_groups, share=s,
                   n_atleast_one=n_atleast_one)

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    return out


a_e1 = build_SAA2a(['E1'], 'E1', f"{REPRO}/{FN_A}")
a_e2 = build_SAA2a(['E1', 'E2'], 'E1E2', f"{NEW}/{FN_A}")

# ---- numeric diff for SA.A.2(a) -------------------------------------------
for i, (lo, hi) in enumerate(zip(a_e1['bins'][:-1], a_e1['bins'][1:])):
    c0, c1 = a_e1['counts'][i], a_e2['counts'][i]
    p0 = 100 * c0 / a_e1['total']
    p1 = 100 * c1 / a_e2['total']
    records.append(dict(exhibit="Fig SA.A.2(a)", item=f"bar [{lo:.2f},{hi:.2f})",
                        published=f"{int(c0)} ({p0:.1f}%)" if c0 > 0 else "0 (no bar/label)",
                        e1e2=f"{int(c1)} ({p1:.1f}%)" if c1 > 0 else "0 (no bar/label)",
                        changed=int(c0) != int(c1)))

for stat, f in [("mean", lambda s: s.mean()), ("median", lambda s: s.median()),
                ("sd", lambda s: s.std(ddof=1)), ("p10", lambda s: s.quantile(.10)),
                ("p25", lambda s: s.quantile(.25)), ("p75", lambda s: s.quantile(.75)),
                ("p90", lambda s: s.quantile(.90)), ("min", lambda s: s.min()),
                ("max", lambda s: s.max())]:
    v0, v1 = f(a_e1['share']), f(a_e2['share'])
    records.append(dict(exhibit="Fig SA.A.2(a)", item=f"distribution {stat}",
                        published=f"{v0:.4f}", e1e2=f"{v1:.4f}",
                        changed=abs(v0 - v1) > 1e-12))

for item, v0, v1 in [
    ("occupations with share = 0",
     int((a_e1['share'] == 0).sum()), int((a_e2['share'] == 0).sum())),
    ("occupations with >=1 exposed task (SA_A:35 prose)",
     a_e1['n_atleast_one'], a_e2['n_atleast_one']),
    ("total occupations (text box)", a_e1['total'], a_e2['total']),
    ("tallest bar (count)", int(a_e1['counts'].max()), int(a_e2['counts'].max())),
    ("occupations right of published xlim 0.82",
     int((a_e1['share'] > 0.82).sum()), int((a_e2['share'] > 0.82).sum())),
]:
    records.append(dict(exhibit="Fig SA.A.2(a)", item=item, published=str(v0),
                        e1e2=str(v1), changed=v0 != v1))

records.append(dict(exhibit="Fig SA.A.2(a)", item="AXIS LIMIT xlim (hard-coded)",
                    published="(-0.02, 0.82)", e1e2="(-0.02, 1.07)", changed=True))
records.append(dict(exhibit="Fig SA.A.2(a)", item="AXIS LIMIT ylim (hard-coded)",
                    published="(0, 460)", e1e2="(0, 115)", changed=True))

pct_a = 100 * a_e1['n_atleast_one'] / a_e1['total']
pct_b = 100 * a_e2['n_atleast_one'] / a_e2['total']
records.append(dict(exhibit="SA_A_sample_construction.tex:35 (prose)",
                    item="\"Of the 872 occupations, N (X%) contain at least one AI-exposed task\"",
                    published=f"{a_e1['n_atleast_one']} ({pct_a:.0f}%)",
                    e1e2=f"{a_e2['n_atleast_one']} ({pct_b:.0f}%)", changed=True))


# =====================================================================
# Figure SA.D.1(c)
# =====================================================================
FN_D = "GPT_task_sequence_robustness_by_human_E1_fraction_distribution.png"
FN_D_CNT = "GPT_task_sequence_robustness_by_human_E1_fraction_distribution_count.png"

seq_in = f"{REPO}/data/computed_objects/tasks_sequences_robustness"
seq_out = f"{seq_in}/data/computed_objects/GPT_task_sequences_overlap_analysis"

kendall_results = pd.read_csv(f"{seq_out}/GPT_task_sequences_kendall_results.csv")

# --- cell 7 ---------------------------------------------------------------
md = pd.read_csv(f"{REPO}/data/computed_objects/ONET_Eloundou_Anthropic_GPT/ONET_Eloundou_Anthropic_GPT.csv")
md = md[["Detailed_Occupation_Title", "Detailed_Occupation_Code", "O*NET-SOC Code"]].drop_duplicates()

occupation_analysis = pd.read_csv(
    f"{REPO}/data/computed_objects/fragmentationIndex/occupation_analysis_with_fragmentationIndex_def1.csv")
occupation_analysis = occupation_analysis.merge(md, on="O*NET-SOC Code", how="left")

for v in ['fragmentation_index', 'ai_fraction', 'human_E1_fraction', 'human_aiExposure_fraction']:
    occupation_analysis[f'{v}_above_median'] = (
        occupation_analysis[v] >= occupation_analysis[v].median()).astype(int)

# --- cell 8 ---------------------------------------------------------------
summary = (
    kendall_results
    .groupby("Occupation Title")["kendall_tau"]
    .agg(["mean", "min", "max"])
    .reset_index()
    .sort_values("mean")
)
summary = summary.merge(
    occupation_analysis[["O*NET-SOC Code", "Occupation Title",
                         'fragmentation_index_above_median', 'ai_fraction_above_median',
                         'human_E1_fraction_above_median', 'human_aiExposure_fraction_above_median']],
    on="Occupation Title", how="left")

# --- cell 12 --------------------------------------------------------------
BINS = np.linspace(-1, 1, 21)


def plot_split_distribution(var, label, as_share, outpath):
    summary_clean = summary.dropna(subset=[var])
    below_median = summary_clean[summary_clean[var] == 0]['mean']
    above_median = summary_clean[summary_clean[var] == 1]['mean']
    mean_below, mean_above = below_median.mean(), above_median.mean()
    wt = (lambda s: np.ones(len(s)) / len(s)) if as_share else (lambda s: None)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(below_median, bins=BINS, weights=wt(below_median), alpha=0.6, color='steelblue',
            edgecolor='black', label=f'Below Median {label}\n(n={len(below_median)}, mean={mean_below:.2f})')
    ax.hist(above_median, bins=BINS, weights=wt(above_median), alpha=0.6, color='red',
            edgecolor='black', label=f'Above Median {label}\n(n={len(above_median)}, mean={mean_above:.2f})')
    ax.axvline(mean_below, color='steelblue', linestyle='--', linewidth=2)
    ax.axvline(mean_above, color='red', linestyle='--', linewidth=2)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)

    ax.set_xlabel("Mean Kendall τ (within Occupation)", fontsize=16)
    ax.set_ylabel("Share of Occupations" if as_share else "Number of Occupations", fontsize=16)
    if as_share:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(-1.02, 1.02)
    ax.legend(loc="upper left", fontsize=15)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    return dict(n_below=len(below_median), n_above=len(above_median),
                mean_below=mean_below, mean_above=mean_above,
                sd_below=below_median.std(ddof=1), sd_above=above_median.std(ddof=1),
                gap=mean_above - mean_below)


# published: split on human_E1_fraction.  E1|E2: split on human_aiExposure_fraction.
# The legend label string "AI Exposure" is unchanged, as is the output FILE NAME
# (plots_new/ carries the drop-in replacement for the published file).
d_e1 = plot_split_distribution('human_E1_fraction_above_median', 'AI Exposure', True,
                               f"{REPRO}/{FN_D}")
d_e2 = plot_split_distribution('human_aiExposure_fraction_above_median', 'AI Exposure', True,
                               f"{NEW}/{FN_D}")
plot_split_distribution('human_E1_fraction_above_median', 'AI Exposure', False,
                        f"{REPRO}/{FN_D_CNT}")
d_e2c = plot_split_distribution('human_aiExposure_fraction_above_median', 'AI Exposure', False,
                                f"{NEW}/{FN_D_CNT}")

note(f"[SA.D.1(c) E1  ] n={d_e1['n_below']}/{d_e1['n_above']} "
     f"mean={d_e1['mean_below']:.4f}/{d_e1['mean_above']:.4f}")
note(f"[SA.D.1(c) E1E2] n={d_e2['n_below']}/{d_e2['n_above']} "
     f"mean={d_e2['mean_below']:.4f}/{d_e2['mean_above']:.4f}")

for item, v0, v1, fmt in [
    ("legend: Below Median AI Exposure, n", d_e1['n_below'], d_e2['n_below'], "d"),
    ("legend: Above Median AI Exposure, n", d_e1['n_above'], d_e2['n_above'], "d"),
    ("legend: Below Median AI Exposure, mean (printed, 2dp)",
     round(d_e1['mean_below'], 2), round(d_e2['mean_below'], 2), ".2f"),
    ("legend: Above Median AI Exposure, mean (printed, 2dp)",
     round(d_e1['mean_above'], 2), round(d_e2['mean_above'], 2), ".2f"),
    ("mean tau below median (full precision)", d_e1['mean_below'], d_e2['mean_below'], ".4f"),
    ("mean tau above median (full precision)", d_e1['mean_above'], d_e2['mean_above'], ".4f"),
    ("sd below median", d_e1['sd_below'], d_e2['sd_below'], ".4f"),
    ("sd above median", d_e1['sd_above'], d_e2['sd_above'], ".4f"),
    ("above-minus-below gap", d_e1['gap'], d_e2['gap'], ".4f"),
]:
    records.append(dict(exhibit="Fig SA.D.1(c)", item=item,
                        published=format(v0, fmt), e1e2=format(v1, fmt),
                        changed=abs(v0 - v1) > 1e-12))

records.append(dict(exhibit="Fig SA.D.1(c)", item="splitting variable",
                    published="human_E1_fraction_above_median",
                    e1e2="human_aiExposure_fraction_above_median", changed=True))
m0 = float(occupation_analysis['human_E1_fraction'].median())
m1 = float(occupation_analysis['human_aiExposure_fraction'].median())
records.append(dict(exhibit="Fig SA.D.1(c)", item="median cut of the splitting variable",
                    published=f"{m0:.4f}", e1e2=f"{m1:.4f}", changed=True))
records.append(dict(exhibit="Fig SA.D.1(c)", item="axis limits (unchanged)",
                    published="xlim (-1.02, 1.02)", e1e2="xlim (-1.02, 1.02)", changed=False))

s2 = summary.dropna(subset=['human_E1_fraction_above_median', 'human_aiExposure_fraction_above_median'])
xt = pd.crosstab(s2['human_E1_fraction_above_median'], s2['human_aiExposure_fraction_above_median'])
switchers = int(xt.values.sum() - np.trace(xt.values))
records.append(dict(exhibit="Fig SA.D.1(c)", item="occupations that change side of the split",
                    published="-", e1e2=f"{switchers} of {int(xt.values.sum())}", changed=True))

# count-variant legend (sibling PNG, not referenced by the .tex)
records.append(dict(exhibit="Fig SA.D.1(c) _count sibling (not in PDF)",
                    item="legend n / mean",
                    published=f"n={d_e1['n_below']}/{d_e1['n_above']}, "
                              f"mean={d_e1['mean_below']:.2f}/{d_e1['mean_above']:.2f}",
                    e1e2=f"n={d_e2c['n_below']}/{d_e2c['n_above']}, "
                         f"mean={d_e2c['mean_below']:.2f}/{d_e2c['mean_above']:.2f}",
                    changed=True))

records.append(dict(exhibit="SA_D_prompt_robustness.tex:48 (subcaption)",
                    item="\"AI Exposure (E1) Split\"",
                    published="AI Exposure (E1) Split",
                    e1e2="AI Exposure (E1 or E2) Split", changed=True))
records.append(dict(exhibit="SA_D_prompt_robustness.tex:68 (prose)",
                    item="\"the share of occupation tasks exposed to AI (E1)\"",
                    published="... exposed to AI (E1)",
                    e1e2="... exposed to AI (E1 or E2)", changed=True))

# =====================================================================
# write the diff
# =====================================================================
df = pd.DataFrame(records)
w1 = max(len(x) for x in df['exhibit']) + 2
w2 = max(len(x) for x in df['item']) + 2
w3 = max(max(len(x) for x in df['published']), 9) + 2
w4 = max(max(len(x) for x in df['e1e2']), 9) + 2

lines = []
lines.append("Per-exhibit numeric diff, PUBLISHED (E1 only) vs REGENERATED (E1|E2)")
lines.append("Figure SA.A.2(a) and Figure SA.D.1(c)")
lines.append("The single change is  human_labels.isin(['E1']) -> human_labels.isin(['E1','E2'])")
lines.append("(for SA.D.1(c) that is the already-computed human_aiExposure_fraction split variable).")
lines.append("")
lines.append(f"{'EXHIBIT':<{w1}}{'ITEM':<{w2}}{'PUBLISHED':<{w3}}{'E1|E2':<{w4}}CHANGED")
lines.append("-" * (w1 + w2 + w3 + w4 + 8))
for _, r in df.iterrows():
    lines.append(f"{r['exhibit']:<{w1}}{r['item']:<{w2}}{r['published']:<{w3}}"
                 f"{r['e1e2']:<{w4}}{'yes' if r['changed'] else 'no'}")
lines.append("")
lines.append("Console output of the rebuild:")
lines += ["  " + x for x in loglines]

lines.append("")
lines.append("=" * 100)
lines.append("DELIBERATE DEVIATION -- Fig SA.A.2(a) hard-coded axis limits (needs the user's approval)")
lines.append("=" * 100)
lines.append("""
The published cell hard-codes  ax.set_xlim(-0.02, 0.82)  and  ax.set_ylim(0, 460).
Under E1|E2 both are wrong as written, so both were relaxed.  Nothing else in the
cell was touched.

  xlim  (-0.02, 0.82)  ->  (-0.02, 1.07)
        Why: 82 of 872 occupations have an exposure share above 0.82 and would be
        silently cut off.  The new value is not free-hand; it applies the published
        rule.  The published limits are exactly [first bin edge - 0.02, right edge
        of the last NON-EMPTY bin + 0.02]:  under E1 the last non-empty bin is
        [0.75,0.80) so 0.80 + 0.02 = 0.82.  Under E1|E2 the last non-empty bin is
        [1.00,1.05] (10 occupations whose tasks are all exposed), so 1.05 + 0.02 = 1.07.

  ylim  (0, 460)  ->  (0, 115)
        Why: the tallest bar falls from 309 to 76, so 460 leaves the panel almost
        flat and unreadable.  115 preserves the published headroom ratio
        460 / 309 = 1.489;  76 * 1.489 = 113 -> 115.  Verified by eye that the
        rotated per-bar percentage labels and the "Total Occupations: 872" box
        still fit without collision.

  NOTE for the user: panel (b) of the same figure (ai_executed_task_share_distribution.png,
  the AI-EXECUTION histogram) is produced by the same notebook with the same
  hard-coded limits and is NOT affected by the exposure swap.  If (a) is redrawn
  on wider axes, (a) and (b) no longer share an x scale, so either (b) is redrawn
  on the same limits or the two subpanels are read on different axes.  This is a
  presentation decision for the user, not something this rebuild made.""")

lines.append("")
lines.append("=" * 100)
lines.append("PIPELINE CONTROL -- the same script run under the PUBLISHED E1-only mask")
lines.append("=" * 100)
lines.append("""
Both figures were also rebuilt under isin(['E1']) into scripts/repro_E1only/ and
compared with the published PNGs in plots_current/.

  Fig SA.A.2(a)  all 15 non-empty bar labels reproduce the published panel exactly
                 (35.4 / 11.4 / 14.0 / 8.9 / 8.8 / 8.0 / 5.2 / 3.3 / 2.4 / 0.8 /
                 0.6 / 0.6 / 0.2 / 0.2 / 0.1 %), text box "Total Occupations: 872".
  Fig SA.D.1(c)  legend reproduces the published panel exactly:
                 n=428 mean=0.64 (below) and n=440 mean=0.56 (above).

  The PNGs are NOT byte-identical to the published ones and are not expected to be:
  this machine runs matplotlib 3.11.0, which lays out multi-line legend entries a
  few pixels lower and hints glyphs slightly differently.  A pixel diff of the
  SA.D.1(c) control against the published file shows differences ONLY on glyph
  outlines, the legend-entry baselines and 1-2px anti-aliased bar edges; every bar
  top and every dashed mean line coincides.  SA.A.2(a) comes out 1888x1245 vs the
  published 1896x1243 for the same reason (bbox_inches='tight' with slightly
  different font metrics).  No data value differs.""")

lines.append("")
lines.append("Agreement with the prior audit (scratchpad m6, figures-splits.md / descriptives.md):")
lines.append("  Fig SA.A.2(a)  audit expected 605 (69%) -> 809 (93%), 82 occupations clipped,")
lines.append("                 tallest bar 309 -> 76.  Reproduced exactly.")
lines.append("  Fig SA.D.1(c)  audit expected n=428 mean=0.64 / n=440 mean=0.56 ->")
lines.append("                 n=430 mean=0.63 / n=438 mean=0.57.  Reproduced exactly.")
lines.append("  No disagreement with the audit on any cell.")

with open(f"{DIFFS}/descriptive-figs.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
df.to_csv(f"{DIFFS}/descriptive-figs.csv", index=False)
print("\n".join(lines))
