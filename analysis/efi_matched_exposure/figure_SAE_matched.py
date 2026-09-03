"""
SA.E frequency-robustness heatmap, MATCHED exposure specification.

Redraws writeup/plots/fragmentationIndex_weeklyTasks/frag_logic_threshold_heatmap_def1.png
in the same visual style, but on the MATCHED estimates (the E1|E2 share as the exposure
regressor instead of the E1 share alone). The published exhibit is read only and is never
touched; everything written here goes to

    writeup/plots/efi_matched_exposure/
        frag_logic_threshold_heatmap_def1_matched.png
        frag_logic_threshold_heatmap_def1_comparison.png

Estimates come from sae_frequency_sweep_old_vs_matched.csv, produced by
SAE_frequency_robustness.py in this folder. If that CSV is absent, this script runs the
producer first.

Styling is copied from analysis/make_frag_def1_heatmap.py: the same row order
(All tasks / Daily+ / SeveralDaily+ / Hourly+), the same threshold columns
(>=20/35/50/65%), the same separator rule under the all-tasks baseline row, the same
"coef stars \\n N=" cell annotation, the same significance-star cutoffs (1/5/10%), the
same RdBu_r diverging colormap (blue negative, red positive) and the same
TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=+vmax) centring.

THE ONE DELIBERATE DEPARTURE IS THE COLOUR LIMIT.
The published script autoscales each panel separately with

    vmax = np.nanmax(np.abs(M))                      # per panel
    TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

That rule is fine when every cell is of comparable size, as in the published spec.  It
fails under the matched spec: 37 of the 39 matched cells lie inside +/-0.33, but the
Hourly+ >=65% corner cut has only 20 occupations and its EFI coefficient is +0.95 (no FE)
and +1.01 (major-group FE).  Per-panel autoscaling would set vmax near 1.0 in two of the
three panels, painting every other cell in those panels an identical near-white, and it
would give the three panels three different scales so they could not be compared.

Instead: ONE scale shared by all three panels, still diverging and still centred exactly
at zero so the sign reads, with the limit set by the cells outside that 20-occupation
corner cut and rounded up to the next 0.05 (VLIM below, 0.35 on the current data).  The
two corner cells then saturate; they are drawn in a distinct out-of-range colour, hatched,
and their annotation carries an explicit "off scale" tag with the true coefficient still
printed.  A colorbar with an upper extend arrow makes the clipping visible.
"""

import os as _os
import subprocess as _subprocess
import sys as _sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Repo root and output dirs are derived from this file's own location, so the folder
# can move and no home directory is baked in.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.abspath(_os.path.join(_HERE, "..", ".."))

PLOTS = _os.path.abspath(_os.path.join(REPO, "writeup", "plots", "efi_matched_exposure"))
_os.makedirs(PLOTS, exist_ok=True)

OUT_MATCHED = _os.path.join(PLOTS, "frag_logic_threshold_heatmap_def1_matched.png")
OUT_COMPARE = _os.path.join(PLOTS, "frag_logic_threshold_heatmap_def1_comparison.png")

CSV_NAME = "sae_frequency_sweep_old_vs_matched.csv"
# The producer's output directory has moved once already (analysis/.../results ->
# data/computed_objects/<analysis folder>, the repo convention). Look in both, newest wins.
CSV_CANDIDATES = [
    _os.path.abspath(_os.path.join(REPO, "data", "computed_objects", "efi_matched_exposure", CSV_NAME)),
    _os.path.abspath(_os.path.join(_HERE, "results", CSV_NAME)),
]
PRODUCER = _os.path.join(_HERE, "SAE_frequency_robustness.py")

DPI = 300

# ---------------------------------------------------------------- published styling
# Row/column order, labels, FE panels and star cutoffs: verbatim from make_frag_def1_heatmap.py.
SWEEP_THRESHOLDS = [20, 35, 50, 65]
fam_order = ["daily", "sevdaily", "hourly"]
fam_label = {"daily": "Daily+", "sevdaily": "SeveralDaily+", "hourly": "Hourly+"}
row_order = ["all"] + fam_order
row_label = {"all": "All tasks", **fam_label}
FE_SPECS = [("none", "No Fixed Effects"), ("Major", "SOC Major Group FE"), ("Minor", "SOC Minor Group FE")]

CMAP_NAME = "RdBu_r"            # published: blue negative, red positive
OVER_COLOR = "#4a0009"          # darker than the top of RdBu_r, for the clipped cells
CELL_FS = 8                     # published annotation font size
WHITE_TEXT_FRAC = 0.6           # published rule: white text once |coef| >= 0.6 * vmax

# The cut whose tiny sample drives the outlier; excluded from the colour-limit calculation
# only, never from the figure.
OUTLIER_CUT = ("hourly", 65)


def star(p):
    return "***" if (pd.notna(p) and p < .01) else "**" if (pd.notna(p) and p < .05) else "*" if (pd.notna(p) and p < .1) else ""


# ------------------------------------------------------------------ estimates
def find_results_csv():
    present = [p for p in CSV_CANDIDATES if _os.path.exists(p)]
    if not present:
        return None
    return max(present, key=_os.path.getmtime)


def load_sweep():
    path = find_results_csv()
    if path is None:
        print(f"{CSV_NAME} not found; running {_os.path.basename(PRODUCER)} first", file=_sys.stderr)
        _subprocess.run([_sys.executable, PRODUCER], check=True)
        path = find_results_csv()
        if path is None:
            raise SystemExit(f"{PRODUCER} ran but did not write {CSV_NAME} to any of {CSV_CANDIDATES}")
    print("estimates from", path)
    return pd.read_csv(path)


def cell(sub, fam, t):
    """One (family, threshold) cell of one spec x FE slice, or None.

    The all-tasks baseline is a single regression, so it fills all four threshold
    columns of its row: same convention as the published script.
    """
    x = sub[sub["family"] == "all"] if fam == "all" else sub[(sub["family"] == fam) & (sub["threshold"] == t)]
    if len(x) and pd.notna(x["coef"].iloc[0]):
        r = x.iloc[0]
        return dict(coef=float(r["coef"]), se=float(r["se"]), pval=float(r["pval"]), n=int(r["N_occ"]))
    return None


def grid(sweep, spec, fe):
    """4x4 arrays of coefficient, p-value, SE and occupation count for one panel."""
    sub = sweep[(sweep["spec"] == spec) & (sweep["FE"] == fe)]
    shape = (len(row_order), len(SWEEP_THRESHOLDS))
    M = np.full(shape, np.nan)
    P = np.full(shape, np.nan)
    S = np.full(shape, np.nan)
    N = np.full(shape, -1, dtype=int)
    for i, fam in enumerate(row_order):
        for j, t in enumerate(SWEEP_THRESHOLDS):
            c = cell(sub, fam, t)
            if c:
                M[i, j], P[i, j], S[i, j], N[i, j] = c["coef"], c["pval"], c["se"], c["n"]
    return M, P, S, N


def colour_limit(sweep, spec):
    """Shared, zero-centred colour limit for all three panels.

    Set by every cell EXCEPT the 20-occupation Hourly+ >=65% cut, rounded up to the next
    0.05 so the number is a round one and the choice is reproducible. Cells beyond it
    saturate and are flagged in the figure rather than being allowed to flatten the scale.
    """
    s = sweep[sweep["spec"] == spec]
    keep = s[~((s["family"] == OUTLIER_CUT[0]) & (s["threshold"] == OUTLIER_CUT[1]))]
    raw = float(np.nanmax(np.abs(keep["coef"])))
    return float(np.ceil(raw / 0.05) * 0.05), raw


# ------------------------------------------------------------------- heatmap
def draw_panel(ax, M, P, N, vlim, cmap, show_ylabels=True):
    """Published convention: each panel autoscales to its own largest absolute coefficient,
    TwoSlopeNorm centred at zero, no clipping and no hatching. `vlim` is ignored.

    The Hourly+ / >=65% cut is blanked, following the neighbour heatmaps, which leave that cell
    white with an en-dash. It retains 20 occupations, too few for the three-regressor
    specification to say anything, and on the matched spec it returns +0.95 and +1.01, which would
    otherwise set the colour scale for the whole panel."""
    M = M.copy()
    i_out, j_out = row_order.index(OUTLIER_CUT[0]), SWEEP_THRESHOLDS.index(OUTLIER_CUT[1])
    M[i_out, j_out] = np.nan
    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(SWEEP_THRESHOLDS)))
    ax.set_xticklabels([f"≥{t}%" for t in SWEEP_THRESHOLDS])
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels([row_label[f] for f in row_order] if show_ylabels else [""] * len(row_order))
    ax.axhline(0.5, color="black", lw=2)
    for i in range(len(row_order)):
        for j in range(len(SWEEP_THRESHOLDS)):
            v = M[i, j]
            if np.isnan(v):
                # blank cell, as in the neighbour heatmaps
                ax.text(j, i, "\u2013", ha="center", va="center", fontsize=CELL_FS + 2,
                        color="black")
                continue
            colr = "white" if abs(v) >= WHITE_TEXT_FRAC * vmax else "black"
            ax.text(j, i, f"{v:.2f}{star(P[i, j])}\nN={N[i, j]}", ha="center", va="center",
                    fontsize=CELL_FS, color=colr, zorder=3)
    ax.set_xlabel("Threshold")
    return im


def figure_matched(sweep, vlim, raw):
    cmap = plt.get_cmap(CMAP_NAME).copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    for ax, (fe_code, fe_name) in zip(axes, FE_SPECS):
        M, P, S, N = grid(sweep, "MATCHED", fe_code)
        draw_panel(ax, M, P, N, vlim, cmap)
        ax.set_title(fe_name, fontsize=11)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.14, wspace=0.28)
    fig.savefig(OUT_MATCHED, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved", OUT_MATCHED)


# ---------------------------------------------------------------- comparison
PUB_COLOR = "#26456e"     # published spec (exposure = E1 share)
MAT_COLOR = "#d95f02"     # matched spec (exposure = E1 or E2 share)
XLO, XHI = -1.15, 1.50    # comparison-panel x limits; whiskers beyond get a caret
LAB_PAD = 0.05            # gap between a whisker end and its value label


def _row_labels():
    labs, keys = [], []
    for fam in row_order:
        if fam == "all":
            labs.append("All tasks")
            keys.append(("all", 0))
        else:
            for t in SWEEP_THRESHOLDS:
                labs.append(f"{fam_label[fam]} ≥{t}%")
                keys.append((fam, t))
    return labs, keys


def figure_comparison(sweep):
    labs, keys = _row_labels()
    ypos = list(range(len(labs)))[::-1]      # first cut at the top
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.6), sharey=True)
    for ax, (fe_code, fe_name) in zip(axes, FE_SPECS):
        ax.axvline(0, color="#2b2b2b", lw=1, ls="--", zorder=1)
        ax.axhspan(len(labs) - 1.5, len(labs) - 0.5, color="0.93", zorder=0)   # all-tasks baseline band
        for y, (fam, t) in zip(ypos, keys):
            o = sweep[(sweep.spec == "OLD") & (sweep.FE == fe_code) & (sweep.family == fam) & (sweep.threshold == t)]
            m = sweep[(sweep.spec == "MATCHED") & (sweep.FE == fe_code) & (sweep.family == fam) & (sweep.threshold == t)]
            if not len(o) or not len(m):
                continue
            o, m = o.iloc[0], m.iloc[0]
            ax.annotate("", xy=(m.coef, y), xytext=(o.coef, y),
                        arrowprops=dict(arrowstyle="-|>", color="0.55", lw=1.0,
                                        shrinkA=4, shrinkB=4), zorder=2)
            # Value labels sit on the marker's own line, just outside the whisker end
            # (published to the left, matched to the right), so they cannot collide with
            # the neighbouring cut's markers.
            for r, col, mk, off, side in [(o, PUB_COLOR, "o", +0.18, -1), (m, MAT_COLOR, "s", -0.18, +1)]:
                lo, hi = r.coef - 1.96 * r.se, r.coef + 1.96 * r.se
                lo_c, hi_c = max(lo, XLO), min(hi, XHI)
                ax.plot([lo_c, hi_c], [y + off, y + off], color=col, lw=1.4, zorder=3)
                for edge, val, mkr in [(XLO, lo, "<"), (XHI, hi, ">")]:
                    if (val < XLO) if mkr == "<" else (val > XHI):
                        ax.plot([edge], [y + off], marker=mkr, ms=5, color=col, zorder=4)
                ax.plot([r.coef], [y + off], marker=mk, ms=6.5, color=col,
                        mec="white", mew=0.6, zorder=5)
                tx = (lo_c - LAB_PAD) if side < 0 else (hi_c + LAB_PAD)
                ha, ty = ("right" if side < 0 else "left"), y + off
                if tx < XLO + 0.06 or tx > XHI - 0.06:
                    # CI runs off the panel: park the label beside the point estimate
                    # instead of on top of the out-of-range caret.
                    tx, ha = r.coef, "center"
                    ty = y + off + (0.28 if off > 0 else -0.28)
                ax.text(tx, ty, f"{r.coef:+.2f}{star(r.pval)}", ha=ha, va="center",
                        fontsize=7, color=col, zorder=6,
                        bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.6))
        ax.set_xlim(XLO, XHI)
        ax.set_ylim(-0.9, len(labs) - 0.1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labs)
        ax.axhline(len(labs) - 1.5, color="black", lw=1.2)      # separator under the baseline row
        ax.set_title(fe_name, fontsize=11)
        ax.set_xlabel("EFI coefficient (standardized), 95% CI")
        ax.grid(axis="x", color="0.9", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    # occupation counts on the right edge of the last panel
    axr = axes[-1].twinx()
    axr.set_ylim(axes[-1].get_ylim())
    axr.set_yticks(ypos)
    axr.set_yticklabels([f"N={int(sweep[(sweep.spec=='OLD') & (sweep.FE=='none') & (sweep.family==f) & (sweep.threshold==t)].N_occ.iloc[0])}"
                         for f, t in keys], fontsize=8, color="0.35")
    axr.tick_params(length=0)
    handles = [Line2D([], [], color=PUB_COLOR, marker="o", ls="-", lw=1.4, ms=6,
                      label="Published spec: exposure = E1 share"),
               Line2D([], [], color=MAT_COLOR, marker="s", ls="-", lw=1.4, ms=6,
                      label="Matched spec: exposure = E1 or E2 share"),
               Line2D([], [], color="0.55", lw=1.0, label="shift from published to matched")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.105, right=0.945, top=0.93, bottom=0.145, wspace=0.10)
    fig.savefig(OUT_COMPARE, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved", OUT_COMPARE)


# --------------------------------------------------------------------- report
def report(sweep, vlim, raw):
    print()
    print("=" * 104)
    print("PLOTTED VALUES -- MATCHED spec (exposure = E1|E2 share, num_E1E2_tasks control kept)")
    print("=" * 104)
    labs, keys = _row_labels()
    for fe_code, fe_name in FE_SPECS:
        print(f"\n--- {fe_name} (FE={fe_code}) ---")
        print(f"{'cut':<22}{'N':>6}{'matched coef':>14}{'se':>8}{'p':>9}{'  ':>2}"
              f"{'published coef':>16}{'se':>8}{'p':>9}")
        for lab, (fam, t) in zip(labs, keys):
            m = sweep[(sweep.spec == "MATCHED") & (sweep.FE == fe_code) & (sweep.family == fam) & (sweep.threshold == t)].iloc[0]
            o = sweep[(sweep.spec == "OLD") & (sweep.FE == fe_code) & (sweep.family == fam) & (sweep.threshold == t)].iloc[0]
            flag = "  <-- off scale" if abs(m.coef) > vlim else ""
            print(f"{lab:<22}{int(m.N_occ):>6}{m.coef:>+14.4f}{m.se:>8.3f}{m.pval:>9.4f}{star(m.pval):<3}"
                  f"{o.coef:>+13.4f}{o.se:>8.3f}{o.pval:>9.4f}{star(o.pval):<3}{flag}")
    mt = sweep[sweep.spec == "MATCHED"]
    keep = mt[~((mt.family == OUTLIER_CUT[0]) & (mt.threshold == OUTLIER_CUT[1]))]
    print()
    print(f"colour limit: max |coef| outside the {row_label[OUTLIER_CUT[0]]} >={OUTLIER_CUT[1]}% cut "
          f"= {raw:.4f}  ->  VLIM = +/-{vlim:.2f}")
    print(f"matched coef range, all 39 cells      : [{mt.coef.min():+.4f}, {mt.coef.max():+.4f}]")
    print(f"matched coef range, 36 in-scale cells : [{keep.coef.min():+.4f}, {keep.coef.max():+.4f}]")
    clipped = mt[mt.coef.abs() > vlim]
    print(f"clipped cells ({len(clipped)}):")
    for _, r in clipped.iterrows():
        print(f"    {r['cut']:<22} FE={r.FE:<6} coef={r.coef:+.4f} se={r.se:.3f} p={r.pval:.4f} N={int(r.N_occ)}")
    print(f"significant at 5%, matched: {int(((mt.pval < .05)).sum())}/39   "
          f"at 10%: {int(((mt.pval < .10)).sum())}/39")
    old = sweep[sweep.spec == "OLD"]
    print(f"significant at 5%, published: {int((old.pval < .05).sum())}/39   "
          f"negative and significant at 5%: {int(((old.coef < 0) & (old.pval < .05)).sum())}/39")


def main():
    sweep = load_sweep()
    vlim, raw = colour_limit(sweep, "MATCHED")
    figure_matched(sweep, vlim, raw)
    figure_comparison(sweep)
    report(sweep, vlim, raw)


if __name__ == "__main__":
    main()
