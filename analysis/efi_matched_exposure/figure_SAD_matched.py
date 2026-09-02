"""
SA.D prompt-robustness figure, MATCHED exposure specification.

Redraws writeup/plots/fragmentationIndex_robustness/fragmentation_index_robustness_definition_1.png
in the same visual style, but on the MATCHED estimates (E1|E2 share as the exposure
regressor instead of the E1 share alone). The published exhibit is read only and is never
touched; everything written here goes to

    writeup/plots/efi_matched_exposure/
        fragmentation_index_robustness_definition_1_matched.png
        fragmentation_index_robustness_definition_1_comparison.png

Estimates come from sad_prompt_robustness_old_vs_matched.csv, produced by
SAD_prompt_robustness.py in the same folder. That producer's output directory has moved
once (analysis/efi_matched_exposure/results -> data/computed_objects/efi_matched_exposure),
so both locations are searched; if the CSV is in neither, this script runs the producer.

Styling is copied from plot_six_coeffs() in analysis/onet_fragmentationIndex_robustness.ipynb
(cell 8). The ONE deliberate departure is the y-axis limits: the published call hardcodes
YLIMS_BY_DEF[1] = {"ai_exposure": (-0.65, 0.65), "fragmentation_index": (-0.65, 0.65)},
which clips four of the eleven matched exposure whiskers in the no-FE panel (the largest
reaches +0.720) and squashes the near-zero matched EFI estimates into the middle fifth of
the panel. Limits are recomputed from the plotted data instead; see pick_ylim().
"""

import os as _os
import subprocess as _subprocess
import sys as _sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Repo root and output dirs are derived from this file's own location, so the folder
# can move and no home directory is baked in.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.abspath(_os.path.join(_HERE, "..", ".."))

PLOTS = _os.path.abspath(_os.path.join(REPO, "writeup", "plots", "efi_matched_exposure"))
_os.makedirs(PLOTS, exist_ok=True)

CSV_NAME = "sad_prompt_robustness_old_vs_matched.csv"
# The producer's output directory has moved once already (analysis/.../results ->
# data/computed_objects/<analysis folder>, the repo convention). Look in both, newest wins.
CSV_CANDIDATES = [
    _os.path.abspath(_os.path.join(REPO, "data", "computed_objects", "efi_matched_exposure", CSV_NAME)),
    _os.path.abspath(_os.path.join(_HERE, "results", CSV_NAME)),
]
PRODUCER = _os.path.join(_HERE, "SAD_prompt_robustness.py")


def find_results_csv():
    present = [p for p in CSV_CANDIDATES if _os.path.exists(p)]
    if not present:
        return None
    return max(present, key=_os.path.getmtime)

DPI = 300
DEFINITION = 1

# ---------------------------------------------------------------- published styling
# Verbatim from plot_six_coeffs() in onet_fragmentationIndex_robustness.ipynb, cell 8.
TERM_LABEL = {"ai_exposure": "AI Exposure", "fragmentation_index": "EFI"}
MODEL_LABEL = {
    "noFE": "No Fixed Effects",
    "majorFE": "SOC Major Group Fixed Effects",
    "minorFE": "SOC Minor Group Fixed Effects",
}
TERM_COLOR = {"ai_exposure": "#1f77b4", "fragmentation_index": "#C49000"}  # blue, dark yellow

CI_MULT = 1.645          # 90% CI, as published
DOT_MS = 6               # robustness prompts
ORIGINAL_MS = 7.5        # prompt 0
ORIGINAL_COLOR = "red"
ZERO_COLOR = "#2b2b2b"
LEGEND_FS = 13
TITLE_FS = 16
LABEL_FS = 14

MODELS = ("noFE", "majorFE", "minorFE")
TERMS = ("ai_exposure", "fragmentation_index")

# Published hardcoded limits, kept only so the script can report what it changed.
PUBLISHED_YLIMS = {"ai_exposure": (-0.65, 0.65), "fragmentation_index": (-0.65, 0.65)}


# ---------------------------------------------------------------- data
def load_results():
    path = find_results_csv()
    if path is None:
        print(f"results CSV absent in {CSV_CANDIDATES}, running {PRODUCER}", flush=True)
        r = _subprocess.run([_sys.executable, PRODUCER], capture_output=True, text=True)
        _sys.stdout.write(r.stdout[-2000:])
        if r.returncode != 0:
            _sys.stderr.write(r.stderr)
            raise RuntimeError(f"{PRODUCER} exited {r.returncode}")
        path = find_results_csv()
        if path is None:
            raise FileNotFoundError(f"{PRODUCER} ran but wrote none of {CSV_CANDIDATES}")
    print(f"estimates read from: {path}")
    df = pd.read_csv(path)
    df["prompt"] = df["prompt"].astype(int)
    for c in ("spec", "model", "term"):
        df[c] = df[c].astype(str)
    return df


# ---------------------------------------------------------------- y-axis limits
def pick_ylim(sub, pad_frac=0.12, headroom_frac=0.28, zero_margin=0.10, step=0.05):
    """Row-wide limits from the plotted data. Replaces the published hardcoded (-0.65, 0.65).

    Four rules, in order:
      1. cover every point and every 90% whisker end in the row;
      2. always include zero, so the zero reference line is on the panel;
      3. pad by 12% of the covered range, and add a further 28% of headroom at the top so
         the legend (pinned upper right, as in the published exhibit) never sits on data;
      4. keep the zero line at least 10% of the panel height off the bottom edge, so it
         reads as a reference line rather than as the axis;
    then round outward to the nearest 0.05 so the tick labels stay round.
    """
    lo_data = float((sub["coef"] - CI_MULT * sub["se"]).min())
    hi_data = float((sub["coef"] + CI_MULT * sub["se"]).max())
    lo = min(lo_data, 0.0)
    hi = max(hi_data, 0.0)
    span = hi - lo
    lo, hi = lo - pad_frac * span, hi + pad_frac * span + headroom_frac * span
    if lo > -zero_margin * (hi - lo):
        lo = -zero_margin * (hi - lo)
    lo = step * np.floor(lo / step)
    hi = step * np.ceil(hi / step)
    return (float(lo), float(hi))


def clip_report(sub, ylim):
    lo, hi = ylim
    w_lo = sub["coef"] - CI_MULT * sub["se"]
    w_hi = sub["coef"] + CI_MULT * sub["se"]
    return {
        "points_out": int(((sub["coef"] < lo) | (sub["coef"] > hi)).sum()),
        "whisker_lo_clipped": int((w_lo < lo).sum()),
        "whisker_hi_clipped": int((w_hi > hi).sum()),
    }


# ---------------------------------------------------------------- panel drawing
def draw_panel(ax, d, term, show_ylabel, show_yticklabels, title, xlabel):
    """One panel, styled exactly as plot_six_coeffs draws one."""
    term_color = TERM_COLOR[term]
    term_disp = TERM_LABEL[term]

    d = d.sort_values("prompt")
    x = d["prompt"].to_numpy()
    y = d["coef"].to_numpy()
    ci = CI_MULT * d["se"].to_numpy()
    yerr = np.vstack([ci, ci])

    mask0 = (x == 0)
    m = ~mask0

    if m.any():
        ax.errorbar(
            x[m], y[m], yerr=yerr[:, m],
            fmt="o", ms=DOT_MS, color=term_color, ecolor=term_color,
            elinewidth=1.6, capsize=3.5, capthick=1.6, alpha=0.95,
            zorder=3, label="Robustness Prompts + 90% CI",
        )
    if mask0.any():
        ax.errorbar(
            x[mask0], y[mask0], yerr=yerr[:, mask0],
            fmt="o", ms=ORIGINAL_MS, color=ORIGINAL_COLOR, ecolor=ORIGINAL_COLOR,
            elinewidth=1.8, capsize=3.5, capthick=1.8, alpha=1.0,
            zorder=5, label="Main Prompt + 90% CI",
        )

    y_mean = float(np.nanmean(y))
    ax.axhline(y_mean, color=term_color, linestyle="--", lw=2, alpha=0.9,
               label=f"Mean (across prompts) = {y_mean:.2f}")
    ax.axhline(0, color=ZERO_COLOR, linestyle="--", lw=1.6, alpha=0.85)

    if title:
        ax.set_title(title, fontsize=TITLE_FS)
    if show_ylabel:
        ax.set_ylabel(f"Estimated {term_disp} Coefficient", fontsize=LABEL_FS)
    else:
        ax.set_ylabel("")
    if not show_yticklabels:
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    return y_mean


# ---------------------------------------------------------------- figure 1: matched only
def figure_matched(df, ylims):
    d = df[(df["definition"] == DEFINITION) & (df["spec"] == "MATCHED")]

    fig, axes = plt.subplots(
        nrows=len(TERMS), ncols=len(MODELS),
        figsize=(6 * len(MODELS), 4.5 * len(TERMS)), sharex=True,
    )
    means = {}
    for r, term in enumerate(TERMS):
        for c, model in enumerate(MODELS):
            ax = axes[r, c]
            sub = d[(d["term"] == term) & (d["model"] == model)]
            means[(term, model)] = draw_panel(
                ax, sub, term,
                show_ylabel=(c == 0), show_yticklabels=(c == 0),
                title=MODEL_LABEL[model] if r == 0 else "",
                xlabel="" if r == 0 else "GPT Prompt",
            )
            ax.set_ylim(*ylims[term])
            ax.legend(loc="upper right", fontsize=LEGEND_FS)

    plt.tight_layout()
    path = _os.path.join(PLOTS, f"fragmentation_index_robustness_definition_{DEFINITION}_matched.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path, means


# ---------------------------------------------------------------- figure 2: comparison
SPEC_HEADER = {
    "OLD": "Published specification: exposure control = E1 share",
    "MATCHED": "Matched specification: exposure control = E1 or E2 share",
}


def figure_comparison(df, ylims):
    d = df[df["definition"] == DEFINITION]

    fig, axes = plt.subplots(
        nrows=len(TERMS), ncols=2 * len(MODELS),
        figsize=(5.5 * 2 * len(MODELS), 4.8 * len(TERMS)), sharex=True,
    )
    means = {}
    for r, term in enumerate(TERMS):
        for si, spec in enumerate(("OLD", "MATCHED")):
            for c, model in enumerate(MODELS):
                col = si * len(MODELS) + c
                ax = axes[r, col]
                sub = d[(d["term"] == term) & (d["spec"] == spec) & (d["model"] == model)]
                means[(spec, term, model)] = draw_panel(
                    ax, sub, term,
                    show_ylabel=(col == 0), show_yticklabels=(c == 0),
                    title=MODEL_LABEL[model] if r == 0 else "",
                    xlabel="" if r == 0 else "GPT Prompt",
                )
                ax.set_ylim(*ylims[term])
                ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout(rect=(0, 0, 1, 0.945))

    # Group headers over each half, plus a divider down the middle.
    left = axes[0, 0].get_position().x0
    mid_l = axes[0, len(MODELS) - 1].get_position().x1
    mid_r = axes[0, len(MODELS)].get_position().x0
    right = axes[0, -1].get_position().x1
    top = axes[0, 0].get_position().y1
    fig.text(0.5 * (left + mid_l), top + 0.055, SPEC_HEADER["OLD"],
             ha="center", va="bottom", fontsize=19, fontweight="bold")
    fig.text(0.5 * (mid_r + right), top + 0.055, SPEC_HEADER["MATCHED"],
             ha="center", va="bottom", fontsize=19, fontweight="bold")
    xmid = 0.5 * (mid_l + mid_r)
    fig.add_artist(Line2D([xmid, xmid], [0.02, top + 0.05],
                          color="#8a8a8a", lw=1.6, linestyle="-"))

    path = _os.path.join(PLOTS, f"fragmentation_index_robustness_definition_{DEFINITION}_comparison.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path, means


# ---------------------------------------------------------------- run
def main():
    df = load_results()
    d1 = df[df["definition"] == DEFINITION]
    matched = d1[d1["spec"] == "MATCHED"]

    # Limits for the matched-only figure: from the matched estimates.
    ylims_matched = {t: pick_ylim(matched[matched["term"] == t]) for t in TERMS}
    # Limits for the comparison: shared across both specs, so the shift is readable.
    ylims_both = {t: pick_ylim(d1[d1["term"] == t]) for t in TERMS}

    print("=" * 104)
    print("Y-AXIS LIMITS. Published call hardcodes YLIMS_BY_DEF[1] = "
          "{'ai_exposure': (-0.65, 0.65), 'fragmentation_index': (-0.65, 0.65)}.")
    for t in TERMS:
        pub = PUBLISHED_YLIMS[t]
        cr = clip_report(matched[matched["term"] == t], pub)
        print(f"  {t:20s} published {pub} on MATCHED data -> "
              f"points off-panel {cr['points_out']}, whiskers clipped "
              f"lo {cr['whisker_lo_clipped']} hi {cr['whisker_hi_clipped']}")
        for name, yl, sub in (("matched-only", ylims_matched[t], matched[matched["term"] == t]),
                              ("comparison  ", ylims_both[t], d1[d1["term"] == t])):
            cr2 = clip_report(sub, yl)
            print(f"    -> {name} limits {yl}  clipped: "
                  f"pts {cr2['points_out']}, lo {cr2['whisker_lo_clipped']}, "
                  f"hi {cr2['whisker_hi_clipped']}")

    p1, means1 = figure_matched(df, ylims_matched)
    p2, means2 = figure_comparison(df, ylims_both)

    print("\n" + "=" * 104)
    print("PLOTTED VALUES, Definition 1, MATCHED specification. "
          "coef (SE) [90% CI], clustered on O*NET-SOC code.")
    for term in ("fragmentation_index", "ai_exposure"):
        print(f"\n--- {TERM_LABEL[term]} ({term}) ---")
        header = f"{'prompt':>6s} |"
        for model in MODELS:
            header += f"{MODEL_LABEL[model]:>34s} |"
        print(header)
        for prompt in range(11):
            line = f"{prompt:>6d} |"
            for model in MODELS:
                r = matched[(matched["term"] == term) & (matched["model"] == model)
                            & (matched["prompt"] == prompt)].iloc[0]
                lo = r["coef"] - CI_MULT * r["se"]
                hi = r["coef"] + CI_MULT * r["se"]
                line += f" {r['coef']:+.4f} ({r['se']:.4f}) [{lo:+.3f},{hi:+.3f}] |"
            print(line)
        line = f"{'mean':>6s} |"
        for model in MODELS:
            line += f" {means1[(term, model)]:+.4f} {'':>25s}|"
        print(line)

    print("\nwrote:", p1)
    print("wrote:", p2)

    # Verify the PNGs.
    try:
        from PIL import Image
        for p in (p1, p2):
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                print(f"  {_os.path.basename(p)}: {im.format} {im.size[0]}x{im.size[1]} px, "
                      f"dpi {im.info.get('dpi')}, {_os.path.getsize(p)/1e6:.2f} MB")
    except ImportError:
        print("  (PIL not available, skipping image verification)")


if __name__ == "__main__":
    main()
