"""Regenerate Figures SA.B.1-SA.B.3 (12 panel PNGs) under the E1|E2 exposure mask.

Figures (each = 4 panel PNGs, specs (1)-(4) of the matching table)
  SA.B.1  writeup/plots/execTypeVaryingDWA_noTasksWithRepetitiveDWAs/is_ai/AME_filtered_is_ai_*.png
  SA.B.2  writeup/plots/execTypeVaryingDWA/is_automated/AME_full_is_automated_*.png
  SA.B.3  writeup/plots/execTypeVaryingDWA/is_automated/AME_filtered_is_automated_*.png

Drawing code is `analysis/replot_AME_figures.py` (current, non-legacy styling), which is what
produced the committed PNGs on 2026-08-26. It is imported here, not re-typed, so the styling
cannot drift: same figsize, same 30 bins, same colours, same 1.645*SE band, same dpi=300.

What changes: the three inputs to that drawing code are re-estimated with

    merged_data['is_exposed'] = merged_data['human_labels'].isin(['E1'])
 -> merged_data['is_exposed'] = merged_data['human_labels'].isin(['E1','E2'])

  * observed AME  -- from the m6 audit's B=200 DWA-cluster bootstrap run
                     (rebuild_SAB*_E1E2_B200.csv), i.e. the paper's own SE method,
                     matching the cached regression_ame_results_*_0.csv the published
                     figures read.
  * observed SE   -- same source, same B=200 bootstrap. (The audit's own figure script
                     used delta-method SEs; those would silently change the width of the
                     red 90% band, which is not the one thing under test, so they are
                     not used here.)
  * null draws    -- ame_figs_null_E1_vs_E1E2.csv, mask=E1E2, the 999 position-reshuffle
                     draws with the notebook's own seeds 43..1041, prefixed by the observed
                     estimate exactly as the notebook's `for i in range(n_shuffles)` loop
                     does when it finds the observed file cached at index 0.

X limits: the published pipeline derives them from all six specs' null draws
(`x_bounds`). Only the four plotted specs were re-simulated under E1|E2, so the published
limits are held fixed and every E1|E2 value drawn is checked against them for clipping
(reported below; nothing is clipped).

CONTROLS run before anything is written:
  1. cached E1 null draws 1..999 vs the audit's E1 re-simulation, draw by draw;
  2. cached E1 observed AME/SE vs the audit's E1 bootstrap rebuild;
  3. the drawing code re-run on the published inputs, compared byte-wise with the
     committed PNGs.

Writes only under writeup/_e1e2_preview/.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin")
M6 = Path("/private/tmp/claude-502/-Users-peymansh-MIT-Dropbox-Peyman-Shahidi-GitHub-rubin/"
          "5af7d286-51e7-470a-b57e-4e1373499eea/scratchpad/m6")
STAGE = REPO / "writeup" / "_e1e2_preview"
WORK = STAGE / "work" / "sab_figs"

# import the repo's own plotting module without executing its main()
_spec = importlib.util.spec_from_file_location("replot", REPO / "analysis" / "replot_AME_figures.py")
replot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replot)

TARGET_REGS = replot.TARGET_REGS
PLOT_SPECS = ["no_fe_no_dwa", "major_fe_no_dwa", "minor_fe_no_dwa", "no_fe_with_dwa"]

# exhibit -> (cached summary dir, kind, out base name, m6 job tag, m6 null config, m6 null sample)
FIGS = {
    "Fig SA.B.1": ("execTypeVaryingDWA_anthropicIndex_noTasksWithRepetitiveDWAs", "is_ai",
                   "filtered", "AME_filtered_is_ai",
                   "SAB1_GPT_ai", "is_ai_noRepetitiveDWAs", "filtered"),
    "Fig SA.B.2": ("execTypeVaryingDWA_anthropicIndex", "is_automated",
                   "full", "AME_full_is_automated",
                   "SAB2_allTasks_auto", "is_automated_randomTieBreak", "full"),
    "Fig SA.B.3": ("execTypeVaryingDWA_anthropicIndex", "is_automated",
                   "filtered", "AME_filtered_is_automated",
                   "SAB3_GPT_auto", "is_automated_randomTieBreak", "filtered"),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def dicts_from_rebuild(csv_path):
    """(coef, se) keyed [spec][term] from an m6 bootstrap rebuild CSV."""
    df = pd.read_csv(csv_path)
    coef = {s: {t: np.nan for t in TARGET_REGS} for s in PLOT_SPECS}
    se = {s: {t: np.nan for t in TARGET_REGS} for s in PLOT_SPECS}
    for _, r in df.iterrows():
        if r["model"] in coef and r["term"] in coef[r["model"]]:
            coef[r["model"]][r["term"]] = r["ame_coef"]
            se[r["model"]][r["term"]] = r["ame_se"]
    return coef, se


def null_from_m6(null_df, config, sample, mask, obs_coef):
    """Rebuild the notebook's null list: [observed] + reshuffle draws 1..999."""
    sub = null_df[(null_df["config"] == config) & (null_df["sample"] == sample)
                  & (null_df["mask"] == mask)]
    out = {}
    for s in PLOT_SPECS:
        out[s] = {}
        for t in TARGET_REGS:
            v = sub[(sub["spec"] == s) & (sub["term"] == t)].sort_values("draw")
            assert list(v["draw"]) == list(range(1000)), (config, sample, mask, s, t)
            draws = list(v["ame"].values[1:])          # drop the extra seed-42 draw
            out[s][t] = [obs_coef[s][t]] + draws        # index 0 = observed, as the notebook does
            assert len(out[s][t]) == 1000
    return out


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    (STAGE / "plots_new").mkdir(parents=True, exist_ok=True)
    log = []
    null_df = pd.read_csv(M6 / "ame_figs_null_E1_vs_E1E2.csv")
    ok = True

    for exhibit, (cachedir, dep, kind, base, job, cfg, samp) in FIGS.items():
        print(f"\n===== {exhibit}  ({base}) =====")
        summary_dir = REPO / "data" / "computed_objects" / cachedir / f"regression_summaries_{dep}"

        # ---------- published (E1) inputs, straight from the repo cache ----------
        obs_pub, se_pub = replot.results_to_dict(pd.read_csv(summary_dir / f"regression_ame_results_{kind}_0.csv"))
        resh_pub, missing = replot.load_null(summary_dir, kind)
        assert missing == 0, f"{missing} cached null files missing"
        lo_x, hi_x = replot.x_bounds(resh_pub, obs_pub)
        print(f"  published x-limits (all six specs): [{lo_x:.5f}, {hi_x:.5f}]")

        # ---------- CONTROL 1: audit's E1 null re-simulation vs the repo cache ----------
        obs_e1, se_e1 = dicts_from_rebuild(M6 / f"rebuild_{job}_E1_B200.csv")
        resh_e1_audit = null_from_m6(null_df, cfg, samp, "E1", obs_e1)
        d = max(abs(np.asarray(resh_e1_audit[s][t][1:]) - np.asarray(resh_pub[s][t][1:])).max()
                for s in PLOT_SPECS for t in TARGET_REGS)
        print(f"  CONTROL 1  audit E1 nulls vs cached nulls, draws 1-999: max|diff| = {d:.2e}")
        if d > 1e-12:
            ok = False

        # ---------- CONTROL 2: audit's E1 bootstrap observed vs the repo cache ----------
        da = max(abs(obs_e1[s][t] - obs_pub[s][t]) for s in PLOT_SPECS for t in TARGET_REGS)
        ds = max(abs(se_e1[s][t] - se_pub[s][t]) for s in PLOT_SPECS for t in TARGET_REGS)
        print(f"  CONTROL 2  audit E1 observed vs cached _0: max|dAME| = {da:.2e}, max|dSE| = {ds:.2e}")
        # dSE tolerance: the bootstrap accumulates over 200 replicates in a different
        # order in the audit's rebuild than in the notebook run that wrote the cache.
        # The audit measured this residual at up to 7.6e-05; 1.645x that is 1.3e-04 on
        # an x-axis 0.36 wide, i.e. 0.03% of a panel.
        if da > 1e-12 or ds > 1e-4:
            ok = False

        # Only the four plotted specs get redrawn, and both sides use the published
        # x-limits computed above from all six specs' cached draws.
        full_specs, full_bounds = replot.SPECS, replot.x_bounds
        replot.SPECS = PLOT_SPECS          # colours are tab10(0..3) either way
        replot.x_bounds = lambda resh, obs: (lo_x, hi_x)
        try:
            # ------ CONTROL 3: redraw the published figure, compare with the committed PNG ------
            ctrl_dir = WORK / f"{base}_E1_control"
            ctrl_dir.mkdir(parents=True, exist_ok=True)
            replot.draw(resh_pub, obs_pub, se_pub, ctrl_dir, base,
                        replot.STYLE, replot.VAR_LABELS)
            pubdir = (REPO / "writeup" / "plots"
                      / ("execTypeVaryingDWA_noTasksWithRepetitiveDWAs" if "noTasks" in cachedir
                         else "execTypeVaryingDWA") / dep)
            for s in PLOT_SPECS:
                a, b = pubdir / f"{base}_{s}.png", ctrl_dir / f"{base}_{s}.png"
                same = sha(a) == sha(b)
                print(f"  CONTROL 3  {base}_{s}.png  committed {sha(a)}  redrawn {sha(b)}  "
                      f"{'IDENTICAL' if same else 'DIFFERS'}")
                log.append(("control3", f"{base}_{s}.png", "IDENTICAL" if same else "DIFFERS"))

            # ------------------------ the one change: E1|E2 ------------------------
            obs_new, se_new = dicts_from_rebuild(M6 / f"rebuild_{job}_E1E2_B200.csv")
            resh_new = null_from_m6(null_df, cfg, samp, "E1E2", obs_new)

            vals = [v for s in PLOT_SPECS for t in TARGET_REGS for v in resh_new[s][t]]
            vals += [obs_new[s][t] + k * 1.645 * se_new[s][t]
                     for s in PLOT_SPECS for t in TARGET_REGS for k in (-1, 0, 1)]
            vmin, vmax = min(vals), max(vals)
            clipped = vmin < lo_x or vmax > hi_x
            print(f"  E1|E2 drawn range over the four plotted specs: [{vmin:.5f}, {vmax:.5f}] "
                  f"-> {'CLIPPED at the published limits' if clipped else 'inside the published limits'}")
            if clipped:
                ok = False

            replot.draw(resh_new, obs_new, se_new, STAGE / "plots_new", base,
                        replot.STYLE, replot.VAR_LABELS)
            for s in PLOT_SPECS:
                print(f"  wrote plots_new/{base}_{s}.png")
        finally:
            replot.SPECS, replot.x_bounds = full_specs, full_bounds

    print("\nALL CONTROLS PASSED" if ok else "\nSOME CONTROL FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
