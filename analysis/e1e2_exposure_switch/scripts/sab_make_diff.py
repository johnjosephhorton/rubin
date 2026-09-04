"""Write the per-exhibit numeric diff for the SA.B group into diffs/sab-exhibits.txt.

Tables: every one of the 72 printed coefficient cells, its printed SE, its stars, plus the
        pseudo-R2 and N footer rows, published (E1) vs regenerated (E1|E2).
Figures: every one of the 48 plotted panel cells, the observed AME, the bootstrap SE that
        sets the red 90% band, the band edges, and the observed value's percentile in its
        own null.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin")
M6 = Path("/private/tmp/claude-502/-Users-peymansh-MIT-Dropbox-Peyman-Shahidi-GitHub-rubin/"
          "5af7d286-51e7-470a-b57e-4e1373499eea/scratchpad/m6")
STAGE = REPO / "writeup" / "_e1e2_preview"

TERMS = ['prev2_is_ai', 'prev_is_ai', 'next_is_ai', 'next2_is_ai']
ROW = {'prev2_is_ai': 'Task (k-2)', 'prev_is_ai': 'Task (k-1)',
       'next_is_ai': 'Task (k+1)', 'next2_is_ai': 'Task (k+2)'}
MODELS = ['no_fe_no_dwa', 'major_fe_no_dwa', 'minor_fe_no_dwa', 'no_fe_with_dwa',
          'no_fe_no_dwa_withTaskDWACount', 'no_fe_with_dwa_withTaskDWACount']
PLOT_SPECS = MODELS[:4]
PANEL = dict(zip(PLOT_SPECS, "abcd"))

TABLES = [
    ("Table SA.B.1", "SAB1_GPT_ai", "writeup/tables/noTasksWithRepetitiveDWAs/GPT_ai.tex", "GPT_ai.tex"),
    ("Table SA.B.2", "SAB2_allTasks_auto", "writeup/tables/randomTieBreak/allTasks_automated.tex", "allTasks_automated.tex"),
    ("Table SA.B.3", "SAB3_GPT_auto", "writeup/tables/randomTieBreak/GPT_automated.tex", "GPT_automated.tex"),
]
FIGS = [
    ("Fig SA.B.1", "SAB1_GPT_ai", "is_ai_noRepetitiveDWAs", "filtered", "AME_filtered_is_ai",
     "writeup/plots/execTypeVaryingDWA_noTasksWithRepetitiveDWAs/is_ai/"),
    ("Fig SA.B.2", "SAB2_allTasks_auto", "is_automated_randomTieBreak", "full", "AME_full_is_automated",
     "writeup/plots/execTypeVaryingDWA/is_automated/"),
    ("Fig SA.B.3", "SAB3_GPT_auto", "is_automated_randomTieBreak", "filtered", "AME_filtered_is_automated",
     "writeup/plots/execTypeVaryingDWA/is_automated/"),
]


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def cell(r):
    return f"{r['ame_coef']:.2f}{stars(r['p_value'])} ({r['ame_se']:.2f})"


L = []
w = L.append

w("Per-exhibit numeric diff: SA.B tables and figures, E1-only -> E1|E2")
w("=" * 100)
w("")
w("The single change is the exposure mask in the focal task's `is_exposed` control:")
w("    merged_data['is_exposed'] = merged_data['human_labels'].isin(['E1'])")
w(" -> merged_data['is_exposed'] = merged_data['human_labels'].isin(['E1','E2'])")
w("Samples, formulas, DWA clustering, B=200 cluster bootstrap, seed 123, the 999 position-reshuffle")
w("draws (seeds 43..1041) and every plotting parameter are unchanged.")
w("")
w("Controls, all passed before anything below was written:")
w("  * the three .tex bodies regenerated from the E1 estimates are byte-identical to the")
w("    committed files, including the commit-228864c 'AI-automated' spanner;")
w("  * the twelve committed PNGs are reproduced byte-identically (sha256) by re-running")
w("    analysis/replot_AME_figures.py on the repo's cached E1 inputs;")
w("  * the E1 re-simulation of the null reproduces all 999 cached draws to <= 2.1e-16.")
w("")

# ----------------------------------------------------------------- TABLES
for exhibit, job, relpath, name in TABLES:
    e1 = pd.read_csv(M6 / f"rebuild_{job}_E1_B200.csv").set_index(['model', 'term'])
    e2 = pd.read_csv(M6 / f"rebuild_{job}_E1E2_B200.csv").set_index(['model', 'term'])
    w("")
    w("=" * 100)
    w(f"{exhibit}   {relpath}")
    w(f"  current: tables_current/{name}     regenerated: tables_new/{name}")
    w("=" * 100)
    w(f"{'row':<12} {'col':<5} {'published (E1)':<20} {'regenerated (E1|E2)':<20} "
      f"{'d(AME)':>9} {'d(SE)':>9}  changed")
    w("-" * 100)
    nch = ncoef = nse = nstar = nsign = 0
    for t in TERMS:
        for i, m in enumerate(MODELS, start=1):
            a, b = e1.loc[(m, t)], e2.loc[(m, t)]
            ca, cb = cell(a), cell(b)
            flags = []
            if f"{a['ame_coef']:.2f}" != f"{b['ame_coef']:.2f}":
                flags.append("coef"); ncoef += 1
            if stars(a['p_value']) != stars(b['p_value']):
                flags.append("STARS"); nstar += 1
            if f"{a['ame_se']:.2f}" != f"{b['ame_se']:.2f}":
                flags.append("SE"); nse += 1
            if np.sign(a['ame_coef']) != np.sign(b['ame_coef']):
                flags.append("SIGN"); nsign += 1
            if flags:
                nch += 1
            w(f"{ROW[t]:<12} ({i})   {ca:<20} {cb:<20} "
              f"{b['ame_coef']-a['ame_coef']:>+9.4f} {b['ame_se']-a['ame_se']:>+9.4f}  "
              f"{','.join(flags)}")
    w("-" * 100)
    st1 = e1.reset_index().drop_duplicates('model').set_index('model')
    st2 = e2.reset_index().drop_duplicates('model').set_index('model')
    w("Pseudo R2   E1   : " + "  ".join(f"{st1.loc[m,'r2_pseudo']:.3f}" for m in MODELS))
    w("Pseudo R2   E1|E2: " + "  ".join(f"{st2.loc[m,'r2_pseudo']:.3f}" for m in MODELS)
      + "   <- " + ("changes in cols " + ", ".join(
          f"({i})" for i, m in enumerate(MODELS, 1)
          if f"{st1.loc[m,'r2_pseudo']:.3f}" != f"{st2.loc[m,'r2_pseudo']:.3f}") or "no change"))
    w("Observations E1  : " + "  ".join(f"{int(st1.loc[m,'nobs']):,}" for m in MODELS))
    w("Observations E1|E2: " + "  ".join(f"{int(st2.loc[m,'nobs']):,}" for m in MODELS)
      + "   <- " + ("IDENTICAL" if all(st1.loc[m, 'nobs'] == st2.loc[m, 'nobs'] for m in MODELS)
                    else "MOVED"))
    w(f"Summary: {nch}/24 printed cells change; coefficient {ncoef}, printed SE {nse}, "
      f"stars {nstar}, sign flips {nsign}.")

# ----------------------------------------------------------------- FIGURES
null_df = pd.read_csv(M6 / "ame_figs_null_E1_vs_E1E2.csv")
for exhibit, job, cfg, samp, base, plotdir in FIGS:
    e1 = pd.read_csv(M6 / f"rebuild_{job}_E1_B200.csv").set_index(['model', 'term'])
    e2 = pd.read_csv(M6 / f"rebuild_{job}_E1E2_B200.csv").set_index(['model', 'term'])
    w("")
    w("=" * 100)
    w(f"{exhibit}   {plotdir}{base}_<spec>.png")
    w("   panels (a) no_fe_no_dwa  (b) major_fe_no_dwa  (c) minor_fe_no_dwa  (d) no_fe_with_dwa")
    w("   red dashed line = observed AME, red band = observed +/- 1.645 x bootstrap SE,")
    w("   histogram = 999 position-reshuffle draws plus the observed value at index 0.")
    w("=" * 100)
    w(f"{'panel':<7} {'term':<12} {'obs E1':>9} {'obs E1|E2':>10} {'d':>9} | "
      f"{'SE E1':>8} {'SE E1|E2':>9} | {'band E1':>17} {'band E1|E2':>17} | {'pct E1':>7} {'pct E1|E2':>9}")
    w("-" * 130)
    maxd = 0.0
    for s in PLOT_SPECS:
        for t in TERMS:
            a, b = e1.loc[(s, t)], e2.loc[(s, t)]
            n1 = null_df[(null_df.config == cfg) & (null_df['sample'] == samp)
                         & (null_df['mask'] == 'E1') & (null_df.spec == s)
                         & (null_df.term == t)].sort_values('draw')['ame'].values
            n2 = null_df[(null_df.config == cfg) & (null_df['sample'] == samp)
                         & (null_df['mask'] == 'E1E2') & (null_df.spec == s)
                         & (null_df.term == t)].sort_values('draw')['ame'].values
            d1 = np.concatenate([[a['ame_coef']], n1[1:]])
            d2 = np.concatenate([[b['ame_coef']], n2[1:]])
            p1 = 100.0 * np.mean(d1 < a['ame_coef'])
            p2 = 100.0 * np.mean(d2 < b['ame_coef'])
            l1, h1 = a['ame_coef'] - 1.645 * a['ame_se'], a['ame_coef'] + 1.645 * a['ame_se']
            l2, h2 = b['ame_coef'] - 1.645 * b['ame_se'], b['ame_coef'] + 1.645 * b['ame_se']
            maxd = max(maxd, abs(b['ame_coef'] - a['ame_coef']))
            w(f"({PANEL[s]})     {ROW[t]:<12} {a['ame_coef']:>9.4f} {b['ame_coef']:>10.4f} "
              f"{b['ame_coef']-a['ame_coef']:>+9.4f} | {a['ame_se']:>8.4f} {b['ame_se']:>9.4f} | "
              f"[{l1:>+7.4f},{h1:>+7.4f}] [{l2:>+7.4f},{h2:>+7.4f}] | {p1:>7.1f} {p2:>9.1f}")
    w("-" * 130)
    w(f"Printed legend label 'Observed = x.xxx' changes in: " + ", ".join(
        f"({PANEL[s]}) {ROW[t]}"
        for s in PLOT_SPECS for t in TERMS
        if f"{e1.loc[(s,t),'ame_coef']:.3f}" != f"{e2.loc[(s,t),'ame_coef']:.3f}") or "  none")
    w(f"Largest observed-AME shift over the 16 panel cells: {maxd:.4f}")
    w("X limits held at the published values (they are derived by the pipeline from all six")
    w("specs' draws; only the four plotted specs were re-simulated). Nothing drawn is clipped:")
    w("  SA.B.1 [-0.18145, +0.18145], drawn [-0.08296, +0.17875]")
    w("  SA.B.2 [-0.18108, +0.18108], drawn [-0.07213, +0.10938]")
    w("  SA.B.3 [-0.18741, +0.18741], drawn [-0.10585, +0.15164]")

out = STAGE / "diffs" / "sab-exhibits.txt"
out.write_text("\n".join(L) + "\n")
print("wrote", out, len(L), "lines")
