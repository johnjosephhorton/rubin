"""Regenerate the three SA.B Prediction-#2 tables under the E1|E2 exposure mask.

Tables
  SA.B.1  writeup/tables/noTasksWithRepetitiveDWAs/GPT_ai.tex        (is_ai,        GPT-filtered, noRepetitiveDWA)
  SA.B.2  writeup/tables/randomTieBreak/allTasks_automated.tex       (is_automated, all tasks,    randomTieBreak)
  SA.B.3  writeup/tables/randomTieBreak/GPT_automated.tex            (is_automated, GPT-filtered, randomTieBreak)

Inputs are the audited estimation output in the m6 scratchpad (`rebuild_SAB*_{E1,E1E2}_B200.csv`),
produced by `pred2_build.py`/`pred2_estimate.py`, which copy cells 9-14 and 17 of
analysis/onet_antrhopicIndex_execTypeVaryingDWA.ipynb verbatim and change exactly one line:

    merged_data['is_exposed'] = merged_data['human_labels'].isin(['E1'])
 -> merged_data['is_exposed'] = merged_data['human_labels'].isin(['E1','E2'])

Same samples, same six formulas, same DWA-clustered B=200 bootstrap, seed 123.

The LaTeX emitter below is cell 17's generate_latex_table(), minus the trailing
\footnotesize{...} line that the committed .tex bodies do not carry, and with the
column spanner keyed on the outcome as fixed in commit 228864c.

CONTROL: the same emitter is run on the E1 estimates and the result is diffed
against the committed file. It must come back byte-identical.

Writes only under writeup/_e1e2_preview/.
"""
from pathlib import Path
import filecmp
import sys

import pandas as pd

M6 = Path("/private/tmp/claude-502/-Users-peymansh-MIT-Dropbox-Peyman-Shahidi-GitHub-rubin/"
          "5af7d286-51e7-470a-b57e-4e1373499eea/scratchpad/m6")
REPO = Path("/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin")
STAGE = REPO / "writeup" / "_e1e2_preview"

TARGET_REGS = ['prev2_is_ai', 'prev_is_ai', 'next_is_ai', 'next2_is_ai']
TABLE_VAR_LABELS = {
    'prev2_is_ai': 'Task ($k-2$) is AI-executed',
    'prev_is_ai': 'Task ($k-1$) is AI-executed',
    'next_is_ai': 'Task ($k+1$) is AI-executed',
    'next2_is_ai': 'Task ($k+2$) is AI-executed',
}
MODEL_ORDER = ['no_fe_no_dwa', 'major_fe_no_dwa', 'minor_fe_no_dwa', 'no_fe_with_dwa',
               'no_fe_no_dwa_withTaskDWACount', 'no_fe_with_dwa_withTaskDWACount']

# job -> (published .tex path relative to writeup/tables, dataset comment, spanner outcome word)
JOBS = {
    "SAB1_GPT_ai":        ("noTasksWithRepetitiveDWAs/GPT_ai.tex",  "filtered_0", "AI-executed",  "Table SA.B.1"),
    "SAB2_allTasks_auto": ("randomTieBreak/allTasks_automated.tex", "full_0",     "AI-automated", "Table SA.B.2"),
    "SAB3_GPT_auto":      ("randomTieBreak/GPT_automated.tex",      "filtered_0", "AI-automated", "Table SA.B.3"),
}


def generate_latex_table(df, dataset_to_show, dep_label):
    """cell 17's generate_latex_table(), without the trailing \\footnotesize line."""
    buf = []
    w = buf.append
    subset = df.copy()
    w(f"% --- LaTeX Table for {dataset_to_show} ---")

    def fmt(row):
        p = row['p_value']
        st = ""
        if pd.notna(p):
            st = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        return f"{row['ame_coef']:.2f}{st}", f"({row['ame_se']:.2f})"

    f = subset.apply(fmt, axis=1, result_type='expand')
    subset['coef_str'], subset['se_str'] = f[0], f[1]
    pc = subset.pivot(index='term', columns='model', values='coef_str').reindex(TARGET_REGS)
    ps = subset.pivot(index='term', columns='model', values='se_str').reindex(TARGET_REGS)
    vm = [m for m in MODEL_ORDER if m in pc.columns]
    stats = subset[['model', 'nobs', 'r2_pseudo', 'fe_label', 'dwa_fe']] \
        .drop_duplicates('model').set_index('model')

    w("\\begin{tabular}{l" + "c" * len(vm) + "}")
    w(r"\toprule")
    w(" & " + r"\multicolumn{" + str(len(vm)) + r"}{c}{Probability that Focal Task ($k$) is "
      + dep_label + r"} \\")
    w(r"\cmidrule(lr){2-" + str(len(vm) + 1) + "}")
    w(" & " + " & ".join(f"({i+1})" for i in range(len(vm))) + r" \\")
    w(r"\midrule")
    for v in TARGET_REGS:
        w(f"{TABLE_VAR_LABELS[v]} & " + " & ".join(pc.loc[v, m] for m in vm) + r" \\")
        w(" & " + " & ".join(ps.loc[v, m] for m in vm) + r" \\")
        w(r"\addlinespace")
    w(r"\midrule")
    w("Pseudo $R^2$ & " + " & ".join(f"{stats.loc[m, 'r2_pseudo']:.3f}" for m in vm) + r" \\")
    w("Observations & " + " & ".join(f"{int(stats.loc[m, 'nobs']):,}" for m in vm) + r" \\")
    fe = []
    for m in vm:
        lab = str(stats.loc[m, 'fe_label'])
        fe.append("Major" if "Major" in lab else "Minor" if "Minor" in lab else "")
    w("SOC Group FE & " + " & ".join(fe) + r" \\")
    w("DWA FE & " + " & ".join(r"\checkmark" if stats.loc[m, 'dwa_fe'] else "" for m in vm) + r" \\")
    w("NumTasks in DWA-Occupation Control & " +
      " & ".join(r"\checkmark" if "withTaskDWACount" in m else "" for m in vm) + r" \\")
    w(r"\bottomrule")
    w(r"\end{tabular}")
    return "\n".join(buf) + "\n"


def main():
    ok = True
    control_dir = STAGE / "work" / "sab_tables_E1_control"
    control_dir.mkdir(parents=True, exist_ok=True)

    for job, (relpath, dataset, dep_label, exhibit) in JOBS.items():
        name = Path(relpath).name
        published = REPO / "writeup" / "tables" / relpath

        # ---- control: rebuild the published (E1) side and demand byte identity ----
        e1 = pd.read_csv(M6 / f"rebuild_{job}_E1_B200.csv")
        e1_tex = control_dir / name
        e1_tex.write_text(generate_latex_table(e1, dataset, dep_label))
        identical = filecmp.cmp(str(e1_tex), str(published), shallow=False)
        print(f"[{exhibit}] E1 control vs {relpath}: "
              f"{'IDENTICAL' if identical else 'DIFFERS'}")
        if not identical:
            ok = False
            import difflib
            a = published.read_text().splitlines()
            b = e1_tex.read_text().splitlines()
            for line in difflib.unified_diff(a, b, 'published', 'E1 rebuild', lineterm=''):
                print("   ", line)

        # ---- the one change: E1|E2 ----
        e12 = pd.read_csv(M6 / f"rebuild_{job}_E1E2_B200.csv")
        out = STAGE / "tables_new" / name
        out.write_text(generate_latex_table(e12, dataset, dep_label))
        print(f"[{exhibit}] wrote {out}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
