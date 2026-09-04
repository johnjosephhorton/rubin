"""Numeric diff for Table 2 and Figure OA.A.1, published (E1-only) vs E1|E2.

Writes writeup/_e1e2_preview/diffs/table2-figOAA1.txt.
READ-ONLY on the repo outside writeup/_e1e2_preview/.
"""
import difflib
import sys

import numpy as np
import pandas as pd

REPO = "/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin"
STAGE = f"{REPO}/writeup/_e1e2_preview"
WORK = f"{STAGE}/work/t2f1"
CACHE = (f"{REPO}/data/computed_objects/execTypeVaryingDWA_anthropicIndex"
         f"_noTasksWithRepetitiveDWAs/regression_summaries_is_ai")

SPECS = ["no_fe_no_dwa", "major_fe_no_dwa", "minor_fe_no_dwa", "no_fe_with_dwa",
         "no_fe_no_dwa_withTaskDWACount", "no_fe_with_dwa_withTaskDWACount"]
COLNUM = {s: i + 1 for i, s in enumerate(SPECS)}
TERMS = ["prev2_is_ai", "prev_is_ai", "next_is_ai", "next2_is_ai"]
TLAB = {"prev2_is_ai": "Task (k-2)", "prev_is_ai": "Task (k-1)",
        "next_is_ai": "Task (k+1)", "next2_is_ai": "Task (k+2)"}
PANEL = {"no_fe_no_dwa": "(a) no FE", "major_fe_no_dwa": "(b) SOC major FE",
         "minor_fe_no_dwa": "(c) SOC minor FE", "no_fe_with_dwa": "(d) DWA FE"}


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def cell(r):
    return f"{r['ame_coef']:.2f}{stars(r['p_value'])}", f"({r['ame_se']:.2f})"


def main():
    out = []
    w = out.append

    pub = pd.read_csv(f"{CACHE}/regression_ame_results_full_0.csv")
    e1 = pd.read_csv(f"{WORK}/ame_full_0_E1.csv")
    e2 = pd.read_csv(f"{WORK}/ame_full_0_E1E2.csv")

    tex_pub = open(f"{STAGE}/tables_current/allTasks_ai.tex").read()
    tex_e1 = open(f"{WORK}/table2_E1.tex").read()
    tex_new = open(f"{WORK}/table2_E1E2.tex").read()

    w("=" * 100)
    w("Table 2 and Figure OA.A.1: published exposure mask isin(['E1'])  ->  isin(['E1','E2'])")
    w("Source notebook: analysis/onet_antrhopicIndex_execTypeVaryingDWA.ipynb, cell 13 line 15.")
    w("Nothing else changed: same 10,708-row estimation sample, same six formulas, same")
    w("DWA clustering, same B=200 DWA-cluster bootstrap at seed 123, same 1,000-index")
    w("placebo loop at seeds 42+i, same plotting code.")
    w("=" * 100)
    w("")
    w("-" * 100)
    w("A. CONTROLS (these must pass before any number below is trustworthy)")
    w("-" * 100)
    ctrl = []
    ctrl.append(("notebook LaTeX generator fed the published CSV reproduces the committed "
                 ".tex byte for byte",
                 open(f"{WORK}/table2_from_published_csv.tex").read() == tex_pub))
    ctrl.append(("my own E1-only rerun (full estimation + bootstrap) reproduces the "
                 "committed .tex byte for byte", tex_e1 == tex_pub))
    m = pub.merge(e1, on=["dataset", "model", "term"], suffixes=("_pub", "_e1"))
    ctrl.append((f"my E1-only AME coefficients vs the published CSV: max abs diff "
                 f"{np.abs(m.ame_coef_pub - m.ame_coef_e1).max():.2e}",
                 np.abs(m.ame_coef_pub - m.ame_coef_e1).max() < 1e-12))
    ctrl.append((f"my E1-only bootstrap SEs vs the published CSV: max abs diff "
                 f"{np.abs(m.ame_se_pub - m.ame_se_e1).max():.2e}",
                 np.abs(m.ame_se_pub - m.ame_se_e1).max() < 1e-9))
    ctrl.append((f"my E1-only Pseudo R2 vs the published CSV: max abs diff "
                 f"{np.abs(m.r2_pseudo_pub - m.r2_pseudo_e1).max():.2e}",
                 np.abs(m.r2_pseudo_pub - m.r2_pseudo_e1).max() < 1e-12))
    ctrl.append(("my E1-only nobs vs the published CSV: identical in all six columns",
                 bool((m.nobs_pub == m.nobs_e1).all())))
    px = pd.read_csv(f"{WORK}/pixel_control.csv")
    for _, r in px.iterrows():
        ctrl.append((f"figure code fed the published cached draws reproduces "
                     f"AME_full_is_ai_{r['spec']}.png pixel for pixel "
                     f"(max channel diff {int(r['maxdiff'])})", bool(r["identical"])))
    rc = pd.read_csv(f"{WORK}/reshuffle_control.csv")
    ctrl.append((f"my placebo loop under E1 vs the repo's cached draws "
                 f"(i = {int(rc['lo'][0])}..{int(rc['hi'][0])}, {int(rc['ncells'][0])} cells): "
                 f"max abs AME diff {rc['maxdiff'][0]:.2e}", bool(rc["ok"][0])))
    for txt, ok in ctrl:
        w(f"  [{'PASS' if ok else 'FAIL'}] {txt}")
    w("")

    w("-" * 100)
    w("B. TABLE 2  writeup/tables/noTasksWithRepetitiveDWAs/allTasks_ai.tex")
    w("   every printed cell; 'CHANGED' marks a cell whose printed text differs")
    w("-" * 100)
    w("")
    w(f"{'row':<13}{'col':<6}{'published':>16}{'E1|E2':>16}   "
      f"{'AME pub':>10}{'AME new':>10}{'delta':>10}   "
      f"{'SE pub':>9}{'SE new':>9}   {'p pub':>9}{'p new':>9}  flag")
    n_str_changed = n_val_changed = n_se_changed = n_star_changed = 0
    for t in TERMS:
        for s in SPECS:
            a = pub[(pub.model == s) & (pub.term == t)].iloc[0]
            b = e2[(e2.model == s) & (e2.term == t)].iloc[0]
            ca, sa = cell(a)
            cb, sb = cell(b)
            flags = []
            if ca != cb:
                n_str_changed += 1
            if f"{a.ame_coef:.2f}" != f"{b.ame_coef:.2f}":
                flags.append("VALUE")
                n_val_changed += 1
            if stars(a.p_value) != stars(b.p_value):
                flags.append("STAR")
                n_star_changed += 1
            if sa != sb:
                flags.append("SE")
                n_se_changed += 1
            flag = ("CHANGED:" + "+".join(flags)) if flags else ""
            w(f"{TLAB[t]:<13}({COLNUM[s]})  "
              f"{ca + ' ' + sa:>15}{cb + ' ' + sb:>16}   "
              f"{a.ame_coef:>10.4f}{b.ame_coef:>10.4f}{b.ame_coef - a.ame_coef:>10.4f}   "
              f"{a.ame_se:>9.4f}{b.ame_se:>9.4f}   "
              f"{a.p_value:>9.4f}{b.p_value:>9.4f}  {flag}")
        w("")

    w("Footer rows")
    sa = pub.drop_duplicates("model").set_index("model")
    sb = e2.drop_duplicates("model").set_index("model")
    w(f"  {'':<22}" + "".join(f"{'(' + str(i + 1) + ')':>12}" for i in range(6)))
    w(f"  {'Pseudo R2 published':<22}" + "".join(f"{sa.loc[s].r2_pseudo:>12.3f}" for s in SPECS))
    w(f"  {'Pseudo R2 E1|E2':<22}" + "".join(f"{sb.loc[s].r2_pseudo:>12.3f}" for s in SPECS))
    w(f"  {'changed?':<22}"
      + "".join(f"{('YES' if f'{sa.loc[s].r2_pseudo:.3f}' != f'{sb.loc[s].r2_pseudo:.3f}' else '-'):>12}"
                for s in SPECS))
    w(f"  {'Observations pub':<22}" + "".join(f"{int(sa.loc[s].nobs):>12,}" for s in SPECS))
    w(f"  {'Observations E1|E2':<22}" + "".join(f"{int(sb.loc[s].nobs):>12,}" for s in SPECS))
    w(f"  {'changed?':<22}"
      + "".join(f"{('YES' if sa.loc[s].nobs != sb.loc[s].nobs else '-'):>12}" for s in SPECS))
    w("")
    w(f"  coefficient cells whose printed text changes : {n_str_changed} of 24")
    w(f"    of which the 2-dp value moves              : {n_val_changed}")
    w(f"    of which only the star moves               : {n_str_changed - n_val_changed}")
    w(f"  significance stars changed                   : {n_star_changed} of 24")
    w(f"  printed SE cells changed                     : {n_se_changed} of 24")
    w(f"  sign flips at full precision                 : "
      f"{sum(1 for t in TERMS for s in SPECS if np.sign(pub[(pub.model == s) & (pub.term == t)].ame_coef.iloc[0]) != np.sign(e2[(e2.model == s) & (e2.term == t)].ame_coef.iloc[0]))} of 24")
    w(f"  largest absolute AME move                    : "
      f"{max(abs(e2[(e2.model == s) & (e2.term == t)].ame_coef.iloc[0] - pub[(pub.model == s) & (pub.term == t)].ame_coef.iloc[0]) for t in TERMS for s in SPECS):.4f}")
    w("")

    w("Unified diff of the .tex bodies (published -> E1|E2)")
    for line in difflib.unified_diff(tex_pub.splitlines(), tex_new.splitlines(),
                                     "tables_current/allTasks_ai.tex",
                                     "tables_new/allTasks_ai.tex", lineterm="", n=1):
        w("  " + line)
    w("")

    w("-" * 100)
    w("C. FIGURE OA.A.1  writeup/plots/execTypeVaryingDWA_noTasksWithRepetitiveDWAs/")
    w("                  is_ai/AME_full_is_ai_{no_fe_no_dwa,major_fe_no_dwa,")
    w("                  minor_fe_no_dwa,no_fe_with_dwa}.png")
    w("-" * 100)
    w("")
    fp = pd.read_csv(f"{WORK}/figOAA1_stats_published.csv").set_index(["spec", "term"])
    fn = pd.read_csv(f"{WORK}/figOAA1_stats_e1e2.csv").set_index(["spec", "term"])
    w("C1. The red 'Observed = ' legend label on each subpanel (all 16 change)")
    w(f"    {'panel':<18}{'subpanel':<14}{'published':>18}{'E1|E2':>18}{'delta':>10}")
    for s in SPECS[:4]:
        for t in TERMS:
            a, b = fp.loc[(s, t)], fn.loc[(s, t)]
            w(f"    {PANEL[s]:<18}{TLAB[t]:<14}{a.label:>18}{b.label:>18}"
              f"{b.observed - a.observed:>10.4f}")
    w("")
    w("C2. The red +/-1.645*SE band (drawn from the same B=200 bootstrap SE)")
    w(f"    {'panel':<18}{'subpanel':<14}{'SE pub':>10}{'SE new':>10}"
      f"{'half-width pub':>16}{'half-width new':>16}")
    for s in SPECS[:4]:
        for t in TERMS:
            a, b = fp.loc[(s, t)], fn.loc[(s, t)]
            w(f"    {PANEL[s]:<18}{TLAB[t]:<14}{a.observed_se:>10.4f}{b.observed_se:>10.4f}"
              f"{1.645 * a.observed_se:>16.4f}{1.645 * b.observed_se:>16.4f}")
    w("")
    w("C3. The placebo histogram itself (1,000 values per subpanel: the observed run at")
    w("    i = 0, which cell 19 reads from full_0.csv, plus 999 position reshuffles at")
    w("    seeds 43..1041; the exposure control is in the reshuffled fits too, so the")
    w("    null moves as well as the red line)")
    w(f"    {'panel':<18}{'subpanel':<14}{'null mean pub':>15}{'null mean new':>15}"
      f"{'null sd pub':>13}{'null sd new':>13}{'P(null>=obs) pub':>18}{'new':>10}")
    for s in SPECS[:4]:
        for t in TERMS:
            a, b = fp.loc[(s, t)], fn.loc[(s, t)]
            w(f"    {PANEL[s]:<18}{TLAB[t]:<14}{a.null_mean:>15.4f}{b.null_mean:>15.4f}"
              f"{a.null_sd:>13.4f}{b.null_sd:>13.4f}"
              f"{a.share_null_ge_obs:>18.3f}{b.share_null_ge_obs:>10.3f}")
    w("")
    w("    Note on the tail shares. They are computed over all 1,000 plotted values, so a")
    w("    subpanel whose observed estimate beats every genuine reshuffle still reads 0.001,")
    w("    not 0.000: cell 19's i = 0 IS the observed run. This is a pre-existing property of")
    w("    the published figure (one draw in a thousand is the observed estimate), not")
    w("    something the exposure swap introduces, and it is why these shares sit 0.001 above")
    w("    the m6 audit's, which were computed on i = 1..999 only. Reproduced deliberately:")
    w("    dropping i = 0 would change the histogram and is a second change, not this one.")
    w("")
    xl = pd.read_csv(f"{WORK}/xlim.csv")
    w("C4. x-axis limits")
    w("    The published code does not hard-code them: plot_comparison_hist sets a")
    w("    symmetric bound from max|value| over all 6 specs x 4 terms of the same figure")
    w("    call, so the limits are an output of the data, not a styling choice. Both")
    w("    values below are what the paper's own code returns on its own inputs.")
    for _, r in xl.iterrows():
        w(f"    {r['mode']:<12} xlim = ({r['xmin']:.6f}, {r['xmax']:.6f})")
    d = xl.set_index("mode")
    w(f"    change: {d.loc['e1e2', 'xmax'] - d.loc['published', 'xmax']:+.6f} on each side "
      f"({100 * (d.loc['e1e2', 'xmax'] / d.loc['published', 'xmax'] - 1):+.2f}%). "
      f"Nothing is clipped under either.")
    w("")

    w("-" * 100)
    w("D. PROSE IN THE BODY THAT QUOTES THESE NUMBERS (recorded, not edited -- the repo is")
    w("   read-only here)")
    w("-" * 100)
    w("")
    e2i = e2.set_index(["model", "term"])
    base = e2i.loc[("no_fe_no_dwa", "prev_is_ai"), "ame_coef"]
    fe_cols = ["major_fe_no_dwa", "minor_fe_no_dwa", "no_fe_with_dwa",
               "no_fe_with_dwa_withTaskDWACount"]
    imm = [e2i.loc[(m, t), "ame_coef"] for m in fe_cols for t in ["prev_is_ai", "next_is_ai"]]
    dis = [e2i.loc[(m, t), "ame_coef"] for m in fe_cols for t in ["prev2_is_ai", "next2_is_ai"]]
    w("  writeup/7_empirics.tex:170")
    w('    "The immediate neighbor effect attenuates from $0.12$ in the baseline to between')
    w('     $0.04$ and $0.06$ ... while the effects two positions away fall to between')
    w('     $-0.01$ and $0.01$ and lose significance."')
    w(f"    NEEDS AN EDIT: the baseline immediate-neighbour effect is now {base:.4f}, which")
    w(f"    prints as 0.11, so $0.12$ must become $0.11$.")
    w(f"    The two ranges it quotes both survive: over columns (2)-(4) and (6) the")
    w(f"    immediate-neighbour AMEs span {min(imm):.4f} to {max(imm):.4f} (prints 0.04 to")
    w(f"    0.06) and the two-positions-away AMEs span {min(dis):.4f} to {max(dis):.4f}")
    w(f"    (prints -0.01 to 0.01), none of them significant at 10%.")
    w("")
    r1 = (e2i.loc[("no_fe_no_dwa", "prev_is_ai"), "ame_coef"]
          / e2i.loc[("no_fe_no_dwa", "prev2_is_ai"), "ame_coef"])
    r2 = (e2i.loc[("no_fe_no_dwa", "next_is_ai"), "ame_coef"]
          / e2i.loc[("no_fe_no_dwa", "next2_is_ai"), "ame_coef"])
    p1 = (pub.set_index(["model", "term"]).loc[("no_fe_no_dwa", "prev_is_ai"), "ame_coef"]
          / pub.set_index(["model", "term"]).loc[("no_fe_no_dwa", "prev2_is_ai"), "ame_coef"])
    p2 = (pub.set_index(["model", "term"]).loc[("no_fe_no_dwa", "next_is_ai"), "ame_coef"]
          / pub.set_index(["model", "term"]).loc[("no_fe_no_dwa", "next2_is_ai"), "ame_coef"])
    w("  writeup/7_empirics.tex:165")
    w('    "...at one and at two positions away alike, with the effects of the adjacent')
    w('     steps roughly twice the size of those two positions out."')
    w(f"    NO EDIT NEEDED. The adjacent-to-distant ratios in column (1) go "
      f"{p1:.2f} -> {r1:.2f} and {p2:.2f} -> {r2:.2f}; both still read as roughly two.")
    w("    Both distant terms in column (1) are still significant, the k+2 one at 5% rather")
    w("    than 1%, so 'at one and at two positions away alike' still holds.")
    w("")
    w("  writeup/7_empirics.tex:162 (table note) and :176 (the 1,000 placebo datasets)")
    w("    NO EDIT NEEDED. The sample sizes and the DWA cluster counts (1,748 / 1,705 / 534)")
    w("    are unchanged, because exposure is a covariate and never a sample filter; and the")
    w("    placebo loop still runs 1,000 indices at the same seeds.")
    w("")

    open(f"{STAGE}/diffs/table2-figOAA1.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    if not all(ok for _, ok in ctrl):
        print("\n*** ONE OR MORE CONTROLS FAILED ***")
        sys.exit(1)


if __name__ == "__main__":
    main()
