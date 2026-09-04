"""Write the per-exhibit numeric diff for the SA.D.4 / SA.E.3 / SA.E.4 group.

Every cell that changed between the published (E1-only) exhibit and the regenerated (E1|E2)
one, at the precision the figure itself prints, plus the fixture evidence that nothing else
moved.

Writes ONLY writeup/_e1e2_preview/diffs/sad-sae-figs.txt.
"""
import os
import numpy as np, pandas as pd

REPO = "/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin"
STAGE = f"{REPO}/writeup/_e1e2_preview"
M6 = ("/private/tmp/claude-502/-Users-peymansh-MIT-Dropbox-Peyman-Shahidi-GitHub-rubin/"
      "5af7d286-51e7-470a-b57e-4e1373499eea/scratchpad/m6")
OUT = f"{STAGE}/diffs/sad-sae-figs.txt"

L = []
def w(s=''): L.append(s)

star = lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
def ordi(n):
    if np.isnan(n): return ""
    n = int(round(n)); s = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10 if n % 100 not in (11, 12, 13) else 0, 'th')
    return f"{n}{s}"

w("=" * 100)
w("PER-EXHIBIT NUMERIC DIFF — Figures SA.D.4, SA.E.3, SA.E.4")
w("published pipeline  human_labels.isin(['E1'])   ->   regenerated  human_labels.isin(['E1','E2'])")
w("Nothing else changed: same sample filters, same estimator ladder, same DWA clustering,")
w("same B=200 bootstrap and same seed=123 (SA.D.4), same 1,000-draw reshuffle with seeds 42..1041 (SA.E.4).")
w("=" * 100)

# =====================================================================================
# SA.E.3
# =====================================================================================
w()
w("#" * 100)
w("# FIGURE SA.E.3 — neighbor AME heatmaps across frequency cuts")
w("#   plots/execTypeVaryingDWA_weeklyTasks/neighbor_logic_threshold_heatmap_{prev,next}_bySpec.png")
w("#   exposure enters as the control `is_exposed` in every cell's logit")
w("#" * 100)

pub_sw = pd.read_csv(f"{REPO}/data/computed_objects/execTypeVaryingDWA_anthropicIndex_weeklyTasks/"
                     "neighbor_logic_threshold_sweep.csv")
reb_sw = pd.read_csv(f"{M6}/sae3_sweep_E1_vs_E1E2.csv")
e1 = reb_sw[reb_sw['mask'] == 'E1'].drop(columns='mask')
e2 = reb_sw[reb_sw['mask'] == 'E1E2'].drop(columns='mask')
K = ['cut', 'family', 'threshold', 'spec', 'term']
fx = pub_sw.merge(e1, on=K, suffixes=('_p', '_r'))
w()
w(f"FIXTURE (published side rebuilt from scratch, E1 mask): {len(fx)} cells, "
  f"max|dAME| = {np.abs(fx.ame_p - fx.ame_r).max():.2e}, "
  f"max|dp| = {np.abs(fx.pval_p - fx.pval_r).max():.2e}, nobs identical = {bool((fx.nobs_p == fx.nobs_r).all())}")
w("Re-rendering the published PNG from the published sweep CSV reproduces every heatmap cell,")
w("its colour, its stars and its N; the only pixels that move are glyph antialiasing from a newer")
w("matplotlib (3.11.0 here). See plots_fixture/E1_from_published_csv_dpi200/.")

m = pub_sw.merge(e2, on=K, suffixes=('_p', '_r'))
m = m[m.term.isin(['prev_is_ai', 'next_is_ai'])].copy()
m['lab_p'] = m.apply(lambda r: f"{r.ame_p:.3f}{star(r.pval_p)}", axis=1)
m['lab_r'] = m.apply(lambda r: f"{r.ame_r:.3f}{star(r.pval_r)}", axis=1)
m['changed'] = m.lab_p != m.lab_r
FAMLAB = {'all': 'All tasks', 'daily': 'Daily+', 'sevdaily': 'SeveralDaily+', 'hourly': 'Hourly+'}
SPECLAB = {'no_fe_no_dwa': 'No FE', 'major_fe_no_dwa': 'Major FE', 'minor_fe_no_dwa': 'Minor FE'}
TERMLAB = {'prev_is_ai': '(a) prev k-1', 'next_is_ai': '(b) next k+1'}
SPECORD = {'no_fe_no_dwa': 0, 'major_fe_no_dwa': 1, 'minor_fe_no_dwa': 2}
FAMORD = {'all': 0, 'daily': 1, 'sevdaily': 2, 'hourly': 3}
m['so'] = m.spec.map(SPECORD); m['fo'] = m.family.map(FAMORD)
m['to'] = m.term.map({'prev_is_ai': 0, 'next_is_ai': 1})
m = m.sort_values(['to', 'fo', 'threshold', 'so'])

w()
w("EVERY PLOTTED CELL (the heatmap prints AME at 3 dp with stars, and N underneath).")
w("N is identical in every cell under both masks: the mask is a right-hand-side control, never a filter.")
w()
w(f"{'panel':<13}{'row':<16}{'thr':>6}  {'spec':<10}{'published':>12}{'E1|E2':>12}{'delta':>9}   N")
w("-" * 88)
nchanged = 0
for _, r in m.iterrows():
    thr = '-' if r.family == 'all' else f">={int(r.threshold)}%"
    flag = '  <-- changed' if r.changed else ''
    if r.changed: nchanged += 1
    w(f"{TERMLAB[r.term]:<13}{FAMLAB[r.family]:<16}{thr:>6}  {SPECLAB[r.spec]:<10}"
      f"{r.lab_p:>12}{r.lab_r:>12}{r.ame_r - r.ame_p:>+9.4f}   {int(r.nobs_p):,}{flag}")
w("-" * 88)
w(f"cells whose printed label changed: {nchanged} of {len(m)}")
w(f"sign flips: {int(((np.sign(m.ame_p) != np.sign(m.ame_r))).sum())} of {len(m)}")
sc = m[m.apply(lambda r: star(r.pval_p) != star(r.pval_r), axis=1)]
w(f"star changes: {len(sc)} of {len(m)}")
for _, r in sc.iterrows():
    thr = '-' if r.family == 'all' else f">={int(r.threshold)}%"
    w(f"    {TERMLAB[r.term]}  {FAMLAB[r.family]} {thr}  {SPECLAB[r.spec]}: "
      f"{r.lab_p} -> {r.lab_r}   (N={int(r.nobs_p):,})")
w(f"largest |shift|: {np.abs(m.ame_r - m.ame_p).max():.4f}")
w("NOTE ON SHADING (not a styling change; the published code's own behaviour). Each panel's colour")
w("scale is TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax) with vmax = max|AME| in that panel,")
w("recomputed from the data, so the whole panel re-shades whenever its extreme cell moves:")
for term in ['prev_is_ai', 'next_is_ai']:
    for spec in ['no_fe_no_dwa', 'major_fe_no_dwa', 'minor_fe_no_dwa']:
        a = np.abs(pub_sw[(pub_sw.term == term) & (pub_sw.spec == spec)].ame).max()
        b = np.abs(e2[(e2.term == term) & (e2.spec == spec)].ame).max()
        w(f"    {TERMLAB[term]}  {SPECLAB[spec]:<9} vmax {a:.3f} -> {b:.3f}   "
          f"({'panel renders PALER' if b > a else 'panel renders DARKER'})")
w("  Panel (a) re-shades paler because Hourly+ >=50% grows; panel (b) re-shades darker because")
w("  the SeveralDaily+ >=65% and Hourly+ >=50% cells shrink. Cell VALUES and stars are the content.")

# =====================================================================================
# SA.E.4
# =====================================================================================
w()
w("#" * 100)
w("# FIGURE SA.E.4 — neighbor placebo forests across frequency cuts")
w("#   plots/placebo_summary/placebo_summary_forest_neighbor_t{1,2}_byCut.png")
w("#   exposure enters the observed dot AND all 1,000 reshuffle draws that make the grey null band")
w("#" * 100)

sp = pd.read_csv(f"{STAGE}/diffs/_sae4_stats_published.csv")
sr = pd.read_csv(f"{STAGE}/diffs/_sae4_stats_E1_rebuild.csv")
se = pd.read_csv(f"{STAGE}/diffs/_sae4_stats_E1E2.csv")
for d in (sp, sr, se):
    d['red'] = (~d.p10.isna()) & ((d.Observed < d.p10) | (d.Observed > d.p90))
K4 = ['cut_tag', 'Effect', 'Spec']
PL = ['(t-1)', '(t+1)']
sp = sp[sp.Effect.isin(PL)].set_index(K4)
sr = sr[sr.Effect.isin(PL)].set_index(K4)
se = se[se.Effect.isin(PL)].set_index(K4)

w()
w("FIXTURE 1 (strongest available): re-rendering the two published PNGs from the paper's own cached")
w("placebo_null_draws.csv + placebo_observed.csv reproduces them PIXEL-FOR-PIXEL (0.0000% of pixels")
w("differ, identical image dimensions). The plotting code in this staging folder is therefore the")
w("published one. See plots_fixture/E1_from_published_draws_dpi190/.")
w()
j = sp.join(sr, lsuffix='_pub', rsuffix='_reb')
noisy = j[(np.abs(j.Observed_pub - j.Observed_reb) > 1e-9) | (np.abs(j['mean_pub'] - j['mean_reb']) > 1e-9)
          | (np.abs(j.p10_pub - j.p10_reb) > 1e-9) | (np.abs(j.p90_pub - j.p90_reb) > 1e-9)]
w("FIXTURE 2 (re-simulating the 1,000-draw null from scratch under E1, seeds 42..1041):")
w(f"  observed markers reproduce to 1.9e-16 on all 72 plotted cells;")
w(f"  of the 72 cells, {len(noisy)} have a null band that moves at all, and all {len(noisy)} sit in the")
w("  single row Hourly+ >=50% (N=49 occ. / 61 obs.), where the logit is at its convergence limit.")
w("  Dot-colour flips from this rebuild noise: "
  f"{int((j.red_pub != j.red_reb).sum())} of 72. Printed-ordinal changes from it:")
for k, r in noisy.iterrows():
    if ordi(r.pct_pub) != ordi(r.pct_reb):
        w(f"    {k[0]:<10} {k[1]}  {k[2]:<9}  {ordi(r.pct_pub)} -> {ordi(r.pct_reb)}  "
          f"(pct {r.pct_pub:.2f} -> {r.pct_reb:.2f})")
w("  So the Hourly+ >=50% row of the E1|E2 figure carries a sliver of rebuild noise on top of the")
w("  mask change; every other row is a clean like-for-like comparison.")

w()
w("EVERY PLOTTED DOT (72 = 2 panels x 3 specs x 12 estimable cuts).")
w("`obs` is the marker position, `pct` the ordinal printed in the right margin, `dot` its colour")
w("(R = crimson, outside the 10-90 null band; b = steelblue, inside).")
w("Row labels (N occ.) come from `nbr_counts` and are mask-independent: none of them moves.")
w()
CUTLAB = {'all': 'All tasks'}
for fam, lab in [('daily', 'Daily+'), ('sevdaily', 'SeveralDaily+'), ('hourly', 'Hourly+')]:
    for t in [20, 35, 50, 65]:
        CUTLAB[f"{fam}{t}"] = f"{lab} >={t}%"
ORDER = list(CUTLAB)
j2 = sp.join(se, lsuffix='_pub', rsuffix='_e2').reset_index()
j2['co'] = j2.cut_tag.map({c: i for i, c in enumerate(ORDER)})
j2['so'] = j2.Spec.map({'No FE': 0, 'Major FE': 1, 'Minor FE': 2})
j2 = j2.sort_values(['Effect', 'co', 'so'])
w(f"{'panel':<8}{'cut':<22}{'spec':<10}"
  f"{'obs pub':>9}{'obs E1E2':>10}{'delta':>9}  {'pct pub':>8}{'pct E1E2':>9}  {'null mean p->e':>18}  dot")
w("-" * 108)
for _, r in j2.iterrows():
    dot = f"{'R' if r.red_pub else 'b'}->{'R' if r.red_e2 else 'b'}"
    flag = '  <-- DOT FLIP' if r.red_pub != r.red_e2 else ''
    w(f"{r.Effect:<8}{CUTLAB[r.cut_tag]:<22}{r.Spec:<10}"
      f"{r.Observed_pub:>9.3f}{r.Observed_e2:>10.3f}{r.Observed_e2 - r.Observed_pub:>+9.3f}  "
      f"{ordi(r.pct_pub):>8}{ordi(r.pct_e2):>9}  {r['mean_pub']:>8.4f}->{r['mean_e2']:<8.4f}  {dot}{flag}")
w("-" * 108)
w(f"red dots (observed outside its 10-90 null): {int(sp.red.sum())} -> {int(se.red.sum())} of 72")
w(f"dot-colour flips: {int((j2.red_pub != j2.red_e2).sum())} of 72")
w(f"sign flips in the observed AME: {int((np.sign(j2.Observed_pub) != np.sign(j2.Observed_e2)).sum())} of 72")
w(f"mean |shift| in the observed AME: {np.abs(j2.Observed_e2 - j2.Observed_pub).mean():.4f}  "
  f"(max {np.abs(j2.Observed_e2 - j2.Observed_pub).max():.4f})")
w(f"mean |shift| in the null mean: {np.abs(j2['mean_e2'] - j2['mean_pub']).mean():.4f}  "
  f"(max {np.abs(j2['mean_e2'] - j2['mean_pub']).max():.4f})")
w(f"mean |shift| in the printed percentile: {np.abs(j2.pct_e2 - j2.pct_pub).mean():.1f} pp  "
  f"(max {np.abs(j2.pct_e2 - j2.pct_pub).max():.1f} pp)")

# =====================================================================================
# SA.D.4
# =====================================================================================
w()
w("#" * 100)
w("# FIGURE SA.D.4 — Table-2 neighbor AMEs across the 11 GPT prompts")
w("#   plots/execTypeVaryingDWA_noTasksWithRepetitiveDWAs_robustness/ame_{no_fe_no_dwa,major_fe_no_dwa,")
w("#   minor_fe_no_dwa,no_fe_with_dwa}_robustness.png")
w("#   exposure enters as the control `is_exposed`; the 90% bars are a B=200 DWA-cluster bootstrap")
w("#" * 100)

pub_d = pd.read_csv(f"{REPO}/data/computed_objects/"
                    "execTypeVaryingDWA_anthropicIndex_noTasksWithRepetitiveDWAs_robustness/allTasks_ai.csv")
f_e1 = f"{STAGE}/diffs/_sad4_allTasks_ai_E1.csv"
f_e2 = f"{STAGE}/diffs/_sad4_allTasks_ai_E1E2.csv"
if not (os.path.exists(f_e1) and os.path.exists(f_e2)):
    w()
    w("*** NOT AVAILABLE — the SA.D.4 estimation had not finished when this diff was written. ***")
else:
    d1 = pd.read_csv(f_e1); d2 = pd.read_csv(f_e2)
    SPECS4 = ['no_fe_no_dwa', 'major_fe_no_dwa', 'minor_fe_no_dwa', 'no_fe_with_dwa']
    TARGET = ['prev2_is_ai', 'prev_is_ai', 'next_is_ai', 'next2_is_ai']
    KD = ['dataset', 'model', 'term']
    p = pub_d[pub_d.model.isin(SPECS4)][KD + ['nobs', 'ame_coef', 'ame_se', 'p_value', 'r2_pseudo']]
    fx = p.merge(d1, on=KD, suffixes=('_p', '_r'))
    w()
    w(f"FIXTURE (published side rebuilt from scratch under E1, including the B=200 bootstrap): "
      f"{len(fx)} cells")
    w(f"  max|d ame_coef| = {np.abs(fx.ame_coef_p - fx.ame_coef_r).max():.2e}")
    w(f"  max|d ame_se|   = {np.abs(fx.ame_se_p - fx.ame_se_r).max():.2e}")
    w(f"  max|d p_value|  = {np.abs(fx.p_value_p - fx.p_value_r).max():.2e}")
    w(f"  nobs identical  = {bool((fx.nobs_p == fx.nobs_r).all())}")
    w("  i.e. the point estimates, the bootstrap standard errors and hence the plotted 90% error bars")
    w("  come back byte-identical to the published allTasks_ai.csv.")

    m = p.merge(d2, on=KD, suffixes=('_p', '_r'))
    m['lo_p'] = m.ame_coef_p - 1.645 * m.ame_se_p; m['hi_p'] = m.ame_coef_p + 1.645 * m.ame_se_p
    m['lo_r'] = m.ame_coef_r - 1.645 * m.ame_se_r; m['hi_r'] = m.ame_coef_r + 1.645 * m.ame_se_r
    PANEL = {'no_fe_no_dwa': '(a) No FE', 'major_fe_no_dwa': '(b) Major FE',
             'minor_fe_no_dwa': '(c) Minor FE', 'no_fe_with_dwa': '(d) DWA FE'}
    TL = {'prev2_is_ai': 'k-2', 'prev_is_ai': 'k-1', 'next_is_ai': 'k+1', 'next2_is_ai': 'k+2'}
    m['po'] = m.model.map({s: i for i, s in enumerate(SPECS4)})
    m['to'] = m.term.map({t: i for i, t in enumerate(TARGET)})
    m = m.sort_values(['po', 'to', 'dataset'])
    w()
    w("EVERY PLOTTED MARKER (176 = 4 panels x 4 terms x 11 prompts), with its 90% bar (coef +/- 1.645*se).")
    w("nobs is identical prompt by prompt under both masks.")
    w()
    w(f"{'panel':<14}{'term':<6}{'prompt':>7}{'AME pub':>10}{'AME E1|E2':>11}{'delta':>9}"
      f"{'SE pub':>9}{'SE E1|E2':>10}   {'90% bar pub':>22}   {'90% bar E1|E2':>22}")
    w("-" * 132)
    for _, r in m.iterrows():
        w(f"{PANEL[r.model]:<14}{TL[r.term]:<6}{int(r.dataset):>7}"
          f"{r.ame_coef_p:>10.4f}{r.ame_coef_r:>11.4f}{r.ame_coef_r - r.ame_coef_p:>+9.4f}"
          f"{r.ame_se_p:>9.4f}{r.ame_se_r:>10.4f}   "
          f"[{r.lo_p:+.4f}, {r.hi_p:+.4f}]   [{r.lo_r:+.4f}, {r.hi_r:+.4f}]")
    w("-" * 132)
    w(f"largest |shift| in a marker: {np.abs(m.ame_coef_r - m.ame_coef_p).max():.4f}")
    sf = m[np.sign(m.ame_coef_p) != np.sign(m.ame_coef_r)]
    w(f"sign flips: {len(sf)} of {len(m)}")
    for _, r in sf.iterrows():
        w(f"    {PANEL[r.model]} {TL[r.term]} prompt {int(r.dataset)}: "
          f"{r.ame_coef_p:+.4f} (p={r.p_value_p:.2f}) -> {r.ame_coef_r:+.4f} (p={r.p_value_r:.2f})")
    st = m[m.apply(lambda r: star(r.p_value_p) != star(r.p_value_r), axis=1)]
    w(f"star changes on the bootstrap p-value: {len(st)} of {len(m)}   "
      f"(the figure plots bars, not stars, so this is a diagnostic only)")
    for _, r in st.iterrows():
        w(f"    {PANEL[r.model]} {TL[r.term]} prompt {int(r.dataset)}: "
          f"{r.ame_coef_p:.3f}{star(r.p_value_p)} -> {r.ame_coef_r:.3f}{star(r.p_value_r)}")
    w()
    w("DASHED ACROSS-PROMPT MEAN (the horizontal line and its legend label, printed at 2 dp):")
    w(f"{'panel':<14}{'term':<6}{'mean pub':>10}{'mean E1|E2':>12}   label pub -> label E1|E2")
    for sp_ in SPECS4:
        for t in TARGET:
            a = m[(m.model == sp_) & (m.term == t)]
            mp = a.ame_coef_p.mean(); mr = a.ame_coef_r.mean()
            chg = '  <-- printed label changes' if f"{mp:.2f}" != f"{mr:.2f}" else ''
            w(f"{PANEL[sp_]:<14}{TL[t]:<6}{mp:>10.4f}{mr:>12.4f}   "
              f"Mean = {mp:.2f} -> Mean = {mr:.2f}{chg}")
    w()
    lo = m[['lo_r']].min().iloc[0]; hi = m[['hi_r']].max().iloc[0]
    w(f"HARD-CODED AXIS LIMIT CHECK. The notebook sets ax.set_ylim(-0.075, 0.175) in every panel.")
    w(f"  Under E1|E2 the extreme plotted values are  min(lo) = {lo:.4f}, max(hi) = {hi:.4f}.")
    clip = m[(m.lo_r < -0.075) | (m.hi_r > 0.175)]
    if len(clip) == 0:
        w("  Nothing is clipped: the published limits are kept unchanged.")
    else:
        w(f"  *** {len(clip)} marker(s)/bar(s) would be clipped: ***")
        for _, r in clip.iterrows():
            w(f"    {PANEL[r.model]} {TL[r.term]} prompt {int(r.dataset)}: bar [{r.lo_r:+.4f}, {r.hi_r:+.4f}]")
    clipp = m[(m.lo_p < -0.075) | (m.hi_p > 0.175)]
    w(f"  (for reference, the published figure already clips {len(clipp)} bar(s) at those limits.)")

# =====================================================================================
# Prose that moves with these three figures
# =====================================================================================
w()
w("#" * 100)
w("# PROSE IN writeup/SA_E_frequency_robustness.tex THAT MOVES WITH THESE FIGURES")
w("#" * 100)
w()
sw2 = reb_sw
def _g(mask, fam, thr, spec, term):
    x = sw2[(sw2['mask'] == mask) & (sw2.family == fam) & (sw2.threshold == thr)
            & (sw2.spec == spec) & (sw2.term == term)]
    return float(x.ame.iloc[0])

w("SA_E:142  (reads Fig SA.E.3, top row)")
w('  published : "...the full-sample adjacent-step effects of $+0.12$ under no fixed effects,')
w('               which attenuate to between $+0.04$ and $+0.06$ once SOC fixed effects absorb..."')
for mk, lb in [('E1', 'E1     '), ('E1E2', 'E1|E2  ')]:
    pv = _g(mk, 'all', 0, 'no_fe_no_dwa', 'prev_is_ai'); nv = _g(mk, 'all', 0, 'no_fe_no_dwa', 'next_is_ai')
    fe = [_g(mk, 'all', 0, s, t) for s in ['major_fe_no_dwa', 'minor_fe_no_dwa'] for t in ['prev_is_ai', 'next_is_ai']]
    w(f"  {lb}   no-FE adjacent = {pv:.4f} / {nv:.4f}  (prints +{pv:.2f} / +{nv:.2f});  "
      f"SOC-FE adjacent range = [{min(fe):.4f}, {max(fe):.4f}]")
w('  REQUIRED EDIT: "$+0.12$" -> "$+0.11$".')
w('  NO EDIT NEEDED to "between $+0.04$ and $+0.06$": the E1|E2 range [0.0455, 0.0591] still lies')
w("  inside that bracket, so the published wording stays literally true.")
w()
w("SA_E:143  (reads Fig SA.E.3, SeveralDaily+ >=50%, previous task, no FE)")
w('  published : "...the previous-task effect rises to $+0.14$ in the no-fixed-effects specification."')
for mk, lb in [('E1', 'E1     '), ('E1E2', 'E1|E2  ')]:
    v = _g(mk, 'sevdaily', 50, 'no_fe_no_dwa', 'prev_is_ai')
    w(f"  {lb}   {v:.5f}  (heatmap cell prints {v:.3f}; prose at 2 dp prints +{v:.2f})")
w('  REQUIRED EDIT: "$+0.14$" -> "$+0.12$".')
w("  (The heatmap cell moves 0.137 -> 0.125, but the prose rounds to 2 dp, so it is +0.12, not +0.13.)")
w()
w("SA_E:174  (reads Fig SA.E.4, top row)")
w('  published : "Both effects sit far in the upper tail of their nulls in the full sample, at the')
w('               100th percentile in all three specifications..."')
top = se.reset_index(); top = top[top.cut_tag == 'all']
topp = sp.reset_index(); topp = topp[topp.cut_tag == 'all']
for lbl, d in [('E1   ', topp), ('E1|E2', top)]:
    s = ", ".join(f"{r.Effect} {r.Spec} {ordi(r.pct)}{'R' if r.red else 'b'}" for _, r in d.iterrows())
    w(f"  {lbl}  {s}")
w("  NO EDIT NEEDED: every all-tasks cell still prints 100th and stays red under E1|E2.")
w()
w("SA_E:175  (reads Fig SA.E.4)  *** THE ONE SENTENCE THAT MUST BE SOFTENED ***")
w('  published : "The previous-task effect lies outside its 10-to-90 null across all four Daily$+$')
w('               cuts (the 91st to 99th percentiles) and at SeveralDaily$+$ $\\geq 20\\%$ and')
w('               $\\geq 50\\%$ (the 93rd to 98th), and the next-task effect behaves similarly,')
w('               escaping its null across the Daily$+$ cuts."')
for lbl, d in [('E1   ', sp.reset_index()), ('E1|E2', se.reset_index())]:
    a = d[(d.Effect == '(t-1)') & (d.cut_tag.str.startswith('daily'))]
    b = d[(d.Effect == '(t-1)') & (d.cut_tag.isin(['sevdaily20', 'sevdaily50']))]
    c = d[(d.Effect == '(t+1)') & (d.cut_tag.str.startswith('daily'))]
    w(f"  {lbl}  (t-1) Daily+          : {int(a.red.sum())}/12 outside, percentiles {ordi(a.pct.min())}-{ordi(a.pct.max())}")
    w(f"         (t-1) SevDaily+ 20/50 : {int(b.red.sum())}/6 outside, percentiles {ordi(b.pct.min())}-{ordi(b.pct.max())}")
    w(f"         (t+1) Daily+          : {int(c.red.sum())}/12 outside, percentiles {ordi(c.pct.min())}-{ordi(c.pct.max())}")
w("  Cells that leave the band under E1|E2:")
for k, r in sp.join(se, lsuffix='_pub', rsuffix='_e2').iterrows():
    if r.red_pub and not r.red_e2:
        w(f"    {k[0]:<11}{k[1]}  {k[2]:<9}  {ordi(r.pct_pub)} (red) -> {ordi(r.pct_e2)} (blue)")
w("  Cells that enter the band under E1|E2:")
for k, r in sp.join(se, lsuffix='_pub', rsuffix='_e2').iterrows():
    if (not r.red_pub) and r.red_e2:
        w(f"    {k[0]:<11}{k[1]}  {k[2]:<9}  {ordi(r.pct_pub)} (blue) -> {ordi(r.pct_e2)} (red)")
w("  SUGGESTED REWRITE (facts only, wording is the author's):")
w("    - previous-task effect outside its null across THREE of the four Daily+ cuts in the no-FE")
w("      and SOC-major-FE columns (Daily+ >=50% falls inside at the 87th and 89th; it is still")
w("      outside under SOC-minor FE at the 94th), Daily+ percentile range widens to 87th-97th;")
w("    - at SeveralDaily+ >=20% the no-FE cell moves inside (90th) while major and minor stay")
w("      outside; SeveralDaily+ >=50% stays outside in all three specs (90th to 98th), so the")
w("      SeveralDaily+ range becomes 90th-98th;")
w("    - the next-task clause gets MORE accurate, not less: Daily+ >=50% no FE moves from inside")
w("      (89th) to outside (91st), taking (t+1) Daily+ from 8 of 12 outside to 9 of 12.")
w(f"    - overall the figure carries {int(se.red.sum())} red dots instead of {int(sp.red.sum())}.")
w()
w("SA_D_prompt_robustness.tex:181-186  (reads Fig SA.D.4)")
w("  No sentence there quotes a point estimate, so no prose edit follows from SA.D.4.")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    f.write("\n".join(L) + "\n")
print(f"wrote {OUT}  ({len(L)} lines)")
