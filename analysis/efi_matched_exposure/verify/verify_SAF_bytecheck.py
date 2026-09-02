"""Byte-for-byte check of my replication of the PUBLISHED APQC table, plus spec C on the pooled sample."""
import os
import warnings, difflib, numpy as np, pandas as pd, statsmodels.formula.api as smf

import os as _os
# Repo root and output dir are derived from this file's own location, so the folder
# can move and no home directory is baked in. Generated output follows the repo
# convention: data/computed_objects/<name of this analysis folder>/.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
OUT = _os.path.join(REPO, "data", "computed_objects", "efi_matched_exposure")
_os.makedirs(OUT, exist_ok=True)

warnings.filterwarnings('ignore')
MAIN = REPO
SNAP = os.path.join(os.path.join(_HERE, "..", "published_reference"), "apqc_fragmentation_index_regression.tex")
L = pd.read_csv(f"{MAIN}/data/computed_objects/apqc_pred3_industry/industry_leaf_matches.csv", dtype={'hid': str})
L['sk'] = L['hid'].map(lambda h: tuple(int(x) for x in h.split('.')))
L = L.sort_values(['uid', 'sk']).reset_index(drop=True)
L['category'] = L['hid'].str.split('.').str[0]
c = L['similarity'] >= 0.73
L['exposed'] = (c & L['human_labels'].isin(['E1', 'E2'])).astype(int)
L['e1'] = (c & (L['human_labels'] == 'E1')).astype(int)
L['executed'] = (c & L['label'].isin(['Augmentation', 'Automation'])).astype(int)
L = L.groupby('uid').filter(lambda g: len(g) >= 5)

def efi_of(s):
    sw = np.ones(len(s)); sw[:-1][(s[:-1] == 1) & (s[1:] == 1)] = 0; return sw.mean()

g = {u: (gg['executed'].to_numpy(), gg['exposed'].to_numpy(), gg['e1'].to_numpy(),
         gg['category'].iloc[0], gg['framework'].iloc[0]) for u, gg in L.groupby('uid', sort=False)}
panel = pd.DataFrame([{'unit': u, 'ai_fraction': v[0].mean(), 'ai_exposure': v[2].mean(),
                       'ai_exposure_E1E2': v[1].mean(), 'num_E1E2_tasks': float(v[1].sum()),
                       'fragmentation_index': efi_of(v[1]), 'category': str(v[3]),
                       'framework': str(v[4])} for u, v in g.items()])
for cc in ('category', 'framework'):
    panel[cc] = panel[cc].astype('object')
for cc in ['ai_fraction', 'ai_exposure', 'ai_exposure_E1E2', 'fragmentation_index', 'num_E1E2_tasks']:
    panel[cc] = (panel[cc] - panel[cc].mean()) / panel[cc].std()

star = lambda p: '***' if p < .01 else '**' if p < .05 else '*' if p < .1 else ''
base = 'ai_fraction ~ fragmentation_index + ai_exposure + num_E1E2_tasks'
mods = [smf.ols(f, panel).fit(cov_type='HC1') for f in (base, base + ' + C(category)', base + ' + C(framework)')]
row = lambda lab, v: [f"{lab} & " + " & ".join(f"{m.params[v]:.2f}{star(m.pvalues[v])}" for m in mods) + r" \\",
                      " & " + " & ".join(f"({m.bse[v]:.2f})" for m in mods) + r" \\"]
tex = [r"\setlength{\tabcolsep}{12pt}", r"\begin{tabular}{lccc}", r"\toprule",
       r" & \multicolumn{3}{c}{Share of AI-executed Tasks} \\", r"\cmidrule(lr){2-4}",
       r" & (1) & (2) & (3) \\", r"\midrule", r"\addlinespace"]
tex += row("Share of AI-exposed Tasks", 'ai_exposure') + [r"\addlinespace"]
tex += row("Empirical Fragmentation Index (Definition 1)", 'fragmentation_index')
tex += [r"\hline\\[-1.25em]", r"Fixed Effect & & PCF Category & Framework \\",
        r"Number of AI-able Steps Control & \checkmark & \checkmark & \checkmark \\",
        "R-squared & " + " & ".join(f"{m.rsquared:.2f}" for m in mods) + r" \\",
        "Observations & " + " & ".join(f"{int(m.nobs)}" for m in mods) + r" \\", r"\bottomrule",
        r"\multicolumn{4}{l}{\footnotesize Standardized coefficients. Robust standard errors in parentheses. "
        r"*** p$<$0.01, ** p$<$0.05, * p$<$0.1} \\", r"\end{tabular}"]
mine = "\n".join(tex) + "\n"
open(f"{OUT}/apqc_published_REPLICATED.tex", 'w').write(mine)
pub = open(SNAP).read()
print("published snapshot has", len(pub.splitlines()), "lines; my replication has", len(mine.splitlines()))
if mine == pub:
    print("BYTE-FOR-BYTE IDENTICAL to the published table")
else:
    # the snapshot has been hand-trimmed of the footnote row; compare the tabular body
    pb = [l for l in pub.splitlines() if not l.startswith(r"\multicolumn{4}")]
    mb = [l for l in mine.splitlines() if not l.startswith(r"\multicolumn{4}")]
    print("identical after dropping the footnote row:", pb == mb)
    for d in difflib.unified_diff(pub.splitlines(), mine.splitlines(), 'published', 'mine', lineterm='', n=0):
        print("   ", d)

print("\nunrounded published-spec values:")
for nm, m in zip(['(1)', '(2)', '(3)'], mods):
    print(f"  {nm} EFI {m.params['fragmentation_index']:+.6f} se {m.bse['fragmentation_index']:.6f} "
          f"p {m.pvalues['fragmentation_index']:.6f} | exposure {m.params['ai_exposure']:+.6f} "
          f"se {m.bse['ai_exposure']:.6f} p {m.pvalues['ai_exposure']:.6f} | R2 {m.rsquared:.6f} n {int(m.nobs)}")

print("\nspec C, pooled: both exposure regressors together, with the step-count control")
cbase = 'ai_fraction ~ fragmentation_index + ai_exposure + ai_exposure_E1E2 + num_E1E2_tasks'
for nm, f in [('no FE', cbase), ('Category FE', cbase + ' + C(category)'), ('Framework FE', cbase + ' + C(framework)')]:
    m = smf.ols(f, panel).fit(cov_type='HC1')
    print(f"  {nm:<14} EFI {m.params['fragmentation_index']:+.3f}{star(m.pvalues['fragmentation_index']):<3} "
          f"(se {m.bse['fragmentation_index']:.3f}, p {m.pvalues['fragmentation_index']:.3f})  "
          f"E1 {m.params['ai_exposure']:+.3f}{star(m.pvalues['ai_exposure'])}  "
          f"E1|E2 {m.params['ai_exposure_E1E2']:+.3f}{star(m.pvalues['ai_exposure_E1E2'])}  R2 {m.rsquared:.3f}")
