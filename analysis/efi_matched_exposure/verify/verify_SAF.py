"""
INDEPENDENT SPOT-CHECK, SA.F (External validation on APQC PCF), Prediction #2 leg.

Written from scratch by the verification agent. The repo is read only and writeup/tables is never written; output goes to data/computed_objects/efi_matched_exposure/. No published exhibit is touched.

Construction reimplemented from analysis/apqc_pooled_predictions.py:
  SRC        = data/computed_objects/apqc_pred3_industry/industry_leaf_matches.csv
  sort key   = ['uid', sk] with sk = tuple(int(x) for x in hid.split('.'))  (documented order)
  category   = hid.split('.')[0]
  SIM_FLOOR  = 0.73, MIN_STEPS = 5 (groups with fewer steps dropped)
  carried    = similarity >= SIM_FLOOR   (steps below the floor are KEPT but coded unlabeled)
  exposed    = carried & human_labels in {E1,E2}      -> EFI Definition 1 input
  e1         = carried & human_labels == 'E1'         -> PUBLISHED exposure regressor
  executed   = carried & label in {Augmentation, Automation}   -> outcome
  EFI        = mean over positions of sw, sw[i]=0 iff seq[i]==1 and seq[i+1]==1, else 1
  panel      = one row per uid: ai_fraction, ai_exposure, num_E1E2_tasks, EFI, category, framework
  z-score all four, then OLS with cov_type='HC1' (NOT clustered)
  three columns: no FE, + C(category), + C(framework)

PUBLISHED  exposure regressor = e1 share
MATCHED    exposure regressor = exposed share (E1|E2), everything else unchanged
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import os as _os
# Repo root and output dir are derived from this file's own location, so the folder
# can move and no home directory is baked in. Generated output follows the repo
# convention: data/computed_objects/<name of this analysis folder>/.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
OUT = _os.path.join(REPO, "data", "computed_objects", "efi_matched_exposure")
_os.makedirs(OUT, exist_ok=True)


SRC = os.path.join(REPO, "data/computed_objects/apqc_pred3_industry/industry_leaf_matches.csv")

SIM_FLOOR = 0.73
MIN_STEPS = 5

L = pd.read_csv(SRC, dtype={"hid": str})
L["sk2"] = L["hid"].map(lambda h: tuple(int(x) for x in h.split(".")))
L = L.sort_values(["uid", "sk2"]).reset_index(drop=True)
L["category"] = L["hid"].str.split(".").str[0]

carried = L["similarity"] >= SIM_FLOOR
L["exposed"] = (carried & L["human_labels"].isin(["E1", "E2"])).astype(int)
L["e1"] = (carried & (L["human_labels"] == "E1")).astype(int)
L["executed"] = (carried & L["label"].isin(["Augmentation", "Automation"])).astype(int)

L = L.groupby("uid").filter(lambda g: len(g) >= MIN_STEPS)

print("=" * 90)
print("SAMPLE")
print("=" * 90)
print(f"floor {SIM_FLOOR} | {L.uid.nunique():,} groups | {len(L):,} steps | "
      f"{len(L)/L.uid.nunique():.1f} steps per group")
print(f"  frameworks: {L['framework'].nunique()}  ({sorted(L['framework'].unique())[:4]} ...)")
print(f"  AI-exposed (E1|E2) {L.exposed.mean()*100:.1f}%   E1 {L.e1.mean()*100:.1f}%   "
      f"AI-executed {L.executed.mean()*100:.1f}%")
print(f"  steps clearing the floor: {int((L['similarity']>=SIM_FLOOR).sum()):,}")

seq_exec = {u: g["executed"].to_numpy() for u, g in L.groupby("uid", sort=False)}
seq_exp = {u: g["exposed"].to_numpy() for u, g in L.groupby("uid", sort=False)}
seq_e1 = {u: g["e1"].to_numpy() for u, g in L.groupby("uid", sort=False)}
cat_of = L.groupby("uid")["category"].first().to_dict()
fw_of = L.groupby("uid")["framework"].first().to_dict()
units = list(seq_exec)


def efi_of(seq):
    sw = np.ones(len(seq))
    sw[:-1][(seq[:-1] == 1) & (seq[1:] == 1)] = 0
    return sw.mean()


panel0 = pd.DataFrame([{
    "unit": u, "num_steps": len(seq_exec[u]),
    "ai_fraction": seq_exec[u].mean(),
    "exp_E1": seq_e1[u].mean(),
    "exp_E1E2": seq_exp[u].mean(),
    "num_E1E2_tasks": float(seq_exp[u].sum()),
    "fragmentation_index": efi_of(seq_exp[u]),
    "category": str(cat_of[u]), "framework": str(fw_of[u]),
} for u in units])
panel0.to_csv(os.path.join(OUT, "ind_saf_panel.csv"), index=False)


def fit(exposure_col, p_in=None):
    p = (panel0 if p_in is None else p_in).copy()
    p["ai_exposure"] = p[exposure_col]
    for c in ("category", "framework"):
        p[c] = p[c].astype("object")
    for c in ["ai_fraction", "ai_exposure", "fragmentation_index", "num_E1E2_tasks"]:
        p[c] = (p[c] - p[c].mean()) / p[c].std()
    base = "ai_fraction ~ fragmentation_index + ai_exposure + num_E1E2_tasks"
    return [smf.ols(f, p).fit(cov_type="HC1")
            for f in (base, base + " + C(category)", base + " + C(framework)")]


def star(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


names = ["(1) no FE", "(2) PCF Category FE", "(3) Framework FE"]
store = {}
for leg, col in [("PUBLISHED_E1", "exp_E1"), ("MATCHED_E1E2", "exp_E1E2")]:
    mods = fit(col)
    store[leg] = mods
    print("\n" + "=" * 90)
    print(f"{leg}  (exposure regressor = {col})")
    print("=" * 90)
    for nm, m in zip(names, mods):
        ci = m.conf_int()
        print(f"{nm:<22} EFI {m.params['fragmentation_index']:+.6f} "
              f"({m.bse['fragmentation_index']:.6f}) p {m.pvalues['fragmentation_index']:.6f} "
              f"{star(m.pvalues['fragmentation_index']):<3} "
              f"CI [{ci.loc['fragmentation_index',0]:+.3f}, {ci.loc['fragmentation_index',1]:+.3f}]")
        print(f"{'':<22} EXP {m.params['ai_exposure']:+.6f} ({m.bse['ai_exposure']:.6f}) "
              f"p {m.pvalues['ai_exposure']:.6f} {star(m.pvalues['ai_exposure']):<3} "
              f"CI [{ci.loc['ai_exposure',0]:+.3f}, {ci.loc['ai_exposure',1]:+.3f}]  "
              f"| R2 {m.rsquared:.6f}  n {int(m.nobs)}")

# ---- rebuild the exact LaTeX the emitter would write, for both legs ----
def emit(mods):
    row = lambda lab, v: [f"{lab} & " + " & ".join(f"{m.params[v]:.2f}{star(m.pvalues[v])}" for m in mods) + r" \\",
                          " & " + " & ".join(f"({m.bse[v]:.2f})" for m in mods) + r" \\"]
    tex = [r"\setlength{\tabcolsep}{12pt}", r"\begin{tabular}{lccc}", r"\toprule",
           r" & \multicolumn{3}{c}{Share of AI-executed Tasks} \\", r"\cmidrule(lr){2-4}",
           r" & (1) & (2) & (3) \\", r"\midrule", r"\addlinespace"]
    tex += row("Share of AI-exposed Tasks", "ai_exposure") + [r"\addlinespace"]
    tex += row("Empirical Fragmentation Index (Definition 1)", "fragmentation_index")
    tex += [r"\hline\\[-1.25em]",
            r"Fixed Effect & & PCF Category & Framework \\",
            r"Number of AI-able Steps Control & \checkmark & \checkmark & \checkmark \\",
            "R-squared & " + " & ".join(f"{m.rsquared:.2f}" for m in mods) + r" \\",
            "Observations & " + " & ".join(f"{int(m.nobs)}" for m in mods) + r" \\",
            r"\bottomrule",
            r"\multicolumn{4}{l}{\footnotesize Standardized coefficients. Robust standard errors in parentheses. "
            r"*** p$<$0.01, ** p$<$0.05, * p$<$0.1} \\",
            r"\end{tabular}"]
    return "\n".join(tex) + "\n"


pub_tex = emit(store["PUBLISHED_E1"])
open(os.path.join(OUT, "ind_saf_published_regen.tex"), "w").write(pub_tex)
open(os.path.join(OUT, "ind_saf_matched_regen.tex"), "w").write(emit(store["MATCHED_E1E2"]))

published_on_disk = open(os.path.join(REPO, "writeup/tables/apqc_fragmentation_index_regression.tex")).read()
snap = open(os.path.join(os.path.join(_HERE, "..", "published_reference"), "apqc_fragmentation_index_regression.tex")).read()
print("\n" + "=" * 90)
print("LATEX BYTE COMPARISON")
print("=" * 90)
print("repo working-tree table == OLD_SNAPSHOT table :", published_on_disk == snap)
print("my regenerated PUBLISHED table == on-disk     :", pub_tex == published_on_disk)
a = pub_tex.rstrip("\n").split("\n")
b = published_on_disk.rstrip("\n").split("\n")
print(f"  line counts: mine {len(a)}, on disk {len(b)}")
extra = [l for l in a if l not in b]
missing = [l for l in b if l not in a]
print("  lines in mine but not on disk:", extra)
print("  lines on disk but not in mine:", missing)
a2 = [l for l in a if not l.startswith(r"\multicolumn{4}{l}{\footnotesize")]
print("  after dropping the emitter footnote row, identical:", a2 == b)

# ---- why it behaves differently from O*NET: EFI decomposition diagnostics ----
print("\n" + "=" * 90)
print("EFI = 1 - k/m + r/m  DECOMPOSITION DIAGNOSTICS (APQC pooled)")
print("=" * 90)
km = panel0["exp_E1E2"]
efi = panel0["fragmentation_index"]
rm = efi - 1 + km
import statsmodels.api as sm
r2 = sm.OLS(efi, sm.add_constant(km)).fit().rsquared
resid_sd = sm.OLS(efi, sm.add_constant(km)).fit().resid.std(ddof=1)
print(f"  mean E1|E2 share k/m      {km.mean():.4f}")
print(f"  mean r/m                  {rm.mean():.4f}")
print(f"  EFI mean / sd             {efi.mean():.4f} / {efi.std():.4f}")
print(f"  corr(EFI, k/m)            {np.corrcoef(efi, km)[0,1]:.4f}")
print(f"  R2 of EFI on k/m          {r2:.4f}")
print(f"  sd of EFI net of k/m      {resid_sd:.4f}  ({resid_sd/efi.std()*100:.0f}% of raw)")
adj = (panel0["num_E1E2_tasks"] - rm * panel0["num_steps"])
print(f"  adjacent AI-able pairs (k-r) per group: mean {adj.mean():.2f}, "
      f"share zero {float((adj<=0).mean()):.2%}")

# ---- within-group reshuffle placebo on the EFI coefficient (no published counterpart) ----
print("\n" + "=" * 90)
print("WITHIN-GROUP RESHUFFLE PLACEBO on the EFI coefficient (no FE), 1000 draws")
print("=" * 90)
rng = np.random.default_rng(20260901)
for leg, col in [("PUBLISHED_E1", "exp_E1"), ("MATCHED_E1E2", "exp_E1E2")]:
    obs = store[leg][0].params["fragmentation_index"]
    draws = []
    for _ in range(1000):
        efi_d = np.array([efi_of(rng.permutation(seq_exp[u])) for u in units])
        p = panel0.copy()
        p["fragmentation_index"] = efi_d
        draws.append(fit(col, p)[0].params["fragmentation_index"])
    draws = np.array(draws)
    z = (obs - draws.mean()) / draws.std(ddof=1)
    print(f"  {leg:<14} observed {obs:+.4f} | null mean {draws.mean():+.4f} sd {draws.std(ddof=1):.4f} "
          f"| z {z:+.2f} | draws below observed {int((draws<obs).sum())}/1000")
print("\nwrote", os.path.join(OUT, "ind_saf_panel.csv"))
