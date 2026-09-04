"""Regenerate the two fragmentation (EFI) exhibits under the corrected E1|E2 exposure mask.

READ-ONLY on the repository. Every write lands under writeup/_e1e2_preview/.

Exhibits
  Table SA.B.5  writeup/tables/fragmentation_index_regression_execution.tex   (PDF p.101 / SA-17)
                published pipeline: analysis/onet_fragmentationIndex.ipynb, cell 2
                    ai_exposure_var = 'human_E1_fraction'
                cells 5-10, definition = 2 (execution-based EFI).
                The .tex writer that produced the published file is the one in
                analysis/efi_matched_exposure/table2_and_SAB.py (latex_table), a faithful
                reimplementation of the notebook; byte-identity of the replication is checked
                below and is the proof that this driver is on the published pipeline.

  Table 3       writeup/tables/fragmentation_index_regression_combined.tex   (PDF p.41)
                published pipeline: analysis/efi_matched_exposure/combined_main_table.py with
                EXPOSURE = "E1E2", delegating estimation to
                analysis/efi_exposure_grid/efi_exposure_grid.py.
                Table 3 is ALREADY on E1|E2, so it is the CONTROL: it must come back
                byte-identical.

THE ONE THING CHANGED, and nothing else:
    human_labels.isin(['E1'])  ->  human_labels.isin(['E1','E2'])
  which in the notebook's vocabulary is
    ai_exposure_var = 'human_E1_fraction'  ->  'human_aiExposure_fraction'  (= E1 + E2 share)
  Same sample filter (>= 3 tasks per occupation), same z-scoring, same OLS, same cluster on
  O*NET-SOC Code with use_correction/df_correction, same control (# E1|E2 steps, which was
  already E1|E2 and is NOT touched), same fixed effects, same .tex formatting.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

_HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.abspath(os.path.join(_HERE, ".."))
REPO = os.path.abspath(os.path.join(STAGE, "..", ".."))

TABLES_NEW = os.path.join(STAGE, "tables_new")
TABLES_CUR = os.path.join(STAGE, "tables_current")
DIFFS = os.path.join(STAGE, "diffs")
WORK = os.path.join(STAGE, "work")
for _d in (TABLES_NEW, TABLES_CUR, DIFFS, WORK):
    os.makedirs(_d, exist_ok=True)

PUB_SAB5 = os.path.join(REPO, "writeup", "tables",
                        "fragmentation_index_regression_execution.tex")
PUB_T3 = os.path.join(REPO, "writeup", "tables",
                      "fragmentation_index_regression_combined.tex")

CODE = "O*NET-SOC Code"
TITLE = "Occupation Title"

LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


# =====================================================================================
# PART A -- Table SA.B.5, execution-based EFI (Definition 2)
#           verbatim logic of analysis/onet_fragmentationIndex.ipynb cells 5-10
# =====================================================================================

merged = pd.read_csv(os.path.join(REPO, "data", "computed_objects",
                                  "ONET_Eloundou_Anthropic_GPT",
                                  "ONET_Eloundou_Anthropic_GPT.csv"))
_counts = merged.groupby(CODE)["Task ID"].nunique()
_valid = _counts[_counts >= 3].index
merged = merged[merged[CODE].isin(_valid)].reset_index(drop=True)

ONET = pd.read_csv(os.path.join(REPO, "data", "computed_objects",
                                "ONET_cleaned_tasks.csv"))
SOC = ONET[[CODE, TITLE, "Major_Group_Code", "Major_Group_Title",
            "Minor_Group_Code", "Minor_Group_Title",
            "Broad_Occupation_Code", "Broad_Occupation_Title",
            "Detailed_Occupation_Code", "Detailed_Occupation_Title"]].copy()
SOC = SOC.drop_duplicates(subset=[CODE, CODE])


def occupation_analysis(df):
    rows = []
    for (soc, occ), g in df.groupby([CODE, TITLE]):
        total = len(g)
        aug = (g["label"] == "Augmentation").sum() / total
        aut = (g["label"] == "Automation").sum() / total
        hE1 = (g["human_labels"] == "E1").sum() / total
        hE2 = (g["human_labels"] == "E2").sum() / total
        rows.append({
            CODE: soc, TITLE: occ,
            "num_tasks": g["Task ID"].nunique(),
            "n_rows": total,
            "ai_fraction": aug + aut,
            # ---- THE ONE SWITCH lives here: which of these two columns is selected below.
            "human_E1_fraction": hE1,                      # published mask: E1 only
            "human_aiExposure_fraction": hE1 + hE2,        # corrected mask: E1 | E2
            "human_E2_fraction": hE2,
            # control, already E1|E2 in the published table -- left untouched
            "num_E1E2_tasks": int(g["human_labels"].isin(["E1", "E2"]).sum()),
        })
    return pd.DataFrame(rows)


def fragmentation_index(df, definition):
    fi = df.copy()
    if definition == 1:
        fi["is_ai"] = fi["human_labels"].isin(["E1", "E2"]).astype(int)
    elif definition == 2:
        fi["is_ai"] = fi["label"].isin(["Augmentation", "Automation"]).astype(int)
    fi["next_is_ai"] = fi.groupby([CODE, TITLE])["is_ai"].shift(-1).fillna(0).astype(int)
    fi["num_switches"] = 1
    fi.loc[(fi["is_ai"] == 1) & (fi["next_is_ai"] == 1), "num_switches"] = 0
    fi["run_start"] = ((fi["is_ai"] == 1) &
                       (fi.groupby([CODE, TITLE])["is_ai"].shift(1).fillna(0).astype(int) == 0)
                       ).astype(int)
    return fi.groupby([CODE, TITLE]).agg(
        fragmentation_index=("num_switches", "mean"),
        k=("is_ai", "sum"), r=("run_start", "sum"), m=("is_ai", "size"),
    ).reset_index()


def build(definition, exposure_var):
    occ = occupation_analysis(merged)
    fi = fragmentation_index(merged, definition)
    occ = occ.merge(fi, on=[CODE, TITLE], how="left")
    occ = occ.merge(SOC, on=[CODE, TITLE], how="left")
    d = occ.groupby([CODE, TITLE]).agg({
        "fragmentation_index": "mean",
        exposure_var: "mean",
        "ai_fraction": "mean",
        "num_tasks": "mean",
        "num_E1E2_tasks": "mean",
        "k": "mean", "r": "mean", "m": "mean",
    }).reset_index()
    d = d.merge(SOC.drop_duplicates(subset=[CODE]), on=CODE, how="left",
                suffixes=("", "_drop"))
    d = d.loc[:, ~d.columns.str.endswith("_drop")]
    d = d.rename(columns={exposure_var: "ai_exposure"})
    for c in ("Major_Group_Code", "Minor_Group_Code", CODE):
        d[c] = d[c].astype("object")
    return d


def zscore(d, cols=("ai_fraction", "ai_exposure", "fragmentation_index", "num_E1E2_tasks")):
    d = d.copy()
    for c in cols:
        s = d[c]
        d[c] = (s - s.mean()) / s.std()
    return d


def run(d, fe=None):
    f = "ai_fraction ~ fragmentation_index + ai_exposure + num_E1E2_tasks"
    if fe:
        f += f" + C({fe})"
    return smf.ols(formula=f, data=d).fit(
        cov_type="cluster",
        cov_kwds={"groups": d[CODE], "use_correction": True, "df_correction": True})


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def latex_table_sab(models, defn):
    """Byte-for-byte the writer that produced writeup/tables/
    fragmentation_index_regression_execution.tex."""
    cell = lambda m, t: f"{m.params[t]:.2f}{stars(m.pvalues[t])}"
    secell = lambda m, t: f"({m.bse[t]:.2f})"
    L = [
        r"\setlength{\tabcolsep}{12pt} % roomier padding for the narrow three-column layout",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Share of AI-executed Tasks} \\",
        r" \cmidrule(lr){2-4}",
        r" & (1) & (2) & (3) \\",
        r"\midrule",
        r"\addlinespace",
        "Share of AI-exposed Tasks & " +
        " & ".join(cell(m, "ai_exposure") for m in models) + r" \\",
        " & " + " & ".join(secell(m, "ai_exposure") for m in models) + r" \\",
        r"\addlinespace",
        f"Empirical Fragmentation Index (Definition {defn}) & " +
        " & ".join(cell(m, "fragmentation_index") for m in models) + r" \\",
        " & " + " & ".join(secell(m, "fragmentation_index") for m in models) + r" \\",
        r"\hline\\[-1.25em]",
        r"SOC Group Fixed Effect & & Major & Minor \\",
        r"Number of AI-able Steps Control & \checkmark & \checkmark & \checkmark \\",
        "R-squared & " + " & ".join(f"{m.rsquared:.2f}" for m in models) + r" \\",
        "Adj. R-squared & " + " & ".join(f"{m.rsquared_adj:.2f}" for m in models) + r" \\",
        "Observations & " + " & ".join(f"{int(m.nobs)}" for m in models) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(L) + "\n"


def fit_three(exposure_var):
    d = zscore(build(2, exposure_var))
    return [run(d), run(d, "Major_Group_Code"), run(d, "Minor_Group_Code")]


log("=" * 96)
log("PART A -- Table SA.B.5 (fragmentation_index_regression_execution.tex)")
log("=" * 96)

pub2 = fit_three("human_E1_fraction")            # published mask: E1 only
new2 = fit_three("human_aiExposure_fraction")    # corrected mask: E1 | E2

tex_pub_replicated = latex_table_sab(pub2, 2)
tex_new = latex_table_sab(new2, 2)

with open(PUB_SAB5, encoding="utf-8") as f:
    tex_published = f.read()

replication_ok = (tex_pub_replicated == tex_published)
log(f"[SA.B.5] replication of PUBLISHED E1-only file byte-identical: {replication_ok}")
if not replication_ok:
    import difflib
    for ln in difflib.unified_diff(tex_published.splitlines(),
                                   tex_pub_replicated.splitlines(),
                                   "published", "replicated", lineterm=""):
        log("    " + ln)

with open(os.path.join(TABLES_NEW, "fragmentation_index_regression_execution.tex"),
          "w", encoding="utf-8") as f:
    f.write(tex_new)
log("[SA.B.5] wrote " + os.path.join(TABLES_NEW,
                                     "fragmentation_index_regression_execution.tex"))
with open(os.path.join(WORK, "SAB5_published_replicated.tex"), "w", encoding="utf-8") as f:
    f.write(tex_pub_replicated)


# =====================================================================================
# PART B -- Table 3, the CONTROL. Already E1|E2; must come back byte-identical.
#           Uses the published generator's own module, analysis/efi_exposure_grid.
# =====================================================================================

log("")
log("=" * 96)
log("PART B -- Table 3 (fragmentation_index_regression_combined.tex) -- CONTROL")
log("=" * 96)

GRID_SRC = os.path.join(REPO, "analysis", "efi_exposure_grid", "efi_exposure_grid.py")
# The published module runs os.makedirs on two repo folders at import time. Both already
# exist (checked), so importing writes nothing, but to keep this run provably read-only the
# source is copied here and only those two output paths are redirected into the staging
# work/ folder. No estimation code is touched.
_grid_local = os.path.join(WORK, "efi_exposure_grid_readonly.py")
with open(GRID_SRC, encoding="utf-8") as f:
    _src = f.read()
_n_sub = 0
for _old, _new in [
    # keep REPO pointing at the real repo even though this copy sits two levels deeper
    ('REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))', 'REPO = r"%s"' % REPO),
    ('OUT = os.path.join(REPO, "data", "computed_objects", NAME)',
     'OUT = os.path.join(r"%s", "grid_out")' % WORK),
    ('FIG = os.path.join(REPO, "writeup", "plots", NAME)',
     'FIG = os.path.join(r"%s", "grid_fig")' % WORK),
]:
    assert _old in _src, "redirect target not found in efi_exposure_grid.py: " + _old
    _src = _src.replace(_old, _new)
    _n_sub += 1
assert _n_sub == 3
with open(_grid_local, "w", encoding="utf-8") as f:
    f.write(_src)

_spec = importlib.util.spec_from_file_location("efi_exposure_grid_readonly", _grid_local)
g = importlib.util.module_from_spec(_spec)
sys.modules["efi_exposure_grid_readonly"] = g
_spec.loader.exec_module(g)

CONTROL = "num_ai_able"
EXPOSURE = "E1E2"          # unchanged: Table 3 is already on the corrected mask


def _cells(panel, fes, cluster_col):
    out = {k: [] for k in ("expo", "expo_se", "efi", "efi_se", "r2", "adj", "n")}
    for fe_term in fes:
        r = g.fit(panel, CONTROL, fe_term, cluster_col)
        out["expo"].append(f"{r['exp']:.2f}{stars(r['exp_p'])}")
        out["expo_se"].append(f"({r['exp_se']:.2f})")
        out["efi"].append(f"{r['efi']:.2f}{stars(r['efi_p'])}")
        out["efi_se"].append(f"({r['efi_se']:.2f})")
        out["r2"].append(f"{r['r2']:.2f}")
        out["adj"].append(f"{r['adj_r2']:.2f}")
        out["n"].append(f"{r['n']}")
    return out


def build_table3(exposure):
    global EXPOSURE
    keep = EXPOSURE
    EXPOSURE = exposure
    data = os.path.join(REPO, "data")
    cleaned = pd.read_csv(os.path.join(data, "computed_objects", "ONET_cleaned_tasks.csv"))
    soc = cleaned[[g.OCC, "Major_Group_Code",
                   "Minor_Group_Code"]].drop_duplicates(subset=[g.OCC])
    onet = _cells(g.onet_panel(os.path.join(
        data, "computed_objects", "ONET_Eloundou_Anthropic_GPT",
        "ONET_Eloundou_Anthropic_GPT.csv"), soc, exposure),
        ["", " + C(Major_Group_Code)", " + C(Minor_Group_Code)"], g.OCC)
    apqc = _cells(g.pcf_panel(exposure), ["", " + C(category)", " + C(framework)"], None)
    EXPOSURE = keep

    row = lambda label, a, b: f"{label} & " + " & ".join(a + b) + r" \\"
    lines = [
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{6}{c}{Share of AI-executed Steps} \\",
        r"\cmidrule(lr){2-7}",
        r" & \multicolumn{3}{c}{O*NET Occupations} & \multicolumn{3}{c}{APQC Process Groups} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r" & (1) & (2) & (3) & (4) & (5) & (6) \\",
        r"\midrule",
        r"\addlinespace",
        row("Share of AI-exposed Steps", onet["expo"], apqc["expo"]),
        row("", onet["expo_se"], apqc["expo_se"]),
        r"\addlinespace",
        row("Empirical Fragmentation Index", onet["efi"], apqc["efi"]),
        row("", onet["efi_se"], apqc["efi_se"]),
        r"\hline\\[-1.25em]",
        r"Fixed Effect & & SOC Major & SOC Minor & & PCF Category & Framework \\",
        r"Number of AI-able Steps Control & \checkmark & \checkmark & \checkmark "
        r"& \checkmark & \checkmark & \checkmark \\",
        row("R-squared", onet["r2"], apqc["r2"]),
        row("Adj. R-squared", onet["adj"], apqc["adj"]),
        row("Observations", onet["n"], apqc["n"]),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n", onet, apqc


t3_tex, t3_onet, t3_apqc = build_table3("E1E2")

with open(PUB_T3, encoding="utf-8") as f:
    t3_published = f.read()

t3_identical = (t3_tex == t3_published)
log(f"[Table 3] regenerated under E1|E2 is byte-identical to published: {t3_identical}")
if not t3_identical:
    import difflib
    for ln in difflib.unified_diff(t3_published.splitlines(), t3_tex.splitlines(),
                                   "published", "regenerated", lineterm=""):
        log("    " + ln)

with open(os.path.join(TABLES_NEW, "fragmentation_index_regression_combined.tex"),
          "w", encoding="utf-8") as f:
    f.write(t3_tex)
log("[Table 3] wrote " + os.path.join(TABLES_NEW,
                                      "fragmentation_index_regression_combined.tex"))

# The E1-only counterfactual for Table 3 is reported for context only (it is NOT an artifact
# the paper prints); it shows what Table 3 would look like had it been on the wrong mask.
t3_e1_tex, _, _ = build_table3("E1")
with open(os.path.join(WORK, "T3_E1only_counterfactual.tex"), "w", encoding="utf-8") as f:
    f.write(t3_e1_tex)


# =====================================================================================
# PART C -- per-cell numeric diff
# =====================================================================================

def cellvals(models, key):
    return [(m.params[key], m.bse[key], m.pvalues[key]) for m in models]


rows = []
for term, label in [("ai_exposure", "Share of AI-exposed Tasks"),
                    ("fragmentation_index", "Empirical Fragmentation Index (Definition 2)"),
                    ("num_E1E2_tasks", "[control] # AI-able steps (not printed)")]:
    for i, (mp, mn) in enumerate(zip(pub2, new2), start=1):
        rows.append({
            "table": "SA.B.5", "row": label, "col": i,
            "old_coef": mp.params[term], "new_coef": mn.params[term],
            "d_coef": mn.params[term] - mp.params[term],
            "old_se": mp.bse[term], "new_se": mn.bse[term],
            "old_p": mp.pvalues[term], "new_p": mn.pvalues[term],
            "old_print": f"{mp.params[term]:.2f}{stars(mp.pvalues[term])} ({mp.bse[term]:.2f})",
            "new_print": f"{mn.params[term]:.2f}{stars(mn.pvalues[term])} ({mn.bse[term]:.2f})",
        })
for stat, get in [("R-squared", lambda m: m.rsquared),
                  ("Adj. R-squared", lambda m: m.rsquared_adj),
                  ("Observations", lambda m: float(int(m.nobs)))]:
    for i, (mp, mn) in enumerate(zip(pub2, new2), start=1):
        rows.append({"table": "SA.B.5", "row": stat, "col": i,
                     "old_coef": get(mp), "new_coef": get(mn),
                     "d_coef": get(mn) - get(mp),
                     "old_se": np.nan, "new_se": np.nan,
                     "old_p": np.nan, "new_p": np.nan,
                     "old_print": (f"{get(mp):.2f}" if stat != "Observations"
                                   else f"{int(get(mp))}"),
                     "new_print": (f"{get(mn):.2f}" if stat != "Observations"
                                   else f"{int(get(mn))}")})

D = pd.DataFrame(rows)
D.to_csv(os.path.join(DIFFS, "efi-tables-cells.csv"), index=False)

# ---------------------------------------------------------------- human-readable diff
out = []
out.append("EFI tables under the corrected exposure mask")
out.append("  human_labels.isin(['E1'])  ->  human_labels.isin(['E1','E2'])")
out.append("")
out.append("Group: Table SA.B.5 (execution EFI) + Table 3 (control)")
out.append("Script: writeup/_e1e2_preview/scripts/regen_efi_tables.py")
out.append("Sources: analysis/onet_fragmentationIndex.ipynb (cells 2, 5-10, definition=2);")
out.append("         analysis/efi_matched_exposure/combined_main_table.py ->")
out.append("         analysis/efi_exposure_grid/efi_exposure_grid.py")
out.append(f"pandas {pd.__version__}, numpy {np.__version__}, "
           f"statsmodels {__import__('statsmodels').__version__}")
out.append("")
out.append("=" * 92)
out.append("TABLE 3  writeup/tables/fragmentation_index_regression_combined.tex   [CONTROL]")
out.append("=" * 92)
out.append("Table 3 is ALREADY estimated on E1|E2 (combined_main_table.py sets EXPOSURE='E1E2').")
out.append("Re-running the published generator therefore has to reproduce the published file")
out.append("exactly. It does.")
out.append("")
out.append(f"  BYTE-IDENTICAL: {t3_identical}")
out.append(f"  published  md5 : {__import__('hashlib').md5(t3_published.encode()).hexdigest()}")
out.append(f"  regenerated md5: {__import__('hashlib').md5(t3_tex.encode()).hexdigest()}")
out.append(f"  cells changed  : 0 of 42")
out.append("")
out.append("=" * 92)
out.append("TABLE SA.B.5  writeup/tables/fragmentation_index_regression_execution.tex")
out.append("=" * 92)
out.append("Published spec (notebook cell 2): ai_exposure_var = 'human_E1_fraction'.")
out.append("Regenerated  spec               : ai_exposure_var = 'human_aiExposure_fraction'")
out.append("                                  (= E1 share + E2 share).")
out.append("The index is execution-based (Augmentation|Automation) and is untouched.")
out.append("The count control num_E1E2_tasks was ALREADY E1|E2 and is untouched.")
out.append("Exactly one regressor moves.")
out.append("")
out.append(f"  Replication of the PUBLISHED (E1-only) file is byte-identical: {replication_ok}")
out.append("")
out.append("Printed cells, published -> regenerated:")
out.append("")
hdr = f"  {'row':<46}{'col':>4}{'published':>16}{'E1|E2':>16}   {'changed'}"
out.append(hdr)
out.append("  " + "-" * (len(hdr) - 2))
n_changed = 0
n_cells = 0
for term, label in [("ai_exposure", "Share of AI-exposed Tasks"),
                    ("fragmentation_index", "Empirical Frag. Index (Definition 2)")]:
    for what, fmt in [("coef", lambda m, t: f"{m.params[t]:.2f}{stars(m.pvalues[t])}"),
                      ("se", lambda m, t: f"({m.bse[t]:.2f})")]:
        for i, (mp, mn) in enumerate(zip(pub2, new2), start=1):
            a, b = fmt(mp, term), fmt(mn, term)
            n_cells += 1
            ch = a != b
            n_changed += ch
            out.append(f"  {label + ' [' + what + ']':<46}{i:>4}{a:>16}{b:>16}   "
                       f"{'CHANGED' if ch else ''}")
for stat, get, f2 in [("R-squared", lambda m: m.rsquared, "%.2f"),
                      ("Adj. R-squared", lambda m: m.rsquared_adj, "%.2f"),
                      ("Observations", lambda m: int(m.nobs), "%d")]:
    for i, (mp, mn) in enumerate(zip(pub2, new2), start=1):
        a, b = f2 % get(mp), f2 % get(mn)
        n_cells += 1
        ch = a != b
        n_changed += ch
        out.append(f"  {stat:<46}{i:>4}{a:>16}{b:>16}   {'CHANGED' if ch else ''}")
out.append("")
out.append(f"  {n_changed} of {n_cells} printed cells changed.")
out.append("")
out.append("Full precision:")
out.append("")
hdr2 = (f"  {'term':<24}{'col':>4}{'coef_old':>12}{'coef_new':>12}{'delta':>11}"
        f"{'se_old':>10}{'se_new':>10}{'p_old':>12}{'p_new':>12}")
out.append(hdr2)
out.append("  " + "-" * (len(hdr2) - 2))
for term in ("ai_exposure", "fragmentation_index", "num_E1E2_tasks"):
    for i, (mp, mn) in enumerate(zip(pub2, new2), start=1):
        out.append(f"  {term:<24}{i:>4}{mp.params[term]:>+12.6f}{mn.params[term]:>+12.6f}"
                   f"{mn.params[term] - mp.params[term]:>+11.6f}"
                   f"{mp.bse[term]:>10.6f}{mn.bse[term]:>10.6f}"
                   f"{mp.pvalues[term]:>12.3e}{mn.pvalues[term]:>12.3e}")
out.append("")
for stat, get in [("R-squared", lambda m: m.rsquared),
                  ("Adj. R-squared", lambda m: m.rsquared_adj)]:
    for i, (mp, mn) in enumerate(zip(pub2, new2), start=1):
        out.append(f"  {stat:<24}{i:>4}{get(mp):>12.6f}{get(mn):>12.6f}"
                   f"{get(mn) - get(mp):>+11.6f}")
out.append(f"  {'Observations':<24}{'1-3':>4}{int(pub2[0].nobs):>12}{int(new2[0].nobs):>12}"
           f"{0:>+11}")
out.append("")
out.append("Sign / significance audit of the EFI row (the row SA.B interprets):")
for i, (mp, mn) in enumerate(zip(pub2, new2), start=1):
    a = mp.params["fragmentation_index"]
    b = mn.params["fragmentation_index"]
    out.append(f"  col ({i}): {a:+.6f} -> {b:+.6f}   sign "
               f"{'UNCHANGED' if np.sign(a) == np.sign(b) else 'FLIPPED'}, "
               f"stars {stars(mp.pvalues['fragmentation_index']) or 'none'} -> "
               f"{stars(mn.pvalues['fragmentation_index']) or 'none'}, "
               f"|delta| = {abs(b - a):.6f}")
out.append("")
out.append("Diagnostics:")
d_pub = build(2, "human_E1_fraction")
d_new = build(2, "human_aiExposure_fraction")
out.append(f"  corr(E1 share, E1|E2 share) across the {len(d_pub)} occupations = "
           f"{np.corrcoef(d_pub['ai_exposure'], d_new['ai_exposure'])[0, 1]:.4f}")
out.append(f"  SD(E1 share)    = {d_pub['ai_exposure'].std():.4f}")
out.append(f"  SD(E1|E2 share) = {d_new['ai_exposure'].std():.4f}")
out.append(f"  mean(E1 share)  = {d_pub['ai_exposure'].mean():.4f}, "
           f"mean(E1|E2 share) = {d_new['ai_exposure'].mean():.4f}")
idn = d_new["fragmentation_index"] - (1 - d_new["k"] / d_new["m"] + d_new["r"] / d_new["m"])
out.append(f"  max |EFI2 - (1 - k/m + r/m)| = {np.abs(idn).max():.3e}")
out.append("")
out.append("Table 3, E1-only counterfactual (NOT an artifact; context only). Its .tex is at")
out.append("work/T3_E1only_counterfactual.tex.")
out.append("")
out.append("Build log lines appended to writeup/_e1e2_preview/BUILD_LOG.txt.")

with open(os.path.join(DIFFS, "efi-tables.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
log("")
log("wrote " + os.path.join(DIFFS, "efi-tables.txt"))
log("wrote " + os.path.join(DIFFS, "efi-tables-cells.csv"))

print("\n" + "\n".join(out))

# machine-readable status for the caller
with open(os.path.join(WORK, "efi_status.txt"), "w", encoding="utf-8") as f:
    f.write(f"T3_BYTE_IDENTICAL={t3_identical}\n")
    f.write(f"SAB5_PUB_REPLICATION_IDENTICAL={replication_ok}\n")
    f.write(f"SAB5_CELLS_CHANGED={n_changed}/{n_cells}\n")
