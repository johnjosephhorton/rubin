"""
SA.D GPT-prompt robustness: OLD spec vs MATCHED spec reimplementation.

Reimplements analysis/onet_fragmentationIndex_robustness.ipynb exactly, but runs the
regression twice per (prompt, FE) cell:
    OLD     : ai_exposure = human_E1_fraction          (published)
    MATCHED : ai_exposure = human_aiExposure_fraction  (E1|E2, same base as the EFI)

Output goes to data/computed_objects/efi_matched_exposure/. No published exhibit is touched.
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import os as _os
# Repo root and output dir are derived from this file's own location, so the folder
# can move and no home directory is baked in. Generated output follows the repo
# convention: data/computed_objects/<name of this analysis folder>/.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
OUT = _os.path.join(REPO, "data", "computed_objects", "efi_matched_exposure")
_os.makedirs(OUT, exist_ok=True)


DATA = os.path.join(REPO, "data")
OCC_CODE = "O*NET-SOC Code"
OCC_TITLE = "Occupation Title"


# ---------------------------------------------------------------- notebook code
def create_occupation_analysis(df):
    rows = []
    for (soc_code, occ_title), group in df.groupby([OCC_CODE, OCC_TITLE]):
        num_tasks = group["Task ID"].nunique()
        total = len(group)
        augmentation_fraction = (group["label"] == "Augmentation").sum() / total
        automation_fraction = (group["label"] == "Automation").sum() / total
        ai_fraction = augmentation_fraction + automation_fraction
        human_E1_fraction = (group["human_labels"] == "E1").sum() / total
        human_E2_fraction = (group["human_labels"] == "E2").sum() / total
        rows.append({
            OCC_CODE: soc_code,
            OCC_TITLE: occ_title,
            "num_tasks": num_tasks,
            "ai_fraction": ai_fraction,
            "human_E1_fraction": human_E1_fraction,
            "human_E2_fraction": human_E2_fraction,
            "human_aiExposure_fraction": human_E1_fraction + human_E2_fraction,
            "num_E1E2_tasks": int(group["human_labels"].isin(["E1", "E2"]).sum()),
        })
    return pd.DataFrame(rows)


def construct_fragmentation_index(df, desired_definition=1):
    fi = df.copy()
    if desired_definition == 1:
        fi["is_ai"] = fi["human_labels"].isin(["E1", "E2"]).astype(int)
    elif desired_definition == 2:
        fi["is_ai"] = fi["label"].isin(["Augmentation", "Automation"]).astype(int)
    fi["next_is_ai"] = fi.groupby([OCC_CODE, OCC_TITLE])["is_ai"].shift(-1).fillna(0).astype(int)
    fi["num_switches"] = 1
    fi.loc[(fi["is_ai"] == 1) & (fi["next_is_ai"] == 1), "num_switches"] = 0
    fi = fi.groupby([OCC_CODE, OCC_TITLE])["num_switches"].mean().reset_index()
    return fi.rename(columns={"num_switches": "fragmentation_index"})


def build_panel(merged_data, SOC_mappings, definition):
    occ = create_occupation_analysis(merged_data)
    fi = construct_fragmentation_index(merged_data, desired_definition=definition)
    occ = occ.merge(fi, on=[OCC_CODE, OCC_TITLE], how="left")
    occ = occ.merge(SOC_mappings, on=[OCC_CODE, OCC_TITLE], how="left")
    agg = occ.groupby([OCC_CODE, OCC_TITLE]).agg({
        "fragmentation_index": "mean",
        "human_E1_fraction": "mean",
        "human_aiExposure_fraction": "mean",
        "ai_fraction": "mean",
        "num_tasks": "mean",
        "num_E1E2_tasks": "mean",
    }).reset_index()
    agg = agg.merge(SOC_mappings.drop_duplicates(subset=[OCC_CODE]),
                    on=OCC_CODE, how="left", suffixes=("", "_drop"))
    agg = agg.loc[:, ~agg.columns.str.endswith("_drop")]
    for c in ("Major_Group_Code", "Minor_Group_Code", OCC_CODE):
        if c in agg.columns:
            agg[c] = agg[c].astype("object")
    return agg


FORMULAS = {
    "noFE":    "ai_fraction ~ fragmentation_index + ai_exposure + num_E1E2_tasks",
    "majorFE": "ai_fraction ~ fragmentation_index + ai_exposure + num_E1E2_tasks + C(Major_Group_Code)",
    "minorFE": "ai_fraction ~ fragmentation_index + ai_exposure + num_E1E2_tasks + C(Minor_Group_Code)",
}


def run_cell(agg, exposure_col):
    d = agg.copy().rename(columns={exposure_col: "ai_exposure"})
    for c in ["ai_fraction", "ai_exposure", "fragmentation_index", "num_E1E2_tasks"]:
        s = d[c]
        d[c] = (s - s.mean()) / s.std()
    groups = d[OCC_CODE]
    out = {}
    for name, f in FORMULAS.items():
        res = smf.ols(f, data=d).fit(
            cov_type="cluster",
            cov_kwds={"groups": groups, "use_correction": True, "df_correction": True})
        out[name] = res
    return out


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ---------------------------------------------------------------- run
ONET = pd.read_csv(os.path.join(DATA, "computed_objects", "ONET_cleaned_tasks.csv"))
SOC_mappings = ONET[[OCC_CODE, OCC_TITLE, "Major_Group_Code", "Major_Group_Title",
                     "Minor_Group_Code", "Minor_Group_Title", "Broad_Occupation_Code",
                     "Broad_Occupation_Title", "Detailed_Occupation_Code",
                     "Detailed_Occupation_Title"]].copy()
SOC_mappings = SOC_mappings.drop_duplicates(subset=[OCC_CODE, OCC_CODE])

base = os.path.join(DATA, "computed_objects", "ONET_Eloundou_Anthropic_GPT")
files = [os.path.join(base, "ONET_Eloundou_Anthropic_GPT.csv")]
for x in range(1, 11):
    files.append(os.path.join(base, f"ONET_Eloundou_Anthropic_GPT_{x}.csv"))

DEFS = [1, 2]
recs = []
for prompt, path in enumerate(files):
    merged = pd.read_csv(path)
    for definition in DEFS:
        agg = build_panel(merged, SOC_mappings, definition)
        for spec, col in (("OLD", "human_E1_fraction"),
                          ("MATCHED", "human_aiExposure_fraction")):
            fits = run_cell(agg, col)
            for model, res in fits.items():
                for term in ("fragmentation_index", "ai_exposure"):
                    recs.append({
                        "prompt": prompt, "definition": definition, "spec": spec,
                        "model": model, "term": term,
                        "coef": res.params[term], "se": res.bse[term],
                        "t": res.tvalues[term], "p": res.pvalues[term],
                        "ci_lo": res.conf_int().loc[term, 0],
                        "ci_hi": res.conf_int().loc[term, 1],
                        "nobs": int(res.nobs), "r2": float(res.rsquared),
                    })
    print(f"prompt {prompt} done  (N={int(recs[-1]['nobs'])})", flush=True)

res_df = pd.DataFrame(recs)
res_df.to_csv(os.path.join(OUT, "sad_prompt_robustness_old_vs_matched.csv"), index=False)


# ---------------------------------------------------------------- sanity: main-notebook sample
print("\n" + "=" * 100)
print("STEP 0. Does the robustness notebook's prompt 0 equal the MAIN-TEXT sample?")
merged0 = pd.read_csv(files[0])
tc = merged0.groupby(OCC_CODE)["Task ID"].nunique()
print(f"  occupations in prompt-0 file : {merged0[OCC_CODE].nunique()}")
print(f"  min tasks per occupation     : {tc.min()}  (main notebook filter is >= 3, "
      f"binds for {(tc < 3).sum()} occupations)")

# also verify the published CSV row for prompt 0 def 1 the main notebook wrote
print("\n" + "=" * 100)
print("STEP 1. REPLICATION, Definition 1, Prompt 0 (main prompt).")
p0 = res_df[(res_df.prompt == 0) & (res_df.definition == 1)]
print(f"{'spec':9s} {'model':8s} {'term':20s} {'coef':>9s} {'se':>9s} {'stars':>5s} {'N':>6s} {'R2':>7s}")
for spec in ("OLD", "MATCHED"):
    for model in ("noFE", "majorFE", "minorFE"):
        for term in ("fragmentation_index", "ai_exposure"):
            r = p0[(p0.spec == spec) & (p0.model == model) & (p0.term == term)].iloc[0]
            print(f"{spec:9s} {model:8s} {term:20s} {r.coef:9.3f} {r.se:9.3f} {stars(r.p):>5s} "
                  f"{int(r.nobs):6d} {r.r2:7.3f}")

TARGET_OLD = {"noFE": (-0.261, 0.073, 0.241, 0.046, 0.36),
              "majorFE": (-0.380, 0.064, 0.109, 0.042, 0.63),
              "minorFE": (-0.283, 0.071, 0.092, 0.047, 0.71)}
TARGET_NEW = {"noFE": (-0.007, 0.101, 0.494, 0.102),
              "majorFE": (-0.086, 0.086, 0.479, 0.094),
              "minorFE": (-0.040, 0.093, 0.390, 0.093)}
print("\n  Deviations from the numbers quoted in the task brief (main-text table):")
for model, (fc, fs, ec, es, r2) in TARGET_OLD.items():
    r = p0[(p0.spec == "OLD") & (p0.model == model)]
    f = r[r.term == "fragmentation_index"].iloc[0]
    e = r[r.term == "ai_exposure"].iloc[0]
    print(f"   OLD     {model:8s} EFI {f.coef:+.4f} vs {fc:+.3f} (d={f.coef-fc:+.4f}) | "
          f"SE {f.se:.4f} vs {fs:.3f} | exp {e.coef:+.4f} vs {ec:+.3f} | R2 {f.r2:.4f} vs {r2}")
for model, (fc, fs, ec, es) in TARGET_NEW.items():
    r = p0[(p0.spec == "MATCHED") & (p0.model == model)]
    f = r[r.term == "fragmentation_index"].iloc[0]
    e = r[r.term == "ai_exposure"].iloc[0]
    print(f"   MATCHED {model:8s} EFI {f.coef:+.4f} vs {fc:+.3f} (d={f.coef-fc:+.4f}) | "
          f"SE {f.se:.4f} vs {fs:.3f} | exp {e.coef:+.4f} vs {ec:+.3f}")


# ---------------------------------------------------------------- full table
def big_table(definition):
    print("\n" + "=" * 118)
    print(f"STEP 2. ALL ELEVEN PROMPTS, Definition {definition}. "
          f"EFI coefficient (clustered SE), OLD vs MATCHED; exposure coefficient below.")
    d = res_df[res_df.definition == definition]
    for term in ("fragmentation_index", "ai_exposure"):
        label = "EFI (fragmentation_index)" if term == "fragmentation_index" else "AI exposure"
        print(f"\n--- {label} ---")
        hdr = f"{'prompt':>6s} {'N':>5s} |"
        for model in ("noFE", "majorFE", "minorFE"):
            hdr += f"{model:>32s} |"
        print(hdr)
        print(f"{'':>6s} {'':>5s} |" + ("      OLD              MATCHED     |") * 3)
        for prompt in range(11):
            dp = d[(d.prompt == prompt) & (d.term == term)]
            n = int(dp.iloc[0].nobs)
            line = f"{prompt:>6d} {n:>5d} |"
            for model in ("noFE", "majorFE", "minorFE"):
                o = dp[(dp.spec == "OLD") & (dp.model == model)].iloc[0]
                m = dp[(dp.spec == "MATCHED") & (dp.model == model)].iloc[0]
                line += (f" {o.coef:+.3f} ({o.se:.3f}){stars(o.p):<3s}"
                         f" {m.coef:+.3f} ({m.se:.3f}){stars(m.p):<3s}|")
            print(line)
    return d


for definition in DEFS:
    d = big_table(definition)

    print("\n" + "=" * 118)
    print(f"STEP 3. COUNTS AND DISPERSION, Definition {definition} "
          f"(11 prompts per FE column).")
    frag = d[d.term == "fragmentation_index"]
    exp_ = d[d.term == "ai_exposure"]
    hdr = (f"{'FE':9s} {'spec':8s} | {'neg&sig5%':>9s} {'neg&sig10%':>10s} {'anySig5%':>8s} | "
           f"{'min':>7s} {'max':>7s} {'mean':>7s} {'sd':>7s} | {'exp>0&sig5%':>11s} {'exp mean':>8s}")
    print(hdr)
    for model in ("noFE", "majorFE", "minorFE"):
        for spec in ("OLD", "MATCHED"):
            f = frag[(frag.model == model) & (frag.spec == spec)]
            e = exp_[(exp_.model == model) & (exp_.spec == spec)]
            neg5 = int(((f.coef < 0) & (f.p < 0.05)).sum())
            neg10 = int(((f.coef < 0) & (f.p < 0.10)).sum())
            any5 = int((f.p < 0.05).sum())
            eps = int(((e.coef > 0) & (e.p < 0.05)).sum())
            print(f"{model:9s} {spec:8s} | {neg5:>4d}/11    {neg10:>5d}/11     {any5:>4d}/11  | "
                  f"{f.coef.min():+7.3f} {f.coef.max():+7.3f} {f.coef.mean():+7.3f} "
                  f"{f.coef.std(ddof=1):7.3f} | {eps:>7d}/11 {e.coef.mean():+8.3f}")

    # extra: how many matched EFI point estimates are even negative
    print("\n  sign counts of the MATCHED EFI point estimate (out of 11):")
    for model in ("noFE", "majorFE", "minorFE"):
        f = frag[(frag.model == model) & (frag.spec == "MATCHED")]
        print(f"   {model:9s} negative: {(f.coef < 0).sum():2d}   positive: {(f.coef > 0).sum():2d}   "
              f"90% CI excludes 0: {((f.coef - 1.645*f.se) * (f.coef + 1.645*f.se) > 0).sum():2d}")


# ---------------------------------------------------------------- figure geometry
print("\n" + "=" * 118)
print("STEP 4. WHAT THE REGENERATED FIGURE WOULD LOOK LIKE (Definition 1, 90% CI bars, "
      "hardcoded ylim (-0.65, 0.65)).")
YLO, YHI = -0.65, 0.65
d1 = res_df[res_df.definition == 1]
for term in ("ai_exposure", "fragmentation_index"):
    print(f"\n  row = {term}")
    for spec in ("OLD", "MATCHED"):
        for model in ("noFE", "majorFE", "minorFE"):
            s = d1[(d1.term == term) & (d1.spec == spec) & (d1.model == model)]
            lo = (s.coef - 1.645 * s.se)
            hi = (s.coef + 1.645 * s.se)
            clipped_lo = int((lo < YLO).sum())
            clipped_hi = int((hi > YHI).sum())
            pt_clip = int(((s.coef < YLO) | (s.coef > YHI)).sum())
            print(f"    {spec:8s} {model:8s} pts [{s.coef.min():+.3f}, {s.coef.max():+.3f}] "
                  f"mean {s.coef.mean():+.3f} | whiskers [{lo.min():+.3f}, {hi.max():+.3f}] "
                  f"| points outside ylim: {pt_clip} | whisker ends clipped: "
                  f"lo {clipped_lo}, hi {clipped_hi}")

print("\nwrote:", os.path.join(OUT, "sad_prompt_robustness_old_vs_matched.csv"))
