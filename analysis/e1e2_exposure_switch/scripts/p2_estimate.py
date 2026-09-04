"""Table 2 / Figure OA.A.1 observed estimates under E1-only and under E1|E2.

Estimation logic copied VERBATIM from cell 17 of
analysis/onet_antrhopicIndex_execTypeVaryingDWA.ipynb (build_ame_df, run_regressions_on,
generate_latex_table).  Exactly one thing differs from the published pipeline:

    cell 13:  merged_data['is_exposed'] = merged_data['human_labels'].isin(['E1'])
    here   :  merged_data['is_exposed'] = merged_data['human_labels'].isin(['E1','E2'])

The estimation sample is the paper's own committed frame
    data/computed_objects/execTypeVaryingDWA_anthropicIndex_noTasksWithRepetitiveDWAs/
        similarTasks_allEligibleTasks.csv          (10,708 rows / 1,748 DWAs)
which cell 13 writes.  Nothing in the sample construction touches `is_exposed`
(it is a right-hand-side control, never a filter), so re-deriving `is_exposed`
from the `human_labels` column already in that file changes the mask and nothing else.

The E1 branch is run as a control: its .tex must come back byte-identical to
writeup/tables/noTasksWithRepetitiveDWAs/allTasks_ai.tex.

READ-ONLY on the repo outside writeup/_e1e2_preview/.
"""
import os
import sys
import time
import warnings
from io import StringIO
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

REPO = "/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin"
STAGE = f"{REPO}/writeup/_e1e2_preview"
WORK = f"{STAGE}/work/t2f1"
SAMPLE = (f"{REPO}/data/computed_objects/execTypeVaryingDWA_anthropicIndex"
          f"_noTasksWithRepetitiveDWAs/similarTasks_allEligibleTasks.csv")

# ---- notebook cell 4 / cell 5 constants, verbatim -------------------------------------
dependent_var = "is_ai"
TARGET_REGS = ["prev2_is_ai", "prev_is_ai", "next_is_ai", "next2_is_ai"]
SPECS = ["no_fe_no_dwa", "major_fe_no_dwa", "minor_fe_no_dwa", "no_fe_with_dwa",
         "no_fe_no_dwa_withTaskDWACount", "no_fe_with_dwa_withTaskDWACount"]
VAR_LABELS = {
    "prev2_is_ai": "Task ($k-2$) is AI",
    "prev_is_ai": "Task ($k-1$) is AI",
    "next_is_ai": "Task ($k+1$) is AI",
    "next2_is_ai": "Task ($k+2$) is AI",
}
TABLE_VAR_LABELS = {
    "prev2_is_ai": "Task ($k-2$) is AI-executed",
    "prev_is_ai": "Task ($k-1$) is AI-executed",
    "next_is_ai": "Task ($k+1$) is AI-executed",
    "next2_is_ai": "Task ($k+2$) is AI-executed",
}


# ======================================================================================
# cell 17, section 1: build_ame_df -- verbatim except for the bootstrap frame assembly
# ======================================================================================
def build_ame_df(res, df_used, dataset_name, model_name, target_regs, fe_label, dwa_fe,
                 formula, calculate_standard_errors=True, cluster_col="DWA ID",
                 B=200, seed=123):
    try:
        np.random.seed(seed)

        pr2 = res.prsquared
        k = res.params.shape[0]
        adj_pr2 = 1 - (res.llf - k) / res.llnull
        nobs = int(res.nobs)

        df_base = df_used.copy()
        df_base.columns = df_base.columns.str.strip()

        # ---------- point estimates ----------
        ame_point = {}
        for var in target_regs:
            if var not in df_base.columns:
                continue
            df1 = df_base.copy()
            df0 = df_base.copy()
            df1[var] = 1
            df0[var] = 0
            p1 = res.predict(df1)
            p0 = res.predict(df0)
            ame_point[var] = np.mean(p1 - p0)

        ame_se = {v: 0.0 for v in ame_point.keys()}
        ame_p = {v: 0.0 for v in ame_point.keys()}

        if calculate_standard_errors:
            ame_boot = {v: [] for v in ame_point.keys()}
            clusters = df_base[cluster_col].unique()
            # SPEED ONLY.  The notebook assembles each bootstrap frame as
            #   pd.concat([df_base[df_base[cluster_col] == c] for c in sampled_clusters],
            #             ignore_index=True)
            # which rescans the frame len(clusters) times per replicate.  Pre-indexing the
            # cluster row positions and .take()-ing them in the same order gives the same
            # rows in the same order with the same dtypes.  The RNG stream is untouched:
            # np.random.choice is still called once per replicate, before anything else.
            # Verified below by the byte-identical E1 control table.
            _pos = {c: np.flatnonzero((df_base[cluster_col] == c).to_numpy()) for c in clusters}
            for _ in range(B):
                sampled_clusters = np.random.choice(clusters, size=len(clusters), replace=True)
                df_b = df_base.take(
                    np.concatenate([_pos[c] for c in sampled_clusters])
                ).reset_index(drop=True)
                try:
                    res_b = smf.logit(formula, data=df_b).fit(disp=False)
                    for var in ame_point.keys():
                        df1 = df_b.copy()
                        df0 = df_b.copy()
                        df1[var] = 1
                        df0[var] = 0
                        p1 = res_b.predict(df1)
                        p0 = res_b.predict(df0)
                        ame_boot[var].append(np.mean(p1 - p0))
                except Exception:
                    continue

            for var in ame_point.keys():
                if len(ame_boot[var]) > 1:
                    se = np.std(ame_boot[var], ddof=1)
                    z = ame_point[var] / se if se > 0 else 0.0
                    p = 2 * (1 - norm.cdf(abs(z))) if se > 0 else 0.0
                else:
                    se, p = 0.0, 0.0
                ame_se[var] = se
                ame_p[var] = p

        rows = [{"term": v, "ame_coef": a, "ame_se": ame_se[v], "p_value": ame_p[v]}
                for v, a in ame_point.items()]
        ame_df = pd.DataFrame(rows)

        return pd.DataFrame({
            "dataset": dataset_name, "model": model_name, "fe_label": fe_label,
            "dwa_fe": dwa_fe, "nobs": nobs, "r2_pseudo": pr2, "r2_adj_pseudo": adj_pr2,
            "term": ame_df["term"], "ame_coef": ame_df["ame_coef"],
            "ame_se": ame_df["ame_se"], "p_value": ame_df["p_value"],
        })
    except Exception as e:
        print(f"Error calculating AME for {model_name}: {e}")
        return pd.DataFrame()


# ======================================================================================
# cell 17, section 2: run_regressions_on -- verbatim (output CSV redirected to staging)
# ======================================================================================
def run_regressions_on(df, dataset_name, dependent_var, regressors,
                       calculate_standard_errors=True, out_dir=None):
    df = df.copy()
    all_cols = regressors + [dependent_var, "is_exposed", "num_tasks", "DWA ID"]
    existing_cols = [c for c in all_cols if c in df.columns]
    numeric_cols = [c for c in existing_cols if c != "DWA ID"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    base_formula = f"{dependent_var} ~ " + " + ".join(regressors)
    ame_list = []
    models = {}

    # 1) No SOC FE, No DWA FE
    try:
        formula = base_formula + " + is_exposed + num_tasks"
        res = smf.logit(formula, data=df).fit(
            disp=False, cov_type="cluster",
            cov_kwds={"groups": df["DWA ID"], "use_correction": True})
        models["no_fe_no_dwa"] = res
        ame_list.append(build_ame_df(res, df, dataset_name, "no_fe_no_dwa", regressors,
                                     fe_label="None", dwa_fe=False, formula=formula,
                                     calculate_standard_errors=calculate_standard_errors))
    except Exception as e:
        print(f"[{dataset_name}] No-FE, No-DWA failed: {e}")

    # 2) Major FE, No DWA FE
    if "Major_Group_Code" in df.columns:
        try:
            formula = base_formula + " + C(Major_Group_Code) + is_exposed + num_tasks"
            df_fe_major = df.groupby("Major_Group_Code").filter(
                lambda g: g[dependent_var].nunique() > 1)
            res = smf.logit(formula, data=df_fe_major).fit(
                disp=False, cov_type="cluster",
                cov_kwds={"groups": df_fe_major["DWA ID"], "use_correction": True})
            models["major_fe_no_dwa"] = res
            ame_list.append(build_ame_df(res, df_fe_major, dataset_name, "major_fe_no_dwa",
                                         regressors, fe_label="Major Group", dwa_fe=False,
                                         formula=formula,
                                         calculate_standard_errors=calculate_standard_errors))
        except Exception as e:
            print(f"[{dataset_name}] Major FE, No-DWA failed: {e}")

    # 3) Minor FE, No DWA FE
    if "Minor_Group_Code" in df.columns:
        try:
            formula = base_formula + " + C(Minor_Group_Code) + is_exposed + num_tasks"
            df_fe_minor = df.groupby("Minor_Group_Code").filter(
                lambda g: g[dependent_var].nunique() > 1)
            res = smf.logit(formula, data=df_fe_minor).fit(
                disp=False, cov_type="cluster",
                cov_kwds={"groups": df_fe_minor["DWA ID"], "use_correction": True})
            models["minor_fe_no_dwa"] = res
            ame_list.append(build_ame_df(res, df_fe_minor, dataset_name, "minor_fe_no_dwa",
                                         regressors, fe_label="Minor Group", dwa_fe=False,
                                         formula=formula,
                                         calculate_standard_errors=calculate_standard_errors))
        except Exception as e:
            print(f"[{dataset_name}] Minor FE, No-DWA failed: {e}")

    # 4) No SOC FE, With DWA FE
    try:
        formula = base_formula + " + C(DWA_ID) + is_exposed + num_tasks"
        df["DWA_ID"] = df["DWA ID"]
        df_dwa = df.groupby("DWA_ID").filter(lambda g: g[dependent_var].nunique() > 1)
        res = smf.logit(formula, data=df_dwa).fit(
            disp=False, cov_type="cluster",
            cov_kwds={"groups": df_dwa["DWA_ID"], "use_correction": True})
        models["no_fe_with_dwa"] = res
        ame_list.append(build_ame_df(res, df_dwa, dataset_name, "no_fe_with_dwa", regressors,
                                     fe_label="None", dwa_fe=True, formula=formula,
                                     calculate_standard_errors=calculate_standard_errors))
    except Exception as e:
        print(f"[{dataset_name}] No-FE, With-DWA failed: {e}")

    # 5) No SOC FE, No DWA FE, With Same DWA Task Counts in Occupation
    try:
        formula = base_formula + " + is_exposed + num_tasks + num_tasks_in_dwa_within_occupation"
        df["DWA_ID"] = df["DWA ID"]
        res = smf.logit(formula, data=df_dwa).fit(
            disp=False, cov_type="cluster",
            cov_kwds={"groups": df_dwa["DWA_ID"], "use_correction": True})
        models["no_fe_no_dwa_withTaskDWACount"] = res
        ame_list.append(build_ame_df(res, df_dwa, dataset_name,
                                     "no_fe_no_dwa_withTaskDWACount", regressors,
                                     fe_label="None", dwa_fe=False, formula=formula,
                                     calculate_standard_errors=calculate_standard_errors))
    except Exception as e:
        print(f"[{dataset_name}] No-FE, With-DWA failed: {e}")

    # 6) No SOC FE, With DWA FE, With Same DWA Task Counts in Occupation
    try:
        formula = (base_formula
                   + " + C(DWA_ID) + is_exposed + num_tasks + num_tasks_in_dwa_within_occupation")
        df["DWA_ID"] = df["DWA ID"]
        df_dwa = df.groupby("DWA_ID").filter(lambda g: g[dependent_var].nunique() > 1)
        res = smf.logit(formula, data=df_dwa).fit(
            disp=False, cov_type="cluster",
            cov_kwds={"groups": df_dwa["DWA_ID"], "use_correction": True})
        models["no_fe_with_dwa_withTaskDWACount"] = res
        ame_list.append(build_ame_df(res, df_dwa, dataset_name,
                                     "no_fe_with_dwa_withTaskDWACount", regressors,
                                     fe_label="None", dwa_fe=True, formula=formula,
                                     calculate_standard_errors=calculate_standard_errors))
    except Exception as e:
        print(f"[{dataset_name}] No-FE, With-DWA failed: {e}")

    combined = pd.concat(ame_list, ignore_index=True) if ame_list else pd.DataFrame()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        combined.to_csv(f"{out_dir}/regression_ame_results_{dataset_name}.csv", index=False)
    return models, combined


# ======================================================================================
# cell 17, section 3: generate_latex_table
# ------------------------------------------------------------------------------------
# The committed writeup/tables/.../allTasks_ai.tex carries no trailing \footnotesize note
# (the paper puts the note in its own table environment), so the writer stops at
# \end{tabular}.  With that single omission this reproduces the committed file byte for
# byte from the published CSV -- asserted in main() before any E1|E2 output is written.
# ======================================================================================
def generate_latex_table(df_results, out_file):
    if df_results.empty:
        raise ValueError("The input DataFrame is empty. Cannot generate LaTeX table.")

    buf = StringIO()
    w = lambda s: buf.write(s + "\n")

    dataset_to_show = df_results["dataset"].unique()[0]
    subset = df_results[df_results["dataset"] == dataset_to_show].copy()

    w("% --- LaTeX Table for full_0 ---")

    def fmt(row):
        stars = ""
        p = row["p_value"]
        if pd.notna(p):
            if p < 0.01:
                stars = "***"
            elif p < 0.05:
                stars = "**"
            elif p < 0.10:
                stars = "*"
        return f"{row['ame_coef']:.2f}{stars}", f"({row['ame_se']:.2f})"

    formatted = subset.apply(fmt, axis=1, result_type="expand")
    subset["coef_str"] = formatted[0]
    subset["se_str"] = formatted[1]

    pivot_coef = subset.pivot(index="term", columns="model", values="coef_str")
    pivot_se = subset.pivot(index="term", columns="model", values="se_str")

    valid_vars = [v for v in TARGET_REGS if v in pivot_coef.index]
    pivot_coef = pivot_coef.reindex(valid_vars)
    pivot_se = pivot_se.reindex(valid_vars)

    model_order = ["no_fe_no_dwa", "major_fe_no_dwa", "minor_fe_no_dwa", "no_fe_with_dwa",
                   "no_fe_no_dwa_withTaskDWACount", "no_fe_with_dwa_withTaskDWACount"]
    valid_models = [m for m in model_order if m in pivot_coef.columns]

    stats = subset[["model", "nobs", "r2_pseudo", "r2_adj_pseudo", "fe_label", "dwa_fe"]] \
        .drop_duplicates("model").set_index("model")

    col_def = "l" + "c" * len(valid_models)
    w(f"\\begin{{tabular}}{{{col_def}}}")
    w(r"\toprule")

    header_nums = [f"({i+1})" for i in range(len(valid_models))]
    dep_label = {"is_ai": "AI-executed", "is_automated": "AI-automated"}[dependent_var]
    w(" & " + r"\multicolumn{" + str(len(valid_models)) + r"}{c}"
      + r"{Probability that Focal Task ($k$) is " + dep_label + r"} \\")
    w(r"\cmidrule(lr){2-" + str(len(valid_models) + 1) + "}")
    w(" & " + " & ".join(header_nums) + r" \\")
    w(r"\midrule")

    for var in valid_vars:
        label = TABLE_VAR_LABELS.get(var, VAR_LABELS.get(var, var.replace("_", " ")))
        c_vals = [pivot_coef.loc[var, m] if m in pivot_coef.columns else "" for m in valid_models]
        s_vals = [pivot_se.loc[var, m] if m in pivot_se.columns else "" for m in valid_models]
        w(f"{label} & " + " & ".join(c_vals) + r" \\")
        w(" & " + " & ".join(s_vals) + r" \\")
        w(r"\addlinespace")

    w(r"\midrule")
    w("Pseudo $R^2$ & " + " & ".join(
        f"{stats.loc[m, 'r2_pseudo']:.3f}" if m in stats.index else "" for m in valid_models)
      + r" \\")
    w("Observations & " + " & ".join(
        f"{int(stats.loc[m, 'nobs']):,}" if m in stats.index else "" for m in valid_models)
      + r" \\")

    fe_vals = []
    for m in valid_models:
        if m in stats.index:
            label = stats.loc[m, "fe_label"]
            if pd.isna(label) or str(label) == "None":
                fe_vals.append("")
            elif "Major" in str(label):
                fe_vals.append("Major")
            elif "Minor" in str(label):
                fe_vals.append("Minor")
            else:
                fe_vals.append(str(label))
        else:
            fe_vals.append("")
    w("SOC Group FE & " + " & ".join(fe_vals) + r" \\")

    dwa_vals = [(r"\checkmark" if (m in stats.index and stats.loc[m, "dwa_fe"]) else "")
                for m in valid_models]
    w("DWA FE & " + " & ".join(dwa_vals) + r" \\")

    cnt_vals = [(r"\checkmark" if (m in stats.index and "withTaskDWACount" in m) else "")
                for m in valid_models]
    w("NumTasks in DWA-Occupation Control & " + " & ".join(cnt_vals) + r" \\")

    w(r"\bottomrule")
    w(r"\end{tabular}")

    text = buf.getvalue()
    if out_file is not None:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            f.write(text)
    return text


# ======================================================================================
def load_sample(mask):
    """The paper's own estimation frame, with `is_exposed` rebuilt under `mask`."""
    d = pd.read_csv(SAMPLE)
    published = d["is_exposed"].to_numpy().copy()
    d["is_exposed"] = d["human_labels"].isin(mask).astype(int)
    if mask == ["E1"]:
        assert (d["is_exposed"].to_numpy() == published).all(), \
            "E1 rebuild does not reproduce the committed is_exposed column"
    return d


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    os.makedirs(WORK, exist_ok=True)

    # ---- control 0: the generator reproduces the committed .tex from the published CSV
    pub_csv = (f"{REPO}/data/computed_objects/execTypeVaryingDWA_anthropicIndex"
               f"_noTasksWithRepetitiveDWAs/regression_summaries_is_ai/"
               f"regression_ame_results_full_0.csv")
    pub = pd.read_csv(pub_csv)
    committed = open(f"{REPO}/writeup/tables/noTasksWithRepetitiveDWAs/allTasks_ai.tex").read()
    regen = generate_latex_table(pub, f"{WORK}/table2_from_published_csv.tex")
    print(f"[control 0] generator(published CSV) == committed .tex : {regen == committed}",
          flush=True)
    assert regen == committed

    jobs = []
    if which in ("both", "e1"):
        jobs.append(("E1", ["E1"]))
    if which in ("both", "e12"):
        jobs.append(("E1E2", ["E1", "E2"]))

    for tag, mask in jobs:
        d = load_sample(mask)
        print(f">>> mask={mask}  exposed={int(d['is_exposed'].sum()):,}/{len(d):,} "
              f"({d['is_exposed'].mean():.2%})", flush=True)
        t0 = time.time()
        _, res = run_regressions_on(d, "full_0", dependent_var, TARGET_REGS,
                                    calculate_standard_errors=True, out_dir=WORK)
        res.to_csv(f"{WORK}/ame_full_0_{tag}.csv", index=False)
        generate_latex_table(res, f"{WORK}/table2_{tag}.tex")
        print(f"    wrote ame_full_0_{tag}.csv / table2_{tag}.tex in {time.time()-t0:.0f}s",
              flush=True)

    if which in ("both", "e1"):
        mine = open(f"{WORK}/table2_E1.tex").read()
        print(f"[control 1] my E1 rerun .tex == committed .tex : {mine == committed}",
              flush=True)


if __name__ == "__main__":
    main()
