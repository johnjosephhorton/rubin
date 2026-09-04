"""Rebuild the Figure SA.D.4 input table (`allTasks_ai.csv`) under a chosen exposure mask,
INCLUDING the B=200 cluster bootstrap that produces the figure's 90% error bars.

Estimation logic copied out of
  analysis/onet_antrhopicIndex_execTypeVaryingDWA_robustness.ipynb  cells 8, 10, 12, 14   (prompts 1-10)
  analysis/onet_antrhopicIndex_execTypeVaryingDWA.ipynb             cells 13, 17, 18      (prompt 0)
with ONE change:
      merged_data['is_exposed'] = merged_data['human_labels'].isin(['E1'])
  ->  merged_data['is_exposed'] = merged_data['human_labels'].isin(EXPOSURE_LABELS)

Everything else is held fixed: same sample filters, same six-spec ladder, same DWA-clustered
cov_type, same B=200, same seed=123, same np.random.choice bootstrap over DWA clusters.

The one implementation change is a pure-performance rewrite of the bootstrap's row gather:
      pd.concat([df_base[df_base[cluster_col] == c] for c in sampled], ignore_index=True)
  ->  df_base.iloc[np.concatenate([pos[c] for c in sampled])].reset_index(drop=True)
which builds the byte-identical resampled frame (same rows, same order) far faster. It is
validated by the E1 fixture: the rebuilt SEs must equal the published ones to the last digit.

Only the four specifications SA.D.4 actually plots are estimated
(no_fe_no_dwa, major_fe_no_dwa, minor_fe_no_dwa, no_fe_with_dwa).

Writes ONLY under writeup/_e1e2_preview/.
Usage: python3 sad4_estimate.py E1        # fixture, must reproduce the published table
       python3 sad4_estimate.py E1E2      # the rebuild
"""
import os, sys, glob, time
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
import warnings; warnings.filterwarnings('ignore')

REPO = "/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin"
STAGE = f"{REPO}/writeup/_e1e2_preview"
input_data_path = f"{REPO}/data"

MASK = sys.argv[1] if len(sys.argv) > 1 else 'E1E2'
EXPOSURE_LABELS = ['E1'] if MASK == 'E1' else ['E1', 'E2']

dependent_var = 'is_ai'
TARGET_REGS = ['prev2_is_ai', 'prev_is_ai', 'next_is_ai', 'next2_is_ai']
# SA.D.4 plots panels (a)-(d) = these four of the notebook's six specs.
PLOT_SPECS = ['no_fe_no_dwa', 'major_fe_no_dwa', 'minor_fe_no_dwa', 'no_fe_with_dwa']

# ---------------- notebook cells 10 / 12: DWA helper objects ----------------
dwa_task_mapping = pd.read_csv(f"{input_data_path}/computed_objects/similar_dwa_tasks/dwa_task_mapping.csv")
dwa_task_counts = dwa_task_mapping.groupby('Task ID')['DWA ID'].nunique().reset_index(name='num_dwas_per_task')
unique_task_dwa_mapping = dwa_task_counts[dwa_task_counts['num_dwas_per_task'] == 1]['Task ID'].tolist()

KEEP_COLS = ['O*NET-SOC Code', 'Occupation Title', 'Task ID', 'Task Title',
             'Task Position', 'Task Type',
             'Major_Group_Code', 'Major_Group_Title',
             'Minor_Group_Code', 'Minor_Group_Title',
             'Broad_Occupation_Code', 'Broad_Occupation_Title',
             'Detailed_Occupation_Code', 'Detailed_Occupation_Title',
             'gpt4_exposure', 'human_labels',
             'automation', 'augmentation', 'label']


def build_prompt_data(path):
    """Cell 14 (robustness) / cell 13 (main) preprocessing, verbatim apart from the mask line."""
    merged_data = pd.read_csv(path)
    merged_data = merged_data[KEEP_COLS]
    merged_data['is_ai'] = merged_data['label'].isin(['Augmentation', 'Automation']).astype(int)
    merged_data['is_automated'] = merged_data['label'].isin(['Automation']).astype(int)
    merged_data['is_exposed'] = merged_data['human_labels'].isin(EXPOSURE_LABELS).astype(int)  # <-- ONE CHANGED LINE

    n = merged_data.groupby('O*NET-SOC Code')['Task ID'].nunique().reset_index().rename(columns={'Task ID': 'num_tasks'})
    merged_data = merged_data.merge(n, on='O*NET-SOC Code', how='left')

    merged_data['Task Position'] = pd.to_numeric(merged_data['Task Position'], errors='coerce')
    merged_data = merged_data.sort_values(['O*NET-SOC Code', 'Task Position']).reset_index(drop=True)
    g = merged_data.groupby('O*NET-SOC Code')['is_ai']
    merged_data['prev_is_ai'] = g.shift(1); merged_data['prev2_is_ai'] = g.shift(2)
    merged_data['next_is_ai'] = g.shift(-1); merged_data['next2_is_ai'] = g.shift(-2)
    nb = ['prev_is_ai', 'prev2_is_ai', 'next_is_ai', 'next2_is_ai']
    merged_data = merged_data.dropna(subset=nb).reset_index(drop=True)
    for c in nb: merged_data[c] = merged_data[c].astype(int)

    merged_data = merged_data.merge(dwa_task_mapping,
                                    on=['Task ID', 'Task Title', 'O*NET-SOC Code', 'Occupation Title'], how='left')
    merged_data = merged_data[merged_data['Task ID'].isin(unique_task_dwa_mapping)].reset_index(drop=True)
    merged_data = merged_data.drop_duplicates(subset=['O*NET-SOC Code', 'Task ID']).reset_index(drop=True)
    merged_data = merged_data[~merged_data['DWA ID'].isna()].reset_index(drop=True)
    occ = merged_data.groupby('DWA ID')['O*NET-SOC Code'].nunique().reset_index(name='num_occupations')
    keep = occ[occ['num_occupations'] > 1]['DWA ID'].unique().tolist()
    merged_data = merged_data[merged_data['DWA ID'].isin(keep)].reset_index(drop=True)
    cnt = merged_data.groupby(['DWA ID', 'O*NET-SOC Code'])['Task ID'].nunique().reset_index(
        name='num_tasks_in_dwa_within_occupation')
    merged_data = merged_data.merge(cnt, on=['DWA ID', 'O*NET-SOC Code'], how='left')
    sc = merged_data.select_dtypes(include=['string']).columns
    merged_data[sc] = merged_data[sc].astype(object)
    return merged_data


# ---------------- notebook cell 8: build_ame_df (B=200 cluster bootstrap) ----------------
def build_ame_df(res, df_used, dataset_name, model_name, target_regs, fe_label, dwa_fe, formula,
                 cluster_col="DWA ID", B=200, seed=123):
    np.random.seed(seed)

    pr2 = res.prsquared
    k = res.params.shape[0]
    adj_pr2 = 1 - (res.llf - k) / res.llnull
    nobs = int(res.nobs)

    df_base = df_used.copy()
    df_base.columns = df_base.columns.str.strip()

    ame_point = {}
    for var in target_regs:
        if var not in df_base.columns:
            continue
        df1 = df_base.copy(); df0 = df_base.copy()
        df1[var] = 1; df0[var] = 0
        ame_point[var] = np.mean(res.predict(df1) - res.predict(df0))

    ame_se = {v: 0.0 for v in ame_point}
    ame_p = {v: 0.0 for v in ame_point}

    ame_boot = {v: [] for v in ame_point}
    clusters = df_base[cluster_col].unique()
    # positional index per cluster, in original row order -> identical rows to the notebook's
    # pd.concat([df_base[df_base[cluster_col] == c] for c in sampled], ignore_index=True)
    codes = pd.Series(np.arange(len(df_base)), index=df_base[cluster_col].values)
    pos = {c: codes.loc[[c]].values for c in clusters}

    for _ in range(B):
        sampled_clusters = np.random.choice(clusters, size=len(clusters), replace=True)
        df_b = df_base.iloc[np.concatenate([pos[c] for c in sampled_clusters])].reset_index(drop=True)
        try:
            res_b = smf.logit(formula, data=df_b).fit(disp=False)
            for var in ame_point:
                df1 = df_b.copy(); df0 = df_b.copy()
                df1[var] = 1; df0[var] = 0
                ame_boot[var].append(np.mean(res_b.predict(df1) - res_b.predict(df0)))
        except Exception:
            continue

    for var in ame_point:
        if len(ame_boot[var]) > 1:
            se = np.std(ame_boot[var], ddof=1)
            z = ame_point[var] / se if se > 0 else 0.0
            p = 2 * (1 - norm.cdf(abs(z))) if se > 0 else 0.0
        else:
            se, p = 0.0, 0.0
        ame_se[var] = se; ame_p[var] = p

    return pd.DataFrame({
        "dataset": dataset_name, "model": model_name, "fe_label": fe_label, "dwa_fe": dwa_fe,
        "nobs": nobs, "r2_pseudo": pr2, "r2_adj_pseudo": adj_pr2,
        "term": list(ame_point), "ame_coef": [ame_point[v] for v in ame_point],
        "ame_se": [ame_se[v] for v in ame_point], "p_value": [ame_p[v] for v in ame_point]})


# ---------------- notebook cell 8: run_regressions_on, restricted to the four plotted specs ----
def run_one(df, dataset_name, model_name):
    df = df.copy()
    all_cols = TARGET_REGS + [dependent_var, 'is_exposed', 'num_tasks', 'DWA ID']
    existing = [c for c in all_cols if c in df.columns]
    num = [c for c in existing if c != 'DWA ID']
    df[num] = df[num].apply(pd.to_numeric, errors='coerce')
    base_formula = f'{dependent_var} ~ ' + ' + '.join(TARGET_REGS)
    df['DWA_ID'] = df['DWA ID']

    if model_name == 'no_fe_no_dwa':
        formula = base_formula + ' + is_exposed + num_tasks'; data = df; cl = 'DWA ID'; fe = "None"; dwafe = False
    elif model_name == 'major_fe_no_dwa':
        formula = base_formula + ' + C(Major_Group_Code) + is_exposed + num_tasks'
        data = df.groupby("Major_Group_Code").filter(lambda g: g[dependent_var].nunique() > 1)
        cl = 'DWA ID'; fe = "Major Group"; dwafe = False
    elif model_name == 'minor_fe_no_dwa':
        formula = base_formula + ' + C(Minor_Group_Code) + is_exposed + num_tasks'
        data = df.groupby("Minor_Group_Code").filter(lambda g: g[dependent_var].nunique() > 1)
        cl = 'DWA ID'; fe = "Minor Group"; dwafe = False
    elif model_name == 'no_fe_with_dwa':
        formula = base_formula + ' + C(DWA_ID) + is_exposed + num_tasks'
        data = df.groupby("DWA_ID").filter(lambda g: g[dependent_var].nunique() > 1)
        cl = 'DWA_ID'; fe = "None"; dwafe = True
    else:
        raise ValueError(model_name)

    res = smf.logit(formula, data=data).fit(disp=False, cov_type='cluster',
                                            cov_kwds={'groups': data[cl], 'use_correction': True})
    return build_ame_df(res, data, dataset_name, model_name, TARGET_REGS,
                        fe_label=fe, dwa_fe=dwafe, formula=formula)


_CACHE = {}
def _job(args):
    ds, model_name = args
    if ds not in _CACHE:
        path = (f"{input_data_path}/computed_objects/ONET_Eloundou_Anthropic_GPT/ONET_Eloundou_Anthropic_GPT.csv"
                if ds == 0 else
                f"{input_data_path}/computed_objects/ONET_Eloundou_Anthropic_GPT/ONET_Eloundou_Anthropic_GPT_{ds}.csv")
        _CACHE[ds] = build_prompt_data(path)
    t0 = time.time()
    out = run_one(_CACHE[ds], ds, model_name)
    print(f"  [{MASK}] prompt {ds:>2} {model_name:<16} {time.time()-t0:6.1f}s", flush=True)
    return out


if __name__ == '__main__':
    from multiprocessing import Pool
    os.makedirs(f"{STAGE}/diffs", exist_ok=True)
    jobs = [(ds, m) for ds in range(0, 11) for m in PLOT_SPECS]
    t0 = time.time()
    with Pool(processes=int(os.environ.get('NPROC', '10'))) as p:
        parts = p.map(_job, jobs, chunksize=1)
    master = pd.concat(parts, ignore_index=True)
    f = f"{STAGE}/diffs/_sad4_allTasks_ai_{MASK}.csv"
    master.to_csv(f, index=False)
    print(f"[{MASK}] wrote {f} ({len(master)} rows) in {(time.time()-t0)/60:.1f} min")
