"""Re-render Figure SA.E.4 (neighbor placebo forests across frequency cuts) under a chosen
exposure mask.

Plotting and statistics code copied VERBATIM out of analysis/onet_placebo_summary.py
(functions `stats`, `ordi`, `panel_bycut`, `forest_3col`, the legend handles, figsize,
dpi=190, tight_layout rect, w_pad, marker sizes, colours). The row labels NOCC_N / NOBS_N
come from `nbr_counts` and are mask-independent, so they are computed exactly as published.

The ONLY thing that differs between the two renders is the pair of placebo tables fed in,
which themselves differ only by
    human_labels.isin(['E1'])  ->  human_labels.isin(['E1','E2'])
inside `build_pool` of analysis/onet_neighborAI_weeklyTasks_placebo.ipynb (cell 5).

Placebo inputs:
  published : data/computed_objects/.../placebo_null_draws.csv + placebo_observed.csv
              (the cached files the published PNG was literally drawn from)
  E1  rebuild / E1|E2 rebuild : the audited 1,000-draw reshuffle in
              sae4_placebo_null_E1_vs_E1E2.csv + sae4_placebo_observed_E1_vs_E1E2.csv,
              same mechanic and same seeds (random_state = 42 + i, i = 0..999).

Writes ONLY under writeup/_e1e2_preview/.
Usage: python3 render_sae4_forests.py
"""
import os, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = "/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin"
STAGE = f"{REPO}/writeup/_e1e2_preview"
M6 = ("/private/tmp/claude-502/-Users-peymansh-MIT-Dropbox-Peyman-Shahidi-GitHub-rubin/"
      "5af7d286-51e7-470a-b57e-4e1373499eea/scratchpad/m6")

nd = f"{REPO}/data/computed_objects/execTypeVaryingDWA_anthropicIndex_weeklyTasks"
mf = f"{REPO}/data/computed_objects/ONET_Eloundou_Anthropic_GPT/ONET_Eloundou_Anthropic_GPT.csv"

# ================= verbatim from onet_placebo_summary.py =================
CUTS = [('all', 'All tasks')]
for fam, lab in [('daily', 'Daily+'), ('sevdaily', 'SeveralDaily+'), ('hourly', 'Hourly+')]:
    for t in [20, 35, 50, 65]:
        CUTS.append((f"{fam}{t}", f"{lab} >={t}%"))
SPEC_N = {'no_fe_no_dwa': 'No FE', 'major_fe_no_dwa': 'Major FE', 'minor_fe_no_dwa': 'Minor FE'}
TERM_N = {'prev2_is_ai': '(t-2)', 'prev_is_ai': '(t-1)', 'next_is_ai': '(t+1)', 'next2_is_ai': '(t+2)'}

mg = pd.read_csv(mf); mg['Task Position'] = pd.to_numeric(mg['Task Position'], errors='coerce')
FTm = {'daily': ['FT_Daily', 'FT_Several times daily', 'FT_Hourly or more'],
       'sevdaily': ['FT_Several times daily', 'FT_Hourly or more'], 'hourly': ['FT_Hourly or more']}
def parse(tag):
    if tag == 'all': return None, None
    m = re.match(r'([a-z]+)(\d+)', tag); return FTm[m.group(1)], int(m.group(2))
dwa = pd.read_csv(f"{REPO}/data/computed_objects/similar_dwa_tasks/dwa_task_mapping.csv")
uniq = dwa.groupby('Task ID')['DWA ID'].nunique().reset_index(name='n'); uniq = uniq[uniq.n == 1]['Task ID'].tolist()
ALL_FT = ['FT_Yearly or less', 'FT_More than yearly', 'FT_More than monthly', 'FT_More than weekly',
          'FT_Daily', 'FT_Several times daily', 'FT_Hourly or more']
pp = mg[['O*NET-SOC Code', 'Occupation Title', 'Task ID', 'Task Title', 'Task Position', 'label'] + ALL_FT].copy()
pp['is_ai'] = pp['label'].isin(['Augmentation', 'Automation']).astype(int)
pp = pp.merge(dwa, on=['Task ID', 'Task Title', 'O*NET-SOC Code', 'Occupation Title'], how='left')
pp = pp[pp['Task ID'].isin(uniq)].drop_duplicates(subset=['O*NET-SOC Code', 'Task ID']); pp = pp[~pp['DWA ID'].isna()]
occ = pp.groupby('DWA ID')['O*NET-SOC Code'].nunique(); pp = pp[pp['DWA ID'].isin(occ[occ > 1].index)]
def nbr_counts(tag):
    cols, thr = parse(tag); d = pp if cols is None else pp[pp[cols].sum(axis=1) >= thr]
    d = d.sort_values(['O*NET-SOC Code', 'Task Position']); g = d.groupby('O*NET-SOC Code')['is_ai']
    fl = pd.concat([g.shift(1), g.shift(2), g.shift(-1), g.shift(-2)], axis=1); keep = d[fl.notna().all(axis=1)]
    return int(keep['O*NET-SOC Code'].nunique()), int(len(keep))
_nb = {t: nbr_counts(t) for t, _ in CUTS}
NOCC_N = {t: v[0] for t, v in _nb.items()}; NOBS_N = {t: v[1] for t, v in _nb.items()}

def stats(v, o):
    v = np.asarray([x for x in v if not np.isnan(x)], float)
    if not len(v) or np.isnan(o): return dict(mean=np.nan, sd=np.nan, p10=np.nan, p90=np.nan, pct=np.nan, p=np.nan)
    return dict(mean=v.mean(), sd=v.std(ddof=1), p10=np.percentile(v, 10), p90=np.percentile(v, 90),
                pct=100 * np.mean(v < o), p=(1 + np.sum(np.abs(v) >= abs(o))) / (1 + len(v)))
def ordi(n):
    if np.isnan(n): return ""
    n = int(round(n)); s = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10 if n % 100 not in (11, 12, 13) else 0, 'th'); return f"{n}{s}"

AME_UNIT = 'Average marginal effect on P(focal task is AI)'
leg = [Line2D([0], [0], color='0.55', lw=5, alpha=0.5, label='Placebo null 10–90%'),
       Line2D([0], [0], marker='|', ls='', ms=12, mew=2, color='0.35', label='Placebo mean'),
       Line2D([0], [0], marker='o', ls='', ms=9, color='crimson', mec='k', label='Observed (outside 10–90% null)'),
       Line2D([0], [0], marker='o', ls='', ms=9, color='steelblue', mec='k', label='Observed (within 10–90% null)')]
BYCUT_SPECS = ['No FE', 'Major FE', 'Minor FE']
SPEC_TITLE = {'No FE': 'No fixed effects', 'Major FE': 'SOC major-group FE', 'Minor FE': 'SOC minor-group FE'}

def panel_bycut(ax, data, unit, title, ylabels):
    n = len(data); ys = list(range(n))[::-1]
    for (clab, r, nocc), y in zip(data, ys):
        if r is None or np.isnan(r['Observed']):
            continue
        ax.plot([r['p10'], r['p90']], [y, y], color='0.55', lw=5, alpha=0.5, solid_capstyle='round', zorder=1)
        ax.plot(r['mean'], y, marker='|', ms=14, mew=2, color='0.35', zorder=2)
        sig = (not np.isnan(r['p10'])) and (r['Observed'] < r['p10'] or r['Observed'] > r['p90'])
        ax.plot(r['Observed'], y, 'o', ms=9, color=('crimson' if sig else 'steelblue'), markeredgecolor='k', mew=0.6, zorder=3)
        ax.text(1.015, y, ordi(r['pct']), transform=ax.get_yaxis_transform(), clip_on=False,
                va='center', ha='left', fontsize=7, color='dimgray')
    ax.axvline(0, color='k', lw=1, alpha=0.5)
    ax.axhline(n - 1.5, color='black', lw=1)
    ax.set_yticks(ys)
    if ylabels:
        ax.set_yticklabels([f"{clab}  (N={nocc})" for clab, _, nocc in data], fontsize=8.5)
    else:
        ax.tick_params(labelleft=False)
    ax.set_xlabel(unit, fontsize=9); ax.grid(axis='x', ls=':', alpha=0.5); ax.margins(x=0.22)
    ax.set_title(title, fontsize=10.5, fontweight='bold')
# ======================= end verbatim block =======================


def build_df(nn, no):
    """`rows` loop of onet_placebo_summary.py, restricted to the Neighbor exercise
    (the two SA.E.4 figures never touch the Fragmentation rows)."""
    rows = []
    for ct, cl in CUTS:
        for tm in ['prev_is_ai', 'next_is_ai', 'prev2_is_ai', 'next2_is_ai']:
            for sp in SPEC_N:
                o = no[(no.cut_tag == ct) & (no.spec == sp) & (no.term == tm)]['ame']
                if not len(o) or pd.isna(o.iloc[0]): continue
                o = float(o.iloc[0]); v = nn[(nn.cut_tag == ct) & (nn.spec == sp) & (nn.term == tm)]['ame'].values
                rows.append(dict(Exercise='Neighbor', cut_tag=ct, Cut=cl, Effect=TERM_N[tm],
                                 Spec=SPEC_N[sp], Observed=o, **stats(v, o)))
    return pd.DataFrame(rows)


def render(nn, no, outdir, dpi):
    os.makedirs(outdir, exist_ok=True)
    df = build_df(nn, no)

    def by_cut(exr, eff, spec):
        res = []
        for ct, cl in CUTS:
            r = df[(df.cut_tag == ct) & (df.Exercise == exr) & (df.Effect == eff) & (df.Spec == spec)]
            res.append((cl, (r.iloc[0].to_dict() if len(r) else None), NOCC_N[ct]))
        return res

    def forest_3col(exr, eff, unit, fname):
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 7.4), sharey=True)
        for j, (ax, sp) in enumerate(zip(axes, BYCUT_SPECS)):
            panel_bycut(ax, by_cut(exr, eff, sp), unit, SPEC_TITLE[sp], ylabels=(j == 0))
        fig.tight_layout(rect=[0, 0.06, 1, 0.99], w_pad=2.5)
        fig.legend(handles=leg, loc='lower center', ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.01))
        fig.savefig(f"{outdir}/{fname}", dpi=dpi, bbox_inches='tight'); plt.close(fig)

    forest_3col('Neighbor', '(t-1)', AME_UNIT, "placebo_summary_forest_neighbor_t1_byCut.png")
    forest_3col('Neighbor', '(t+1)', AME_UNIT, "placebo_summary_forest_neighbor_t2_byCut.png")
    return df


if __name__ == '__main__':
    SP = list(SPEC_N)
    pub_nn = pd.read_csv(f"{nd}/placebo_null_draws.csv")
    pub_no = pd.read_csv(f"{nd}/placebo_observed.csv")
    reb_nn = pd.read_csv(f"{M6}/sae4_placebo_null_E1_vs_E1E2.csv")
    reb_no = pd.read_csv(f"{M6}/sae4_placebo_observed_E1_vs_E1E2.csv")
    print(f"draws: published={pub_nn.draw.nunique()}  rebuild={reb_nn.draw.nunique()}")

    e1_nn = reb_nn[reb_nn['mask'] == 'E1']; e1_no = reb_no[reb_no['mask'] == 'E1']
    e2_nn = reb_nn[reb_nn['mask'] == 'E1E2']; e2_no = reb_no[reb_no['mask'] == 'E1E2']

    # fixture on the observed markers
    m = pub_no[pub_no.spec.isin(SP)].merge(e1_no[e1_no.spec.isin(SP)],
                                           on=['cut_tag', 'spec', 'term'], suffixes=('_p', '_r'))
    print(f"[fixture] observed markers, E1 rebuild vs published: n={len(m)} "
          f"max|dAME|={np.nanmax(np.abs(m.ame_p - m.ame_r)):.2e} max|dSE|={np.nanmax(np.abs(m.se_p - m.se_r)):.2e}")

    df_pub = render(pub_nn, pub_no, f"{STAGE}/plots_fixture/E1_from_published_draws_dpi190", dpi=190)
    df_e1 = render(e1_nn, e1_no, f"{STAGE}/plots_fixture/E1_rebuild_dpi190", dpi=190)
    df_e2 = render(e2_nn, e2_no, f"{STAGE}/plots_fixture/E1E2_dpi190", dpi=190)
    df_e2b = render(e2_nn, e2_no, f"{STAGE}/plots_new", dpi=300)

    for nm, d in [('published', df_pub), ('E1_rebuild', df_e1), ('E1E2', df_e2)]:
        d.to_csv(f"{STAGE}/diffs/_sae4_stats_{nm}.csv", index=False)
    print("done")
