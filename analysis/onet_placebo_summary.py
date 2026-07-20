"""
Placebo summary: combined forest plots + table for the neighbor and fragmentation-index
position-reshuffle placebos. Reads the saved placebo null/observed CSVs and renders, per cut,
a 2-row forest plot (t-1 | t+1 on top, EFI Definition 2 (preferred exposure measure) centered below). "Consistent with null" = observed
within the placebo 10-90% band. Panel headers show sample size (occupations, and observations for
the task-level neighbor regression). Both predictions share the >=5-step frequency-pruned-workflow universe;
fragmentation uses all such occupations while the neighbor regression uses their DWA-eligible neighbored subset.

Run:  python analysis/onet_placebo_summary.py   (from the repo root)
"""
import pandas as pd, numpy as np, re, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

nd = "data/computed_objects/execTypeVaryingDWA_anthropicIndex_weeklyTasks"
fd = "data/computed_objects/fragmentationIndex_weeklyTasks"
outd = "writeup/plots/placebo_summary"
mf = "data/computed_objects/ONET_Eloundou_Anthropic_GPT/ONET_Eloundou_Anthropic_GPT.csv"
import os; os.makedirs(outd, exist_ok=True)

nn = pd.read_csv(f"{nd}/placebo_null_draws.csv"); no = pd.read_csv(f"{nd}/placebo_observed.csv")
fn = pd.read_csv(f"{fd}/frag_placebo_null_draws.csv"); fo = pd.read_csv(f"{fd}/frag_placebo_observed.csv")
N = int(nn['draw'].nunique())

CUTS = [('all', 'All tasks')]
for fam, lab in [('daily', 'Daily+'), ('sevdaily', 'SeveralDaily+'), ('hourly', 'Hourly+')]:
    for t in [20, 35, 50, 65]: CUTS.append((f"{fam}{t}", f"{lab} >={t}%"))
SPEC_N = {'no_fe_no_dwa': 'No FE', 'major_fe_no_dwa': 'Major FE', 'minor_fe_no_dwa': 'Minor FE'}
TERM_N = {'prev2_is_ai': '(t-2)', 'prev_is_ai': '(t-1)', 'next_is_ai': '(t+1)', 'next2_is_ai': '(t+2)'}
SPEC_F = {'none': 'No FE', 'Major': 'Major FE', 'Minor': 'Minor FE'}

# ---- sample sizes per cut (neighbor occupations/observations; fragmentation uses the SAME occupations) ----
mg = pd.read_csv(mf); mg['Task Position'] = pd.to_numeric(mg['Task Position'], errors='coerce')
FTm = {'daily': ['FT_Daily', 'FT_Several times daily', 'FT_Hourly or more'],
       'sevdaily': ['FT_Several times daily', 'FT_Hourly or more'], 'hourly': ['FT_Hourly or more']}
def parse(tag):
    if tag == 'all': return None, None
    m = re.match(r'([a-z]+)(\d+)', tag); return FTm[m.group(1)], int(m.group(2))
dwa = pd.read_csv("data/computed_objects/similar_dwa_tasks/dwa_task_mapping.csv")
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
# Shared >=5 universe: both predictions run on occupations whose frequency-pruned workflow keeps >=5 steps.
# Fragmentation regresses on all such occupations; the neighbor regression uses the DWA-eligible neighbored
# subset of them (so NOCC_N <= NOCC_F). Read the fragmentation N from the heatmap sweep so the forest panel
# header matches the heatmap cell exactly.
_fsw = pd.read_csv(f"{fd}/frag_logic_threshold_sweep.csv"); _fsw = _fsw[(_fsw.FI_def == 'v1') & (_fsw.FE == 'none')]  # v1 = paper Definition 1 (exposure E1|E2) after the July 2026 renumbering; rerun the sweep notebook before this script
NOCC_F = {('all' if r['family'] == 'all' else f"{r['family']}{int(r['threshold'])}"): int(r['N_occ'])
          for _, r in _fsw.iterrows()}

def stats(v, o):
    v = np.asarray([x for x in v if not np.isnan(x)], float)
    if not len(v) or np.isnan(o): return dict(mean=np.nan, sd=np.nan, p10=np.nan, p90=np.nan, pct=np.nan, p=np.nan)
    return dict(mean=v.mean(), sd=v.std(ddof=1), p10=np.percentile(v, 10), p90=np.percentile(v, 90),
                pct=100 * np.mean(v < o), p=(1 + np.sum(np.abs(v) >= abs(o))) / (1 + len(v)))
def ordi(n):
    if np.isnan(n): return ""
    n = int(round(n)); s = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10 if n % 100 not in (11, 12, 13) else 0, 'th'); return f"{n}{s}"

rows = []
for ct, cl in CUTS:
    for tm in ['prev_is_ai', 'next_is_ai', 'prev2_is_ai', 'next2_is_ai']:
        for sp in SPEC_N:
            o = no[(no.cut_tag == ct) & (no.spec == sp) & (no.term == tm)]['ame']
            if not len(o) or pd.isna(o.iloc[0]): continue
            o = float(o.iloc[0]); v = nn[(nn.cut_tag == ct) & (nn.spec == sp) & (nn.term == tm)]['ame'].values
            rows.append(dict(Exercise='Neighbor', cut_tag=ct, Cut=cl, Effect=TERM_N[tm], Spec=SPEC_N[sp], Observed=o, **stats(v, o)))
    for d_ in ['v2', 'v1']:
        for fe in SPEC_F:
            o = fo[(fo.cut_tag == ct) & (fo.FI_def == d_) & (fo.FE == fe)]['coef']
            if not len(o) or pd.isna(o.iloc[0]): continue
            o = float(o.iloc[0]); v = fn[(fn.cut_tag == ct) & (fn.FI_def == d_) & (fn.FE == fe)]['coef'].values
            rows.append(dict(Exercise='Fragmentation', cut_tag=ct, Cut=cl, Effect=f'FI {d_}', Spec=SPEC_F[fe], Observed=o, **stats(v, o)))
df = pd.DataFrame(rows)
out = df.copy(); out['Placebo_10_90'] = out.apply(lambda r: f"[{r.p10:.3f}, {r.p90:.3f}]" if not np.isnan(r.p10) else "", axis=1)
out['N_occupations'] = out.apply(lambda r: NOCC_N[r.cut_tag] if r.Exercise == 'Neighbor' else NOCC_F[r.cut_tag], axis=1)
out['N_observations'] = out.apply(lambda r: NOBS_N[r.cut_tag] if r.Exercise == 'Neighbor' else NOCC_F[r.cut_tag], axis=1)
out = out.rename(columns={'mean': 'Placebo_mean', 'sd': 'Placebo_SD', 'pct': 'Obs_pctile', 'p': 'rand_p'})
out[['Exercise', 'Cut', 'Effect', 'Spec', 'N_occupations', 'N_observations', 'Observed', 'Placebo_mean', 'Placebo_SD', 'Placebo_10_90', 'Obs_pctile', 'rand_p']].to_csv(f"{outd}/placebo_summary.csv", index=False)

def panel(ax, sub, unit, title, nocc, nobs=None):
    cnt = f"{nocc:,} occ." if nobs is None else f"{nocc:,} occ. · {nobs:,} obs."
    title = f"{title}\n{cnt}"
    if not len(sub):
        ax.text(0.5, 0.5, 'Not estimable\nfor this cut', ha='center', va='center', transform=ax.transAxes, fontsize=11, color='dimgray')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10.5, fontweight='bold'); return
    sub = sub.iloc[::-1].reset_index(drop=True)
    for i, r in sub.iterrows():
        ax.plot([r['p10'], r['p90']], [i, i], color='0.55', lw=5, alpha=0.5, solid_capstyle='round', zorder=1)
        ax.plot(r['mean'], i, marker='|', ms=14, mew=2, color='0.35', zorder=2)
        sig = (not np.isnan(r['p10'])) and (r['Observed'] < r['p10'] or r['Observed'] > r['p90'])
        ax.plot(r['Observed'], i, 'o', ms=9, color=('crimson' if sig else 'steelblue'), markeredgecolor='k', mew=0.6, zorder=3)
        ax.text(1.012, i, ordi(r['pct']), transform=ax.get_yaxis_transform(), clip_on=False, va='center', ha='left', fontsize=8, color='dimgray')
    ax.axvline(0, color='k', lw=1, alpha=0.5)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels([r['Spec'] for _, r in sub.iterrows()], fontsize=9)
    ax.set_xlabel(unit, fontsize=9.5); ax.grid(axis='x', ls=':', alpha=0.5); ax.set_ylim(-0.6, len(sub) - 0.4); ax.margins(x=0.16)
    ax.set_title(title, fontsize=10.5, fontweight='bold')

ospec = {'No FE': 0, 'Major FE': 1, 'Minor FE': 2}
AME_UNIT = 'Average marginal effect on P(focal task is AI)'; FI_UNIT = 'Standardized fragmentation-index coefficient'
leg = [Line2D([0], [0], color='0.55', lw=5, alpha=0.5, label='Placebo null 10–90%'),
       Line2D([0], [0], marker='|', ls='', ms=12, mew=2, color='0.35', label='Placebo mean'),
       Line2D([0], [0], marker='o', ls='', ms=9, color='crimson', mec='k', label='Observed (outside 10–90% null)'),
       Line2D([0], [0], marker='o', ls='', ms=9, color='steelblue', mec='k', label='Observed (within 10–90% null)')]
for ct, cl in CUTS:
    d = df[df.cut_tag == ct]
    def get(exr, eff):
        s = d[(d.Exercise == exr) & (d.Effect == eff)].copy(); s['k'] = s.Spec.map(ospec); return s.sort_values('k')
    t1 = get('Neighbor', '(t-1)'); t2 = get('Neighbor', '(t+1)'); fi = get('Fragmentation', 'FI v2')
    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(2, 4, hspace=0.7, wspace=0.85, top=0.88, bottom=0.12, left=0.09, right=0.95)
    ax1 = fig.add_subplot(gs[0, 0:2]); ax2 = fig.add_subplot(gs[0, 2:4]); ax3 = fig.add_subplot(gs[1, 1:3])
    panel(ax1, t1, AME_UNIT, 'Previous task (t-1) is AI', NOCC_N[ct], NOBS_N[ct])
    panel(ax2, t2, AME_UNIT, 'Next task (t+1) is AI', NOCC_N[ct], NOBS_N[ct])
    panel(ax3, fi, FI_UNIT, 'Fragmentation index (EFI Definition 2)', NOCC_F[ct])
    fig.legend(handles=leg, loc='lower center', ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"Placebo summary — observed effect vs its null distribution  —  {cl}\n"
                 f"(N={N} reshuffles; panel header = sample size (occ. / obs.); right margin = observed percentile in null)",
                 fontsize=12, fontweight='bold')
    fig.savefig(f"{outd}/placebo_summary_forest_{ct}.png", dpi=125, bbox_inches='tight'); plt.close(fig)
# ===================== PER-EFFECT FORESTS (rows = cuts; columns = specifications) =====================
# One graph per effect, showing the same effect across every frequency cut (rows) under each of the three
# regression specifications (columns): no FE, SOC major-group FE, SOC minor-group FE.
BYCUT_SPECS = ['No FE', 'Major FE', 'Minor FE']
SPEC_TITLE  = {'No FE': 'No fixed effects', 'Major FE': 'SOC major-group FE', 'Minor FE': 'SOC minor-group FE'}

def by_cut(exr, eff, spec):
    res = []
    for ct, cl in CUTS:
        r = df[(df.cut_tag == ct) & (df.Exercise == exr) & (df.Effect == eff) & (df.Spec == spec)]
        nocc = NOCC_N[ct] if exr == 'Neighbor' else NOCC_F[ct]
        res.append((cl, (r.iloc[0].to_dict() if len(r) else None), nocc))
    return res

def panel_bycut(ax, data, unit, title, ylabels):
    n = len(data); ys = list(range(n))[::-1]   # first cut (All tasks) at top
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
    ax.axhline(n - 1.5, color='black', lw=1)   # separate the All-tasks baseline row
    ax.set_yticks(ys)
    if ylabels:
        ax.set_yticklabels([f"{clab}  (N={nocc})" for clab, _, nocc in data], fontsize=8.5)
    else:
        ax.tick_params(labelleft=False)   # hide without clearing the shared-axis labels
    ax.set_xlabel(unit, fontsize=9); ax.grid(axis='x', ls=':', alpha=0.5); ax.margins(x=0.22)
    ax.set_title(title, fontsize=10.5, fontweight='bold')

def forest_3col(exr, eff, unit, suptitle, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 7.4), sharey=True)
    for j, (ax, sp) in enumerate(zip(axes, BYCUT_SPECS)):
        panel_bycut(ax, by_cut(exr, eff, sp), unit, SPEC_TITLE[sp], ylabels=(j == 0))
    fig.tight_layout(rect=[0, 0.06, 1, 0.99], w_pad=2.5)
    fig.legend(handles=leg, loc='lower center', ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.01))
    # figure titles live in the writeup, not the PNG (suptitle argument kept for call-site compatibility)
    fig.savefig(f"{outd}/{fname}", dpi=190, bbox_inches='tight'); plt.close(fig)

forest_3col('Fragmentation', 'FI v2', FI_UNIT,
            "Fragmentation index (EFI Definition 2): observed coefficient vs position-reshuffle null\n"
            f"across frequency cuts (rows) and specifications (columns)   (N={N} reshuffles; right margin = observed percentile in null)",
            "placebo_summary_forest_fragmentation_byCut.png")
forest_3col('Neighbor', '(t-1)', AME_UNIT,
            "Previous task (t-1) is AI: observed average marginal effect vs position-reshuffle null\n"
            f"across frequency cuts (rows) and specifications (columns)   (N={N} reshuffles; right margin = observed percentile in null)",
            "placebo_summary_forest_neighbor_t1_byCut.png")
forest_3col('Neighbor', '(t+1)', AME_UNIT,
            "Next task (t+1) is AI: observed average marginal effect vs position-reshuffle null\n"
            f"across frequency cuts (rows) and specifications (columns)   (N={N} reshuffles; right margin = observed percentile in null)",
            "placebo_summary_forest_neighbor_t2_byCut.png")

print(f"Saved placebo_summary.csv, {len(CUTS)} per-cut forests, and 3 per-effect (by-cut x spec) forests to {outd}")
