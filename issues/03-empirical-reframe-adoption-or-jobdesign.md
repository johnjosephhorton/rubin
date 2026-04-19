# Issue 03 — Reframe paper: theory of AI adoption OR theory of job design

## Raised by
**R4 (second bullet).** R1 hints at the same ("clarify whether its primary contribution lies in the AI-chaining mechanism, [or] the implications of AI for the specialization–coordination tradeoff").

R4 offers two explicit options:

> **Option 1: theory of AI adoption.** Unique angle: the model speaks to exposure-vs-execution discrepancy. Currently tested only by one regression (eq. 21) with mixed significance. *Expand* with data like same-task-across-occupations comparisons, or historical non-AI automation episodes.
>
> **Option 2: theory of job design.** More interesting but harder. Juxtapose the model's predictions for job design with vacancy data (Lightcast). Currently underexplored — the only near-testable hint is the p.24 "AI deployment may reduce worker specialization" paragraph.

## Verification against source
- The single fragmentation regression is `empirics.tex` §5 (the "p.41" content confirmed at `empirics.tex:288`).
- `discussion.tex:131` has the "technically possible" specialization remark — the closest job-design prediction (and R3 also calls this out as under-committed; see Issue 22).
- No Lightcast or vacancy data pipeline exists in the repo (spot-checked `etl/`, `data/`, `analysis/`); choosing Option 2 would require a new data source.

## Options to address

1. **Go with Option 1 (AI adoption).** Keep close to current empirical strategy; add complementary tests:
   - Same task across occupations with different neighbors (quasi-identification)
   - Historical analog: how did clustering predict prior automation waves?
   - Heterogeneity across platforms (not only Anthropic AEI)
2. **Go with Option 2 (job design).** Requires new data (Lightcast vacancies). Test: does AI exposure predict changes in skill/time composition within job postings? Does specialization drop in AI-exposed occupations?
3. **Hybrid** — keep both but tier them: one as the headline, the other as a cross-cut.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. AI-adoption reframe | Medium (2–3 weeks data work) | High — builds on existing pipeline; addresses R4 + editor |
| 2. Job-design reframe | High (6–10 weeks; new data acquisition + cleaning) | **Very high** if credible — this is the novel contribution the field doesn't have |
| 3. Hybrid | High+ | High but dilutes focus |

**Recommendation:** Option 1 is the low-risk path. Option 2 is the upside play — R4 explicitly says "I see big potential in this direction" and it would differentiate the paper in a way that lands on Acemoglu-level territory. Discuss with John whether the Lightcast pipeline is feasible before the meeting.
