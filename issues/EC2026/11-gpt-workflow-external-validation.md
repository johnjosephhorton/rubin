# Issue EC 11 — GPT-generated workflows lack external validation

## Raised by
**Review #2016D** (and secondarily #2016B).

Review #2016D: *"The GPT-generated workflows in the empirical application are interesting, though the validation is focused on internal consistency and not external validity. It would have been helpful to have a benchmark."*

Review #2016B (major comment 1, excerpt): *"The authors show task orderings are robust to alternative prompts (Appendix I, Kendall's τ ≈ 0.6), but this measures consistency across prompts, not accuracy relative to actual production processes."*

> "As a piece of recommendation, the authors might provide additional validation of GPT-generated sequences against direct observation where possible (e.g., detailed work studies, time-motion data, or interviews with practitioners in selected occupations)."

## Why this matters
The paper's empirical strategy rests on the GPT-ordered task sequences. Current robustness checks (alternative prompts) demonstrate the ordering is reproducible given similar inputs to the LLM. But none of them establish that the ordering reflects *reality* — i.e., that the resulting fragmentation index corresponds to actual workflow dispersion in the occupations studied. This is the same concern as QJE R2's fragility point but framed as a call for specific external validation.

## Verification against source
- `writeup/appendix-GPTprompts_robustness.tex` contains the alternative-prompt robustness check (Kendall's τ ≈ 0.6 across prompt variants).
- Producer notebook: `analysis/onet_task_and_occ_sequence_robustness.ipynb`.
- No external benchmark data in the repo (`data/` doesn't contain time-motion studies, interview transcripts, or production-process maps).

## Options to address

1. **Pick a handful of occupations, validate against BLS Time-Use or OES data.** Some occupations have detailed time-use breakdowns (ATUS). Compare the GPT-generated ordering against the time-use activity order. Even partial validation for a few occupations would be a substantial upgrade.
2. **Interviews with practitioners.** Survey practitioners in 3–5 well-defined occupations (programmers, public relations specialists, radiologists, etc.) with a simple "given these tasks, order them typically" prompt. Compare with GPT output. Low sample size but high informational content.
3. **Industrial-engineering literature check.** Production-process maps exist in the operations management literature for many standardized occupations. Cite a few, compare orderings for those.
4. **Sensitivity bounds.** Without external validation, explicitly bound the empirical estimates' sensitivity to ordering errors. Report: "under the null that GPT orderings are random, our fragmentation coefficient would be 0. Our observed coefficient is X, inconsistent with random ordering at the p < 0.01 level." This is a weaker statement but defensible.
5. **Publish the orderings openly.** (Already raised in QJE Issue 13.) Combined with external validation, gives any subsequent researcher the tools to critique or extend.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. ATUS comparison | Medium (1–2 weeks; new data) | **High** — gives a concrete external benchmark |
| 2. Practitioner interviews | Medium-high (2–3 weeks; IRB + recruitment) | High but slow |
| 3. Lit-based validation | Low-medium (1 week) | Medium — a few anecdotes, not systematic |
| 4. Sensitivity bounds | Low (½ week) | Medium — defensive framing |
| 5. Publish orderings | Trivial (already recommended) | Medium |

**Recommendation:** Option 1 (ATUS, if feasible) + Option 4 (bounds as a safety net). Option 5 should happen regardless. The combination answers both R#2016B and R#2016D without requiring extensive new infrastructure.
