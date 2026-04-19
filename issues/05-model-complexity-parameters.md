# Issue 05 — Model is parameter-heavy and difficult to build intuition from

## Raised by
**R1 (§2).** Echoed by R2 ("veers into computer science and computability") and R4 ("verbose theory section with overly long/numerous examples").

> "Each step is endowed with six distinct exogenous parameters, in addition to the global AI quality parameter. With just three steps, the reader must keep track of nineteen parameters."
>
> "The combination of extensive parameterization and layered optimization makes it difficult to develop a clear intuition about how the model behaves… it forces the authors to rely heavily on carefully selected examples."

## Verification against source
Per-step parameters currently instantiated (inferred from `main.tex:104–128` macros):
- `\manualTime{i}` — manual execution time
- `\AItime{i}` — AI execution time
- `\manualSkill{i}` — manual skill requirement
- `\AIskill{i}` — AI skill requirement
- `\handofftime{i}` — handoff time cost
- plus AI success probability per step (from `discussion.tex` / `optimization.tex`)

Six per-step parameters × 3 steps + 1 global = 19. R1's count is exact.

## Options to address

1. **Normalize aggressively.** Absorb degrees of freedom: set `\AItime = 1` uniformly (already done in Appendix FI per `appendix-FI.tex:9` comment `\AItime{i} = 1`); merge skill and time into a single "cost per step" in the baseline model and reintroduce the split only in an extension section.
2. **Two-parameter baseline + extensions.** Present the core intuition with one time parameter and one success probability per step. Each additional parameter gets its own subsection showing what it adds.
3. **Leaner running example.** Instead of Examples I–V with increasingly many numerical values, use one schematic example (like `intro.tex:61` Figure 1) and report comparative statics in figure form rather than exhaustive tables.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Aggressive normalization | Medium (3–5 days) | High — directly shrinks parameter count; may weaken some results |
| 2. Two-param baseline | Medium-high (1 week) | **High** — preserves all results, best pedagogy |
| 3. Leaner examples | Low (1–2 days) | Medium — addresses symptom not cause |

**Recommendation:** Option 2 is the cleanest response. It lets you keep the full model as a second-pass generalization while giving readers a tractable skeleton first.
