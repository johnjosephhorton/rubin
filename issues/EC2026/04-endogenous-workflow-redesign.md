# Issue EC 04 — Exogenous task sequences preclude workflow redesign

## Raised by
**Review #2016B (Major Comment 1).**

> "The model assumes production follows a fixed, exogenously specified sequence of steps. While analytically tractable, which severely limits the framework's ability to capture how firms actively redesign workflows in response to AI capabilities, arguably the most consequential margin of adjustment."
>
> "Real-world AI deployment frequently involves workflow reengineering: reordering tasks to enable earlier validation, parallelizing previously sequential activities, or restructuring production to exploit AI's comparative advantage in specific cognitive operations."
>
> "The empirical reliance on GPT-generated task orderings (Appendix H) partially addresses this by producing 'typical' workflows, but the validation is limited… measurement error in task ordering could substantially bias the fragmentation index and weaken tests of local complementarities."
>
> Recommendation: "The long-run model (Section 6.1) should be extended to allow firms to choose task orderings jointly with AI deployment and job design. Even a simplified version, say, allowing permutations of a subset of tasks, might be helpful."

## Why this matters
This is perhaps the most fundamental critique of the paper's long-run relevance. The current model holds the step sequence fixed and lets the firm optimize over only AI-deployment patterns and job boundaries. But real AI adoption often involves **reorganizing what gets done when** — splitting historically sequential tasks into parallel streams, pulling verification earlier, or changing the order entirely to exploit where AI is strong. None of this is capturable under an exogenous sequence.

## Verification against source
- `model.tex` treats the step sequence as a given ordered tuple `(s_1, ..., s_n)`.
- `optimization.tex` optimizes over partitions into chains and jobs but preserves the order.
- The long-run problem (Section 6.1, `model.tex:260–…`) jointly optimizes job design and AI strategy — **but not sequence**.

## Options to address

1. **Allow local permutations.** Let the firm pick from a set of "admissible orderings" (e.g., within-block swaps allowed, across-block swaps not) and optimize jointly with AI deployment. Adds one layer to the optimization but tractable via DP.
2. **Full permutation on a subset.** Designate some subset of steps as "order-flexible" (e.g., preparation-phase steps that don't have strict dependencies) and allow any permutation on them. Matches real-world partial reorganization.
3. **DAG instead of a linear sequence.** Replace the sequence with a partial-order DAG specifying which steps must precede which. The firm optimizes over topological orderings consistent with the DAG. More general but much harder to calibrate empirically.
4. **Defend the assumption as a scope choice.** Acknowledge that the paper studies deployment within a fixed workflow, not workflow redesign. Frame workflow redesign as a separate research question. Weaker response; likely unsatisfying for this reviewer.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Local permutations | Medium (1–2 weeks theory + numerical example) | **High** — directly addresses the reviewer with minimal conceptual overhaul |
| 2. Subset permutations | Medium-high (2–3 weeks) | High — richer theoretical object, more to say about endogenous workflow |
| 3. Full DAG | High (new theory paper's worth of work) | Very high but out of scope for this revision |
| 4. Scope defense | Low (½ day) | Low — reviewer explicitly calls this "the most consequential margin of adjustment" |

**Recommendation:** Option 1 for this revision cycle; Option 3 as a natural sequel paper.
