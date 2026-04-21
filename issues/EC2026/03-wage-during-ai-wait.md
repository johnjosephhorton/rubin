# Issue EC 03 — Does the firm pay wages while the worker waits for AI?

## Raised by
**Review #2016A.**

> "Does the firm pay the human worker while she is waiting for the AI tasks to be completed? If yes, the paper benefits from justifying this."

## Why this matters
If AI takes time to execute a step (or chain) and the human worker is idle during that time but still paid, the effective "labor cost of an AI chain" is not just the end-of-chain verification — it's verification plus the wait-time wage. This changes the chain-length tradeoff: longer chains mean longer waits, which if paid, offsets the per-atom verification savings.

The ambiguity comes from how the wage equation compresses "time" — it's unclear whether the `Σ time_i` factor includes AI-execution time during which the human is idle.

## Verification against source
- `aggregation.tex:32-34`: `wage_bill_j = (... skill-sum ...) × (Σ time_i)`. The time sum is over tasks in the job, but it's not disambiguated whether AI-only time counts.
- The chain-length tradeoff in `model.tex` assumes verification is a fixed cost per chain — this implicitly assumes the worker isn't "on the clock" for the AI's execution time.
- Appendix FI normalizes `\AItime{i} = 1` but does not say whose clock this is denominated in.

## Options to address

1. **Assumption + one-paragraph justification.** State explicitly: "The human worker is compensated only for execution and verification time, not for idle time during AI chain execution." Justify by appeal to modern work arrangements (parallel task assignment, shift work, contract-based compensation). Cheapest fix.
2. **Two compensation regimes as cases.**
   - Case A (hourly wages): worker paid for all hours on-shift, including AI wait. Wage bill includes AI execution time.
   - Case B (piece-rate / output-based): worker paid only for tasks completed. Wage bill excludes AI wait.
   Show that main results are qualitatively similar but chain-length thresholds differ.
3. **Concurrent work model.** Allow each worker to juggle multiple chains in parallel: while Chain 1's AI is running, she verifies Chain 2's output. This eliminates the idle-wait problem entirely and introduces a concurrency limit as a new parameter.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Justification paragraph | Trivial (½ day) | Medium — directly answers the reviewer's literal question |
| 2. Two regimes | Medium (2–3 days; short additional propositions) | High — shows robustness to compensation structure |
| 3. Concurrent work | High (1+ week; new parameter, new optimization problem) | High but adds complexity counter to R1's parameter-load critique |

**Recommendation:** Option 1 at minimum. Option 2 if you want to demonstrate robustness without adding modelling machinery. Skip Option 3 unless the paper is reframing around gig-economy / contract labor.

*Response to this comment is tracked in [`referee_responses.md`](referee_responses.md).*

---

## Proposed structural solution — joint with [Issue 10](10-verification-cost-chain-length.md)

We now prefer a structural modification that jointly addresses this comment and R#2016D's verification-cost-scaling critique. The per-step AI-management cost becomes chain-length-dependent:

```
t^{AI, new}_i  =  t^{AI, old}_i  +  (n − 1) · t^{Chain}
```

where `(n − 1) · t^{Chain}` captures the additional human time associated with a longer chain. Under the wage-during-AI-wait reading of this comment, that term is the human's engaged / oversight time while the AI runs the chain's intermediate steps. Under R#2016D's verification-difficulty reading, the same term is the extra per-step verification effort for longer AI outputs. The firm compensates the human for this time via the wage equation, and the dependence on chain length is now explicit. Full specification and expected implications are in [Issue 10](10-verification-cost-chain-length.md#proposed-structural-solution--concrete-formulation).

The earlier "AI execution time is second-order" framing is a special case of the structural solution at `t^{Chain} = 0`.
