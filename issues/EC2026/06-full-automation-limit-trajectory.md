# Issue EC 06 — AI trajectory and the full-automation limit

## Raised by
**Review #2016B (Major Comment 3).**

> "The model treats AI quality ($\alpha$, governing success probability $q_i = \alpha^{d_i}$) as a fixed parameter… Long-run implication: as $\alpha \to 1$, the model predicts near-complete automation of all but the final verification step in each job."
>
> "Today's 'augmented' tasks (human verification of AI output) increasingly become tomorrow's 'automated' tasks as AI reliability improves and human verification costs exceed error costs. The more profound question, IMHO, which the model hints at but does not fully explore, concerns what work remains for humans in the limit."
>
> "The current framework assumes humans provide two irreplaceable functions: (1) verification/evaluation of AI outputs (augmented steps), and (2) hand-offs between jobs (coordination). But on (1): As AI error rates fall and outputs become more reliable, the marginal value of human verification declines… On (2): The model's hand-off costs ($t_i^H$) implicitly assume human-to-human coordination, but AI increasingly mediates these interfaces (workflow management systems, automated handoffs between AI agents)."
>
> "I think the long-run equilibrium of full automation is worthy of discussion."

## Why this matters
The paper's mechanisms assume humans are indispensable — at minimum as verifiers. But the reviewer correctly notes that this is endogenous: once AI is reliable enough, firms stop verifying. Once AI handles handoffs, coordination friction disappears. In the limit, the model predicts a degenerate equilibrium in which humans do nothing.

This matters for two reasons:
1. **Prediction validity.** If AI quality is on a trajectory (and the paper implicitly assumes so via its J-curve narrative), the paper should be able to describe where that trajectory ends — not just the near-term mechanics.
2. **Theoretical completeness.** A paper that offers a "theory of AI automation" should take a stand on the limiting case. Silence here leaves a visible gap.

## Verification against source
- `\alpha` treated as a fixed parameter throughout.
- `discussion.tex` and `model.tex` describe threshold effects as `\alpha` rises past critical values, but the paper does not characterize the limit `\alpha \to 1`.
- No formal treatment of "AI-mediated handoffs" or "AI verifying AI."

## Options to address

1. **Limiting-case proposition.** Add a proposition: "As `\alpha \to 1`, the cost-minimizing organizational structure converges to a single verifier per job + fully-automated chain for everything else." Derive formally. Discuss welfare implications.
2. **Endogenous verification intensity.** Add a parameter for "verification effort" `v_i ∈ [0, 1]` with `v_i = 1` meaning full human verification and `v_i = 0` meaning no verification. Firms trade off verification cost against error cost. As `\alpha → 1`, `v_i → 0` endogenously. Substantially more theoretical machinery.
3. **AI-to-AI handoffs.** Add a second handoff cost parameter `t_H^{AI-to-AI}` which is lower than human-to-human. Show comparative statics as this falls. Natural way to model workflow-management systems.
4. **Discussion paragraph only.** Add a paragraph in Section 6 or Conclusion speculating about the limit without deriving it formally. Cheapest response; likely unsatisfying for this reviewer.
5. **Two-era framing.** Position the paper as "the theory of AI adoption in the *transition* era, before full automation becomes feasible for most tasks." Scope-limit rather than extend. Requires adjusting the title/framing.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Limiting-case prop | Low-medium (2–3 days) | High — directly answers reviewer, small marginal modelling work |
| 2. Endogenous verification | High (3–4 weeks; restructures much of Section 4) | Very high — addresses this + more, but expensive |
| 3. AI-to-AI handoffs | Medium (1 week) | Medium — partial answer, one mechanism of automation |
| 4. Discussion only | Trivial (½ day) | Low |
| 5. Scope-limit framing | Low (1 day) | Medium — honest, but reviewer wants engagement with the limit, not exclusion of it |

**Recommendation:** Option 1 as the minimum — one limiting-case proposition costs little and satisfies the reviewer's explicit request. Option 2 if the paper is being heavily restructured anyway (e.g., per QJE Issue 01's two-mechanism separation).
