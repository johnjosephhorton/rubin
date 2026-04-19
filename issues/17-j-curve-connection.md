# Issue 17 — J-curve connection needs clarification

## Raised by
**R1 (§4).**

> "The paper argues that the model provides a microfoundation for the 'productivity J-curve' phenomenon. However, the connection between the model and this empirical concept could be clarified more precisely."
>
> "In the empirical literature, a J-curve refers to the time path of observed productivity following adoption… By contrast, the model appears to generate discontinuities in organizational choices and in the marginal value of the technology when AI quality crosses certain thresholds."

## Verification against source
- The J-curve claim likely appears in `intro.tex` and/or `discussion.tex`. Model threshold discontinuities are a property of the long-run job-design optimization.
- The connection from "discrete organizational reorganization on AI-quality improvement" to "slow-then-fast time path of measured productivity" is not formally made — R1 is correct.

## Options to address

1. **Formalize the mapping.** Assume AI quality improves on a smooth trajectory q(t). Under the model, optimal organizational structure changes discretely at q-thresholds. Plot measured productivity (output per unit labor) as q(t) crosses thresholds → shows flat periods followed by jumps. Whether this looks "J-shaped" depends on timing of thresholds and the convexity of q(t).
2. **Scope-of-claim hedge.** Replace "microfoundation for productivity J-curves" with "a model that produces threshold effects in measured productivity, complementary to the intangibles-complementarity explanation of Brynjolfsson, Rock & Syverson (2021)".
3. **Remove the claim.** Drop the J-curve framing entirely; keep threshold discontinuity as a standalone result.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Formalize | Medium (1 week; numerical simulation + one proposition) | **High** — directly answers R1; adds empirical content |
| 2. Scope hedge | Low (½ day) | Medium — avoids reviewer ire without adding substance |
| 3. Remove | Low (½ day) | Low — loses a contribution that might be valuable |

**Recommendation:** Option 1. If the model actually produces J-shaped productivity paths under reasonable q(t), formalizing it is a genuine contribution worth preserving. Fall back to Option 2 if the numerical exercise doesn't deliver.
