# Issue 12 — Single-chain-per-occupation assumption

## Raised by
**R4 (footnote 1).**

> "In how the model is taken to the data, I do have mild reservations about the idea that *every* step within an occupation must belong to a single chain, rather than having some separate, independent step chains within an occupation (teaching and research may both be steps an economics professor must complete within their job, but which comes first?)."

## Verification against source
- Verified in `model.tex` and `empirics.tex`: the model defines chains as "contiguous blocks" of AI-executed steps, which does **not** inherently force a single chain per occupation.
- However, the **empirical operationalization** prompts GPT to return *one* linear ordering per occupation — so the data-level object is effectively a single sequence. Multiple chains are then defined as contiguous runs *within* that single sequence, not as parallel independent workflows.
- R4's concern is specifically about the linear-sequence assumption in the empirics, not about the theory.

## Options to address

1. **Allow multiple orderings per occupation.** Prompt GPT to return a DAG or a set of independent sequences. The AI-chain definition adapts naturally: chains are contiguous runs within any sequence.
2. **Keep linear ordering but test sensitivity.** Ask GPT to identify "natural break points" in the sequence and treat segments as independent. Re-run the fragmentation and spillover regressions within-segment and show they are robust.
3. **Qualitative caveat only.** Acknowledge the assumption in a paragraph; argue most occupations have enough linearity (especially at daily-workflow level) that the bias is small; cite literature on occupational routineness.
4. **Theoretical extension.** Generalize the model to allow parallel chains (workflow as a partial order) and show main results survive.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. DAG prompting | High (3–4 weeks; heavy reprocessing) | High — answers R4 structurally |
| 2. Segment-robustness | Medium (1 week) | **High** — cheap, preserves existing infrastructure |
| 3. Qualitative caveat | Low (½ day) | Low-medium — safest but least convincing |
| 4. Theory extension | Medium (2 weeks theory work) | Medium — addresses the conceptual point but not the empirical one |

**Recommendation:** Options 2 + 3 together. Option 2 handles the empirical concern at low cost; Option 3 transparently scopes the claim. Consider Option 1 if a future revision has more bandwidth.
