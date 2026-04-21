# Issue EC 09 — The role of the fragmentation index is unclear

## Raised by
**Reviews #2016A and #2016C.**

Review #2016A: *"Clarifying the main takeaway from Section 5: Does this say one cannot benefit much from knowing the successes and reordering the steps?"*

Review #2016C:

> "My main reservation is that I did not fully understand the point of the fragmentation index and, by extension, the role of the empirical section. Proposition 5 shows that the fragmentation index approximates the short-run optimum up to constants, but I was left uncertain what exactly this contributes beyond a rough proxy for the intuition that clustering matters. It is not the main algorithmic tool, nor does it seem central to the main task-allocation results."
>
> "Similarly, although the empirical section documents patterns consistent with chaining, its connection to the paper's main organizational-design contribution feels somewhat indirect."

## Why this matters
Two referees independently flag that the fragmentation index feels disconnected from the paper's central contribution. #2016C is the most direct: "What does this contribute beyond a rough proxy?" If the index is the empirical section's linchpin but the theoretical section treats it as an approximation curiosity, the paper's internal structure looks weakly-integrated. This is a framing issue as much as a content issue.

## Verification against source
- Fragmentation index defined in `discussion.tex:62-64` with the approximation result `1/8 · OPT ≤ FI ≤ 5/4 · OPT`.
- Full proofs in `appendix-FI.tex`.
- Empirical tests (Tables/Figures in `empirics.tex`) use the fragmentation index as the key regressor.
- But in `model.tex` and `optimization.tex` the DP approach to the optimum doesn't reference the fragmentation index — it's a parallel analytical device.

## Options to address

1. **Elevate the fragmentation index as the paper's conceptual centerpiece.** Restructure so that the index is introduced as the measure of "cost of fragmentation" early, and the DP analysis becomes the proof that this intuition is approximately correct. Moves Section 5 earlier. The empirical section's use of the index then flows naturally from the theory.
2. **Add a "bridging" subsection.** Explicitly state why the fragmentation index is the right empirical object: the true optimum requires knowing `p_i` for all steps, which the econometrician doesn't observe; the index is observable (it only needs the binary exposure pattern) and is provably within constant factors of the optimum. This justifies its use for empirics even though it's not the theoretical optimum.
3. **De-emphasize the index.** If the index is really not central, remove it from the theory section and recast the empirical regressions in terms of direct clustering measures (chain length, contiguous-run count) rather than a composite index. Simpler but throws away a tractable empirical tool.
4. **Clarify in the introduction.** Add one paragraph explicitly previewing the index's role: "We develop a fragmentation index F that approximates the true cost of fragmentation within constant factors (§5), use it as an observable proxy in our empirical analysis (§7), and show it has theoretically clean comparative statics…"

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Restructure | Medium-high (1–2 weeks rewriting) | **High** — makes the index the through-line of the paper |
| 2. Bridging subsection | Low (1 day) | **High per cost** — cheapest way to answer both reviewers at once |
| 3. De-emphasize | Medium (rewrite empirical regressions) | Medium — throws out useful object |
| 4. Intro clarification | Trivial (½ day) | Medium — previews but doesn't fix the structural issue |

**Recommendation:** Option 2 as minimum fix (cheap, targeted). Option 1 if the paper is being heavily restructured. Note: both reviewers are otherwise favorable — Option 2 alone may be sufficient.
