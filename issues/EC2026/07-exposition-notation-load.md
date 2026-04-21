# Issue EC 07 — Notation load and exposition density

## Raised by
**Review #2016B (Minor Comment 4).**

> "The paper is technically rigorous but dense, particularly Sections 3–5. The model introduces considerable notation (steps $s_i$, tasks $T_b$, jobs $J_j$, execution modes, skill costs $c_i^M$, time costs $t_i^M$, hand-off costs $t_i^H$, AI probabilities $q_i$, etc.) that accumulates quickly. Feels a lot of notations floating around and have been having a hard time following the exposition."
>
> Suggestions:
> - "Add a notational summary table early in Section 3."
> - "Include a worked numerical example in the main text (currently relegated to Appendix B) to build intuition before the general analysis."
> - "Consider moving Proposition 1 (short-run dynamic programming) to an appendix and leading with the fragmentation index intuition (currently Section 5), which is more central to the empirical application."

## Why this matters
The notation load is symptomatic of the parameter count itself (a 6-per-step × n-step model needs a lot of symbols). Solving the underlying parameter-reduction problem (see QJE Issue 05 — model complexity) would also solve most of this. But even without cutting the model, cosmetic fixes are cheap and reduce the reviewer's cognitive load.

## Verification against source
- `model.tex` introduces the notation sequentially across the section, with no consolidated table.
- `main.tex:104–128` defines the macros (`\manualTime`, `\skillCost`, etc.) but these are TeX macros, not a reader-facing glossary.
- Appendix B (numerical example) is in `appendix-FI.tex` or similar — not up front.
- Proposition 1 on short-run DP: in `optimization.tex` (Section 4 of the body).

## Options to address

1. **Add notation summary table.** One table early in Section 3 listing every primitive symbol with one-line definition. Trivial; purely cosmetic.
2. **Promote a worked numerical example to the main body.** Extract the cleanest example from Appendix B and insert as Section 3.5 or 4.1. Builds intuition before the general propositions.
3. **Move Proposition 1 (short-run DP) to appendix.** Leave the statement in the body; move proof and DP construction to an appendix. Consistent with what QJE R3 and QJE Issue 06 also suggest.
4. **Lead with the fragmentation-index intuition.** Restructure so that Section 5 (fragmentation) comes before Section 4 (DP optimization). Frames the fragmentation concept as the central tool, DP as the computational appendix. Bigger structural change.
5. **Reduce notation at the source.** Solve QJE Issue 05 (model complexity) — the notation load falls along with the parameter count. Most ambitious; addresses the root cause.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Notation table | Trivial (½ day) | Medium — direct answer to reviewer; helps other readers too |
| 2. Worked example up front | Low (1 day; reorganization) | High — pedagogically strong, low risk |
| 3. Move Prop 1 proof | Low (½ day) | High — also asked by QJE R3 |
| 4. Restructure section order | Medium (2–3 days) | High — but also risky; may break the narrative flow |
| 5. Reduce notation itself | High (whole-paper revision, see QJE Issue 05) | Very high — root cause fix |

**Recommendation:** Implement Options 1, 2, 3 in the same editing pass (they take together ~2 days). Option 5 belongs to the model-complexity revision discussed elsewhere; when that lands, Option 4 follows naturally.
