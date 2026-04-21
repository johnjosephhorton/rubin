# Issue 08 — Introduction needs streamlining

## Raised by
**R1 (§5).** Related to Issue 02 (motivation).

> "The introduction spends considerable time developing the conceptual distinctions between steps, tasks, and jobs. As a result, the model and its key trade-offs are introduced gradually and in a somewhat fragmented way, with the main idea only becoming fully clear after several pages of exposition."
>
> "A more streamlined structure could help sharpen the paper's message. In particular, the introduction could first (i) explain what is missing in the existing literature, (ii) present the central mechanism and key trade-offs, (iii) summarize the main results and how they change conventional wisdom."

## Verification against source
- `intro.tex` = 617 lines. Currently opens with the concept of steps/tasks/jobs (line 1–100) rather than a puzzle or contribution statement.
- Figure 1 (illustrative, `intro.tex:61`) shows the division-of-labor example — this is intuitive but doesn't lead with a question.

## Options to address

1. **R1's three-beat rewrite.** (i) gap in literature; (ii) mechanism + trade-off; (iii) main results + conventional-wisdom shift. Replaces the current slow taxonomy buildup.
2. **Empirical-puzzle lede** (overlaps with Issue 02 Option 1). Open with the three data facts and frame the theory as required to rationalize them.
3. **Abstract-driven outline.** Tighten the abstract first (4–5 sentences), then mirror its structure in the intro's first 2 pages.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. R1's three-beat | Medium (2 days) | **High** — directly implements what R1 asked for |
| 2. Empirical-puzzle lede | Medium (2 days) | High — also addresses R4 and editor |
| 3. Abstract-driven | Low (1 day) | Medium — cosmetic but fast |

**Recommendation:** Combine Options 1 and 2 into a single intro rewrite: empirical puzzle → theory gap → mechanism → results. This is the single highest-leverage edit in the paper.
