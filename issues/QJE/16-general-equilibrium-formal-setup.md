# Issue 16 — General equilibrium treatment lacks formal setup

## Raised by
**R4.**

> "The generalization of the theory, which is at first introduced as a firm optimization problem, to general equilibrium, seems to lack a formal treatment of the model environment, which makes it harder to comprehend. Right now, its sole purpose is the formalization of the aggregation result. It would arguably be more interesting to expand the GE setting, formalizing it carefully, and then use it to think about economy-wide changes in job design as a direct effect of automation."

## Verification against source
- The GE setup is folded into Section 6 (`aggregation.tex`). It introduces a distribution of firms with heterogeneous effective AI quality but does not lay out preferences, market-clearing, or equilibrium definitions in full.
- The only payoff currently is the CES aggregation (see Issue 07).

## Options to address

1. **Move GE to appendix (link to Issue 07).** If the macro section goes to the appendix, this concern largely dissolves — the remaining body statement is informal by design.
2. **Formalize the GE carefully in the body.** Standard market structure (representative consumer, competitive firms, labor market with two labor types, capital), well-defined equilibrium. Use it to study economy-wide changes in job design as R4 suggests.
3. **Hybrid.** Keep a formally-defined GE environment in an appendix; use a short section in the body to show how job design responds to economy-wide AI improvements (comparative statics).

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Move GE to appendix | Low (part of Issue 07) | Low for this specific issue, but satisfies multiple referees in aggregate |
| 2. Formalize in body | High (2–3 weeks; genuine new theory) | **Very high** if the paper reframes as job-design (Issue 03 Option 2) — unlocks economy-wide predictions |
| 3. Hybrid | Medium (1 week) | High — best of both worlds |

**Recommendation:** If the paper reframes around job design (Issue 03 Option 2), do Option 2 — it becomes the centerpiece contribution. Otherwise, Option 1 is fine as the cheapest response. Option 3 is a worthy middle ground.
