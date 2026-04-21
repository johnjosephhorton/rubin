# Issue 07 — Macroeconomic extension (Section 6) to appendix

## Raised by
**R1 (§3), R2 ("weak link" section), R3 (other comments).** Three of the four referees explicitly flag this as a candidate for relocation.

> R1: "The result is essentially a **rationalization exercise**… does not appear to deliver strong testable implications or sharp restrictions beyond demonstrating that such an aggregation is possible… I would consider moving the macroeconomic extension to an appendix or an online appendix."
>
> R2: "If we assume heterogeneity is shaped *just so*, we can recover CES. That's nice, but I had trouble seeing it as an essential task. Maybe an appendix point."
>
> R3: "We already know e.g. from Sato (1975) that firm-level Leontief production functions can aggregate to CES, so is there much added by exploring one narrow way this could occur? To my mind this would be a better fit for a supplemental appendix."

## Verification against source
- Section 6 lives at `writeup/aggregation.tex` — 223 lines.
- CES aggregation result: `aggregation.tex:156` (`\label{eq:macro_agg_prod}`).
- Already has a mirror appendix at `writeup/appendix-aggregation.tex` — structure for relocation is partly in place.

## Options to address

1. **Move to appendix wholesale.** Retain one paragraph in the main body saying "the firm-level cost function admits a macro CES aggregation; see Appendix D". Moves ~7–8 pages out of the body.
2. **Move + add value per R1's suggestion.** Keep in body but add a comparison with Acemoglu (2025) or Aghion–Bunel (2024), showing that the chaining mechanism delivers macro predictions those don't. This addresses R1's "make the value explicit" ask.
3. **Hybrid:** move the derivation to appendix, keep a ½-page "macro implications" subsection in the body that contrasts with Acemoglu/Aghion.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Move wholesale | Low (½ day) | **High** — three referees explicitly asked for this; aligns with paper-length concerns |
| 2. Add Acemoglu/Aghion comparison | Medium (3–5 days: lit review + new derivation) | Medium — only addresses R1, risks more complexity |
| 3. Hybrid | Medium (2–3 days) | High — cheapest way to satisfy all three referees while keeping a macro hook in the body |

**Recommendation:** Option 3. Three referees said "move to appendix" — that's a strong signal. Keep a small visible macro tie-in so the paper still says something at aggregate level.
