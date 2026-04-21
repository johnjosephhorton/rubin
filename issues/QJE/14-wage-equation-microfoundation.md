# Issue 14 — Wage equation (1) lacks explicit micro-foundation

## Raised by
**R4.**

> "I was left unsure about what micro-foundation would deliver the wage formulation in equation (1). I think it would be very helpful to spell this out much more clearly and explicitly. Otherwise, this equation, which is at the heart of the theory's prediction for wages, seems to be too ad-hoc to be believable."

## Verification against source
- The wage/compensation expression is in `aggregation.tex:32–34` (label `eq:new_wage`), currently:
  ```latex
  w_{\manualLetter} \left(\sum_{T_b^{\manualLetter} \in J} \manualSkill{b}\right)
  + w_{\AIletter} \left(\sum_{T_b^{\AIletter} \in J} \AIskill{b}\right)
  ```
- The wage bill formula (skill × time) is closer to what R4 likely means as "equation (1)". In the current body it's in the intro of Section 3 (model). No compensating-differentials or matching-theoretic derivation is provided.

## Options to address

1. **Hedonic / compensating-differentials derivation.** Add a subsection deriving the wage bill from a worker who faces multiple jobs, each a bundle of skill requirements, and chooses the highest-utility bundle. Standard Rosen/Sattinger framework.
2. **Assignment-model derivation.** Workers differ in ability; firms post job bundles. In equilibrium, the marginal worker assigned to job J earns the bundle-specific wage rate. Cite Sattinger (1993) or Costinot–Vogel (2010).
3. **Efficiency-wage / effort derivation.** Worker effort scales with skill-weighted time; wage compensates accordingly. Less standard but quickest.
4. **Direct assumption with caveat.** Add a paragraph explicitly framing the wage equation as a reduced-form primitive, motivated by Becker–Murphy (human capital) logic. Least work but weakest response.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Compensating differentials | Medium (2–3 days) | **High** — standard labor econ; easy to defend |
| 2. Assignment model | Medium-high (1 week) | High — connects to existing task-based literature (Autor-Thompson style) |
| 3. Efficiency-wage | Low (1 day) | Medium — convenient but less central |
| 4. Assumption + caveat | Low (½ day) | Low — doesn't satisfy R4 |

**Recommendation:** Option 2 aligns best with how the rest of the paper frames itself (task-based, heterogeneous worker/AI). Option 1 is a faster second-best. Pair whichever you choose with a dedicated micro-foundations subsection that R4 can grade on.
