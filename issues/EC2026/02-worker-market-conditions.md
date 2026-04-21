# Issue EC 02 — Worker-side of the market is unmodeled

## Raised by
**Review #2016A.**

> "The paper also benefits from a discussion of the worker side of the market. For instance, what happens if workers are strategic and charge based on the marginal value of their work to the firm, or if there is a shortage or abundance of skilled workers?"

## Why this matters
The current model takes wages as exogenous (the wage bill is `(v_L + Σ skill_i) × (Σ time_i)`, with `v_L` and skill-pricing implicit). There is no labor-supply side: no strategic workers, no matching frictions, no scarcity of high-skill workers that could be driven up by demand. This leaves the reviewer uncertain whether the paper's predictions (e.g., upskilling reduces the skill wage premium) would survive with an explicit labor market.

## Verification against source
- `aggregation.tex:32-34` treats `w_M` and `w_A` as given; no clearing condition.
- No labor-demand curve, no participation constraint.
- The GE/aggregation section (`aggregation.tex`, Section 6) aggregates firms with a representative consumer but does not model a heterogeneous worker distribution or an equilibrium wage schedule.

## Options to address

1. **Acknowledge the partial-equilibrium scope.** Add a paragraph at the end of Section 3 or start of Section 6 explicitly stating that the model takes worker wages as given — appropriate for firm-level analysis, insufficient for general-equilibrium claims about wage premiums. Flag GE treatment as future work.
2. **Short competitive closure.** Assume a continuum of workers with a distribution of skills; firms post jobs with skill requirements; workers match to jobs of their skill level; wages adjust to clear the market. This gives an explicit `w(skill)` function. Probably one page of additional text.
3. **Strategic worker bargaining.** Each worker's outside option is the marginal value she creates; wages are set by Nash bargaining. More involved; possibly a separate subsection. Connects to R4 QJE's specialist-wages concern.
4. **Capacity constraint on skilled labor.** Model a fixed supply of high-skill workers. AI-induced demand for verification raises the skill wage premium; scarcity binds. A simple comparative static exercise.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Scope acknowledgment | Trivial (½ day) | Low-medium — honest but doesn't deliver new content |
| 2. Competitive closure | Medium (1 week) | High — adds substantive content, answers the reviewer, strengthens Section 6 |
| 3. Strategic bargaining | High (2–3 weeks; new propositions) | High — would also address the specialist-wage concern raised elsewhere |
| 4. Capacity constraint | Low (2–3 days comparative statics) | Medium — targeted, less ambitious |

**Recommendation:** Option 2 in Section 6 alongside the CES aggregation, or Option 1 as a defensive minimum. Option 3 if the revision reframes as a job-design theory and wants to take worker heterogeneity seriously.

*Response to this comment is tracked in [`referee_responses.md`](referee_responses.md).*
