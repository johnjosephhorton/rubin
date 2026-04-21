# Issue 15 — Specialists-earn-less assumption contradicts intuition

## Raised by
**R4.**

> "The model seems to predict/assume that specialists should earn a lower wage rate than generalists. In fact, this is a key part of the tradeoff firms face. Does that not contradict how we would typically think of wages vary with specialization? At least it seems to require significant discussion."

## Verification against source
- Current model: workers hold jobs whose wage bill is (total skill required) × (total time). Narrowing the job's task set reduces its skill requirement → lower per-unit-time wage rate. This is R4's concern.
- Standard intuition: specialists (surgeons, ML engineers) earn **more**, not less.
- The confusion is between "skill requirement per unit time" (what your worker needs to know) and "scarcity value" (what the market pays). The model's wage rate is purely skill-technical; it has no scarcity component.

## Options to address

1. **Add a scarcity margin.** Augment the wage equation with a market-clearing term: wage = skill-requirement-based floor + scarcity premium from labor supply. Specialists have high scarcity premium → earn more despite narrow task set. Preserves the model's mechanism while reconciling with data.
2. **Reinterpret.** Claim the "specialist wage" in the real world reflects investment in depth (skill level, not skill breadth), which the model does not nominally capture. Add a paragraph saying depth can be modeled by increasing `\skillcost{i}` for specialized steps, which raises wages of specialists in the model.
3. **Narrow the scope.** Argue that the "specialist" in the model is better interpreted as "narrower job within a given skill level" — the model compares jobs that vary on breadth, not depth. Specialists in the data are narrower *and* deeper; the wage comparison is ambiguous.
4. **Add a discussion-section paragraph** engaging the apparent contradiction head-on. Cheapest response but doesn't solve anything.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Scarcity margin | High (needs new theorem + rewriting related results) | Very high if successful — resolves the critique |
| 2. Reinterpretation (depth) | Medium (2–3 days) | **High** — preserves model, reframes language |
| 3. Narrow the scope | Low (½ day) | Medium — defensive but defensible |
| 4. Discussion paragraph | Low (½ day) | Low — R4 specifically asks for "significant discussion", which implies more than a paragraph |

**Recommendation:** Option 2, implemented via a subsection contrasting "breadth" vs. "depth" specialization. Optional Option 3 if reviewers push back. Option 1 is aggressive but might make the paper a better job-design theory (see Issue 03 Option 2 — which makes this more valuable).
