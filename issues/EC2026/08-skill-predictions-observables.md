# Issue EC 08 — Connect skill predictions to measurable outcomes

## Raised by
**Review #2016B (Minor Comment 5).**

> "Section 6.1's discussion of worker skill effects (Examples E.1–E.3) is interesting but underdeveloped. The model shows AI can increase or decrease skill requirements depending on task structure, but the paper does not connect these predictions to measurable outcomes. Do we expect AI-exposed occupations to exhibit wage polarization? Changes in educational requirements? Skill premium shifts?"

## Why this matters
The paper makes qualitative predictions (upskilling vs. deskilling) but doesn't link them to the kinds of outcomes labor economists actually measure: wage distributions, education requirements by occupation, the college premium. For an empirical audience, this is the natural follow-through question. Not answering it weakens the empirical section's external relevance.

## Verification against source
- `discussion.tex` (Section 6.1 area) contains the upskilling/deskilling examples.
- The empirical section (`empirics.tex`) focuses on clustering, fragmentation, and adjacency regressions — not on wages or education.
- No connection between the theoretical skill predictions and wage data (BLS OES, ACS, CPS) or education-requirement data (O\*NET zones).

## Options to address

1. **Descriptive comparative statics.** Add a subsection in the discussion explicitly deriving predictions: (a) wage polarization (what share of AI-exposed occupations see skill distributions widen?), (b) shifts in education requirements, (c) skill-premium changes. No new data; theoretical extensions.
2. **Empirical validation against wage / education data.** Test the predictions: do AI-exposed occupations in Anthropic AEI show wage dispersion changes over time? Using ACS wage microdata, cross-check with exposure labels. Moderate new empirical work.
3. **Literature discussion only.** Add a paragraph in the conclusion connecting the paper's predictions to existing empirical literature on AI wage impacts (Autor, Acemoglu, Brynjolfsson et al.). Cheap but limited.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Comparative statics | Medium (1 week) | High — theoretical enrichment, sharpens predictions |
| 2. Empirical validation | High (3–4 weeks; new data pipeline) | **Very high** — directly answers the reviewer with evidence |
| 3. Literature discussion | Low (½ day) | Low-medium — signaling only |

**Recommendation:** Option 1 for the next revision. Option 2 is a natural follow-up paper if the reframing settles toward a more labor-economics-facing narrative.
