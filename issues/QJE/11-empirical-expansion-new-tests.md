# Issue 11 — Empirical section needs expansion beyond one regression

## Raised by
**R4.** Also editor's "concerns with the plausibility and durability of your core empirical work".

> "The validation approach is limited to one regression (equation 21) which does not have significant results for one of its two definitions of fragmentation… Complementing this analysis with other data points would therefore be valuable to convince the reader that the theory does well in predicting AI adoption."

## Verification against source
- The fragmentation regression is the p.41 paragraph at `empirics.tex:288` with results in Table III (also called out by R3 in Issue 19).
- Under **Definition 1** of fragmentation, the coefficient is insignificant in all three specifications; under **Definition 2** it is highly significant in all three. R3's reframing of this is in Issue 19.
- The paper has three empirical predictions tested: clustering (placebo-reshuffle benchmark), fragmentation regression, neighbor-spillovers regression. R4 considers tests 2 and 3 as variants of one idea ("adjacency fuels automation").

## Concrete test ideas (from R4 + R3 + plausible extensions)

1. **Same task across occupations.** The same O*NET task appears in multiple occupations. Compare AI-execution rate for that task in occupations where it has adjacent AI-able neighbors vs. occupations where it is isolated. Within-task, between-occupation variation.
2. **Historical automation analogs.** Did clustering of routine-cognitive tasks predict 1990s–2000s computerization? Test chaining prediction on RTI (routine task intensity) measures from Autor–Levy–Murnane (2003) and ALM successors.
3. **Multi-platform execution labels.** Cross-validate AEI (Anthropic) with other platform logs: OpenAI taxonomy, enterprise productivity telemetry. Directly addresses R2's platform-specificity critique.
4. **Firm-level adoption.** If firm AI-adoption survey data is accessible, regress adoption rate on occupation-weighted fragmentation index.
5. **Eloundou-only vs Eloundou+AEI.** Test whether chaining predicts *exposure-predicted-but-not-executed* (the gap).

## Effort vs. payoff

| Test | Effort | Payoff |
|---|---|---|
| 1. Same task across occupations | Medium (1–2 weeks; data exists) | **High** — leverages within-task variation, hard to dismiss |
| 2. Historical analogs | Medium-high (3–4 weeks; extensive lit review) | Medium — addresses durability, less central |
| 3. Multi-platform | High (data acquisition) | High if obtainable |
| 4. Firm-level | High (survey access) | Very high if obtainable |
| 5. Gap analysis | Low (1 week; uses existing data) | Medium — refines current story |

**Recommendation:** Ship Tests 1 and 5 in the next revision. They are cheap and directly address R4's "same task across occupations" suggestion. Test 3 is the best durability hedge (ties to Issue 10). Tests 2 and 4 for follow-on work.
