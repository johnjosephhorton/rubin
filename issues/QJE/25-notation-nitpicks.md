# Issue 25 — Notation and phrasing nitpicks

## Raised by
**R3 (nitpicks bullet list).** Low-importance individually; bundled here for efficiency.

## Individual items and verification

### 25.1 — p.17 "both … along with"
**R3:** "Instead of 'both… along with', either cut 'both' or use 'both… and'."

**Verification:** Confirmed at `optimization.tex:77` — "choose both the set of AI chains to implement---yielding a vector of skill and time costs---along with a job design".

**Fix:** Replace with either "both the set of AI chains… and a job design" or "the set of AI chains… along with a job design". Trivial.

### 25.2 — p.22 "Write F for the set of steps that fail, and C… for the random variable"
**R3:** Both F and C are random variables; one is a function of the other. Rephrase: "Write F as the random variable representing the set of steps that fail, and C = {C₁, …, Cₖ} as the corresponding collection…". Also "Given a realization of C and F" suggests independence — should be "Given a realization of F". And replace "non-failed" with "successful".

**Verification:** Not found via exact phrase grep in current source. Likely already revised or paraphrased. Verify against submitted PDF and apply R3's suggestion if still present.

### 25.3 — p.28 expression (6) notation
**R3:** Sum across tasks in manual/AI-assisted subsets of J — name the subsets (e.g. `J^M`, `J^A`). Also "total compensation" is a weighted *sum*, not an average.

**Verification:** Current source uses `T_b^{\manualLetter}` and `T_b^{\AIletter}` with sums like `\sum_{T_b^{\manualLetter} \in J}`. The subsets themselves (`J^M`, `J^A`) are not explicitly named. See `aggregation.tex:32`.

**Fix:** Define `J^M ≡ {T_b \in J : b executed manually}` and `J^A` analogously. Rewrite the sum as `\sum_{T_b \in J^M}`. Replace "average" with "sum" if "weighted sum" is what's meant.

### 25.4 — p.32 ρ < 0 complementarity
**R3:** "ρ < 0 indicates some degree of complementarity — but this would be true of any ρ < 1. What ρ < 0 entails is *gross* complementarity. But do you use ρ < 0? It seems to me that you only use ρ < 1."

**Verification:** Confirmed at `aggregation.tex:165` — "we assume that ρ<0 (which implies elasticity of substitution σ<1), indicating that macro-level production exhibits some degree of complementarity".

**Fix:** If the paper only uses ρ<1, weaken the assumption to ρ<1 and replace "some degree of complementarity" with "complementarity" (which holds for ρ<1). If ρ<0 is actually required somewhere, rephrase as "gross complementarity" and cite where it is used.

## Options to address

1. **One editing pass** through all five items. Total: under 2 hours.
2. **Verify against submitted PDF** for items 25.2 — if already fixed, add revision-response notes.

## Effort vs. payoff

Combined effort: **Very low** (one afternoon). Combined payoff: **Medium-high** per-cost — cheap signal of care; a reviewer who sees their nitpicks addressed feels listened to.

**Recommendation:** Do all of these in a single sweep during the first editing pass.
