# Issue 19 — p.41 "nearly all specifications" framing of fragmentation regression

## Raised by
**R3 (third substantive bullet).**

> "On p.41, you say 'the coefficient on the empirical fragmentation index is negative in nearly all specifications… The only exception arises when we use the stricter fragmentation measure in Definition 1 without SOC controls in column (1)…'"
>
> "From Table III, I think it would be more appropriate to just say that the fragmentation index is always statistically insignificant under Definition 1 (but always highly significant under Defn. 2). Certainly the -0.01 of regression 3 can hardly count toward the 'nearly all' (5 of 6) specifications."

## Verification against source
- The paragraph exists at `empirics.tex:288`: "…negative in nearly all specifications… the only exception arises when we use the stricter fragmentation measure in Definition 1…". R3's quote matches.
- Table III is not under a single file search. Typical referent is `empirics.tex` or in `tables/` subfolder `fragmentation_index_regression_execution.tex` / `fragmentation_index_regression_exposure.tex`.

## Options to address

1. **Rewrite honestly per R3's specification.** Say directly: "Under Definition 1 the coefficient is statistically insignificant across specifications; under Definition 2 it is negative and highly significant." Present results as heterogeneous-by-definition rather than mostly-significant-with-one-exception.
2. **Add a motivating discussion of Definition 1 vs Definition 2.** Explain why Def 2 is the preferred measure if theory pushes that way, or acknowledge Def 1 as a conservative check the data doesn't support.
3. **Drop Definition 1** if it is genuinely a weaker operationalization that adds no information.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Honest rewrite | Trivial (½ hour) | **Very high** — direct R3 ask, cheapest trust-builder |
| 2. Motivate two definitions | Low (½ day) | High — preemptively defends against "why two definitions?" |
| 3. Drop Definition 1 | Low | Medium — only if you're confident Def 2 is the right one |

**Recommendation:** Option 1 immediately, Option 2 in the same editing pass. Option 3 only if there's an affirmative theoretical reason to prefer Def 2.
