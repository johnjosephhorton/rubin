# Issue 18 — "Human-generated" label terminology for Eloundou et al.

## Raised by
**R3 (first substantive bullet).**

> "On p.34 you say you use 'human-generated labels for AI exposure of O*NET tasks from Eloundou et al. (2024)'. But this confuses me, since almost all of the AI exposure labels from Eloundou et al. are AI-generated; they just validate them against a small set of human-generated labels."
>
> "If you used the whole dataset, don't refer to the labels as human-generated."

## Verification against source
- Grep for "human-generated labels" in `writeup/*.tex` returned **no matches** — either the phrase was rewritten already, or R3 was paraphrasing slightly and the real text differs.
- Worth verifying against the *submitted* PDF version (Feb 2026). Possible that a later revision silently fixed this, in which case this is a non-issue; otherwise the current text may still describe the Eloundou labels inaccurately.

## Options to address

1. **Verify and correct.** Grep `empirics.tex` for any language characterizing Eloundou labels; rephrase to something like "human-validated, AI-generated exposure labels" or just "exposure labels from Eloundou et al. (2024)".
2. **Cite carefully.** Whatever the exact phrasing, make sure it matches Eloundou et al.'s own framing.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Verify + correct | Trivial (15 min) | **High** (per-cost) — a direct reviewer ask, costs nothing to satisfy |

**Recommendation:** Do this in the first editing pass. If the text has already been updated, add a line to the revision response memo noting the fix.
