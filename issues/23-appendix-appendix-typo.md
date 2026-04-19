# Issue 23 — "Appendix Appendix [letter]" formatting artifact

## Raised by
**R3 (nitpicks, last item).**

> "Just a formatting issue, but on six occasions the phrase 'Appendix Appendix [letter]' appears."

## Verification against source
- **Not reproducible in current source.** Grep across all `writeup/*.tex` returns zero matches for "Appendix Appendix" or variants like "Appendix\s+Appendix\s+[A-Z]".
- Either (a) the issue was already silently fixed between the submitted and the current version, or (b) the submitted PDF rendered `\ref{sec:appendix-X}` inside a sentence that already said "Appendix", producing a duplication when R3 read the rendered paper.

## Options to address

1. **Verify against the submitted PDF version.** Open the Feb 2026 PDF and grep for "Appendix Appendix". If present, trace back to which `\ref{...}` or `\appendix` usage caused it.
2. **Double-check common culprits.** Searches for patterns like `Appendix~\ref{sec:...}` where the label's title itself starts with "Appendix" will render as "Appendix Appendix X". Confirm none in the current source.
3. **Add a revision-response note.** "Thanks — fixed." (If Option 1 confirms already-fixed.)

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Verify submitted PDF | Trivial (10 min) | High — establishes whether there's anything left to fix |
| 2. Pattern audit | Trivial (15 min) | High — preventive |
| 3. Response note only | Trivial (2 min) | Medium |

**Recommendation:** Options 1 + 2 together. If nothing turns up, just note the fix in the response memo.
