# Issue 24 — Table C.III notes run off the page

## Raised by
**R3 (fourth substantive bullet).**

> "The notes under table C.III run off the page, so I couldn't evaluate them. I don't doubt they're fine, of course, but please do make them visible in a revision!"

## Verification against source
- **No table labeled "C.III" currently exists in source.** Appendix tables appear to use numbers like C.1–C.4. The "C.III" referent may have been renumbered, merged, or removed since the Feb 2026 submission.
- Candidate files: `writeup/appendix-tables.tex` contains `\emph{Notes:}` blocks at lines 14, 32, 68, 86, 106, 126 — any of which could be the culprit depending on final numbering.

## Options to address

1. **Identify the culprit table.** Compile the submitted version; locate the actual table R3 calls C.III; find its corresponding `\emph{Notes:}` block in current source.
2. **Apply a robust fix** once identified:
   - Wrap the note in a `minipage` sized to the text width.
   - Use `\parbox{\textwidth}{...}` or `\begin{tablenotes}` (if using the `threeparttable` package).
   - Reduce note font size: `\footnotesize` or `\scriptsize`.
   - Shorten the note itself.
3. **Systematic pass.** Audit *all* appendix-table notes for width compliance using `\showframe` or a compile with overfull-hbox warnings enabled. Fix any that extend past `\textwidth`.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Identify + 2. Fix single | Low (1–2 hours) | **High** — direct reviewer ask |
| 3. Systematic audit | Low (½ day) | High — prevents recurrence |

**Recommendation:** Option 3. Cheap, avoids future reviewer complaints across other tables, ships a visibly cleaner document.
