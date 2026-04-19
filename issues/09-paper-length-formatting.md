# Issue 09 — Paper is too long and densely formatted

## Raised by
**R2 (the rant).** Also implicit in editor's language ("streamlined and more focused version").

> "Who submits a paper that is single spaced, maybe 11 point font and that is 83 pages long? I mean, come on… for normal situations, we just go for the large font, readable pre-print."

## Verification against source
- `main.tex:1` — `\documentclass[11pt]{article}`
- No `\onehalfspacing`/`\doublespacing`/`\setstretch` command present — so default single spacing.
- Current compiled length: 80 pages (PDF).
- `\usepackage{fullpage}` (`main.tex:3`) tightens margins further.

## Options to address

1. **Reformat for readability.** Switch to 12pt, one-and-a-half spacing, normal margins. Page count will swell but perceived density drops. Estimated new length: ~120–140 pages.
2. **Actually shorten.** Move Sections IV (Issue 06) and VI (Issue 07) to appendix; compress examples (Issue 05 Option 3); tighten intro (Issue 08). Combined with Option 1, final length around 60–80 double-spaced pages.
3. **Split into two papers.** "Theory of automation chaining" + "Empirical evidence for chaining and fragmentation". Riskier — harder to place the theory paper alone.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Reformat only | ½ hour (preamble edit) | Low — doesn't reduce content, but removes the specific gripe |
| 2. Reformat + content cuts | Part of Issues 05–08 | **High** — addresses length AND complexity together |
| 3. Split | Very high (new paper) | Uncertain — could be great or could orphan half the work |

**Recommendation:** Option 2 as a natural byproduct of implementing Issues 05–08. Keep 11pt but add `\onehalfspacing` in the revision submission to signal effort without forcing all the cuts at once.
