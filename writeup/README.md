# Chaining Tasks, Redefining Work: A Theory of AI Automation

The paper, set in the house format of
`Allocative Efficiency in Bilateral Oligopoly/1_draft`.

Sources, exhibits, and bibliography all live in this folder. `0_main.tex` is the
main file; `plots/` and `tables/` hold everything it inputs, so `0_main.tex` only
needs

```latex
\graphicspath{{./}}
```

This draft previously lived in a `draft_mert/` subfolder that shadowed the parent's
`plots/` and `tables/`. It was consolidated here on 2026-08-26: the newer copy of
each duplicated exhibit was kept, the superseded `main.tex` and its section files
were removed, and the two-level search path was collapsed.

## Build

```
./build.sh
```

produces `0_main.pdf` (129 pages) and prints a summary of the log. The Online
Appendix uses `bibunits`, so `bibtex` has to run once on `0_main.aux` and once on
`bu1.aux`, and plain `latexmk` does not pick the `bu*.aux` files up on its own.

`latexmkrc` in this folder teaches it to, so

```
latexmk -pdf 0_main.tex
```

also produces a complete document, Online Appendix bibliography included. That is
the path Overleaf takes; keep the file alongside `0_main.tex` when syncing.

## Layout

| File | Contents |
|---|---|
| `0_main.tex` | Main file: title page, section list, bibliography, Online Appendix scaffolding |
| `preamble.tex` | House format (identical to the bilateral-oligopoly preamble), plus a clearly marked block of paper-specific definitions at the bottom |
| `1_introduction.tex` … `8_conclusion.tex` | Body sections; the numeric prefix is the section number in the paper |
| `A_omitted_proofs.tex` … `I_external_validation.tex` | Online Appendix sections; the letter prefix is the appendix letter in the paper |
| `rubin.bib` | Bibliography |
| `plots/` | Figures: PNGs from the analysis notebooks, plus `TikZ_visualization/` for the diagrams drawn in TeX |
| `tables/` | Regression and illustration tables, written by the analysis notebooks or by hand |
| `_exhibit_options/` | Standalone builds of the examples and of the fragmentation exhibit. `_labels.tex` is a snapshot of `\newlabel`/`\bibcite` from `0_main.aux` so these resolve to the draft's own numbers; **refresh it whenever the draft's numbering changes** (the header of the file says how) |
| `referee_prep/` | Archived referee-lens findings and the one-lens verifier that consumes them. The findings predate the appendix reorganization, so `verify_lens.js` carries an old-name to new-name map for the agent |

## Appendix organization

The Online Appendix runs theory first, then empirics. Each appendix is one file and
one `\section`; `0_main.tex` `\clearpage`s between them.

| | Appendix | File | What belongs in it |
|---|---|---|---|
| A | Omitted Proofs | `A_omitted_proofs.tex` | Proofs of the propositions stated in the body, in body order |
| B | Macro-level Production Function | `B_macro_production.tex` | The Leontief-to-CES aggregation and the effective-AI-quality distribution behind it |
| C | Additional Tables for the Theory Sections | `C_theory_tables.tex` | Exhibits the body refers to but does not print: the notation summary, the job-design costs of Section 5.4, the configuration costs of Section 4.3 |
| D | Construction Details of the Main Sample | `D_sample_construction.tex` | How the four data sources are assembled, and the model-chain vs. Anthropic-label discrepancy |
| E | Additional Robustness Tests for Predictions #2 and #3 | `E_prediction_robustness.tex` | Execution-based EFI; placebo reshuffles, the GPT-filtered similarity sample, and AI-automation outcomes |
| F | GPT-5-mini Prompts | `F_gpt_prompts.tex` | The two prompts, verbatim |
| G | Robustness to Alternative GPT Prompts | `G_prompt_robustness.tex` | All three predictions re-run on ten alternative orderings |
| H | Robustness to Frequently-Executed Tasks Sample Restriction | `H_frequency_robustness.tex` | All three predictions re-run on frequency-pruned samples |
| I | External Validation of the Sequencing Results | `I_external_validation.tex` | APQC PCF and 4TU event-log benchmarks |

The rule C encodes is the one worth keeping: an exhibit belongs in C when the body
cites it but no appendix discusses it, and in the appendix that discusses it otherwise. C used to be folded together with E in a single "Additional Tables and
Robustness Tests" appendix, which put the notation table and the Prediction #3 placebo
figures under one heading.

### Appendix numbering

Every numbered object in the appendix carries the letter of the appendix it sits in and
restarts at 1 there: `Table C.1`, `Figure E.3`, `Equation (B.15)`, `Example A.2`. The
scheme is one macro in `0_main.tex`, applied to each counter:

```latex
\newcommand{\oaNumberWithin}[1]{%
  \expandafter\renewcommand\csname the#1\endcsname{\thesection.\arabic{#1}}%
  \counterwithin*{#1}{section}%
}
```

`\thesection` is already `\Alph{section}` by then, and `\counterwithin*` installs the
per-section reset without touching the printed form the `\renewcommand` just set.

The starred `\counterwithin` also fixes the hyperlinks, which is the part worth knowing
about. hyperref builds PDF anchors out of a hidden second name per counter,
`\theH<counter>`, not out of the printed number. Nothing ever set it under the old `OA-`
scheme, so appendix `Equation (OA.1)` and body `Equation (1)` both anchored at
`equation.1`; the appendix anchor was dropped as a duplicate and every appendix link
landed in the main text. hyperref hooks `\@addtoreset`, so `\counterwithin*` redefines
`\theH<counter>` to `\theHsection.\arabic{<counter>}` as a side effect and the anchors
come out as `equation.B.15`. Setting `\theH<counter>` by hand in the macro does nothing:
`\counterwithin*` runs afterwards and overwrites it.

The log went from 56 `destination with the same identifier` warnings to 15: 13 `cite.*`
anchors that `bibunits` necessarily duplicates between the two reference lists, plus
`page.OA-1` and `page.OA-2`, which collide because the appendix resets the page counter
twice, once for its title page and contents and once for its first content page.

Page numbers are unaffected and stay `OA - n`.

Two of the four floats that deviate from the house `[!t]` do so for this reason. The
leading table of Appendix C and of Appendix E are `[t]`, and each appendix opens with `\suppressfloats[t]`: `!` overrides
`\suppressfloats`, so with `[!t]` those two tables were typeset at the top of the page
carrying their own appendix heading, above it. Every later float keeps `[!t]` and
follows the deferred leader in order.

### Displays and tables that ran into the right margin

The text block is 469.76pt. Four exhibits in the appendix were wider than that and
overhung the right margin; all four are fixed:

| Exhibit | Natural width | Fix | After |
|---|---|---|---|
| `(B.14)`, the CES identity in aggregate variables | 505.0pt | `\small` | 463.6pt |
| `(B.15)`, the effective AI quality distribution | 563.5pt | `\small` + broken after `(\bar\alpha)^{1/(\rho-1)}` | 266.4 / 255.6pt |
| the `\Gamma'(u)` derivative and `(B.21)` in B.3 | same | same | same |
| `Table C.3`, configuration costs | 518.6pt | `\footnotesize`, as every other table float | 445pt |

`\footnotesize` alone was not enough for (B.15): it comes to 471.8pt, still 2pt over.
`\scriptsize` fits but is 8.5pt type in a 12pt document, so the display is set `\small`
and broken instead. Table C.3 was the source of the document's long-standing
`Overfull \hbox (48.87pt too wide)`, which is now gone; the only overfull boxes left in
the log are the two inside the landscape DWA table's `\resizebox`, which are harmless
because the box is scaled afterwards.

Four places still put glyphs past the right margin. None is new and none was in scope
for the appendix pass, but they are worth a look:

| PDF page | What | Over by |
|---|---|---|
| 15 | the `(8, 4, 0.9)` parameter display in Section 4 | 30pt |
| 38 | the neighbour regression, Equation (12), in Section 7 | 26pt |
| OA - 3 | a `\min\{V_1, V_2, V_3\} <` line in Appendix A | 6pt |
| OA - 16 | a display in Appendix A | 13pt |

The same walk also flags PDF page 97, the landscape DWA table; that one is the
known clipping documented under "the landscape DWA table overflows its page" below.

Setting a display `\small` needs care in this document. A display does not end the
paragraph around it, so the lines TeX has already accumulated for that paragraph are
contributed to the page when the display opens, at whatever `\baselineskip` is current
then. A bare `\begingroup\small` before the display therefore re-leads the text above
it, 14.44pt down to 13.55pt, which is the same trap the removed
`\AtBeginEnvironment{table}{\singlespacing}` hook fell into. `B_macro_production.tex`
defines `\smalldisplay`, which saves `\baselineskip` before `\small` and restores it
after, and the four displays use `\begingroup\smalldisplay ... \endgroup`.

To re-check, walk the glyph boxes rather than trusting the log, which stays quiet for
some display math:

```bash
python3 -c "import fitz;d=fitz.open('0_main.pdf');[print(i+1,round(max(c['bbox'][2] for b in d[i].get_text('rawdict')['blocks'] if b['type']==0 for l in b['lines'] for s in l['spans'] for c in s['chars']),1)) for i in range(len(d))]" | awk '$2>544'
```

## What the house format changed relative to the superseded `main.tex`

Formatting only. No prose, math, notation, labels, cross-references, or exhibits were
edited. Section files are renamed to the house `N_name.tex` convention: the body is
`1_introduction.tex` .. `8_conclusion.tex` and the appendix is `A_..` .. `I_..`, both in
document order. `rubin.bib` keeps its name for continuity with the parent project.

- `\documentclass[12pt, final]{article}` instead of `11pt` + `fullpage`; `geometry`
  with 1in margins.
- Fonts: `newpxtext` / `newpxmath` (Palatino) instead of Computer Modern.
- Section heads: `titlesec` / `sectsty`, 14pt sections and 13pt subsections, house
  spacing above and below.
- Captions above floats (`\captionsetup{position=top}`), bold labels, centered
  justification, `(a)`-style subfigure labels; `\singlespacing` inside floats.
- References: `chicago` via `natbib`, `\bibsep = 0.2ex`, printed at
  `\begin{spacing}{0.98}`, instead of `aer`.
- In-text citations are uniform: one or two authors named in full, three or more
  collapsed to "First et al.", including on a reference's first appearance. This is
  `chicago.bst`'s own `format.lab.names` rule (`numnames > 2`), reached by dropping the
  house preamble's `[longnamesfirst]` option, which would otherwise spell out every
  author the first time a work is cited. Reference-list entries still name all authors.
- Hyperlink colors: dark blue for internal links, dark red for citations and URLs.
- Footnotes: `footmisc` with the house `\footnotesep` and `\footnotemargin`.
- Title page in the house style: affiliations and emails as per-author `\thanks`
  footnotes, acknowledgments on the last author's footnote, `titling` with a
  `\droptitle` offset. Keywords and JEL codes are the one house element the paper
  does not yet have; the lines are in `0_main.tex`, commented out, ready to fill in.
- Online Appendix in the house style: its own title page, its own `etoc`-filtered
  table of contents, `OA - n` page numbers, and its own reference list via `bibunits`.
  Figures, tables, equations and theorem-like environments carry the letter of the
  appendix they sit in and restart there (`Table C.1`, `Equation (B.15)`); see
  "Appendix numbering" below.
- The `\ifoptionfinal` switch from the house format: dropping `final` from the
  document class turns on the draft table of contents, list of figures, list of
  tables and `todonotes`.

Three adjustments were needed because the house format sets 12pt type in a narrower
text block:

- `F_gpt_prompts.tex`: the two verbatim prompt blocks are set `\footnotesize` so
  they stay inside their `tcolorbox`.
- `I_external_validation.tex`: the two APQC tables are wrapped in
  `\adjustbox{max width=\textwidth}{...}`.
- `0_main.tex`: `\droptitle` is `-7em` rather than the house `-4em`, so that a
  two-line title, five authors and a three-line date block still leave room for the
  whole abstract on the title page.

## Exhibit conventions

Matched to the bilateral-oligopoly paper:

- **Placement.** 42 of the 46 live floats are `[!t]`, so exhibits sit at the top of a
  page, as they do in the bilateral draft (which uses `[t]`/`[!t]` throughout the main
  text). Four are not: `Figure E.1` and `Table E.5` are `[p]`, the first because it is a
  four-panel full-page figure and the second because a landscape float cannot sit at the
  top of a portrait page; and the leading table of Appendix C and of Appendix E are `[t]`
  so that `\suppressfloats[t]` can keep them off their appendix's heading page (see
  "Appendix numbering" above).
- **Caption first.** All 41 floats already had `\caption` before the graphic or tabular,
  matching the house `position=top` caption setup. Nothing to change.
- **Notes.** All 38 live notes blocks were rewritten from
  `\footnotesize{\emph{Notes:} ...}` into the house block:

  ```latex
  \begin{minipage}{1\linewidth}
  \begin{spacing}{0.2}
  {\fontsize{9.5pt}{10pt}\selectfont Notes: ...}
  \end{spacing}
  \end{minipage}
  ```

  So notes are now 9.5pt on a 10pt body at `spacing{0.2}`, full measure, with a plain
  (not italicised) `Notes:` label, exactly as in the bilateral draft. Note text is
  unchanged: all 41 blocks were diffed character by character after conversion.

Three floats are commented out in the source and were left exactly as they are. The
small example tables and panel figures inside `example` environments are not floats at
all -- they are inline `center` blocks -- so they stay where they sit in the argument,
which is what you want for them.

### Pre-existing: the landscape DWA table is too tall for its page

`Table E.5`, the landscape DWA table, is the document's one remaining float warning:

```
LaTeX Warning: Float too large for page by 49.28372pt
```

Nothing is clipped in the current build: both panels and the notes render in full on
PDF page 97, and an earlier note in this file claiming Panel (b) was cut off is out of
date. What the warning costs is legibility, since `\resizebox{0.545\textheight}` is
shrinking Panel (b) to about 5pt type to make it fit. This is not new: the original
`main.tex` reported seven `Float too large` warnings, this table among them, at 64pt
over at 11pt type. Splitting it across two pages is still the real fix.

## Exhibit and bibliography pass

- Every live float (46 as of this pass) carries a `Notes:` block. Three tables had none and the notes
  for them are newly drafted -- worth a read: the Table 1 horizons summary, the model
  Notation Summary, and the job-design notation table.
- All table floats are `\footnotesize`.
- The two-part table footnotes are gone: the significance line that regression tables
  carried as a `\multicolumn` row inside the tabular ("Standardized coefficients.
  Clustered standard errors in parentheses...") is now the first sentence of the single
  `Notes:` block. Seven table files were stripped this way; the
  rest were already in the right shape.
- Caption-to-exhibit and exhibit-to-notes spacing now match the other paper structurally:
  37 floats were using a `center` environment (which adds `\topsep` at both ends) where
  the house format uses `\centering`, and stray `\vspace` commands sat between the caption
  and the exhibit or between the exhibit and the notes. Both are removed. `\arraystretch`
  was also dropped -- the original GenAI preamble set it to 1.25, which stretched every
  table row by a quarter and was the only remaining reason table geometry differed.
- The Example figure (cost and marginal benefit as AI quality rises) is now a proper
  float with a caption, a label, house `\subcaption` panel labels and notes. It was
  given `[H]` at the time because the surrounding text read "the panels below"; that
  prose has since been rewritten and the figure is `[!t]` like the rest. No float in the
  document uses `[H]` any more.
- Equation (6) was running past the right margin; it is now broken over two lines in an
  `aligned` block.

### Bibliography

`rubin.bib` was normalised to the conventions of the bilateral paper:

- All 82 titles are double-braced so `chicago.bst` cannot lowercase them. This was the
  main source of the mess -- "GPTs are GPTs: Labor Market Impact Potential of LLMs" was
  printing as "Gpts are gpts: Labor market impact potential of llms". Three entries added
  after that pass had gone back to single braces and were printing lowercased again
  ("How ai redraws job boundaries", "A task-interdependency model...", "What makes new
  work different..."); they are double-braced now, so the invariant holds for all 82.
- 14 working papers and preprints moved from `@techreport`/`@unpublished`/`@misc` to
  `@article` with the series in `journal`: `NBER Working Paper, No. 32872`,
  `arXiv Preprint arXiv:2503.04761`, and so on.
- Dropped throughout: `month` (the source of "(2021, January)"), plus `type`, `series`,
  `institution`, `eprint`, `archivePrefix`, `primaryClass`, `urldate`, `issn`, `isbn`,
  `doi`, and the `organization` field that was printing a publisher's street address into
  the reference list.
- Field names lowercased, field order made uniform, `@inbook` folded into `@incollection`.

## Subfigure panels

Every figure with panels now uses the standard `subcaption` structure: `\includegraphics`
first, then `\subcaption{...}` below the plot, auto-numbered (a), (b), (c), ... by the
preamble's `\renewcommand\thesubfigure{(\alph{subfigure})}` in the house bold-label style.
34 panels across 12 figures were converted from hand-typed labels above the plot
(`\captionsetup{labelformat=empty}` plus `\caption{Panel (A): ...}` or `\caption{(a) ...}`).
Prose references to Panel (A)/(B)/(C)/(D) were lower-cased to match, including the two
stacked panels of the landscape DWA table.

In the appendix, `Figure E.1` and `Table E.5` are `[p]`, so each takes a full page with
no body text on it. No main-text figure is `[p]`.

## A second bug: footnotes did not match the house format

The original GenAI preamble loads `\usepackage{dialogue}`. The paper never uses a dialogue
environment, but the package redefines `\@makefntext`, which defeats `footmisc`'s `marginal`
option: the footnote marker was being set at `\parindent` with flush-left continuation lines
instead of hanging in the margin. Measured against the bilateral paper:

| | marker x | continuation x |
|---|---|---|
| bilateral | 68.51pt | 72.00pt |
| this paper, with `dialogue` | 86.45pt | 72.00pt |
| this paper, `dialogue` removed | 68.51pt | 72.00pt |

Isolated by bisecting the preamble packages against a minimal document. `color-edits`, the
other unused carryover, is harmless and stays. The superseded `main.tex` had this same
problem, since it loaded `dialogue` too.

## One bug fixed in the house preamble

The bilateral-oligopoly preamble carries

```latex
\AtBeginEnvironment{table}{\vspace{-0.5\baselineskip}\singlespacing}
\AtBeginEnvironment{figure}{\vspace{-0.5\baselineskip}\singlespacing}
```

Both begin-hooks are removed here. `\AtBeginEnvironment` fires *before* the float box
is opened, so neither line does what it looks like it does:

- `\singlespacing` lands on the paragraph TeX is still building whenever
  `\begin{table}` or `\begin{figure}` follows body text with no blank line, and the
  whole preceding paragraph is then set single-spaced. 22 paragraphs in this draft
  were affected; measured line pitch was 14.4pt against 17.9pt for normal body text.
  It buys nothing in exchange, because `\@xfloat` calls `\@parboxrestore` and float
  captions, bodies and notes are single-spaced either way.
- `\vspace` stays behind at the anchor point rather than travelling with the float, so
  any float that migrates to the top or bottom of a page leaves half a line of
  negative space at an unrelated spot in the running text.

The `\AtEndEnvironment` hooks are kept: they fire inside the float box and trim its
bottom, which is harmless.

After the fix, 1,534 of 1,572 full-measure body lines sit at 17.9pt; the remainder are
footnotes at 15.3pt, which are single-spaced by design and permitted by REStud.

Floats now take standard `\intextsep` above, so they sit about 9pt closer to the
preceding text than they did with the bug in place. If you want them looser, set
`\setlength{\intextsep}{...}` in `preamble.tex` — unlike the `\vspace`, that travels
with the float.

## Theorem-like environments

The house format leaves propositions, lemmas, definitions and examples unboxed, so
the blue and red `tcolorbox` frames of the previous draft are off. To bring them
back, set

```latex
\boxedresultstrue
```

in `preamble.tex` (the switch and the box definitions are already there).

## Resolved: the truncated DWA exhibit

`plots/execTypeVaryingDWA_noTasksWithRepetitiveDWAs/is_ai/AME_filtered_is_ai_no_fe_no_dwa.png`
used to be truncated (163,833 bytes, no `IEND` chunk), which made `pdflatex` abort.
The draft carried a recovered copy at the same relative path and relied on the
`{./}` before `{../}` search order to override it.

The file has since been re-synced and is intact (441,510 bytes, valid `IEND`,
7172x1467). The recovered copy and `_broken_assets/` are gone, and the build reads
the good file directly.
