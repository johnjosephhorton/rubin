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

produces `0_main.pdf` (128 pages: the paper, then the Online Appendix, then the
Supplementary Appendix) and prints a summary of the log. The Online
Appendix uses `bibunits`, so `bibtex` has to run once on `0_main.aux` and once per
appendix on `bu1.aux` and `bu2.aux`, and plain `latexmk` does not pick the `bu*.aux`
files up on its own.

`latexmkrc` in this folder teaches it to, so

```
latexmk -pdf 0_main.tex
```

also produces a complete document, Online Appendix bibliography included. That is
the path Overleaf takes; keep the file alongside `0_main.tex` when syncing.

## Layout

| File | Contents |
|---|---|
| `0_main.tex` | Main file: title page, section list, bibliography, and the scaffolding for both appendices |
| `preamble.tex` | House format (identical to the bilateral-oligopoly preamble), plus a clearly marked block of paper-specific definitions at the bottom |
| `1_introduction.tex` … `8_conclusion.tex` | Body sections; the numeric prefix is the section number in the paper |
| `OA_A_*.tex` … `OA_C_*.tex` | Online Appendix sections; the prefix is the part and the appendix letter |
| `SA_A_*.tex` … `SA_F_*.tex` | Supplementary Appendix sections, same convention |
| `rubin.bib` | Bibliography |
| `plots/` | Figures: PNGs from the analysis notebooks, plus `TikZ_visualization/` for the diagrams drawn in TeX |
| `tables/` | Regression and illustration tables, written by the analysis notebooks or by hand |
| `_exhibit_options/` | Standalone builds of the examples and of the fragmentation exhibit. `_labels.tex` is a snapshot of `\newlabel`/`\bibcite` from `0_main.aux` so these resolve to the draft's own numbers; **refresh it whenever the draft's numbering changes** (the header of the file says how) |
| `referee_prep/` | Archived referee-lens findings and the one-lens verifier that consumes them. The findings predate the appendix reorganization, so `verify_lens.js` carries an old-name to new-name map for the agent |

## Appendix organization

The document builds as three parts in one PDF: the paper, then the Online Appendix,
then the Supplementary Appendix. Each appendix is self-contained, with its own title
page, its own contents list, its own page sequence and its own reference list, so that
either can be lifted out as a standalone file later.

The split exists because ReStud caps the online appendix at 30 pages. What a referee
must have to check the paper goes in the Online Appendix; the data construction and the
further tests go in the Supplementary Appendix.

**Online Appendix** (`OA - n` pages, sections `OA.A` to `OA.C`):

| | Appendix | File | What belongs in it |
|---|---|---|---|
| OA.A | Additional Tables and Figures | `OA_A_tables_and_figures.tex` | Exhibits the body cites but does not print: the notation summary, the job-design costs of Section 5.4, the configuration costs of Section 4.3, and the Prediction #3 position-reshuffle placebo figure |
| OA.B | Omitted Proofs | `OA_B_omitted_proofs.tex` | Proofs of the propositions stated in the body, in body order |
| OA.C | CES Representation at Macro Level | `OA_C_CES_representation.tex` | The Leontief-to-CES aggregation and the effective-AI-quality distribution behind it |

**Supplementary Appendix** (`SA - n` pages, sections `SA.A` to `SA.F`):

| | Appendix | File | What belongs in it |
|---|---|---|---|
| SA.A | Construction Details of the Main Sample | `SA_A_sample_construction.tex` | How the four data sources are assembled, and the model-chain vs. Anthropic-label discrepancy |
| SA.B | Alternative Definitions of Empirical Fragmentation, Step Similarity, and AI Execution | `SA_B_alternative_definitions.tex` | Predictions #2 and #3 re-estimated under different definitions of the objects they are built from, sequences held fixed |
| SA.C | GPT-5-mini Prompts | `SA_C_gpt_prompts.tex` | The two prompts, verbatim |
| SA.D | Robustness to Alternative GPT Prompts | `SA_D_prompt_robustness.tex` | All three predictions re-run on ten alternative orderings |
| SA.E | Robustness to Frequently-Executed Tasks Sample Restriction | `SA_E_frequency_robustness.tex` | All three predictions re-run on frequency-pruned samples |
| SA.F | External Validation of the Sequencing Results | `SA_F_external_validation.tex` | APQC PCF and 4TU event-log benchmarks |

**As of the last build the Online Appendix runs to `OA - 33`, three pages over the
ReStud limit.** OA.B is 18 pages of it and OA.C 11, so that is where any further
trimming has to come from.

Two rules the layout encodes, worth keeping. An exhibit belongs in OA.A when the body
cites it but no appendix discusses it, and in the appendix that discusses it otherwise.
And SA.B, SA.D, SA.E and SA.F are all robustness, but their titles rank them: SA.B
varies the *definitions* while holding the sequences fixed, whereas SA.D, SA.E and SA.F
vary the *sequences*, which is where the paper's identifying assumption lives, so those
three keep the heavier "Robustness to ..." and "External Validation of ..." forms.

OA.A goes first within its part because it is the only cross-cutting appendix: it holds
exhibits cited from Sections 3, 4, 5 and 7, so it belongs to neither the theory block
nor the empirical block and reads as an interruption anywhere between them. Of the two
ends, the front wins because its most-consulted item is a lookup aid rather than a
result: `Table OA.A.1` is the notation summary, cited from the fourteenth line of
Section 3.

### Appendix numbering

Every numbered object carries the part marker and the letter of the appendix it sits in,
and restarts at 1 there: `Table OA.A.1`, `Equation (OA.C.15)`, `Example OA.B.2`,
`Table SA.B.5`. The part marker is what makes a bare `\ref` unambiguous, since both
appendices have an Appendix A.

`0_main.tex` does this with one macro per part:

```latex
\startappendixpart{OA}{Online Appendix}
...
\startappendixpart{SA}{Supplementary Appendix}
```

which sets `\thesection` to `OA.\Alph{section}`, resets every counter, restarts the page
sequence at `OA - 1`, and lays out the part's title page. To go back to bare appendix
letters, drop the marker from `\thesection` there; nothing else depends on it.

`\theHsection` gets the marker too, and that part is not optional. hyperref builds PDF
anchors from a hidden second name per counter, `\theH<counter>`, not from the printed
number. Without the marker the two Appendix A's would both anchor at `A`, every exhibit
under them would collide, and the duplicates would be dropped, which is the bug the old
`OA-` scheme had. The counters themselves are attached to `section` by

```latex
\newcommand{\oaNumberWithin}[1]{%
  \expandafter\renewcommand\csname the#1\endcsname{\thesection.\arabic{#1}}%
  \counterwithin*{#1}{section}%
}
```

`\counterwithin*` installs the per-section reset without touching the printed form, and,
because hyperref hooks `\@addtoreset`, it redefines `\theH<counter>` to
`\theHsection.\arabic{<counter>}` as a side effect. Setting `\theH<counter>` by hand in
that macro does nothing: `\counterwithin*` runs afterwards and overwrites it.

Separating the two appendices takes three pieces beyond the numbering:

- **Contents.** Three `etoc` depth tags now, `main`, `onlineappendix` and
  `suppappendix`, one `\etocdepthtag.toc` per part. Each `\tableofcontents` shows its
  own tag and hides the other two.
- **References.** One `bibunit` per part, each with its own `\putbib[rubin]` and its own
  `\refname`, so the build writes `bu1.aux` and `bu2.aux`. `build.sh` already loops over
  `bu*.aux`, so nothing there changes.
- **Contents column widths.** The labels are `OA.A` and `SA.B.1`, not `A` and `B.1`, so
  tocloft's defaults are too narrow and the label collides with the title.
  `\cftsetindents` widens both, set once after the draft front matter is typeset.

The log carries 15 `destination with the same identifier` warnings, all benign: 13
`cite.*` anchors that `bibunits` necessarily duplicates between the two reference lists,
plus `page.OA-1` and `page.SA-1`, which collide because each part restarts its page
counter once for its title page and again for its first content page.

### Displays that ran into the right margin

The text block is 469.76pt. Nine places, in the body as well as the appendices, put ink
past the right margin. All nine are fixed, and no glyph in the document now sits right
of 540pt except on the landscape DWA table page, where the check measures the rotated
page in portrait coordinates and the reading is meaningless.

| Where | Natural width | Fix | After |
|---|---|---|---|
| p. 15, the three parameter triples of Section 4.1 | 500.3pt | `\smalldisplay` | 456.5pt |
| p. 38, `(12)`, the neighbour regression | 522.7pt | broken after the `\beta_2` term | 312.2 / 205.2pt |
| `(OA.C.14)`, the CES identity in aggregate variables | 505.0pt | `\smalldisplay` | 463.6pt |
| `(OA.C.15)`, the effective AI quality distribution | 563.5pt | `\smalldisplay` + broken after `(\bar\alpha)^{1/(\rho-1)}` | 266.4 / 255.6pt |
| the `\Gamma'(u)` derivative and `(OA.C.21)` in OA.C.3 | same | same | same |
| `Table OA.A.3`, configuration costs | 518.6pt | `\footnotesize`, as every other table float | 445pt |
| OA - 20, the three-option `\min` recursion for `R` | 481.4pt | `\smalldisplay` | 439.9pt |
| OA - 30, the sentence citing `(OA.C.11)`, `(OA.C.12)` and `(OA.C.13)` | 18pt overfull line | `sloppypar` | fits |
| OA - 7, the `Reduction 1` paragraph | 6pt overfull line | `sloppypar` | breaks before the `\min` |

Two of these needed more than a size change. `\footnotesize` alone still leaves (OA.C.15)
2pt over (471.8pt) and `\scriptsize` is 8.5pt type in a 12pt document, so it is `\small`
*and* broken. Equation (12) is 477.3pt at `\small`, still over, so it is broken at full
size. The OA - 7 case is not a wide box at all but a line TeX could not break within the
house `\tolerance`; `\mbox`-ing the formula makes it worse (25pt over), and `sloppypar`
is the fix. Table OA.A.3 was the source of the document's long-standing
`Overfull \hbox (48.87pt too wide)`, now gone; the only overfull boxes left in the log
are the two inside the landscape DWA table's `\resizebox`, harmless because the box is
scaled afterwards.

**The log will not tell you about any of this.** `preamble.tex` sets `\hfuzz=100pt`, so
TeX reports nothing that overhangs by less than 100pt, and display math can overrun
without a word either way. Six of the eight above were invisible in a clean build log.
Walk the glyph boxes instead, with the command below.

Setting a display `\small` needs care in this document. A display does not end the
paragraph around it, so the lines TeX has already accumulated for that paragraph are
contributed to the page when the display opens, at whatever `\baselineskip` is current
then. A bare `\begingroup\small` before the display therefore re-leads the text above
it, 14.44pt down to 13.55pt, which is the same trap the removed
`\AtBeginEnvironment{table}{\singlespacing}` hook fell into. `preamble.tex` defines
`\smalldisplay`, which saves `\baselineskip` before `\small` and restores it after.
Write `\begingroup\smalldisplay ... \endgroup` around a display, never a bare
`\begingroup\small`.

To re-check, walk the glyph boxes. Everything on a page should end at 540pt or less;
page 97 is the rotated landscape table and always reads as an outlier.

```bash
python3 -c "import fitz;d=fitz.open('0_main.pdf');[print(i+1,round(max(c['bbox'][2] for b in d[i].get_text('rawdict')['blocks'] if b['type']==0 for l in b['lines'] for s in l['spans'] for c in s['chars']),1)) for i in range(len(d))]" | awk '$2>540.5'
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
  appendix they sit in and restart there (`Table A.1`, `Equation (C.15)`); see
  "Appendix numbering" below.
- The `\ifoptionfinal` switch from the house format: dropping `final` from the
  document class turns on the draft table of contents, list of figures, list of
  tables and `todonotes`.

Three adjustments were needed because the house format sets 12pt type in a narrower
text block:

- `SA_C_gpt_prompts.tex`: the two verbatim prompt blocks are set `\footnotesize` so
  they stay inside their `tcolorbox`.
- `SA_F_external_validation.tex`: the two APQC tables are wrapped in
  `\adjustbox{max width=\textwidth}{...}`.
- `0_main.tex`: `\droptitle` is `-7em` rather than the house `-4em`, so that a
  two-line title, five authors and a three-line date block still leave room for the
  whole abstract on the title page.

## Exhibit conventions

Matched to the bilateral-oligopoly paper:

- **Placement.** 40 of the 46 live floats are `[!t]`, so exhibits sit at the top of a
  page, as they do in the bilateral draft (which uses `[t]`/`[!t]` throughout the main
  text). Six are not: `Figure OA.A.1` and `Table SA.B.5` are `[p]`, the first because it
  is a four-panel full-page figure and the second because a landscape float cannot sit at
  the top of a portrait page; OA.A's three tables are `[h!]`, placed by hand between
  `\newpage`s; and SA.B's leading table is `[t]` so that `\suppressfloats[t]` can keep it
  off the appendix's heading page (see "Appendix numbering" above).
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

`Table SA.B.5`, the landscape DWA table, is the document's one remaining float warning:

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

In the appendices, `Figure OA.A.1` and `Table SA.B.5` are `[p]`, so each takes a full
page with no body text on it. No main-text figure is `[p]`.

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
