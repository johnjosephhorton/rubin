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

produces `0_main.pdf` (130 pages) and prints a summary of the log. The Online
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
| `A_omitted_proofs.tex` … `H_external_validation.tex` | Online Appendix sections; the letter prefix is the appendix letter in the paper |
| `rubin.bib` | Bibliography |
| `plots/` | Figures: PNGs from the analysis notebooks, plus `TikZ_visualization/` for the diagrams drawn in TeX |
| `tables/` | Regression and illustration tables, written by the analysis notebooks or by hand |

## What the house format changed relative to the superseded `main.tex`

Formatting only. No prose, math, notation, labels, cross-references, or exhibits were
edited. Section files are renamed to the house `N_name.tex` convention (body 1--6,
appendix 10--18, in document order); `rubin.bib` keeps its name for continuity with
the parent project.

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
  table of contents, `OA-`-numbered figures, tables, equations and theorem-like
  environments, `OA - n` page numbers, and its own reference list via `bibunits`.
  (The previous draft used `A.1`, `B.1`, … appendix numbering.)
- The `\ifoptionfinal` switch from the house format: dropping `final` from the
  document class turns on the draft table of contents, list of figures, list of
  tables and `todonotes`.

Three adjustments were needed because the house format sets 12pt type in a narrower
text block:

- `appendix-prompts.tex`: the two verbatim prompt blocks are set `\footnotesize` so
  they stay inside their `tcolorbox`.
- `appendix-external_validation.tex`: the two APQC tables are wrapped in
  `\adjustbox{max width=\textwidth}{...}`.
- `0_main.tex`: `\droptitle` is `-7em` rather than the house `-4em`, so that a
  two-line title, five authors and a three-line date block still leave room for the
  whole abstract on the title page.

## Exhibit conventions

Matched to the bilateral-oligopoly paper:

- **Placement.** Every float is `[!t]`, so exhibits sit at the top of a page, as they do
  in the bilateral draft (which uses `[t]`/`[!t]` throughout the main text). The one
  exception is the landscape DWA table, which is a full-page rotated float and keeps
  `[p]`; a landscape float cannot sit at the top of a portrait page.
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

### Pre-existing: the landscape DWA table overflows its page

`Table OA-8` is taller than the page and Panel (B) is clipped. This is not new: the
original `main.tex` reports seven `Float too large` warnings, including this table at
64pt over at 11pt type. At the required 12pt it is about 187pt over. The table needs to
be split across two pages or shrunk further; the `\resizebox{0.545\textheight}` is
already doing a lot of work.

## Exhibit and bibliography pass

- Every live float (42) now carries a `Notes:` block. Three tables had none and the notes
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
  float with a caption, a label, house `\subcaption` panel labels and notes. It keeps
  `[H]` placement because the surrounding text reads "the panels below".
- Equation (6) was running past the right margin; it is now broken over two lines in an
  `aligned` block.

### Bibliography

`rubin.bib` was normalised to the conventions of the bilateral paper:

- All 78 titles are double-braced so `chicago.bst` cannot lowercase them. This was the
  main source of the mess -- "GPTs are GPTs: Labor Market Impact Potential of LLMs" was
  printing as "Gpts are gpts: Labor market impact potential of llms".
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

Figure 5 in the main text is `[p]`, so it takes a full page with no body text on it.

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

The file has since been re-synced and is intact (219,770 bytes, valid `IEND`,
7169x1461). The recovered copy and `_broken_assets/` are gone, and the build reads
the good file directly.
