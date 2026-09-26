# Prose review of "Chaining Tasks, Redefining Work" (2026-09-26)

Draft reviewed: all 18 `.tex` sources in `writeup/` (main text, Online Appendix, Supplementary Appendix) at commit `792c8c1`. Page numbers are from a fresh 129-page compile of those sources: body `p. N`, Online Appendix `p. OA-n`, Supplementary Appendix `p. SA-n`.

This file replaces the 2026-09-08 prose list. All 16 of that list's still-open items are folded in below. Substantive problems are in `REVIEW_issues_2026-09-26.md`.

The draft is mechanically clean: no doubled words, no straight quotes in prose, no stray `\%` spacing. Most of what follows comes from the latest round: the Overleaf edits introduced three or four grammar slips, and the APQC-to-PCF renaming stopped short of the appendices.

---

## House style violations

### S1. Em dash (the only one left in the prose)
- **Location**: p. 6, "Long-Run Job Design" (`1_introduction.tex:116`)
- **Now**: "while the same reduction---and any direct saving in coordination---makes each boundary cheaper to draw."
- **Fix**: "while the same reduction, together with any direct saving in coordination, makes each boundary cheaper to draw."

### S2. Explanatory colons
Each one splices a sentence to its own unpacking. Colons that introduce genuine lists, such as "three things about every step:", can stay.

| Location | Now | Fix |
|---|---|---|
| p. 5, `1_introduction.tex:112` | "How broadly to define each job involves a trade-off: adding more tasks ..." | "How broadly to define each job involves a trade-off. Adding more tasks ..." |
| p. 10, Fig. 2 notes, `3_shortrun.tex:46` | "The bottom layer shows the resulting tasks: a run of automated steps ..." | "In the bottom layer, a run of automated steps ending in an augmented step forms an AI chain ..." |
| p. 17, Fig. 4 notes, `4_implications.tex:129` | "which the firm executes as a single AI chain: Step 1 is automated and Step 2 is augmented and verified." | "which the firm executes as a single AI chain, automating Step 1 and verifying Step 2." |
| p. 30, fn., `6_extensions.tex:35` | "asks more of the economy than the firm-level argument does: capital productivity is common ..." | "asks more of the economy than the firm-level argument does. It requires that capital productivity be common ..." |
| p. 35, `7_empirics.tex:68` | "The two placebos isolate different margins: the first holds ..." | "The two placebos isolate different margins. The first holds ..." |
| p. OA-5, `OA_B:5` | "The subsections follow the order in which the results appear in the body: Appendix B.1 proves ..." | "The subsections follow the body's order. Appendix B.1 proves ..." |
| p. SA-27, `SA_E:57, 65` | "the object used in the main analysis: the occupation's sixteen O*NET tasks ..." | "the object used in the main analysis, which lists the occupation's sixteen O*NET tasks ..." |
| p. SA-26, `SA_E:66` | "one of the most demanding in the grid: seven of the sixteen tasks ..." | "one of the most demanding in the grid, which keeps seven of the sixteen tasks ..." |
| p. SA-28, `SA_E:82` | "as a forest plot: each row is a frequency cut ..." | "as a forest plot in which each row is a frequency cut ..." |
| p. SA-29, Fig. SA.E.2 notes, `SA_E:93` | "for that cut: the mean length of a maximal run ..." | "for that cut, that is, the mean length of a maximal run ..." |
| p. SA-39, `SA_F:180` | "the recurring case is the one anticipated above: parents whose children are ..." | "the recurring case is the one anticipated above, namely parents whose children are ..." |
| p. SA-39, `SA_F:196` | "is about exactly this object: a workflow partitioned into jobs, ..." | "is about exactly this object, a workflow partitioned into jobs, ..." |

### S3. AI-tell phrasing
| Location | Now | Fix |
|---|---|---|
| p. 4, `1_introduction.tex:56` | "they differ in one crucial respect:" | "they differ in one respect that matters for everything below:" |
| p. 14, `4_implications.tex:39` | "Importantly, step k's own verification cost ..." | "Step k's own verification cost ..." |
| p. 21, `4_implications.tex:242` | "It is also worth noting that the jump at α ≈ 0.92 is larger than the one at α = 0.50. Forming the chain pulls ..." | "The jump at α ≈ 0.92 is also larger than the one at α = 0.50, because forming the chain pulls ..." |
| p. 21, `5_longrun.tex:5` | "Moreover, adjustments in the labor market let ..." | "Adjustments in the labor market also let ..." |
| p. 26, `5_longrun.tex:185` | "Notably, the hand-off shares the skill requirements of the worker doing it for an extra amount of time added to the last task in their job." | "The hand-off is paid at the skill level of the worker who performs it, as extra time appended to the last task in their job." |
| p. 29, `6_extensions.tex:30` | "and it is worth stating what makes it work." | "and we state what makes it work." |
| p. 30, `6_extensions.tex:40` | "Importantly, the organization of work does not disappear ..." | "The organization of work does not disappear ..." |
| p. SA-1, `SA_A:18` | "to fully leverage AI capabilities" | "to make full use of AI capabilities" |
| p. SA-2, `SA_A:79` | "it is worth emphasizing a few points about the dataset" | "we note three features of the dataset" |
| p. SA-39, `SA_F:174` | "and it is worth being clear about what they are" | "and they deserve a closer look" |
| pp. OA-26, OA-28, OA-31, `OA_C:68, 106, 183`; pp. OA-16, OA-18, `OA_B:402, 403, 406, 461` | "Furthermore", "Moreover" (six), "Crucially" | Delete each one. The following sentence already reads as an addition. |
| p. 6 and p. 28, `1_introduction.tex:117` and `5_longrun.tex:233` | "not a prediction about job breadth but an account of what determines it" (twice, almost word for word) | Keep it in Section 5.4. In the intro, write "The framework therefore identifies what determines job breadth, and what would have to be measured to settle the question empirically." |
| `1_introduction.tex:89`, `4_implications.tex:59`, `6_extensions.tex:56` | "not just ... but also" | "its neighbors' characteristics as much as its own" / "on its neighbors' cost parameters as well as its own" / "carries both the time and the skill" |

---

## Grammar and punctuation

### G1. Pronoun and verb agreement
- p. 6, `1_introduction.tex:115`. "Which way improvements in AI move job boundaries is ambiguous, because **it presses** on both sides" → "because **they press** on both sides".
- p. 26, `5_longrun.tex:187`. "where those two prices balance **determine** the firm's degree of specialization" → "**determines**".
- p. SA-33, `SA_E:212`. "all three implications of our model **appears** to operate ... and **does** not appear" → "appear ... do not appear". The substance of this sentence also has to change; see REVIEW M1.
- p. OA-27, `OA_C:86`. "the required amount of skill-adjusted time ... **are** fixed" → "**is** fixed".
- p. OA-25, `OA_C:37`. "type of labor that **need** to perform them" → "the type of labor that performs them".

### G2. Missing articles, wrong prepositions (OA.C)
- p. OA-25, `OA_C:26`. "the more tasks firm includes in a job, the higher required compensation" → "the more tasks the firm includes in a job, the higher the required compensation".
- p. OA-26, `OA_C:62`. "Specifically, production function of the firm can be represented" → "Specifically, the firm's production function can be represented".
- p. OA-26 and OA-27, `OA_C:61, 91`. "by the help of AI" → "with the help of AI".
- p. OA-26, `OA_C:48`. "the fraction appearing behind t" → "the fraction multiplying t".

### G3. Rendering typos
- p. OA-24, `OA_C:4`. `Sections~\ref{sec:shortrun}--~\ref{sec:longrun}` prints "Sections 3– 5". Use `Sections~\ref{sec:shortrun}--\ref{sec:longrun}`.
- p. OA-29, `OA_C:135`. "for existence conditions.)." → "for existence conditions)."
- p. SA-30, `SA_E:145` (fn.). `\footnote{ To save space` has a leading space, and "two **position** away" should be "two **positions** away".
- pp. OA-14 and OA-16, `OA_B:336, 406`. Double spaces: "charging  the", "a  total".
- p. 4, Fig. 1 notes, `1_introduction.tex:68`. Literal Unicode "Steps 2–4" → `Steps~2--4`.

### G4. Sentence construction
- p. 2, `1_introduction.tex:9` (Overleaf edit). "A step is the primitive unit of work in our framework, what classic models call a 'task.'" The appositive clause doesn't attach. → "..., corresponding to what classic models call a 'task.'"
- p. 9, `3_shortrun.tex:19`. "each is executed in one of three modes: \emph{manually}, \emph{augmented}, or \emph{automated}" mixes an adverb with participles → "\emph{manual}, \emph{augmented}, or \emph{automated}".
- p. 11, `3_shortrun.tex:74` (fn.). "The forces described by the model remain unaffected even if we assumed ..." → "would remain unaffected if we assumed ...".
- p. 39, `7_empirics.tex:206`. "The last prediction of the model that we test is how the positioning of steps ... matters" → "The last prediction we test concerns how the positioning of steps ... matters".
- p. 23, `5_longrun.tex:56` (fn.). "namely that each step demand some increment" → "namely that each step demands some increment".
- p. SA-25, `SA_E:5`. "rarely-executed tasks, which a worker might perform only rarely" is tautological → "tasks that a worker performs only occasionally".
- p. SA-19, `SA_D:8`. "only change the sentence starting with ... with an alternative" → "replace only the sentence starting with ... with an alternative".
- p. 3, `1_introduction.tex:44`. "three modes of step completion are recognized" → "our model distinguishes three modes of step completion".

### G5. Punctuation
- p. 41, `7_empirics.tex:268`. "at −0.35, −0.26 and −0.34" needs the serial comma the rest of the draft uses: "−0.26, and −0.34".
- p. 37, `7_empirics.tex:132-134`. The footnote mark sits before the colon ("regression\footnote{...}:"). Move it after the colon, or after the displayed equation.
- p. 19, `4_implications.tex:175`. "(i.e., t^M_i ≥ 1) every component" needs a comma after the parenthesis.
- p. 19, `4_implications.tex:176` (fn.). `Equation~\ref{eq:fragmentation_closed_form}` prints "Equation 4". Everywhere else the draft prints "(4)"; use `\eqref`.

### G6. Headings
- p. 30, `6_extensions.tex:44`. "Computation of Firm's Short- and Long-run Optimums" → "Computing the Firm's Short- and Long-Run Optima". The paragraph heads "The Short Run Optimization." and "The Long Run Optimization." → "Short-Run Optimization." and "Long-Run Optimization.", matching "Short-Run Production" and "Long-Run Production".
- p. SA-25, SA.E title. "Robustness to Frequently-Executed Tasks Sample Restriction" → "Robustness to Restricting the Sample to Frequently Performed Tasks".
- p. SA-38, Fig. SA.F.1 caption. "Overlap between GPT-orderings and Original APQC-PCF Order" → "Overlap Between GPT Orderings and the Original PCF Order", with "Between" capitalized as in the other captions.
- p. SA-21, `SA_D:107`. "In the following Subsections" → "subsections".

---

## Clarity and consistency

### C1. Sample names were only half renamed
Section 7 now says "O*NET sample" and "PCF sample". The rest of the draft does not:
- **"main sample"** survives at `SA_A:1` (the appendix title, "Construction Details of the Main Sample", p. SA-1), `OA_A:201`, `SA_B:30`, `SA_D:118, 144`, and `SA_F:47, 352, 361, 396, 418, 449`. → "O*NET sample".
- **"APQC Process Groups"** is the Table 3 column header (p. 40), while Figure 9(b) and the table notes say "PCF". → "PCF Process Groups".
- **"corpus/corpora"** refers to the PCF 16 times in SA.F, plus `7_empirics.tex:35`. **"PCF dataset"** appears at `SA_F:394`. Use "PCF sample" for the analysis data, and keep "corpus" only in the SA.F.1 validation exercise if you want the distinction.
- `7_empirics.tex:39, 173` say "O*NET dataset" where the sample is meant.

### C2. "AI-able" is never defined
`7_empirics.tex:11` sends the reader to "the sense of Section 4" for "AI-able", but Section 4 never defines the term. It is first used at `1_introduction.tex:91` and `4_implications.tex:3`, and the closest definition is the footnote on p. 15 ("One can think of an AI-easy step as one that is 'exposed' to AI"). `7_empirics.tex:237` uses both terms in one sentence. Define "AI-able" once at `4_implications.tex:3` ("steps AI performs reliably, which we measure empirically as AI-exposed"). The Prediction #3 heading could then read "AI-exposed" to match the test.

### C3. Section 7 subsection titles are not parallel
- 7.1: "Tendency to Have Runs of Consecutive AI-executed **Tasks**" (noun phrase)
- 7.2: "AI Execution Is More Likely Next to AI-executed **Steps**" (clause)
- 7.3: "AI Execution Is Lower Where **AI-able** Steps Are Dispersed" (clause)

→ Change 7.1 to "Prediction #1: AI-Executed Steps Form Contiguous Runs". Also decide whether "executed" is capitalized after the hyphen in title case ("AI-executed" or "AI-Executed").

### C4. Capitalization of "Step" and "Task" before a numeral
- p. 4, `1_introduction.tex:76` says "the manual step~1, the AI chain spanning steps~2--4, ... step~4, and the manual step~5", while the same paragraph (`:72-75`) writes "Steps 1 and 5" and "Step 4".
- p. 10, Fig. 2 notes, `3_shortrun.tex:46`: "steps 4--6 become Task~4", "step~2".
- p. 20, `4_implications.tex:208`: "relabelled steps~1 and~2".
- p. 27 to 28, `5_longrun.tex:215-222` and `OA_A:106`: "tasks 1 and 2", "task 3", against "Tasks~1 and~2" at `5_longrun.tex:19`.

→ Capitalize throughout, since the body mostly does.

### C5. Spelling variants
The draft is American ("labeled", "modeling", "judgment"). The British forms are "labelled" (`4_implications.tex:84`), "relabelled" (`:208`), "grey" (`4_implications.tex:93`; `SA_E:82, 94`; `SA_F:387`), "judgement" (`SA_F:365`) and "honours" (a comment only). "Gray" already appears at `3_shortrun.tex:46`.

### C6. Hyphens after -ly adverbs
"frequently-executed" (SA.E title), "rarely-executed" (`SA_E:5`), "frequently-performed" (`SA_E:212`) and "rarely-performed" (`SA_E:208`) are all hyphenated, while the body has "more frequently performed tasks" (`7_empirics.tex:45`). Drop the hyphens. Also "highly frequently performed" (`SA_F:21`) → "frequently performed".

### C7. "Subsection" and "Section" for the same objects
The body writes `Section~\ref{sec:comparative_advantage}` for subsections, which prints "Section 4.1". But `7_empirics.tex:195` has "Subsection 7.1", and `SA_E` (3 times), `SA_F` (8 times), `SA_D` and `OA_C` also use "Subsection". Pick "Section".

### C8. Number style
`SA_D:7` says "10 alternative prompts" where Section 7 and SA.F say "ten". Tables SA.F.2 and SA.F.4 also print "10 alternative prompts".

### C9. Unclear sentences worth rewriting
- p. 4, `1_introduction.tex:84`. "compares the cost of manual execution to the unified AI-based execution cost". "Unified" is unclear → "compares the cost of manual execution with the cost of executing the step with AI as part of a chain".
- p. 3, `1_introduction.tex:51-52`. Step 3 is "building an analysis pipeline", then "running the analysis". Step 2 is "finding and fetching data", then "finding data". Use one wording per step.
- p. 12, `3_shortrun.tex:100`. "an AI chain (automated or augmented)" reads as if a chain were one or the other → "an AI chain (of length one or more)".
- p. 12, `3_shortrun.tex:127`. "rich dynamics" in a static problem → "rich patterns in how AI is deployed".
- p. 16, `4_implications.tex:101`. "The previous section examined" → "The previous subsection examined".
- p. 16, `4_implications.tex:141`. "A single verification now covers two steps instead of none" → "A single verification now covers two steps, whereas panel (a) used AI on none".
- p. 17, `4_implications.tex:149`. "having a measure of that arrangement would be useful, one that stands in for them" ("them" has no referent) → "a measure of that arrangement that can be computed without solving the firm's problem would be useful".
- p. 34, `7_empirics.tex:44`. "restores the correct workflow order to a substantial degree against ground truth" is redundant. → "recovers much of the known workflow order". Also see REVIEW Minor 3 on "ground truth".
- p. 34, `7_empirics.tex:53`. "we use the terms ... interchangeably and expect the reader to keep these distinctions in mind" contradicts itself → "we use the terms ... interchangeably, relying on the mapping above".
- p. 35, `7_empirics.tex:117`. "in one sitting between strong AI performers" → "in one sitting between AI-executed steps".

---

## Substantive issues noticed in passing
These are all in `REVIEW_issues_2026-09-26.md`:
- the unqualified Prediction #3 summaries (M1);
- the 151-word abstract (C1);
- SA.F's validation-only framing (D1);
- SA.E's misreading of its own heatmaps (D2, D3).
