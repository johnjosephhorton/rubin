# Prose review of "Chaining Tasks, Redefining Work" — 2026-09-08

Draft reviewed: all 18 `.tex` sources in `writeup/`, main text plus Online and
Supplementary Appendices. Page numbers from `0_main.pdf`, 133 pages, compiled
2026-09-08 from the current working tree (which includes uncommitted edits).

The draft is mechanically clean. No doubled words, no straight quotes in prose,
no `\%` spacing errors, no lowercase `figure~\ref`/`table~\ref`, no missing
commas after `e.g.`/`i.e.`, no comma splices found. The findings below are
mostly consistency rather than error, and the terminology item (C1) is the one
worth acting on regardless of the rest.

---

## House style violations

### S1. Em dash, two in one sentence
- **Location**: p. 6, "Long-Run Job Design" (`1_introduction.tex:116`)
- **Now**: "Absorbing steps into chains strips skill out of the tasks a worker retains, making broad jobs cheaper to sustain, while the same reduction---and any direct saving in coordination---makes each boundary cheaper to draw."
- **Fix**: "Absorbing steps into chains strips skill out of the tasks a worker retains, making broad jobs cheaper to sustain, while the same reduction, together with any direct saving in coordination, makes each boundary cheaper to draw."

This is the only em dash in the prose. The five other `---` hits are rules in
`preamble.tex` comments and are fine.

### S2. Explanatory colon, plus "crucial"
- **Location**: p. 4, "Automation versus Augmentation" (`1_introduction.tex:56`)
- **Now**: "Although augmented and automated production steps both involve AI, they differ in one crucial respect: AI augmentation demands that a human verify the AI's output, whereas automation does not."
- **Fix**: "Although augmented and automated production steps both involve AI, they differ in one respect that drives everything below. AI augmentation demands that a human verify the AI's output, whereas automation does not."

### S3. Explanatory colon
- **Location**: p. 35, Section 7.1 (`7_empirics.tex:70`)
- **Now**: "The two placebos isolate different margins: the first holds each occupation's task composition fixed and randomizes the ordering of its workflow, while the second holds each occupation's size and each major group's AI intensity fixed and randomizes which tasks, and hence which execution labels, an occupation contains."
- **Fix**: Replace the colon with a full stop and start "The first holds...".

### S4. "not just X, but Y" cadence, three instances
- `1_introduction.tex:89` (p. 5): "What tips a step into automation is therefore not just its own characteristics but also those of its neighbors." → "What tips a step into automation is therefore its neighbors' characteristics as much as its own."
- `4_implications.tex:59`: "depends not just on its own cost parameters, as comparative advantage logic would predict, but also on those of its neighbors" → "depends on its neighbors' cost parameters as well as its own, which comparative advantage logic would not predict".
- `6_extensions.tex:56`: "carries not just the time but also the skill of the worker" → "carries both the time and the skill of the worker".

### S5. "Moreover" and "Furthermore" pile-up
Seven instances, six of them in the appendices. Each can be deleted outright
without loss, since the sentence that follows already reads as an addition.
- `5_longrun.tex:5`; `OA_C_CES_representation.tex:68` (p. 75), `:106`, `:183`;
  `OA_B_omitted_proofs.tex:402`, `:406`, `:461`.
- `OA_B_omitted_proofs.tex:403` also opens "Crucially, the contribution from each $C$..." → "The contribution from each $C$...".

### S6. "leverage" as a verb
- **Location**: `SA_A_sample_construction.tex:18`
- **Now**: "require additional software or tools to fully leverage AI capabilities"
- **Fix**: "require additional software or tools to make full use of AI capabilities"

---

## Grammar and punctuation

### G1. Literal Unicode en dash in a figure note
- **Location**: p. 4, notes to Figure 1 (`1_introduction.tex:68`)
- **Now**: "while Steps 2–4 form an AI chain task"
- **Fix**: "while Steps~2--4 form an AI chain task"

The rest of the draft uses `--` throughout; this is the only raw Unicode dash.
It renders as a hyphen-width dash, visibly shorter than the `2--4` on the same page.

### G2. Comma before a compound predicate
- **Location**: p. 36, Section 7.1 (`7_empirics.tex:73`)
- **Now**: "We observe that the average AI chain length in the data is 1.45, and is noticeably larger than in both placebo distributions."
- **Fix**: "We observe that the average AI chain length in the data, 1.45, is noticeably larger than in both placebo distributions."

### G3. Passive where the agent matters
- **Location**: p. 3, "Automation versus Augmentation" (`1_introduction.tex:44`)
- **Now**: "In our model, three modes of step completion are recognized: manual, augmented, and automated."
- **Fix**: "Our model distinguishes three modes of step completion: manual, augmented, and automated."

The colon here introduces a genuine list and stays.

### G4. Doubled spaces mid-sentence
- `OA_B_omitted_proofs.tex:336` ("by charging  the realized") and `:406` ("adds up to a  total value").

---

## Clarity and consistency

### C1. Three names for one concept, none of them defined
This is the most consequential item. The draft calls the same object:

| Term | Count | First use |
|---|---|---|
| AI-exposed | 19 | Section 4 |
| AI-able | 24 | `4_implications.tex:106`, p. 13 |
| AI-suitable | 1 | `4_implications.tex:3`, p. 13 |

"AI-able" carries a subsection title ("Prediction \#3: Dispersion of AI-able
Steps Lowers AI Execution"), a table caption, and two figure subcaptions, yet no
sentence in the draft ever says what it means or that it is a synonym for
AI-exposed. A reader meets it first in a figure subcaption on p. 13.

"AI-suitable" appears exactly once, in the Section 4 roadmap that promises the
fragmentation index, and never again.

**Fix**: pick one term. "AI-exposed" is the one tied to the data, since the
labels are Eloundou et al.'s exposure categories, so it is the safest choice.
If you keep "AI-able" for the theory because exposure is a data construct, say
so once at first use, at `4_implications.tex:106`.

### C2. Corpus naming drift
The two datasets are referred to seven ways in the body: "O\*NET sample" (9),
"O\*NET dataset" (9), "O\*NET data" (1), "PCF corpus" (5), "PCF sample" (2),
"PCF dataset" (1), "APQC corpus" (1). Section 7.1 alone uses "PCF corpus" and
"the corpus" and "APQC's documented sequences" for the same thing.

**Fix**: settle on "the O\*NET sample" and "the PCF sample", and reserve "APQC"
for the organization rather than the data.

Related leftover: `7_empirics.tex:110` still says "the main sample's" where the
surrounding text now says "O\*NET". Six more "main sample" occurrences survive
in the appendices.

### C3. Step versus step before a numeral
The intro uses capital "Step" 19 times and lowercase 4 times, and one paragraph
mixes both:
- p. 4 (`1_introduction.tex:72-75`): "Steps 1 and 5 are performed manually...", "Step 4 is augmented..."
- p. 4 (`1_introduction.tex:76`), immediately after: "the manual step~1, the AI chain spanning steps~2--4, whose only human input is verifying the output of step~4"

**Fix**: capitalize in line 76 to match its own paragraph. Same fix at
`3_shortrun.tex:46` (figure notes) and `4_implications.tex:211` (p. 20,
"relabelled steps~1 and~2").

`4_implications.tex:211` also has British "relabelled"; the draft is otherwise
American throughout.

### C4. fixed-effect versus fixed-effects, attributive
- `SA_F_external_validation.tex:369` and `:386`: "the three fixed-effect specifications"
- `SA_E_frequency_robustness.tex:143` (p. 115): "fixed-effects specification"

**Fix**: use the singular attributive, "fixed-effect specification", in both places.

---

## Substantive issues noticed in passing

Not a prose pass's job; run `paper-review` for a real correctness check.

1. **`SA_F_external_validation.tex`, Prediction \#1 paragraph**: reports
   `z = 10.2` for the within-category reassignment null. Re-running
   `analysis/apqc_pooled_predictions.py` at the 0.71 floor gives 9.91, and at
   0.73 gives 9.73. The companion `z = 6.2` does reproduce, from the
   threshold-sweep script. The verdict is unaffected, since both are at the
   100th percentile.

2. **`7_empirics.tex`, Section 7.1, PCF paragraph**: "What the corpus can still
   ask is whether the runs that do form are longer than the same labels arranged
   at random along the same sequences, and they are." The sentence that carried
   the supporting numbers was removed, so "and they are" now rests only on Panel
   (b) and the appendix.

3. **`1_introduction.tex`, Prediction \#3**: "workflows whose AI-exposed steps
   are more dispersed across the production sequence execute a lower share of
   their steps with AI" is stated without qualification, while Table 3 shows the
   coefficient significant in the PCF and not distinguishable from zero in
   O\*NET.

4. **`7_empirics.tex`, Section 7.3**: the paragraph explaining why a
   workflow-level test leans on the ordering harder than a step-level one has
   been removed, but the later passage explaining the O\*NET/PCF contrast still
   assumes the reader has that argument.
