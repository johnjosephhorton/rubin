# Review of "Chaining Tasks, Redefining Work: A Theory of AI Automation" — 2026-09-04

Draft reviewed: writeup/*.tex (main text + Online Appendix + Supplementary Appendix), compiled PDF of 3 September 2026 (134 pages).

## Scope and method

Three independent passes were run over the whole draft and then consolidated here.

1. **Per-file deep read** of all 18 .tex sources (main text, OA.A-OA.C, SA.A-SA.F), each read against the compiled PDF, the generating notebooks in `analysis/`, the table sources in `writeup/tables/`, and the rendered figures in `writeup/plots/`.
2. **Twelve whole-draft cross-cutting lenses**, each tracking one object across all 134 pages (the chain indices $\ell$ and $r$; $\alpha$, $d_i$ and $q_i$; the E1/E2 exposure labels; FI versus EFI; the EFI definition numbering; the Prediction #3 hedging; partitions versus jobs; sample units and Ns; prompt numbering; terminology and thresholds; the CES claims; time symbols).
3. **Number verification**, in which every numeric claim in the prose was traced back to the CSV, table file, figure PNG, or notebook that produces it, and recomputed from the source data where the source permitted it.

**Coverage of the number check**: 287 numeric claims traced to source, 252 verified correct, 0 unlocatable. Every drafted number was matched to a source object; nothing had to be reported as "cannot be checked".

**Verification standard.** Every finding below survived an adversarial refute pass whose explicit instruction was to default to rejection when uncertain. Roughly 40 candidate findings were killed that way, including several that looked like real contradictions until the underlying file or figure was opened. Findings that survived carry a re-located quote and, wherever a number is involved, an independent recomputation from the source rather than a reading of the drafted text.

**Mechanical cross-reference check.** 175 `\label` definitions, 0 dangling `\ref`, 0 labels defined in more than one file. Missing-cross-reference errors are therefore ruled out. Every cross-reference finding below is about a pointer aimed at the wrong object or at a non-`\ref` string, never a missing target.

**What was NOT covered.** A dedicated line-by-line re-derivation pass over the OA.B proofs was planned as a separate module and **did not run** (the session hit a usage limit). The OA.B and OA.C proofs were read and partly re-derived by the per-file pass, including a hand-verification of the full OA.C.3 algebra chain from (OA.C.20) through (OA.C.22) to (OA.C.16), and the OA.B and OA.C findings below (D3, D10, D18, D19, D20 and N24 to N31) come from that reading. But they did not get the dedicated adversarial proof audit the other material got, so **the review is thinner on the proofs of Propositions 1-4 and Lemmas OA.B.1-OA.B.4 than anywhere else in the draft.** Treat the absence of further OA.B findings as unproven, not as a clean bill of health.

Also not covered: bibliography and citation accuracy; the substantive economics of the literature positioning; anything about grammar, style, or wording, which is excluded by design (findings whose only defect was phrasing were dropped).

Two lenses (`sample-units`, `prompt-numbering`) completed their refute passes late in the session. Both did produce verified output, so their findings are reported in the body with the rest rather than as unverified leads. The one item nothing closed is listed in the appendix.

**Independence from prior reviews.** Three prior review files in `writeup/` (`REVIEW_issues.md`, `paper_review_findings.md`, `paper_review_language.md`) were excluded from this review at the author's instruction. None was read by any pass. This report is therefore independent of them, and overlap with them, where it exists, is corroboration rather than repetition.

## Summary

| Severity | Count |
|---|---|
| Major | 5 |
| Medium | 46 |
| Minor | 56 |
| **Total entries** | **107** |

107 entries were consolidated from 165 raw findings; 58 were duplicates of an entry already listed, merged into it. Where two passes disagreed on severity, the disagreement is recorded at the end of this section.

| Category | Entries |
|---|---|
| Number / text-vs-exhibit mismatch | 38 |
| Illogical or unsupported argument | 27 |
| Notation and definition collisions | 19 |
| Cross-section inconsistency | 14 |
| Proof and math errors | 9 |

**The five issues that matter most.**

1. **Prediction #3 is a null on the paper's main sample and is stated as a finding in the abstract, the introduction and the conclusion** (M1). O*NET occupations, Table 3 columns (1)-(3): -0.01 (0.10), -0.09 (0.09), -0.04 (0.09), n = 872, no stars. The significant estimates -0.35\*\*\* (0.10), -0.26\*\* (0.11), -0.34\*\*\* (0.11) are 525 APQC PCF process groups. Found independently by five passes.
2. **AI exposure is declared to be E1 or E2 and implemented as E1-only in four places** (M2). SA.A reports 605 (69%) of occupations containing an AI-exposed task where the declared rule gives 809 (93%); Table 2's exposure control codes 3,109 of the 4,632 E1|E2 tasks as unexposed; Table SA.B.5's exposure row is 0.11/0.07/0.07 against 0.49/0.48/0.39 in Table 3 on the identical 872 occupations.
3. **The execution-based EFI "falsification check" is reproduced by an independence null** (M4). Observed -0.780; within-occupation reshuffle of the execution labels, which makes position independent of execution by construction, gives -0.755 (sd 0.008 over 50 draws). The two fixed-effects columns give z = -1.7 and -1.1 against their nulls, i.e. no separation at all.
4. **The Anthropic attrition step is wrong and mixes two units** (M3). Drafted 1,017 fully filtered / 2,347 remaining; the source file gives 1,066 / 2,298, and 2,347 is a count of merged occupation-task rows, not of Anthropic tasks.
5. **SA.D's "no evidence of systematically different orderings" is contradicted by its own figure** (M5). The printed panel means are 0.64 vs 0.56 (AI exposure), 0.62 vs 0.58 (AI execution), 0.56 vs 0.63 (EFI), with Welch t of -6.90, -3.37 and +4.96. The occupations that drive Predictions #2 and #3 are the ones whose GPT orderings are least stable.

**Severity reconciliations** (where two passes disagreed).

- **SA.F exact-recovery claim (D42).** Rated major by the per-file pass, downgraded to medium by the number-verification pass, which recomputed it from `ordering_accuracy.csv` and found that the error **understates** the instrument (the draft says no branch of five or more steps is recovered exactly; 21 of 208 are). **Downgrade taken**: the error does not flatter the paper's own result.
- **Anthropic attrition counts (M3).** Medium in the per-file pass, major in the number pass, which read the source CSV directly. **Major taken.**
- **The 605 (69%) exposure statistic (folded into M2).** Medium in the per-file pass, major in the number pass, which recomputed both variants (605 E1-only, 809 E1|E2). **Major taken.**
- **Figure SA.D.1 Panel (b) staleness (D32).** Minor in the per-file pass, medium in the number pass, which re-ran the notebook's own split rule on the current data (433/435, not 328/540) and established the data revision from file timestamps. **Medium taken.**
- **Adjacent-versus-two-away in Panel (d) (D33).** Medium in the per-file pass, minor in the number pass. Both re-derived from the same AME CSVs. **Medium kept**: the per-file pass documents that min(adjacent) < max(two-away) for 6 of 11 prompts and that the figure notes repeat the unqualified claim, which the number pass's charitable reading of one sentence does not dispose of.
- **The worked frequency-cut example (N44).** Medium in the per-file pass, minor in the number pass. Neither rests on data. **Minor taken**: the defect is an unstated stipulation in an illustrative example, with no estimate downstream.
- **Proposition 2 as the warrant for Prediction #3 (D4).** Minor in the per-file pass, medium in the cross-cutting lens, which inverted the sandwich and showed the ranking needs FI_A/FI_B > 5. **Medium taken.**
- **Prediction #3 hedging (M1).** Medium in the abstract-level finding, major in four other passes. **Major taken.**

## Status of fixes

Entries addressed so far are marked ✅ in their heading and carry a **Status** line giving what changed.
Everything not listed here is still open.

**Addressed (14 of 107):**

| Entry | Landed in |
|---|---|
| M3 | `01a9ae1` |
| D25 | `228864c` |
| M4 | `1d64cb0` (removed SA.B.2) |
| D28 | working tree |
| N7 | `f9d4a4a` |
| N24 | `f9d4a4a` |
| N25 | `f9d4a4a` |
| N26 | `f9d4a4a` |
| N27 | `f9d4a4a` |
| N47 | `f9d4a4a` |
| N49 | `f9d4a4a` |
| N51 | `f9d4a4a` |
| N53 | `f9d4a4a` |
| N55 | `f9d4a4a` |

_Last updated 2026-09-04._

## Major

### M1. Prediction #3 is null on the main O*NET sample and is asserted as a finding in the abstract, introduction, section heading, conclusion and two appendix summaries
- **Location**: p. 1 (PDF 1) abstract, `0_main.tex:76`; p. 5 (PDF 5) `1_introduction.tex:94`; p. 6 (PDF 6) `:129`; p. 7 (PDF 7) `:135` and `:141`; p. 39 (PDF 39) subsection heading `7_empirics.tex:186`; p. 42 (PDF 42) `7_empirics.tex:281-282`; p. 44 (PDF 44) `8_conclusion.tex:17-18`; p. SA-24 (PDF 108) `SA_B_alternative_definitions.tex:264`; p. SA-35 (PDF 119) `SA_E_frequency_robustness.tex:212`.
- **Issue**: In Table 3 (`tables/fragmentation_index_regression_combined.tex`) the Empirical Fragmentation Index coefficient on the main O*NET sample, columns (1)-(3), is **-0.01 (0.10), -0.09 (0.09), -0.04 (0.09)** over **872 occupations**, with no significance stars (|t| = 0.10, 1.00, 0.44). The significant estimates **-0.35\*\*\* (0.10), -0.26\*\* (0.11), -0.34\*\*\* (0.11)** are columns (4)-(6), **525 APQC PCF process groups**, a different corpus whose exposure and execution labels are transferred from O*NET by embedding match. Subsection 7.3 says so plainly ("the point estimates are small and none is statistically distinguishable from zero. On O*NET occupations we do not detect the workflow-level channel"), and that one sentence is the only acknowledgment of a null anywhere in `0_main.tex`, `1_introduction.tex`, `7_empirics.tex` and `8_conclusion.tex`. Two of the outside statements are explicitly about the wrong unit: the introduction says "occupations whose AI-exposed steps are more dispersed ... are associated with a lower share of their steps executed by AI" (p. 7), and the contribution paragraph says "jobs" (p. 5). The introduction's robustness sentence and SA.E's closing sentence go further and frame the PCF estimate as a re-confirmation ("continue to hold", "all three implications operate"), when it is the only place the result exists and SA.E's own body calls it "the null of Subsection 7.3".
- **Why it's a problem**: A reader who does not reach Subsection 7.3 comes away believing the workflow-level prediction is confirmed on the paper's 872-occupation main sample. It is not. The evidentiary structure is inverted at every point where the paper summarizes itself.
- **Proposed fix**: Adopt one attribution rule and apply it at all nine locations. Name the corpus whenever the fragmentation result is asserted, and never assert it of occupations or jobs. Abstract: "... and (3) in a corpus of practitioner-documented process sequences, dispersion of AI-exposed steps predicts lower AI execution at the workflow level." Introduction p. 5 and p. 7: replace "jobs"/"occupations" with the PCF corpus and add that the O*NET estimate carries the predicted sign but is not distinguishable from zero. Conclusion: split the clause and name the corpus. Introduction p. 7 and SA.E:212: replace "continue to hold" and "all three implications operate" with the split verb (the chaining result is reproduced on PCF; the fragmentation channel is detected there and not on O*NET).
- **Corroboration**: found independently by five passes (per-file abstract, per-file introduction, per-file conclusion, the Prediction #3 hedging lens, and the front-matter number check). Ten raw findings merged.

### M2. AI exposure is declared to be E1 or E2 and implemented as E1-only in four separate places
- **Location**: declared at p. 34 (PDF 34) `7_empirics.tex:25` and p. SA-1 (PDF 85) `SA_A_sample_construction.tex:16`. Implemented E1-only at: p. 37-38 (PDF 37-38) Table 2's exposure control, `7_empirics.tex:143` and note at `:162`, from `analysis/onet_antrhopicIndex_execTypeVaryingDWA.ipynb` cell 13; p. SA-2 (PDF 86) `SA_A_sample_construction.tex:34` and Figure SA.A.2 panel (a); p. SA-17 (PDF 101) Table SA.B.5, `tables/fragmentation_index_regression_execution.tex`, claim at `SA_B_alternative_definitions.tex:284`; p. SA-23 (PDF 107) Figure SA.D.1 panel (c), `SA_D_prompt_robustness.tex:68`.
- **Issue**: The declared rule is "we treat tasks with a human-assigned E1 or E2 label as exposed to AI". Table 3 honours it (its note says "the share of steps carrying an E1 or E2 label"). Four places do not.
  - **SA.A statistic**: drafted **605 (69%)** of the 872 occupations contain at least one AI-exposed task. Recomputed from `ONET_taskPosition_AImeasures.csv`: E1-only gives 605 (69.4%), **E1|E2 gives 809 (92.8%)**. A 204-occupation, 23-percentage-point gap. The AI-execution half of the same sentence, 555 (63.6%), is correct.
  - **Table 2's control**: the notebook sets `is_exposed = human_labels.isin(['E1'])`. In the saved 10,708-row estimating sample this codes all 1,523 E1 tasks as exposed and all 3,109 E2 tasks as unexposed, so **3,109 of the 4,632 E1|E2 tasks (67.1%)** sit in the "unexposed" bin of a control the table note describes as "the AI exposure status of task k". The sibling notebook `analysis/onet_neighborAI_E1E2exposureControls.ipynb` annotates the same line "# paper control: E1 only".
  - **Table SA.B.5**: estimated with `build(2, "human_E1_fraction")`; the replicated output is byte-identical to the published table. Its exposure row is **0.11, 0.07, 0.07** against **0.49, 0.48, 0.39** in Table 3 on the identical 872 occupations. The claim at `:284` that "the specification mirrors the main-text regression exactly ... the only difference is that fragmentation is measured over realized AI execution" is therefore false; there are two differences. A matched version already exists at `data/computed_objects/efi_matched_exposure/fragmentation_index_regression_execution_MATCHED.tex` (exposure 0.21\*\*\*/0.22\*\*\*/0.17\*\*\*).
  - **Figure SA.D.1 panel (c)**: splits on `human_E1_fraction`, while `human_aiExposure_fraction` (E1+E2) is computed in the same notebook and left unplotted, so the panel does not split on the exposure dimension it says is "discussed in Subsection 7.3".
- **Why it's a problem**: The paper's central primitive is defined one way and used another way in its main Prediction #2 table, in the appendix that documents the sample, in the one table of the execution-based index, and in a robustness split. Substantively, Proposition 1 predicts a neighbor effect *over and above* a step's own AI-ability, and an exposure control that misses two thirds of the exposed tasks does not hold that fixed. For SA.B.5 the matching of exposure to the index's label set is exactly what nets out the level term -k/m, so the E1-only regressor leaves the level in.
- **Proposed fix**: Re-estimate on E1|E2 wherever the declared rule is meant to apply (the matched SA.B.5 table already exists; `onet_neighborAI_E1E2exposureControls.ipynb` already builds the Table 2 control), and update the SA.A statistic to 809 (93%) with Figure SA.A.2 panel (a) regenerated. Where the narrow cut is deliberate, invoke SA.A's "unless explicitly mentioned otherwise" clause explicitly and say why, in the table note as well as the text.
- **Corroboration**: found independently by five passes (per-file SA.A, per-file SA.D, the exposure-label lens, the EFI-definition lens, and the empirics and SA.A number checks). Seven raw findings merged.

### ✅ M3. The Anthropic attrition step is wrong by 49 tasks and states two different units as one
- **Status**: ✅ **Addressed** 2026-09-04 (commit `01a9ae1`). SA.A now states 1,066 fully filtered / 2,298 labeled, and derives the 2,347 occupation-task records through the 1,830 matched task statements. A footnote was added for the 23 statements whose augmenting and automating shares tie and break toward automation (28 of the 721 automation records).
- **Location**: p. SA-1 (PDF 85), `SA_A_sample_construction.tex:28-29`.
- **Issue**: Drafted: 3,364 tasks, of which **1,017** are fully filtered, leaving **2,347**. Source `data/Anthropic_EconomicIndex/automation_vs_augmentation_by_task.csv` has 3,364 rows (matching), of which **1,066** have `filtered == 1`, leaving **2,298**. The construction notebook prints this verbatim ("Number of rows after filtering: 2,298", with Automation 566 + Augmentation 1,732 = 2,298). The drafted 2,347 is a different object, the number of *occupation-task rows* in the merged O*NET panel carrying a non-Manual label (1,626 Augmentation + 721 Automation), spanning only 1,830 distinct task titles; 1,017 is then back-derived as 3,364 - 2,347.
- **Why it's a problem**: Neither count is reproducible from the cited source, the sentence understates by 49 the number of tasks Anthropic could not classify, and the next sentence ("we assign each of these 2,347 tasks an AI execution label") assigns labels in the wrong unit.
- **Proposed fix**: "In total, conversations are linked to 3,364 tasks, of which 1,066 have all of their conversations filtered, leaving 2,298 with at least one non-filtered conversation. Of these, 1,830 match an O*NET task title, yielding 2,347 labeled occupation-task observations (1,626 augmentation, 721 automation) in the final 17,925-row sample."
- **Corroboration**: found by two passes; severity reconciled upward to major on the number pass's direct read of the source file.

### ✅ M4. The execution-based EFI "falsification check" is reproduced almost exactly by an independence null, so the stated inference does not follow
- **Status**: ✅ **Addressed** 2026-09-04 (working tree). Subsection SA.B.2 was **removed** rather than repaired, together with its table and the falsification argument. The check cannot do the job it was included for: since EFI = 1 - k/m + r/m and the execution-based variant sets k to AI-executed steps, the regressor contains the dependent variable by identity (verified exactly on the 872-occupation panel), and a within-occupation reshuffle that destroys clustering by construction still reproduces 96% of the estimate. The AI-exposed count control cannot net out a level term that is the execution share, so the note at `:280` went with it. Nothing referenced the subsection. The section title and the SA.B roadmap were rescoped to Prediction #2 alone.
- **Location**: p. SA-18 (PDF 102), `SA_B_alternative_definitions.tex:292-294`; table note at `:280`; exhibit `tables/fragmentation_index_regression_execution.tex`.
- **Issue**: The subsection argues that a strong negative EFI coefficient could not arise if AI adoption were independent of position, and reads the observed **-0.78 / -0.70 / -0.68** as evidence of clustered, chain-based adoption. Two things break it.
  - By Equation (13), EFI_w = 1 - k_w/m_w + r_w/m_w, and for the execution-based index k_w counts AI-**executed** steps, so k_w/m_w *is* the dependent variable (the identity reproduces to 1.1e-16 on the 872-occupation panel). Under independent placement of k executed steps among m, E[EFI | k] = 1 - k(k-1)/m^2, strictly decreasing in the execution share with zero clustering, so the stated null is false as stated.
  - Re-estimating the identical specification after reshuffling each occupation's execution labels uniformly at random within the occupation, which makes position independent of execution by construction and holds m and k exactly fixed, gives a standardized EFI coefficient of **-0.755 (sd 0.008 over 50 draws, range -0.771 to -0.738)**, i.e. the independence null reproduces **97%** of the published -0.780. The two fixed-effects columns are worse: null means **-0.678** and **-0.659** against observed **-0.703** and **-0.679**, i.e. **z = -1.7 and -1.1**, not distinguishable from the null.
  - Relatedly, the note at `:280` carries over the main table's justification that controlling for the AI-exposed step count "nets out the level term". For the execution-based index the level term is the *execution* share, not the exposure share, so the count control does not net it out and the coefficient is not identified off arrangement.
- **Why it's a problem**: These two sentences carry the entire interpretive payoff of Subsection SA.B.2 ("The strong negative relationship ... therefore indicates that AI adoption follows the clustered, chain-based pattern implied by the model"). No benchmark is reported against which -0.78 could be judged strong, and when one is computed the coefficient is not strong.
- **Proposed fix**: Replace the verbal null with the within-occupation reshuffle the paper already runs in Subsections 7.1-7.2. Report the observed -0.78 against the null mean of about -0.76, state that only the residual gap is informative about clustering, and note that the gap is significant only in column (1). Correct the note at `:280` so it does not claim a netting-out property this specification lacks. If the reshuffle gap will not carry the claim, delete the falsification reading.
- **Corroboration**: found independently by three passes (per-file SA.B logic, per-file SA.B note, and the SA.B number check, which re-implemented the specification from raw data).

### M5. SA.D's "no evidence of systematically different orderings" is contradicted by the printed means in its own figure, and the gap runs against the paper
- **Location**: p. SA-21 to SA-23 (PDF 105-107), `SA_D_prompt_robustness.tex:5, 67, 69`; Figure SA.D.1 panels (b)-(d).
- **Issue**: The text says the split means are "very similar" and that there is "no evidence that GPT generates systematically different task orderings across different types of occupations". The figure legends print: **AI exposure split 0.64 (below median) vs 0.56 (above)**; **AI execution split 0.62 vs 0.58**; **EFI split 0.56 vs 0.63**. Recomputing from `GPT_task_sequences_kendall_results.csv` merged to `occupation_analysis_with_fragmentationIndex_def1.csv` (N = 868, mean tau 0.6018, sd 0.1701) reproduces those means exactly and gives Welch t = **-6.90 (p = 1.0e-11)** for E1 exposure, **-3.37 (p = 7.8e-04)** for AI execution and **+4.96 (p = 8.6e-07)** for the EFI, with Cohen's d = -0.47, -0.23, +0.34. Splitting on the E1|E2 measure gives the same verdict (0.635 vs 0.570, t = -5.75). Spearman rho of mean tau with the E1 share is -0.236 (p = 2.1e-12).
- **Why it's a problem**: This paragraph, plus the appendix's opening claim, is the only evidence offered that prompt-induced ordering noise is not differential across the occupation dimensions used in the tests. The sign is the one that matters: the occupations that are **more** AI-exposed and **more** AI-executed, exactly those driving the Prediction #2 and #3 estimates, are the ones whose GPT orderings are **least** stable across prompts.
- **Proposed fix**: State the numbers and the direction, then make the argument that the size does not overturn the results. For example: "Mean Kendall's tau is 0.06 to 0.08 lower in the above-median AI-exposure and AI-execution groups and about 0.06 higher in the above-median EFI group, differences of roughly a quarter to a half of the cross-occupation standard deviation of 0.17." Then argue from the size, not from the absence of a difference.
- **Corroboration**: found independently by two passes, both of which recomputed the split means from the source CSV rather than reading the panel.

## Medium

### D1. The introduction bills the CES aggregation as a micro-foundation for AI's aggregate labor-market and productivity effects, broader than what OA.C derives and caveats
- **Location**: p. 6 (PDF 6) `1_introduction.tex:121`; p. OA-32 (PDF 80) `OA_C_CES_representation.tex:254`; caveat at `OA_C_CES_representation.tex:244-247`.
- **Issue**: OA.C's own caveat restricts the object: because tau_M is common to all firms, (OA.C.16) gives M = tau_M Y at every wage vector while K is normalized to 1, so the economy "traces a one-dimensional locus along which M/Y and K are both fixed", and (OA.C.9) "should accordingly be read as a representation valid along that locus rather than as an identification of the economy's technology over arbitrary (A, M, K)". It "says nothing testable about the manual-input-per-output or capital margins". Those are two of the three margins an aggregate labor-demand or productivity exercise needs.
- **Why it's a problem**: The introduction and OA.C's own closing sentence promise strictly more than the appendix concedes it can support, and the caveat is on the same page as the overclaim.
- **Proposed fix**: Restrict both sentences to the margin the representation covers, matching the Section 6.1 wording (substitution between AI management labor and the rest of production), rather than "the labor market and total productivity".

### D2. The short-run setup posits several workers with fixed job boundaries; the rest of the paper treats the short run as one worker holding the whole workflow
- **Location**: p. 9 (PDF 9) `3_shortrun.tex:9-14`, against `3_shortrun.tex:114` (p. 12), the note to Table OA.A.1, `5_longrun.tex:1` and `6_extensions.tex` Section 6.2.
- **Issue**: The setup describes several workers each holding a fixed block of steps. Section 3.3 says "The worker carries out the entire step sequence ... and is paid at the normalized rate of w = 1"; Table OA.A.1's note says "With the whole workflow assigned to one worker at a fixed wage"; Section 5 opens "we held the set of steps assigned to the worker fixed" (singular). Problem (1) correspondingly minimizes over **all** contiguous partitions of S, with no constraint that a block lie inside one worker's block.
- **Why it's a problem**: Under the multi-worker reading, Problem (1) is not the firm's feasible problem, since it admits AI chains straddling a job boundary, which the long-run formulation forbids (Definition 6 makes a job a partition of the task sequence).
- **Proposed fix**: State in Section 3 that the whole workflow is held by a single worker at a fixed wage, as three other places already do, and drop "each worker's job covers a fixed block of steps". If several workers are intended, add the refinement constraint to Definition 5 and Problem (1).

### D3. The domains of alpha and d_i are never specified, and three downstream results need them
- **Location**: p. 10 (PDF 10) `3_shortrun.tex:22, 35`; Table OA.A.1 row at `OA_A_tables_and_figures.tex:34` (p. OA-1); `OA_B_omitted_proofs.tex:441, 446` (p. OA-17).
- **Issue**: Neither alpha nor d_i is given a domain in the main text. Table OA.A.1 supplies alpha in (0,1] but gives d_i no range at all. Three consequences. (i) q_i = alpha^{d_i} lies in (0,1] only if d_i >= 0, and Definition 2's asserted monotonicity fails at alpha = 1 (which the table permits) and at d_i = 0 (where q_i is identically 1). (ii) OA.B asserts D_c = sum d_i >= 1 for every chain as if it followed from the setup; it does not, and the appendix's own Examples `ex:FI.tight` and `ex:FI.necessity` use steps with q_i = 1, while OA.C writes d_b = 0 for a manual step. (iii) OA.B calls the cost curve "a polynomial in 1/alpha" whose chains contribute "a term of degree equal to its total difficulty", which requires the d_i to be non-negative integers. The strings "integer", `\mathbb{N}` and `\mathbb{Z}` appear in none of the 18 .tex files, though every d value used in the draft is an integer (d = (11,1) in Example 2, d = (2,1,2,1) in the transversality footnote).
- **Why it's a problem**: Downstream results assume more than Section 3 delivers, in three different appendices.
- **Proposed fix**: State the domains where q_i is introduced (alpha in (0,1), d_i > 0, or d_i >= 1 if the degree bound is to be inherited rather than assumed), add the d_i range to Table OA.A.1, reconcile (0,1] with the (0,1) used in the proofs, and either declare integrality or replace "polynomial"/"degree" with the "finite sum of powers" language the proof itself already uses at `OA_B:483`.

### D4. Proposition 2's constant-factor sandwich is too loose to rank arrangements, yet two passages cite it as the warrant for the arrangement comparative static
- **Location**: p. 13 (PDF 13) `4_implications.tex:3`; p. 39 (PDF 39) `7_empirics.tex:191`. Proposition statement at `4_implications.tex:163-169`.
- **Issue**: Proposition 2 gives only (1/8)OPT <= FI <= (5/4)OPT, tightened to (1/4)OPT <= FI <= (5/4)OPT when t^M_i >= 1. Inverted, OPT lies in [(4/5)FI, 4FI], so concluding OPT_A > OPT_B from FI_A > FI_B requires **FI_A/FI_B > 5**. Across re-orderings of a fixed step set the index cannot move nearly that much, so the sandwich orders no pair of arrangements. Proposition 2 is also a statement about **cost**, whereas the regression's outcome is the share of steps executed by AI. The monotone content the test actually rests on is the order-invariance decomposition in eq. (3) (FI decreasing in sum q_{i-1} q_i across re-orderings) plus Example 1.
- **Why it's a problem**: "This is the empirical content of Proposition 2" sends a reader checking the theoretical warrant for beta_2 < 0 to the one result in Subsection 4.2 that cannot supply it.
- **Proposed fix**: Attribute the arrangement result to Example 1 and eq. (3), and reserve Proposition 2 for the separate level claim that the index proxies the firm's optimal cost. Drop the "so that" in `4_implications.tex:3`.

### D5. The set-monotonicity clause after Figure 3 overreaches Proposition 1(ii), and is false inside the proposition's own three-step block
- **Location**: p. 16 (PDF 16), `4_implications.tex:96`.
- **Issue**: Proposition 1(ii) establishes upward closure only for the focal, human-advantaged step k. The "so" clause draws the stronger conclusion that the whole set of AI-executed steps is monotone in AI reliability. Counterexample using the appendix's own pricing (V_0 all singletons; V_1 = m\*\_{k-1} + t^A_{k+1}/(q_k q_{k+1}); V_2 = t^A_{k+1}/(q_{k-1} q_k q_{k+1}); V_3 = t^A_k/(q_{k-1} q_k) + m\*\_{k+1}): take (t^M, t^A, q)\_{k-1} = (2, 100, 0.5), (t^M, t^A, q)\_k = (1, 1, 0.8) (human-advantaged, 1 < 1.25), t^M_{k+1} = 10, t^A_{k+1} = 1. At q_{k+1} = 0.45, V3 = 4.7222 < V1 = 4.7778 < V0 = 5.2222 < V2 = 5.5556, so AI executes all three steps. At q_{k+1} = 0.55, V1 = 4.2727 < V3 = 4.3182, so improving q_{k+1} **removes** step k-1 from the AI-executed set.
- **Why it's a problem**: It converts a proved statement about one step into a false comparative static about the whole set, stated as the general lesson of part (ii), which is exactly the claim a reader carries into the empirical section on neighbor effects.
- **Proposed fix**: Restrict the clause to the focal step ("a step AI already executes is never returned to the human when its own or its neighbors' reliability improves") and drop the claim that improvements "only ever add steps to those AI executes".

### D6. "Returns jump at each reorganization threshold as longer AI chains become worth deploying" is not what Lemma OA.B.4 proves, and fails at the first threshold of the paper's own Example 2
- **Location**: p. 20 (PDF 20) `4_implications.tex:206`; same gloss at `0_main.tex:75`, `1_introduction.tex:106`, `2_literature.tex:9`, `7_empirics.tex:68`. Formal result at `OA_B_omitted_proofs.tex:457-507`; contradicting exhibit Table OA.A.3.
- **Issue**: Thresholds are defined one sentence earlier as "the values of alpha at which the optimal strategy changes", so "at each of them" is universal. Lemma OA.B.4 derives the jump from one fact only, that at a crossing the cost difference changes sign, hence the newly optimal strategy is more alpha-elastic (larger sum of t^A_r D_c). Neither the statement nor the proof mentions chain length. Example 2 realizes the counterexample: by Table OA.A.3 the threshold at alpha = 0.50 switches the optimum from "Both steps manual" to "Step 2 augmented", a single length-one chain with D_c = 1. No chain lengthens.
- **Why it's a problem**: This is the mechanism sentence for the paper's third headline implication and its J-curve micro-foundation, offered as what the formal result establishes.
- **Proposed fix**: State the jump in the terms the lemma proves (the newly optimal strategy is more sensitive to alpha, so the marginal return jumps), add that this typically happens because AI takes over more of the workflow but need not involve a longer chain, and cite the alpha = 0.50 threshold of Example 2 as the exception.

### D7. In eq. (8) the AI strategy ranges over P(S), the set of unlabeled contiguous partitions, but Definition 5 makes a strategy a partition plus a mode label on every singleton block
- **Location**: p. 26 (PDF 26), `5_longrun.tex:141` (definition of P(X)) and the display at `:143-149`; Definition 5 at `3_shortrun.tex:98-102`.
- **Issue**: P(X) is defined as the set of contiguous partitions of a sequence. Only the inner minimization over jobs is exact, since a job design really is nothing but a contiguous partition. An AI strategy is a partition **plus** a mode label on every singleton block, because a singleton admits both labels and they cost differently (c^M_i, t^M_i versus c^A_i, t^A_i/q_i). As written the outer minimand is not a function of the object it ranges over.
- **Why it's a problem**: The long-run objective multiplies a job's total skill by its total time, so the manual-versus-augmented label on a singleton cannot be resolved outside the minimization the way it can in the short run, where minimizing time alone selects min{t^M_i, t^A_i/q_i} step by step.
- **Proposed fix**: Introduce a separate symbol for the strategy set, e.g. A(S) = {(P, l) : P a contiguous partition of S, l a mode in {M, A} for each singleton block, every block of size >= 2 an AI chain}, and write min over T in A(S), reserving P(.) for the job design.
- **Corroboration**: found independently by two passes.

### D8. The body presents (9) as a derived three-input CES; OA.C normalizes K = 1 and posits the capital term
- **Location**: p. 30 (PDF 30) `6_extensions.tex:21-28` and footnote 22; `OA_C_CES_representation.tex:162, 244-247`.
- **Issue**: Equation (9) is introduced as what the firm-level technology "can be aggregated to" and glossed as a function over economy-wide A, M and capital K. OA.C states that the capital term is not derived: "Aggregate capital is normalized to 1 as described above, so the third term is constant at 1 - theta_A - theta_M; its exponent is part of the CES form we posit rather than something the aggregation derives." Neither the body nor footnote 22 mentions the normalization or that K never varies.
- **Why it's a problem**: The display presents K as a third input of a derived CES when in the appendix it is a constant with an imposed exponent.
- **Proposed fix**: Add to the gloss after (9), or to footnote 22, that aggregate capital is normalized to K = 1 so the third term is constant with a posited exponent, and that (9) is a representation valid along a one-dimensional locus rather than an identification over arbitrary (A, M, K).

### D9. The two-stage commitment timing is credited with delivering the CES; OA.C says the same timing is why the object is not a genuine three-input CES
- **Location**: p. 30 (PDF 30) `6_extensions.tex:34`; `OA_C_CES_representation.tex:248-249`.
- **Issue**: The sentence sits under "it is worth stating what makes it work" and credits the commit-then-learn timing with producing the heterogeneity behind (9). OA.C states the cost of the same timing: "recovering a genuine three-input CES requires firms to differ in at least two of their input requirements ... Our two-stage timing rules this out by construction, since every firm commits to the same T and J before learning alpha-bar." The timing also does not generate the dispersion; the density phi is reverse-engineered from the target CES parameters, and the timing's role is to make T and J common so that alpha-bar is the only dimension of heterogeneity.
- **Why it's a problem**: The status of the subsection's central claim differs between body and appendix.
- **Proposed fix**: Say the appendix takes the dispersion in realized effective AI quality as given and uses the timing to make T and J common, and add that this one-dimensional heterogeneity is also why the construction falls short of a genuine three-input CES, as the appendix notes.

### D10. The CES share restriction plus a positive capital weight silently requires min(tau_A, tau_M) < 1, and "for any such rho" says nothing needs checking
- **Location**: p. 30 (PDF 30) `6_extensions.tex:27`; p. OA-31 (PDF 79) `OA_C_CES_representation.tex:226-229` and `:331`.
- **Issue**: The construction needs three things at once: theta_A, theta_M > 0; the restriction theta_A tau_A^rho + theta_M tau_M^rho = 1; and 1 - theta_A - theta_M > 0, since otherwise the prefactor ((1 - theta_A - theta_M)/(theta_A tau_A^rho))^{1/rho} is not a positive real (and at equality Gamma is identically 0). But if tau_A^rho <= 1 and tau_M^rho <= 1 then 1 = theta_A tau_A^rho + theta_M tau_M^rho <= theta_A + theta_M < 1, a contradiction. Hence at least one of tau_A^rho, tau_M^rho must exceed one, which for rho < 0 means **min(tau_A, tau_M) < 1**. Neither condition is ever stated; `OA_C:331` records only the weaker 1 - theta_M tau_M^rho > 0.
- **Why it's a problem**: The restriction is presented as "the condition under which a density consistent with (OA.C.14) exists at all", but on its own it is not sufficient. For a large set of (tau_A, tau_M) the admissible weight set is empty and the aggregation result is vacuous, and the body tells the reader nothing has to be checked.
- **Proposed fix**: Add 1 - theta_A - theta_M > 0 where (OA.C.9) is posited, and state alongside the restriction that admissible weights exist if and only if min(tau_A, tau_M) < 1, noting how that interacts with the assumption w_A tau_A + w_M tau_M < 1 already made in the footnote.
- **Corroboration**: found independently by two passes.

### D11. The chain index letters ell and r are reused with four incompatible meanings, and the prose introducing eq. (10) is off by one against its own equation
- **Location**: canonical convention at `3_shortrun.tex:59-68, 80, 101`, `5_longrun.tex:46`, Table OA.A.1 rows for t_b and c_b, `OA_B_omitted_proofs.tex:444-446`. Reused: p. 31 (PDF 31) `6_extensions.tex:62-65` and `OA_B:523-530` (ell = last step **before** the chain); `6_extensions.tex:83` and `OA_B:572, 583, 602-604` (r = last step **not** in the chain); `OA_B:284-286, 307` (ell = **last** step of the chain); `OA_C_CES_representation.tex:46` (ell = a task index inside a job).
- **Issue**: Section 3 fixes the convention once: a chain spans (s_ell, ..., s_r), ell is its first step and r is the augmented endpoint whose verification time t^A_r prices it. Table OA.A.1 independently confirms it. Four later blocks invert or repurpose the same letters. In the worst case the prose at `6_extensions.tex:62` says the chain "reaches back to step ell" while Equation (10) runs the product over i = ell+1 to k with the underbrace "step k ends a chain begun at ell+1", so ell is the last step **not** in the chain. Taken literally, step ell is both inside the chain and charged inside C[ell].
- **Why it's a problem**: These letters are what tells the reader which end of a chain is verified, and the economics of chaining rests on the cost being t^A_r at the endpoint rather than something that scales with length.
- **Proposed fix**: Adopt one convention. Either keep (s_ell, ..., s_r) everywhere and write the recursion as C[k] = min{C[k-1] + t^M_k, min_{ell <= k}(C[ell-1] + t^A_k/prod_{i=ell}^{k} q_i)}, or keep the cut-variable form and rename the cut to a neutral letter in eq. (10), eq. (11) and their proofs. Fix the off-by-one in the prose either way.
- **Corroboration**: found independently by two passes.

### D12. The empirical "average AI chain length" is the mean maximal run of AI-executed steps, which the model does not equate with a chain
- **Location**: p. 36 (PDF 36) `7_empirics.tex:60, 65-68`; measure defined at p. SA-30 to SA-31 (PDF 114-115) `SA_E_frequency_robustness.tex:79, 93`; also `0_main.tex:76`, `SA_D_prompt_robustness.tex:115-128`, `SA_F_external_validation.tex:384-401`.
- **Issue**: SA.E defines the statistic as "the average length of a maximal run of contiguous AI-executed steps". In the model an AI strategy is a partition into contiguous blocks and nothing forces a chain to be a maximal run, so a run of n AI-executed steps is anywhere from one chain of length n to n chains of length one. The model normally identifies chains inside a run from the augmented/automated labels, but Section 7 pools those labels precisely because the Anthropic labels do not respect sequencing, so the empirical run cannot be cut into chains. The paper's own Example OA.B.4 has an optimum of roughly m/3 adjacent chains sitting inside a single run of length m.
- **Why it's a problem**: Chaining, as opposed to co-located augmentation, is the paper's distinctive mechanism and the whole content of Prediction #1. Naming the statistic after the model object invites the reader to read 1.45 as an estimate of average chain length.
- **Proposed fix**: Rename the statistic ("average run of contiguous AI-executed steps") and state once in Subsection 7.1, repeated in SA.D, SA.E and SA.F, that a maximal run is an upper bound on chain length because the model permits adjacent chains and the pooled labels cannot separate one chain of length n from n adjacent length-one chains.

### D13. The 10,708-task estimating sample is produced by an undisclosed workflow-position restriction; the two stated restrictions alone give 13,407 tasks
- **Location**: p. 37 (PDF 37) `7_empirics.tex:115-117`; N carried in the Table 2 note at `:162` and in `tables/noTasksWithRepetitiveDWAs/allTasks_ai.tex`; the omitted restriction surfaces only at `SA_E_frequency_robustness.tex:16` (p. SA-27).
- **Issue**: Applying exactly the two restrictions the text names to the 17,925-row merged file yields **13,407 tasks across 1,888 DWAs**, not 10,708 across 1,748. The reported figures reproduce exactly (10,708 / 1,748 / 871 occupations) only after a third, unstated restriction: every task lacking two predecessors **and** two successors inside its occupation's workflow is dropped. The notebook does this explicitly (neighbour flags built with shift(1), shift(2), shift(-1), shift(-2), then "Drop rows where ANY neighbor flag is NA").
- **Why it's a problem**: A replicator following the stated construction lands on a different sample from the one every column of Table 2 and Figure OA.A.1 is estimated on. The dropped rows are precisely the first two and last two steps of every workflow, which is not neutral for a neighbor regression.
- **Proposed fix**: State the third restriction where the sample is defined ("we further keep only tasks with two steps on each side inside the workflow, since Equation (12) requires all four neighbour indicators") and report the counts after each filter in sequence.

### D14. The Prediction #2 placebo null is estimated on a 32-41% smaller sample and under a different adjacency definition than the observed estimates it is plotted against
- **Location**: p. 39 (PDF 39) `7_empirics.tex:175-178`; figure note p. OA-4 (PDF 52) `OA_A_tables_and_figures.tex:202-203`.
- **Issue**: The note presents the histogram and the red observed line as the same regression on reshuffled versus actual orderings. Observed N (from `regression_ame_results_full_0.csv`, identical to Table 2): 10,708 (panel a), 10,708 (b), 9,861 (c), 4,096 (d). The 999 reshuffle draws: **7,257** in every draw in (a); 7,096-7,257 in (b); 6,123-6,716 (median 6,545) in (c); 2,209-2,561 (median 2,403) in (d), i.e. **68%, 68%, 66% and 59%** of the observed samples. The cause is that the shuffle starts from `merged_data`, which has already been dropna'd on the four neighbour flags, then recomputes neighbours **within the DWA-eligible subsample** and drops again. So the placebo's neighbours are neighbours of a different sequence.
- **Why it's a problem**: The observed AME is drawn against a null built on a different estimation sample and a different adjacency definition, so the position of the null relative to the observed point is not interpretable as a p-value.
- **Proposed fix**: Reshuffle positions in the full occupation workflow, recompute neighbours there, and only then apply the DWA restrictions, so the placebo samples equal 10,708 / 10,708 / 9,861 / 4,096. Otherwise state the placebo Ns in the note and say the null is estimated on the re-neighboured subset.
- **Corroboration**: found independently by two passes of the number check (main text and OA.A).

### D15. Figure OA.A.1's placebo comparison does not hold in panel (d): the observed immediate-neighbor effects sit inside the DWA-fixed-effects placebo distribution
- **Location**: p. OA-4 (PDF 52) `OA_A_tables_and_figures.tex:199`; main-text sentence p. 39 (PDF 39) `7_empirics.tex:178`.
- **Issue**: Percentile of the observed AME within its own 999-draw placebo distribution: panel (a) 100.0 / 100.0 for (k-1)/(k+1); panel (b) 100.0 / 100.0; panel (c) 100.0 / 100.0; **panel (d) only 87.4** for (k-1) (observed 0.0517 vs null mean 0.0220, null sd 0.0262) and **78.4** for (k+1) (observed 0.0413 vs null mean 0.0213, null sd 0.0259). One-sided placebo p-values in panel (d) are **0.126 and 0.216**.
- **Why it's a problem**: Panel (d) is the DWA fixed-effects specification, the most demanding comparison in the figure and the one used to rule out task-type heterogeneity. The main-text sentence "In each case, the actual orderings deliver stronger immediate-neighbor effects" is not true of that case.
- **Proposed fix**: Qualify both the note and the main-text sentence, and report the placebo percentile or p-value in each subpanel.

### D16. Subsection 7.3's reconciliation paragraph denies a cross-panel disagreement the paper's own standard errors reject, counts the O*NET null as evidence, and gives a power explanation the standard errors do not support
- **Location**: p. 41-42 (PDF 41-42), `7_empirics.tex:256-263`.
- **Issue**: Three problems in one paragraph.
  - "The contrast between the two panels is not a disagreement between the corpora." Differencing paired coefficients in the units the table reports gives **0.34 with s.e. sqrt(0.10^2 + 0.10^2) = 0.141, t = 2.40** for (1) vs (4), and **0.30 with s.e. 0.142, t = 2.11** for (3) vs (6). The O*NET 95% intervals are [-0.21, 0.19], [-0.27, 0.09], [-0.22, 0.14], and the APQC point estimates -0.35 and -0.34 lie outside the first and third.
  - The closing sentence says "both panels give evidence consistent with the mechanism". The O*NET panel is a failure to reject (-0.01, -0.09, -0.04 against 0.10, 0.09, 0.09).
  - The divergence is explained by differing residual identifying variation ("only about 30% of the index's standard deviation survives conditioning" vs 56%). All coefficients are standardized within sample, and the reported EFI standard errors are **0.10, 0.09, 0.09 (O*NET) against 0.10, 0.11, 0.11 (APQC)**, so the two panels are equally precise in the units reported. In standardized units the stated mechanism predicts an O*NET standard error roughly 1.5x the APQC one; it is not observed.
- **Why it's a problem**: This paragraph is the hinge of the empirical section, where a null and a significant estimate are converted into one supported prediction, and it is the only textual justification the abstract, introduction and conclusion have for stating Prediction #3 without qualification (M1).
- **Proposed fix**: Report that the channel is detected on PCF and not on O*NET; state what the O*NET interval does and does not exclude (it admits effects up to roughly 0.2 sd but excludes an effect of the APQC magnitude, so imprecision alone does not reconcile the panels); either report the raw-unit slopes and residual standard deviations so the power claim is checkable, or drop the "better powered" framing and rest on the caveat already in the table note that standardized coefficients are not comparable in levels across panels.
- **Corroboration**: found independently by two passes.

### D17. The conclusion claims the model explains why firms invest heavily in AI capabilities; the model has no investment decision
- **Location**: p. 44 (PDF 44), `8_conclusion.tex:14`.
- **Issue**: AI quality alpha is an exogenous parameter of a general-purpose technology. The firm's only choices are the AI deployment strategy (short run) and the job design (long run). There is no investment decision, no cost of raising alpha, and no dynamics anywhere in the paper. Lemma OA.B.4 is a static comparative statement about -dC/dalpha at exogenous alpha. Section 4.3 and the introduction make only the weaker J-curve claim.
- **Why it's a problem**: The stated explanandum (firms sinking resources into AI capability ahead of realized returns) requires an anticipation-of-threshold-crossing argument in a dynamic investment problem the paper does not write down.
- **Proposed fix**: Replace with the claim the model supports ("measured returns to AI can look flat for a stretch and then jump once a reorganization threshold is crossed") and drop the reference to firms' investment in AI capabilities, or flag it explicitly as a conjecture requiring a dynamic extension.

### D18. The upper-bound accounting in Lemma OA.B.2 omits one component in the "some step fails" case
- **Location**: p. OA-12 (PDF 60), `OA_B_omitted_proofs.tex:234-237`.
- **Issue**: In the branch where at least one step of T_b fails, the number of connected components meeting T_b is at most **1 + |F ∩ T_b|**, not |F ∩ T_b|. The component beginning at the chain's own first surviving step is created by a failure outside T_b, or by no failure at all, and is charged to nobody. Concretely, one AI chain covering the whole workflow with t^M = (1.0, 1.4, 1.1, 0.8) and q = (1.0, 0.3, 1.0, 0.8) has **FI = 2.84** while the bound the two stated cases deliver, prod q_i + sum (1-q_i)(1 + min t), equals **2.28**.
- **Why it's a problem**: The displayed inequality does not follow from the two cases as stated. It is nevertheless true, because the unconditional leading +1 absorbs the uncharged component, but the proof never says so, and this is the key inequality of the upper bound.
- **Proposed fix**: State the allocation explicitly. Charge each component to the leftmost AI chain it meets (components meeting no chain to their manual singletons via omega(C) <= sum t^M_i), note that at most 1 + |F ∩ T_b| components can be charged to T_b, and conclude the expected charge is at most 1 + sum (1 - alpha^{d_i})(1 + min t).

### D19. Effective AI quality is fixed deterministically by (OA.C.7) yet treated as a firm-specific draw; no primitive is named that could make it vary
- **Location**: p. OA-29 (PDF 77), `OA_C_CES_representation.tex:143`; (OA.C.7) at `:113`; footnote 38 at `:141`; `:147-149, 183, 249`; `6_extensions.tex:35`.
- **Issue**: (OA.C.7) defines alpha-bar as (sum_{b<=k} tau^A_b) / (sum_{b<=k} tau^A_b alpha^{-d_b}). All three ingredients are common across firms in this appendix: alpha is "the quality of the general-purpose AI technology", and `OA_C:147` ("they choose identical AI strategies and job designs") together with `:249` ("every firm commits to the same T and J before learning alpha-bar") fix tau^A_b and d_b. Alpha-bar is therefore a single number and the density phi that (OA.C.11)-(OA.C.13) integrate over is degenerate. `OA_C:183` confirms the tension inadvertently, since the upper endpoint alpha-bar = 1 is reached "when alpha -> 1^-", i.e. by moving the economy-wide parameter, not a firm-level realization.
- **Why it's a problem**: Cross-firm variation in alpha-bar is the sole input to the Leontief-to-CES aggregation, and footnote 38 says heterogeneity enters "through variations in effective AI quality alone".
- **Proposed fix**: Name the primitive that varies. Introduce a firm-specific realized quality alpha_f drawn around the common technology level, inducing alpha-bar_f through (OA.C.7), distinguish alpha from alpha_f in the text and in Table OA.A.1, and restate `OA_C:183` as alpha-bar_f -> 1 as alpha_f -> 1.

### D20. Nothing in the model determines firm scale y(alpha-bar); output per firm is imposed, not derived
- **Location**: p. OA-31 (PDF 79), `OA_C_CES_representation.tex:190-194`; see also `:140-142` and the footnote at `:148`.
- **Issue**: The firm technology (OA.C.4), y = min{alpha-bar l_A/tau_A, l_M/tau_M}, is constant returns to scale in the two labor inputs and contains no capacity term. With p = 1 and w_A l_A + w_M l_M <= y, profit equals y[(1 - w_M tau_M) - w_A tau_A/alpha-bar], strictly increasing in y for every alpha-bar strictly above the threshold u = w_A tau_A/(1 - w_M tau_M), so profit-maximizing scale is unbounded there and indeterminate at alpha-bar = u. Yet the derivation assigns each type a specific finite output phi(alpha-bar), invariant to the wage vector. In Houthakker (1955) and Levhari (1968) the distribution being integrated is a distribution of exogenously fixed plant capacities, and that fixed capacity is what makes each active unit's output finite and price-independent. The one device that would pin scale here (capital) is announced and then voided by the K = 1 normalization.
- **Why it's a problem**: The density that carries the whole aggregation is a density of an object the model does not determine.
- **Proposed fix**: Either write the technology as y = min{alpha-bar l_A/tau_A, l_M/tau_M, kappa k_f} with an exogenous firm-level capital endowment whose distribution across alpha-bar is the primitive being solved for, so that phi is a capacity schedule; or state explicitly that with CRS firm-level scale is indeterminate and phi is an assumed capacity distribution rather than a derived object.

### D21. 789 of 17,925 task records carry no Eloundou label and are silently pooled with the unexposed, against an appendix that states an exhaustive partition
- **Location**: p. SA-1 (PDF 85) `SA_A_sample_construction.tex:16-21`; the variable enters `7_empirics.tex:195-207` (the EFI), `:217-222` (Eq. 14), the notes at `:236-238` and the 44% figure at `:262`.
- **Issue**: The stated rule presents an exhaustive E1/E2-versus-E0 partition ("Any task that is neither E1 nor E2 is labeled E0"). In the constructed sample `human_labels` takes E0 (9,318), E2 (5,287), E1 (2,531) and **missing (789)** out of 17,925, i.e. **4.4%** carry no human label; the merge left them unmatched (the notebook prints "Number of unmatched tasks: 791"). Every implementation codes exposure with `.isin([...])`, which returns False on a missing value, so all 789 are coded unexposed and the effective unexposed group is 10,107, not 9,318. The pooling is not innocuous: unlabeled tasks are AI-executed at **19.6% (155 of 789)** against **5.1%** for E0 and **22.0%** for E1|E2. Unlabeled versus E0 gives chi-squared p = 2.3e-59; unlabeled versus E1|E2 is indistinguishable (p = 0.14). They behave like exposed tasks and are coded as the opposite.
- **Why it's a problem**: The exposure share and the EFI both enter Equation (14) with measurement error correlated with the dependent variable, and a replicator cannot infer the coding choice from the text.
- **Proposed fix**: State the count and the rule in SA.A ("789 of 17,925 tasks carry no human label from Eloundou et al.; we group these with the unexposed"), then either drop them from the exposure and EFI computations or report that the results hold when they are dropped. Drop SOC 33-3051.00 from the fragmentation sample, since it contributes exposure 0 and EFI 1 with no underlying information.
- **Corroboration**: found independently by three passes.

### D22. The claim that a workflow sequence is generated for every O*NET occupation is false, and five tasks are assigned two positions each
- **Location**: p. SA-2 (PDF 86), `SA_A_sample_construction.tex:72`.
- **Issue**: `data/computed_objects/tasks_sequences` contains **872 files for the 873 occupations** with rated tasks; occupation 47-2053.00 has no sequence and is dropped. Set-differencing the 17,953 rated (SOC, Task ID) pairs against the sequence pairs leaves **33 records with no Task Position**, 26 of them in 47-2053.00 and 7 scattered across six other occupations. Separately, the GPT ordering assigned **two positions to each of 5 tasks** within the same occupation (11-9032.00 task 5216 at positions 10 and 32; 29-1021.00/20562 at 4 and 20; 29-9099.01/17471 at 8 and 36; 39-3011.00/4452 at 10 and 21; 43-3031.00/2499 at 6 and 30), so the final file has 17,925 rows for only 17,920 distinct occupation-task pairs.
- **Why it's a problem**: Every position-based measure in the paper (fragmentation index, neighbor AI status, chain length) reads Task Position, and five tasks sit at two contradictory points in their own workflow.
- **Proposed fix**: Say a sequence is generated for 872 of the 873 occupations with rated tasks and that 33 rated records receive no position; then either de-duplicate the 5 doubly-positioned tasks or state that they are retained.

### D23. The Anthropic labels are described as feasibility-based, contradicting Section 7's characterization of the same variable
- **Location**: p. SA-4 (PDF 88) `SA_A_sample_construction.tex:100`; same wording in the footnote at `7_empirics.tex:27`.
- **Issue**: Section 7 draws the opposite contrast: "Whereas these labels capture what AI could in principle do, the Anthropic Economic Index captures what AI actually does". Feasibility is the Eloundou E1/E2 rubric. The Anthropic augmentation/automation split classifies the observed interaction pattern of the conversations covering a task, as the very next sentence of the same appendix states ("A task is labeled automated if the conversations covering it are predominantly directive").
- **Why it's a problem**: The empirical design rests on exposure (what AI could do) versus execution (what AI actually does). Calling the Anthropic labels feasibility-based collapses that distinction, and the paragraph's own argument needs "realized interaction pattern, agnostic to workflow position", not "feasibility".
- **Proposed fix**: Replace with "whereas the Anthropic classifications are based on the observed interaction pattern of the conversations covering a task and are agnostic to workflow position", in both places.

### D24. "AI automation" is framed as the model's notion of automation, contradicting SA.A's account of the same label
- **Location**: p. SA-9 (PDF 93), `SA_B_alternative_definitions.tex:33`.
- **Issue**: SA.A says the opposite about this measure: "The automated versus augmented distinction in our model is defined by production sequencing and task dependencies, whereas the Anthropic classifications are ... agnostic to workflow position", and "For this reason the empirical analysis in Section 7 does not distinguish tasks labeled automated from those labeled augmented". The `is_automated` label is assigned by the majority share of directive conversations, not by workflow position.
- **Why it's a problem**: The exercise is sold as tightening the outcome to the concept the model implies, but by the paper's own account the label does not measure that concept.
- **Proposed fix**: Reframe as a robustness check on the subset of AI-executed tasks whose conversations are predominantly directive, with a pointer to SA.A noting that this label is not the model's automated step.

### ✅ D25. Tables SA.B.2 and SA.B.3 carry the wrong dependent-variable header
- **Status**: ✅ **Addressed** 2026-09-04 (commit `228864c`). The spanner in both tables now reads "is AI-automated". `generate_latex_table` keys the label on `dependent_var`, so re-running the notebooks cannot reintroduce it.
- **Location**: p. SA-11 to SA-12 (PDF 95-96), `tables/randomTieBreak/allTasks_automated.tex:4` and `GPT_automated.tex:4`, input at `SA_B_alternative_definitions.tex:65, 83`.
- **Issue**: Both tables carry the spanning header "Probability that Focal Task (k) is AI-executed", while the estimated outcome is `is_automated`: the captions read "Task's AI Automation Likelihood", the notes say the dependent variable is `is_automated_k`, and the notebook calls `run_regressions_on(..., dependent_var='is_automated')`. The printed numbers are correct for `is_automated` (verified cell by cell against `regression_summaries_is_automated/regression_ame_results_full_0.csv`: 0.01864, 0.04592, 0.05318, 0.02711, nobs 13,786).
- **Why it's a problem**: The header names the outcome, so as printed the tables claim to be the same regressions as Table SA.B.1 on a different sample when they are a different outcome.
- **Proposed fix**: Change the spanning header in both files to "Probability that Focal Task ($k$) is AI-automated".
- **Corroboration**: found independently by two passes.

### D26. The note to Figure SA.B.2 claims an immediate-neighbor effect its own placebo panels do not support
- **Location**: p. SA-14 (PDF 98), `SA_B_alternative_definitions.tex:174`.
- **Issue**: In panel (d) (DWA fixed effects) the observed AMEs are **-0.014 (k-2), 0.018 (k-1), 0.013 (k+1), 0.019 (k+2)**: the two-positions-away effect is the largest, none is significant in column (4) of Table SA.B.2, and all four sit at the centre of the position-reshuffle placebo distribution. Even in panel (a) the observed immediate-neighbor AMEs (0.046, 0.053) sit just right of a placebo mode near 0.035-0.04, unlike the AI-execution figure where the observed value is in the far right tail.
- **Why it's a problem**: The note asserts as an established pattern what the body of the same appendix correctly reports as "smaller in magnitude and often statistically insignificant". The exhibit's own benchmark does not separate the automation estimates from chance.
- **Proposed fix**: Rewrite the note to say the automation estimates keep the sign of the AI-execution estimates but are smaller, mostly indistinguishable from zero, and not separated from the position-reshuffle placebo, especially under DWA fixed effects.

### D27. The note to Table SA.B.5 misdescribes the fragmentation regressor as built on AI-able steps
- **Location**: p. SA-17 (PDF 101), `SA_B_alternative_definitions.tex:278`.
- **Issue**: In this table the index is built on AI-**executed** steps, as the very next sentence of the same note says, not on AI-able/AI-exposed steps. (The same sentence also calls the exposure control "E1", which is the separate problem in M2.)
- **Why it's a problem**: The fragmentation regressor is described as measuring the dispersion of a different set of steps from the one it is built on, which is the entire point of the table.
- **Proposed fix**: "The variable 'AI Exposure' denotes the share of AI-exposed (E1 or E2) steps in the occupation, while the 'Empirical Fragmentation Index' here captures how dispersed the AI-executed steps are across the occupation's workflow."

### ✅ D28. The EFI variants are cited as "Definition 1" and "Definition 2", numbers already taken by Manual Step and Augmented Step, and never defined anywhere
- **Status**: ✅ **Addressed** 2026-09-04 (working tree). The six "Definition~1" labels in SA.D and SA.E were removed. Four were plain parentheticals; the other two were reworded so the sentence still reads, keeping the cross-reference and the E1/E2 clause. The "Definition~2" variant went with the removal of SA.B.2 under M4. No EFI variant is now cited by number, so the collision with Definition~1 (Manual Step) and Definition~2 (Augmented Step) is gone; those two remain the only numbered definitions in the PDF.
- **Location**: p. SA-17 (PDF 101) `tables/fragmentation_index_regression_execution.tex:12`; p. SA-23 to SA-25 (PDF 107-109) `SA_D_prompt_robustness.tex:68, 157, 172`; p. SA-34 to SA-35 (PDF 118-119) `SA_E_frequency_robustness.tex:192, 202`. Colliding targets `3_shortrun.tex:26, 32` (p. 10).
- **Issue**: These are the only hard-coded numbered cross-references in the draft, and both numbers are occupied. `preamble.tex:154` declares a single running `definition` counter, and the only `\begin{definition}` environments are in `3_shortrun.tex` (Manual Step, Augmented Step, Automated Step, AI Chain, AI Deployment Strategy) and `5_longrun.tex` (Job Design), so **Definition 1 = Manual Step** and **Definition 2 = Augmented Step**, both on PDF page 10. Subsection 7.3 introduces the EFI in running text with no numbered environment, and no "Definition 1" EFI variant appears anywhere in the compiled document. `SA_E:202` goes further and cites "Definition 1 of Section 7.3", which that section does not contain.
- **Why it's a problem**: A reader chasing "Definition 1" lands on Manual Step. Table SA.B.5 is the only place the execution-based index is tabulated, so its row label is the reader's only handle on which index it is.
- **Proposed fix**: Drop the numbers and use the descriptive names the draft's own prose already uses. Rename the SA.B row to "Empirical Fragmentation Index (execution-based)", replace "(Definition 1)" with "exposure-based EFI" throughout SA.D and SA.E, and rewrite `SA_E:202` accordingly. Same fix in the four orphan table sources that carry the tags.
- **Corroboration**: found independently by five passes. Five raw findings merged.

### D29. Prompt #2 caps the GPT sample at one task per occupation, so the columns (5)-(6) control is constant and column (6) reproduces column (4)
- **Location**: p. SA-20 (PDF 104) `SA_C_gpt_prompts.tex:44`; exhibits `tables/noTasksWithRepetitiveDWAs/GPT_ai.tex` (Table SA.B.1) and the GPT-filtered automation table; description at `SA_B_alternative_definitions.tex:26-28, 55, 91`.
- **Issue**: Prompt #2 instructs the model to "Return only the most relevant task for every occupation", so within each DWA the GPT-filtered sample has one task per occupation by construction. In `GPT_ai.tex` columns **(4) and (6) are numerically identical in every coefficient (0.04, 0.06\*\*, 0.09\*\*\*, -0.00), every standard error, the observation count (2,544) and the pseudo R-squared (0.198)**, i.e. the added control is collinear with a constant. The tables nonetheless carry a "NumTasks in DWA-Occupation Control" checkmark in columns (5) and (6), and SA.B never states the one-task-per-occupation restriction.
- **Why it's a problem**: The main text describes that control as ruling out "mechanical proximity inflating the estimates when several same-DWA tasks sit in one workflow", but in this sample several same-DWA tasks never sit in one workflow. The undisclosed restriction is also part of why the filtered sample falls to 3,689 tasks.
- **Proposed fix**: State the restriction in SA.B, and either drop columns (5)-(6) from the GPT-filtered tables, mark the control not applicable, or note explicitly that it is constant so column (6) reproduces column (4).

### D30. SA.D never discloses that the alternative prompts returned orderings for fewer occupations than the main prompt
- **Location**: p. SA-21 to SA-26 (PDF 105-110), `SA_D_prompt_robustness.tex:7, 61, 144, 155, 158, 171-176, 219-225`.
- **Issue**: The alternative-prompt datasets do not cover the sample. `ONET_Eloundou_Anthropic_GPT_{1..10}.csv` hold **868, 867, 868, 868, 868, 866, 868, 867, 868 and 785** occupations against **872** for the main file, and 16,067 task rows for prompt 10 against 17,925. In the Prediction #3 grid the regression N is **872 at Prompt 0 and 785 at Prompt 10**; in the Prediction #2 panels it is **10,708 at Prompt 0 and 9,415 at Prompt 10** with no fixed effects, and 4,096 versus 3,569 with DWA fixed effects. In Figure SA.D.1 the note's "across 11 prompts" holds for only **737 of the 869** occupations plotted (118 have 10, 11 have 9, 3 have 8).
- **Why it's a problem**: The figures' whole point is that the eleven points are comparable, and `SA_D:158` says the alternative estimates "fall inside the confidence interval of the main prompt estimate". Prompt 10 is estimated on a 10% smaller occupation set and a 12% smaller task set.
- **Proposed fix**: Report per-prompt occupation and task counts in the figure notes or a small table, say why prompt 10 is short 87 occupations, and note that occupations whose orderings failed for a given prompt are dropped from that prompt's Kendall comparisons, in the same way SA.F reports its 2 of 3,795 failures.

### D31. "Prompt 0" denotes two different ordering datasets inside SA.D, and Figure SA.D.1's version reproduces the analysis ordering for only 15 of 868 occupations
- **Location**: p. SA-22 (PDF 106), `SA_D_prompt_robustness.tex:61-62, 25-26, 66`; contrast `:129, :175, :225`; used at `7_empirics.tex:35`.
- **Issue**: In Figures SA.D.2, SA.D.3 and SA.D.4 the Prompt 0 point is the main-analysis sample, and its cells reproduce the published tables exactly (N = 872 with 0.49/0.48/0.39 and -0.01/-0.09/-0.04; N = 10,708 with 0.07/0.12/0.12/0.05). Figure SA.D.1 does not. Its prompt 0 is read from `tasks_sequences_robustness/<occ>/<occ>_0.csv`, a **separate re-run** of the baseline wording (the generating notebook labels index 0 "Main spec - preserved verbatim"). For Accountants and Auditors, Actors and Actuaries the analysis ordering matches `tasks_sequences/<occ>.csv` at tau = 1.000, while the re-run gives tau = 0.686 and similar.
- **Why it's a problem**: `SA_D:66` attributes the measured 0.60 to "the diversity of prompt formulations considered" and `7_empirics.tex:35` uses it to say the orderings overlap substantially across ten varied prompts. Much of the measured disagreement is run-to-run instability of the **same** prompt, not prompt variation.
- **Proposed fix**: Either recompute Figure SA.D.1 with the main-analysis sequences as the prompt-0 leg, or keep the re-run and report the re-run-versus-main tau of about 0.70 as a same-prompt benchmark against which the cross-prompt 0.60 should be read. Say which dataset "Prompt 0" means in each figure.

### D32. Figure SA.D.1 Panel (b) plots a superseded EFI split
- **Location**: p. SA-22 (PDF 106), `SA_D_prompt_robustness.tex:38-42, 62, 68`.
- **Issue**: The Panel (b) legend prints **n = 328 / 540** (mean 0.56 / 0.63), a 38/62 split, while Panels (c) and (d) print 428/440 and 430/438. Re-running the notebook's own split rule on the current `occupation_analysis_with_fragmentationIndex_def1.csv` gives **n = 433 (mean 0.573) / 435 (mean 0.630)**. The cause is a data revision: the current index has median **0.8095** with only 0.7% of occupations tied there, whereas the plotted version had median 1.0, matching the notebook comment that "62.3% of occupations sit on that ceiling, so its median IS 1.0" (540/868 = 62.2%). File times confirm the ordering (PNG Sep 1 17:15, CSV Sep 1 18:48).
- **Why it's a problem**: The panel a reader is told is the EFI split of the current sample is drawn from a superseded series, and anyone re-running the notebook gets different group sizes.
- **Proposed fix**: Re-run `analysis/GPT_task_sequences_overlap.ipynb` cells 7-12 against the current file and regenerate the panel and its `_count` twin. Update the now-false cell-7 comment and revisit whether the ">=" tie rule is still needed at a median of 0.81.
- **Severity note**: rated minor by the per-file pass and medium by the number pass, which re-ran the split rule; medium taken.

### D33. The adjacent-beats-two-away claim is unqualified but fails in Panel (d) of Figure SA.D.4 for 6 of 11 prompts
- **Location**: p. SA-24 (PDF 108), `SA_D_prompt_robustness.tex:148-149` (body) and `:219-221` (figure notes).
- **Issue**: In the DWA fixed-effects panel the printed cross-prompt means are **(t-2) 0.01, (t-1) 0.06, (t+1) 0.03, (t+2) 0.02**, with mean standard errors 0.0213 and 0.0210 for (t+1) and (t+2), so the two-away effect is essentially the same size and nominally the more precisely estimated. Per-prompt AMEs show min(adjacent) < max(two-away) for **prompts 1, 2, 5, 6, 7 and 8**, i.e. 6 of 11. Examples: prompt 2 next2 = +0.060 (p = 0.006) versus next = +0.014 (p = 0.57); prompt 6 next2 = +0.044 (p = 0.09) versus next = +0.006 (p = 0.77); prompt 8 next2 = +0.042 (p = 0.05) versus next = +0.014 (p = 0.50). The companion claim that two-away effects "attenuate and become less distinguishable from zero" as fixed effects are added also fails for (t+2), whose across-prompt mean rises from 0.0034 under the main prompt to 0.0251.
- **Why it's a problem**: Locality is the substance of Prediction #2, and this is the entire robustness content for it under alternative prompts, stated without qualification in both the body and the figure notes.
- **Proposed fix**: Restrict the statement to Panels (a)-(c) and report Panel (d) as it is: under DWA fixed effects the across-prompt means for (t+1) and (t+2) are 0.03 and 0.02 with comparable standard errors, so that specification does not separate one step from two on the forward side, although the (t-1) effect (mean 0.055) remains clearly the largest.
- **Severity note**: medium in the per-file pass, minor in the number pass; medium kept (see reconciliations above).

### D34. SA.E's claim that all three predictions run on one common set of occupations fails at every pruned cut, and its stated justification is false
- **Location**: p. SA-27 (PDF 111) `SA_E_frequency_robustness.tex:14-16`; consequence at `:193` (p. SA-34).
- **Issue**: The parenthetical asserts that no occupation is lost when Prediction #2 is restricted to neighboured tasks, "which a five-task workflow always admits". A five-task workflow contains exactly one such position (the third), and that task is then subject to the DWA filters. Replicating the pipeline gives, for the frequency cuts, **816 / 689 / 560 / 411 / 513 / 330 / 177 / 74 / 295 / 111 / 32 / 4** occupations in the Prediction #2 sample against **832 / 725 / 602 / 475 / 564 / 388 / 236 / 112 / 345 / 162 / 75 / 20** in the Prediction #1 and #3 samples. Only the unpruned all-tasks row shares an occupation set (871 in both). At the strictest cut the two samples are 4 and 20 occupations.
- **Why it's a problem**: The stated design of the appendix is that one sample definition is held fixed so the three predictions can be read against one another down the grid.
- **Proposed fix**: Drop the "which a five-task workflow always admits" clause and say the neighbour regression retains the eligible tasks of these occupations, hence the subset that keeps at least one such task after the DWA filters, reporting both occupation counts per cut. Adjust the Hourly+ >= 65% note on p. SA-34 so the number it cites belongs to the exercise it justifies.

### D35. Chain length is called an upper-tail result that "falls back only in the Hourly+ cuts" while the same sentence quotes a 58th percentile; 8 of 12 pruned cuts are inside the null band
- **Location**: p. SA-31 (PDF 115), `SA_E_frequency_robustness.tex:104`.
- **Issue**: The quoted percentiles are correct (Daily+ 98.4 / 99.6 / 86.9 / 96.1; SeveralDaily+ 99.5 / **58.1** / 80.8 / 80.4; Hourly+ 39.8 / 68.0 / 58.2 / 55.2), but the 58th percentile is the middle of the null, not its upper tail. Applying the figure's own colouring rule, **8 of the 12 pruned cuts plot inside the 10-90 band**: Daily+ >= 50%, the three stricter SeveralDaily+ cuts, and all four Hourly+ cuts. Only 4 pruned cuts plot outside.
- **Why it's a problem**: The subsection opens "The result survives pruning", while its exhibit shows the chain-length statistic indistinguishable from its reshuffle null in two thirds of the pruned cuts.
- **Proposed fix**: Say the observed value stays outside the null band at Daily+ >= 20%, >= 35%, >= 65% and SeveralDaily+ >= 20%, and falls inside at the other eight, and drop "upper tail" for the 58th-percentile cell.

### D36. SA.E attributes the +0.04 attenuation endpoint to SOC fixed effects, but +0.04 exists only in Table 2's DWA-fixed-effects columns, which SA.E's grid does not contain
- **Location**: p. SA-31 (PDF 115), `SA_E_frequency_robustness.tex:142`; grid defined at `:169, :194`.
- **Issue**: Table 2's SOC-fixed-effects columns give **0.06 and 0.06** (column 2, SOC major) and **0.05 and 0.05** (column 3, SOC minor), so under SOC fixed effects the range is +0.05 to +0.06. The value **+0.04 appears only in columns (4) and (6)**, the DWA-fixed-effects columns. The main text is correct because it scopes the range to "columns (2)-(4) and (6)"; SA.E imports the same range while scoping it to SOC fixed effects. SA.E's sweep carries only `no_fe_no_dwa`, `major_fe_no_dwa` and `minor_fe_no_dwa`, whose all-tasks cells are 0.1239/0.1166, 0.0607/0.0572 and 0.0506/0.0479. **No cell of SA.E's grid equals +0.04.**
- **Why it's a problem**: The sentence tells the reader the top row of the figure reproduces a main-text range against which the pruned cells below can be read. That benchmark is one the figure's own specifications cannot generate.
- **Proposed fix**: Change the range to "+0.05 to +0.06", or add the DWA specification to the frequency grid, or say the main text's lower endpoint comes from a specification not re-estimated here.

### D37. "Neighbor effects often grow" cites only cells the next sentence dismisses as too sparse
- **Location**: p. SA-32 (PDF 116), `SA_E_frequency_robustness.tex:143`.
- **Issue**: In the no-fixed-effects previous-task panel the AME is **0.124** in the all-tasks row and **0.093, 0.091, 0.086, 0.113, 0.090, 0.073, 0.137, 0.204, 0.093, 0.083, 0.149** across the eleven pruned cuts. Only three exceed the baseline, and those three are SeveralDaily+ >= 50% (N = 574), SeveralDaily+ >= 65% (N = 161) and Hourly+ >= 50% (N = 61), the three smallest samples in the panel. Across every inclusive and moderate cut the effect shrinks by roughly a quarter. Pooling all three specifications, about ten of thirty-three previous-task cells exceed their baseline, all at the strict, sparse end.
- **Why it's a problem**: The very next sentence says the estimates "weaken and lose significance only in the sparsest cells", so the appendix cites one of its smallest cells as evidence that the effect grows and then dismisses small cells as uninformative.
- **Proposed fix**: "The positive adjacent-step effect persists across the inclusive and moderate cuts, attenuating modestly (0.07 to 0.11 under no fixed effects against a 0.12 baseline); the larger point estimates in the grid, up to +0.20, occur only in the sparsest cuts, where they are correspondingly imprecise."

### D38. "Lose significance only in the sparsest cells, where task observations fall into the low hundreds" is contradicted by insignificant cells with 2,533 and 1,412 observations
- **Location**: p. SA-32 (PDF 116), `SA_E_frequency_robustness.tex:144`.
- **Issue**: Insignificant immediate-neighbor cells at the 10% level include **SeveralDaily+ >= 20% / SOC-minor FE / next task, AME 0.017, p = 0.253, N = 2,533**; **SeveralDaily+ >= 35% / SOC-major FE / previous task, AME 0.032, p = 0.107, N = 1,412**; SeveralDaily+ >= 35% / SOC-minor FE, p = 0.299 and 0.201, N = 854; Hourly+ >= 20% / SOC-minor FE, p = 0.144 and 0.144, N = 657. The heatmap prints these Ns beneath each estimate, so a reader can check the claim directly.
- **Why it's a problem**: This sentence is the appendix's defence of the neighbor result under pruning, and it attributes every loss of significance to sample size.
- **Proposed fix**: "The estimates weaken at the stricter thresholds and, under SOC minor-group fixed effects, lose significance from SeveralDaily+ >= 20% onward; they are uniformly insignificant in the Hourly+ >= 35% and >= 50% cells, where task observations fall into the low hundreds." Report the cell counts alongside.
- **Corroboration**: found independently by two passes.

### D39. The next-task effect does not escape its null "across the Daily+ cuts"
- **Location**: p. SA-34 (PDF 118), `SA_E_frequency_robustness.tex:175`.
- **Issue**: Of the 12 Daily+ by specification cells for (k+1), **4 lie inside the 10-90 band and are plotted blue**: Daily+ >= 50% no FE (observed 0.0842, band [0.038, 0.086], 88.9th percentile), and the whole Daily+ >= 65% row (0.0882 in [0.035, 0.092], 85.9th; 0.0424 in [-0.001, 0.043], 89.8th; 0.0343 in [-0.017, 0.035], 89.7th). The parallel previous-task clause does check out exactly (Daily+ percentiles 91.3 to 98.7, matching the drafted "91st to 99th").
- **Why it's a problem**: The claim asserts red dots where the figure shows blue ones, in the sentence carrying the Prediction #2 placebo robustness claim.
- **Proposed fix**: "The next-task effect escapes its null at Daily+ >= 20% and >= 35% in all three specifications and at Daily+ >= 50% under both fixed-effect specifications, but sits inside the band at Daily+ >= 65%."
- **Corroboration**: found independently by two passes.

### D40. "Retreat to null only at the sparsest cuts" is false for the next-task panel, and one of the three named cuts has 1,557 observations
- **Location**: p. SA-34 (PDF 118), `SA_E_frequency_robustness.tex:176`.
- **Issue**: The next-task effect sits inside its 10-90 null in **all three specifications** at SeveralDaily+ >= 20% (**63.3 / 69.4 / 62.2 percentiles, 493 occupations and 3,028 task observations**), SeveralDaily+ >= 50% (68.4 / 76.8 / 85.6), SeveralDaily+ >= 65% (85.1 / 86.9 / 88.9), Hourly+ >= 20% (74.2 / 85.2 / 89.0) and Daily+ >= 65%, none of which the sentence names. SeveralDaily+ >= 20% is the fifth-largest sample in the grid and sits essentially at the middle of the null. Separately, SeveralDaily+ >= 35%, one of the three cuts the sentence does name, has **1,557** neighboured task observations, not "low hundreds" (only Hourly+ >= 35% at 428 and Hourly+ >= 50% at 126 do). The 315 printed on that row is an occupation count, not an observation count.
- **Why it's a problem**: The "only" is load-bearing, since it is what lets the next sentence conclude that Prediction #2 "weakens only through lost statistical power at the strictest thresholds".
- **Proposed fix**: Split the two panels. The previous-task effect stays outside its null through SeveralDaily+ >= 50% and retreats at Hourly+ >= 35% and >= 50% and SeveralDaily+ >= 35%; the next-task effect retreats from SeveralDaily+ >= 20% and Daily+ >= 65% onward. Quote the actual counts (1,557 / 428 / 126) and say whether they are occupations or task observations.
- **Corroboration**: found independently by three passes.

### D41. Figure SA.E.4's row labels are neither task observations nor occupation counts and match no regression in the paper
- **Location**: p. SA-33 to SA-34 (PDF 117-118), `SA_E_frequency_robustness.tex:168-170` (notes) and `:176` (the prose reading them); inherited convention at `:92`; contrasting convention at `:133`.
- **Issue**: Three exhibits label sample sizes with a bare "N=" under three different conventions, and the prose reads them under a fourth. Figures SA.E.2 and SA.E.5 print **occupation** counts (871 for all tasks, 20 for Hourly+ >= 65%). Figure SA.E.3 prints **task observations** (10,708; 1,486; 288; 61). Figure SA.E.4, whose note says its labels are "as in Figure SA.E.2", prints **865 / 792 / 662 / 523 / 384 / 493 / 315 / 178 / 76 / 282 / 123 / 49 / 6**, which are neither: replicating its own regressions gives **10,708 / 6,877 / 4,920 / 3,458 / 2,115 / 3,099 / 1,486 / 574 / 161 / 1,171 / 288 / 61 / 4** task observations across **871 / 816 / 689 / 560 / 411 / 513 / 330 / 177 / 74 / 295 / 111 / 32 / 4** occupations.
- **Why it's a problem**: The power argument closing SA.E.3 rests entirely on these labels, and they are not the sample size of the regression plotted beside them under any convention.
- **Proposed fix**: Regenerate the row labels from the same pipeline that produces Figure SA.E.3 and print both counts explicitly ("SeveralDaily+ >= 35% (330 occ., 1,486 obs.)"). State the unit in the note rather than inheriting it.

### D42. "None of the branches with five or more steps" is exactly recovered is false; 21 of 208 are, and the error comes from a float-equality test
- **Location**: p. SA-41 (PDF 125), `SA_F_external_validation.tex:175`.
- **Issue**: Exact recovery is the event tau = +1. In `ordering_accuracy.csv`, `exact_p00 == 1` for **84 of 345 branches (24.3%**, the 0.24 in Table SA.F.1), of which only 63 have three or four steps. The other **21 have five or more**: 18 of 70 five-step (26%), 1 of 35 six-step, 2 of 40 seven-step, i.e. **21 of 208 (10.1%)**. The figure's rightmost bin [0.9, 1.0] contains 29 branches with five or more steps against 63 with three or four. The error traces to notebook cell 20, which computes the share at tau = +1 as `(g['tau_main'] == 1).mean()`; for n = 5, 6, 8, 10, 12 a perfect ordering returns 0.9999999999999999, so the strict equality test silently reports 0.000 for those bands.
- **Why it's a problem**: The sentence tells the reader the instrument never reproduces a documented sequence exactly once it is longer than four steps, and it cannot be reconciled with the 0.24 exact-order recovery reported in the same subsection.
- **Proposed fix**: Recompute with `np.isclose(tau, 1)` or directly from `exact_p00`, and rewrite as "56% of three-step branches and 38% of four-step branches sit there, against 26% of five-step branches and 10% of all branches with five steps or more".
- **Severity note**: rated major by the per-file pass, medium by the number-verification pass, which recomputed it and found the error **understates** the instrument. Downgrade taken.

### D43. The Kendall tau granularity illustration is wrong in both directions
- **Location**: p. SA-41 (PDF 125), `SA_F_external_validation.tex:174`.
- **Issue**: On a three-step branch there are 3 pairs, so a single discordant pair gives tau = (2-1)/3 = **+1/3**, which is positive; two of the three pairs must invert before tau turns negative. The claim also contradicts the value set {-1, -1/3, +1/3, +1} listed in the same sentence, which contains no large negative value reachable by one inversion. On a ten-step branch there are 45 pairs, so one discordant pair moves tau from 1 to 43/45 = 0.956, a cost of **2/45 = 0.044**, not 0.02. The 0.02 is the smallest attainable |tau| (1/45 = 0.022), which the notebook computes as `smallest_possible_abs_tau` = 2/(n(n-1)) and the text relabels as the cost of one inversion.
- **Why it's a problem**: The whole "reading the left tail" argument is priced on how much a single inversion costs at each branch length, and both numbers used to price it are wrong.
- **Proposed fix**: "On a three-step branch two of the three pairs must invert before tau turns negative, so any negative value there is already large; on a ten-step branch a single inversion costs 2/45 = 0.04 and leaves tau at +0.96."

### D44. The long-branch negative tau range is stated on a different sample from the 14% it accompanies, and the interpretive gloss misreads a small negative tau
- **Location**: p. SA-41 (PDF 125), `SA_F_external_validation.tex:178`.
- **Issue**: The "14% in both cases" clause is true only for the figure's own bands, three-step branches (9 of 63, 14.3%) and eight-or-more-step branches (9 of 63, 14.3%). Under that definition of "longest" the nine negative taus are **-0.429, -0.357, -0.214, -0.200, -0.167, -0.156, -0.111, -0.083, -0.030**, so the range is **-0.03 to -0.43** and three of the nine fall outside the quoted -0.03 to -0.20. The quoted interval is exactly the set obtained if "longest" means nine or more steps (six branches), so the two halves of the sentence use different definitions. Separately, the gloss "a handful of misordered pairs within an otherwise correct sequence" is wrong at any cut: tau = -0.03 on a twelve-step branch means 34 of 66 pairs discordant; tau = -0.43 on an eight-step branch means 20 of 28, against the 14 a random ordering delivers.
- **Why it's a problem**: The paragraph's job is to argue the long-branch left tail is benign. The stated range understates the spread and the gloss describes a near-random or worse-than-random ordering as nearly correct.
- **Proposed fix**: Use one definition of "longest" and report the true range and its meaning: "among the 63 branches with eight or more steps, 9 (14%) are negative, from -0.03 to -0.43; even the mildest of these has slightly more than half its pairs inverted, so these are near-random rather than nearly correct orderings."
- **Corroboration**: found independently by two passes.

### D45. The inference from the coincidence of the three pair-level accuracies is near-vacuous because 95.4% of pairs are determinate
- **Location**: p. SA-43 (PDF 127), `SA_F_external_validation.tex:262`.
- **Issue**: The paper reports two paragraphs earlier that 1,124 of 1,178 pairs (95.4%) are determinate. The all-pairs and determinate-pairs samples therefore overlap by 95%, and once the determinate figure is 0.789 the all-pairs figure is arithmetically confined to **[0.753, 0.799]** whatever happens on the 54 indeterminate pairs. Their coinciding at 0.788 / 0.789 / 0.788 is close to forced (verified from the microdata: 0.7878, 0.7891, 0.7593 on n = 54).
- **Why it's a problem**: An inference ("the aggregate is not being propped up by easy cases") is drawn from a comparison with essentially no power to detect what it is invoked against.
- **Proposed fix**: Drop the "itself informative" inference, or replace it with the comparison that does have power, accuracy across determinacy bands, which is flat to non-monotone in the microdata (0.77 on 0.5-0.6, 0.94 on 0.6-0.7, 0.83 on 0.7-0.8, 0.75 on 0.8-0.9, 0.78 on 0.9-1.0).

### D46. The Prediction #2 power counts in SA.F are the 0.73-floor numbers, not the 0.71 floor the appendix uses
- **Location**: p. SA-47 (PDF 131), `SA_F_external_validation.tex:409-411`.
- **Issue**: Drafted **981 observations, 137 DWAs, 9 DWAs** containing both an AI-executed and a non-executed step. The similarity threshold set two pages earlier and marked on Figure SA.F.3 is **0.71**. Re-running the power block at 0.71 over the same inputs gives **1,446 steps / 198 recurring DWAs / 23 varying DWAs**; the drafted triple is exactly the 0.73-floor output. Every other number in the subsection is at 0.71 (2,067 cleared steps, 15.3% share, mean cosine 0.75, 12.2% exposed, 4.0% executed, EFI sd 0.057).
- **Why it's a problem**: One paragraph reports a different, 32% smaller sample from the one it names, making the matched corpus look less powered than it is.
- **Proposed fix**: Replace with the 0.71-floor values: 1,446 observations across 198 DWAs, of which 23 contain both an AI-executed and a non-executed step.

## Minor

### N1. "Appending a step costs only a lower probability of end-to-end success" holds only for a predecessor
- **Location**: p. 4 (PDF 4), `1_introduction.tex:86`.
- **Issue**: By Definition 4 a chain over (s_ell, ..., s_r) costs t^A_r / prod_{i=ell}^{r} q_i, so the verification time is the **last** step's. Extending backwards to ell-1 leaves t^A_r unchanged and multiplies cost by 1/q_{ell-1}, exactly as described. Extending **forwards** to r+1 moves the endpoint, so t^A_r becomes t^A_{r+1} and the previous endpoint's verification is dropped.
- **Why it's a problem**: The claim is false for forward extensions, and it is the sentence carrying the "verification is a fixed cost of the chain" intuition into the introduction. Section 4.1 is careful about this.
- **Proposed fix**: Add the direction: "Folding a **preceding** step into an existing chain therefore costs only a lower probability of end-to-end success, since the chain is still verified once, at its unchanged endpoint."

### N2. The Prediction #1 preview drops the qualification that the mean AI chain length is 1.45
- **Location**: p. 6 (PDF 6), `1_introduction.tex:130-131`.
- **Issue**: Section 7.1 reports a mean of **1.45** and says "the modest magnitude of 1.45 indicates that long AI chains remain rare"; Section 7.2 adds that "tasks two positions away rarely fall in the same chain". A mean of 1.45 means the typical AI chain is a single step. The established finding is an excess over two randomization benchmarks, not that AI execution generally spans consecutive steps.
- **Proposed fix**: "AI-executed steps cluster into chains significantly longer than either of two randomization placebos produces, though the observed mean chain length of 1.45 shows that long chains are still rare at this stage of adoption."

### N3. The robustness claim covers three measures but all three cited exercises vary only the sequence
- **Location**: p. 7 (PDF 7), `1_introduction.tex:137-141`.
- **Issue**: The three measures named are the sequence of steps, AI exposure and AI execution. The three exercises listed (alternative prompts SA.D, frequency pruning SA.E, external corpora SA.F) all vary the sequence. `SA_B_alternative_definitions.tex:19` says so itself: "Appendices SA.D, SA.E, and SA.F vary the sequences themselves rather than the definitions." The execution definition is varied only in SA.B, which the introduction never cites, and the exposure labels are never re-constructed anywhere.
- **Proposed fix**: Add a fourth item citing SA.B, or narrow the opening claim to robustness of the task ordering.

### N4. The mode labels M and A are reused as the aggregate labor inputs of Section 6.1 and OA.C
- **Location**: p. 9 (PDF 9) `3_shortrun.tex:20-21`; p. 30 (PDF 30) `6_extensions.tex` eq. (9); throughout `OA_C_CES_representation.tex`; `preamble.tex:378-386`.
- **Issue**: M and A are introduced as labels for the two execution modes, used only in sub/superscript position. Equation (9) then uses the same bare letters as economy-wide labor quantities. The preamble makes the reuse explicit (`\manualLetter` and `\aggregateManualLabor` both expand to M). Footnote 21 uses c^M_i and c^A_i within a page of eq. (9), and OA.C carries tau_A alongside tau^A_b.
- **Proposed fix**: Rename the aggregate inputs to L_A and L_M in eq. (9) and throughout OA.C, leaving M and A as mode labels only.

### N5. t^A_i is prompting plus verification in Sections 3.1 and 4.2, verification only in Table OA.A.1, Definition 2 and Section 5.1
- **Location**: p. 10 (PDF 10) `3_shortrun.tex:21` and Definition 2 at `:35`; `OA_A_tables_and_figures.tex:34`; `5_longrun.tex:44`.
- **Issue**: Section 3.1 defines t^A_i as the time spent "prompting and verifying" one AI attempt and Section 4.2 repeats it; the other three places define it as verification only.
- **Why it's a problem**: If t^A includes prompting, charging a chain exactly t^A_r per attempt assumes prompting r-ell+1 steps costs the same as prompting one.
- **Proposed fix**: Adopt one wording everywhere, and if prompting is included add a clause to footnote 6 that a chain is assumed to be prompted at the cost of a single step.

### N6. The composite-chain footnote supplies no standalone counterpart for the collapsed successor, so Proposition 1's pricing does not carry over "unchanged"
- **Location**: p. 13 (PDF 13), `4_implications.tex:15`; arrangements V_0 to V_3 at `OA_B_omitted_proofs.tex:17-24`.
- **Issue**: The footnote collapses a successor chain spanning k+1, ..., r into one pseudo-step with two composite parameters (t^A_r and prod_{i=k+1}^{r} q_i). That suffices for eq. (2), which prices only the chain and the successor-alone chain. It does not suffice for Proposition 1, whose V_0 and V_3 price the successor at min{t^M_{k+1}, t^A_{k+1}/q_{k+1}}: substituting gives min{t^M_{k+1}, t^A_r/prod q_i}, whose first branch is the manual time of a single step while the second covers the whole run, leaving k+2, ..., r unpriced.
- **Proposed fix**: State a third composite parameter (the minimum cost of running k+1, ..., r under any arrangement not containing step k), or limit the footnote's claim to eq. (2) and drop "what follows applies unchanged".

### ✅ N7. The notes to Figure 5 point to "the table below", which is Table OA.A.3, thirty pages away
- **Status**: ✅ **Addressed** 2026-09-04 (commit `f9d4a4a`). The notes now cite `Table~\ref{tab:nonmonotone_costs}` in `Appendix~\ref{app:tables_and_figures}`; the semicolon before it became a full stop, since the clause no longer describes panel (a).
- **Location**: p. 21 (PDF 21), `4_implications.tex:235`; target `tab:nonmonotone_costs` on p. OA-3 (PDF 51).
- **Issue**: No table appears below or near Figure 5. The body text one sentence later cites the same object correctly as "Table~\ref{tab:nonmonotone_costs} in Appendix~\ref{app:tables_and_figures}".
- **Proposed fix**: Replace "the table below" with that `\ref`.
- **Corroboration**: found independently by three passes.

### N8. Figure 5 panel (a) plots four of the five configurations the text and Table OA.A.3 enumerate
- **Location**: p. 21 (PDF 21), notes at `4_implications.tex:235`, text at `:240`, table at `OA_A_tables_and_figures.tex:130-144`.
- **Issue**: The text says the firm chooses among five configurations and Table OA.A.3 lists five. `plots/example5_costs.png` has four strategy curves plus the envelope; the missing one is "Step 1 augmented, Step 2 manual", C = 3.5 alpha^{-11} + 8, which equals 11.5 at alpha = 1 and 19.2 at alpha = 0.9, both inside the plotted y-range of roughly 4.3 to 25.5. In `analysis/example5_plot.ipynb` the series is computed and stacked into the min but never passed to `plt.plot`.
- **Proposed fix**: Plot the missing curve, or reword the note to say panel (a) shows four of the five and name the omission.
- **Corroboration**: found independently by two passes.

### N9. Equation (8) equates a minimized value to a job-design-dependent expression
- **Location**: p. 26 (PDF 26), `5_longrun.tex:143-150`.
- **Issue**: The display reads min_T min_J TotalCost(J;T) = sum_{J_j in J} WageBill_j = ..., so a number is set equal to an expression whose free variable J the minimization has already bound.
- **Proposed fix**: Split the display. Write TotalCost(J;T) = sum_j [(sum c_b)(t^H(J_j) + sum t_b)] as a definition, then state the program separately, or attach the sum with "where" rather than "=".

### N10. Hand-off time is defined over steps and indexed by task in the Section 5.4 example, against the notation table's own convention
- **Location**: p. 27 (PDF 27) `5_longrun.tex:191-192, 217`; definition at `:112`; Table OA.A.1 at `OA_A_tables_and_figures.tex:54-55, 64`.
- **Issue**: Table OA.A.1 defines t^H_i on steps and its note states "Step-level primitives are indexed by i, task-level objects by b". The example writes (c_b, t_b, t^H_b) for b = 1, 2, 3 and reads t^H_1 = 3 as the hand-off after task 1. Since a task may be an AI chain spanning several steps, t^H_1 (step) and t^H_1 (task) are different objects, and t^H_b is never defined anywhere.
- **Proposed fix**: Either state that the example takes one step per task, or define t^H(T_b) as t^H at the final step of T_b and add that row to Table OA.A.1.
- **Corroboration**: found independently by three passes.

### N11. Figures 7 and 8 label the hand-off blocks h_1 and h_2, a symbol the model never defines
- **Location**: p. 27 (PDF 27), `5_longrun.tex:170-184, 196-214`; images `plots/job_design.png`, `plots/combined_grid_with_handoff.png`.
- **Issue**: The plots print h_1 and h_2 inside the pink rectangles while the model's hand-off time is t^H_i. Figure 7 uses both conventions in one panel (the block is labeled h_1 and its width t^H_1); Figure 8 has no t^H label anywhere. The symbol h appears in no .tex source and not in Table OA.A.1.
- **Proposed fix**: Regenerate the two PNGs with the blocks labeled t^H_1 and t^H_2, or add "h_b denotes the hand-off cost c_b t^H_b" to the figure notes.

### N12. The extensive margin is described as an AI adoption choice; in OA.C it is a produce-or-exit threshold
- **Location**: p. 30 (PDF 30), `6_extensions.tex:33`; `OA_C_CES_representation.tex:180-182`.
- **Issue**: In OA.C every producing firm has committed to the same T and J, so no firm chooses whether AI is worthwhile and there is no all-manual option in the second stage. The margin that moves with wages is the participation threshold alpha-bar >= w_A tau_A/(1 - w_M tau_M), above which firms produce and below which they exit.
- **Proposed fix**: "the set of firms for which production is profitable at all has moved".

### N13. The CES weights theta_A and theta_M are used in (9) and never defined in the body
- **Location**: p. 30 (PDF 30), `6_extensions.tex:22-26`.
- **Issue**: The gloss after (9) defines A, M, K, Y, rho and sigma but never theta_A and theta_M, nor that they are nonnegative with theta_A + theta_M < 1. They are defined only in OA.C and do not appear in Table OA.A.1, which is scoped to Sections 3-5. Footnote 22 then makes a claim about them.
- **Proposed fix**: Add to the gloss: "and CES weights theta_A, theta_M >= 0 with theta_A + theta_M < 1 on AI management labor and manual labor, the remainder falling on capital".

### N14. Section 6.1 glosses the CES inputs A and M as AI management labor and manual labor; in OA.C the same symbols are skill-adjusted aggregates
- **Location**: p. 30 (PDF 30), `6_extensions.tex:16-17, 26`; `OA_C_CES_representation.tex:12, 17, 169`.
- **Issue**: OA.C defines these inputs as skill-adjusted everywhere it introduces them, and the skill adjustment is not a units convention: by the definition of tau_b the weight on a task's time is the total compensation of the job the task belongs to divided by the base wage rate of its execution mode, so it moves with the chosen job design. "Skill-adjusted" appears in the entire main text exactly once, in footnote 22.
- **Proposed fix**: Say in the gloss after (9) that A and M are economy-wide aggregates of **skill-adjusted** AI management and manual labor, with the adjustment defined in OA.C and depending on the firm's job design.

### N15. Proposition 4's cost range [1/B, B] is empty unless B >= 1, but only B > 0 is assumed
- **Location**: p. 32 (PDF 32) `6_extensions.tex:95`; proof at `OA_B_omitted_proofs.tex:618-621`.
- **Issue**: [1/B, B] is nonempty only when B >= 1. The same slip appears in the proof, whose subsequent bounds (total skill in [1/B, mB], total time at most 2mB^3) also implicitly use B >= 1. For B < 1 the hypothesis is vacuous and the stated running time O(m^4 eps^{-2} log^2(mB)) is not meaningful, since log(mB) can be non-positive.
- **Proposed fix**: State the hypothesis as "for some B >= 1" in both the proposition and the proof.

### N16. The second placebo is said to isolate a distinct margin although it nests the first
- **Location**: p. 35 (PDF 35), `7_empirics.tex:62-63`.
- **Issue**: The preceding sentence defines placebo 2 as reassigning tasks across occupations "preserving each occupation's task count and randomizing task positions within each occupation", so it randomizes ordering as well as composition. It is placebo 1 plus a composition shuffle, not a separate margin.
- **Why it's a problem**: The claim that the two designs separate two margins supports the inference at `:96-98` that the result speaks separately to the GPT ordering and to the distribution of execution labels.
- **Proposed fix**: Describe placebo 2 as nesting placebo 1, i.e. a strictly more permissive benchmark.

### N17. The Table 2 note calls column (5) directly comparable with column (4), but the two differ in a control as well as in fixed effects
- **Location**: p. 38 (PDF 38), `7_empirics.tex:162`; `tables/noTasksWithRepetitiveDWAs/allTasks_ai.tex`.
- **Issue**: "DWA FE" is checked in columns (4) and (6); "NumTasks in DWA-Occupation Control" is checked in (5) and (6). Columns (4) and (5) therefore differ in two dimensions at once. The clean contrast on the restricted sample is (5) versus (6).
- **Why it's a problem**: The main-text inference at `:171-172` that the attenuation reflects DWA heterogeneity rather than the sample restriction is presented as a (4)-versus-(5) comparison.
- **Proposed fix**: "Column (5) shares the restricted sample of columns (4) and (6) and is the no-fixed-effects counterpart of column (6), so the pair (5)-(6) isolates the DWA fixed effects."
- **Corroboration**: found independently by two passes.

### N18. The EFI is described as one minus a share of adjacent pairs, but the divisor in the formula is m_w, not m_w - 1
- **Location**: p. 40 (PDF 40), footnote 27 at `7_empirics.tex:202`.
- **Issue**: By eq. (13), EFI_w = 1 - (k_w - r_w)/m_w, and k_w - r_w is the count of adjacent AI-able pairs. The subtracted term divides that count by the number of **steps**, whereas a workflow of m_w steps contains m_w - 1 adjacent positions.
- **Proposed fix**: "the EFI is exactly one minus the number of adjacent AI-able pairs per step, (k_w - r_w)/m_w", or normalize by m_w - 1 and change eq. (13).

### N19. The EFI footnote calibrates q_i in {0,1}, a value the model's own primitive excludes
- **Location**: p. 40 (PDF 40), footnote 27 at `7_empirics.tex:200`; Definition 2 at `3_shortrun.tex:35`; Table OA.A.1 at `OA_A_tables_and_figures.tex:28`.
- **Issue**: Definition 2 declares q_i = alpha^{d_i} in (0,1] and the notation table repeats alpha in (0,1]. No (alpha, d_i) with finite d_i delivers q_i = 0, so the footnote that bridges Proposition 2 to the empirical measure uses a parameterization half outside the model's parameter space.
- **Proposed fix**: State the calibration as a limit: t^M_i = t^A_i = 1, q_i = 1 for AI-exposed steps and q_i -> 0 for the rest, noting min{t^M_i, 1/q_i} -> 1 so the index is well defined in the limit.

### N20. The stated range for r_w is unattainable in 40% of the estimation sample and wrong at k_w = 0
- **Location**: p. 40 (PDF 40), `7_empirics.tex:212`.
- **Issue**: The number of maximal AI-able blocks obeys **r_w <= min(k_w, m_w - k_w + 1)**, not r_w <= k_w, since k_w isolated blocks need k_w - 1 separators. In the 872-occupation panel **348 occupations (39.9%)** have k_w > m_w - k_w + 1, so the stated upper endpoint is infeasible for them. The lower endpoint is also wrong for the 63 occupations with k_w = 0, where r_w = 0; the paper's own five-step example at `:198` requires exactly this, since EFI = 1 there.
- **Proposed fix**: Write 1 <= r_w <= min(k_w, m_w - k_w + 1) for k_w >= 1, and note r_w = 0 when k_w = 0.

### N21. The note to Table 3 says the raw index captures dispersion; the text one page earlier says it can be read that way only after conditioning
- **Location**: p. 41 (PDF 41) `7_empirics.tex:237`, against `:204`, `:213` and `:260`.
- **Issue**: The subsection says "Two components govern the empirical fragmentation index, and only one is the subject of Prediction #3" and "To read the index as a measure of dispersion, we must control for the share of AI-able steps". It then quantifies the gap: regressing the EFI on the AI-able share alone gives **R-squared = 0.91**, with about **30%** of its standard deviation surviving conditioning.
- **Proposed fix**: In the note, replace "the index itself captures how dispersed AI-exposed steps are across the workflow" with "conditional on that exposure share, the residual variation in the index is the arrangement of those steps (see eq. 13)".

### N22. The notation table's lead-in claims Panel A covers Section 4; Panel A's header and contents are Section 3 only
- **Location**: p. OA-1 (PDF 49), `OA_A_tables_and_figures.tex:11-12, 23`.
- **Issue**: Every symbol in Panel A (alpha, s_i, S, t^M_i, t^A_i, d_i, q_i, T_b, T, t_b) is introduced in Section 3, and the panel header reads "(Section 3: Short Run)". None of Section 4's notation appears: the failure set F, the component collection C, the weight omega(C), FI and OPT are all absent.
- **Proposed fix**: Either narrow the lead-in to Section 3, or add a Section 4 block with F, C, omega(C), FI and OPT and update the header.

### N23. Figure OA.A.1's histograms contain 999 reshuffles plus the observed fit, not 1,000 reshuffles
- **Location**: p. OA-4 (PDF 52) `OA_A_tables_and_figures.tex:202`; same count at `7_empirics.tex:175` (p. 39).
- **Issue**: The notebook writes `regression_ame_results_full_0.csv` from the **unshuffled** data in the preceding cell, then loops `for i in range(n_shuffles)` starting at i = 0. Only files i = 1..999 are genuine reshuffles, so the observed value is counted as one of its own placebos.
- **Proposed fix**: Drop i = 0 from the null and state 999 reshuffles, or generate a 1,000th shuffle; adjust the note and the main-text sentence accordingly.

### ✅ N24. The ratio in the Lemma OA.B.2 upper bound is maximized at alpha^{d} = 1/2, not alpha^{-d} = 1/2
- **Status**: ✅ **Addressed** 2026-09-04 (commit `f9d4a4a`). Now reads "maximized at $\alpha^{d(T_b)} = 1/2$ (equivalently $\alpha^{-d(T_b)} = 2$)". The tightness example below it builds a chain with $q_1q_2q_3 = 1/2$, which corroborates the corrected orientation.
- **Location**: p. OA-13 (PDF 61), `OA_B_omitted_proofs.tex:256`.
- **Issue**: With u = alpha^{d(T_b)} in (0,1], the ratio (1 + u^{-1} - u)/u^{-1} = u + 1 - u^2 is maximized at u = 1/2, i.e. alpha^{d(T_b)} = 1/2, equivalently alpha^{-d(T_b)} = 2. Since q_i in (0,1] forces alpha^{-d} >= 1, the stated maximizer lies outside the feasible range. The value 5/4 is correct.
- **Proposed fix**: "is maximized at alpha^{d(T_b)} = 1/2 (equivalently alpha^{-d(T_b)} = 2), achieving a value of 5/4".

### ✅ N25. The constant term of the fragmentation index in Example OA.B.2 is 0.5, not 1
- **Status**: ✅ **Addressed** 2026-09-04 (commit `f9d4a4a`). Constant changed from 1 to 0.5. Confirmed two ways: the closed form (12), and brute-force enumeration over all $2^m$ failure realizations, which agree at $FI = 0.62132\,m + 0.5$.
- **Location**: p. OA-15 (PDF 63), `OA_B_omitted_proofs.tex:364`.
- **Issue**: With q_i = 1/sqrt(2) and t^M_i = sqrt(2) for all i, FI = m(1-q)sqrt(2) + q + (m-1)q(1-q) = **0.62132 m + 0.5**. The additive constant is 0.5, not 1, so the displayed expression is wrong for every finite m (m = 10 gives FI = 6.7132, not 7.2132). Verified by exhaustive enumeration over failure realizations. The limiting ratio 4 sqrt(2)/(9(sqrt(2)-1)) = 1.5174 is unaffected.
- **Proposed fix**: Replace "1 + m x 0.6213" with "0.5 + m x 0.6213".

### ✅ N26. The post-threshold marginal benefit in Example 2 is 134.1, not 133.9
- **Status**: ✅ **Addressed** 2026-09-04 (commit `f9d4a4a`). 133.9 changed to 134.1. At $\alpha_0 = 0.9239886$, $48/\alpha_0^{13} = 134.145$; the old value was $48/0.9241^{13}$.
- **Location**: p. OA-19 (PDF 67), `OA_B_omitted_proofs.tex:506`.
- **Issue**: Solving 6 + 4/alpha = 4 alpha^{-12} (the two cost expressions in Table OA.A.3) gives alpha_0 = **0.9239886**. There 4/alpha_0^2 = 4.685 (matching the quoted 4.7) but 48/alpha_0^{13} = **134.145**, not 133.9. The printed value is 48/0.9241^13, i.e. the threshold was rounded to four decimals before substitution into a quantity of degree 13. The plotting grid lands on 0.9240570 and gives 134.016, so the plotted value is not 133.9 either.
- **Proposed fix**: Replace 133.9 with 134.1, or state the threshold as alpha_0 = 0.92399 and the jump as 4.69 to 134.15. The 0, 16.0 and 4.7 figures are correct.
- **Corroboration**: found independently by two passes.

### ✅ N27. Job compensation is stated without base wage rates, contradicting Eq. (OA.C.1) four lines later
- **Status**: ✅ **Addressed** 2026-09-04 (commit `f9d4a4a`). Job 1's compensation now reads $w_M c^M_1 + w_A c^A_2$, the two-task instance of (OA.C.1).
- **Location**: p. OA-25 (PDF 73), `OA_C_CES_representation.tex:25`.
- **Issue**: w_M and w_A are introduced two paragraphs earlier and (OA.C.1) gives compensation as w_M(sum c^M_b) + w_A(sum c^A_b). For Job 1 that is w_M c^M_1 + w_A c^A_2, not c^M_1 + c^A_2. The sentence reproduces the Section 5 wage, which normalizes w_A = w_M = 1, inside the one subsection whose purpose is to un-normalize them.
- **Proposed fix**: Change to "... and equals w_M c^M_1 + w_A c^A_2".

### N28. "Skill-adjusted labor" is announced as defined, but only skill-adjusted **time** is defined, and l_b's units are never fixed
- **Location**: p. OA-25 (PDF 73), `OA_C_CES_representation.tex:39`, `:44-50`, first use at `:84`.
- **Issue**: The paragraph defines only tau_b. The labor variable l_b is never defined and is described at first use merely as "the amount of labor assigned to task b". Yet the accounting requires l_b in skill-adjusted units, since the denominators in (OA.C.2) are skill-adjusted times and the wage bill is w_A l_A + w_M l_M with w_A, w_M described as rates "for a unit of skill-adjusted" labor. If l_b were raw worker hours, (OA.C.2) would be dimensionally inconsistent.
- **Proposed fix**: Add the promised sentence after tau_b is defined: "l_b denotes skill-adjusted labor devoted to task b, so that its cost is w_{E(b)} l_b", and change the description at `:84`.

### N29. Appendix OA.C attaches task subscripts to the step-level primitives c^M, c^A, t^M, t^A
- **Location**: p. OA-25 (PDF 73), `OA_C_CES_representation.tex:25, 32, 42, 46`; convention stated at `OA_A_tables_and_figures.tex:64`.
- **Issue**: These four primitives are step-level everywhere else in the draft, and Table OA.A.1's note states "Step-level primitives are indexed by i, task-level objects by b, and jobs by j", giving the task-level aggregates their own symbols t_b and c_b (with c_b = c^A_r for a chain augmented at step r). OA.C is the only place that breaks it, writing c^M_b, c^A_b inside (OA.C.1), t^M_b, t^A_b at `:42`, and summing c^M_ell, c^A_ell over tasks T_ell four lines after ell denoted a chain's first step. For a chain longer than one step, b and the endpoint r(b) are different numbers.
- **Proposed fix**: State the map once at the head of OA.C, or write c^A_{r(b)} and t^A_{r(b)} and say r(b) denotes task b's augmented endpoint. Also rename the task-summation index off ell.
- **Corroboration**: found independently by two passes.

### N30. The two-input Leontief (OA.C.4) is the efficiently-allocated envelope of (OA.C.2), not the same function
- **Location**: p. OA-27 (PDF 75), `OA_C_CES_representation.tex:91-103`.
- **Issue**: By the mediant inequality, min_b l_b/a_b <= (sum l_b)/(sum a_b) with equality only when labor is allocated within each group in fixed proportions. (OA.C.4) is therefore the value of (OA.C.2) **maximized over within-group allocations**, which the text never says. Relatedly, the premise offered for (OA.C.3) is a property of the equilibrium **allocation** while the conclusion is a property of the production **function**.
- **Proposed fix**: State the reduction directly, that for any (l_A, l_M) the max of (OA.C.2) over within-group allocations equals min{alpha-bar l_A/tau_A, l_M/tau_M}, and add that (OA.C.4) is the aggregate function over efficiently allocated group inputs.

### N31. The boundary check in OA.C.3 is justified by a constant of integration that never appears in the derivation
- **Location**: p. OA-34 (PDF 82), `OA_C_CES_representation.tex:317`.
- **Issue**: The displayed Gamma(u) is not obtained by integrating Gamma' = -phi. It is obtained algebraically: (OA.C.20) is differentiated to express Psi^{rho-1} in terms of Gamma^{rho-1} u, that expression is substituted back into the undifferentiated (OA.C.20), and the result is solved for Gamma(u). No constant of integration enters. The real reason the boundary check is needed is that differentiating discards information, so the pair (Gamma, Psi) solving the combined system need not satisfy Gamma(1) = Psi(1) = 0.
- **Proposed fix**: "Differentiating (OA.C.20) is necessary but not sufficient, so the Gamma obtained by combining the differentiated and undifferentiated equations must still be checked against the boundary condition Gamma(1) = 0 that (OA.C.18) imposes by construction."

### N32. SA.A says Section 7 lists four data sources; Section 7 lists five
- **Location**: p. SA-1 (PDF 85), `SA_A_sample_construction.tex:4`.
- **Issue**: `7_empirics.tex:11` reads "We draw on five sources:" and enumerates five, the fifth being the APQC Process Classification Framework; `:19` then says "The first four build our main sample".
- **Proposed fix**: "from the first four of the five sources listed in Section 7" (the fifth, APQC, documented in Appendix SA.F).

### N33. Figure SA.A.1's bars sum to 17,920 but the figure states a total of 17,925
- **Location**: p. SA-2 (PDF 86), `SA_A_sample_construction.tex:33` and the figure at `:36-49`.
- **Issue**: The printed figure reads Manual **15,573 (86.9%)**, Augmentation 1,626 (9.1%), Automation 721 (4.0%) with an inset "Total Tasks: 17,925". The bars sum to **17,920**, five short, and the percentages use the larger denominator. The Manual count in the estimation sample is **15,578**. The cause is a `drop_duplicates(subset=['Task ID'])` in the plotting cell applied to the numerator but not the denominator, and the five duplicates are the doubly-positioned tasks of D22.
- **Proposed fix**: Use one unit throughout. Either count occupation-task records (15,578 / 1,626 / 721 against 17,925) or count distinct tasks and set the total to 17,920.
- **Corroboration**: found independently by two passes.

### N34. Fully filtered tasks are recoded as manual, not excluded, so the stated power justification does not describe what was done
- **Location**: p. SA-3 (PDF 87), `SA_A_sample_construction.tex:80-82`.
- **Issue**: One sentence earlier the appendix says these tasks are labeled manual, and `:32` confirms they are kept. They are therefore retained with a known-wrong execution label, not lost. Losing observations and misclassifying the outcome are different problems.
- **Why it's a problem**: The paragraph's conclusion that the estimates are "a conservative signal of potentially much stronger effects" needs an attenuation argument about non-differential misclassification, not a sample-size argument.
- **Proposed fix**: Replace "excluding them limits our statistical power" with "recoding them as manual introduces measurement error in the execution indicator, which attenuates the associations we estimate", and note that the misclassification is plausibly unrelated to workflow position.

### N35. The notes to Figures SA.A.3 and SA.A.4 overstate the AI-executed share of the two example occupations
- **Location**: p. SA-6 to SA-7 (PDF 90-91), `SA_A_sample_construction.tex:137, 155`.
- **Issue**: Computer Programmers (15-1251.00) has 17 tasks, **10** labeled Augmentation or Automation, i.e. **59% (10/17)**, not "about two thirds" (which would be 11 or 12). Public Relations Specialists (27-3031.00) is **7 of 18, i.e. 39%**, not "about one third".
- **Proposed fix**: "about three fifths (10 of 17 tasks)" and "about two fifths (7 of 18 tasks)", or drop the fractions and state the counts.

### N36. The GPT-filtered estimates do not "closely mirror" the baseline in magnitude at two positions away
- **Location**: p. SA-9 (PDF 93) `SA_B_alternative_definitions.tex:31`; same claim at `7_empirics.tex:118` (footnote 26).
- **Issue**: Main sample column (1): (k-2) **0.07\*\*\***, (k-1) 0.12\*\*\*, (k+1) 0.12\*\*\*, (k+2) **0.05\*\*\***. GPT-filtered column (1): (k-2) **0.04 (p = 0.120)**, 0.11\*\*\*, 0.11\*\*\*, (k+2) **0.01 (p = 0.609)**. Unrounded, 0.0651 -> 0.0449 and 0.0495 -> 0.0142, so the k+2 estimate falls to 29% of its baseline and both go from 1% significance to insignificant. Section 7.2 reads the baseline as showing an effect "at one and at two positions away alike, with the effects of the adjacent steps roughly twice the size"; that ratio is not preserved.
- **Proposed fix**: "the immediate-neighbour estimates are essentially unchanged under the stricter similarity filter, while the two-positions-away estimates shrink toward zero and lose significance". Same edit in the main-text footnote.
- **Corroboration**: found independently by two passes.

### N37. The similarity-filter criteria described in SA.B do not match Prompt #2 in SA.C
- **Location**: p. SA-9 (PDF 93), `SA_B_alternative_definitions.tex:26`; prompt at `SA_C_gpt_prompts.tex:27-60`.
- **Issue**: Prompt #2 asks the model to determine which tasks are similar "in nature and in terms of their objectives, methods, or required skills". "Execution complexity" is not one of its criteria. The prompt also imposes a restriction the text never mentions ("Return only the most relevant task for every occupation"), which is a sample restriction rather than a similarity criterion and part of why the filtered sample falls to 3,689 tasks (see D29).
- **Proposed fix**: Describe the filter with the prompt's own criteria, and state that at most one task per occupation is retained within each DWA.

### N38. The automation attenuation is attributed to prevalence, but the estimating sample also changes
- **Location**: p. SA-10 (PDF 94), `SA_B_alternative_definitions.tex:38` with the footnote at `:36`.
- **Issue**: The automation regressions are run on the random-tie-break sample, which retains tasks mapped to multiple DWAs and assigns each to one at random, giving **13,786** observations in Table SA.B.2 against **10,708** in Table 2, and **5,156** in Table SA.B.3 against **3,689** in Table SA.B.1. Section 7.2 defines the main sample by dropping exactly those tasks. No `is_ai` benchmark on the tie-break sample is reported.
- **Proposed fix**: Add a column or table reporting the `is_ai` regression on the same random-tie-break sample and read the attenuation against that benchmark.

### N39. The notes to Figures SA.B.2 and SA.B.3 describe the regressors as neighbours' automation status; they are AI execution
- **Location**: p. SA-14 (PDF 98), `SA_B_alternative_definitions.tex:174`, by reference `:217-219`.
- **Issue**: Only the dependent variable changes in this exercise (`:34`, `:176`). The regressors remain `prev2_is_ai`, `prev_is_ai`, `next_is_ai`, `next2_is_ai`, as the figure caption, the table row labels and the source CSV term names all show.
- **Proposed fix**: "those sitting in a more AI-**executed** local context exhibit higher probabilities of being AI-automated".
- **Corroboration**: found independently by two passes.

### N40. Table SA.B.5 reports clustered standard errors, unlevelled, while the main-text table it claims to mirror reports robust ones
- **Location**: p. SA-17 (PDF 101), `SA_B_alternative_definitions.tex:275, 284`.
- **Issue**: The note says "Clustered standard errors in parentheses" without naming the cluster level, in a regression with one observation per occupation, while Table 3 reports "Heteroskedasticity-robust standard errors" for the same equation on the same 872 occupations, and `:284` claims the specification mirrors it exactly.
- **Proposed fix**: Make the description match the main table, or name the cluster level.

### N41. Prompt #1 elicits the "typical" order; the text describes it as the "most reasonable" order
- **Location**: p. SA-19 (PDF 103), `SA_C_gpt_prompts.tex:13-14`, against `7_empirics.tex:31` and `SA_A_sample_construction.tex:73`.
- **Issue**: Prompt #1 asks for the descriptive modal order ("the typical sequential order in which these tasks are performed in a real-world workflow"). Both descriptions say the model is asked for "the most reasonable order". The paper itself treats descriptive and normative framings as distinct, since SA.D's alternative #3 asks to "minimize rework, waiting, and unnecessary handoffs" and alternative #10 asks for "how the work is most commonly carried out in practice".
- **Proposed fix**: Change both descriptions to match Prompt #1's own wording.

### N42. Three different descriptions of what Prompt #2 asks, none matching the prompt
- **Location**: p. SA-20 (PDF 104), `SA_C_gpt_prompts.tex:42-43` versus `SA_B_alternative_definitions.tex:24, 26, 54, 89`.
- **Issue**: The prompt asks for tasks similar "in nature and in terms of their objectives, methods, or required skills". SA.B gives "skill requirements and execution complexity" (`:26`), "similar execution nature and skill characteristics" (`:54`, `:89`) and "objectives, execution nature, or required skills" (`:24`). "Execution complexity" and "methods" are neither elicited nor mentioned in the prompt.
- **Proposed fix**: Use the prompt's own wording at `:26` and repeat that identical phrase in the notes to both tables.

### N43. Figure SA.D.2's dashed line is the alternative-prompt mean in the text and the 11-prompt mean in the notes
- **Location**: p. SA-23 (PDF 107), `SA_D_prompt_robustness.tex:118` versus `:131`.
- **Issue**: The two quantities are **1.44927** (ten alternatives) and **1.44906** (all eleven), against a main-prompt value of **1.44698**, so no printed number changes, but the same line is defined two ways. If the plotted line is the eleven-prompt mean it contains the main-prompt value it is being compared to.
- **Proposed fix**: Make the body match the notes ("the mean across all eleven prompts is 1.449, essentially the main-prompt value of 1.447"), or plot and describe the ten-prompt mean in both places.

### N44. The worked frequency-cut example does not pin down the Daily+ share, so the 65% conclusion does not follow
- **Location**: p. SA-28 (PDF 112), `SA_E_frequency_robustness.tex:30-32`.
- **Issue**: The example stipulates only that 60% of workers report the task "several times daily", "with the remaining responses spread over less frequent categories". Under SeveralDaily+ the frequent share is exactly 60% and failure at 65% follows. Under Daily+ the frequent share is 60% plus the "daily" share, and "daily" is one of the less frequent categories the stipulation permits, so the Daily+ share can be anywhere in [60%, 100%]. An even spread of the remaining 40% over the six lower categories gives about 6.7% daily, which clears 65%.
- **Why it's a problem**: The sentence claims failure at 65% under both remaining logics, and the preceding sentence says "Nothing stricter retains it, in either direction". This is the paper's only worked illustration that the cuts nest.
- **Proposed fix**: Pin down the residual ("with the remaining 40% spread over categories no more frequent than more than weekly"), or restrict the last clause to the SeveralDaily+ logic.
- **Severity note**: medium in the per-file pass, minor in the number pass; minor taken (an unstated stipulation in an illustrative example, with no estimate downstream).

### N45. The Allergists example's dropped and retained step lists account for only 13 of the 16 tasks
- **Location**: p. SA-30 (PDF 114), `SA_E_frequency_robustness.tex:68`.
- **Issue**: Panel (b) keeps positions **1, 2, 5, 6, 7, 11, 12** and therefore drops **nine** steps: 3, 4, **8**, 9, 10, **13**, 14, 15, 16. The sentence lists seven drops, omitting step 8 ("Assess the risks and benefits of therapies", 41.40% hourly-or-more) and step 13 ("Provide therapies ... to treat immune conditions", 45.81%). The retained list names only six of the seven survivors, omitting step 6 ("Diagnose or treat allergic or immunologic conditions", 64.72%).
- **Why it's a problem**: The rhetorical point is "what the filter removes is what one would expect it to remove", but two core clinical steps are removed and left off the list. The next sentence also relies on step 8 being dropped ("once the intervening manual steps 8 through 10 are dropped").
- **Proposed fix**: State the full pruned set and add "diagnosing or treating" to the retained clinical loop.
- **Corroboration**: found independently by two passes.

### N46. The chain-length range 1.24 to 1.31 excludes Daily+ >= 20% (1.35), which is not an Hourly+ corner
- **Location**: p. SA-30 (PDF 114), `SA_E_frequency_robustness.tex:101`.
- **Issue**: The sweep gives Daily+ >= 20% = **1.354**, printed as 1.35 in the forest plot's second row. That value lies above the stated band while not being one of the "two sparsest Hourly+ corners" (1.40 and 1.50). The remaining nine pruned cuts do fall in 1.24 to 1.31.
- **Proposed fix**: "to between 1.24 and 1.35 across the remaining cuts, with the largest values in the two sparsest Hourly+ corners (1.40 and 1.50, with 75 and 20 occupations)".

### ✅ N47. The neighbor heatmap note defers its significance-star convention to a figure that shows no stars
- **Status**: ✅ **Addressed** 2026-09-04 (commit `f9d4a4a`). The star convention is now stated in place (*** p<0.01, ** p<0.05, * p<0.1, from the DWA-clustered coefficient test, matching the generator). The deferral to Figure SA.E.4 now covers layout only.
- **Location**: p. SA-32 (PDF 116), `SA_E_frequency_robustness.tex:133`.
- **Issue**: The referenced figure is the fragmentation heatmap two subsections later, whose own note says "No cell in the grid is significant at the 10% level, so no significance stars are shown". The neighbor heatmaps do carry stars, and the cutoffs (1/5/10% in the code) are never stated in the appendix.
- **Proposed fix**: State the convention in place and refer only to the layout of the other figure.

### N48. The Hourly+ >= 65% fragmentation cells are described as unestimable, but they are estimated (+0.95, +1.01, +0.22)
- **Location**: p. SA-34 (PDF 118), `SA_E_frequency_robustness.tex:193`.
- **Issue**: The matched sweep reports EFI coefficients of **+0.95 (p = 0.126), +1.01 (p = 0.107) and +0.22 (p = 0.725)** on the 20 occupations of that cut. The plotting script's own comment gives the operative reason, that those values "would otherwise set the colour scale for the whole panel". The stated parallel is also inexact, since the neighbor heatmaps genuinely have no estimate for that cut while the chain-length figure reports it (1.50).
- **Why it's a problem**: The three suppressed cells are the largest positive coefficients in the grid and are excluded from the summary that follows ("negative in 13 of 33, median absolute estimate 0.05").
- **Proposed fix**: State the actual reason and the actual numbers, and say the cell is excluded from the summary counts.

### ✅ N49. The text says the fragmentation heatmap prints significance stars; the figure's own note says none are shown
- **Status**: ✅ **Addressed** 2026-09-04 (commit `f9d4a4a`). The text now says the number of occupations is printed beneath each cell, with no stars since no cell is significant at the 10% level. Consistent with the fix to N47.
- **Location**: p. SA-35 (PDF 119), `SA_E_frequency_robustness.tex:201` versus `:193`.
- **Issue**: The rendered heatmap prints only "coef" and "N=" in each of its 36 cells.
- **Proposed fix**: "with the number of occupations printed beneath (no cell is significant at the 10% level, so no stars appear)".

### N50. The same prompt is "Prompt 0" in the SA.F table and "Prompt #1" in the SA.F text, and the two numbering schemes collide
- **Location**: p. SA-39 (PDF 123), `tables/apqc_pcf_ordering_validation.tex:7` versus `SA_F_external_validation.tex:108`; schemes at `SA_C_gpt_prompts.tex:4-27` and `SA_D_prompt_robustness.tex:9-20, 129`.
- **Issue**: SA.C numbers two functionally different prompts as #1 (ordering, the baseline) and #2 (similarity filter). SA.D indexes the eleven ordering prompts 0-10 with "Prompt 0 corresponds to the baseline prompt used in the main text", and all three SA.D figures print a "GPT Prompt" axis running 0-10. So "Prompt 1" is the baseline under SA.C and the first alternative under SA.D, and "Prompt 2" is the similarity filter under SA.C and the input-output-logic alternative under SA.D. SA.F uses both conventions on one page.
- **Proposed fix**: Keep SA.C's "Prompt #1 (ordering)" and "Prompt #2 (similarity)" as prompt names, relabel SA.D's eleven variants as "Ordering variant 0" through "Ordering variant 10", and change the APQC table row to "Main prompt (Prompt #1 of Appendix SA.C)".
- **Corroboration**: found independently by two passes.

### ✅ N51. The "Deliver Services" category mean tau is reported as 0.67; the source gives 0.66
- **Status**: ✅ **Addressed** 2026-09-04 (commit `f9d4a4a`). 0.67 changed to 0.66 (`tau_main` = 0.664683). The other four values in the sentence round correctly to their sources.
- **Location**: p. SA-40 (PDF 124), `SA_F_external_validation.tex:169`.
- **Issue**: `category_summary.csv` gives tau_main = **0.664683** for category 5, which rounds to 0.66. Recomputed from `ordering_accuracy.csv` as the unweighted mean over the 12 category-5 branches, the same value. The other four category figures quoted in the sentence (0.69, 0.65, 0.25, 0.27) match their source to two decimals.
- **Proposed fix**: Change 0.67 to 0.66.

### N52. The mass at tau = -1 is not entirely three- and four-step branches
- **Location**: p. SA-41 (PDF 125), `SA_F_external_validation.tex:175`.
- **Issue**: Four branches sit at tau = -1: 9.6 "Process accounts payable and expense reimbursements" (n = 3), 10.1.4 "Manage facilities operations" (n = 3), 1.1.5.3 "Analyze deal options" (n = 4) and **13.9.3 "Monitor and manage EHS program" (n = 6**, stored as -0.9999999999999999, a full reversal of all 15 pairs). Same float-equality artifact as D42.
- **Proposed fix**: Soften to "consists almost entirely of three- and four-step branches", or name the exception.
- **Corroboration**: found independently by two passes.

### ✅ N53. The pooled PCF corpus uses seventeen industry-specific frameworks, not sixteen
- **Status**: ✅ **Addressed** 2026-09-04 (commit `f9d4a4a`). "sixteen" changed to "seventeen" in SA.F and in the docstrings of `analysis/apqc_industry_leaf_matching.py` and `analysis/apqc_pooled_predictions.py`. Re-ran the loader's own file selection: 18 frameworks, Cross-Industry plus 17. The Telecommunications PCF is a `.xls` and so never matches the `K*.xlsx` glob.
- **Location**: p. SA-45 (PDF 129), `SA_F_external_validation.tex:341`.
- **Issue**: The matched-step file behind the 13,482 steps and 525 process groups carries **18** distinct framework labels, CrossIndustry plus **17** industry frameworks (Aerospace and Defense, Airline, Automotive, Banking, Broadcasting, City Government, Consumer Electronics, Consumer Products, Education, Health Insurance Payor, Healthcare Provider, Life Sciences, Petroleum Downstream, Petroleum Upstream, Property and Casualty Insurance, Retail, Utilities).
- **Why it's a problem**: Column (6) of Table 3 is a framework fixed effect, so the count is the number of fixed effects absorbed.
- **Proposed fix**: Change "sixteen" to "seventeen", here and in the docstrings of `analysis/apqc_industry_leaf_matching.py` and `analysis/apqc_pooled_predictions.py`, which repeat the count.
- **Corroboration**: found independently by two passes.

### N54. Columns (4)-(6) of the PCF fragmentation table are called three fixed-effect specifications although column (4) has none
- **Location**: p. SA-46 (PDF 130), `SA_F_external_validation.tex:368` and figure note at `:385`.
- **Issue**: In `tables/fragmentation_index_regression_combined.tex` the "Fixed Effect" row for columns (4)-(6) reads blank, "PCF Category", "Framework". Only two of the three carry fixed effects, as the same sentence implicitly concedes with "without fixed effects".
- **Proposed fix**: "under each of the three specifications of columns (4)-(6) (no fixed effects, PCF Category, Framework), reaching the 5% level at eight of the eleven floors in the specification without fixed effects."

### ✅ N55. The chain-length z is 6.6 in the text and 6.2 on the figure printed on the same page
- **Status**: ✅ **Addressed** 2026-09-04 (commit `f9d4a4a`). 6.6 changed to 6.2, matching `chain_z` = 6.191137 in the stored 0.71 sweep row and the annotation on Figure SA.F.3. The companion figures (1.09, 0.01, 1.1627) already matched that row.
- **Location**: p. SA-47 (PDF 131), `SA_F_external_validation.tex:397-399` and Figure SA.F.3.
- **Issue**: The text gives z = **6.6** for the within-group step-order reshuffle at the 0.71 floor. Figure SA.F.3, on the same page immediately above the paragraph, annotates the 0.71 point as **z = 6.2**, computed deterministically from the stored sweep row (chain_z = 6.1911). Neither pipeline reproduces 6.6; the pooled script's own seed gives 6.06. The observed 1.1627, null mean 1.0878 and null sd 0.0121 quoted in the text all match the stored run.
- **Proposed fix**: Quote the value the figure is drawn from (z = 6.2), or regenerate the figure from the draw the text reports.

### N56. Cross-corpus agreement, not the distance from the 0.5 null, is billed as the strongest evidence of sequence over topic
- **Location**: p. SA-48 (PDF 132), `SA_F_external_validation.tex:430-432`.
- **Issue**: That the documented benchmark gives 0.700 and the observed benchmark 0.679 on adjacent pairs shows the measured accuracy is stable across two corpora. Stability alone does not discriminate sequence from topic, since an order-symmetric, topically driven procedure could deliver similar accuracies in both. What discriminates is that both sit far above the 0.5 null, because topical clustering is order-symmetric and predicts exactly 0.5. The two preceding sentences already make that argument, so the superlative is attached to the wrong statistic.
- **Proposed fix**: "The distance from the 0.5 null in both corpora is what rules out topical clustering, which is order-symmetric; the agreement of the two benchmarks to within two percentage points shows in addition that the measured accuracy is not specific to one corpus."

## Appendix: unverified leads

Everything in the body above survived the adversarial refute pass. The items here did not go through it and are **NOT independently verified**. Treat them as things to check, not as findings.

1. **Five table sources in `writeup/tables/` are not `\input` by any .tex file.** From the mechanical pre-check: `apqc_fragmentation_index_regression.tex`, `apqc_fragmentation_index_regression_exposure.tex`, `fragmentation_index_regression_E1E2control.tex`, `fragmentation_index_regression_exposure.tex`, `eventlog_example_case.tex`. Three of them carry the "(Definition 1)" / "(Definition 2)" EFI row labels flagged in D28, and one is an E1|E2-controlled fragmentation regression of exactly the kind M2 says is missing. **What to check**: whether any prose in SA.F or Subsection 7.3 quotes a number that comes from one of these tables without displaying the table. A quoted result with no displayed exhibit would be a major finding; unused table sources left over from an earlier draft are not an error at all. No pass established which of the two this is.

2. **The OA.B proof audit did not run.** As stated in Scope and method, the dedicated line-by-line re-derivation of the proofs of Propositions 1-4 and Lemmas OA.B.1-OA.B.4 was cut. The OA.B and OA.C entries above (D3, D10, D18, D19, D20, N24 to N31) come from the per-file read, which re-derived the OA.C.3 algebra chain by hand and checked several OA.B examples numerically but did not audit every proof step. **The absence of further OA.B findings is not evidence that the proofs are clean.**

3. **Unused figure files.** 63 distinct plot paths are referenced against 88 files in `plots/`. Unused figure files are not errors and were not investigated, but the stale-exhibit problems in D32 and N33 both trace to a PNG that predates the CSV behind it, so a systematic timestamp check of every referenced PNG against its generating CSV would be worth one pass.
