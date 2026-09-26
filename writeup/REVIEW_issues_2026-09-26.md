# Review of "Chaining Tasks, Redefining Work: A Theory of AI Automation" (2026-09-26)

Draft reviewed: `writeup/0_main.tex` and every file it inputs (Sections 1 to 8, Online Appendix A to C, Supplementary Appendix SA.A to SA.F, `tables/`), at commit `792c8c1`. Compiled fresh from those sources: 129 pages, 0 LaTeX errors, 0 undefined references or citations. Page numbers are printed page labels: body `p. N` (PDF page N), Online Appendix `p. OA-n` (PDF page n+46), Supplementary Appendix `p. SA-n` (PDF page n+81).

Scope, as requested: internal consistency, text against exhibits, and mistakes. Proofs were not checked. Grammar and style are in `PROSE_issues.md`. Items marked *(carried over)* were raised in the 2026-09-04 audit and are still in the text. Each one was re-checked against the current source.

**Counts: 1 submission-compliance item, 1 Major, 11 Medium, 32 Minor.**

---

## Submission compliance

### C1. The abstract is 151 words, over ReStud's 150-word limit
- **Location**: p. 1 (`0_main.tex:71-75`)
- **Issue**: Commit `5da9f6a` trimmed the abstract to 147 words for the limit. The rewrite in `df31bc2` brought it back to 151 (hyphenated compounds counted as one word).
- **Proposed fix**: Delete "just" in "beyond just how many there are" (150). Cutting "existing" in "existing automation patterns" as well leaves some margin (149).

---

## Major

### M1. Prediction #3 is reported as a general finding, but it holds only in the PCF sample *(carried over; the Sept 4 fix has regressed)*
- **Location**: p. 5 (`1_introduction.tex:94`), p. 7 (`1_introduction.tex:137-138`), p. 42 (`8_conclusion.tex:17`), p. SA-33 (`SA_E_frequency_robustness.tex:212`). The abstract (`0_main.tex:75`) is borderline.
- **Issue**: Table 3 gives O*NET fragmentation coefficients of -0.01, -0.09 and -0.04 (s.e. 0.09 to 0.10), none significant. The PCF gives -0.35\*\*\*, -0.26\*\* and -0.34\*\*\*. Section 7.3, SA.D (30 of 33 cells negative, none significant at 5%) and SA.E (13 of 33 negative, none significant at 10%) all describe the O*NET result as a null. The summaries do not:
  - Intro, p. 7: "Controlling for the share of steps exposed to AI, workflows whose AI-exposed steps are more dispersed ... execute a lower share of their steps with AI." No sample is named.
  - Intro, p. 5: "jobs with higher fragmentation see a weaker translation". The PCF unit is a process group, not a job.
  - Conclusion, p. 42: "Using a task-level dataset ... workflows whose AI-exposed steps are more fragmented exhibit lower realized AI execution conditional on exposure." The dataset is singular and the PCF is never mentioned.
  - SA.E, p. SA-33: "all three implications of our model appears to operate among the subset of frequently-performed tasks". Two paragraphs earlier, the same appendix says "Pruning does not recover the relationship."
- **Why it's a problem**: The paper's own tables contradict its headline empirical claim. A referee who reads Section 7.3 and then the conclusion will see this immediately.
- **Proposed fix**: Name the sample every time Prediction #3 is summarized. For example: intro p. 7, "In practitioner-documented business processes, workflows whose AI-exposed steps are more dispersed ... ; on O*NET occupations the estimate has the predicted sign but is imprecise." Conclusion: "Using O*NET tasks and APQC process data ... and, in documented business processes, that workflows whose ...". SA.E close: "The chain-length and neighbor patterns survive among frequently performed tasks, and the O*NET fragmentation null is unchanged." The abstract could say "... cluster in documented business workflows" if the word budget allows.

---

## Medium

### D1. SA.F still presents the PCF as a validation-only corpus, which contradicts Section 7 and SA.F.2
- **Location**: p. SA-34 (`SA_F:4-7`), p. SA-46 (`SA_F:445-446`), p. SA-47 (`SA_F:449-453`)
- **Issue**:
  - The opening says "The task sequences underlying our empirical analysis are not observed ... Every adjacency-based object in the paper rests on that imputation." The PCF results do not rest on an imputed order.
  - The limitations paragraph says "neither the PCF nor the event logs is crosswalked to O*NET tasks." SA.F.2 does exactly that crosswalk, by nearest-task embedding match.
  - "Third, the PCF serves as a validation sample rather than as the main sample because ... There is no corresponding measure for PCF process elements, and constructing one would mean generating the very labels ..." SA.F.2 constructs that measure, and Section 7 (`7_empirics.tex:20, 35-40, 271`) puts the two samples on equal footing.
- **Why it's a problem**: These passages predate the equal-footing restructure (`f8a7297`). They now contradict the main text and the next subsection of the same appendix.
- **Proposed fix**: Rewrite the opening as "The O*NET sequences are not observed ...". In limitation 1, say the ordering procedure is validated on corpora whose own orderings are not O*NET occupations. Replace limitation 3 with the actual caveat: PCF labels are transferred by a lossy embedding match (15% of steps clear the floor), so the PCF sample is thin in AI content.

### D2. SA.E's Prediction #2 paragraph misreads Figure SA.E.3 *(carried over)*
- **Location**: p. SA-29 to SA-30 (`SA_E:142-144`)
- **Issue**:
  - "The top row reproduces the full-sample adjacent-step effects of +0.12 under no fixed effects". The figure's top row shows 0.113 (k-1) and 0.105 (k+1), and Table 2 col. (1) shows 0.11 and 0.11. The next sentence says the SeveralDaily+ ≥50% effect "rises to +0.12", which is no rise if the baseline were +0.12.
  - "between +0.04 and +0.06 once SOC fixed effects" should be 0.05 to 0.06 (0.046 to 0.059). The value 0.04 appears only with DWA fixed effects.
  - "often grows in magnitude as the workflow is restricted to frequent tasks". Every Daily+ cell is smaller than the all-tasks row: no-FE k-1 is 0.070 to 0.092 against 0.113, SOC-major 0.031 to 0.049 against 0.059, and SOC-minor 0.028 to 0.037 against 0.050. Growth appears only in the sparsest cells (N = 41 to 574).
  - "The estimates weaken and lose significance only in the sparsest cells ... low hundreds". The opposite holds. The sparsest cells carry the largest and mostly significant estimates (0.279\*\*\*, 0.328\*\*\*, 0.283\*\*, 0.194\*\*, 0.226\*\*). Significance is lost in mid-sized cells: k-1 Daily+ ≥50% SOC-minor, N = 2,690, 0.028; k+1 SeveralDaily+ ≥20% SOC-minor, N = 2,533, 0.014.
- **Proposed fix**: Rewrite the paragraph from the figure. The effect stays positive in every estimable cell, shrinks under the inclusive cuts, and is noisy in the mid-sized SeveralDaily+ and Hourly+ cells.

### D3. SA.E overstates how well Prediction #1 and #2 survive pruning *(carried over)*
- **Location**: p. SA-29 (`SA_E:103-104`), p. SA-32 (`SA_E:176-177`), and the matching summary in Section 7, p. 34 (`7_empirics.tex:45`, "leaves our conclusions unchanged")
- **Issue**:
  - Figure SA.E.2 has 8 of the 12 pruned cuts inside the 10-90 null band (blue dots). The text says "The result survives pruning" and that it "falls back toward the middle of the null only in the Hourly+ cuts", while the same sentence quotes the 58th percentile for SeveralDaily+.
  - For the neighbor placebo (Figure SA.E.4), the text says the effects retreat to the middle "only at the sparsest cuts, the Hourly+ ≥35% and ≥50% cuts and SeveralDaily+ ≥35%". SeveralDaily+ ≥35% (315 occupations) is not sparser than SeveralDaily+ ≥50% (178) or ≥65% (76), which sit at the 82nd to 98th percentiles. Hourly+ ≥50% sits at the 81st to 88th, which is not the middle.
- **Proposed fix**: Report the counts plainly ("outside the null band in 4 of 12 pruned cuts, all under the Daily+ and SeveralDaily+ ≥20% logics"). Tone down "leaves our conclusions unchanged" in Section 7 to "leaves the sign of each result unchanged".

### D4. SA.E's footnote says the all-tasks estimates are identical to the main text, but they are not
- **Location**: p. SA-25 (`SA_E:16`) against p. SA-33 (`SA_E:204`)
- **Issue**: The footnote says "Point estimates for the 'all tasks' specification are identical to those reported in the main text." The fragmentation row is -0.01, -0.09, -0.05 on 871 occupations. Table 3 is -0.01, -0.09, -0.04 on 872. The appendix itself calls them "closely matching". The appendix also says all three predictions are evaluated "on this common set of occupations", but Figure SA.E.4 prints N = 865, 792, ..., 6 against 871, 832, ..., 20 in Figures SA.E.2 and SA.E.5.
- **Proposed fix**: Change "identical" to "identical for Predictions #1 and #2 and within 0.01 for #3 (871 of 872 occupations retain five tasks)". Say what the N in Figure SA.E.4 counts.

### D5. SA.F's exact-recovery-by-length claim is the Kendall-τ float-equality artifact *(carried over)*
- **Location**: p. SA-39 (`SA_F:176`)
- **Issue**: The text says "The mass at τ = -1 consists entirely of three- and four-step branches, and so does most of the spike at τ = +1, where 56% of three-step branches sit against none of the branches with five steps or more." Recomputing from `data/computed_objects/apqc_pcf_ordering/ordering_accuracy.csv` with `np.isclose(tau, 1)` gives 21 of 208 five-plus-step branches (10.1%) recovered exactly, 18 of them five-step. The 56% for three-step branches is right. One of the four τ = -1 branches has six steps. The claim also contradicts p. SA-37, which motivates exact recovery with the "one in 120 for a five-step branch" odds.
- **Proposed fix**: "... where 56% of three-step branches sit against 10% of branches with five or more steps."

### D6. SA.F points to a Section 7.3 argument that no longer exists
- **Location**: p. SA-44 (`SA_F:362`)
- **Issue**: The text says "The caveat applies to the fragmentation estimates in those columns as much as to the chain-length test below, though Subsection 7.3 explains why the same sparsity works in the fragmentation test's favor." After the Overleaf edits, Section 7.3 argues recorded order against imputed order (`7_empirics.tex:270-271`). It no longer makes a sparsity argument. "Those columns" also has no antecedent in the paragraph.
- **Proposed fix**: Either restore a sentence in 7.3 (at low label density the level term k/m no longer dominates the EFI, so the r/m arrangement term is identified) or state that argument here and drop the pointer. Replace "those columns" with "columns (4) to (6) of Table 3".

### D7. SA.A defines E1 backwards
- **Location**: p. SA-1 (`SA_A:17`)
- **Issue**: The text says "E1 tasks as those that an AI can perform in at least half the time required by a human". That reverses the rubric. Section 7 (`7_empirics.tex:25`) states it correctly: access to an LLM reduces the time "by at least half".
- **Proposed fix**: "... that an LLM can help complete in at most half the time a human would need, at equal quality."

### D8. Causal wording survives in OA and SA exhibits after the main text moved to association
- **Location**: Figure A.1 caption, p. OA-4 (`OA_A:161`). Tables SA.B.1 to SA.B.3 and Figures SA.B.1 to SA.B.3 captions, p. SA-10 to SA-15 (six instances of "Effect of Neighboring Tasks' ..."). SA.D "spillover results" and "spillover effect", p. SA-21 and SA-24 (`SA_D:142, 219`). SA.E "The positive effect of adjacent-step AI execution on the AI execution likelihood", p. SA-32 (`SA_E:177`). Figure A.1 notes "with the effect concentrated".
- **Issue**: The latest round retitled Table 2 "Neighboring Tasks' AI Execution Status and a Task's AI Execution", changed "raises" to "is associated with", and added the associational caveat on p. 41. The appendix exhibits still use causal titles.
- **Proposed fix**: Use the Table 2 pattern throughout, e.g. "Neighboring Tasks' AI Execution Status and a Task's AI Automation (GPT-filtered Sample)", and "association" in place of "spillover".

### D9. Section 7.2 says the placebo shows the pattern cannot arise by chance "in each case", but the DWA-FE panel does not show that *(carried over)*
- **Location**: p. 38 (`7_empirics.tex:191-192`); Figure A.1 notes, p. OA-4
- **Issue**: The text says "In each case, the actual orderings deliver stronger immediate-neighbor effects ... This implies that local work context and proximity do work that the reshuffled orderings cannot reproduce by chance." In panel (d), the observed 0.053 (k-1) and 0.042 (k+1) sit inside the body of the reshuffle distribution, which has substantial mass above them. Panels (a) to (c) support the claim.
- **Proposed fix**: "In the specifications without DWA fixed effects the observed immediate-neighbor effects lie in the upper tail of the placebo distribution. With DWA fixed effects they lie above the placebo mean but within its range."

### D10. Prediction #3 is called "the empirical content of Proposition 2", but Proposition 2 cannot rank arrangements *(carried over)*
- **Location**: p. 39 (`7_empirics.tex:208`); also p. 12 (`4_implications.tex:3`, "so that the gains from AI depend on the arrangement")
- **Issue**: Proposition 2 only bounds FI between OPT/8 and 5·OPT/4. That does not imply that a more dispersed arrangement of the same steps has a higher OPT. The arrangement comparison comes from Example 1 and from the closed form in Equation (4) for FI itself.
- **Proposed fix**: "Prediction #3 is the empirical counterpart of the fragmentation index of Section 4.2 (Example 1 and Equation (4)), which Proposition 2 shows tracks the optimal cost up to constant factors."

### D11. Section 6.1 presents a three-input CES that OA.C says it does not derive *(carried over)*
- **Location**: p. 29 (`6_extensions.tex:21-28, 33-35`), p. 6 (`1_introduction.tex:120-121`), against p. OA-30 and OA-32 (`OA_C:162, 244-250`)
- **Issue**:
  - Section 6.1 says the firm-level technology "can be aggregated to" Equation (10) "over economy-wide AI management labor, manual labor, and capital K", for any ρ < 0. OA.C says the capital term's exponent "is part of the CES form we posit rather than something the aggregation derives", and that the representation holds only along a one-dimensional locus with L_M/Y and K fixed.
  - Section 6.1 says firms "have access to the same AI technology but differ in how effectively they are able to deploy it". In OA.C, however, ᾱ in (C.7) is a deterministic function of α, τ^A_b and d_b, which are common to all firms once they share T and J. The text never says where the cross-firm variation in ᾱ comes from.
- **Proposed fix**: Add one sentence to Section 6.1: "The representation holds along the locus the aggregation traces, with capital fixed, and should not be read as identifying substitution over arbitrary input bundles." In OA.C, state that realized effective quality ᾱ is a firm-level draw, and not the value (C.7) computes from the common α.

---

## Minor

Ordered by location.

1. **p. 6** (`1_introduction.tex:134`). "AI execution operates over consecutive steps far more often than chance would produce". Section 7.1 calls the magnitude "modest" (O*NET 1.45 against a placebo mean of 1.38; PCF 1.16 against 1.09). Use "more often than chance".
2. **p. 7** (`1_introduction.tex:140-143`). "robust to the way each of these measures is constructed" (sequence, exposure, execution). All three checks listed concern the sequence. Either name SA.B (similarity and automation outcome) or say "robust to how the sequence is constructed".
3. **p. 7 and p. 34** (`1_introduction.tex:143`, `7_empirics.tex:44`). The PCF orders "come from actual work practice" and are "ground truth". SA.F (p. SA-36) says they are "not immutable ground truth" but "a documentation convention produced by committees". Use "documented by practitioners".
4. **p. 9 and p. 12** (`3_shortrun.tex:9` against `:115`). The short run has "each worker's job covers a fixed block of steps", then "The worker carries out the entire step sequence". The w.l.o.g. single-worker reduction on `:12` needs one clause of justification, or the fixed-blocks sentence should go. *(carried over)*
5. **p. 9 and p. OA-17** (`3_shortrun.tex:22, 35`; `OA_B:441-446`). The domain of d_i is never stated. OA.B.3 writes C_T as "a polynomial in 1/α" with "D_c ≥ 1", which requires integer d_i ≥ 1. α ∈ (0,1] appears only in Table A.1, and "decreasing in d_i" fails at α = 1. *(carried over)*
6. **p. 16** (`4_implications.tex:93`). "improvements in AI reliability only ever add steps to those AI executes". Proposition 1(ii) covers step k only. Moving right from the orange band can switch step k-1 from AI to manual. Say "only ever move step k toward AI". *(carried over)*
7. **p. 18** (`4_implications.tex:170`). "the fragmentation index is the expected cost of that strategy". A step the prophet knows will fail its first attempt costs 1 + 1/q_i in expectation if augmented, not 1/q_i. Say "approximately the expected cost". *(carried over)*
8. **p. 20** (`4_implications.tex:203`). "at each of them those returns jump upward as longer AI chains become worth deploying". At α = 0.50 the firm starts a length-one chain and nothing gets longer.
9. **p. 27** (`5_longrun.tex:190-191`). The example indexes hand-off costs by task, h_b, but h_i is defined per step (`:111`). Note that h_b is the hand-off at task b's last step.
10. **p. 25** (`5_longrun.tex:144`). T ∈ P(S) ranges over unlabeled partitions, but an AI strategy also has to label each singleton block manual or augmented. *(carried over)*
11. **p. 31** (`6_extensions.tex:62`). "an AI chain reaching back to some earlier step ℓ" conflicts with the recursion, where the chain begins at ℓ+1. In Section 3, ℓ is the chain's first step and C also names components and cost curves. Say "reaching back to step ℓ+1". *(carried over)*
12. **p. 32 and p. OA-22** (`6_extensions.tex:95`; `OA_B:618`). "[1/B, B] for some B > 0" is empty for B < 1. Use B ≥ 1.
13. **p. 33** (`7_empirics.tex:20`). "the fifth builds our PCF sample". The PCF sample also uses sources 1 to 3 through the label transfer.
14. **p. 33** (`7_empirics.tex:28`, footnote; `SA_A:101`). The Anthropic labels "are assigned on task-level feasibility". They come from the observed mix of conversation types (`SA_A:102`), and `:27` itself says they capture "what AI actually does". Replace "feasibility" with "the conversation mix for each task". *(carried over)*
15. **p. 35** (`7_empirics.tex:109`). "and they are" has no numbers in the main text. Add the PCF null mean (1.09) and z.
16. **p. 37 to 38** (Section 7.2, Table 2). The regression sample is restricted to tasks with two neighbors on each side (`SA_E:16`), but Section 7.2 never says so. *(carried over)*
17. **p. 38** (Table 2 and Table SA.B.1). Columns (4) and (6) are identical to the last digit, including pseudo-R² 0.197 (0.199 in SA.B.1). So the same-DWA count control drops out once DWA fixed effects are in, either because it is collinear with them or because it is constant on this sample. The text credits columns (5) and (6) with ruling out mechanical proximity (`7_empirics.tex:182`). Say which of the two applies. The notes also call column (5) "directly comparable with column (4)", but (5) adds the control that (4) lacks.
18. **p. 39** (`7_empirics.tex:215-217`, footnote). "q_i ∈ {0,1}" falls outside the model's (0,1], and 1/q_i is undefined at 0. Write "q_i → 0". "One minus the share of adjacent AI-able pairs" is really adjacent exposed pairs divided by m, not by the m-1 pairs.
19. **p. 40** (Table 3 header). "APQC Process Groups". The text, table notes and Figure 9(b) now say "PCF sample" and "PCF Process Groups".
20. **p. 42** (`8_conclusion.tex:14`). "helps explain why firms invest so heavily in AI capabilities". The model has no investment decision. Use "is consistent with firms investing ...". *(carried over)*
21. **p. OA-30 and OA-34** (`OA_C:161, 331`). The capital weight 1-θ_A-θ_M > 0 is never assumed. *(carried over)*
22. **p. SA-1** (`SA_A:16`). Unexposed is defined as "an E0 label". Section 7 says "all other tasks". The earlier audit found 789 of 17,925 records with no Eloundou label; say how they are coded.
23. **p. SA-2** (`SA_A:73`). "a workflow sequence for every O*NET occupation". The sample has 872 occupations; state the count. *(carried over)*
24. **p. SA-4 to SA-7** (`SA_A:102`, Figure SA.A.3 and SA.A.4 notes). Automated is described as "predominantly directive", but it is Directive plus Feedback Loop. Computer Programmers "about two thirds" is 10 of 17 (59%). Public Relations "about one third" is 7 of 18 (39%).
25. **p. SA-9 and SA-14** (`SA_B:28, 169`). The automation outcome is called "directly implied by our model", which SA.A (p. SA-4) rejects. The Figure SA.B.2 notes say "more AI-automated local context", but the regressors are neighbors' AI execution (table rows "Task (k-1) is AI-executed").
26. **p. SA-21 and SA-22** (`SA_D:118, 148-149`). "the mean across alternative prompts (the dashed colored line)". The notes say it is the mean of all 11 prompts. "The two adjacent steps still have larger ... effects than the steps two positions away" does not hold for k+1 against k+2 under DWA FE (prompt means 0.03 against 0.02; prompt 2 gives k+2 ≈ 0.06 against k+1 ≈ 0.01).
27. **p. SA-27 and SA-28** (`SA_E:68`). The Allergists passage lists steps 3-4, 9-10 and 14-16 as dropped. The figure also drops 8 and 13 (9 dropped, 7 kept).
28. **p. SA-38 to SA-39** (`SA_F:161, 170, 175, 179`).
    - "p = 0.001" from 1,000 permutations should be p < 0.001.
    - The category range "from 0.66 ..., 0.69 ..., and 0.65 ... down to 0.25" is unordered.
    - "a single inverted pair already registers as a large negative number" is wrong: one inversion on three steps gives τ = +1/3.
    - "on a ten-step branch the same inversion costs 0.02" should be 2/45 ≈ 0.04.
    - τ between -0.03 and -0.20 is close to random order, not "a handful of misordered pairs within an otherwise correct sequence".
29. **p. SA-37** (`SA_F:110`). "GPT-5-mini, temperature 0". The notebooks pass `temperature=0.0` through EDSL, but to my knowledge OpenAI's API accepts only the default temperature for GPT-5-family reasoning models. Confirm the setting took effect, or drop the claim.
30. **p. SA-41** (`SA_F:263`). "That these three numbers coincide is itself informative." With 95.4% of pairs determinate, the all-pairs and determinate-pairs accuracies must nearly coincide mechanically. *(carried over)*
31. **p. SA-45** (`SA_F:401-402`). The text gives z = 10.2 for the within-category reassignment. The saved `data/computed_objects/apqc_chainLength_placebo/chain_length_placebo_summary.csv` (observed 1.163, null mean 1.055) gives 9.91, and the Sept 8 pass reproduced 9.91 at the 0.71 floor. The within-group z = 6.2 matches the threshold-sweep run but not the saved summary (6.06). Quote both z's from one run. *(carried over)*
32. **References, p. 43 to 45.** Bloom, Sadun and Van Reenen (2016), "Management as a Technology?", is listed as *QJE* 132(3), 1-60. That looks like the NBER working paper (w22327) with a journal volume attached, so please verify. Levhari (1968) is missing volume 36(1). Autor et al. (2026) and Garicano et al. (2026) use sentence-case titles where every other entry uses title case.

---

Not re-checked here because they are proof-internal (per your instruction): A5, A13, A14 and R-D18 from the 2026-09-04 audit. Excluded because they are recorded author decisions: SA.D's "no evidence of systematically different orderings" wording (A21/R-M5).
