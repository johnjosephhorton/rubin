# Audit of *Chaining Tasks, Redefining Work: A Theory of AI Automation*

Full pass over the draft compiled on 2026-09-04 (134 pp.): body (Sections 1–8), Online Appendix OA.A–OA.C, Supplementary Appendix SA.A–SA.F, plus `tables/` and the TikZ sources in `plots/`. Grammar, style and formatting were deliberately ignored; this lists only substantive mistakes — notation errors, math and proof errors, inconsistencies across sections, and arguments that do not follow.

**65 distinct issues: 3 major, 24 medium, 38 minor.**

**Status as of 2026-09-04: 13 closed, 3 partially addressed, 49 open.** Closed entries carry ✅ and
partial ones ◐, each with a **Status.** line recording what was checked in the current source. Four of the
thirteen are marked closed because their target text was deleted rather than repaired (19, 20, 56, 57 all
went with Subsection SA.B.2 and Table SA.B.5). Every status was verified against the source rather than
inferred from a commit message, and anything that could not be confirmed was left open.

## How this was produced

21 independent auditors each took a section or a cross-cutting dimension (notation inventory across all files; recomputation of every stated number; reference and claim integrity; end-to-end logical coherence). They returned 175 raw findings. Every finding then went to two adversarial reviewers — one checking that the quoted text really says what the finding claims, one instructed to *refute* the finding on the merits and to redo any algebra itself — with a third adjudicator wherever the two disagreed. 59 candidate issues were refuted and dropped; the rest collapse to the 65 distinct issues below. Numeric claims were re-derived rather than taken on trust.

**Reading the flags.**

- *Reviewers split; settled by a third adjudicator* (23 findings) — the two reviewers reached opposite verdicts, and an independent adjudicator settled it from the source. Where the adjudicator narrowed a finding, the wording below is theirs, not the original auditor's; two findings were dropped outright at this stage.
- ⚠️ *Single review* (5 findings) — one of the two reviewers was lost to a spurious API safeguard error rather than to any judgement about the finding.
- *Flagged independently by N auditors* — N auditors who never saw each other's work reported the same problem. A confidence signal, not a severity signal.

**Severity.** *Major* = invalidates or materially weakens a stated result, or an empirical claim the paper's own tables do not support. *Medium* = a real error that is locally fixable without overturning a result. *Minor* = notation slips, undefined symbols, small numerical mismatches, imprecise statements.

Page numbers are the printed pages of the compiled PDF: plain integers in the body, `OA - n` in the Online Appendix, `SA - n` in the Supplementary Appendix.

## The three that matter most

1. **The abstract, introduction and conclusion assert Prediction #3 as an established occupation-level finding; Section 7.3 reports a null on exactly that sample.** The O*NET fragmentation coefficients are −0.01, −0.09, −0.04, none significant, and the body says so plainly ("On O*NET occupations we do not detect the workflow-level channel"). The significant estimates come from the APQC corpus, whose units are process groups, not occupations. Twelve of the 21 auditors flagged this independently. (p. 1, 7, 44)
2. ✅ **The power explanation offered for that null is not supported by the paper's own standard errors.** *(Addressed 2026-09-04 in `78fce8f`; the paragraph now reads the divergence as one of corpus scope. Detail at entry 2.)* Section 7.3 argues the workflow-level test is "better powered where AI-able material is sparse," but the standardized standard errors are 0.10/0.09/0.09 in O*NET against 0.10/0.11/0.11 in APQC, on 872 versus 525 observations — O*NET is if anything more precise. The two panels' coefficients also differ significantly from each other (p ≈ 0.016), so the contrast *is* a disagreement between the corpora. (p. 42)
3. **Two different AI-exposure definitions appear to be in circulation.** The descriptive appendix's counts imply a far lower exposure share than the 44% the main text uses, and one appendix table note defines the variable as E1-only against the paper's E1-or-E2 convention. Since exposure is both the key regressor in Prediction #3 and the input to the Empirical Fragmentation Index, this needs to be pinned down before the Section 7.3 discussion can be read. (p. SA - 2, and the related note issue at p. SA - 17)

Beyond these, the recurring pattern in the appendices is a **summary sentence that its own table or figure does not support** — SA.E's closing claim that all three predictions survive frequency pruning against its own Prediction #3 subsection, SA.B's figure note describing a specification that was not estimated, several table notes naming the wrong dependent variable. Those are individually small and collectively worth one careful pass.

On the theory side the errors are smaller and cleanly fixable: a maximizer stated at an infeasible point in the 5/4 bound (p. OA - 13), `D_c ≥ 1` asserted where nothing bounds step difficulty away from zero, hand-off time indexed by task when it is defined on steps, the chain cut-point letter swapped between the two DP recursions, and the greedy algorithm in Section 4.2 missing the branch that makes it approximately optimal.

---

## Major (3)

### ◐ 1. Introduction states the workflow-level fragmentation result as an established finding on O*NET occupations, but Section 7.3 reports a null there

**Page 7** · `1_introduction.tex, "Empirical Evidence" paragraph (lines 134-135; also line 94) vs. 7_empirics.tex, Sec. 7.3 / Table 3 (tab:fragmentation_index_regression_exposure)` · *inconsistency*
  
Flagged independently by **12** auditors

> Third, the model emphasizes that fragmentation of AI-exposed steps plays a central role in determining the returns to AI automation.
> In the data, controlling for the share of steps exposed to AI, occupations whose AI-exposed steps are more dispersed across the production workflow are associated with a lower share of their steps executed by AI.

**Issue.** The stated result is about *occupations*, i.e. the O*NET sample. In Section 7.3 the fragmentation coefficient in the O*NET occupation columns of Table 3 is -0.01, -0.09, -0.04 with standard errors 0.10, 0.09, 0.09, and the body says: "In the main sample it carries the predicted negative sign in all three specifications, but the point estimates are small and none is statistically distinguishable from zero. On O*NET occupations we do not detect the workflow-level channel." The only significant estimates (-0.35***, -0.26**, -0.34***) come from the APQC PCF corpus, whose units are process groups, not occupations. The same intro also writes "We then provide empirical evidence that jobs with higher fragmentation see a weaker translation from AI exposure to AI execution" (p. 5), and "the fragmentation and chaining results continue to hold when we re-estimate them on the practitioner-ordered PCF corpus" (p. 7) — the phrase "continue to hold" presupposes a main-sample result that does not exist. The intro's closing claim that "the share of steps exposed to AI can be a misleading predictor of occupational impacts" (p. 7) is likewise not what the O*NET columns show: there exposure is a strong predictor (0.39-0.49***) and fragmentation adds nothing measurable.

**Why it matters.** This is one of the paper's three headline empirical claims, and the introduction asserts on the main sample exactly the result that the main sample fails to deliver. A reader who never reaches Section 7.3 will believe the workflow-level channel is documented for occupations. The paper's own Supplementary Appendix calls it "the fragmentation null of Subsection 7.3" (SA - 27) and states "the null of Subsection 7.3 is a feature of the O*NET sample rather than an artifact of counting rarely-performed tasks as steps."

**Status.** ◐ Partially addressed 2026-09-04 (`310052f`). 1_introduction.tex was changed in three of the five flagged places, and two flagged sentences are untouched. FIXED: :135 now reads "controlling for the share of steps exposed to AI, process groups whose AI-exposed steps are more dispersed across the production workflow are associated with a lower share of their steps executed by AI. The same coefficient is negative but imprecisely estimated on O*NET occupations." :141 now reads "when we re-estimate on the practitioner-ordered PCF corpus, the chaining results replicate and the fragmentation channel is detected" ('continue to hold' is gone). NOT FIXED: :94 still says "We then provide empirical evidence that, on practitioner-documented workflows, jobs with higher fragmentation see a weaker translation from AI exposure to AI execution" - only a scope clause was prepended; the significant result is still attributed to 'jobs', while the PCF units are process groups, which is exactly the substitution the fix asked for. :145 is byte-identical to the audited text: "This is why the share of steps exposed to AI can be a misleading predictor of occupational impacts when production steps are technologically interdependent" - the audit flagged this claim explicitly, and the O*NET columns show exposure predicting strongly (0.39-0.49***) with fragmentation adding nothing. The abstract (0_main.tex:76) and conclusion (8_conclusion.tex:17) are unchanged since the audit (git diff a10af1f..HEAD is empty for both files). The commit message for 310052f itself records this as partial.

**Fix.** Rewrite as: "In the main O*NET sample the coefficient carries the predicted negative sign but is not statistically distinguishable from zero; on APQC's practitioner-documented process sequences, process groups whose AI-able steps are more dispersed do convert less of their exposure into execution." Replace "occupations"/"jobs" with "process groups" wherever the significant result is invoked, and drop "continue to hold" in favor of "is detected on".

### ✅ 2. The "identifying variation / better powered where AI-able material is sparse" explanation for the O*NET-APQC contrast is contradicted by the paper's own standard errors and is combinatorially backwards

**Page 42** · `7_empirics.tex, Sec. 7.3, final paragraph ("The contrast between the two panels is not a disagreement between the corpora..."), vs. tab:fragmentation_index_regression_exposure and SA_F_external_validation.tex, "Reading the PCF fragmentation coefficient"` · *empirics*
  
Reviewers split; **settled by a third adjudicator** · Flagged independently by **4** auditors

> The workflow-level test is thus better powered where AI-able material is sparse, since that is the regime in which arrangement varies most freely at a given level of exposure.

**Issue.** Three independent checks contradict this. (i) All variables are standardized within sample, so the standard errors are directly comparable across panels; they are 0.10, 0.09, 0.09 in O*NET and 0.10, 0.11, 0.11 in APQC. O*NET is if anything MORE precise, so it is not less powered. (ii) The paper's "only about 30% of the index's standard deviation survives conditioning" versus "about 56%" are within-sample percentages of index SDs that differ by a factor of four: SA.F (SA-48) reports the EFI standard deviation is 0.26 in O*NET and 0.06 in APQC, so in raw index units the surviving identifying variation is 0.30x0.26=0.078 in O*NET against 0.56x0.06=0.034 in APQC -- more than twice as much in O*NET. (iii) The stated mechanism is false: at a given exposure level, arrangement varies LESS freely when AI-able steps are sparse, because sparse steps are almost always isolated. With m=26 and k=3 (the PCF averages), 77.8% of arrangements have r=k=3 and the simulated SD of r/m is 0.017; with m=20 and k=9 (the O*NET averages) only 0.13% have r=k and the simulated SD of r/m is 0.055. Finally, the two panels' coefficients are statistically distinguishable: cols (1) vs (4) give z=(-0.01+0.35)/sqrt(0.10^2+0.10^2)=2.40, p=0.016; cols (3) vs (6) give z=2.11, p=0.035 -- so the contrast IS a disagreement between the corpora at conventional levels, contrary to "The contrast between the two panels is not a disagreement between the corpora."

**Why it matters.** The paper explains away its main-sample null with a power story that its own table refutes and its own appendix numbers reverse. This changes what the result means: the honest reading is that two corpora give significantly different answers, not that one is too noisy to speak.

**Status.** ✅ Addressed 2026-09-04 (`78fce8f`). The closing paragraph of Subsection 7.3 was rewritten and the power explanation is gone, along with the "not a disagreement between the corpora" sentence, the "both panels give evidence" close, and the 30-versus-56 percent surviving-variation comparison that supported it. The divergence is now read as one of scope. O*NET spans the breadth of the economy, including task lists that describe a repertoire of activities rather than a process with an intrinsic order, so their sequence has to be imputed, while the PCF documents operational business processes that practitioners record as ordered flows. Prediction \#3 leans hardest on the ordering, so it speaks most clearly where sequences are documented rather than inferred. The paragraph closes by stating the result plainly, that the channel is found on the PCF corpus and not detected on O*NET. The same sparsity story appeared in a cross-reference at `SA_F_external_validation.tex:416` and was reworded to point at the discussion rather than assert the mechanism.

**Fix.** Drop the power explanation, or replace it with the correct one. Report that the standardized standard errors are essentially equal across panels, that the two coefficients differ significantly (p=0.016 in cols 1 vs 4), and give a substantive rather than statistical account of the divergence -- e.g. GPT-imputed versus practitioner-documented ordering, or the difference in the raw scale of the index (SD 0.26 vs 0.06) that SA.F already flags.

### ✅ 3. Figure SA.A.2(a) and the "605 (69%)" statistic imply ~15% of O*NET steps are AI-exposed, but Section 7 and SA.F state 44%

**Page SA - 2** · `SA_A_sample_construction.tex, line 34 and Figure \ref{fig:occupation_ai_share} panel (a) (plots/ONET_Eloundou_Anthropic_GPT/ai_exposed_task_share_distribution.png); conflicts with 7_empirics.tex line 259 and SA_F_external_validation.tex line 357` · *inconsistency*

> Of the 872 occupations, 605 (69\%) contain at least one AI-exposed task and 555 (64\%) contain at least one AI-executed task.

**Issue.** SA.A states the exposure definition is E1 or E2 ("we treat tasks with a human-assigned E1 or E2 label as exposed to AI"), and reports that only 605 of 872 occupations have ANY AI-exposed task, i.e. 267 occupations (30.6%) have zero. Figure SA.A.2(a) puts 35.4% of occupations in the [0,0.05) exposure bin and yields an occupation-level mean AI-exposed share of 14.6%. But Section 7 (p. 41) says "In O*NET, where $44\%$ of steps are AI-able" and SA.F repeats "$44\%$ and $13\%$ in the O*NET task universe", where "AI-able" is explicitly the E1-or-E2 label set from which the EFI is built. A 44% task-weighted share is arithmetically unreachable from the figure's distribution. 14.6% is precisely Eloundou et al.'s human E1-only (alpha) share, while ~46% is their E1+E2 (zeta) share, so the SA.A figures appear to be built on E1 only while the main text quotes E1+E2. Note SA.B's Table SA.B.5 note independently says "AI-exposed (E1)", corroborating that two different exposure definitions are in circulation.

**Why it matters.** The AI-exposure variable is the right-hand-side regressor in Prediction #3 and the object from which the Empirical Fragmentation Index is constructed. If the descriptive appendix and the estimation use different exposure definitions, the reader cannot tell which definition produced Table 3, and the paper's key power argument ("where 44% of steps are AI-able, the level component dominates ... only about 30% of the index's standard deviation survives conditioning") rests on a share that the paper's own descriptive figure contradicts by a factor of three.

**Status.** ✅ Addressed 2026-09-04. SA_A_sample_construction.tex:35 now reads "Of the 872 occupations, 809 (93\%) contain at least one AI-exposed task and 555 (64\%) contain at least one AI-executed task" - the audited "605 (69\%)" is gone (grep for '605' and '69\%' across all .tex returns nothing). Figure SA.A.2(a), plots/ONET_Eloundou_Anthropic_GPT/ai_exposed_task_share_distribution.png, was regenerated (mtime Sep 4 13:38) and I read it: the [0,0.05) bin is now 8.7%, not 35.4%, and the printed bar shares give a bin-midpoint-weighted occupation mean of 43.6%, i.e. the 44% the main text quotes, versus the 14.6% the audited figure implied. 7_empirics.tex:259 ($44\%$ of O*NET steps AI-able) and SA_F:357 ($44\%$ and $13\%$) are consistent with it. SA_A:16 still declares the single E1-or-E2 rule, and grep for 'E1' across all .tex and tables/ shows no surviving 'AI-exposed (E1)' note; the Table SA.B.5 note that carried it went with the table under commit 1d64cb0.

**Fix.** Determine which exposure definition (E1 only vs. E1 or E2) actually generated Figure SA.A.1/SA.A.2, the 605/555 counts, and the 44%/13% figures, and make all of them use one definition; state that definition once in SA.A and repeat it identically in Section 7 and SA.B/SA.F. If E1-only is used anywhere deliberately, flag it explicitly as a different measure.


---

## Medium (24)

### 4. Prediction #2's neighbor regression conditions only on the focal step's binary AI exposure, so it cannot separate chain complementarity from correlated AI-ability of adjacent steps — a confound the paper raises in SA.F but concedes it cannot test for this prediction

**Page 1** · `0_main.tex abstract, prediction (2); test implemented in 7_empirics.tex eq:DWA_regression_ai / tables/noTasksWithRepetitiveDWAs/allTasks_ai.tex; interpreted in 8_conclusion.tex ("the local complementarities created by AI chains", p. 44)` · *empirics*
  
Reviewers split; **settled by a third adjudicator**

> (2) a step's local context, the extent to which the work around it is AI-executed, increases the likelihood that the step itself is AI-executed

**Issue.** Equation (7.1) regresses the focal step's AI-execution indicator on its four neighbors' AI-execution indicators, controlling for the focal step's own AI exposure, the occupation's task count, and (in columns 4 and 6) DWA fixed effects. Under the no-chaining null — each step assigned by its own comparative advantage alone — a positive neighbor coefficient still arises whenever adjacent steps have correlated AI-ability, because the E1/E2 flag is binary and DWA is a coarse grouping, so substantial residual variation in the focal step's q_k survives the controls and neighbors' execution proxies for it. The paper itself supplies every premise of this alternative: SA.B.1 (p. SA-9) concedes that "even within a given DWA tasks may still somewhat differ in their objectives, execution nature, or required skills"; SA.F (p. SA-36) warns that the LLM ordering "may place semantically similar items next to one another" and that, since "AI exposure and execution are themselves correlated with what a task is about, such an ordering could manufacture the contiguity we document ... and contaminate our other exercises"; and SA.F (p. SA-37, SA-49) then states that the external, practitioner-ordered PCF corpus that answers this concern for Prediction #1 is too underpowered to test Prediction #2 (only 9 of 137 matched DWAs contain both an executed and a non-executed step). The position-reshuffle placebo cannot arbitrate either, since reshuffling destroys the adjacency of similarly AI-able steps just as it destroys chains. Neighbors' exposure labels are observed for every task but are never entered in the main table, in SA.B.1, in SA.D, or in SA.E, and no placebo anywhere holds the arrangement of the exposure labels fixed. The paper does offer partial discriminating evidence — the sharp fall from 0.04-0.06 at distance one to about zero at distance two in the fixed-effects columns (argued at p. 39), and the GPT-filtered similarity sample of SA.B.1, which tightens exactly the within-DWA comparability at issue — but the distance-two null is estimated conditional on distance-one execution, which itself proxies for any local ability factor and would absorb it, so the contrast is suggestive rather than decisive. The result is that the design supports an association, while the abstract ("increases the likelihood"), the p. 39 summary, and the conclusion's attribution to "the local complementarities created by AI chains" state it as the model's mechanism.

**Why it matters.** The abstract states the relationship causally ("increases the likelihood") and the conclusion attributes it to "the local complementarities created by AI chains." As implemented, the test cannot discriminate the model's mechanism from the leading alternative, so it does not identify the object the model predicts. The DWA fixed effects in columns (4) and (6) hold the *focal* step's type fixed but leave the neighbors' AI-ability free.

**Fix.** Add a diagnostic aimed at the correlated-ability channel and report it alongside Table 2. Two cheap options, neither requiring new data: (i) re-estimate Equation (7.1) adding prev/next (and two-position) is_exposed indicators and report whether the neighbor-execution coefficients survive — an imperfect check, since exposure is the same binary proxy and is a determinant of the treatment, so it should be framed as a robustness diagnostic rather than a clean separator; (ii) run the Prediction #1 run-length placebo, or the neighbor regression itself, on the AI-exposure labels instead of the execution labels, which measures directly how much adjacency of AI-ability the GPT orderings produce and therefore how much of the distance-one effect could be generated under the no-chaining null. Whatever the diagnostics show, note explicitly in Subsection 7.2 that the SA.F ordering concern is answered externally for Prediction #1 but, by the paper's own admission, not for Prediction #2, and soften the abstract's "increases the likelihood" and the p. 39 summary to the associational phrasing already used in the introduction and conclusion.

### 5. The greedy algorithm described in the body is not the greedy of the proof, and as literally described it is not approximately optimal

**Page 19** · `4_implications.tex, Section 4.2, last sentence before Section 4.3; vs. OA_B_omitted_proofs.tex, proof of Lemma~\ref{lem:FI.lower.bound.4} / \ref{lem:FI.lower.bound.8}` · *inconsistency*

> A further implication of the proof is a natural and approximately optimal greedy algorithm for constructing AI strategies, where the firm groups steps into a chain as long as the probability of success stays sufficiently high, terminating the chain and starting a new one when it falls too low, and running an entire chain manually whenever its steps' total manual time is below the cost of one verification.

**Issue.** The greedy in the proof has three rules, not two. Besides (a) extend the chain while success probability ≥ 1/2 and (c) run a constructed block manually when Σ t^M_i < 1, it has rule (b): "If that set is empty (i.e., the first step has success probability strictly less than 1/2) then the greedy gives up on chaining and executes s_1 on its own, either manually or augmented, whichever is cheaper" — i.e. cost min{t^M_i, 1/q_i}, the "individual" tasks T_I that the whole charging argument is built around. The body sentence omits rule (b): a step that cannot start a chain reaching probability 1/2 becomes "a new chain" of length one, run manually only if t^M_i < 1 rather than if t^M_i < 1/q_i.

**Why it matters.** Without rule (b) the described algorithm has unbounded approximation ratio, so the sentence's claim that it is "approximately optimal" and "a further implication of the proof" is false for the algorithm as stated. A reader implementing the body's description gets a different, arbitrarily bad algorithm.

**Fix.** Add the missing branch, e.g. "...terminating the chain when it falls too low; a step on which no chain can start is executed on its own in whichever mode — manual or augmented — is cheaper; and running an entire chain manually whenever its steps' total manual time is below the cost of one verification."

### 6. The "prophet" interpretation is stated as an identity but the fragmentation index is not the expected cost of that strategy

**Page 19** · `4_implications.tex, Section 4.2, paragraph after Proposition~\ref{prop:fragmentation}; same claim in 1_introduction.tex ("Optimizing Automation and Fragmentation" paragraph, printed page 5)` · *logic*

> Runs of steps that will all succeed are natural candidates for a chain, so the prophet chains each such run, unless running that whole run manually is cheaper, and executes every remaining step on its own in whichever of the two modes is cheaper for it; the fragmentation index is the expected cost of that strategy.

**Issue.** The prophet is defined as knowing "which steps AI would complete on its first attempt." For a run C that will all succeed, the index charges ω(C) = min{1, Σ t^M} — the *realized* cost of one successful attempt, which is correct given the foresight. But for a step s_i in F, the index charges min{t^M_i, 1/q_i}, the *unconditional* expected cost of augmenting it. A prophet who knows s_i fails on the first attempt pays 1 + 1/q_i in expectation if it augments the step (one wasted attempt, then a fresh geometric restart), so the correct charge is min{t^M_i, 1 + 1/q_i}. The index therefore mixes a conditional cost for the succeeding runs with an unconditional cost for the failing steps, and the stated equality fails. The paper itself notes elsewhere that "a long chain ... may be worth attempting even when the prophet knows it will fail on the first try," which is exactly the conditioning that the failed-step term ignores.

**Why it matters.** This is the paper's headline economic interpretation of its central new object, and it is repeated in the introduction as the definition ("the expected cost of a simple execution plan endowed with foresight into AI's successes and failures"). As stated it is not correct, so the index has no exact behavioural reading.

**Fix.** Either weaken to an inequality/heuristic ("the fragmentation index is a lower bound on, and a close proxy for, the expected cost of that strategy" — min{t^M_i, 1/q_i} ≤ min{t^M_i, 1 + 1/q_i} always), or say explicitly that a failed step is charged the cost of executing it from scratch, ignoring the wasted first attempt.

### 7. The intuition in Section 6 attributes aggregate substitution to a change in relative wages, but the OA.C derivation is only valid when relative wages are held fixed

**Page 30** · `6_extensions.tex, Sec. 6.1 (paragraph "Arriving at Equation~\eqref{eq:lr_macro_ces}..."); vs OA_C_CES_representation.tex eq:new_wage (OA.C.1) and Sec. OA.C.3` · *logic*
  
Flagged independently by **2** auditors

> Firms have access to the same AI technology but differ in how effectively they are able to deploy it, and when AI management labor becomes relatively cheaper the economy's input mix moves because the set of firms for which using AI is worthwhile has moved, not because any firm has changed its own mix.

**Issue.** In OA.C the stage-one problem is eq:totalcost_with_handoff with the job wage given by eq:new_wage, $w_M\sum c^M_b+w_A\sum c^A_b$ (footnote 32 makes the correspondence to eq:wage explicit: the body's $c_b$ is $w_{E(b)}c^{E(b)}_b$). That objective is homogeneous of degree one in $(w_A,w_M)$, so the cost-minimizing $\mathcal{T}$ and $\mathcal{J}$ — and hence $\tau_A$ and $\tau_M$ — depend on the *ratio* $w_A/w_M$. But the derivation in Sec. OA.C.3 differentiates (OA.C.17) with respect to $u=w_A\tau_A/(1-w_M\tau_M)$ holding $\tau_A,\tau_M,\theta_A,\theta_M,\rho$ constant. Holding $\tau_A,\tau_M$ fixed while $u$ varies is legitimate only if $(w_A,w_M)$ moves along a ray, i.e. proportionally — precisely the case in which AI management labor does *not* become relatively cheaper. A change in $w_A/w_M$ changes every firm's chosen AI strategy and job design (its own mix), changes $\tau_A$ and $\tau_M$, and therefore, through Restriction (OA.C.15), changes the CES weights $\theta_A,\theta_M$ along the very locus over which eq:lr_macro_ces is supposed to be a fixed-parameter CES.

**Why it matters.** The sentence quoted is the paper's entire economic account of where aggregate substitution comes from, and the clause "not because any firm has changed its own mix" is false in the model as specified. The comparative static that the appendix actually supports is a proportional fall in the wage level relative to the output price, not a change in the relative price of AI management labor.

**Fix.** Restate the comparative static in Section 6 as a movement in the participation threshold driven by the wage level relative to output price (holding $w_A/w_M$ fixed), or add to OA.C an explicit assumption that the cost-minimizing $\mathcal{T},\mathcal{J}$ are invariant to $w_A/w_M$ over the relevant range, and say so in the Section 6 footnote of what the representation asks.

### 8. Prediction #1's two placebos both randomize step positions, so the chain-length result cannot separate AI chaining from clustering of AI exposure in the observed ordering

**Page 35** · `7_empirics.tex, Sec. 7.1 (sec:chainLength_prediction), fig:aiChains_graphs_def1` · *logic*
  
Reviewers split; **settled by a third adjudicator**

> This prediction follows directly from Definition~\ref{def:ai_chain} on how AI chains are formed in the model.

**Issue.** Section 7.1 benchmarks the observed average AI chain length against exactly two nulls: permuting task positions within an occupation, and reassigning tasks-with-execution-labels across occupations within a SOC major group (which also "randomiz[es] task positions within each occupation"). Both destroy the observed arrangement of AI *exposure*, which the model treats as a primitive (the step difficulties d_i; Sec. 7 calls exposure "the difficulty of the step for AI"). A no-chaining benchmark in which each step is AI-executed iff AI has the comparative advantage on it in isolation therefore reproduces "observed > null" whenever AI-exposed steps happen to sit adjacent in real workflows. The section cannot discriminate the chaining mechanism from that alternative, yet the result is read as evidence for chaining in the abstract ("AI-executed steps co-occur in chains"), the introduction ("the chaining mechanism implies that AI-executed steps should appear in contiguous blocks"), and the conclusion. The same gap is also used at p.31 to validate the GPT ordering as capturing "meaningful workflow structure" — but if the ordering groups content-similar steps, AI-ability clusters mechanically, which is exactly the confound. The section's stated caveats cover only two other alternatives (arbitrary GPT sequences, randomly distributed execution labels), and no exposure-arrangement-preserving chain-length null appears anywhere in the paper (SA_D, SA_E and SA_F all reuse the same two nulls; SA_B's execution-based EFI conditions on the exposure share and count, not on its arrangement). Note that the confound is addressed elsewhere in Section 7 — Prediction #2 controls for the focal step's exposure and adds DWA fixed effects, and Prediction #3 conditions on exposure to isolate the arrangement term — so this weakens the standalone interpretation of Test #1, not the paper's overall evidence.

**Why it matters.** Prediction #1 is the paper's first and most-cited piece of empirical support for chaining, and both the abstract ("AI-executed steps co-occur in chains") and the conclusion rest on it. If the observed clustering is inherited from clustered exposure, the test provides no evidence for chaining over the standard step-by-step comparative-advantage benchmark the paper is departing from. The paper recognises exactly this concern elsewhere - Section 7.2 adds DWA fixed effects "addressing differences in AI-ability across types of task" - but not here.

**Fix.** Add a third benchmark to Section 7.1 that holds the observed arrangement of AI exposure fixed — e.g. reshuffle execution labels only within the set of exposed steps, keeping each step's position and exposure status as observed, or report the observed AI-execution chain length against the chain length of AI-exposed steps in the same ordering. If that is not feasible, add one sentence in 7.1 stating that the two placebos randomize positions and so cannot separate chaining from clustered exposure, and point forward to Predictions #2 and #3, which do condition on exposure. Separately (optional wording only, not an error): the clause "follows directly from Definition~\ref{def:ai_chain}" is loose, since the definition alone admits fully scattered length-one chains; the mechanism given in the next sentence is what delivers the prediction, and citing it there would read more precisely.

### 9. Baseline column (1) of Table 2 is read as a "local context" effect, but the paper's own placebo shows the reshuffle null for every neighbor coefficient is ≈0.089, so the two-positions-away estimates are BELOW their null

**Page 37** · `7_empirics.tex, Sec. 7.2 (sec:DWA_prediction), discussion of tab:DWA_regression_aiExecution_mainSample; placebo evidence in OA_A_tables_and_figures.tex, fig:DWA_regression_aiExecution_mainSample panel (a)` · *empirics*
  
⚠️ **Single review** — only one of the two reviewers returned

> The baseline in column~(1) shows that AI execution anywhere in the focal task's local context raises the probability that the task is AI-executed, at one and at two positions away alike, with the effects of the adjacent steps roughly twice the size of those two positions out.

**Issue.** Equation (12) contains no occupation-level controls (only SOC major/minor groups in cols (2)-(3)), so the column-(1) coefficients are dominated by between-occupation variation in AI intensity: in an AI-intensive occupation both the focal task and all four of its neighbors are more likely AI-executed. The paper's own placebo figure quantifies this. In panel (a) of Figure OA.A.1 (no fixed effects) the position-reshuffle null distributions are centred at roughly +0.089 for ALL FOUR regressors (k-2, k-1, k+1, k+2), not at zero. The observed values are 0.124 (k-1), 0.117 (k+1), 0.065 (k-2), 0.050 (k+2). So only about 0.035 of the reported 0.124 immediate-neighbor effect is attributable to workflow position; and the two-positions-away estimates lie BELOW their nulls (0.065 sits at roughly the 3rd percentile of a null running 0.05-0.13; 0.050 lies entirely below the null's support). There is therefore no positive two-positions-away local-context effect in column (1) — if anything a negative one — contrary to the quoted sentence's "at one and at two positions away alike." The sentence also contradicts the paper's own later statement on p. 39 that "the actual orderings deliver ... weaker distant-neighbor effects than the placebos."

**Why it matters.** Column (1) is presented as the headline result for Prediction #2 and its 0.12 magnitude is the number quoted throughout ("attenuates from 0.12 in the baseline"). If ~72% of that number is reproduced by randomly reordering tasks, the baseline is not evidence of local context at all, and the claimed decay pattern (adjacent ≈ twice distant) is an artifact: null-adjusted, the immediate-neighbor effect is +0.035 while the two-away effect is −0.038, a sign flip rather than a factor of two.

**Fix.** Report the placebo-null-adjusted estimates (observed minus reshuffle mean, with reshuffle percentiles) alongside the raw AMEs, and add occupation fixed effects (or at minimum state that the column-(1) coefficients are not net of occupation composition). Rewrite the quoted sentence to say that only the immediate-neighbor coefficients exceed their reshuffle null and that the two-positions-away coefficients fall below theirs.

### 10. The main text never says how the neighbor indicators are defined at workflow boundaries; SA.E states the regression uses only tasks "with two neighbors on either side", a restriction the 10,708 sample count is not attributed to

**Page 37** · `7_empirics.tex, Sec. 7.2 sample construction, vs. SA_E_frequency_robustness.tex, sample-definition paragraph` · *empirics*

> In our main sample, we drop tasks that are mapped to multiple DWAs.
> We further restrict attention to DWAs that contain tasks appearing in more than one occupation.
> Together, these restrictions reduce the sample to 10{,}708 tasks spread across 1{,}748 DWAs.

**Issue.** Equation (12) requires prev2, prev, next and next2 to exist, which they do not for the first two and last two tasks of any workflow. The main text is silent on whether such tasks are dropped or coded as zero — a choice that matters, because coding a missing neighbor as "not AI-executed" mechanically attaches non-AI neighbors to boundary positions. SA.E states the rule: "for Prediction~\#2 the neighbor regression uses the eligible tasks of the same occupations, namely those with two neighbors on either side, which a five-task workflow always admits", and its footnote adds that "Point estimates for the ``all tasks'' specification are identical to those reported in the main text." If the same eligibility rule underlies the main sample, then roughly 4 x 871 ≈ 3,500 boundary tasks are also dropped, and the main text's attribution of the 10,708 count to the two DWA restrictions alone ("Together, these restrictions") is incomplete.

**Why it matters.** The sample definition is the basis for every number in Table 2 and for the reader's ability to reproduce the exercise; a reader following the main text's stated restrictions will not recover 10,708 tasks. It also matters substantively, since dropping the first and last steps of every workflow removes exactly the positions where chains are truncated.

**Fix.** State in Section 7.2 how boundary tasks are handled and include that restriction in the accounting for the 10,708 figure (e.g. "we further require the focal task to have two predecessors and two successors in its workflow, leaving 10,708 tasks").

### 11. Table 2's standard errors are clustered only on DWA, a dimension that does not span the within-occupation dependence Equation (12) builds in; in the DWA-FE column the paper's own within-occupation placebo distribution is ~33% wider than the reported SE

**Page 38** · `7_empirics.tex, note to tab:DWA_regression_aiExecution_mainSample; specification eq:DWA_regression_ai` · *empirics*
  
Reviewers split; **settled by a third adjudicator**

> Standard errors are bootstrapped using $B=200$ replications and clustered at the DWA level.

**Issue.** The unit of observation in Equation (12) is an O*NET task within an occupation, and the four regressors are the AI-execution indicators of the two tasks before and after it in the same occupation's workflow. Each observation's dependent variable is therefore mechanically four other same-occupation observations' regressor, and unobserved occupation-level determinants of AI use are common to all ~12.3 sample tasks in an occupation (10,708 tasks, ≤872 occupations). That dependence runs within OCCUPATION. The reported variance estimator clusters only on DWA (1,748 / 1,705 / 534 clusters), a non-nested dimension: adjacent tasks in one occupation almost always sit in different DWAs, so the covariance the design creates is treated as zero. Occupation is never used as a clustering dimension anywhere in the paper (only three clustering statements exist — 7_empirics.tex:161, SA_E:16, SA_E:134 — all DWA), no occupation fixed effects appear in any column, and the dependence structure is never discussed or the DWA level justified. On direction, the paper's own within-occupation reshuffle is the only internal evidence, and it cuts both ways: re-extracting Figure OA.A.1, the placebo SD is 0.011 in column (1) and ~0.010 in columns (2)-(3), i.e. at or below the reported SEs of 0.02 and 0.01, so no understatement is demonstrated there; but in the DWA-FE column (panel d) the placebo SD is 0.0265 against a reported SE of 0.02. Since the reshuffle holds each occupation's set of AI-executed tasks fixed and permutes only order, it conditions the occupation-intensity component out and is a lower bound on what occupation clustering would capture — so column (4) is direct internal evidence that DWA clustering understates the relevant sampling variability there. That matters for what the text claims: column (4)'s adjacent-neighbor AMEs of 0.05** and 0.04** (t = 2.5 and 2.0 at SE 0.02) fall to t ≈ 1.9 and 1.5 at 0.027, and columns (5)-(6) plus every neighbor regression in SA_B, SA_D and SA_E have no placebo counterpart at all, so their stars rest entirely on DWA-clustered standard errors.

**Why it matters.** The reported standard errors (0.02 in the baseline, 0.01-0.03 elsewhere) are the basis for the significance stars that carry Prediction #2. Since the paper's own reshuffle placebo shows the null AME for the baseline is 0.089 with a spread of roughly ±0.02, the analytic/bootstrap SE around zero is not the right yardstick, and the DWA clustering understates the dependence that actually generates that spread.

**Fix.** Report standard errors clustered on occupation, or two-way on occupation and DWA, for Table 2 and for the neighbor regressions in SA_B/SA_D/SA_E, and state in the table note which dependence each dimension is meant to capture. Where the within-occupation reshuffle already exists (columns (1)-(4)), report the randomization-inference p-value next to each estimate rather than only the histogram, since that placebo is the design-based check that actually spans the overlapping-window and occupation-composition dependence; do so in particular for column (4), where the placebo SD (0.027) exceeds the reported SE (0.02) and the observed 0.052 sits inside the placebo support. Full occupation fixed effects are not the right ask here — with ~12 tasks per occupation and only 555 of 872 occupations containing any AI-executed task, a logit would drop the occupations without within-occupation outcome variation, and the within-occupation reshuffle is the design-based analog of that conditioning — so the reshuffle p-values plus a robust clustering dimension are the substitute.

### 12. "The reshuffled orderings cannot reproduce by chance" is false for the DWA-fixed-effects specification — the one that actually implements the stated "same step, different occupations" design

**Page 39** · `7_empirics.tex, Sec. 7.2 (sec:DWA_prediction), placebo paragraph; evidence in OA_A_tables_and_figures.tex, fig:DWA_regression_aiExecution_mainSample panel (d) (fig:ame_dwa)` · *empirics*

> In each case, the actual orderings deliver stronger immediate-neighbor effects and weaker distant-neighbor effects than the placebos.
> This implies that local work context and proximity does work that the reshuffled orderings cannot reproduce by chance.

**Issue.** In panel (d) of Figure OA.A.1 (DWA fixed effects, i.e. Table 2 column (4)) the position-reshuffle null for the k-1 coefficient is centred near +0.02 with a standard deviation of roughly 0.03 and support from about -0.07 to +0.10; the observed value is 0.052, which sits at roughly the 88th percentile (about 12% of the 1,000 reshuffles produce a larger value). For k+1 the null is centred near +0.025 with support -0.06 to +0.11 and the observed value is 0.041, at roughly the 80th percentile. Neither is outside the placebo distribution at any conventional level. The claim that the reshuffled orderings "cannot reproduce by chance" is therefore not supported for the DWA-FE specification, and "in each case" is not true of all four panels. The supplementary appendix is careful here — SA.E reports "the 100th percentile in all three specifications" and those three are no-FE, SOC major and SOC minor, deliberately excluding DWA FE — but the main text makes the stronger blanket claim.

**Why it matters.** Columns (4) and (6) are the only specifications that implement the design the section says is required ("Testing this prediction requires identifying steps that appear across different occupations"), since without DWA fixed effects the comparison is across different DWAs entirely. If the placebo-benchmarked evidence disappears in exactly that specification, the headline claim for Prediction #2 rests on comparisons the paper itself says are contaminated by cross-occupation and cross-task heterogeneity.

**Fix.** Report the reshuffle percentile / placebo p-value for each panel of Figure OA.A.1 in the figure note (as SA.E and SA.F do for their exercises), and qualify the main-text sentence: the observed immediate-neighbor effects exceed their nulls decisively without DWA fixed effects and only weakly (roughly p≈0.12 and p≈0.20) once DWA fixed effects are imposed.

### ✅ 13. Upper-bound proof states the maximizing point as α^{-d(T_b)} = 1/2, which is infeasible (α^{-d} ≥ 1); it should be α^{d(T_b)} = 1/2

**Page OA - 13** · `OA_B_omitted_proofs.tex, proof of Lemma~\ref{lem:FI.upper.bound} (Lemma OA.B.1), last line before \end{proof}` · *math*
  
Flagged independently by **5** auditors

> The ratio between this expression and $\alpha^{-d(T_b)}$ is maximized at $\alpha^{-d(T_b)} = 1/2$, achieving a value of $5/4$ as claimed.

**Issue.** The ratio being maximized is (1 + α^{-d} - α^{d})/α^{-d}. Writing z = α^{d(T_b)} ∈ (0,1], this ratio equals z(1 + 1/z - z) = 1 + z - z², whose maximum over z is at z = 1/2 with value 5/4. So the maximizing point is α^{d(T_b)} = 1/2, i.e. α^{-d(T_b)} = 2 — not α^{-d(T_b)} = 1/2. Since α ∈ (0,1] and d(T_b) ≥ 0, α^{-d(T_b)} ≥ 1 always, so the point the proof names cannot occur; and evaluating the ratio at α^{-d(T_b)} = 1/2 (i.e. z = 2) gives 1 + 2 - 4 = -1, not 5/4.

**Why it matters.** The 5/4 in Proposition 2 is delivered by exactly this line. The constant itself is correct, but as written the key optimization step is evaluated at a point outside the feasible range and gives the wrong value, so the proof of the upper half of Proposition 2 does not read as valid.

**Status.** ✅ Addressed 2026-09-04 (`f9d4a4a`). OA_B_omitted_proofs.tex:256 now reads "The ratio between this expression and $\alpha^{-d(T_b)}$ is maximized at $\alpha^{d(T_b)} = 1/2$ (equivalently $\alpha^{-d(T_b)} = 2$), achieving a value of $5/4$ as claimed." That is the exact replacement the fix specified, and it is feasible: with z = alpha^{d(T_b)} in (0,1], the ratio (1 + 1/z - z)*z = 1 + z - z^2 peaks at z = 1/2 with value 5/4. The preceding line :255 still derives the bound 1 + alpha^{-d} - alpha^{d}, so the algebra now lines up. The commit records it as N24.

**Fix.** Replace with "is maximized at $\alpha^{d(T_b)} = 1/2$ (equivalently $\alpha^{-d(T_b)} = 2$), achieving a value of $5/4$". Optionally add the one-line verification: with $z = \alpha^{d(T_b)}$, $1 + \alpha^{-d} - \alpha^{d} \le \tfrac54 \alpha^{-d}$ ⟺ $z + 1 - z^2 \le 5/4$ ⟺ $(z - 1/2)^2 \ge 0$.

### 14. Discretization grid is sized for a single rounding, contradicting the proof's own compounding argument two paragraphs later

**Page OA - 23** · `OA_B_omitted_proofs.tex, line 623, proof of Prop. 5 (prop:totalcost_optimization_dp), "Discretization and Computation Time"` · *math*

> Each of these ranges contains $O\bigl(\epsilon'^{-1}\log(mB)\bigr) = O\bigl(m\,\epsilon^{-1}\log(mB)\bigr)$ powers of $(1+\epsilon')$; rounding upward can carry a total past the bounds above by at most one factor of $(1+\epsilon')$, which changes these counts only by an additive constant.

**Issue.** The overshoot of a recorded (rounded) running total above its true value is NOT one factor of $(1+\epsilon')$. The proof itself says so on the next page: "the distortions therefore compound along a job rather than being incurred once" and "the recorded skill exceeds the exact total by a factor of at most $(1+\epsilon')^{k}$ and the recorded time by a factor of at most $(1+\epsilon')^{k+1}$", with $k$ up to $m$. So a job whose true skill sits at the stated bound $mB$ can be recorded at up to $(1+\epsilon')^{m}\,mB$, and its time at up to $(1+\epsilon')^{m+1}\cdot 2mB^{3}$. With $\epsilon'=\ln(1+\epsilon)/(2m+1)$ that overshoot factor is $(1+\epsilon)^{m/(2m+1)}\approx\sqrt{1+\epsilon}$, not $(1+\epsilon')$. The number of extra grid points needed is therefore $\approx m$ per coordinate, not "an additive constant".

**Why it matters.** Two problems. (a) Internal inconsistency: the same proof asserts single-rounding overshoot here and $k$-fold compounding there. (b) Taken literally, a table tabulated only to $(1+\epsilon')\cdot mB$ and $(1+\epsilon')\cdot 2mB^{3}$ would prune states that the optimal solution's own DP path actually reaches (its recorded totals can be up to $\sqrt{1+\epsilon}$ times the true ones), so the algorithm as described could fail to return a $(1+\epsilon)$-approximate solution. The final $O(m^{4}\epsilon^{-2}\log^{2}(mB))$ bound is unaffected, since $m = O(m\epsilon^{-1}\log(mB))$, so this is a fixable gap rather than a broken theorem.

**Fix.** Replace with: "rounding upward can carry a recorded total past the bounds above by at most $(1+\epsilon')^{m+1} \le (1+\epsilon)$, so the grids are extended by $O(m)$ further powers of $(1+\epsilon')$, which is dominated by the $O(m\epsilon^{-1}\log(mB))$ points already counted and leaves the counts $O(m\epsilon^{-1}\log(mB))$ unchanged."

### 15. The aggregation implicitly requires $\theta_A+\theta_M<1$ (and hence $\min(\tau_A,\tau_M)<1$), which is never assumed; the paper's well-definedness check covers only the other factor

**Page OA - 34** · `OA_C_CES_representation.tex, eq:phi / eq:gamma_solved and the sentence following it in sec:aggregationDist; claim carried into 6_extensions.tex, Section~\ref{sec:extensions.aggregation} ("The aggregation supports Equation~\eqref{eq:lr_macro_ces} for any such $\rho$")` · *math*
  
⚠️ **Single review** — only one of the two reviewers returned

> It also delivers $1-\theta_{\manualLetter}\,\skillAdjustedTimeLetter_{\manualLetter}^{\rho}=\theta_{\AIletter}\,\skillAdjustedTimeLetter_{\AIletter}^{\rho}>0$, so the real powers appearing throughout are well defined.

**Issue.** $\Gamma(u)$ and $\phi(\bar\alpha)$ both carry the factor $\bigl((1-\theta_A-\theta_M)/(\theta_A\tau_A^\rho)\bigr)^{1/\rho}$. The text verifies only that the denominator $\theta_A\tau_A^\rho$ is positive; it never verifies or assumes $1-\theta_A-\theta_M>0$. Nothing in the paper restricts $\theta_A,\theta_M$ at all — they are introduced only as "the weights of corresponding inputs" — and Restriction~\eqref{eq:ces_share_restriction}, $\theta_A\tau_A^\rho+\theta_M\tau_M^\rho=1$, does not deliver it. With $\rho<0$, non-negative weights and $\theta_A+\theta_M<1$ are jointly feasible only if $\min(\tau_A,\tau_M)<1$, since the smallest attainable $\theta_A+\theta_M$ consistent with the restriction is $1/\max(\tau_A^\rho,\tau_M^\rho)=\min(\tau_A,\tau_M)^{-\rho}$. That is a substantive, unit-dependent restriction on the firm's skill-adjusted time requirements and it is nowhere stated.

**Why it matters.** When it fails, the capital weight $1-\theta_A-\theta_M$ in eq:macro_agg_prod / eq:lr_macro_ces is negative and the "distribution of output across effective AI quality" $\phi$ is negative everywhere, so the object the appendix claims to derive in closed form does not exist. Section 6.1's statement that the aggregation "supports Equation~\eqref{eq:lr_macro_ces} for any such $\rho$" therefore needs a qualification, and so does the footnote's list of what the aggregate representation asks of the economy.

**Fix.** State $\theta_A,\theta_M\ge 0$ and $\theta_A+\theta_M<1$ as maintained assumptions, note that together with eq:ces_share_restriction they require $\min(\skillAdjustedTimeLetter_{\AIletter},\skillAdjustedTimeLetter_{\manualLetter})<1$, and extend the well-definedness sentence to cover $(1-\theta_A-\theta_M)^{1/\rho}$ as well as $\theta_A\tau_A^\rho$.

### ✅ 16. Tables SA.B.2 and SA.B.3 carry the column header "Probability that Focal Task (k) is AI-executed" although their dependent variable is is_automated

**Page SA - 11 and SA - 12** · `SA_B_alternative_definitions.tex, Tables \ref{tab:DWA_regression_aiAutomation_mainSample} and \ref{tab:DWA_regression_aiAutomation_GPTsample}; source rows in tables/randomTieBreak/allTasks_automated.tex and tables/randomTieBreak/GPT_automated.tex` · *inconsistency*
  
Flagged independently by **3** auditors

> & \multicolumn{6}{c}{Probability that Focal Task ($k$) is AI-executed} \\

**Issue.** The caption ("...on Task's AI Automation Likelihood"), the notes ("with dependent variable ($\text{is\_automated}_{k}$) instead of ($\text{is\_ai}_{k}$)"), and the body text all say the outcome is AI automation, but the spanning column header inside both tables still says the outcome is AI execution. The header is stale boilerplate copied from tables/noTasksWithRepetitiveDWAs/*_ai.tex.

**Why it matters.** The two automation tables are the only place the stricter outcome is reported, and their header labels the dependent variable as the very thing they are supposed to replace. A reader comparing Table SA.B.2 (N = 13,786) with Table 2 (N = 10,708) has no on-table signal that a different outcome is being estimated.

**Status.** ✅ Addressed 2026-09-04 (`228864c`). Both generating table sources now carry the right header. tables/randomTieBreak/allTasks_automated.tex:4 and tables/randomTieBreak/GPT_automated.tex:4 both read " & \multicolumn{6}{c}{Probability that Focal Task ($k$) is AI-automated} \\". Both files are still the ones \input by SA_B_alternative_definitions.tex at lines 60 and 78, so the change reaches Tables SA.B.2 and SA.B.3 as printed. No occurrence of "is AI-executed" remains in either header row.

**Fix.** Change the spanning header in tables/randomTieBreak/allTasks_automated.tex and tables/randomTieBreak/GPT_automated.tex to "Probability that Focal Task ($k$) is AI-automated".

### 17. Figure SA.B.2's note describes a specification that was not estimated and asserts a concentration result the underlying table does not show

**Page SA - 14** · `SA_B_alternative_definitions.tex, notes to Figure~\ref{fig:DWA_regression_aiAutomation_mainSample}, line 174` · *inconsistency*

> Notes: These graphs show that, among similar tasks appearing in multiple occupations, those sitting in a more AI-automated local context exhibit higher probabilities of being automated by AI, with the effect concentrated on the steps immediately adjacent to the task.

**Issue.** Two problems. (i) The regressors are not changed: SA.B says only "replacing the dependent variable ... with an indicator for whether step $k$ is AI-automated", the figure's own caption reads "Effect of Neighboring Tasks' AI Execution Status ...", and the table rows are labelled "Task ($k\pm j$) is AI-executed". So the local context is AI-*executed*, not "AI-automated". (ii) The claim that the effect is "concentrated on the steps immediately adjacent" is not what Table SA.B.2 shows: under DWA fixed effects (columns 4 and 6, the specification in panel (d)) the adjacent coefficients are 0.02 and 0.01, both insignificant, while the k+2 coefficient is 0.02; in column (5) the k+2 coefficient is 0.05* and significant. The body text on SA - 10 concedes the estimates "are smaller in magnitude and often statistically insignificant", which the note contradicts.

**Why it matters.** The note is the only prose interpretation attached to the automation figures, and it overstates a result the appendix text itself walks back. It also mislabels which variable defines the "local context", which is exactly the distinction (automated vs. executed) that SA.A spends a page explaining the paper cannot make in the data.

**Fix.** Rewrite the note to say "a more AI-executed local context" and to report the actual pattern: adjacent effects positive in the no-fixed-effect and SOC-fixed-effect columns but attenuated to insignificance once DWA fixed effects are included, consistent with the text on SA - 10.

### ✅ 18. "Definition 1" / "Definition 2" are used for the two EFI variants but are never defined, and collide with the paper's numbered Definitions 1 and 2

**Page SA - 17** · `tables/fragmentation_index_regression_execution.tex, row label of Table SA.B.5; cross-referenced from SA_D_prompt_robustness.tex lines 68, 157, 172 and SA_E_frequency_robustness.tex lines 192, 202` · *inconsistency*
  
Flagged independently by **3** auditors

> Empirical Fragmentation Index (Definition 2) & -0.78*** & -0.70*** & -0.68*** \\

**Issue.** The table row is labelled "(Definition 2)", and five other passages refer to "Definition 1" for the exposure-based EFI, including SA_E line 202: "We use our preferred exposure-based measure throughout (Definition~1 of Section~\ref{sec:fragmentation_prediction}, under which both E1- and E2-exposed tasks may form AI chains)". But Section 7.3 (sec:fragmentation_prediction) never numbers any definition: it defines "the empirical fragmentation index (EFI)" in prose only. There is no "Definition 1" or "Definition 2" of the index anywhere in the paper. Worse, the paper's actual numbered Definition 1 and Definition 2 are "Manual Step" and "Augmented Step" (3_shortrun.tex, printed page 10), so the labels point a reader at the wrong objects.

**Why it matters.** A reader of Table SA.B.5 cannot determine what "Definition 2" is, and the cross-reference in SA_E ("Definition 1 of Section 7.3") sends them to a section that contains no such definition. Since the whole point of Appendix SA.B.2 is that this is a different index from the main-text one, the label carries the distinction and is unusable.

**Status.** ✅ Addressed 2026-09-04. No undefined "Definition 1" or "Definition 2" EFI label survives in any compiled file. grep -rn "Definition" over SA_D_prompt_robustness.tex, SA_E_frequency_robustness.tex, SA_B_alternative_definitions.tex, 7_empirics.tex and SA_F_external_validation.tex returns only 7_empirics.tex:57 ("Definition~\ref{def:ai_chain}", a real numbered definition) and the SA_B section title. The five flagged prose sites are repaired in place: SA_D:68 now reads "the empirical fragmentation index, the share of occupation tasks exposed to AI (E1 or E2)"; SA_D:157 "for the exposure-based empirical fragmentation index"; SA_D:172 "for the exposure-based EFI"; SA_E:192 "The index is the preferred exposure-based measure"; SA_E:202 "(as defined in Section~\ref{sec:fragmentation_prediction}, under which both E1- and E2-exposed tasks may form AI chains)". The "(Definition 2)" row itself went away with tables/fragmentation_index_regression_execution.tex, which no longer exists. Residual but non-printing: three orphaned table sources still carry the string (tables/fragmentation_index_regression_exposure.tex:12, tables/apqc_fragmentation_index_regression.tex:12, tables/apqc_fragmentation_index_regression_exposure.tex:4); none of the three is \input anywhere (the only fragmentation table the paper inputs is tables/fragmentation_index_regression_combined.tex at 7_empirics.tex:231, which has no such label).

**Fix.** Either number the two indices explicitly where they are introduced in Section 7.3 (e.g. "Definition 1 (exposure-based EFI)" / "Definition 2 (execution-based EFI)"), using a name that does not collide with the paper's \begin{definition} counter, or replace "(Definition 2)" in the table row and "(Definition 1)" in SA_D/SA_E with "(execution-based)" / "(exposure-based)".

### ✅ 19. Table SA.B.5's note claims the count control isolates the arrangement component, but the control counts AI-exposed steps while the index is built from AI-executed steps

**Page SA - 17** · `SA_B_alternative_definitions.tex, notes to Table~\ref{tab:fragmentation_index_regression_execution}, line 280` · *logic*

> All specifications additionally control for the number of AI-exposed (E1 or E2) steps in the occupation, so that the fragmentation coefficient is identified from how the AI-able steps are arranged rather than from how many of them the workflow contains.

**Issue.** By the paper's own decomposition (eq. \eqref{eq:efi_decomposition}), EFI = 1 - k_w/m_w + r_w/m_w, and the argument in Section 7 is that the level term k_w/m_w must be netted out to read the index as dispersion. In this table k_w is the number of AI-*executed* steps, but the control is the number of AI-*exposed* steps. Conditioning on exposure does not remove an execution-based level term, so the fragmentation coefficient is NOT identified from arrangement here — it still contains the level of execution, which is the dependent variable. The note therefore asserts an identification property the specification does not have, and it contradicts the appendix's own body text, which says the measure "mechanically amplifies the negative relationship".

**Why it matters.** Readers are told the coefficient measures arrangement when it largely measures 1 minus the dependent variable. It also explains the otherwise unexplained jump in fit relative to the main table (R^2 of 0.84-0.89 versus 0.35-0.72) and the collapse of the exposure coefficient (0.11/0.07/0.07 versus 0.49/0.48/0.39), neither of which the text discusses.

**Status.** ✅ Moot 2026-09-04 (`1d64cb0 / 061f1d8`). The target no longer exists. tables/fragmentation_index_regression_execution.tex is gone from tables/ (ls shows only apqc_*, fragmentation_index_regression_{E1E2control,combined,exposure}.tex), and grep -rn "fragmentation_index_regression_execution" over all writeup .tex files returns nothing, so there is no Table SA.B.5 and no note carrying the "identified from how the AI-able steps are arranged" clause. SA_B_alternative_definitions.tex is now 256 lines and contains no execution-based EFI table, note, or discussion.

**Fix.** Delete the "so that the fragmentation coefficient is identified from ..." clause from this table's note, or replace it with the correct statement — that because the index's level term is built from execution and the control is built from exposure, the level component is not netted out and the coefficient should not be read as arrangement.

### ✅ 20. SA.B.2's "falsification-style check" is a non-sequitur: at the paper's own calibration, step-level independence reproduces essentially the observed -0.78 coefficient on the execution-based EFI

**Page SA - 18** · `SA_B_alternative_definitions.tex, Section SA.B.2 (app:additional_robustness_pred2), lines 290-294` · *logic*
  
Reviewers split; **settled by a third adjudicator**

> If AI execution were independent across steps, then conditional on AI exposure, increases in execution would primarily appear as isolated events and would not substantially reduce fragmentation.
> The strong negative relationship observed between the execution-based EFI and AI execution therefore indicates that AI adoption follows the clustered, chain-based pattern implied by the model.

**Issue.** The paragraph infers clustered, chain-based AI adoption from the magnitude of the execution-based EFI coefficient (-0.78, Table SA.B.5 col. 1), on the premise that "if AI execution were independent across steps... increases in execution would... not substantially reduce fragmentation." That premise is false at the paper's own calibration, and no benchmark for the independence case is reported anywhere in the subsection (grep finds no placebo, permutation or reshuffle in SA_B_alternative_definitions.tex). By the paper's own identity, Eq. (13), EFI = 1 - k/m + r/m, and for a random arrangement E[r|k] = k(m-k+1)/m, so EFI is already a near-deterministic decreasing function of the execution share under independence: E[EFI] = 1 - (m-1)p^2/m. Averaged over the occupation-level execution-share distribution the paper itself reports in SA.A (45.2% of the 872 occupations below 0.05, tail to 0.7; E[p] = 0.135, E[p^2] = 0.0411), independence gives E[EFI] = 0.961 against 0.958 for the paper's observed mean chain length of 1.45 — indistinguishable. (Evaluating at the mean share instead gives a misleadingly benign 0.983; EFI is quadratic in p, so that step is a Jensen's-inequality error, and the regression is identified precisely off the dispersion it discards.) Simulating the regression itself at the paper's calibration — 872 occupations, 20 tasks each, execution and exposure shares drawn from the two SA.A histograms, independence implemented as uniform within-occupation placement, i.e. exactly the paper's own Prediction #1 position-permutation placebo — yields beta_EFI = -0.77 under independence versus -0.80 to -0.82 for clustering tuned to L = 1.45, against the observed -0.78. The clustering signal in this coefficient is on the order of 0.03-0.05 standard deviations, while the coefficient itself ranges from -0.40 to -0.89 across plausible nuisance calibrations (mean workflow length, length dispersion, exposure-execution correlation). The diagnostic therefore has essentially no power to separate the null from the alternative, and the paragraph's inference does not follow. The preceding hedges do not resolve this: conceding that the measure "mechanically amplifies the negative relationship" and is "not intended to provide an independent characterization" disclaims testing Prediction #3, whereas the flagged sentences make the separate, stronger claim that the coefficient's strength diagnoses the form of adoption. The claim is also methodologically inconsistent with the rest of the paper, which uses 1,000-draw randomization placebos for exactly this kind of inference in Predictions #1 and #2.

**Why it matters.** SA.B.2's only substantive conclusion is that the execution-based EFI "functions as a falsification-style check on the model's underlying mechanism." As written it cannot falsify anything: the null (independent adoption) and the alternative (chain-based adoption) both predict a large negative coefficient, and the paper supplies no way to tell them apart. The paragraph also sits awkwardly beside the immediately preceding admission that the measure "mechanically amplifies the negative relationship."

**Status.** ✅ Moot 2026-09-04 (`1d64cb0 / 4c82916`). The target subsection is gone. grep for "additional_robustness_pred2", "falsification" and "execution-based EFI" over every .tex file in writeup/ returns no hits, and SA_B_alternative_definitions.tex now contains no \subsection at all (grep -n "subsection" returns nothing). The two flagged sentences ("If AI execution were independent across steps..." and "The strong negative relationship observed between the execution-based EFI and AI execution therefore indicates...") are absent from the file.

**Fix.** Supply the benchmark the paper already knows how to build, or drop the inference. Concretely: re-estimate Eq. (14) with the execution-based EFI on 1,000 position-permutation placebo datasets — holding each occupation's m_w and its set of AI-executed tasks fixed and randomizing only their positions, exactly as in Prediction #1 — and report the observed -0.78 against that null distribution. Only the gap between the two is evidence about clustering. If, as the calibration above suggests, the observed estimate falls inside the placebo distribution, delete the "diagnostic" and "falsification-style check" sentences (lines 291-294) and let the subsection stand as what the preceding paragraph already says it is: a robustness check on how fragmentation is measured. The clustering claim itself needs no rescue here — it is established properly in Section 7.1, where the observed mean chain length of 1.45 sits at the 100th percentile of both placebo nulls — so the fix costs the paper nothing substantive. In the interim, at minimum replace "would not substantially reduce fragmentation" with the correct statement that independence reduces fragmentation quadratically in the execution share, and note that the coefficient's magnitude is not by itself informative about arrangement.

### 21. SA_D claims prompt-induced ordering noise is not systematic across occupation types, but its own Figure 12 panels show large, statistically decisive gaps — largest exactly along AI exposure

**Page SA - 22** · `SA_D_prompt_robustness.tex, §"Robustness to Alternative GPT Prompts", text at lines 67–69 and Figure fig:kendall_tau_dist panels (b)–(d)` · *empirics*

> More importantly for our analyses, we find no evidence that GPT generates systematically different task orderings across different types of occupations.
> Specifically, we split occupations based on whether they fall above or below the median in the \emph{main prompt sample} along three dimensions discussed in Subsection~\ref{sec:fragmentation_prediction}: the empirical fragmentation index (Definition~1), the share of occupation tasks exposed to AI (E1), and the share of tasks executed by AI.
> Panels (b)\textendash(d) of Figure~\ref{fig:kendall_tau_dist} show that the mean Kendall's $\tau$ is very similar for above- and below-median occupations in all cases.

**Issue.** The three split panels the sentence cites report group means that differ substantially and in a consistent direction. Panel (b): below-median EFI mean tau = 0.56 (n=328) vs above-median 0.63 (n=540). Panel (c): below-median AI exposure 0.64 (n=428) vs above-median 0.56 (n=440). Panel (d): below-median AI execution 0.62 (n=430) vs above-median 0.58 (n=438). Using the full-sample distribution shown in Panel (a) (mean 0.601, implied SD 0.172, n=869), the two-sample t-statistics are 5.8, 6.9 and 3.4 respectively. These are not 'very similar', and they are the exact systematic pattern the appendix's opening paragraph promises to rule out ('we show that these differences do not exhibit systematic patterns across subsets of occupations', SA - 21).

**Why it matters.** The direction of the gap is the damaging one for the paper: GPT's sequences are least reproducible across prompts precisely for the occupations that are most AI-exposed and most AI-executed, and for the low-EFI (clustered) occupations. That is differential measurement error in the running variable of Prediction #3 and in the sample that drives Predictions #1 and #2, so it cannot be dismissed as noise that 'washes out'. As written, the appendix asserts the absence of the very confound its figures document, and Section 7 (p. 33) and the Introduction rely on this appendix for the claim that 'our findings are not artifacts of the particular prompt'.

**Fix.** Replace 'we find no evidence that GPT generates systematically different task orderings across different types of occupations' with an accurate statement of the gaps (e.g. 'mean tau is 0.64 among below-median-exposure occupations and 0.56 among above-median ones'), report the differences with standard errors, and add a sentence on why a ~0.08 gap in ordering stability does not threaten the results — e.g. by showing the headline estimates are unchanged when the sample is restricted to high-tau occupations, or by bounding the attenuation this differential noise implies for the EFI coefficient.

### 22. SA_E claims all three predictions are evaluated on a common set of occupations that a five-task workflow 'always admits', but the Prediction #2 figures use strictly fewer occupations in every cut (6 vs 20 at Hourly+ >=65%)

**Page SA - 27** · `SA_E_frequency_robustness.tex, lines 14–16, vs Figures fig:chainlength_frequency / fig:frag_frequency (N labels) and fig:neighbor_placebo_forest (N labels)` · *inconsistency*

> To keep the sample definition consistent across the three predictions, we restrict each cut to the occupations whose frequency-pruned workflow retains at least five tasks, and we evaluate all three predictions on this common set of occupations.
> [...] for Prediction~\#2 the neighbor regression uses the eligible tasks of the same occupations, namely those with two neighbors on either side, which a five-task workflow always admits.

**Issue.** The occupation counts do not match. The chain-length forest and the fragmentation heatmap use identical counts (871, 832, 725, 602, 475, 564, 388, 236, 112, 345, 162, 75, 20). The neighbor forest uses strictly smaller counts in every row (865, 792, 662, 523, 384, 493, 315, 178, 76, 282, 123, 49, 6). The gap is not explained by the stated eligibility rule: the loss comes from the Prediction #2 sample restrictions described in Section 7 (dropping tasks mapped to multiple DWAs, and keeping only DWAs that appear in more than one occupation), which this passage never mentions. The parenthetical 'which a five-task workflow always admits' asserts that no occupation is lost, when at Hourly+ >=65% only 6 of the 20 occupations survive into the neighbor sample and at Hourly+ >=50% only 49 of 75.

**Why it matters.** The stated point of the paragraph is that the three predictions are compared on a common sample so that differences across the grid reflect pruning rather than sample composition. That premise fails for Prediction #2, and the failure is largest exactly in the strict cuts where the paper attributes the weakening of the neighbor result to 'lost statistical power'. It also makes the note under Figure fig:frag_frequency ('Following the neighbor heatmaps ... the Hourly+ and >=65% cut is left blank, since the 20 occupations it retains are too few') read as if 20 were the neighbor-sample size, when it is 6.

**Fix.** Add the omitted restrictions: 'for Prediction #2 the neighbor regression additionally requires a task to map to a single DWA whose DWA appears in more than one occupation, so the neighbor samples are subsets of the common occupation set (865 of 871 occupations in the all-tasks row, 6 of 20 at Hourly+ >=65%).' Report both counts in the figures so the comparison is transparent.

### 23. SA_E states the chain-length result 'falls back toward the middle of the null only in the Hourly+ cuts', but 8 of the 12 pruned cuts in its own forest plot lie inside the 10–90 null band

**Page SA - 31** · `SA_E_frequency_robustness.tex, §app:frequency_robustness_chains, lines 103–104, vs Figure fig:chainlength_frequency` · *empirics*

> The result survives pruning.
> The observed chain length stays in the upper tail of its null across the inclusive and moderate cuts, at the 87th to 100th percentiles under the Daily$+$ logic and the 58th to 100th under SeveralDaily$+$, and falls back toward the middle of the null only in the Hourly$+$ cuts (the 40th to 68th percentiles), which are the smallest samples.

**Issue.** In the figure, the dot is colored red when the observed value falls outside the 10–90 reshuffle band and blue when inside. Only 4 of the 12 pruned cuts are red: Daily+ >=20% (99th), Daily+ >=35% (100th), Daily+ >=65% (96th) and SeveralDaily+ >=20% (100th). The other 8 are blue, i.e. statistically indistinguishable from the within-occupation position-reshuffle null: Daily+ >=50% (87th), SeveralDaily+ >=35% (58th), >=50% (81st), >=65% (80th), and all four Hourly+ cuts (40th, 68th, 58th, 55th). So the retreat into the null is not confined to Hourly+; it happens in one Daily+ cut and in three of the four SeveralDaily+ cuts, including SeveralDaily+ >=35% at the 58th percentile on 388 occupations, which is not a small-sample corner.

**Why it matters.** Prediction #1 is the paper's headline chaining result. The appendix's summary sentence (line 8, 'The chain-length and neighbor results continue to hold'), Section 7's 'recover the same patterns for all three predictions' (p. 33) and the Introduction's claim that the results are 'not artifacts of the particular way O*NET lists occupation tasks' all rest on this passage. A reader is told the failure is confined to the sparsest cuts when in fact the majority of pruned cuts cannot reject the null.

**Fix.** State the count accurately: 'the observed chain length lies outside its 10–90 null in 4 of the 12 pruned cuts (all Daily+ except >=50%, and SeveralDaily+ >=20%); it falls inside the band in the remaining cuts, including SeveralDaily+ >=35% (58th percentile, 388 occupations).' Then argue explicitly that the observed value never falls below the null mean and that the point estimate stays above 1.24, rather than claiming the result 'survives pruning' outright.

### 24. SA_E attributes the loss of significance in the neighbor heatmaps to 'the sparsest cells' with observations 'in the low hundreds', but its own heatmaps show insignificant cells with 654–2,533 observations

**Page SA - 32** · `SA_E_frequency_robustness.tex, §app:frequency_robustness_neighbor, line 144, vs Figure fig:neighbor_frequency_heatmap Panels (a) and (b)` · *empirics*

> The estimates weaken and lose significance only in the sparsest cells, where the number of surviving task observations falls into the low hundreds.

**Issue.** In Panel (b) (next task), the SOC-minor-FE heatmap reports an insignificant 0.017 at SeveralDaily+ >=20% with N=2,533 task observations, an insignificant 0.035 at SeveralDaily+ >=35% with N=854, and an insignificant 0.048 at Hourly+ >=20% with N=657. In Panel (a) (previous task), the SOC-major-FE heatmap reports an insignificant 0.032 at SeveralDaily+ >=35% with N=1,412, and the SOC-minor panel an insignificant 0.058 at Hourly+ >=20% with N=657. Under SOC minor fixed effects, the next-task effect is insignificant in every single non-Daily+ cell of the grid, regardless of sample size. None of these is a 'low hundreds' cell.

**Why it matters.** The sentence turns a substantive pattern — the next-task effect does not survive the most demanding fixed effects outside the Daily+ logic — into a mechanical power story. That matters because the same power story is used in Section 7 ('Confidence intervals widen as the restriction shrinks the sample ... but the direction of the headline results remains unaffected') to license the robustness claim.

**Fix.** State the pattern by specification rather than by sample size: the previous-task effect stays significant across the Daily+ cuts in all three specifications, while the next-task effect loses significance under SOC minor-group fixed effects in every SeveralDaily+ and Hourly+ cut including those with 650–2,500 observations. Then discuss whether this reflects power or genuine attenuation.

### ◐ 25. SA_E's neighbor placebo claims are contradicted by its own Figure 16 Panel (b): the next-task effect does not escape its null in all Daily+ cuts, and it sits mid-null in cuts with thousands of observations

**Page SA - 34** · `SA_E_frequency_robustness.tex, §app:frequency_robustness_neighbor, lines 175–177, vs Figure fig:neighbor_placebo_forest Panel (b)` · *empirics*

> The previous-task effect lies outside its 10-to-90 null across all four Daily$+$ cuts (the 91st to 99th percentiles) and at SeveralDaily$+$ $\geq 20\%$ and $\geq 50\%$ (the 93rd to 98th), and the next-task effect behaves similarly, escaping its null across the Daily$+$ cuts.
> The effects retreat toward the middle of their nulls only at the sparsest cuts, the Hourly$+$ $\geq 35\%$ and $\geq 50\%$ cuts and SeveralDaily$+$ $\geq 35\%$, where the surviving task observations number in the low hundreds and the reshuffle bands are correspondingly wide.

**Issue.** Two false statements. (i) 'the next-task effect ... escaping its null across the Daily+ cuts': in Panel (b) the Daily+ >=65% dot is blue (inside the band) in all three specifications (86th, 90th, 90th percentile), and the Daily+ >=50% dot is blue in the no-fixed-effects column (89th). (ii) 'The effects retreat toward the middle of their nulls only at the sparsest cuts, the Hourly+ >=35% and >=50% cuts and SeveralDaily+ >=35%': for the next-task effect the SeveralDaily+ >=20% dot is blue in all three specifications at the 63rd/69th/62nd percentile, on 493 occupations and 3,099 task observations (from Figure 15 Panel (b)); SeveralDaily+ >=50%, >=65% and Hourly+ >=20% are also blue. In the no-fixed-effects column of Panel (b), only 2 of the 11 pruned cuts (Daily+ >=20% and >=35%) are red. The previous-task effect in Panel (a) is likewise blue at Hourly+ >=20% (79th/80th) and SeveralDaily+ >=65% (85th), neither of which the sentence lists.

**Why it matters.** The final sentence of the subsection concludes that the neighbor effect 'weakens only through lost statistical power at the strictest thresholds'. That reading is not available: SeveralDaily+ >=20% has 3,099 task observations and a tight reshuffle band, and the observed effect still sits in the middle of the null. Prediction #2 is the paper's cleanest evidence for local chaining, and the frequency robustness check is cited in Section 7 and the Introduction as confirming it.

**Status.** ◐ Partially addressed 2026-09-04 (`b2d11a8 (first sentence only)`). Half fixed. The first flagged sentence was rewritten in b2d11a8 and is now accurate: SA_E:175 reads "lies outside its 10-to-90 null in ten of the twelve Daily$+$ cells (the 87th to 97th percentiles) and in five of the six SeveralDaily$+$ $\geq 20\%$ and $\geq 50\%$ cells (the 90th to 98th), and the next-task effect behaves similarly, escaping its null in nine of the twelve Daily$+$ cells" -- I counted the dots in the current figures and both counts check out (prev-task: 10 red of 12 Daily+ cells; next-task: 9 red of 12, the three blue being Daily+ >=65% at 77th/86th/88th). The second flagged sentence is untouched and still false. SA_E:176 still reads "The effects retreat toward the middle of their nulls only at the sparsest cuts, the Hourly$+$ $\geq 35\%$ and $\geq 50\%$ cuts and SeveralDaily$+$ $\geq 35\%$, where the surviving task observations number in the low hundreds". In the current Panel (b) the next-task dot is blue at SeveralDaily+ >=20% in all three specifications (59th/66th/60th percentile) on 493 occupations and 3,099 task observations, blue at SeveralDaily+ >=50% (47th/64th/75th), at SeveralDaily+ >=65%, at Hourly+ >=20% (59th/80th/86th, 1,171 observations) and at Daily+ >=65% (77th/86th/88th, 2,115 observations); none of those is in the sentence's list. SA_E:177's "weakens only through lost statistical power at the strictest thresholds" is also unchanged.

**Fix.** Report the counts rather than an 'only' clause: state in which cuts and specifications each dot escapes the 10–90 band, note explicitly that the next-task effect is inside the band at SeveralDaily+ >=20% (3,099 observations) in all three specifications, and drop the attribution of every retreat to lost power. Distinguish the previous-task effect (which does hold up across all Daily+ cuts and SeveralDaily+ >=20%/>=50%) from the next-task effect, which does not.

### 26. SA_E's closing sentence claims all three model implications 'operate' among frequent tasks, contradicting its own Prediction #3 subsection, which reports a wrong-signed, uniformly insignificant grid

**Page SA - 35** · `SA_E_frequency_robustness.tex, final line 212, vs §app:frequency_robustness_frag lines 205–208 and Figure fig:frag_frequency` · *inconsistency*
  
Flagged independently by **4** auditors

> In short, all three implications of our model appears to operate among the subset of frequently-performed tasks, and does not appear to be an artifact of the rarely-performed activities catalogued in O*NET.

**Issue.** Six lines earlier the same subsection states 'Pruning does not recover the relationship' and 'Across the $33$ pruned cells the coefficient is negative in $13$, the median absolute estimate is $0.05$, and no cell in the grid reaches significance at the $10\%$ level under any of the three specifications.' Negative in 13 of 33 means the coefficient carries the WRONG (positive) sign in 20 of 33 cells, and none is significant. Prediction #3 therefore does not 'operate' among frequent tasks; the appendix's own honest reading, two lines above, is that the O*NET null is unchanged. The paragraph even concedes the evidence cannot rule the channel out either.

**Why it matters.** This sentence is the summary a reader carries away, and it upgrades a null into a confirmation of the model's third implication. It also propagates: Section 7 (p. 33) says the frequency appendix lets the paper 'recover the same patterns for all three predictions' and that 'the direction of the headline results remains unaffected', which is not true of the fragmentation coefficient's sign in the pruned grid.

**Fix.** Rewrite as: 'In short, the chaining and neighbor results are not artifacts of the rarely-performed activities catalogued in O*NET, and the O*NET fragmentation null of Subsection~\ref{sec:fragmentation_prediction} is likewise a feature of the sample rather than a consequence of counting infrequent tasks as steps.' Correspondingly soften the Section 7 sentence about recovering 'the same patterns for all three predictions'.

### 27. The PCF chain-length test benchmarks text-derived labels against a step-order reshuffle null that the label transfer itself violates: adjacent PCF steps matched to the same O*NET task carry mechanically identical labels, so part of z = 6.6 can come from the transfer rather than from chaining

**Page SA - 47** · `SA_F_external_validation.tex, app:apqc_fragmentation_top, "Prediction \#1" paragraph (lines 391-403), with the label-transfer construction at lines 340-349` · *logic*
  
Reviewers split; **settled by a third adjudicator**

> Since no language model placed these steps in order, the contiguity is a property of the documented workflow
> rather than of the ordering procedure.

**Issue.** The PCF replication does remove the channel the closing sentence claims it removes — no language model ordered these steps, so the contiguity cannot be an artifact of the ordering procedure. But a second channel, specific to this corpus, is left unaddressed and the paper reports no diagnostic bounding it. AI execution labels on PCF steps are not observed: each step is embedded and assigned the label of its nearest O*NET task above a 0.71 cosine floor, with everything below coded not-exposed and not-executed (SA - 45). The paper itself notes that transferred labels are "a deterministic function of the matched task" (SA - 47, Prediction #2 paragraph), so two adjacent steps matched to the same task necessarily carry the same label. PCF leaves are nested (Process Groups > Processes > Activities > Tasks), and the documented order keeps siblings of the same parent contiguous, so textually near-identical steps sit next to each other — the paper's own example, "process/review requisitions" then "approve requisitions." Measurement error in the labels is therefore positively autocorrelated along the sequence, while the within-group step-order reshuffle null destroys that blocking by construction. The measured excess is small enough that very little blocking suffices: 1.16 vs 1.09 with 539 AI-executed steps implies 74.4 observed adjacent AI-executed bonds against 44.5 under the null, an excess of ~30 pairs out of ~12,957 within-group adjacencies (0.23%); nine process groups out of 525 in which a single contiguous sibling block of five near-paraphrase steps all matched the same AI-executed task would reproduce the entire effect. The paper's own Prediction #2 numbers corroborate the premise: of 137 DWAs appearing in more than one process group, only 9 contain both an executed and a non-executed step, so 93% are label-homogeneous. Neither null preserves within-group semantic blocking (the within-category reassignment, z = 10.2, destroys it too), and the appendix's other caveats — the 0.71 floor robustness, the "what survives the threshold" density discussion — bear on label density, not on adjacency structure. The claim that "the co-occurrence of AI-executed content behind it is unambiguous," and the appendix summary's claim that the findings are artifacts of neither channel, are therefore stronger than the test supports.

**Why it matters.** This is the appendix's only language-model-free replication of Prediction #1 and is cited in the introduction ("the fragmentation and chaining results continue to hold when we re-estimate them on the practitioner-ordered PCF corpus") and in Section 7.1's footnote. If the excess chain length is an artifact of semantic label transfer, the exercise does not deliver the independent confirmation it is presented as delivering.

**Fix.** Do not use the finding's original suggestion of reshuffling step order and re-deriving labels: a step's nearest-task match is position-independent, so that null is identical to the one already run. Instead, (i) add a null that preserves within-group semantic blocking — permute labels within parent element (Process or Activity) rather than across the whole process group, or permute contiguous sibling blocks rather than individual steps; (ii) report the within-group lag-1 autocorrelation of match success and of matched-task identity, and the share of adjacent step pairs whose best matches are the same O*NET task or the same DWA, so a reader can see how much of the ~30-bond excess the transfer channel can generate; (iii) as a robustness check, collapse runs of adjacent steps sharing a matched task into a single observation before computing chain lengths. Separately, soften the two sentences the test cannot support: "the co-occurrence of AI-executed content behind it is unambiguous" and the appendix summary's "artifacts neither of how LLMs order a list of tasks in a sequence nor of how O*NET catalogues tasks in occupations," and state explicitly that the PCF exercise removes the ordering channel but not the label-transfer channel.


---

## Minor (38)

### 28. Prop. 1(ii)'s monotonicity is stated for the middle step k only, but the text generalizes it to "a step" / "steps"; the set of AI-executed steps is not monotone in AI reliability

**Page 15** · `4_implications.tex, Sec. 4.1 (discussion of Prop. 1(ii)), line 62 ("A step that AI executes thus never reverts...") and line 97 ("...only ever add steps to those AI executes"); prop:ca_local part (ii)` · *logic*

> A step that AI executes thus never reverts to manual as its neighbors become more reliable, and the same holds when the step's own reliability improves alongside theirs.

**Issue.** Proposition 1(ii) establishes upward closure only for the *middle* step k of the block {k-1,k,k+1}, and only under the restriction that no chain crosses the block boundary. The two sentences drop both qualifications and assert monotonicity for an arbitrary step, and for the whole set of AI-executed steps. That general statement is false in the model: a step that AI executes can revert to manual when an immediate neighbor becomes more reliable, and the AI-executed set can shrink as reliability rises.

**Why it matters.** It reads as if Proposition 1(ii) delivers a general comparative-static ("AI improvements only ever expand AI execution"), which is one of the paper's headline takeaways from the result and is echoed in the Figure 3 discussion. The proposition does not support it, and the claim is false, so a reader who applies it outside the middle-step block (e.g. to a whole workflow, as the empirical sections do) is applying a result the model does not have.

**Fix.** Restrict both sentences to step k under the block restriction, e.g. "the step at the centre of the block never reverts to manual as its own or its two neighbors' reliabilities rise", and delete or explicitly qualify the generalization "improvements in AI reliability only ever add steps to those AI executes" (page 16, line 97) — noting that it describes what happens in Figure 3, where only q_{k-1} and q_k move, and does not hold once other steps' reliabilities change.

### 29. "Lowest precisely when the steps AI performs reliably sit beside one another" is not the correct characterization of the FI-minimizing order

**Page 19** · `4_implications.tex, Section 4.2, sentence after eq:fragmentation_closed_form` · *math*

> Across re-orderings of a given workflow the index is therefore decreasing in $\sum_{i=2}^{m} q_{i-1} q_i$, the total reliability of adjacent pairs of steps, and is lowest precisely when the steps AI performs reliably sit beside one another.

**Issue.** The first clause is right (FI = order-invariant terms − Σ q_{i-1}q_i). But the maximizer of Σ_{i=2}^m q_{i-1}q_i over orderings is the unimodal ("organ-pipe") arrangement, which puts the *least* reliable steps at the two endpoints — where each contributes only one adjacency — and the most reliable ones in the interior. Simply clustering the reliable steps together is not sufficient for minimality: distinct orderings in which the reliable steps all sit beside one another give strictly different index values, so "precisely when" identifies a set of orderings, not the minimizer.

**Why it matters.** The sentence is the paper's formal statement of the comparative static that the whole section (and the empirical fragmentation prediction) rests on; as written it names the wrong optimum, and the mechanism it suggests (cluster the reliable steps) is not the one the formula implies (put the unreliable steps at the ends).

**Fix.** Say instead that the index falls as reliable steps become adjacent, and that it is minimized by the unimodal ordering that places the least reliable steps at the two endpoints of the workflow — or drop "precisely" and keep it as a directional statement.

### 30. Section 4.3 attributes every upward jump to "longer AI chains," but the paper's own first threshold in Example 2 involves no chain at all

**Page 20** · `4_implications.tex, Section 4.3 (sec:nonlinear), paragraph before Example 2; echoed in the abstract (0_main.tex) and 1_introduction.tex l.106` · *logic*
  
Reviewers split; **settled by a third adjudicator** · Flagged independently by **2** auditors

> And when $\alpha$ reaches a level at which the firm re-optimizes its deployment strategy, extending a chain to absorb steps that were previously manual, the marginal return jumps up discontinuously, because improvements in quality now compound over a longer stretch of the workflow.
> We call the values of $\alpha$ at which the optimal strategy changes reorganization thresholds. Between them the returns to AI progress diminish, and at each of them those returns jump upward as longer AI chains become worth deploying.

**Issue.** The stated mechanism for the upward jump is chain lengthening ("extending a chain to absorb steps that were previously manual", "compound over a longer stretch", "as longer AI chains become worth deploying"). At the first threshold of the paper's own Example 2, alpha = 0.50, the optimum switches from "Both steps manual" to "Step 2 augmented", i.e. from no AI at all to a single augmented step, which the model defines as an AI chain of length one (Section 3.1). No chain is extended, nothing compounds over a longer stretch, yet g* jumps from 0 to 16.0 and the paper counts this as a reorganization threshold (Table OA.A.3, Figure 5). The actual mechanism in the proof of Lemma OA.B.4 is the envelope inequality phi'(alpha_0) <= 0, which uses nothing about chain length; the newly optimal curve merely has to be steeper.

**Why it matters.** The intuition paragraph misstates the mechanism of the paper's headline non-monotonicity result and is contradicted by the paper's own worked example two paragraphs later (which itself concedes "a later threshold does not necessarily introduce a longer chain than an earlier one"). The abstract inherits the error, promising jumps "as longer chains become viable", which is not what Lemma OA.B.4 delivers.

**Fix.** Restate the mechanism as "the newly optimal strategy has a steeper cost curve, because AI quality now enters through a larger total difficulty exponent" (which covers both a first deployment and a chain extension), and change "as longer AI chains become worth deploying" to "as more of the workflow is handed to AI". Match the abstract's "as longer chains become viable" accordingly.

### ◐ 31. Figure 5 plots only four of the five configurations the text and Table OA.A.3 enumerate, and its note points to a table that is not below it

**Page 21** · `4_implications.tex, Example 2 / fig:example5_nonmonotone notes; plots/example5_costs.png and plots/example5_marginalBenefit.png; cross-checked against tab:nonmonotone_costs` · *inconsistency*
  
Flagged independently by **2** auditors

> Panel (a) shows the cost of each AI strategy, with the optimal strategy given by their lower envelope (thick black line); the table below gives each strategy's cost and the range of $\alpha$ over which it is optimal.

**Issue.** The text on p. 20 says "the firm chooses among five configurations" and Table OA.A.3 lists five, but Panel (a)'s legend contains only four (Both steps manual; Step 1 manual, Step 2 augmented; Both steps augmented; Steps 1--2 chained). The missing curve is "Step 1 augmented, Step 2 manual", cost 3.5*alpha^{-11} + 8, which equals 11.5 at alpha = 1 — no curve appears at that value on the right edge of the plot. Panel (b) shows only three of the five. Separately, "the table below" is Table OA.A.3, which sits in the Online Appendix on page OA - 3, not below the figure on page 21.

**Why it matters.** The note asserts the panel shows "each AI strategy" when one strategy is absent, so a reader cannot verify the claim in Table OA.A.3's notes that the omitted configuration "never" minimizes cost, and is sent looking for a table that is not there.

**Status.** ◐ Partially addressed 2026-09-04 (`f9d4a4a`). Half fixed. The stale pointer is gone: the Figure 5 note (4_implications.tex:236) now reads "...lower envelope (thick black line). Table~\ref{tab:nonmonotone_costs} in Appendix~\ref{app:tables_and_figures} gives each strategy's cost and the range of $\alpha$ over which it is optimal." `git show f9d4a4a -- writeup/4_implications.tex` shows exactly this one-line change (its item N7). The missing-configuration half is not fixed: plots/example5_costs.png is unchanged (mtime Aug 21 22:04, untracked by git) and I read the image - its legend has four entries (Both steps manual; Step 1 manual, Step 2 augmented; Both steps augmented; Steps 1--2 chained) plus the minimum-cost envelope, with no "Step 1 augmented, Step 2 manual" curve, while OA_A_tables_and_figures.tex:134-142 still lists five configurations and 4_implications.tex:244 still says "the firm chooses among five configurations". The note still asserts "Panel (a) shows the cost of each AI strategy".

**Fix.** Either add the fifth curve to Panel (a) or change the note to "the cost of each AI strategy that is ever cost-minimizing" / "of four of the five AI strategies". Change "the table below" to "Table~\ref{tab:nonmonotone_costs} in Appendix~\ref{app:tables_and_figures}".

### 32. eq:totalcost_with_handoff minimizes over $\mathcal{P}(\mathcal{S})$, but a contiguous partition of the steps does not determine an AI deployment strategy or the costs $(\skillcost{b},\timecost{b})$

**Page 26** · `5_longrun.tex, Section 5.3 (sec:longrun.production), eq:totalcost_with_handoff and the sentence defining $\mathcal{P}(X)$` · *notation*
  
Reviewers split; **settled by a third adjudicator** · Flagged independently by **2** auditors

> Formally, we can write $\mathcal{P}(X)$ for the set of partitions of a sequence $X$ into contiguous subsequences.
> Then the full optimization problem faced by the firm can be expressed as
> \begin{align}
> \label{eq:totalcost_with_handoff}
> \min_{\T \in \mathcal{P}(\mathcal{S})} \
> \min_{\mathcal{J} \in \mathcal{P}(\T)} \
> \text{TotalCost}(\mathcal{J}; \T)

**Issue.** By Definition~\ref{def:ai_strategy} an AI deployment strategy is a contiguous partition of $\mathcal{S}$ *together with* a mode label for each block: a singleton block $\{s_i\}$ may be a manual step, with costs $(\manualSkill{i},\manualTime{i})$, or a length-one AI chain, with costs $(\AIskill{i},\AItime{i}/q_i)$. Both map to the same element of $\mathcal{P}(\mathcal{S})$, so $\skillcost{b}$ and $\timecost{b}$ are not functions of $\T\in\mathcal{P}(\mathcal{S})$ and the displayed objective is not well defined on the stated feasible set. The paper concedes exactly this in the very next clause: "both depending on the selected mode of operation for individual steps" — a choice that is not an element of $\mathcal{P}(\mathcal{S})$.

**Why it matters.** The firm's central optimization problem is stated over the wrong choice set. It also makes eq:totalcost_with_handoff inconsistent with the dynamic program that solves it (OA.B.5 and eq:longrun_recursion), which correctly treats "step $i$ is manual" and "step $i$ ends a chain begun at $r+1$ with $r=i-1$" as two distinct branches.

**Fix.** Define the choice set explicitly, e.g. let $\mathcal{A}(\mathcal{S})$ be the set of pairs (contiguous partition, mode labelling of singleton blocks) as in Definition~\ref{def:ai_strategy}, and write $\min_{\T\in\mathcal{A}(\mathcal{S})}$; keep $\mathcal{P}(\cdot)$ only for the job design $\mathcal{J}\in\mathcal{P}(\T)$, where it is exactly right.

### 33. Hand-off time $t^H$ is indexed by task in the Section 5.4 example, contradicting its step-level definition and the notation table's stated index convention

**Page 27** · `5_longrun.tex line 192 (and line 217) / Section 5.4 sec:longrun.specialization; conflicts with OA_A_tables_and_figures.tex line 54 and the Notes to tab:notation` · *notation*
  
Flagged independently by **3** auditors

> \[ \bigl(\skillcost{b},\, \timecost{b},\, \handofftime{b}\bigr)_{b=1,2,3} = (3,\,1,\,3),\quad (1,\,2,\,0.5),\quad (2,\,2,\,0). \]

**Issue.** $\handofftime{\cdot}$ ($t^H$) is a step-level primitive: 5_longrun.tex line 112 says "given step $s_i$, we write $\handofftime{i}$ for the additional hand-off time spent by a worker for whom the last step of their job is $s_i$", and Table OA.A.1 records it as "$t^H_i$ -- Hand-off time incurred when a worker's final step is $s_i$", with the Notes stating "Step-level primitives are indexed by $i$, task-level objects by $b$, and jobs by $j$". The worked example nonetheless writes $t^H_b$ for $b=1,2,3$ over tasks, and the following text says "the large hand-off time $\handofftime{1}=3$ that a boundary between tasks 1 and 2 would trigger", where the subscript 1 is a task index, not a step index. In this three-task workflow the tasks are not asserted to be single steps (the text only says "under a fixed AI deployment strategy $\T$, has $n=3$ tasks"), so $t^H_1$ is literally the hand-off of step 1, not of task 1.

**Why it matters.** The paper's index convention is stated explicitly in the notation table and is the only thing keeping the three-layer step/task/job hierarchy readable. The one place the hand-off cost is given a number is also the one place it is indexed off-layer, so a reader following the table gets the wrong object; it also propagates to Table OA.A.2, whose Panel (b) column header is the bare "$t^H + \sum t_b$".

**Fix.** Write the primitive at the layer it is defined on -- e.g. give the example's steps and set $\handofftime{}(J)$ per job, or state "each task here is a single step, so $\handofftime{b}$ abbreviates $\handofftime{i}$ for that step" -- and use $\handofftime{}(J_j)$ (the job-level object the table already defines) in the sentence about the boundary between tasks 1 and 2.

### 34. Figures 7 and 8 label the hand-off rectangles $h_1$, $h_2$, a symbol defined nowhere in the paper

**Page 27** · `5_longrun.tex, Section 5.4; plots/job_design.png (Figure~\ref{fig:lr_job_design}) and plots/combined_grid_with_handoff.png (Figure~\ref{fig:job_design}, Panel b)` · *notation*
  
⚠️ **Single review** — only one of the two reviewers returned

> {\fontsize{9.5pt}{10pt}\selectfont Notes: The blue bounded rectangles are tasks, with height the task's skill requirement ($\skillcost{}$) and width its time requirement ($\timecost{}$).
> The shaded areas are the wage bills of jobs.
> In the left panel the two tasks are assigned to two specialized workers, and the pink rectangle is the hand-off cost incurred between them.

**Issue.** The pink hand-off regions are labelled $h_1$ and $h_2$ inside the graphics. The paper never introduces $h$; the model, the notation table, and the body text all write $\handofftime{i} = t^H_i$, and the text discussing Panel~(b) refers to the very rectangle labelled $h_1$ as "the large hand-off time $\handofftime{1}=3$". In Figure~\ref{fig:lr_job_design} the same pink box carries $h_1$ inside it and $\handofftime{1}$ as its width label, so the two symbols appear side by side with no statement of how they relate (one is an area, the other a length).

**Why it matters.** The reader must guess that $h_b$ is the hand-off rectangle whose width is $\handofftime{i}$, in the two figures that carry the entire geometric intuition of Section 5.4. The notes define $\skillcost{}$ and $\timecost{}$ for the figure but not $h$.

**Fix.** Relabel the pink regions with $\handofftime{1}$, $\handofftime{2}$ (or state in the notes that $h_b$ denotes the hand-off rectangle at the boundary after task $b$, of width $\handofftime{}$ and height the job's total skill).

### 35. Section 6.1 presents Equation (9) as a three-input CES delivered by the aggregation, but the appendix states that the capital term is posited rather than derived (K is normalized to 1 and never enters the firm's problem) and that a genuine three-input CES is ruled out by the model's two-stage timing

**Page 30** · `6_extensions.tex, Sec. 6.1 (eq:lr_macro_ces) vs. OA_C_CES_representation.tex, Sec. OA.C.2 (paragraph beginning "Two features of this construction") and the text following eq:macro_agg_prod` · *inconsistency*
  
Reviewers split; **settled by a third adjudicator**

> Appendix~\ref{sec:aggregation} shows that the firm-level technology developed earlier, in which the content of tasks and of jobs is chosen rather than given, can be aggregated to

**Issue.** On p. 30 the body says the firm-level technology "can be aggregated to" Equation (9) "over economy-wide AI management labor $A$, manual labor $M$, and capital $K$" and calls this "the familiar constant elasticity of substitution (CES) form, with which the macroeconomic implications of automation are usually studied." Appendix OA.C is more guarded on two points the body does not carry. (i) The capital term is assumed, not obtained: "We assume capital is a fixed input whose productivity is common across firms, so that firms make their labor decisions conditional on a given capital stock and capital plays no part in the allocation"; "We normalize the economy's aggregate capital stock to $1$"; eq. (OA.C.9) is introduced with "Suppose specifically that the macro-level production function takes the form"; and OA - 30 states "Aggregate capital is normalized to $1$ ..., so the third term is constant at $1-\theta_{A}-\theta_{M}$; its exponent is part of the CES form we posit rather than something the aggregation derives." Consistently, in the aggregation identity (OA.C.11) the third term appears as the bare constant $(1-\theta_A-\theta_M)$ with no $K$. Capital also never enters the firm's problem: Section 3.3 (p. 12) says "labor is the only input used in production," and the micro production functions contain only labor. (ii) OA - 32 states that "recovering a genuine three-input CES requires firms to differ in at least two of their input requirements ... Our two-stage timing rules this out by construction," a consequence the body never mentions while calling eq. (9) the familiar CES form. Footnote 35 in Section 6.1 names the underlying assumptions ("capital productivity is common across firms, every firm organizes identically, realized effective AI quality is the single dimension along which firms differ") but not either consequence. Two qualifications to the original finding: the locus restriction is in fact partly stated in the body two paragraphs later ("the implications people draw from improving AI quality in a CES economy carry over along the margin the representation covers, the substitution between AI management labor and the rest of production"), so only the sharper "$M/Y$ and $K$ are fixed by construction" is missing; and the abstract and the p. 6 introduction are not inconsistent with the appendix — the abstract already hedges with "under additional assumptions ... admits a macroeconomic CES representation," and the p. 6 sentence is close to the appendix's own closing claim about a "micro-founded rationale for using aggregate CES production functions to study labor demand and productivity."

**Why it matters.** The abstract (p.1: "firm-level production admits a macroeconomic CES representation") and the introduction (p.6: "Our framework can thus serve as a micro-foundation for economy-wide production functions and be used to study the aggregate effects of AI on outcomes such as the labor market and total productivity") inherit the body's stronger reading. Since $M/Y$ and $K$ are constant along the derived locus, the representation cannot speak to labor demand per unit output or to capital-labor substitution, which are precisely the aggregate objects the intro promises.

**Fix.** In Section 6.1, add one clause to the paragraph introducing Equation (9) (or extend footnote 35) saying that the CES form over three inputs is posited: capital does not enter the firm's problem, aggregate capital is normalized to one, so the capital term is a constant whose exponent is imposed rather than derived, and the aggregation delivers substitution only along the AI-management-labor margin. Also note there, as the appendix does on OA - 32, that a genuine three-input CES would require firms to differ in at least two input requirements, which the two-stage timing rules out. Optionally sharpen the existing "margin the representation covers" sentence to say that $M/Y$ and $K$ are fixed along the locus. Trim the p. 29 framing that the firm-level adjustments determine "how output responds to capital and to labor," which the aggregation does not support for capital. No change is needed to the abstract or to the p. 6 introduction sentence: both are already hedged and match the appendix's own summary claim.

### 36. The DP cut variable is called ℓ in eq. (10) and r in eq. (11), and in eq. (11) r is the letter the paper's own convention binds to a chain's augmented last step

**Page 32** · `6_extensions.tex, eq:shortrun_recursion and eq:longrun_recursion; vs 3_shortrun.tex def:ai_chain and OA_A_tables_and_figures.tex tab:notation` · *notation*
  
Reviewers split; **settled by a third adjudicator**

> \min_{0 \le r < i}\ R\Bigl(r,\, \skillcost{} + \AIskill{i},\, \timecost{} + \tfrac{\AItime{i}}{\prod_{i'=r+1}^{i} q_{i'}}\Bigr)}_{\text{step $i$ ends a chain begun at } r+1}

**Issue.** Definition~\ref{def:ai_chain} and Definition~\ref{def:ai_strategy} (p. 11), Section~\ref{sec:longrun} (5_longrun.tex:46), Table OA.A.1 (p. OA - 1), OA\_C (line 41) and OA\_B (p. OA - 5) all use the same convention: an AI chain spans $(s_\ell,\dots,s_r)$ with $\ell$ its first step, $s_r$ its augmented last step, and time cost $\AItime{r}/\prod_{i=\ell}^{r} q_i$. In the two dynamic-programming recursions of Section~6 the minimand is the *cut* — the highest-indexed step that is NOT in the chain — and it is given a different letter in each. In eq:shortrun_recursion (p. 31) it is $\ell$, with the chain running $\ell+1,\dots,k$; in eq:longrun_recursion (p. 32) the identical object is $r$, with the chain running $r+1,\dots,i$ (OA\_B, p. OA - 22: "The cut $r$ is the highest-indexed step that is not added to this AI chain"). Two things follow. (a) One role carries two letters in equations one page apart, and the appendix reproduces the same split (OA\_B lines 523 and 572/583). (b) In eq:longrun_recursion, $r$ sits on the side of the chain opposite the one the convention assigns it, so $R\bigl(r,\dots,\AItime{i}/\prod_{i'=r+1}^{i} q_{i'}\bigr)$ reads, to anyone carrying Definition~4, as though it were evaluated at the chain's augmented endpoint when it is in fact evaluated one step before the chain's first step. The short-run $\ell$ is the milder case: it is one step below the convention's $\ell$, on the same side, not the opposite endpoint. Note what is NOT wrong: there is no off-by-one in the DP. Both cost expressions reproduce Definition~4 exactly for the chains $(s_{\ell+1},\dots,s_k)$ and $(s_{r+1},\dots,s_i)$, the degenerate cuts $\ell = k-1$ and $r = i-1$ give the correct singleton costs $\AItime{k}/q_k$ and $\AItime{i}/q_i$, and each equation carries an underbrace naming the chain's true first step ("a chain begun at $\ell+1$" / "at $r+1$"). The defect is a locally rebound symbol that a reader must undo, not an error in the recursion.

**Why it matters.** A reader who has internalized $(s_\ell,\dots,s_r)$ from Definition 4 will misread both recursions — most damagingly eq:longrun_recursion, where $R(r,\dots)$ looks like it is called at the chain's endpoint rather than one step before its start, which would be an off-by-one in the DP.

**Fix.** Use a single letter for the cut in both recursions, and one that the paper has not already bound to a chain endpoint — e.g. $j$: write $\min_{0 \le j < k}\bigl(C[j] + \AItime{k}/\prod_{i=j+1}^{k} q_i\bigr)$ in eq:shortrun_recursion and $\min_{0 \le j < i} R\bigl(j,\, \skillcost{}+\AIskill{i},\, \timecost{}+\AItime{i}/\prod_{i'=j+1}^{i} q_{i'}\bigr)$ in eq:longrun_recursion, and make the matching change in OA\_B (lines 523--530 and 572, 583, 602--604) so body and appendix stay in step. Alternatively, re-index each minimum over the chain's *first* step $\ell$, so the product reads $\prod_{i=\ell}^{k} q_i$ exactly as in Definition~4 and the state argument becomes $C[\ell-1]$ / $R(\ell-1,\dots)$. Either way keep the existing underbraces, which are what currently disambiguates the equations. Also fix the short-run bound for symmetry: eq:shortrun_recursion writes $\min_{\ell < k}$ with no lower limit while eq:longrun_recursion writes $\min_{0 \le r < i}$.

### 37. The measured "average AI chain length" is a maximal run of AI-executed steps, which the model does not identify with a single AI chain; the model itself has a region where two adjacent chains are optimal

**Page 36** · `7_empirics.tex, Sec. 7.1 and the closing paragraph of Sec. 7.2; model objects in 3_shortrun.tex def:ai_chain / def:ai_strategy and 4_implications.tex fig:ca_regions` · *inconsistency*
  
⚠️ **Single review** — only one of the two reviewers returned

> At the same time, the modest magnitude of 1.45 indicates that long AI chains remain rare, perhaps reflecting the early stage of AI adoption; our model predicts chains to lengthen as AI quality improves.

**Issue.** Empirically an "AI chain" is measured as a maximal run of consecutive AI-executed tasks (the paper explicitly refuses to distinguish augmented from automated labels, footnote 24 / SA.A). In the model, an AI deployment strategy is a partition into contiguous blocks each of which is a manual step or an AI chain, and nothing prevents two AI chains from being adjacent: two consecutive stand-alone augmented steps are two chains of length one, not one chain of length two. This is not a corner case — it is a labelled region of the paper's own Figure 3. There, in the orange band, the optimum is "$\{k-1, k\}$ chained ($k$ augmented)" while step $k+1$ remains, by the construction of that subsection, "a successor $k+1$ executed as a standalone augmented step". Steps $k-1$, $k$, $k+1$ are then a contiguous run of three AI-executed steps comprising two distinct chains. The measured statistic is therefore an upper bound on the model's mean chain length, not an estimate of it.

**Why it matters.** The 1.45 figure is used to make quantitative statements about the model's object — "long AI chains remain rare", "our model predicts chains to lengthen", and "implying that tasks two positions away rarely fall in the same chain as the focal task" (p.39). Those readings require run length = chain length, which the model does not deliver, so the inference about how much of a workflow a single verification spans is not tight.

**Fix.** Rename the statistic (e.g. "average run of consecutive AI-executed steps") and state that it weakly over-states the model's chain length because adjacent stand-alone augmented steps register as one run. Where the automation label survives, an automated-step-followed-by-AI-step run is the model-consistent chain and could be reported alongside.

### 38. The linear AI-able-share control removes the level term of the EFI decomposition but not the mechanical dependence of the arrangement term r_w/m_w on k_w and m_w, so beta_2 is a clean arrangement coefficient only if AI execution is linear in exposure

**Page 40** · `7_empirics.tex, Sec. 7.3, discussion of eq:efi_decomposition and eq:fragmentation_index_regression` · *empirics*
  
Reviewers split; **settled by a third adjudicator**

> To read the index as a measure of dispersion, we must control for the share of AI-able steps, which nets out the level term and leaves the arrangement component.

**Issue.** Including EFI_w and ai_exposure_w = k_w/m_w together is equivalent to including r_w/m_w and k_w/m_w, so beta_2 is the coefficient on r_w/m_w, exactly as the paper says. The imprecision is in calling what remains "the arrangement component." For a uniformly random arrangement of k_w AI-able steps among m_w, E[r_w] = k_w(m_w - k_w + 1)/m_w, i.e. E[r_w/m_w | k_w, m_w] = phi(1 - phi) + phi/m_w with phi = k_w/m_w, so EFI_w = 1 - phi^2 + phi/m_w + eta_w. Neither phi^2 nor phi/m_w lies in span{1, phi, k_w} or in that span plus the SOC/PCF dummies, so the regressor that survives the controls is not level-free: in simulations calibrated to the paper's own reported moments (R^2 of EFI on phi = 0.91, SD(EFI) = 0.26 in O*NET; R^2 = 0.68, SD = 0.06 in APQC) with random arrangements, 74-85% (O*NET) and 49-61% (APQC) of the residual EFI variance after the controls is a deterministic function of (m_w, k_w) rather than arrangement. The same point applies to the stronger claim in the notes to SA_B Table (SA_B_alternative_definitions.tex:280), that the coefficient "is identified from how the AI-able steps are arranged rather than from how many of them the workflow contains." The practical consequence is limited, however, and smaller than a first reading suggests. Contamination of beta_2 is exactly zero when AI execution is linear in exposure (simulated beta_2 = 0.00 in every calibration), and it requires curvature in the exposure-execution relation over and above the chain-count channel that r_w/m_w already captures. Its sign is minus the coefficient on phi^2, so a concave outcome -- the case a bounded share makes most plausible -- biases beta_2 upward and makes the paper's negative estimates conservative; only convexity produces a spurious negative. In the APQC calibration, where the paper's significant result sits, plausible mild-to-moderate convexity produces artifacts of only -0.04 to -0.12 against reported coefficients of -0.26 to -0.35, and the sample where the mechanical term is largest, O*NET, is where the paper reports a null it explicitly declines to interpret. Note also that what survives linear projection depends on the dispersion of phi, not its mean, so the relevant contrast between the two corpora is that exposure is more dispersed in O*NET, not that 44% is near the maximum of phi(1-phi). No specification in the paper checks this: footnote 28 only swaps k_w for m_w or drops the count control, and unlike Predictions #1 and #2 the fragmentation test is never benchmarked against a position-reshuffle placebo, which would net out E[r_w | k_w, m_w] directly.

**Why it matters.** The identification claim for the paper's only workflow-level test is that beta_2 isolates arrangement. If a nonlinear function of the exposure level survives the controls, the coefficient is not a clean dispersion effect -- which matters most in O*NET, where exposure share is high (44%) and the mechanical term phi(1-phi) is near its maximum.

**Fix.** Add a quadratic in ai_exposure_w (and ideally 1/m_w, or a flexible spline in the share) to Equation (14) and report that the fragmentation coefficient is unchanged; equivalently, report a random-order-normalized index r_w m_w / (k_w(m_w - k_w + 1)), which is level-free by construction. The cleanest single check is to extend the within-workflow position-reshuffle placebo already used for Predictions #1 and #2 to Prediction #3, since the placebo distribution of beta_2 embeds E[r_w | k_w, m_w] exactly. Also soften the two identification sentences: on p. 40, say that controlling for the share removes the level term of (13) but that the surviving arrangement term still has a random-order mean phi(1-phi) + phi/m that varies with the level and with workflow length; and correspondingly qualify the claim in the notes to the SA_B table that the coefficient is identified from arrangement rather than from how many AI-able steps a workflow contains.

### 39. Footnote 27 calls the EFI a "special case" of eq. (3) at $q_i = 0$, a value Definition 2's range $q_i = \alpha^{d_i} \in (0,1]$ excludes

**Page 40** · `7_empirics.tex, Section~\ref{sec:fragmentation_prediction}, footnote defining the empirical fragmentation index` · *inconsistency*
  
Reviewers split; **settled by a third adjudicator**

> The EFI is, after normalizing by the number of steps, the special case of the realized fragmentation expressed in \eqref{eq:fragmentation} in which every step takes one unit of time in either mode, $\manualTime{i} = \AItime{i} = 1$, and AI either always or never succeeds at a step, $q_i \in \{0,1\}$.

**Issue.** Footnote 27 (p. 40) reads: "The EFI is, after normalizing by the number of steps, the special case of the realized fragmentation expressed in \eqref{eq:fragmentation} in which every step takes one unit of time in either mode, $\manualTime{i} = \AItime{i} = 1$, and AI either always or never succeeds at a step, $q_i \in \{0,1\}$." Definition~\ref{def:augmented_step} states $q_i = \alpha^{d_i} \in (0,1]$, and $\alpha \in (0,1]$ (Table OA.A.1), so $q_i = 0$ is not attainable in the model — it is a limiting case ($d_i \to \infty$), not a special case. This is a wording/parameter-range slip only. The substance is exact, not merely approximate: with $\manualTime{i} = \AItime{i} = 1$ and $q_i \in \{0,1\}$, reading $1/q_i = +\infty$ at $q_i = 0$ (the ordinary convention, since $1/q_i$ is the expected AI cost of a step AI never completes), eq. (3) equals $(m_w - k_w) + r_w = m_w - (k_w - r_w) = m_w \cdot \text{EFI}_w$; I verified this against eq. (4) and eq. (13) for every AI-able pattern up to $m = 8$, and it reproduces the paper's worked examples ($5/5 = 1$, $4/5 = 0.8$). The term $\min\{\manualTime{i}, 1/q_i\}$ is therefore not undefined or limit-only — it is unambiguously $\manualTime{i} = 1$ — but that convention is nowhere stated in the paper, and $q_i = 0$ appears nowhere else in the source.

**Why it matters.** The footnote is what licenses reading the empirical index as the theoretical one, and it does so by evaluating the theoretical object at a parameter value the theory excludes. The substantive claim survives as a limit ($q_i \to 0$, with $\min\{\manualTime{i}, 1/q_i\} \to \manualTime{i} = 1$), so only the statement needs repair.

**Fix.** Reword the footnote so it does not place $q_i$ outside its stated range, e.g.: "...the limiting case of the realized fragmentation expressed in \eqref{eq:fragmentation} in which every step takes one unit of time in either mode, $\manualTime{i} = \AItime{i} = 1$, and AI either always succeeds at a step ($q_i = 1$) or never does ($q_i \to 0$, so that $\min\{\manualTime{i}, 1/q_i\} = \manualTime{i} = 1$)." Keeping "$q_i \in \{0,1\}$" is also fine if the parenthetical convention $1/q_i = +\infty$ at $q_i = 0$ is spelled out. No change to the arithmetic or to the two claims the footnote makes is needed.

### 40. Section 7.4's blanket claim that "the direction of the headline results remains unaffected" does not hold for Prediction #3, whose appendix result is a persisting null with the point estimates positive in 20 of 33 pruned cells

**Page 42** · `7_empirics.tex, sec:robustness (second robustness paragraph) vs. SA_E_frequency_robustness.tex, app:frequency_robustness_frag` · *inconsistency*
  
Reviewers split; **settled by a third adjudicator**

> In Appendix~\ref{app:frequency_robustness} we restrict the sample to progressively more frequently-executed tasks, which makes the surviving sequences more workflow-like, and recover the same patterns for all three predictions.
> Confidence intervals widen as the restriction shrinks the sample, since dropping infrequent tasks also removes AI-executed ones, which are already scarce to begin with, but the direction of the headline results remains unaffected.

**Issue.** On printed p. 42 the frequency-pruning robustness paragraph promises that the exercise "recover[s] the same patterns for all three predictions" and that, although confidence intervals widen, "the direction of the headline results remains unaffected." The second clause is a signed claim, and it is accurate only for Predictions #1 and #2, for which SA.E certifies direction explicitly ("persists ... without reversing sign", SA_E line 177). For Prediction #3 the O*NET headline is a null with negative but insignificant coefficients (-0.01, -0.09, -0.04, Table \ref{tab:fragmentation_index_regression_exposure}), and SA.E states "Pruning does not recover the relationship" and reports the coefficient negative in only 13 of 33 pruned cells (SA - 35). Since none of the 33 estimates is significant and the median absolute estimate is 0.05, the 13/33 split is noise around zero (two-sided binomial p = 0.30), not a sign reversal of a result — so the substantive pattern SA.E reproduces for #3 is the null, and there is no "direction" for pruning to leave unaffected. The problem is therefore loose summary language rather than a conflict between the two sections: SA.E's own framing sentences (lines 8 and 208) agree with the main text, and only its data sentence at line 206 shows the majority sign flip. The same looseness recurs at SA_E line 212, "all three implications of our model appears to operate among the subset of frequently-performed tasks," two sentences after "Pruning does not recover the relationship." A reader who does not open SA.E will infer that the negative O*NET fragmentation estimates survive pruning with their sign intact, which they do not.

**Why it matters.** This is a forward promise about appendix content that the appendix does not deliver as stated; a reader who does not open SA.E will believe the fragmentation channel survives pruning with its sign intact.

**Fix.** In 7_empirics.tex (lines 281-282), qualify the promise so it says only what SA.E delivers for each prediction, e.g.: "...and recover the same patterns for all three predictions: the chain-length and neighbor results retain their sign and their distance from the reshuffle nulls across the inclusive and moderate cuts, while the fragmentation null of Prediction #3 persists under every cut, with point estimates too imprecise to sign." Replace the blanket clause "but the direction of the headline results remains unaffected" with one restricted to Predictions #1 and #2. For consistency, soften SA_E_frequency_robustness.tex line 212 from "all three implications of our model appears to operate among the subset of frequently-performed tasks" to something like "none of the three results is an artifact of the rarely-performed activities catalogued in O*NET: the chain-length and neighbor effects survive pruning, and the fragmentation null is likewise unchanged."

### 41. The conclusion says the framework "thus helps explain why firms invest so heavily in AI capabilities," but the model contains no investment decision and AI quality is exogenous to the firm

**Page 44** · `8_conclusion.tex, paragraph on discrete reorganizations of work` · *logic*
  
Reviewers split; **settled by a third adjudicator** · Flagged independently by **2** auditors

> Our framework thus helps explain why firms invest so heavily in AI capabilities even when short-run returns appear limited, and it is consistent with the J-curve pattern of technology adoption \citep{brynjolfsson2021productivity}.

**Issue.** The sentence uses "thus" to present an explanation of firms' investment behavior as following from the paper's results, but nothing in the model bears on an investment choice. $\alpha$ is "the quality of the general-purpose AI technology" (3_shortrun.tex, Def. 2 and tab:notation, where it is listed as a primitive); the firm's only choice variables are the AI strategy $\T$ in eq. (1) and, in the long run, job design in eq. (8). There is no cost of acquiring or improving AI capability, no dynamics and no forward-looking behavior anywhere in the paper (a grep for "invest" over the full source returns firm investment only in this sentence; the footnote at 3_shortrun.tex mentioning an upfront subscription fee is used only to justify zero marginal cost of automated steps and is never a choice). The binding problem is not merely that the model is static: because $\alpha$ is the quality of the *general-purpose* technology, no firm in the model can buy an increment of it, so the non-convexity in $C^*(\alpha)$ is not a non-convexity in any firm's investment problem, and deployment itself is costless, leaving nothing to invest heavily in. Note that the model's return schedule is not in tension with lumpy investment — Lemma OA.B.4 makes $C^*$ the lower envelope of decreasing convex curves, so marginal returns are flat within a regime and jump at transversal thresholds, and in the paper's own Example 5 $C^*(0.40)-C^*(0.45)=0$ while $C^*(0.40)-C^*(0.95)=6.60$ — but converting that shape into a statement about how much firms spend requires an anticipation or option-value argument the paper does not make. Consistent with this, the three other places the paper invokes the J-curve (Sec. 1, Sec. 2, and the closing paragraph of Sec. 4.3) claim only a micro-foundation for the non-monotone return pattern and say nothing about firm investment; the conclusion is the only place the behavioral claim appears.

**Why it matters.** This is a conclusion-only claim about firm behavior that no result in the paper supports, and it is the sentence a reader is most likely to carry away as the managerial implication of the non-monotonicity result.

**Fix.** Rewrite the clause so it claims only what the results deliver — a statement about the shape of the returns to AI capability rather than about firms' spending. For example: "Our framework thus shows that the return to better AI can be flat for extended stretches and then jump once a reorganization threshold is crossed, so limited short-run gains are a poor guide to the eventual payoff, consistent with the J-curve pattern of technology adoption." If the authors want to keep the investment reading, add an explicit sentence that the model takes AI quality as an exogenous general-purpose technology and contains no investment margin, so the step from the non-convex return schedule to firms' spending decisions is a lumpy-investment/option-value argument outside the model, matching the weaker phrasing already used in Sections 1, 2 and 4.3.

### 42. Proof of Proposition 1 restricts success probabilities to $(0,1)$, but the model and the proposition allow $q_i = 1$

**Page OA - 5** · `OA_B_omitted_proofs.tex, line 10 and first bullet of the proof of part (i), Appendix OA.B.1 (app:ca_local_proof); Proposition 1 (prop:ca_local) in 4_implications.tex` · *proof*
  
⚠️ **Single review** — only one of the two reviewers returned

> Throughout, write $\minTime{i} = \min\left\{\manualTime{i}, \frac{\AItime{i}}{q_i}\right\}$ for the cheapest way of executing step $i$ on its own, whether manually or as a standalone augmented step, and take $q_{k-1}, q_k, q_{k+1} \in (0,1)$.

**Issue.** Definition 2 and Table OA.A.1 put $q_i = \alpha^{d_i} \in (0,1]$, and Proposition 1 imposes no restriction beyond $\manualTime{k} < \AItime{k}/q_k$. The proof adds $q_i \in (0,1)$ and then uses strictness materially: the first bullet of part (i) chooses $0<\mu<\tfrac{\manualTime{k}}{2}\tfrac{1-q_{k-1}}{q_{k-1}}$, "a nonempty interval because $q_{k-1} < 1$", which is empty at $q_{k-1}=1$; part (ii) is proved only on $Q=(0,1)^3$, so the upward-closure statement is not established at boundary triples with some $q'_i = 1$ — precisely the "AI becomes perfectly reliable" case the surrounding text emphasizes.

**Why it matters.** The proposition as stated is not proved on the full parameter range the model permits. The conclusions do survive (part (i) still follows from the second bullet, which does not need $q_{k-1}<1$; part (ii) extends to the closed box by continuity), so this is a coverage gap rather than a false claim.

**Fix.** Either state the restriction $q_i \in (0,1)$ in Proposition 1, or add one sentence noting that the $q_i = 1$ boundary follows by continuity of $V_0,\dotsc,V_3$ (and, for part (i), that the $\{k-1,k,k+1\}$ construction with $\mu = 1$ needs no restriction on $q_{k-1}$).

### 43. ℓ carries three incompatible meanings: a chain's first step, a chain's last step, and a task index

**Page OA - 13** · `OA_B_omitted_proofs.tex, proof of Lemma~\ref{lem:FI.lower.bound.4}; vs 3_shortrun.tex def:ai_chain; vs OA_C_CES_representation.tex` · *notation*

> Otherwise, the sequence $T = (s_1, \dotsc, s_\ell)$ is added to the task sequence as an AI chain, and we call it a \emph{non-individual} task. ... This procedure is then repeated beginning with the next step (which is $s_2$ in the former case, or $s_{\ell+1}$ in the latter case), until all steps have been added to a task.

**Issue.** Definition~\ref{def:ai_chain} and Table OA.A.1 make $\ell$ the *first* step of a chain, $(s_\ell,\dots,s_r)$. In the proof of Lemma OA.B.2 the greedy chain runs $(s_1,\dots,s_\ell)$, so $\ell$ is now the chain's *last* step; the same proof then writes $\overline{T}_b = (s_i,\dots,s_\ell,s_{\ell+1})$ with $\ell$ again the last step of $T_b$. In Appendix OA.C, $\ell$ becomes a *task* index: "$\sum_{T_{\ell}^{\manualLetter} \in J(b)} \manualSkill{\ell}$". A fourth, related use is eq:shortrun_recursion, where $\ell$ is the step immediately before the chain.

**Why it matters.** Within a single appendix a reader must reset the meaning of $\ell$ between subsections, and in OA_C the same letter that denotes a step in $d_b = \sum_{i=\ell}^{r} d_i$ (defined one page earlier) is reused as a task index on the very next page.

**Fix.** Reserve $\ell$ for the first step of a chain throughout. In the Lemma OA.B.2 proof use a neutral letter for the greedy chain's last step (e.g. $T=(s_1,\dots,s_j)$, next step $s_{j+1}$); in OA_C use $b'$ or $a$ for the summation task index.

### ✅ 44. Example OA.B.2's fragmentation index has additive constant 0.5, not 1

**Page OA - 15** · `OA_B_omitted_proofs.tex, line 364, Example OA.B.2 (ex:FI.lower.gap)` · *math*
  
Flagged independently by **3** auditors

> As $m$ grows large, the ratio between $\tfrac{2\sqrt{2}}{3}m$ and $1 + m \times 0.6213$ approaches $\tfrac{4\sqrt{2}}{9(\sqrt{2}-1)} \approx 1.52$.

**Issue.** For the instance $(\manualTime{i},\AItime{i},q_i)=(\sqrt2,1,1/\sqrt2)$ for all $i$, the exact fragmentation index is $FI(m)= m(1-q)\sqrt2 + q + (m-1)q(1-q) = 0.621320\,m + 0.5$ with $q=1/\sqrt2$, not $1 + 0.6213\,m$. The constant term is $q - q(1-q) = 0.5$: the leading component only exists when the last step succeeds (probability $q=0.707$), and step 1's failure creates no new component.

**Why it matters.** Purely a constant-term slip; the limiting ratio $\tfrac{4\sqrt2}{9(\sqrt2-1)}\approx1.517$ that the example is used for is unaffected. But the displayed expression overstates $FI$ by 0.5 at every $m$, so the finite-$m$ ratios a reader computes from it will not match the exact ones.

**Status.** ✅ Addressed 2026-09-04 (`f9d4a4a`). OA_B_omitted_proofs.tex:364 now reads "the ratio between $\tfrac{2\sqrt{2}}{3}m$ and $0.5 + m \times 0.6213$ approaches $\tfrac{4\sqrt{2}}{9(\sqrt{2}-1)} \approx 1.52$" (was "$1 + m \times 0.6213$"). I re-derived FI by brute-force enumeration over all 2^m failure realizations for the instance (t^M,t^A,q)=(sqrt2,1,1/sqrt2): FI(1)=1.12132, FI(3)=2.36396, FI(5)=3.60660, FI(8)=5.47056, FI(12)=7.95584, matching 0.6213203m+0.5 to 5 decimals at every m and never 1+0.6213203m. Limiting ratio 0.94281/0.62132 = 1.5175, consistent with the retained 1.52.

**Fix.** Write "the ratio between $\tfrac{2\sqrt{2}}{3}m$ and $\tfrac12 + m \times 0.6213$", or simply "$0.6213\,m + O(1)$".

### 45. "$D_c = \sum_{i \in c} d_i \ge 1$" is asserted but no assumption in the model bounds $d_i$ away from 0

**Page OA - 17** · `OA_B_omitted_proofs.tex, line 446, eq:strategy_cost in Appendix OA.B.3 (app:nonmonotone)` · *math*
  
Flagged independently by **3** auditors

> with each chain $c$ (with augmented endpoint $r_c$) contributing a term of degree equal to its total difficulty $D_c = \sum_{i \in c} d_i \ge 1$.

**Issue.** Nothing in the model restricts $d_i$. Section 3 and Table OA.A.1 only state $q_i=\alpha^{d_i}\in(0,1]$, which admits $d_i=0$ (an AI that always succeeds at step $i$) and $0<d_i<1$. So $D_c\ge 1$ is an unstated assumption presented as a definitional fact. The consequence is visible in the proof: with $D_c=0$ the term $\AItime{r_c}\alpha^{-D_c}$ is constant, so the parenthetical claim "(when $\T$ uses no chain $g^*(\alpha) = 0$ but is otherwise strictly decreasing)" is false -- a strategy can use chains and still have $g^*\equiv 0$.

**Why it matters.** Minor, because the lemma's conclusions survive with $D_c \ge 0$ (convexity and monotonicity of $C_\T$ need only $D_c\ge 0$), but as written the proof relies on a restriction the model never imposes and states a strictness claim that is false without it.

**Fix.** Either add "$d_i > 0$ for every step" to the standing assumptions of Appendix OA.B.3 (and write $D_c > 0$), or drop the "$\ge 1$" and weaken the parenthetical to "$g^*(\alpha)=0$ when $\T$ deploys no chain of positive total difficulty, and is non-increasing otherwise."

### ✅ 46. Marginal-benefit value at the second threshold of Example 2 does not reproduce: 133.9 should be 134.1

**Page OA - 19** · `OA_B_omitted_proofs.tex, app:nonmonotone, last line of the proof of lem:reorg_threshold` · *math*
  
Flagged independently by **4** auditors

> Both thresholds of Example~\ref{ex:nonmonotone} are transversal, with $g^*$ rising from $0$ to $16.0$ at $\alpha = 0.50$ and from $4.7$ to $133.9$ at $\alpha \approx 0.92$.

**Issue.** The second threshold solves 6 + 4/alpha = 4/alpha^12, giving alpha = 0.9239886. At that alpha the pre-jump marginal benefit is 4/alpha^2 = 4.685 (rounds to 4.7, correct) and the post-jump value is 48/alpha^13 = 134.145, not 133.9. The stated 133.9 corresponds to alpha = 0.92412, which is above the true threshold, so the number appears to have been read off a plotting grid rather than computed at the crossing.

**Why it matters.** It is the only quantitative claim substantiating the Lemma against the example; a reader who recomputes gets a different number and cannot tell whether the discrepancy is a rounding artifact or a modelling one.

**Status.** ✅ Addressed 2026-09-04 (`f9d4a4a`). OA_B_omitted_proofs.tex:506 now reads "...and from $4.7$ to $134.1$ at $\alpha \approx 0.92$" (was 133.9). Recomputed independently: solving 6 + 4/alpha = 4/alpha^12 by bisection gives alpha = 0.9239885872; pre-jump 4/alpha^2 = 4.68519 (rounds to 4.7) and post-jump 48/alpha^13 = 134.1453 (rounds to 134.1). The other two quoted values (0 to 16.0 at alpha = 0.50) are untouched and were not challenged.

**Fix.** Replace "133.9" with "134.1" (or state the threshold as alpha = 0.9240 and quote 134.1).

### 47. Footnote reports the two sides of a non-transversal threshold as 3170.5 and 3165.5, but both one-sided limits of $g^*$ equal 3168 exactly

**Page OA - 19** · `OA_B_omitted_proofs.tex, line 503 (footnote in proof of Lemma OA.B.4 / lem:reorg_threshold)` · *math*
  
Flagged independently by **3** auditors

> Writing $x = 1/\alpha$, their cost difference is $\phi = 12x^{4} - 65x^{3} + 102x^{2} - 12x - 56 = (x-2)^{3}(12x+7)$, which has a triple root at $x = 2$, and $g^*$ falls through $\alpha_0$ from $3170.5$ to $3165.5$ without jumping.

**Issue.** For the stated instance, $C_{\T}(\alpha)=65\alpha^{-3}+56+12\alpha^{-1}$ and $C_{\T'}(\alpha)=102\alpha^{-2}+12\alpha^{-4}$, so $g_{\T}(\alpha)=195\alpha^{-4}+12\alpha^{-2}$ and $g_{\T'}(\alpha)=204\alpha^{-3}+48\alpha^{-5}$. At $\alpha_0=1/2$ both equal exactly $3168$ ($195\cdot16+12\cdot4 = 204\cdot8+48\cdot32 = 3168$) -- which is precisely the content of the triple root ($\phi'(\alpha_0)=0 \Rightarrow g_\T(\alpha_0)=g_{\T'}(\alpha_0)$). The reported 3170.5 and 3165.5 are $g^*$ evaluated at $\alpha=0.4999$ and $\alpha=0.5001$, i.e. at two arbitrary points $10^{-4}$ either side of the threshold, not at the threshold.

**Why it matters.** As written the sentence asserts that $g^*$ drops by 5 across $\alpha_0$, which directly contradicts the lemma being proved one paragraph above ("at every reorganization threshold ... $g^*$ does not fall") and obscures the counterexample's actual point, namely that the left and right limits coincide so there is no jump. A reader checking the lemma against its own illustration finds them in conflict.

**Fix.** State the limits: "...which has a triple root at $x = 2$, so $g_{\T}(\alpha_0)=g_{\T'}(\alpha_0)=3168$ and $g^*$ passes through $\alpha_0$ continuously, declining within each regime rather than jumping."

### 48. Appendix OA.C indexes the step-level primitives $c^M$, $c^A$, $t^M$, $t^A$ by task, so Equation (OA.C.1) names the wrong step for any chain longer than one

**Page OA - 25** · `OA_C_CES_representation.tex lines 27, 34 (eq:new_wage) and 42-43; conflicts with OA_A_tables_and_figures.tex lines 47-49 (tab:notation, Panel B) and 5_longrun.tex line 60` · *notation*
  
Reviewers split; **settled by a third adjudicator** · Flagged independently by **2** auditors

> w_{\manualLetter} \left(\sum_{T_b^{\manualLetter} \in J} \manualSkill{b}\right) + w_{\AIletter} \left(\sum_{T_b^{\AIletter} \in J} \AIskill{b}\right),\label{eq:new_wage}

**Issue.** Table OA.A.1 defines $\manualSkill{i}$ as "Skill required to complete step $i$ manually" and $\AIskill{i}$ as "Skill required to verify the output of one AI attempt at step $i$", and the body (5_longrun) fixes the task-level object as $\skillcost{b}$, equal to $\manualSkill{i}$ for a manual step and $\AIskill{r}$ for a chain augmented at step $r$. OA.C instead writes $\manualSkill{b}$ and $\AIskill{b}$ with $b$ a task index, and likewise redefines $\manualTime{b}$/$\AItime{b}$ ("A task's time requirement ... is written $\timeCostLetter^{E(b)}_{b}$, which is $\manualTime{b}$ when task $b$ is a manual step, and $\AItime{b}$, the cost of verifying the chain's augmented endpoint, when it is an AI chain"). For a chain spanning several steps these subscripts name a step that is not the chain's augmented endpoint. In the paper's own running example (Figure 2/Figure 6), Task 4 is the chain over steps 4-6 augmented at step 6, so its verification skill and time are $c^A_6$ and $t^A_6$, but OA.C's convention writes them $c^A_4$ and $t^A_4$. The illustration OA.C gives ("the skill required for manual Task 1, $\manualSkill{1}$ ... the required skill for AI chain Task 2, $\AIskill{2}$") reads correctly only because in that figure Task 1 = step 1 and Task 2 = step 2 coincidentally.

**Why it matters.** Equation (OA.C.1) is the starting point of the whole CES aggregation (it feeds $\skillAdjustedTimeLetter_b$, then eq:task_level_prod, eq:micro_agg_prod and eq:alpha_bar). Under the notation the rest of the paper and Table OA.A.1 establish, the sums in (OA.C.1) pick the wrong step for every multi-step chain, so the appendix's formulas are literally wrong as written even though the intended object is well defined.

**Fix.** Use the task-level symbols the body already defines: write $\skillcost{b}$ (splitting the sum by mode via $E(b)$ rather than by superscript), or state explicitly at the top of OA.C that for the remainder of the appendix $c^{E(b)}_b$ and $t^{E(b)}_b$ denote the skill/time of task $b$'s executed step (its augmented endpoint for a chain), and add that convention to Table OA.A.1.

### ✅ 49. Job 1's per-unit-time compensation is given as an unweighted sum of skill costs two paragraphs before eq:new_wage defines it as a wage-weighted sum

**Page OA - 25** · `OA_C_CES_representation.tex, Sec. OA.C intro (line 25) vs eq:new_wage (OA.C.1)` · *inconsistency*

> The worker assigned to perform Job 1 is required to obtain not only the skill required for manual Task 1, $\manualSkill{1}$, but must also possess the required skill for AI chain Task 2, $\AIskill{2}$, as the worker's total compensation per unit of time is determined at the job level, and equals $\manualSkill{1} + \AIskill{2}$.

**Issue.** For Job 1 of Figure 6 (manual Task 1 plus AI chain Task 2), eq:new_wage seven lines later gives the compensation as $w_M c^M_1 + w_A c^A_2$, not $c^M_1+c^A_2$. The two expressions agree only under $w_M=w_A=1$, which is exactly the normalization OA.C sets out to abandon — the entire section rests on distinguishing the base wage rates, and the denominator $w_{E(b)}$ in the definition of $\skillAdjustedTimeLetter_b$ is meaningless if they are equal. The same object, a job's compensation per unit of time, is therefore defined two different ways on the same page.

**Why it matters.** The worked example is the reader's first encounter with the two-base-wage formulation, and it states the object it is illustrating incorrectly. The skill-adjusted time $\tau_b$, the aggregates $\tau_A,\tau_M$, and hence Restriction (OA.C.15) all inherit the weighted form, so the example is inconsistent with everything downstream.

**Status.** ✅ Addressed 2026-09-04 (`f9d4a4a`). OA_C_CES_representation.tex:25 now ends "...the worker's total compensation per unit of time is determined at the job level, and equals $w_{\manualLetter}\manualSkill{1} + w_{\AIletter}\AIskill{2}$." The `git show f9d4a4a -- writeup/OA_C_CES_representation.tex` diff is exactly this one-line change from "$\manualSkill{1} + \AIskill{2}$". It now agrees with eq:new_wage seven lines below (OA_C:32-33), which weights each skill cost by its base wage rate.

**Fix.** Write the example compensation as $w_{\manualLetter}\manualSkill{1} + w_{\AIletter}\AIskill{2}$, or move the example after eq:new_wage and state that the unweighted version is the normalized special case of the body.

### 50. The separability condition (OA.C.3) carrying the Leontief-to-Leontief aggregation uses an undefined difference operator that is 0/0 at generic allocations of a min technology, and is inferred from a property of the equilibrium allocation rather than of the production function

**Page OA - 27** · `OA_C_CES_representation.tex, eq:task_subst_rate_indep (OA.C.3), Sec. OA.C.1` · *proof*
  
Reviewers split; **settled by a third adjudicator**

> The fixed ratio of labor inputs implies that the rate of substitution between any pair of tasks is independent of the labor allocated to other tasks:
> \begin{equation}
> \frac{\partial}{\partial \labor{z}}\left(\frac{\frac{\Delta y}{\Delta \labor{a}}}{\frac{\Delta y}{\Delta \labor{b}}}\right)=0,\qquad\forall z\neq a,b.\label{eq:task_subst_rate_indep}
> \end{equation}

**Issue.** (OA.C.3) is the only justification given for the two-input representation (OA.C.4), and three things about it do not hold up — the conclusion itself is correct, so this concerns the argument only. (i) The symbol Δ appears exactly once in the entire paper, here, and is never defined, so no convention fixes what Δy/Δl_a means for the kinked function (OA.C.2). Under either two-sided or forward differences the object is 0/0 at every allocation where a and b are both slack — the generic case — and 0 or ∞ when exactly one of them binds. (ii) The only convention that rescues the inner ratio is left-hand (downward) differences at the equilibrium point where all constraints bind, which gives (τ_b α^{-d_b})/(τ_a α^{-d_a}); but the outer ∂/∂l_z still fails two-sidedly there, since lowering l_z makes z the unique binder and returns both differences to zero. So (OA.C.3) is not true as written under any stated or natural reading. (iii) The premise offered is the fixed input ratio, which is a property of the equilibrium allocation ("In equilibrium, allocation of labor to tasks satisfies..."), whereas the Leontief-Fisher condition invoked in the next paragraph is a property of the production function at arbitrary input bundles; the former cannot establish the latter. Separately, weak separability of the min in the AI/manual partition would deliver the nested aggregator min_{b≤k} l_b/(τ^A_b α^{-d_b}), not the linear aggregate l_A = Σ_{b≤k} l_b that (OA.C.4) and (OA.C.8) actually use, so the cited theorem is not by itself the step that yields (OA.C.4). Footnote 36 addresses only the strictness of the ∀z quantifier and does not bear on any of this. The conclusion is fine: I verified numerically that (OA.C.4)-(OA.C.7) reproduce (OA.C.2) exactly, so no downstream result is affected.

**Why it matters.** The step is presented as verifying the "necessary and sufficient condition" for the two-input aggregation, so a reader checking the appendix finds the key condition asserted rather than shown. The conclusion is in fact true and one line: $\min\{x_1,\dots,x_n\}=\min\{\min_{b\le k}x_b,\ \min_{b>k}x_b\}$, so eq:task_level_prod is weakly separable in the AI-chain and manual groups by inspection, and maximizing output subject to $\sum_{b\le k}l_b=l_A$ gives $l_b\propto\tau_b\alpha^{-d_b}$ and hence eq:micro_agg_prod with $\bar\alpha$ as in eq:alpha_bar.

**Fix.** Replace (OA.C.3) with the two lines that actually do the work. First, the nesting identity min{x_1,...,x_n} = min{min_{b≤k} x_b, min_{b>k} x_b} makes (OA.C.2) weakly separable in the AI-chain and manual groups by inspection, with no derivatives needed. Second, the efficient-allocation computation max{min_{b≤k} l_b/(τ^A_b α^{-d_b}) : Σ_{b≤k} l_b = l_A} = l_A/Σ_b τ^A_b α^{-d_b} = ᾱ l_A/τ_A (and likewise for the manual group) is what licenses using the simple sum as the aggregate input and delivers (OA.C.4) with τ_A, τ_M and ᾱ exactly as in (OA.C.5)-(OA.C.7); the paper's own fixed-ratio observation is this step, so only the computation need be written out. Keep Leontief (1947), Fisher (1965) and Felipe-Fisher (2003) as context for the aggregation question rather than as the load-bearing step. If (OA.C.3) is retained instead, define Δ, state the one-sided convention, and restrict the claim to the region on which it holds.

### 51. In Appendix OA.C, p denotes both the number of jobs and the output price

**Page OA - 30** · `OA_C_CES_representation.tex, Section~\ref{sec:agg_within} vs Section "Cross-Firm Aggregation"` · *notation*

> Normalize the output price to $p=1$, so that $w_{\AIletter}$ and $w_{\manualLetter}$ can be interpreted as real wage rates for a unit of skill-adjusted AI management and manual labor, respectively.

**Issue.** $p$ is the number of jobs throughout the paper (Definition~\ref{def:job_design}: "$\mathcal{J} = (J_1,\dots,J_p)$ ... $p$ such jobs"; Table OA.A.1: "Job design; a partition of the tasks into $p$ jobs"; OA - 26: "$|\J| = p$, indicating these $n$ tasks are grouped into $p$ jobs"; OA - 27: "hand-off tasks $n+1$ to $n+p-1$" and footnote "the final job $p$ requires no hand-off"). Four pages later the same appendix reassigns $p$ to the output price. Both meanings are live in the same derivation: the profitability condition that follows uses the price normalization while $\tau_{\manualLetter}$, which appears in it, is defined via a sum over $j=1,\dots,p-1$ jobs.

**Why it matters.** The reader must disambiguate $p$ by context inside a single appendix; worse, the price normalization is what makes the participation threshold $w_A\tau_A/(1-w_M\tau_M)$ interpretable, and that threshold sits two lines from expressions indexed by the job count $p$.

**Fix.** Rename the output price (e.g. $P=1$, or state "normalize the output price to $1$" without introducing a symbol), keeping $p$ for the number of jobs.

### 52. SA.A calls the E1-or-E2 exposure rule "conservative" without saying relative to what, and on the paper's own taxonomy it is the most inclusive exposure rule available

**Page SA - 1** · `SA_A_sample_construction.tex, lines 16-21` · *inconsistency*
  
Reviewers split; **settled by a third adjudicator**

> Unless explicitly mentioned otherwise, in all analyses we treat tasks with a human-assigned E1 or E2 label as exposed to AI and those with an E0 label as unexposed to AI.
> [...]
> This gives a conservative measure of exposure to AI for all O*NET tasks.

**Issue.** SA_A_sample_construction.tex line 21 asserts "This gives a conservative measure of exposure to AI for all O*NET tasks" without stating the comparison. On the label space the accompanying footnote defines exhaustively ("Any task that is neither E1 nor E2 is labeled E0"), treating E1-or-E2 as exposed is the broadest exposure rule the paper makes available, and it classifies 44% of O*NET steps as AI-able (7_empirics.tex line 259); the paper itself uses the narrower E1-only rule in Appendix SA.D. A reader will therefore most naturally attach "conservative" to the threshold choice, where it points the wrong way. Two other readings are defensible but nowhere stated: human rather than GPT-4 annotations lower measured exposure, and the rubric's bar (at least half the time saved at preserved human-level quality) leaves many tasks AI can partly help with inside E0. Scope note: this is an unexplained, potentially misleading adjective, not an internal contradiction and not a load-bearing step. The second use of "conservative" (SA_A line 79, printed page SA - 2, about coding fully-filtered conversations as manual) points in the same direction, not the opposite one; and no passage in the paper reads the exposure share as a lower bound or uses it to argue the results understate AI's reach (verified by grepping all .tex for "conservat", "understat", "lower bound", "overstat").

**Why it matters.** "Conservative" is load-bearing here: it is the sentence that tells the reader why the exposure share should be read as a lower bound, which in turn supports the paper's framing of its estimates as understating AI's reach. With the broadest exposure definition the framing does not hold.

**Fix.** State the referent in the same sentence, e.g. "Because we use human rather than GPT-4 annotations, and because the rubric requires at least a 50% time saving at preserved output quality, this is a conservative measure of exposure to AI" — or drop "conservative" and instead note that E1-or-E2 is the more inclusive of the available thresholds, cross-referencing the E1-only variant used in Appendix SA.D.

### 53. SA.A states the E1 threshold with the time comparison reversed relative to Section 7's statement of the same definition

**Page SA - 1** · `SA_A_sample_construction.tex, footnote at line 17` · *inconsistency*

> \cite{eloundou2023gpts} define E1 tasks as those that an AI can perform in at least half the time required by a human while preserving human-level output quality.

**Issue.** "Perform in at least half the time required by a human" states t_AI >= 0.5 * t_human, which is a condition satisfied by every task an AI performs no faster than half speed — the opposite of a time saving. Section 7 states the same rubric correctly: "whether access to an LLM would reduce the time required to perform a task by at least half while preserving output quality", i.e. t_AI <= 0.5 * t_human. The two statements of the same threshold are inconsistent, and the SA version is the wrong one.

**Why it matters.** SA.A is the appendix a reader consults for what the exposure label means; as written the stated criterion is unsatisfiable as a definition of exposure and disagrees with the main text.

**Fix.** Change to "in at most half the time required by a human", or mirror Section 7's phrasing ("reduce the time required to perform the task by at least half").

### 54. Figure SA.A.1's three bars sum to 17,920, not the 17,925 total tasks the figure states

**Page SA - 2** · `SA_A_sample_construction.tex, Figure~\ref{fig:task_execution_dist} (plots/ONET_Eloundou_Anthropic_GPT/all_labels_histogram.png)` · *empirics*

> {\fontsize{9.5pt}{10pt}\selectfont Notes: The automation and augmentation labels are drawn from Anthropic's Economic Index dataset, and the universe of tasks comes from the May~2023 O*NET release.}

**Issue.** The figure's box reports "Total Tasks: 17,925" and the three exhaustive execution-mode bars are Augmentation 1,626 (9.1%), Automation 721 (4.0%), Manual 15,573 (86.9%). 1,626 + 721 + 15,573 = 17,920, five tasks short of the stated total, and the exact shares sum to 99.97% rather than 100%. Since the three modes are defined to partition the universe (SA.A: "we treat tasks that do not appear in the Anthropic dataset, as well as those with 100\% filtered conversations, as manual tasks"), no residual category should exist.

**Why it matters.** The figure is the paper's headline description of the constructed dataset; a partition whose parts do not add to the whole signals five tasks dropped or double-counted somewhere in the merge, and the same denominator (17,925) is what makes the 13% AI-execution share quoted in SA.F reproduce.

**Fix.** Reconcile the five missing tasks — either the total is 17,920, or a fourth (unlabelled/unmerged) category exists and should be shown or explained in the note.

### 55. SA.B.1 describes the GPT similarity filter as selecting on "execution complexity," a criterion the cited prompt never uses, and never states the prompt's one-task-per-occupation cap

**Page SA - 9** · `SA_B_alternative_definitions.tex §app:additional_robustness_pred3 (p. SA - 9) vs SA_C_gpt_prompts.tex Prompt #2 (p. SA - 20)` · *inconsistency*
  
Reviewers split; **settled by a third adjudicator**

> Specifically, for each DWA, we ask GPT-5-mini to select the subset of tasks that are most similar in terms of skill requirements and execution complexity.\footnote{The prompt used to identify similar DWA tasks is provided in Appendix~\ref{app:prompts}.}

**Issue.** Two prose/prompt mismatches in the description of the GPT-5-mini similarity filter (SA_B_alternative_definitions.tex line 26, p. SA - 9). (1) The sentence says the filter selects "the subset of tasks that are most similar in terms of skill requirements and execution complexity," but Prompt #2 (SA_C_gpt_prompts.tex, p. SA - 20) asks GPT-5-mini to "Determine which tasks are similar in nature and in terms of their objectives, methods, or required skills"; the strings "complex," "difficult," and "effort" appear nowhere in that file. This is a single-word slip rather than a competing characterization: the paper's own preceding sentence (line 24, "objectives, execution nature, or required skills") and the notes to Tables SA.B.1 and SA.B.3 ("similar execution nature and skill characteristics") all say "execution nature" and all track the prompt. It still misleads, because "execution complexity" reads as holding task difficulty fixed, which the filter does not do. (2) The prompt imposes a restriction the prose never states — "Return only the most relevant task for every occupation" — so the filtered sample keeps at most one task per (DWA, occupation). This is operative and visible in the exhibit: Panel (a) of Table SA.B.4 red-flags the second same-DWA task of Bioinformatics Technicians (DWA 1) and of Climate Change Policy Analysts (DWA 2), and Panel (b) retains exactly one task per occupation per DWA. It is disclosed only in the verbatim prompt reproduced in Appendix SA.C, which the footnote points to; the prose ("the subset of tasks that are most similar," "a set of highly comparable tasks," and the Table SA.B.4 note's "procedure described in text") never conveys the cap. Note that the cap is not a bias problem — by keeping two same-DWA tasks in one workflow from both entering as focal observations, it mitigates rather than aggravates the mechanical-proximity margin that columns (5)-(6) of Table 2 address; the defect is disclosure of a binding sample restriction, not the direction of the estimates.

**Why it matters.** The GPT-filtered sample is the paper's tighter definition of 'the same step across occupations', so what the filter actually selects on determines what the robustness test shows. A reader told the filter conditions on execution complexity would read the exercise as holding task difficulty fixed, which it does not; and the undisclosed one-task-per-occupation rule is exactly the mechanical-proximity margin that columns (5)–(6) of Table~\ref{tab:DWA_regression_aiExecution_mainSample} are built to address.

**Fix.** In SA_B_alternative_definitions.tex line 26, replace "execution complexity" with "execution nature" so the sentence agrees with the prompt and with the paper's three other descriptions of the same filter, and state the cap explicitly, e.g.: "Specifically, for each DWA, we ask GPT-5-mini to identify the tasks that are similar in nature and in terms of their objectives, methods, or required skills, retaining at most one task per occupation." No change is needed to the table notes, which already use the correct wording.

### ✅ 56. SA.B's table note defines the exposure variable as E1-only while the paper defines AI exposure as E1 or E2 everywhere else -- and the same note's control variable uses E1 or E2

**Page SA - 17** · `SA_B_alternative_definitions.tex, notes to tab:fragmentation_index_regression_execution (l.278 vs l.280); vs. 7_empirics.tex l.25 and l.237, SA_A_sample_construction.tex l.16` · *inconsistency*
  
Flagged independently by **6** auditors

> The variable ``AI Exposure'' denotes the share of AI-exposed (E1) steps in the occupation, while the ``Empirical Fragmentation Index'' captures how dispersed AI-able steps are across the occupation's workflow.

**Issue.** Two sentences later the same note says "All specifications additionally control for the number of AI-exposed (E1 or E2) steps in the occupation", so within a single table note the exposure regressor is E1 and the exposure count control is E1-or-E2. The main text states "We use their human-generated labels and treat their E1 and E2 category as exposed to AI" and the main Table 3 note says "the share of steps carrying an E1 or E2 label"; SA.A states "Unless explicitly mentioned otherwise, in all analyses we treat tasks with a human-assigned E1 or E2 label as exposed to AI", and SA.E confirms "both E1- and E2-exposed tasks may form AI chains". SA_D l.68 and its Figure SA.D.2 subcaption "AI Exposure (E1) Split" carry the same E1-only wording.

**Why it matters.** The reader cannot tell which label set produced Table SA.B.5's coefficients, and if the note is right the robustness table is not comparable to the main-text specification it is said to mirror ("The specification mirrors the main-text regression exactly").

**Status.** ✅ Moot 2026-09-04 (`1d64cb0 / 061f1d8 (deletion); b2d11a8 (SA_D wording)`). The named target is gone: commit 1d64cb0 removed Subsection SA.B.2 and its table, and 061f1d8 deleted writeup/tables/fragmentation_index_regression_execution.tex. `grep -c fragmentation SA_B_alternative_definitions.tex` now returns 0, and no .tex references tab:fragmentation_index_regression_execution. Note that the secondary locations flagged in the entry's Issue/Fix WERE repaired: SA_D_prompt_robustness.tex:48 now reads "AI Exposure (E1 or E2) Split" and line 68 "the share of occupation tasks exposed to AI (E1 or E2)" (commit b2d11a8). No E1-only exposure wording remains anywhere.

**Fix.** If the estimation used E1 or E2 (as SA.A's blanket statement implies), correct the note to "(E1 or E2)" in both places; if it genuinely used E1 only, say so explicitly as an "explicitly mentioned otherwise" case and drop the claim that the specification mirrors the main text exactly. Apply the same fix to SA_D l.68 and the Figure SA.D.2 subcaption.

### ✅ 57. Table SA.B.5 reports "Clustered standard errors" while the main-text regression it says it mirrors exactly uses heteroskedasticity-robust errors, and the cluster level is never given

**Page SA - 17** · `SA_B_alternative_definitions.tex, notes to Table~\ref{tab:fragmentation_index_regression_execution}, line 275; conflicts with 7_empirics.tex Table~\ref{tab:fragmentation_index_regression_exposure} note` · *inconsistency*

> {\fontsize{9.5pt}{10pt}\selectfont Notes: Standardized coefficients. Clustered standard errors in parentheses. *** p$<$0.01, ** p$<$0.05, * p$<$0.1.

**Issue.** Section 7's Table 3 note for the same equation reads "Standardized coefficients. Heteroskedasticity-robust standard errors in parentheses", and SA.B.2 asserts "The specification mirrors the main-text regression exactly ... the only difference is that fragmentation is measured over realized AI execution rather than over exposure." Clustered versus robust is a second difference. The clustering level is also never stated, which matters because the regression has one observation per occupation (N = 872), so clustering must be at some coarser grouping (SOC major/minor) that is unspecified.

**Why it matters.** The reported significance stars on the EFI coefficient (-0.78***, -0.70***, -0.68***) depend on the variance estimator; a reader cannot reproduce or assess them without knowing the cluster level, and the "mirrors exactly" claim is inaccurate as printed.

**Status.** ✅ Moot 2026-09-04 (`1d64cb0 / 061f1d8`). Table SA.B.5 (tab:fragmentation_index_regression_execution) and its "Clustered standard errors" note no longer exist: SA.B.2 was removed in 1d64cb0 and the table source deleted in 061f1d8. `grep -rn "mirrors the main-text"` returns nothing. The clustered-vs-robust conflict is therefore gone by deletion, not repair. Main-text Table 3's note (7_empirics.tex:234) still reads "Standardized coefficients. Heteroskedasticity-robust standard errors in parentheses", unchanged.

**Fix.** Either change the note to "Heteroskedasticity-robust standard errors" to match Table 3, or state the clustering level explicitly and amend the "mirrors the main-text regression exactly" sentence to acknowledge the difference.

### 58. SA_E's worked example of the frequency cuts draws a conclusion about the Daily+ >=65% cut that does not follow from its own premises

**Page SA - 28** · `SA_E_frequency_robustness.tex, §app:frequency_robustness_construction, paragraph beginning 'To see how the cuts relate to one another'` · *logic*

> To see how the cuts relate to one another, consider a task that 60\% of surveyed workers report performing several times daily, with the remaining responses spread over less frequent categories.
> [...] Nothing stricter retains it, in either direction. No worker reports the task hourly or more, so it fails every Hourly$+$ cut, and its frequent share falls short of 65\%, so it fails the strictest threshold under both remaining logics.

**Issue.** Under the SeveralDaily+ logic the frequent share is exactly 60% < 65%, so the task does fail that cut. But under the Daily+ logic the frequent share is 60% PLUS the share reporting 'daily', and the premise only says the remaining 40% is 'spread over less frequent categories' — a set that includes 'daily'. If as little as 5 of those 40 percentage points sit in the 'daily' category, the Daily+ share is 65% and the task survives the Daily+ >=65% cut. The conclusion 'it fails the strictest threshold under BOTH remaining logics', and hence 'Nothing stricter retains it, in either direction', is not implied by the stated example.

**Why it matters.** The paragraph exists to teach the reader how the twelve logic-by-threshold cuts nest, and the illustration gets the nesting wrong in the one direction that is not mechanical (raising the threshold under a looser logic can be satisfied by pooling across categories). A reader who takes the example at face value will mis-read the >=65% column of every figure in the appendix.

**Fix.** Pin down the remaining mass, e.g. 'consider a task that 60% of workers report performing several times daily and 40% report performing monthly or less'. Then the Daily+ share is also 60% and the stated conclusion follows.

### 59. In the Allergists pruning illustration, the three categories of "episodic" tasks named as dropping out cover only seven of the nine dropped steps; the two uncovered ones (8 and 13) are the hands-on clinical drops, and step 13 is never mentioned anywhere

**Page SA - 30** · `SA_E_frequency_robustness.tex, §app:frequency_robustness_construction, paragraph beginning 'Panel~(a) shows the occupation's full sequenced workflow', vs Figure fig:frequency_pruning_example Panel (b)` · *empirics*
  
Reviewers split; **settled by a third adjudicator**

> The specialized diagnostic procedures (steps~3 and~4), the coordination and consultation activities (steps~9 and~10), and the research and continuing-education tasks (steps~14 through~16) are performed episodically rather than within the hour, and they drop out; what remains is the recurring clinical loop of documenting histories, examining patients, interpreting results, planning treatment, educating patients, and prescribing.

**Issue.** Panel (b) of Figure~\ref{fig:frequency_pruning_example} keeps steps 1, 2, 5, 6, 7, 11, 12 ("kept 7 of 16"), so nine steps drop: {3, 4, 8, 9, 10, 13, 14, 15, 16}. The sentence supporting "What the filter removes is what one would expect it to remove" names three categories — specialized diagnostic procedures (3, 4), coordination and consultation (9, 10), research and continuing education (14-16) — which account for seven of the nine. Step 8 ("Assess the risks and benefits of therapies") appears two sentences later only incidentally, as one of "the intervening manual steps 8 through 10", and step 13 ("Provide therapies, such as allergen immunotherapy or immunoglobin therapy") is never mentioned anywhere in the paper. Those two are exactly the drops that fit none of the three "episodic" labels, so the enumeration that carries the rhetorical point silently omits the two cases that would test it. Nothing false is stated: the preceding sentence gives the correct count ("seven of the sixteen tasks"), the figure note tells the reader that skipped numbers are the pruned steps, and the chain arithmetic (1.00 -> 1.33) is correct. The survivor gloss ("the recurring clinical loop of documenting histories, examining patients, ...") carries no step numbers and reasonably subsumes the unnamed survivor step 6, so it is not itself a defect. This is an incompleteness in an illustrative supplementary passage; no reported statistic depends on it.

**Why it matters.** The paragraph's rhetorical point is that 'What the filter removes is what one would expect it to remove'. The two unlisted drops are exactly the ones that undercut it: step 13 is a hands-on therapy-delivery task and step 8 a core clinical-judgement task, neither obviously 'episodic'. A reader who cannot reconcile the prose with the figure cannot check the claim.

**Fix.** Make the enumeration non-exhaustive, or complete it. Either replace "The specialized diagnostic procedures (steps~3 and~4), ... and they drop out" with "Among the nine tasks the filter removes are the specialized diagnostic procedures (steps~3 and~4), ...", or add the two remaining drops explicitly, e.g. "... together with the therapy-assessment and therapy-delivery steps (steps~8 and~13), which incumbents likewise report performing less often than hourly." A single added clause resolves it; no change to the figure or to any number is needed.

### ✅ 60. SA_E's body text says the fragmentation heatmap prints significance stars, while the same figure's note says none are shown, and the neighbor heatmap defers its star convention to that starless figure

**Page SA - 35** · `SA_E_frequency_robustness.tex, line 201 vs Figure fig:frag_frequency notes (line 193) and Figure fig:neighbor_frequency_heatmap notes (line 133)` · *inconsistency*
  
Flagged independently by **2** auditors

> Each cell reports the standardized coefficient on the empirical fragmentation index from re-estimating Equation~\eqref{eq:fragmentation_index_regression} for the corresponding cut, with significance stars and the number of occupations printed beneath

**Issue.** The note under the very same figure says 'No cell in the grid is significant at the 10\% level, so no significance stars are shown', and the body two lines later repeats 'no cell in the grid reaches significance at the $10\%$ level'. The figure indeed carries no stars. Separately, the neighbor heatmap's note says 'The heatmap layout and the significance stars are as in Figure~\ref{fig:frag_frequency}' — pointing, for its star convention, at a figure that displays no stars and never defines the thresholds; and fig:frag_frequency's note in turn defers its blank-cell convention to the neighbor heatmaps. Neither note ever states what *, ** and *** mean in the neighbor heatmaps, which do carry stars.

**Why it matters.** The reader is told to look for stars that are not there, and the only figure in the appendix that does carry stars (the neighbor heatmap, with ***/**/*) never has its significance thresholds defined, because both notes point at each other.

**Status.** ✅ Addressed 2026-09-04 (`f9d4a4a`). Both halves are fixed in commit f9d4a4a (verified with `git show f9d4a4a -- writeup/SA_E_frequency_robustness.tex`). Body line 201 now reads "...for the corresponding cut, with the number of occupations printed beneath (no cell is significant at the $10\%$ level, so no stars appear)" — the phrase "with significance stars and" is deleted. The neighbor-heatmap note at line 133-134 now reads "The heatmap layout is as in Figure~\ref{fig:frag_frequency}, except that..." plus an explicit legend "Standard errors are clustered at the DWA level, and significance stars come from the clustered coefficient test. *** p$<$0.01, ** p$<$0.05, * p$<$0.1." The circular deferral is gone.

**Fix.** Delete 'with significance stars and' from line 201, and in the neighbor heatmap note replace 'the significance stars are as in Figure~\ref{fig:frag_frequency}' with an explicit legend (*** p<0.01, ** p<0.05, * p<0.1).

### 61. Kendall's tau cost of a single inversion on a ten-step branch is stated as 0.02; it is 2/45 = 0.044

**Page SA - 41** · `SA_F_external_validation.tex, app:apqc_validation_tail ("Reading the left tail")` · *math*
  
Flagged independently by **4** auditors

> Kendall's $\tau$ is coarse when there are few steps to order. On a three-step branch it can only take the values $\{-1, -1/3, +1/3, +1\}$, so a single inverted pair already registers as a large negative number, whereas on a ten-step branch the same inversion costs $0.02$ and leaves $\tau$ near $+1$.

**Issue.** A ten-step branch has 45 pairs; one discordant pair gives tau = (44-1)/45 = 0.9556, a drop of 2/45 = 0.0444, not 0.02. The figure 0.02 is 1/45, the share of discordant pairs, which is half the tau decrement. Relatedly, on a three-step branch a single inverted pair gives tau = +1/3, not a negative value; it takes two inversions to reach -1/3.

**Why it matters.** The paragraph's whole point is a quantitative contrast between the tau penalty on short and long branches, and the number carrying that contrast is off by a factor of two (and the three-step half of the comparison is mis-stated in sign).

**Fix.** Write "whereas on a ten-step branch the same inversion costs $0.04$ and leaves $\tau$ near $+1$", and replace "a single inverted pair already registers as a large negative number" with something accurate, e.g. "a single inverted pair already costs two-thirds of the scale, and two send $\tau$ negative".

### 62. "That these three numbers coincide is itself informative" rests on comparing a set with its own 95.4% subset; the statistic that actually carries the content — accuracy on the 54 indeterminate pairs — is never reported

**Page SA - 43** · `SA_F_external_validation.tex, app:eventlog_design (lines 251-254) vs app:eventlog_results (lines 260-262); determinate share stated at line 210` · *logic*
  
Reviewers split; **settled by a third adjudicator**

> The pair level is the more informative of the two here, both because it is where the determinacy conditioning
> bites and because it is closer to what the paper's predictions actually use

**Issue.** The Results paragraph reports all-pairs precedence accuracy of 78.8%, determinate-pairs-only 78.9%, and determinacy-weighted 78.8%, then infers: "That these three numbers coincide is itself informative. The model does no better on the pairs that a real organization orders strictly than on pairs in general, so the aggregate is not being propped up by easy cases." But the determinate pairs are 1,124 of the 1,178 all-pairs sample (95.4%), and the determinacy-weighted figure reweights the same pairs, so the three numbers were confined to a narrow window by construction: holding determinate accuracy at 0.789, the all-pairs figure must lie in [0.753, 0.799] whatever the model does on the 54 indeterminate pairs. The coincidence is not literally mechanical — chance performance on the indeterminate pairs would have given 0.776 rather than 0.788, a 1.3pp gap — but all the information is in those 54 pairs, and the paper recovers none of it directly. Backing it out requires a 21.8x (=1178/54) amplification of a difference reported to three decimals; the unique integer solution consistent with both reported figures is 887/1124 determinate and 928/1178 overall, implying 41/54 = 75.9% on the indeterminate pairs, though rounding alone admits [74.6%, 78.9%]. The paper's underlying conclusion is therefore true on the comparison that matters (78.9% determinate vs 75.9% indeterminate), but the sentence claims informativeness for a comparison that could barely have come out otherwise. Note that the separate objection to "it is where the determinacy conditioning bites" (Design paragraph, line 254) does not hold: that sentence compares the pair level with the log level, and determinacy conditioning genuinely cannot be applied to a log-level Kendall tau, so it is correct as written.

**Why it matters.** The whole justification for adding the second benchmark rests on the determinacy conditioning doing work that the APQC benchmark cannot do. If it removes only 4.6% of pairs, the second benchmark's distinguishing feature is inert, and the inference drawn from the coincidence is unsupported.

**Fix.** Report pairwise accuracy on the indeterminate pairs directly — add a row to Table~\ref{tab:eventlog_ordering_validation} ("Indeterminate pairs only", 41 of 54 = 0.76 under the main prompt) — and recast the inference around the comparison that discriminates: the model orders the 1,124 pairs with a real order at 78.9% and the 54 concurrent pairs at 75.9%, so the aggregate is not propped up by easy cases. Replace "That these three numbers coincide is itself informative" with a sentence that notes the determinate set is 95.4% of all pairs and therefore that the all-pairs and weighted figures are near-mechanically close, so the determinate-vs-indeterminate split is what carries the content. Optionally, since a 3pp difference on 54 pairs is imprecise, give a confidence interval or report the split under the pooled ten alternative prompts (540 indeterminate pair-observations).

### 63. The 0.71 similarity floor is given two separate, unreconciled accounts of how it was chosen — manual inspection in the footnote, a label-density rule in the text

**Page SA - 46** · `SA_F_external_validation.tex, footnote at line 344 vs "How much the similarity floor matters" paragraph (lines 371-373)` · *inconsistency*
  
Reviewers split; **settled by a third adjudicator**

> We therefore choose the floor on the density of the transferred labels rather than on the estimates it produces.
> Raising it thins both the AI exposure and the AI execution content of the corpus, and exposure, the more common of the two labels, is the binding one.
> At $0.71$ the corpus retains $12\%$ of its steps as AI-exposed and $4\%$ as AI-executed, and every floor above it drops the exposure share below $10\%$, which is why we settled there.

**Issue.** The appendix states the provenance of the 0.71 label-transfer floor twice, in two places, each definitively and without reference to the other. Footnote 50 (SA - 45, line 344) says "We chose the $0.71$ similarity threshold after manual inspection of some of the matches." The body paragraph (SA - 46, lines 371-373) says the floor was chosen "on the density of the transferred labels rather than on the estimates it produces," and that "every floor above it drops the exposure share below $10\%$, which is why we settled there." The two are not logically contradictory — inspection could have argued against lower floors and the density criterion against higher ones — but the paper never says that, and the footnote's forward reference points to the later paragraph only as reporting "how the results move as it is varied," not as supplying the selection rule. As written, a reader cannot tell which rule was operative or reproduce the choice; the density rule additionally rests on a 10% minimum exposure share that is stated nowhere else and never motivated. The choice is a boundary case: Figure SA.F.3 shows 0.71 is the highest floor at which all three fixed-effect specifications' 90% intervals exclude zero (at 0.72 all three cross zero), which is precisely why the stated provenance should be auditable. The appendix does disclose the full path and the non-monotonicity, and reports the sign as negative at all eleven floors under all three specifications, so this is a transparency defect rather than evidence of selection on the estimates.

**Why it matters.** The floor is consequential: by the paper's own sensitivity analysis the no-fixed-effect fragmentation coefficient runs "from $-0.26$ at $0.65$ to $-0.63$ at $0.69$ and back to $-0.11$ at $0.72$," while the value at the chosen 0.71 is -0.35*** — the paper's only statistically significant evidence for Prediction #3. Section 7.3 presents that estimate without any reference to this sensitivity or to the threshold's provenance. A reader cannot audit a choice for which two incompatible rules are stated.

**Fix.** State one account of the choice in one place. If both considerations were in fact used, say so explicitly — e.g. that manual inspection of matches ruled out lower floors on quality grounds while the requirement that at least 10% of steps remain AI-exposed ruled out higher ones, so 0.71 is where the two meet — and motivate the 10% exposure minimum, which is otherwise an undisclosed criterion. Then either delete the footnote or recast it as a sanity check consistent with that single account. Separately, add a cross-reference from Subsection 7.3 (or the notes to Table 7) to Figure SA.F.3, so the main text discloses that the APQC fragmentation coefficient ranges from about -0.11 to -0.63 across floors from 0.65 to 0.75 while remaining negative throughout.

### 64. The appendix summary calls the two benchmarks' 2pp agreement "the strongest evidence" for sequence-over-topic, but that gap is well inside its own sampling noise and is the weakest item in the appendix's evidence

**Page SA - 48** · `SA_F_external_validation.tex, app:eventlog_results (lines 269-271) and app:external_validation_discussion (lines 430-432)` · *logic*
  
Reviewers split; **settled by a third adjudicator**

> That two independent benchmarks, one documented and one observed, agree to within roughly two percentage
> points on the statistic they share is the strongest evidence here that what the instrument recovers is
> sequence rather than topic.

**Issue.** The summary of Appendix app:external_validation designates one statistic as "the strongest evidence here that what the instrument recovers is sequence rather than topic": that the documented (0.700) and observed (0.679) adjacent-pair accuracies agree to within roughly two percentage points. Two problems, both narrower than originally filed. (i) Precision. The event-log side of that comparison rests on 125 adjacent pairs (133 activities minus 8 logs, from Table tab:eventlog_overview), clustered in 8 logs whose per-log accuracy varies widely. Even ignoring clustering, the binomial standard error on 0.679 is 0.042 and the standard error of the difference is 0.043, so the 95% interval on the 2.1pp gap runs from about -6pp to +11pp, and wider once clustered on 8 logs. The data cannot distinguish "agree to within two points" from "differ by ten points": the likelihood ratio for equality against a true 5pp difference is about 1.1. No standard error, confidence interval, or bootstrap is reported for either adjacent-pair statistic anywhere in the appendix or in either table. Running eleven prompts does not repair this, since all eleven are applied to the same 345 branches and the same 8 logs, so prompt replication removes prompt noise but not the corpus- and log-level variation that drives the interval. (ii) Rank. Invariance of accuracy across two very differently constructed corpora is a legitimate external-validity argument, but it is the weakest and least precisely measured item in the appendix, not the strongest. The two sentences immediately preceding it (directional accuracy above the 0.5 null on corpora with known order, exact sequence recovery in 24% of branches against 1/120 by chance) discriminate sequence from topical clustering more directly, as do the permutation nulls and the determinate-versus-all-pairs comparison. A secondary point compounds this: the two statistics are not quite the same estimand, since APQC adjacency is documented-order adjacency that includes branches the paper itself says have no true order, while event-log adjacency comes from a mean-first-occurrence global ranking and is not conditioned on determinacy.

**Why it matters.** This sentence is explicitly designated "the strongest evidence here," and Section 7.4 repeats it ("the documented and the observed benchmark agree closely with each other"). A reader is being asked to treat a coincidence of two unweighted averages as the load-bearing evidence for the construct claim.

**Fix.** Demote the sentence rather than delete the comparison. Replace "is the strongest evidence here that what the instrument recovers is sequence rather than topic" with a statement that the two benchmarks give similar accuracy despite differing radically in construction, and move the "strongest evidence" designation onto the statistics that actually discriminate the constructs: directional accuracy well above the 0.5 null on corpora whose order was fixed without any language model, exact recovery of the full sequence in 24% of branches against 1/120 for a five-step branch, and the permutation nulls (z = 22.6 for APQC, z = 6.8 for the logs). If the authors want to keep the agreement as evidence in its own right, attach uncertainty: report the adjacent-pair counts (1,572 for APQC, 125 for the event logs) in the table notes and a standard error or interval for each, clustered on branches and on logs respectively, and say plainly what the comparison can and cannot rule out. Consider also noting in the notes that the two adjacent-pair statistics differ in how the pairs are defined, so "the statistic they share" is an approximation. The §7.4 sentence ("the documented and the observed benchmark agree closely with each other") is descriptive and can stand as is.

### 65. Limitation #1's blanket clause "neither the PCF nor the event logs is crosswalked to O*NET tasks" is unqualified for the PCF, whose steps SA.F.2 does match to nearest O*NET tasks

**Page SA - 48** · `SA_F_external_validation.tex, app:external_validation_discussion (lines 443-444) vs app:apqc_fragmentation_top (lines 340-344)` · *inconsistency*
  
Reviewers split; **settled by a third adjudicator**

> First, they validate the ordering \emph{procedure} rather than the specific O*NET orderings used in the paper,
> since neither the PCF nor the event logs is crosswalked to O*NET tasks.

**Issue.** In the Summary and Takeaways (SA - 48, SA_F_external_validation.tex:443-444), limitation #1 reads: "First, they validate the ordering \emph{procedure} rather than the specific O*NET orderings used in the paper, since neither the PCF nor the event logs is crosswalked to O*NET tasks." The clause is correct for the event logs, which are never linked to O*NET. For the PCF it is unqualified: three pages earlier (SA - 45, line 342) Subsection app:apqc_fragmentation_top does "embed each leaf element and each labeled O*NET task with the language embedding model \texttt{all-mpnet-base-v2}, and match every step to its nearest task by cosine similarity," and the paper's own main text names the result "the O*NET-PCF match" (7_empirics.tex:261) and describes "matching each step to its nearest O*NET task" (7_empirics.tex:40). "Crosswalk(ed)" occurs exactly once in the paper, at line 444, so no narrower technical sense of the word is established that would exclude that match. The limitation's substantive conclusion is nonetheless correct, and so is its underlying logic once stated precisely: the match is lossy (only 2,067 steps, 15%, clear the 0.71 floor), is used for label transfer and never for ordering, and draws each step's partner from the entire labeled O*NET task universe, so it never reconstitutes the ordered task list of any single occupation. No estimate or conclusion changes; this is a one-clause wording issue that a reader who has just read SA.F.2 will trip over.

**Why it matters.** The reason given for the limitation is wrong, which obscures the real reason. The real reason is that the PCF branches are different work from the O*NET occupations, not that no mapping exists — indeed a mapping was built and used.

**Fix.** Qualify the clause rather than rewriting the limitation. For example: "...rather than the specific O*NET orderings used in the paper, since neither corpus provides a documented ordering over the task list of an O*NET occupation. The PCF steps are matched to their nearest O*NET tasks in Subsection~\ref{app:apqc_fragmentation_top}, but that match transfers labels rather than establishing sequence: it accepts only 15\% of steps and draws each partner from the whole O*NET task universe rather than from a single occupation, so it cannot recover the orderings the paper imputes." Keep limitations #2 and #3 as they stand.

---

## Verification gaps

Two things in this audit are known-incomplete, listed so they can be weighted properly.

1. **5 findings had only one reviewer**, lost to a spurious API safeguard error rather than to any judgement about the finding. Each is flagged inline.
2. **Figure content was read from rendered PNGs** where a claim depends on what a plot shows (Figures SA.A.1, SA.A.2, SA.B.2, SA.E.1–SA.E.5, Figure 5). Those readings were not cross-checked against the plotting code or the underlying data, which this audit did not have. Findings resting on a number read off a figure say so.

Independently re-derived during assembly, not taken from an agent: the second reorganization threshold of Example 2 solves `6 + 4/α = 4α⁻¹²` at α₀ = 0.9239886, where the marginal benefit rises from 4.685 to **134.145** — the proof of Lemma OA.B.4 reports 133.9.

---

## Checked and dismissed

59 further candidate issues were raised and then refuted under adversarial review — most often because a footnote, a preamble macro or a later sentence already resolved the objection, or because the finder's algebra was wrong and the paper's was right. They are listed here so the same ground is not covered twice.

- **Abstract and introduction state the CES aggregation result far more strongly than Appendix OA.C delivers** — `0_main.tex, abstract (final sentence) and 1_introduction.tex, "Aggregate Production" paragraph (lines 119-121)`
- **Introduction claims empirical counterparts to "all of the model's key objects"; most model primitives are unobserved** — `1_introduction.tex, "Empirical Evidence" paragraph (line 127) vs. 3_shortrun.tex / 5_longrun.tex primitives an`
- **Introduction says the PCF/event-log sequences are "ordered based on actual workplace data"; Appendix SA.F says the PCF is not a record of observed execution** — `1_introduction.tex, robustness paragraph (line 140) vs. SA_F_external_validation.tex, Sec. app:apqc_validation`
- **Introduction's definition of a "task" is broader than Definition 5, and its "tasks optimally collapse" is definitional, not an optimum** — `1_introduction.tex, lines 10-11 vs. 3_shortrun.tex, Definition 5 (def:ai_strategy) and Sec. 3.2`
- **Introduction attributes the empirical patterns to "how firms use AI", but the data contain no firms** — `1_introduction.tex, closing paragraph (line 143) vs. 7_empirics.tex data description and SA_A_sample_construct`
- **The short run's reduction to "a single worker type" is called without loss of generality but is a substantive assumption, and contradicts the multi-worker setup stated three sentences earlier** — `3_shortrun.tex, Section 3 preamble (sec:shortrun), sentence immediately before \subsection{Steps}; objective e`
- **Footnote on chain verification overhead states a sufficient condition that is not sufficient: the overhead is paid on every retry and inflated by 1/prod(q), the appended step's standalone cost is not** — `3_shortrun.tex, footnote attached to "One attempt at the chain therefore costs the same as verifying the last `
- **The admissible range of the step difficulty $d_i$ is never stated, and the stated ranges of $\alpha$ and $q_i$ admit values at which Definition 2's monotonicity claims are false and at which the appendix's assumptions are violated** — `3_shortrun.tex, Definition 2 (def:augmented_step), Section 3.1; cf. Table OA.A.1 (tab:notation) Panel A and OA`
- **Definition 5 classifies an AI chain as "automated or augmented", contradicting Definition 4, under which every AI chain ends in an augmented step** — `3_shortrun.tex, Definition 5 (def:ai_strategy), Section 3.2; conflicts with Definition 4 (def:ai_chain)`
- **The task cost $t_b$, and hence the objective in (1), is not determined by the object being minimized over: a singleton block has two possible costs** — `3_shortrun.tex, Definition 5 (def:ai_strategy) and eq:totalcost_shortterm, Sections 3.2-3.3`
- **The stated rule for how steps map into tasks ("the task absorbs whatever the AI has run automatically since the previous such point") is contradicted by the next sentence and by Definition 5** — `3_shortrun.tex, Section 3.2 (sec:model.tasks), paragraph introducing tasks`
- **$\AItime{i}$ is defined as prompting-plus-verifying time in the text but as verification time only in Definition 2 and Table OA.A.1, and the chain's per-attempt cost is charged at the endpoint's rate although the attempt requires prompting the whole run** — `3_shortrun.tex, Section 3.1 (sec:model.steps): definition of \AItime{i}, Definition 2, and the chain cost sent`
- **The chain's expected-time formula needs independence across retries and full re-execution of the chain on each retry; only independence across steps is stated** — `3_shortrun.tex, Section 3.1, sentence deriving the chain's expected cost (immediately after the overhead footn`
- **"Channel 1" is presented as a condition for step k to be automated, but it is only sufficient for AI execution; step k can end up augmented instead** — `4_implications.tex, Sec. 4.1, lines 35-38 (paragraph following eq:ca_test)`
- **Footnote 9's "applies unchanged" claim does not carry over to Proposition 1, whose pricing needs a manual cost for step k+1 that a composite chain does not have** — `4_implications.tex, Sec. 4.1, line 15 (footnote attached to "a successor $k+1$ executed as a standalone augmen`
- **Footnote 11 states the condition for the orange (augmented) band to disappear against the wrong benchmark** — `4_implications.tex, Sec. 4.1, line 94 (footnote on Figure 3's orange band)`
- **The stated reason why the marginal return jumps at a threshold is a non-sequitur, and is refuted by the paper's own non-transversal counterexample** — `4_implications.tex, Example 2 (ex:nonmonotone), final paragraph`
- **Section 4.3 claims the model reproduces "the same shape" as the productivity J-curve, but Lemma OA.B.4 rules out the decline that defines a J-curve** — `4_implications.tex, Section 4.3 closing paragraph; also 2_literature.tex l.11, 1_introduction.tex l.107, 8_con`
- **The long-run problem does not "nest" the short-run problem: with one job the objective is (Σc)(Σt), not Σt, and the two pick different AI strategies** — `5_longrun.tex, Section 5.3 (sec:longrun.production), sentence after eq:totalcost_with_handoff`
- **The "time channel" is claimed to cut both ways like the skill channel, but within the model shortening tasks pushes unambiguously toward consolidation; the offsetting force requires AI to lower $\handofftime{i}$, which the model treats as an exogenous primitive** — `5_longrun.tex, Section 5.4 (sec:longrun.specialization), second-channel paragraph`
- **Footnote claiming "the results rest on a considerably weaker version" of the additive-skill assumption is too strong: Proposition 4's dynamic program requires additivity, not just monotonicity** — `5_longrun.tex, Section 5.1 (sec:longrun.jobdesign), footnote attached to "The total skill required to complete`
- **Professor board-wiping example presumes a mid-job hand-off and hand-back, which Definition 6's contiguous one-worker-per-block jobs do not allow** — `5_longrun.tex, Section 5.2 (sec:longrun.handoff), closing paragraph`
- **The "c > 0" rule that gates job closure makes the long-run recursion return the wrong optimum whenever a task's skill cost is zero** — `6_extensions.tex, footnote to eq:longrun_recursion (Section 6.2, "The Long Run Optimization"); same argument i`
- **Section 6.1 presents eq:lr_macro_ces as a three-input CES over $(A,M,K)$ that the aggregation delivers, but OA_C says it is neither a genuine three-input CES nor valid off a one-dimensional locus** — `6_extensions.tex, Section~\ref{sec:extensions.aggregation}, sentence introducing eq:lr_macro_ces (and its foot`
- **Proposition 4's hypothesis "$[1/B, B]$ for some $B > 0$" is empty unless $B \ge 1$** — `6_extensions.tex, Proposition~\ref{prop:totalcost_optimization_dp}; identical wording in OA_B_omitted_proofs.t`
- **"Prophet" characterization does not price the strategy it describes, so FI is not literally its expected cost** — `4_implications.tex, lines 172-174, Section 4.2 (sec:fragmentation), interpretation of eq:fragmentation / Propo`
- **$m$ is used both for the number of steps and for a manual time cost inside the same appendix** — `OA_B_omitted_proofs.tex, line 127 (the "claim covering both coordinates" in the proof of Proposition 1, part (`
- **Capital is asserted as a firm input but appears in no firm-level production function; the constant it carries is what makes the whole aggregation non-degenerate** — `OA_C_CES_representation.tex, Sec. OA.C.2 "Cross-Firm Aggregation" (line 140-142), eq:macro_agg_prod (OA.C.9); `
- **Section 6's footnote of what the CES representation "asks more of the economy" omits the two limitations OA.C itself states — that the representation holds only along a one-dimensional locus and is not a genuine three-input CES** — `6_extensions.tex, Sec. 6.1, footnote to eq:lr_macro_ces; vs OA_C_CES_representation.tex lines 243-250`
- **Restriction (OA.C.15) has no admissible solution unless min(tau_A, tau_M) < 1 — an unstated condition on which the entire appendix depends** — `OA_C_CES_representation.tex, eq:ces_share_restriction (OA.C.15); and 6_extensions.tex, Sec. 6.1 ("for any such`
- **The derived heterogeneity distribution makes firm output strictly decreasing in effective AI quality over the standard elasticity range, and zero at the best AI quality** — `OA_C_CES_representation.tex, eq:phi (OA.C.16) together with the output normalization on p. OA - 31 and footnot`
- **Chain difficulty is written two different ways in the two appendices ($d(T_b)$ vs $d_b$) and appears in neither the notation table nor the body's symbol set** — `OA_B_omitted_proofs.tex line 247 vs OA_C_CES_representation.tex line 41; absent from OA_A_tables_and_figures.t`
- **Prediction #2 regresses one endogenous execution outcome on another; Proposition 1 is a comparative static in neighbors' success probabilities q, and the neighbors' AI-exposure — the model-consistent regressor, available in the data — is never controlled for** — `7_empirics.tex, Sec. 7.2 (sec:DWA_prediction), eq:DWA_regression_ai and its controls; theory in 4_implications`
- **Prediction #1 is said to follow "directly from Definition 4", but a definition has no empirical content, and Example 1 of the paper shows the model predicts NO chains when AI-able steps are dispersed** — `7_empirics.tex, Sec. 7.1 (sec:chainLength_prediction), opening paragraph; contrast with 4_implications.tex, ex`
- **The two chain-length placebos are described as isolating different margins, but the second nests the first and produces a numerically identical null, so it supplies no separate check** — `7_empirics.tex, Sec. 7.1 (sec:chainLength_prediction), placebo design and fig:aiChains_graphs_def1`
- **Table 2 note claims column (5) is "directly comparable with column (4)", but column (5) also adds the NumTasks-in-DWA-Occupation control that column (4) lacks** — `7_empirics.tex, note to tab:DWA_regression_aiExecution_mainSample, and the main-text inference immediately fol`
- **The version of Prediction #2 that matches the model's own definitional restriction (AI automation) is null under DWA fixed effects, and this is never mentioned in the main text or abstract** — `SA_B_alternative_definitions.tex, app:additional_robustness_pred3 and tab:DWA_regression_aiAutomation_mainSamp`
- **Prediction #3 is attributed to Proposition 2, which bounds cost and says nothing about the AI-execution share; in the exact calibration that makes the EFI equal the theoretical FI, the tested relationship is knife-edge or absent** — `7_empirics.tex, Sec. 7.3 opening ("This is the empirical content of Proposition~\ref{prop:fragmentation}") and`
- **Section 7.4 claims the frequency-pruning exercise recovers "the same patterns for all three predictions" with directions unaffected; SA.E reports the fragmentation coefficient flips sign in 20 of 33 cells and says pruning does not recover the relationship** — `7_empirics.tex, Sec. 7.4 (Robustness Tests), second paragraph; vs. SA_E_frequency_robustness.tex, app:frequenc`
- **Causal language for a purely cross-sectional workflow-level correlation with no exogenous variation in workflow arrangement** — `7_empirics.tex, Sec. 7.3 heading (sec:fragmentation_prediction) and the paragraph discussing tab:fragmentation`
- **The EFI footnote sets q_i in {0,1}, but the model restricts q_i = alpha^{d_i} to (0,1]; q_i = 0 is outside the model and makes the min in Eq. (3) well defined only as a limit** — `7_empirics.tex, footnote to the EFI definition in Sec. 7.3; vs. 3_shortrun.tex, def:augmented_step and eq:frag`
- **The AI-automation exercise is justified by a model restriction that does not imply it; the model's chains always terminate in an augmented step** — `SA_B_alternative_definitions.tex, Section SA.B.1, lines 33-34`
- **SA.A equates "job" with "occupation", contradicting the model's job-design definition and Section 7's own terminology mapping** — `SA_A_sample_construction.tex, line 11`
- **Every validation statistic measures ordinal direction, not adjacency, so none of them can rule out the topical-clustering alternative the appendix says it refutes** — `SA_F_external_validation.tex, opening of app:external_validation (lines 11-14) and app:external_validation_dis`
- **24% exact-order recovery is benchmarked against 1-in-120 and 1-in-40,320 chance rates, but by the paper's own figure discussion no branch with five or more steps was exactly recovered** — `SA_F_external_validation.tex, app:apqc_validation_results (lines 158-159) vs app:apqc_validation_tail (line 17`
- **SA.F's own opening claim ("our findings are artifacts neither of how LLMs order tasks nor of how O*NET catalogues them") contradicts its own limitation #1 and its admission that Prediction #2 could not be tested** — `SA_F_external_validation.tex, roadmap paragraph (lines 51-55) vs app:external_validation_discussion limitation`
- **"Underpowered by roughly an order of magnitude" for Prediction #2 on PCF uses the wrong comparison; the identifying variation is 9 DWAs against 534, nearly two orders of magnitude** — `SA_F_external_validation.tex, app:apqc_fragmentation_top, "Prediction \#2" paragraph (lines 405-413); compare `
- **The key auxiliary claim in the proof of Proposition 1(ii) reuses m, A, B and c, all four of which already denote model primitives** — `OA_B_omitted_proofs.tex, proof of Proposition~\ref{prop:ca_local} part (ii), "A claim covering both coordinate`
- **The total difficulty of a chain is written four different ways across body and appendices** — `3_shortrun.tex; OA_B_omitted_proofs.tex (app:fragmentation_proof and eq:strategy_cost); OA_C_CES_representatio`
- **Repeated references to "Definition 1 of Section 7.3" point to an object that does not exist; Definition 1 in the paper is "Manual Step" in Section 3.1** — `SA_E_frequency_robustness.tex, app:frequency_robustness_frag (and fig:frag_frequency note); SA_D_prompt_robust`
- **Proposition 2 is a loose constant-factor bound on COST, but it is invoked as a cross-workflow ordering result about AI-EXECUTION shares** — `4_implications.tex, prop:fragmentation and the paragraph following it; 7_empirics.tex, Sec. 7.3 (sec:fragmenta`
- **The short run is described as multiple workers with pre-drawn job boundaries but is formalized as a single worker holding the entire workflow** — `3_shortrun.tex, opening of Sec. 3 vs. Sec. 3.3 (sec:shortrun.production) and eq:totalcost_shortterm; 5_longrun`
- **Section 6.1 asserts that no firm in the model substitutes AI management labor for manual labor, contradicting the firm's own optimization in Sections 3-5** — `6_extensions.tex, Sec. 6.1, paragraph beginning "Arriving at Equation~\eqref{eq:lr_macro_ces} is not straightf`
- **Literature section claims the model micro-founds the Brynjolfsson-Rock-Syverson productivity J-curve, but the model has neither its mechanism nor its shape** — `2_literature.tex, "Complementarities, O-Ring Dynamics, and the Productivity J-Curve" paragraph; same claim rep`
- **Abstract's CES claim omits the appendix's own statement that the aggregation is not a genuine three-input CES and holds only along a one-dimensional locus** — `0_main.tex, abstract, final sentence; conflicts with OA_C_CES_representation.tex, "Two features of this constr`
- **Literature section says the returns to better AI arrive only at thresholds, contradicting Section 4.3** — `2_literature.tex, O-Ring/J-curve paragraph; conflicts with 4_implications.tex Sec. 4.3`
- **Literature section says entire chains can be automated, contradicting the paper's own definition of an AI chain** — `2_literature.tex, first paragraph, last sentence; conflicts with Definition def:automated_step and Definition `
- **SA_E asserts a result about steps two positions away while its attached footnote states those results are not reported** — `SA_E_frequency_robustness.tex, §app:frequency_robustness_neighbor, line 145` *(split verdict, dropped by the adjudicator)*
- **Attenuation of the automation estimates is attributed solely to automation prevalence, but the estimating sample also changes by 29%** — `SA_B_alternative_definitions.tex, Section SA.B.1, lines 36 and 38` *(split verdict, dropped by the adjudicator)*