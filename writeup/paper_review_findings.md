# Review: *Chaining Tasks, Redefining Work — A Theory of AI Automation*

**Reviewed:** 2 September 2026, against the working sources (`0_main.tex` and everything it inputs).
**Page numbers** refer to a fresh local build of the current sources (131 pp.: body pp. 1–46, then `OA - 1`…`OA - 35`, then `SA - 1`…`SA - 48`). Body pages are plain numbers; Online-Appendix pages are written `OA - n`. The compile is clean — 0 undefined references.

**Scope.** §§1–8, `preamble.tex`, Online Appendix OA.A / OA.B / OA.C, and the `tables/` and `plots/TikZ_visualization/` fragments the body inputs. **Excluded by request:** all six `SA_*.tex` files (they were not opened).

**Out of scope by request:** grammar, spelling, wording, style, typography, LaTeX formatting. Nothing below is a language comment.

**Method.** A first pass produced 120 candidate defects across twelve independent readings of the paper (model setup, §4 mathematics, each proof, the CES appendix, the empirics, cross-section consistency, cross-references, verbal arguments, every number, and the assumption inventory). Every candidate was then re-derived from the source by an independent adversarial pass instructed to refute it; **26 were killed** and are listed at the end so you can see what was checked and found sound. Numerical claims were verified by recomputation, and the two central propositions were stress-tested by brute force (details in §"What was verified as correct").


**Notation.** Symbols are those of the main draft (Table OA.A.1 and `preamble.tex`); the draft's macros are written out here so this file renders on its own. Step-level primitives are indexed by $i$, task-level objects by $b$, jobs by $j$; superscripts $M$ and $A$ denote the manual and AI modes of executing a step, and $H$ a hand-off between consecutive workers.

| Symbol | Definition in the draft |
|---|---|
| $\alpha \in (0,1]$ | Quality of the general-purpose AI technology |
| $s_i$, $\mathcal S = (s_1,\dots,s_m)$ | Production step $i$; the firm's sequence of $m$ steps |
| $d_i$, $q_i = \alpha^{d_i}$ | Difficulty of step $i$ for AI; probability AI completes it |
| $t^{M}_{i}$ | Manual time: time cost of completing step $i$ manually |
| $t^{A}_{i}$ | AI time: time cost of verifying one AI attempt at step $i$ |
| $t^{*}_{i} = \min\{t^{M}_{i},\,t^{A}_{i}/q_i\}$ | Cheapest standalone execution of step $i$ (OA.B) |
| $T_b$, $\mathcal T = (T_1,\dots,T_n)$ | Task $b$ (a manual step or an AI chain); AI deployment strategy |
| $t_{b}$ | Time cost of task $b$: $t^{M}_{i}$ if manual, $t^{A}_{r}/\prod_{i=\ell}^{r} q_i$ if a chain over $\ell,\dots,r$ |
| $c^{M}_{i}$, $c^{A}_{i}$ | Skill to complete step $i$ manually; skill to verify one AI attempt at it |
| $c_{b}$ | Skill cost of task $b$: $c^{M}_{i}$ if a manual step, $c^{A}_{r}$ if a chain augmented at step $r$ |
| $J_j$, $\mathcal J = (J_1,\dots,J_p)$ | Job $j$ (a contiguous block of tasks); job design |
| $t^{H}_{i}$, $t^{H}(J_j)$ | Hand-off time when a worker's final step is $s_i$; hand-off time of job $J_j$ |
| $\tau_b$, $\tau^{H}(J_j)$ | Skill-adjusted time of task $b$; of a job's hand-off (OA.C) |
| $l$, $w_{M}$, $w_{A}$ | Labor; the manual and AI base wage rates (OA.C) |
---

## Summary

| | Count |
|---|---|
| **Major** — a wrong result, a broken proof, or a design that does not support the conclusion drawn from it | 3 |
| **Medium** — a real defect a referee would require fixing that does not overturn a result | 9 |
| **Minor** — a local slip: typo-level math error, wrong index, imprecise normalization, inconsistent symbol, mis-pointed reference | 38 |

The three majors are independent of one another: one is a false lemma in OA.B (with an explicit counterexample), one is a false sufficient condition in a §4.2 footnote (with an explicit counterexample), and one is an identification problem in Prediction #2 (reproduced by simulation). None of them touches Proposition 1 or Proposition 2, both of which survived brute-force testing intact.

---

## Reconciliation status

Each finding below carries a **Status** line (Major and Medium) or a status marker on its opening line (Minor).

| | Status | Meaning |
|---|---|---|
| 🟢 | <span style="color:#1a7f37">**RESOLVED**</span> | The fix has landed in the draft |
| 🟡 | <span style="color:#9a6700">**PARTIAL**</span> | Some of the fix landed; the remainder is named in the status note |
| 🔵 | <span style="color:#0969da">**DEFERRED**</span> | Set aside deliberately, with the reason named in the status note |
| 🔴 | <span style="color:#cf222e">**OPEN**</span> | Not yet addressed |

| Severity | 🟢 <span style="color:#1a7f37">Resolved</span> | 🟡 <span style="color:#9a6700">Partial</span> | 🔵 <span style="color:#0969da">Deferred</span> | 🔴 <span style="color:#cf222e">Open</span> | Total |
|---|---|---|---|---|---|
| **Major** | 2 | 0 | 1 | 0 | 3 |
| **Medium** | 4 | 1 | 1 | 3 | 9 |
| **Minor** | 0 | 0 | 4 | 34 | 38 |
| **Total** | **6** | **1** | **6** | **37** | **50** |

Line numbers in the findings are those of the sources as reviewed on 2 September 2026. They drift as fixes land: `4_implications.tex` and `OA_B_omitted_proofs.tex` have both been edited since (MAJ-1), so locations in those two files may sit a few lines lower than stated.

---

## Major

### MAJ-1. Lemma OA.B.4's upward jump — and §4.3's headline claim — are false as stated, not merely unproven

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span> Lemma OA.B.4 now states the weak inequality plus a transversality condition for the strict jump, with the tangency counterexample in a footnote and the genericity argument in the proof; `4_implications.tex:206` and `:250` were brought into line. `:205` and `:244` were left as they stand (see the status note there).

**Where:** Appendix OA.B, `OA_B_omitted_proofs.tex:448–493` — **page OA - 18**; restated in §4.3 at `4_implications.tex:205, 206, 244, 250` — **pages 20–21**.
**Text:** “at every reorganization threshold (where the optimal AI strategy changes), $g^*$ jumps upward”; “the marginal return jumps up discontinuously”; “A differentiable function with such a sign change has nonpositive derivative at $\alpha_0$”.

**Issue.** The proof derives only the weak inequality $\phi'(\alpha_0)\le 0$, hence $g_{\mathcal T}(\alpha_0)\le g_{\mathcal T'}(\alpha_0)$. The lemma then asserts a strict upward *jump*, and §4.3 builds the paper's non-monotonicity result on it. Strictness is never established — and it is not true in general.

**Why it's an issue.** A jump requires $\phi'(\alpha_0)<0$. If $\phi'(\alpha_0)=0$ then $C^*$ is differentiable at $\alpha_0$, $g^*$ is continuous there, and — combined with the lemma's own first claim that $g^*$ is non-increasing *within* each regime — the returns to AI quality would be **monotone**, the exact opposite of the section's conclusion. Since $\phi=C_{\mathcal T'}-C_{\mathcal T}$ is a real-analytic generalised polynomial in $1/\alpha$, a sign change only needs a zero of *odd* order, so the displayed premises do not force strictness.

Here is an admissible workflow in which the jump is exactly zero. Take $m=4$ with

$$d=(2,1,2,1),\qquad t^{A}=(102,\,65,\,1000,\,12),\qquad t^{M}=(1000,\,1000,\,56,\,1000).$$

Writing $x=1/\alpha$: $\mathcal T=\{\text{chain}\{1,2\},\ \text{manual }3,\ \text{chain}\{4\}\}$ costs $65x^{3}+56+12x$, and $\mathcal T'=\{\text{chain}\{1\},\ \text{chain}\{2,3,4\}\}$ costs $102x^{2}+12x^{4}$, so

$$\phi(x)=12x^{4}-65x^{3}+102x^{2}-12x-56=(x-2)^{3}(12x+7),$$

with $\phi(2)=\phi'(2)=\phi''(2)=0$ and $\phi'''(2)=186>0$. Brute force over all 34 strategies confirms that $\mathcal T$ is the unique optimum just below $\alpha_0=0.5$ and $\mathcal T'$ just above it — a genuine reorganization threshold — yet the envelope's marginal benefit passes straight through it, *decreasing*:

| $\alpha$ | 0.4999 | 0.5 | 0.5001 |
|---|---|---|---|
| $g^*(\alpha)$ | 3170.5 | 3168.0 | 3165.5 |

All primitives are admissible: $d_i\ge 1$, $t^M,t^A>0$, $q_i=\alpha^{d_i}\in(0,1)$.

**Proposed fix.** Two options.
1. *Repair the lemma.* Add an explicit transversality condition — $\phi'(\alpha_0)\neq 0$ at every threshold, i.e. the two envelope-adjacent cost curves cross non-tangentially — state it in the lemma, and note that it holds for generic $(t^A,t^M,d)$ and in Example 2. Then "jumps upward" is earned.
2. *Weaken the claim.* State the lemma as: within each regime $g^*$ is non-increasing; **at a reorganization threshold $g^*$ never falls, and it rises strictly whenever the two cost curves cross transversally** (as at both thresholds of Example 2). Then soften `4_implications.tex:205–206, 244, 250` and the Figure 5 notes from "jumps up discontinuously" to "does not fall, and rises discontinuously at every transversal threshold".

Option 1 is preferable — the economics you want is real and generic; it is only the universal quantifier that fails.

---

### MAJ-2. Footnote 12's sufficient condition for an AI-hard step never to be chained is false — and the mechanism it misses is the paper's own

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span> Footnote 12 now states $\manualTime{h} \le 1/q_h - 2$, derived from the merge deviation rather than the extension: merging the chains on either side of $s_h$ saves one of the two verifications they pay separately, so the tightest case is perfectly reliable flanking chains, where $1/q_h$ must cover $2 + \min\{\manualTime{h}, 1/q_h\}$. The body sentence is unchanged, as the review found it correct.

**Where:** §4.2, `4_implications.tex:150` (footnote) — **page 18**.
**Text:** “this holds whenever $\min\{t^M_h,\,1/q_h\}\le 1/q_h-1$ for each AI-hard step $s_h$: extending a chain over $s_h$ multiplies its expected cost by $1/q_h$ while saving only the cost of executing $s_h$ on its own.”

**Issue.** The condition is offered as sufficient for AI-hard steps to be ones "the firm would not fold into a chain under any arrangement". It is not sufficient. The argument prices only one deviation — *extending* an existing chain over $s_h$ — and misses the deviation that *merges two chains across* $s_h$, which saves an entire verification on top of $s_h$'s own standalone cost.

**Why it's an issue.** Counterexample in the normalization the footnote assumes ($t^A_i=1$). Three steps: $q=(1,\,0.5,\,1)$, $t^M=(5,\,1,\,5)$, with $s_2$ the AI-hard step. The condition holds with equality: $\min\{t^M_2,1/q_2\}=\min\{1,2\}=1$ and $1/q_2-1=1$. Yet

| arrangement | cost |
|---|---|
| chain$\{1,2,3\}$ | $1/(1\cdot 0.5\cdot 1)=\mathbf{2.0}$ |
| chain$\{1\}$ + $s_2$ alone + chain$\{3\}$ | $1+1+1=3.0$ |
| chain$\{1,2\}$ + chain$\{3\}$ | $2+1=3.0$ |
| chain$\{1\}$ + chain$\{2,3\}$ | $1+2=3.0$ |

The optimum folds the AI-hard step into a chain. The footnote's inequality compares $C\,(1/q_h-1)$ against $\min\{t^M_h,1/q_h\}$ with $C\ge 1$; that is the right comparison only when $s_h$ is appended to a chain whose *other* endpoint stays put. When $s_h$ bridges two chains, folding it in also removes one verification, worth a further $t^A=1$ — which is exactly the chaining complementarity the paper is about.

**Proposed fix.** The body sentence itself is fine: Example 1's own four steps are in the class it describes. I enumerated all six distinct orderings of $\{$hard, hard, easy, easy$\}$ at the example's parameters and in none of them is an AI-hard step folded into a chain (the optimum is 28.00 in the three orderings where the two easy steps are not adjacent — EHEH, EHHE, HEHE — and 24.24 in the three where they are — EEHH, HEEH, HHEE — with both hard steps performed manually in every case, which is exactly the point Example 1 makes). It is only the footnote's *sufficient condition* that is wrong, and it is wrong because it omits the merge deviation.

Two ways to repair it.

1. *State the correct test.* Absorbing $s_h$ between two adjacent chains of costs $a$ and $b$ costs $ab/(t^A q_h)$ against $a+b+\min\{t^M_h,\,t^A/q_h\}$, so the step is never chained iff that inequality fails for every pair of admissible flanking chain costs (together with the two one-sided cases $a/q_h$ vs. $a+t^*_h$, which is what the footnote currently prices). In the normalization $t^A_i=1$ the crudest bound, taking $a,b\ge 1$, gives the sufficient condition $\min\{t^M_h,1/q_h\}\le 1/q_h-2$ — but note this is very conservative (it assumes the flanking chains can be as cheap as a single verification, which needs $q=1$ neighbours) and it does **not** certify Example 1's own parameters, where $\min\{t^M_h,t^A/q_h\}=6$ against $t^A(1/q_h-2)=4.67$. So it is a poor replacement on its own.
2. *Drop the general condition and verify the example.* Simplest and honest: delete the "this holds whenever …" claim and say instead that for the parameters of Example 1 no arrangement of the four steps chains an AI-hard step (verifiable by enumerating the six orderings, as above). That is all the sentence in the body actually needs.

Either way, the sentence in the footnote explaining *why* — "extending a chain over $s_h$ multiplies its expected cost by $1/q_h$ while saving only the cost of executing $s_h$ on its own" — should be corrected to note that folding a step in can also *merge* two chains, in which case it additionally saves one verification. That is the paper's own central mechanism, so it is worth getting right here.

---

### MAJ-3. Prediction #2's coefficient is not identified from *arrangement*: the count control leaves the mechanical channel open

🔵 <span style="color:#0969da">**Status: DEFERRED.**</span> Prediction #2 is being handled separately; see `EFI_MATCHED_SPECIFICATION.md` and `analysis/efi_matched_exposure/`.

**Where:** §7.2, `7_empirics.tex:102` and Equation (12) at `:95–101`; Table 2 notes at `:135` — **pages 36–37**.
**Text:** “this control ensures that $\beta_2$ is identified from how those steps are arranged rather than how many there are.”

**Issue.** The empirical fragmentation index is $\mathrm{EFI}=1-P/m$ ($P$ = number of adjacent AI-able pairs, $m$ = number of steps). Under a *random* arrangement of $k$ AI-able steps, $\mathbb E[P]=k(k-1)/m$, so $\mathbb E[\mathrm{EFI}]=1-\frac{k}{m}\cdot\frac{k-1}{m}$ — a function of the AI-able **density** $k/m$, not of the count $k$. Equation (12) controls for the count and for the E1 exposure share, but not for $m$ or for the AI-able share, so the density channel is left in $\beta_2$. The dependent variable, the share of *all* steps executed by AI, is itself increasing in that same density.

**Why it's an issue.** The specification returns a strongly negative, highly significant $\beta_2$ under a null in which the arrangement carries **zero** information. Simulating 872 occupations with $m\in[5,40)$, AI-able steps drawn at 44% (E1 nested inside them at 14%), AI-able positions placed **uniformly at random**, and AI execution a fixed fraction of AI-able steps:

| specification (all variables standardized, as in Table 2) | $\beta_{\mathrm{EFI}}$ | $t$ | $p<0.01$ |
|---|---|---|---|
| paper's exact spec (E1 share, EFI, **count** of AI-able steps) | **−0.78** | −42.0 | 100% of reps |
| + control for workflow length $m$ | −0.52 | −25.3 | 100% |
| + control for AI-able **share** $k/m$ | −0.00 | −0.9 | — |

So the sign, the significance, and (given a stylised DGP) the order of magnitude of Table 2's −0.26 / −0.38 / −0.28 are all delivered by a null with no arrangement content. The SOC major/minor fixed effects in columns (2)–(3) absorb only between-group variation in $m$, so they do not repair it. Note too that §7's own Prediction #3 regression *does* control for the number of steps in the occupation (`:192`), so the two designs are inconsistent on exactly this point.

This does not show the finding is spurious — it shows the specification cannot distinguish arrangement from density.

**Proposed fix.** Add the AI-able share $k/m$ (equivalently, workflow length $m$) to Equation (12) and to both EFI tables; in the simulation above that control removes the mechanical channel exactly. Better still, benchmark the EFI against its own random-arrangement expectation — regress on $\mathrm{EFI}-\bigl(1-\frac{k(k-1)}{m^{2}}\bigr)$, or on the EFI residualized on $(k,m)$ — which is the quantity the theory is about. As a complement, make the dependent variable the share of *AI-able* steps that are AI-executed, which is what the prediction as worded at `:6` actually says ("fewer of *those* steps actually executed by AI"). Then rewrite `:102` and the table note to say the coefficient is identified from arrangement conditional on the number **and** the length.

---

## Medium

### MED-1. §6.1 states as a result what OA.C says cannot be concluded

🟡 <span style="color:#9a6700">**Status: PARTIAL.**</span> `6_extensions.tex:26` no longer claims the three aggregate inputs are complements and now attributes $\sigma<1$ to the assumed $\rho<0$, adding that the aggregation supports Equation (9) for any such $\rho$ rather than pinning one; `:37` qualifies "largely carry over" to the margin the representation covers; `OA_C:167` no longer reads the assumption as a finding. **Declined:** moving OA.C's `:242-247` caveats into the `:33` footnote. That footnote already flags that the aggregate representation asks more of the economy and points to Appendix C, where the locus and two-dimension qualifications are stated in full, so nothing in the body is false; it simply does not enumerate them. `OA_C:250-252` left as written.

**Where:** §6.1, `6_extensions.tex:26` and `:37` — **page 30**; the disclaimers are at `OA_C_CES_representation.tex:162, 242–247` — **pages OA - 31 to OA - 32**.
**Text:** “$\sigma=1/(1-\rho)$ lies below one and the three aggregate inputs are complements”; “the implications people draw from improving AI quality in a CES economy largely carry over”.

**Issue.** §6.1 presents Equation (9) as a derived three-input CES over $(\mathcal A,\mathcal M,K)$ whose three inputs are complements. OA.C explicitly denies each half of that: the capital exponent “is part of the CES form we posit rather than something the aggregation derives” (`:162`); because $\tau_{M}$ is common across firms, $\mathcal M=\tau_{M}Y$ identically and $K\equiv 1$, so Equation (9) “should accordingly be read as a representation valid along that locus … [and] says nothing testable about the manual-input-per-output or capital margins, which are fixed by construction” (`:242–244`); and “recovering a genuine three-input CES requires firms to differ in at least two of their input requirements … Our two-stage timing rules this out by construction” (`:245–246`).

**Why it's an issue.** With two of the three input ratios fixed by construction, nothing in the aggregation identifies substitutability among three inputs; $\sigma<1$ follows from the *assumed* $\rho<0$, not from anything derived. The caveat footnote at `6_extensions.tex:33` lists four milder assumptions (common capital productivity, identical organization, one dimension of heterogeneity, weights tied to the AI strategy) but omits precisely the two that void the body's reading. A reader of §6 alone — or of the abstract and `1_introduction.tex:118–119` — takes away a claim the paper's own appendix refutes. Note also that OA.C is not internally consistent about this: its own line 167 (“indicating that macro-level production exhibits some degree of complementarity between aggregate inputs”) and its closing paragraph re-assert the reading its caveat paragraph restricts.

**Proposed fix.** In §6.1 say what is actually delivered: a CES-form representation of how the economy substitutes between AI-management and manual labor **along the participation locus**, with the capital exponent posited and $\mathcal M/Y$ and $K$ fixed by construction. Attribute $\sigma<1$ to the assumed $\rho<0$. Replace “the three aggregate inputs are complements”, qualify “largely carry over” at `:37`, and move OA.C's `:242–247` caveats into the §6.1 footnote. Reconcile OA.C `:167` and its closing paragraph with its own caveat.

---

### MED-2. $\phi$ is given three incompatible readings in OA.C, and the headline sentence describes it as the one thing it is not

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span> Minimum-intervention version. `OA_C:166` drops "probability"; `OA_C:190-191` now defines $\phi$ as the output density, states the normalization that only the product of firm density and firm output is identified (firm density set to one, so $y=\phi$), and licenses "distribution of effective AI quality" as shorthand in that sense; `6_extensions.tex:32` names the object correctly in the body. `:230`, `:238`, `:256` and `:259` keep the shorthand, now covered by the definition.

**Where:** `OA_C_CES_representation.tex:166, 190–191, 230, 238, 259` and the title of §OA.C.3 — **pages OA - 31 to OA - 34**; echoed at `6_extensions.tex:32` — **page 30**.
**Text:** “A firm with effective AI quality $\bar\alpha$ thus produces output $y=\phi(\bar\alpha)$ by definition.”

**Issue.** In the mathematics $\phi$ is an unnormalized **output density** over $\bar\alpha$ (line 190, and Equations (C.11)–(C.13), where $\int_u^1\phi=Y$). But line 166 calls it “the output probability density function”, line 191 equates it to a single firm's output *level*, and line 230, line 238, line 259 and the subsection title call it the distribution of $\bar\alpha$ itself.

**Why it's an issue.** It is provably not a probability density: $\int_0^1\phi=\Gamma(0)=\bigl((1-\theta_A-\theta_M)/(\theta_A\tau_A^{\rho})\bigr)^{1/\rho}$, which equals 1 only on a knife edge. Two admissible parameterizations satisfying restriction (C.15) give total mass 2.789 ($\rho=-0.5$, $\theta_A=0.3$, $\tau_A=0.5$, $\tau_M=0.6$) and 0.900 ($\rho=-3.5$, $\theta_A=0.1$, $\tau_A=0.6$, $\tau_M=0.5$) — verified in closed form and by quadrature. And “$y=\phi(\bar\alpha)$ by definition” is coherent only under an unstated indexing normalization (firms indexed by $\bar\alpha$ with unit density); under that normalization Equation (C.16) is the firm's *output schedule*, so calling it the heterogeneity distribution is doubly wrong. Nothing in the derivation breaks — the ODE argument uses only $\Gamma$ and $\Psi$ — but the section's stated deliverable ("the distribution of effective AI quality", repeated in §6.1) is not what it derives, because firm scale is never pinned separately: only the *product* of firm density and firm output is identified.

**Proposed fix.** Keep $\phi$ as the output density/schedule and say so once, consistently. At `:166` drop “probability” (“the output density in $\bar\alpha$, with $\int\phi=Y$”). At `:190–191` state the normalization explicitly (“index firms by $\bar\alpha$ with unit density; a firm of type $\bar\alpha$ produces $y(\bar\alpha)=\phi(\bar\alpha)$, and firms below the threshold exit”) rather than “by definition”. At `:230, :238, :259`, the subsection title, and `6_extensions.tex:32`, replace “distribution for effective AI quality” with “output schedule across effective AI quality levels”, noting it is unnormalized.

---

### MED-3. “AI-exposed” denotes two different sets inside §7, and the introduction inherits the ambiguity

🔵 <span style="color:#0969da">**Status: DEFERRED.**</span> Prediction #2 is being handled separately; see `EFI_MATCHED_SPECIFICATION.md` and `analysis/efi_matched_exposure/`.

**Where:** `7_empirics.tex:19` (definition), `:102` (control), `:117` (EFI footnote), `:133` and `:135` (Table 2 notes) — **pages 33, 36, 37**; `1_introduction.tex:131` — **page 6**.
**Text:** “We count as AI-exposed both E1- and E2-exposed tasks, which together account for 44% of the tasks.”

**Issue.** Line 19 fixes the paper's convention: “treat their E1 category as exposed to AI and the remaining categories as unexposed, which yields a conservative measure of exposure.” The EFI footnote at `:117` then uses E1 **or** E2 (44% of tasks) for the same word, expressly rejecting E1-only (14%). The Table 2 notes use both at once: `:133` says the “AI Exposure” regressor is “the share of AI-exposed (E1) steps” while `:135` says the count control is “the number of AI-exposed (E1 or E2) steps”. Line 102 calls that control “the number of AI-exposed steps”, which under line 19 would be the E1 count.

**Why it's an issue.** No column is misreported — the notes do disclose which set builds which regressor, and the E1-or-E2 count is in every specification — but a reader cannot map either the introduction's sentence (“controlling for the share of steps exposed to AI, occupations whose AI-exposed steps are more dispersed …”) or line 102 onto the estimated equation without decoding the notes. The two constructs are genuinely different (14% vs 44% of tasks), and line 19's “conservative measure” claim covers only the exposure regressor, not the index the second prediction rests on.

**Proposed fix.** Reserve **AI-exposed** for E1 and **AI-able** for E1-or-E2, everywhere. Rewrite `:117` as “we count a task as AI-able if it is E1- or E2-exposed (44% of tasks)”; change `:102` and Table 2 note `:135` to “AI-able”; change `1_introduction.tex:131` to “occupations whose AI-able steps are more dispersed”. State once at the head of §7.2 that the exposure regressor is the E1 share while the EFI is built on the broader AI-able set, and add a robustness column with both built on the same set.

---

### MED-4. Prediction #3 is stated in the neighbours' AI-*ability* but estimated on their realized *execution*, and reported causally

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>

**Where:** §7.3, `7_empirics.tex:154, 192, 228` and Table 3's caption — **pages 38, 40**.
**Text:** “AI execution of immediate neighbors of the focal task increases the likelihood that the task is executed by AI.”

**Issue.** The prediction as worded (`:154`) and the model's comparative static (Proposition 1(ii), a statement about the neighbours' success probabilities $q_{k\pm1}$) are in the neighbours' AI-ability — a primitive. The regression instead uses the neighbours' *realized execution*, a coordinate of the same cost-minimizing partition that produces the dependent variable. The exposure labels needed for the primitive-based test exist for every task and are the primary measure in Prediction #2, but are never used here, and the neighbours' exposure is not among the controls (`:192` lists only the focal step's exposure and the occupation's task count).

**Why it's an issue.** Under the model both indicators are outputs of one optimization, so the coefficient measures equilibrium co-movement, not an effect of one step's mode on another's — yet it is reported with causal labels (“Effect of Neighboring Tasks' AI Execution Status on Task's AI Execution”; “increases the likelihood”). With the neighbours' exposure omitted, `prev_is_ai` / `next_is_ai` still carry the neighbours' AI-ability, which DWA fixed effects do not absorb (they hold the *focal* task's content fixed) and which the within-occupation placebo cannot net out (it destroys adjacency, so it cannot difference out content-driven correlation between genuinely adjacent steps). The paper applies exactly this standard elsewhere: the Prediction #2 footnote flags the execution-based EFI as “mechanically related to the dependent variable” and declines to read it independently. No such caveat appears here.

**Proposed fix.** Run the exposure-based specification (neighbours' E1/E2 exposure as the regressors of interest) as the headline Prediction #3 test, since exposure is the primitive the prediction names and is not jointly determined with `is_ai`$_k$; keep the execution-based regression as a secondary specification with the neighbours' exposure added to the controls. Retitle Table 3 “Neighboring Tasks' AI Execution and a Task's AI Execution”, rewrite `:228` and the abstract's third bullet as association/co-movement, and add a footnote parallel to the Prediction #2 one.

---

### MED-5. The long-run dynamic program is self-referential at every job-opening state, so it does not determine $V(i)$

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span> `OA_B:556` now defines $V(i)$ directly as the minimum over the manual and chain branches evaluated at the job-opening state, with $V(0)=0$, instead of as the entry $R(i,0,t^{H}_{i})$; the closure bullet restricts that option to $c>0$ and says why (a job must hold at least one task, Definition 6), naming the degenerate $V(i)=\min\{V(i),\cdot\}$ it removes; the fill order is stated, and the circular identification $V(m)=R(m,0,0)$ is replaced by $V(m)$ with $t^{H}_{m}=0$. Equation (14) in `6_extensions.tex:78-85` mirrors all three. Verified against brute force over joint (AI strategy, job design) pairs on random instances: the corrected recursion matches exactly, the printed one returns 0.

**Where:** `OA_B_omitted_proofs.tex:551, 561–566, 585` — **pages OA - 20 to OA - 21**; identically in §6.2 at `6_extensions.tex:77–83` — **page 32**.
**Text:** “The term $c\,t+V(i)$ corresponds to not adding any further steps to the job of the active worker.”

**Issue.** The recursion's first branch is $ct+V(i)$ at the *same* index $i$, and $V(i)$ is defined as the entry $R(i,0,t^{H}_{i})$. Evaluating the recursion at exactly that state gives $V(i)=\min\{0\cdot t^{H}_{i}+V(i),\ A(i)\}=\min\{V(i),A(i)\}$, where $A(i)$ is the min of the manual and chain branches.

**Why it's an issue.** That equation is satisfied by *every* $x\le A(i)$, so it does not determine $V(i)$; a solver filling the table in increasing $i$ has no defined value to use, and value iteration from below returns 0. Concretely with $m=1$: $V(1)=\min\{V(1),\ c^{M}_{1}(t^{H}_{1}+t^{M}_{1}),\ c^{A}_{1}(t^{H}_{1}+t^{A}_{1}/q_1)\}$ is satisfied by $V(1)=0$. Economically, the zero-cost self-loop is the option of *closing a job that holds no tasks*, which Definition 6 does not admit (a job is a block of tasks $(T_b,\dots,T_{b'})$ with $b\le b'$). Nothing about Proposition 4's bound changes — the intended fixed point is the right one — but the recursion as displayed is not well-founded, which matters because the proposition is a claim about an algorithm.

**Proposed fix.** Restrict the closure branch to states with at least one task already assigned (equivalently $c>0$), and define $V(i)$ explicitly as the min over the two non-closure branches evaluated at $(i,0,t^{H}_{i})$; then state the fill order (compute $V(i)$ from entries with index $<i$, then the remaining $R(i,\cdot,\cdot)$). Mirror the change in Equation (14) of `6_extensions.tex`. See also MIN-32: this makes $V(i)$ a $c=0$ entry, which the discretization grid does not currently contain.

---

### MED-6. The introduction names the worker's tasks as $\{1\},\{4\},\{5\}$, contradicting the paper's own definition of a task and the figure eight lines above

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span> `1_introduction.tex:75` now reads "The worker's job therefore consists of three tasks: the manual step 1, the AI chain spanning steps 2--4, whose only human input is verifying the output of step 4, and the manual step 5." The chain is named as one task, matching Definition 5, the Figure 1 notes at `:67` and the TikZ source; the "1, 4, 5" reading is preserved as the human touchpoints rather than as task labels, so nothing suggests tasks are indexed by steps.

**Where:** `1_introduction.tex:75` — **page 4**.
**Text:** “The resulting worker tasks in this job thus become $1$, $4$, and $5$.”

**Issue.** Definition 5 makes the blocks of the partition the firm's tasks, and §3.2 says “a manual step is a task on its own, and an entire AI chain is a single *composite* task.” With Steps 2–3 automated and Step 4 augmented, the chain spans $(s_2,\dots,s_4)$ and is **one** task $\{2,3,4\}$ — which is exactly what the notes to Figure 1 say eight lines earlier (`:67`: “Steps 1 and 5 form separate human tasks, while Steps 2–4 form an AI chain task”) and what the TikZ source draws (a single dashed box fitting `(S2)(S3)(S4)`).

**Why it's an issue.** The paper's central object is described two ways within eight lines, on the page where it is first introduced. Naming the middle task “4” also invites the reading that tasks are indexed by steps, which conflicts with Figure 2, where the five tasks of a seven-step workflow are numbered 1–5 by position.

**Proposed fix.** “The resulting worker tasks in this job are therefore the manual step 1, the AI chain spanning steps 2–4 (delegated at step 2 and received back by verifying the output of step 4), and the manual step 5.”

---

### MED-7. The model cannot produce a J-curve, so “micro-foundation for the productivity J-curve” overstates the connection

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>

**Where:** `1_introduction.tex:105` — **page 5**; also `2_literature.tex:11` (**page 8**), `4_implications.tex:252–253` (**page 21**), `8_conclusion.tex:14` (**page 43**).
**Text:** “Our framework thus provides a micro-foundation for the productivity J-curve phenomenon.”

**Issue.** Lemma OA.B.4's proof establishes $C_{\mathcal T}'(\alpha)\le 0$ for every strategy, so $C^*=\min_{\mathcal T}C_{\mathcal T}$ is non-increasing and measured productivity $1/C^*$ is non-decreasing in $\alpha$ **everywhere**. The model therefore cannot generate the defining feature of a J-curve: an interval on which measured productivity falls below trend.

**Why it's an issue.** In \citet{brynjolfsson2021productivity} the J-curve is a shape in *measured* TFP, produced by unmeasured intangible investment being counted as cost rather than output, and then reversing as that intangible capital pays off. This model has neither an investment margin nor a measurement wedge. What it delivers is a convex, always-non-negative gain profile with upward jumps at reorganization thresholds — delay and lumpiness, but no dip. §4.3's gloss also restates the cited result inaccurately, describing the J-curve as “only modest improvements in the early days of their adoption”, which is the delay, not the dip.

**Proposed fix.** At `1_introduction.tex:105` use the conclusion's own weaker language: the framework generates delayed and lumpy returns to AI quality and is *consistent with, and a complementary mechanism to*, the productivity J-curve, rather than a micro-foundation of it. At `4_implications.tex:252–253` state their shape correctly (measured productivity first falls below trend because complementary intangible investment is counted as cost, then rises above it), then say explicitly that this model delivers the delay and the lumpiness through a different channel — discrete reorganization thresholds — and no dip.

---

### MED-8. The two-stage timing does not *generate* the dispersion in $\bar\alpha$; it removes every other dimension of heterogeneity

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>

**Where:** `6_extensions.tex:32` — **page 30**.
**Text:** “Appendix OA.C obtains that dispersion from the order in which a firm has to commit, settling on its AI strategy and job design before it learns how effectively it can put AI to use.”

**Issue.** The dispersion is a primitive, not a consequence of the timing. OA.C `:143`: “Firms do not know their individual effective AI quality level before production occurs; instead, they share a common belief about the distribution from which these quality levels are drawn.” The distribution $\phi$ is then reverse-engineered from the posited CES parameters, not derived from the timing.

**Why it's an issue.** OA.C `:246` states the opposite of what §6.1 attributes to it: “Our two-stage timing rules this out by construction, since every firm commits to the same $\mathcal T$ and $\mathcal J$ before learning $\bar\alpha$.” What the timing does is make $\tau_{A},\tau_{M}$ common across firms so that $\bar\alpha$ becomes the *single* dimension of heterogeneity — a restriction, not a source.

**Proposed fix.** “Appendix OA.C makes realized AI effectiveness the only dimension of firm heterogeneity, by having firms commit to an AI strategy and job design before learning it, and derives in closed form the distribution of that effectiveness under which Equation (9) holds.” Keep the existing footnote.

---

### MED-9. `acemoglu2022artificial` is cited twice for aggregate production-function modelling; it is an empirical vacancy-postings paper

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span> Fixed by Peyman. `acemoglu2022artificial` is replaced by `acemoglu2022tasks` (Acemoglu and Restrepo 2022, Econometrica) at both `6_extensions.tex:13` and `OA_C:6`, and the key no longer appears in any `.tex`. The third key in each list was also changed, from `acemoglu2024tasks` (Acemoglu, Kong and Restrepo, *Tasks At Work*) to `acemoglu2025simple` (Acemoglu, *The Simple Macroeconomics of AI*), which fits both sentences. `acemoglu2024tasks` remains cited at `2_literature.tex:2`. One typo in that edit, `acemoglu205simple` at `6_extensions.tex:13`, left an undefined citation on page 29; corrected to `acemoglu2025simple`.

**Where:** `6_extensions.tex:13` — **page 29**; `OA_C_CES_representation.tex:6` — **page OA - 24**.
**Text:** “These are the terms the automation literature works in, writing production functions directly at the level of the aggregate economy \citep{acemoglu2018, acemoglu2022artificial, acemoglu2024tasks}.”

**Issue.** `acemoglu2022artificial` resolves in `rubin.bib` to Acemoglu, Autor, Hazell and Restrepo (2022), *“Artificial Intelligence and Jobs: Evidence from Online Vacancies”* (JOLE) — an empirical study of job postings that writes no production function. OA.C `:6` cites it again for “the large literature in which production is modeled directly at the task level and then aggregated to the economy level”.

**Why it's an issue.** Neither sentence describes that paper. `rubin.bib` already contains `acemoglu2022tasks` = Acemoglu and Restrepo (2022), *“Tasks, Automation, and the Rise in U.S. Wage Inequality”* (Econometrica), which is the paper both sentences describe and which `1_introduction.tex:3` already cites correctly — so this reads as a key mix-up between the two 2022 entries. (I checked every `\cite*` key in the body and both appendices against `rubin.bib`: all 64 resolve, so this is a wrong-key problem, not a missing-entry one.)

**Proposed fix.** Replace `acemoglu2022artificial` with `acemoglu2022tasks` at `6_extensions.tex:13` and `OA_C_CES_representation.tex:6`.

---

## Minor

Listed in the order in which they appear in the paper.

### §1 Introduction

🔴 **MIN-1. The gloss of “augmented” inverts who does the work.** `1_introduction.tex:45` — **page 3**. “*Augmented* completion involves a human performing the step with the use of AI.” Definition 2 (`3_shortrun.tex:34`) has the AI executing the step and the human only *verifying* the output; the very next sentence in the introduction describes it correctly. *Fix:* “*Augmented* completion means the AI executes the step and a human verifies its output.”

🔴 **MIN-2. The external-validation sentence covers three predictions; §7 re-runs two.** `1_introduction.tex:139` — **page 7**. “our main predictions continue to hold when we re-estimate them on the practitioner-ordered PCF corpus.” §7 (`:260–261`) says Predictions #1 and #2 come through and “We are unable to re-run Prediction #3, however”. *Fix:* “…and Predictions #1 and #2 continue to hold when we re-estimate them on the practitioner-ordered PCF corpus, while Prediction #3 cannot be re-estimated there because the matched data lacks the required within-step variation.”

### §3 Short Run

🔴 **MIN-3. “Without loss of generality” is doing the work of a $p=1$ assumption.** `3_shortrun.tex:12` — **page 9**. Three lines earlier the short run has several workers, each covering “a fixed block of steps”, and Table 1 lists Job Design: Fixed. What is genuinely without loss is only the wage part — the objective separates across fixed jobs, and each job's optimal AI strategy is independent of its wage rate. Collapsing to the single worker used everywhere downstream (`:114`, the notes to Table OA.A.1, OA.B `:524`, `5_longrun.tex:154`) is a substantive restriction: with $p>1$ a chain cannot cross a fixed job boundary, and the hand-off times at those boundaries never enter Problem (1). Nothing downstream breaks (every object goes through job by job, and the empirical unit is one occupation), so this is a framing slip. *Fix:* state it as an assumption — “we take the workflow to be the block of steps covered by one worker, whose wage we normalize to 1; with several fixed jobs the problem separates across them, since each job's AI strategy minimizes its own execution time whatever its wage, and the hand-off times at fixed boundaries are additive constants” — or make `:9` singular.

### §4 Implications of Chaining

🔴 **MIN-4. Example 1's panel (a): the leading easy step is not “flanked”.** `4_implications.tex:132` (figure notes) and `:138` — **page 17**. Both say every AI-easy step in panel (a) is flanked by AI-hard ones. The panel draws Easy, Hard, Easy, Hard (`plots/TikZ_visualization/dispersed_steps_visualization.tex`), so Step 1 has no predecessor. The claim that does the work — and that is true of the figure — is the next one. *Fix:* replace both with “no two AI-easy steps are adjacent”.

🔴 **MIN-5. “A single closed-form quantity” — the closed form needs $t^{M}_{i}\ge1$.** `4_implications.tex:171` — **page 18**. The sentence reads off Proposition 2, i.e. the general case, but in general $\omega(C)=\min\{1,\sum_{s_i\in C}t^{M}_{i}\}$ depends on the whole component, so the index is an expectation over $2^m$ failure patterns (computable by DP, not closed-form). Only Equation (4), which assumes $t^{M}_{i}\ge1$, is closed-form. *Fix:* “a single quantity, in closed form when $t^{M}_{i}\ge1$,” or move the sentence after Equation (4).

🔴 **MIN-6. The prophet does not have the fragmentation index as its expected cost.** `4_implications.tex:173` — **page 19**. “the fragmentation index is the expected cost of that strategy.” For a step the prophet *knows* fails on its first AI attempt, running it as a standalone augmented step costs $1+1/q_i$ in expectation (the known-failed attempt, then a fresh geometric run), not the $1/q_i$ that the first term of (3) charges through $\min\{t^{M}_{i},1/q_i\}$. At $q_i=0.4$: (3) charges $\min\{t^{M}_{i},2.5\}$ where the described strategy actually costs $\min\{t^{M}_{i},3.5\}$. The component term is exact (a run known to succeed does cost exactly 1), so the mismatch is one-sided: the index *lower-bounds* the prophet strategy's true expected cost. *Fix:* “the fragmentation index is what that strategy is charged”, plus one clause: each failed step is charged its *unconditional* standalone cost rather than its cost conditional on the observed first-attempt failure, so the index is a prophet-flavoured benchmark rather than the expected cost of any single well-defined strategy.

🔴 **MIN-7. “Lowest precisely when the reliable steps sit beside one another” — adjacency is necessary, not sufficient.** `4_implications.tex:187` — **page 19**. Minimizing (4) over orderings means maximizing $\sum_{i=2}^{m}q_{i-1}q_i$, whose maximizer is the unimodal (“organ-pipe”) arrangement with the most reliable steps in the *interior*. With $q=(0.9,0.9,0.1,0.1)$ and $t^{M}_{i}\ge1$: the clustered order gives $0.81+0.09+0.01=0.91$, but $(0.1,0.9,0.9,0.1)$ gives $0.09+0.81+0.09=0.99$ — lower FI, though the reliable steps are side by side in both. (Exhaustive check over all six distinct orderings confirms 0.99 is the max.) *Fix:* “and so is lower the more the steps AI performs reliably sit beside one another; it is minimized by the unimodal arrangement that places the most reliable steps in the interior, where each borders two others.”

🔴 **MIN-8. Not every threshold lengthens a chain.** `4_implications.tex:205–206` — **page 20**. “extending a chain to absorb steps that were previously manual”; “those returns jump upward as longer AI chains become worth deploying.” By Table OA.A.3, at $\alpha=0.50$ the optimum switches from *both steps manual* to *step 2 augmented* — a standalone augmented step, which §3 defines as a chain of length one — and the marginal benefit jumps from 0 to $4/\alpha^2=16$ with no chain extended. Only the second threshold ($\alpha\approx0.92$) lengthens a chain, and Lemma OA.B.4 imposes no such requirement on $\mathcal T'$. *Fix:* at `:206` write “because the newly optimal strategy exposes a larger total difficulty to AI, so its cost falls more steeply in $\alpha$”; keep the chain-lengthening case at `:205` but flag it as the leading example.

🔴 **MIN-9. Figure 5's notes point to a table that is not there, and panel (a) plots four of the five configurations.** `4_implications.tex:235` and `:240` — **page 21**. The notes say “the table below gives each strategy's cost and the range of $\alpha$ over which it is optimal”; no table follows the figure — the table is Table OA.A.3 on **page OA - 3**. Separately, `:240` says “the firm chooses among five configurations” and Table OA.A.3 lists five, but the plotted panel (a) legend has four: *Both steps manual*, *Step 1 manual / Step 2 augmented*, *Both steps augmented*, *Steps 1–2 chained*. The missing curve is *Step 1 augmented, Step 2 manual* ($3.5\alpha^{-11}+8$), which at $\alpha=1$ takes the value 11.5, inside the plotted range — so it is genuinely absent, not merely off-scale, and the note's “Panel (a) shows the cost of each AI strategy” is inaccurate. *Fix:* point the notes at Table OA.A.3 by name and reference, and either add the fifth curve to panel (a) or say the panel omits the never-optimal *Step 1 augmented* strategy.

### §5 Long Run

🔴 **MIN-10. The outer minimization ranges over bare partitions, but an AI strategy carries a mode label.** `5_longrun.tex:141` and Equation (8) — **page 26**. $\mathcal P(X)$ is defined as the set of contiguous partitions, whereas by Definition 5 an AI strategy is a partition *together with* a mode label on each singleton block (manual step, or length-one chain). The label fixes $(c_{b},t_{b})$ for singletons and cannot be optimized out pointwise in the long run, since the objective is the non-separable product $(\sum c_{b})(\cdot+\sum t_{b})$. No result depends on it — the sentence after (8) says $t_{b}$ and $c_{b}$ depend on the selected mode, and the OA.B recursion carries both modes as separate branches. *Fix:* write “$\min$ over AI strategies $\mathcal T$ (Definition 5)” and reserve $\mathcal P(\cdot)$ for the job layer, where $\mathcal J\in\mathcal P(\mathcal T)$ is exactly right.

🔴 **MIN-11. Hand-off time is indexed by tasks where it is defined over steps.** `5_longrun.tex:192` — **page 27**. The three-task example writes $(c_{b},t_{b},t^{H}_{b})_{b=1,2,3}$, but `:112` and Table OA.A.1 define $t^{H}_{i}$ as a step-level primitive (the hand-off of a worker whose last step is $s_i$) and $t^{H}(J_j)$ at the job level; there is no task-indexed $t^{H}_{b}$. (Eight of the twelve independent readings flagged this same line.) *Fix:* write the triples as $(c_{b},t_{b})$ with the hand-off given separately as $t^{H}(\cdot)$ at the block boundaries, or state once that in this example each task is a single step so the two indices coincide.

🔴 **MIN-12. Figures 7 and 8 label the hand-off $h_1,h_2$; the paper's symbol is $t^{H}_{i}$.** `plots/job_design.png`, `plots/combined_grid_with_handoff.png` — **pages 27 and 28**. No symbol $h_i$ is defined anywhere. Figure 7 uses both at once: the pink box is labelled $h_1$ while the width beneath it is labelled $t^{H}_{1}$. Figure 8 shows only $h_1,h_2$. *Fix:* regenerate both with $t^{H}_{1}$, $t^{H}_{2}$.

### §6 Extensions

🔴 **MIN-13. “The set of firms for which using AI is worthwhile” names an adoption margin the appendix does not contain.** `6_extensions.tex:31` — **page 30**. In OA.C the only margin is *participation*: every active firm operates the identical committed strategy $\mathcal T$, and sub-threshold firms exit the market. Readers arriving from §§3–5, where AI deployment is the firm's central choice, will read the phrase as an adoption margin. The mechanism itself (composition, with the marginal firm the AI-management-intensive one) is stated correctly. *Fix:* “…because the set of firms that can produce profitably has moved, and the firms at that margin are the AI-management-labor-intensive ones.”

### §7 Empirical Evaluation

🔵 **MIN-14. Table 2's notes do not state the clustering level.** `7_empirics.tex:130` — **page 37**. “Clustered standard errors in parentheses”, with no unit given, while Table 3's notes (`:210`) specify “clustered at the DWA level”. The unit of observation in Equation (12) is the occupation (872 observations), so clustering must be coarser — presumably SOC major or minor group — and column (1) has no fixed effects to signal which. The claim at `:141` that the EFI coefficient is “statistically significant at the 1% level in all specifications” cannot be assessed without it. *Fix:* “Standard errors clustered at the SOC major-group level (XX clusters) in parentheses”, matching the disclosure already given for Table 3.

🔵 **MIN-15. Table 2's regressor is labelled “(Definition 1)”, which is undefined here and collides with the paper's Definition 1.** `tables/fragmentation_index_regression_exposure.tex` — **page 37**. The body calls the two variants “exposure-based EFI” and “execution-based EFI” (`7_empirics.tex:117–118`); “Definition 1” appears nowhere in the main text or the notes, and the paper's Definition 1 is *Manual Step* (page 10), so the label reads as a cross-reference to it. The same internal naming survives in the figure filename (`..._definition1.png`) and in Figure 9's label. *Fix:* relabel the row “Empirical Fragmentation Index (exposure-based)”, and likewise for Figure 9.

🔵 **MIN-16. The EFI identity has the wrong denominator.** `7_empirics.tex:116` — **page 37**. “the EFI is exactly one minus the share of adjacent AI-able pairs in the workflow”. Working the closed form through: with $t^{M}_{i}=t^{A}_{i}=1$ and $q_i\in\{0,1\}$, $\mathrm{EFI}=(m-P)/m=1-P/m$, where $P$ is the *number* of adjacent AI-able pairs and $m$ the number of steps. A “share of adjacent pairs” would divide by the number of adjacent positions, $m-1$. *Fix:* “one minus the number of adjacent AI-able pairs per step, $P/m$”.

🔵 **MIN-17. $q_i\in\{0,1\}$ is outside the model's stated domain.** `7_empirics.tex:114` — **page 37**. The footnote calls $q_i\in\{0,1\}$ a special case of Equation (3), but Definition 2 fixes $q_i=\alpha^{d_i}\in(0,1]$, so $q_i=0$ is excluded — and the first term of (3) evaluates $\min\{t^{M}_{i},1/q_i\}$ at $1/0$. *Fix:* say the EFI is the limiting case $q_i\to0$ for AI-hard steps and $q_i=1$ for AI-able ones, noting that $\min\{t^{M}_{i},1/q_i\}\to t^{M}_{i}=1$ so (3) is well behaved in the limit.

🔴 **MIN-18. The estimating equation carries an error term inside the logistic link.** `7_empirics.tex:185`, Equation (13) — **page 39**. $\Pr(\mathrm{is\_ai}_k=1\mid X_k)=\Lambda(\beta_0+\cdots+\beta_4\,\mathrm{next2\_is\_ai}_k+\varepsilon_k)$. A conditional probability on the left cannot equal a random variable on the right, and the logistic error is what *generates* $\Lambda$ in the latent-index derivation, so it cannot also sit inside it. The estimation itself is a plain logit (pseudo-$R^2$, AMEs, FE columns) and no reported number changes. *Fix:* delete $+\varepsilon_k$; optionally add the latent-index form $\mathrm{is\_ai}_k=\mathbf 1\{X_k'\beta+\varepsilon_k>0\}$ with $\varepsilon_k$ standard logistic, from which the display follows.

🔴 **MIN-19. The displayed equation omits every control the table reports.** `7_empirics.tex:192` and the Table 3 notes `:209` — **page 39**. Equation (13) shows only the four neighbour indicators and a constant, while every reported specification contains the focal step's AI exposure, the occupation's task count, SOC major/minor or DWA fixed effects in columns (2)–(4) and (6), and the same-DWA task count in columns (5)–(6). *Fix:* write the index as $\Lambda(\beta_0+\beta_1\mathrm{prev2}+\beta_2\mathrm{prev}+\beta_3\mathrm{next}+\beta_4\mathrm{next2}+\gamma'Z_k+\mu_{g(k)})$, define $Z_k$ and $\mu_{g(k)}$, and note which columns include which.

### Appendix OA.A

🔴 **MIN-20. Panel A's header names §3 only; the lead-in says §3 and §4.** `OA_A_tables_and_figures.tex:12` vs the panel header at `:23` — **page OA - 1**. The lead-in says Panel A “covers the step-, task-, and AI strategy-level objects of Sections 3 and 4”, but the header reads “(Section 3: Short Run)”. *Fix:* make the header “(Sections 3–4: Short Run)”.

🔴 **MIN-21. The notation table omits the objects §4.2 introduces.** `OA_A_tables_and_figures.tex:11`, Table OA.A.1 — **page OA - 1**. §3 `:14` sends the reader here for “the notation used throughout the theory sections”, but the table has no entry for $F$, $\mathcal C$, $\omega(C)$, $FI$, $OPT$, or $t^{*}_{i}=\min\{t^{M}_{i},t^{A}_{i}/q_i\}$ — and $t^{*}_{i}$ is used in the body (footnote 12's $\min\{t^{M}_{h},1/q_h\}$) while being defined only in OA.B `:10` and `:194`. Also missing: $w$ (the short-run normalized wage) and $\mathcal P(X)$ from Equation (8). No domain is given for $d_i$. *Fix:* add a Panel C for the fragmentation-index objects, add $w$ and $\mathcal P(X)$ to Panel A, and give the domain of $d_i$ (see MIN-30).

🔴 **MIN-22. Figure OA.A.1's notes state an absolute null where the figure shows a placebo-relative one.** `OA_A_tables_and_figures.tex:198` — **page OA - 4**. “more distant neighbors have little to no effect on a task's AI execution likelihood.” Taken literally this is contradicted by panel (a) / Table 3 column (1), where the distant AMEs are 0.07\*\*\* and 0.05\*\*\*. What the figure does show — and what is true of all four panels — is that the observed distant effects are no larger than the reshuffled benchmark. *Fix:* rewrite in the placebo-relative form the body already uses at `7_empirics.tex:214`.

### Appendix OA.B

🔴 **MIN-23. The proof of Proposition 1 restricts $q$ to the open interval; Definition 2 allows $q_i=1$.** `OA_B_omitted_proofs.tex:10` and `:69` — **page OA - 5**. “take $q_{k-1},q_k,q_{k+1}\in(0,1)$”, and $Q=(0,1)^3$ at `:69`, whereas Definition 2 and Table OA.A.1 give $q_i=\alpha^{d_i}\in(0,1]$ with $\alpha\in(0,1]$, so $q_i=1$ is admissible (any step with $d_i=0$, or any step at $\alpha=1$). The proposition as stated in §4.1 carries no such restriction. *Fix:* either add $q_i<1$ to the proposition's hypotheses, or extend the proof to the boundary (part (i)'s first bullet needs $q_{k-1}<1$; the rest goes through).

🔴 **MIN-24. Strict positivity of the primitives is used but never assumed — and the appendix's own example violates it.** `OA_B_omitted_proofs.tex:62` — **page OA - 7**; also `5_longrun.tex:98` — **page 24**. Part (i)'s third bullet computes $V_0-V_3=t^{M}_{k}$ and needs it strictly positive; `5_longrun.tex:98` says splitting a job “strictly lowers the wage bill”, which needs $c_{b},t_{b}>0$ (the change is $c_{1}t_{2}+c_{2}t_{1}$). No domain for $t^{M}_{i},t^{A}_{i},c^{M}_{i},c^{A}_{i}$ is stated anywhere, and Example OA.B.3 (`:417`) sets $t^{M}_{i}=0$ for its odd steps. Nothing is overturned — part (i) survives $t^{M}_{k}=0$ via the middle configuration, and the third bullet is repaired by a strictly larger $\mu$. *Fix:* add $t^{M}_{i}>0$, $t^{A}_{i}>0$ (and $c_{b}>0$ for §5) to the primitives in §3.1, list the domains in Table OA.A.1, and either exempt Example OA.B.3 explicitly or restate it with $t^{M}_{i}=\epsilon$.

🔴 **MIN-25. Local symbols collide with paper-wide ones inside OA.B.** `OA_B_omitted_proofs.tex:125`, `:279`, `:302`, `:439` — **pages OA - 9, OA - 14, OA - 17**. Three collisions: (i) the single-variable claim at `:125` opens “Let $m,\,A,\,B,\,c>0$”, where $m$ is the number of steps throughout the paper and $c$ is the skill cost in the same appendix's DP; (ii) at `:279` and `:302` a block is written $T_b=(s_i,\dots,s_\ell)$, so $\ell$ is the block's *last* step and $i$ its first — the reverse of Definition 4, Definition 5 and Table OA.A.1, and the reverse of $\ell$'s use in the same appendix's DP recursion at `:509`; (iii) $c$ indexes chains in Equation (OA.B.1). *Fix:* rename the local constants ($m\to\bar t$, $c\to\kappa$) and write the blocks as $(s_\ell,\dots,s_r)$ with $\overline T_b=(s_\ell,\dots,s_r,s_{r+1})$.

🔴 **MIN-26. The charging argument for Lemma OA.B.1 does not literally add up (the bound is nonetheless true).** `OA_B_omitted_proofs.tex:225` — **page OA - 12**. Manual singletons are charged their *marginal* contribution to $\omega(C)$, and marginal contributions to $\omega(C)=\min\{1,\sum t^{M}_{i}\}$ do not sum to $\omega(C)$: for a component of two steps with $t^{M}=0.6$ each, each step's marginal contribution is $1-0.6=0.4$, so the two together account for $0.8<1=\omega(C)$. An upper bound needs an apportionment whose parts sum to at least the realized fragmentation. The stated bound *is* correct; here is an argument that adds up. Split each component $C$ into its manual-singleton steps and its chain steps and use $\min\{1,a+b\}\le b+\min\{1,a\}$, giving
$$\textstyle\sum_C\omega(C)\;\le\;\sum_{\text{manual},\,i\notin F}t^{M}_{i}\;+\;\#\{\text{components containing a chain step}\},$$
and the second term is at most $\sum_{T_b\in\mathcal{T}_{A}}(|F\cap T_b|+1)$. Taking expectations reproduces the displayed bound exactly. *Fix:* replace the two-sentence justification with that argument.

🔴 **MIN-27. The auxiliary inequality is stated on a domain where it is undefined.** `OA_B_omitted_proofs.tex:249` — **page OA - 13**. “for any $x,y\in[0,1]$, $(1-x)(1+\tfrac1x)+(1-y)(1+\tfrac1y)\le(1-xy)(1+\tfrac1{xy})$”, but all three terms involve reciprocals, undefined at $x=0$ or $y=0$. *Fix:* state it on $(0,1]$. (The inequality itself is correct: the difference equals $(1-x)(1-y)\bigl(\tfrac1{xy}-1\bigr)\ge0$, which is worth putting in as the one-line proof.)

🔴 **MIN-28. The maximizer of the final ratio is stated with the exponent inverted, at a value the variable cannot take.** `OA_B_omitted_proofs.tex:251` — **page OA - 13**. “The ratio between this expression and $\alpha^{-d(T_b)}$ is maximized at $\alpha^{-d(T_b)}=1/2$.” Writing $z=\alpha^{d(T_b)}\in(0,1]$, the ratio is $(1+1/z-z)/(1/z)=1+z-z^2$, maximized at $z=1/2$ with value $5/4$. So the maximizer is $\alpha^{\,d(T_b)}=1/2$, i.e. $\alpha^{-d(T_b)}=2$. As printed the condition is infeasible, since $\alpha^{-d(T_b)}\ge1$ always. (Six of the twelve independent readings caught this.) *Fix:* “maximized at $\alpha^{d(T_b)}=1/2$, achieving a value of $5/4$”.

🔴 **MIN-29. Example OA.B.2's additive constant is $1/2$, not $1$.** `OA_B_omitted_proofs.tex:359` — **page OA - 15**. “the ratio between $\tfrac{2\sqrt2}{3}m$ and $1+m\times0.6213$”. Computing the index exactly: $FI=q_m+\sum_{i=1}^{m}(1-q)\sqrt2+\sum_{i=2}^{m}(1-q)q$ with $q=1/\sqrt2$, which is $\tfrac{3}{2}(\sqrt2-1)m+\tfrac12=0.62132\,m+0.5$. The limit $\tfrac{4\sqrt2}{9(\sqrt2-1)}\approx1.5174$ is unaffected (I confirmed it numerically: at $m=1000$ the exact ratio is 1.5166). *Fix:* write $\tfrac12+m\times0.6213$.

🔴 **MIN-30. $D_c\ge1$ is asserted as if it followed from the model.** `OA_B_omitted_proofs.tex:439`, `:456` — **page OA - 17**. Nothing in the model restricts $d_i$: Definition 2 permits $d_i=0$ (via $q_i=1$), Table OA.A.1 gives $d_i$ no domain, and OA.C `:183` explicitly uses $d_b\ge0$. Lemma OA.B.4 and its proof survive with $D_c\ge0$; what fails at $D_c=0$ are two ancillary statements — the strictness parenthetical at `:465` and the footnote at `:469` claiming all-manual is the only threshold-free case. *Fix:* state $d_i>0$ as a maintained assumption (amending Definition 2 to $q_i=\alpha^{d_i}\in(0,1)$ for $\alpha<1$), or declare $d_i\ge1$ a normalization of the difficulty scale, and record it in Table OA.A.1. If $d_i=0$ is to be allowed, weaken `:439` to $D_c>0$ where strictness is used and reword `:465` and `:469`.

🔴 **MIN-31. The sign-change display is written with one-sided limits and is therefore vacuous.** `OA_B_omitted_proofs.tex:478–484` — **page OA - 18**. The display reads $\phi(\alpha_0^{-})\ge0$, $\phi(\alpha_0)=0$, $\phi(\alpha_0^{+})\le0$. Since $\phi$ is continuous and $\phi(\alpha_0)=0$, both one-sided limits *equal* 0, so the display literally says $0\ge0$, $0=0$, $0\le0$ and carries no information — it cannot imply $\phi'(\alpha_0)\le0$. The argument needs the sign condition on a punctured neighbourhood. *Fix:* “there exists $\delta>0$ such that $\phi(\alpha)\ge0$ on $(\alpha_0-\delta,\alpha_0)$, $\phi(\alpha_0)=0$, and $\phi(\alpha)\le0$ on $(\alpha_0,\alpha_0+\delta)$; hence $\phi'(\alpha_0)=\lim_{h\downarrow0}[\phi(\alpha_0+h)-\phi(\alpha_0)]/h\le0$.” (This is presentational; the substantive gap is MAJ-1.)

🔴 **MIN-32. The discretization grid has no cell for the states at which every job opens.** `OA_B_omitted_proofs.tex:598` — **page OA - 22**. Skill costs are tabulated in $[1/B,\,mB]$ and time costs in $[1/B,\,2mB^{3}]$, as powers of $(1+\epsilon')$. But the recursion is entered at skill level 0 for every $i$ — through $V(i)=R(i,0,t^{H}_{i})$ — and the final answer is read at $(0,0)$, since $t^{H}_{m}=0$. Zero is in neither range and is not a power of $(1+\epsilon')$, so “rounding every running total up to the next such power” has no image for it. *Fix:* tabulate skill in $\{0\}\cup(\{(1+\epsilon')^{j}\}\cap[1/B,mB])$ and time in $\{0\}\cup(\{(1+\epsilon')^{j}\}\cap[1/B,2mB^{3}])$, storing 0 exactly; it arises only at the job-opening states $(i,0,t^{H}_{i})$ and at $(m,0,0)$, so it adds $O(1)$ grid points per coordinate and leaves the entry count and the $O(m^{4}\epsilon^{-2}\log^{2}(mB))$ bound unchanged. See MED-5.

### Appendix OA.C

🔴 **MIN-33. Step-level primitives are indexed by the task index.** `OA_C_CES_representation.tex:42` (and `:32`) — **page OA - 25**. Table OA.A.1's notes fix the convention — “Step-level primitives are indexed by $i$, task-level objects by $b$” — and `5_longrun.tex:46` states it correctly. OA.C instead writes $t^{A}_{b}$, $t^{M}_{b}$, $c^{M}_{b}$, $c^{A}_{b}$. For a task $b$ that is a chain over $(s_\ell,\dots,s_r)$ the right objects are $t^{A}_{r}$ and $c^{A}_{r}$, and $r\ne b$ in general: in Figure 2, Task 4 is the chain over Steps 4–6, so its verification cost is $t^{A}_{6}$. Line 42's definition of $t^{E(b)}_{b}$ is circular as a result. No formula is numerically wrong under the intended reading. *Fix:* define $t^{E(b)}_{b}$ the way `5_longrun.tex:46` does — $t^{M}_{i}$ when task $b$ is the manual step $s_i$, and $t^{A}_{r}$ when it is the chain with augmented endpoint $r$ — and in Equation (C.1) use the task-level $c_{b}$ restricted to the manual and AI-chain subsets of the job.

🔴 **MIN-34. $\ell$ is a step index and a task index five lines apart.** `OA_C_CES_representation.tex:41` vs `:46` — **page OA - 26**. Line 41: “$d_b=\sum_{i=\ell}^{r}d_i$ for a chain spanning $(s_\ell,\dots,s_r)$” — $\ell$ is a step. Line 46: the same symbol ranges over tasks $T_\ell\in J(b)$ with skill costs $c^{M}_{\ell},c^{A}_{\ell}$. A reader cannot tell whether the sum runs over tasks or steps. *Fix:* rename the summation index to a fresh task index $b'$, keeping $\ell$ for a chain's first step.

🔴 **MIN-35. The hand-off task's skill-adjusted time is never given a formula, and the verbal definition is right only if $w_{A}=w_{M}$.** `OA_C_CES_representation.tex:72–73` — **page OA - 26**. $\tau^{H}(J_j)$ appears in (C.4) and (C.6) but the footnote at `:73` only names the symbol. The verbal definition — a manual task whose skill equals the job's total skill requirement — prices the job's AI-chain skills at $w_{M}$, which reproduces Equation (7) only under $w_{A}=w_{M}$; otherwise the hand-off's wage bill is off by $c^{A}\,t^{H}\,(w_{M}-w_{A})$. It is also left unsaid whether hand-off tasks enter the skill sums in the numerator of $\tau_b$ at `:46`, which is the assumption the construction needs. *Fix:* give the formula, $\tau^{H}(J_j)=\bigl[w_{M}\sum_{T_b^{M}\in J_j}c^{M}_{b}+w_{A}\sum_{T_b^{A}\in J_j}c^{A}_{b}\bigr]t^{H}(J_j)/w_{M}$, and add a clause stating that hand-off tasks carry the job's compensation rate but are excluded from the skill sums defining $\tau_b$, so that (C.4) reproduces Equation (7) exactly.

🔴 **MIN-36. The second labor input is called “human labor” in the sentence that lists the three CES arguments.** `OA_C_CES_representation.tex:140` — **page OA - 29**. Everywhere else — including the gloss of Equation (C.9) twenty-one lines later — the input is “manual labor”, and the body (`3_shortrun.tex:111`) uses “human labor” for total labor, both types included. *Fix:* “each producing output with AI management labor, manual labor, and capital”. (`:98`'s “human-executed” describes an execution mode, not an input, and is fine as is.)

🔴 **MIN-37. $\bar\alpha=1$ is said to be “achieved” along a limit that excludes it.** `OA_C_CES_representation.tex:183` — **page OA - 31**. “the upper bound of $\bar\alpha$ is 1 and is achieved when $\alpha\to1^{-}$.” A supremum approached along a limit is not attained; and since Table OA.A.1 admits $\alpha=1$, the bound *is* attained — at $\alpha=1$ exactly, which the one-sided limit rules out. *Fix:* “the upper bound of $\bar\alpha$ is 1, attained at $\alpha=1$ (and approached as $\alpha\to1^{-}$)”.

🔴 **MIN-38. The positivity check omits the condition the constant factor actually needs.** `OA_C_CES_representation.tex:328` — **page OA - 34**. The check verifies $\theta_{A}\tau_{A}^{\rho}>0$, but $\Gamma$ and $\phi$ carry the factor $\bigl((1-\theta_{A}-\theta_{M})/(\theta_{A}\tau_{A}^{\rho})\bigr)^{1/\rho}$ and additionally need the capital weight $1-\theta_{A}-\theta_{M}>0$, which is never stated among the parameter restrictions (`:161–162` introduces $\theta_{A},\theta_{M}$ with no range, and the only footnoted parameter assumption, at `:178`, is on $w_{A}\tau_{A}/(1-w_{M}\tau_{M})$). *Fix:* state $0<\theta_{A}$, $0<\theta_{M}$, $\theta_{A}+\theta_{M}<1$ at `:161–162`, and extend `:328` to “…which, together with the maintained $1-\theta_{A}-\theta_{M}>0$, makes the real powers appearing throughout well defined.” Note in passing that with $\rho<0$ the share restriction plus a positive capital weight force $\min\{\tau_{A},\tau_{M}\}<1$, which is a choice of units.

---

## What was verified as correct

Recorded so you know these were checked rather than skipped.

- **Proposition 1 (overturning of comparative advantage).** Part (ii)'s upward-closure claim was brute-forced over 71,262 admissible base points (random $t^M,t^A\in(0.05,40)$, $q\in(0.01,0.999)^3$, restricted to the human holding the advantage on $k$), each with six random coordinatewise increases: **zero violations**. All four arrangements of the block are attainable under human comparative advantage, so part (i)'s coverage claim is right. The proof itself checks out line by line, including the auxiliary claim about $D(q)=A/q-\min\{m,B/q\}-c$ and all three closure arguments.
- **Proposition 2 (fragmentation index).** 60,000 random instances ($m\le7$), exact $FI$ by enumeration over all $2^m$ failure patterns against an exact DP optimum: both bands hold, and $5/4$ is attained (max observed $FI/OPT=1.249999$). The greedy $ALG$ of Lemma OA.B.2 was never below $OPT$ in any instance.
- **The two tightness examples.** Example OA.B.1: $OPT=2$, $FI=5/2$, ratio exactly $5/4$. Example OA.B.2: optimal chain length is 3 ($2^{k/2}/k$ minimized at $k=3$, value $0.9428$), and the limiting ratio is $1.5174 = \tfrac{4\sqrt2}{9(\sqrt2-1)}$ (numerically 1.5166 at $m=1000$). Example OA.B.3: chaining blocks of five is optimal (verified by DP for $K$ up to 120, cost exactly $\tfrac23K$), $FI=1+(K-1)(1-1/\sqrt2)$, limit $2.2761=\tfrac23(2+\sqrt2)$.
- **Every number in §§4–5 and Tables OA.A.2–OA.A.3.** The three-step example ($15.6$ vs $35.4$), Example 1's totals (28.00 and 24.24, matching the TikZ sources), Figure 3's threshold ($q_k\approx0.4255$, printed as 0.43) and its four regions, Example 2's thresholds ($0.50$ exactly; $0.9240$, printed as $\approx0.92$) and the claim that the second jump is larger ($129.5$ vs $16$), all five cost expressions and all five $-\mathrm dC/\mathrm d\alpha$ entries of Table OA.A.3, and all eight job-design costs in Table OA.A.2 (9, 16, 15, 30 and 18.5, 18, 24, 30).
- **The whole OA.C algebra.** Profitability threshold, (C.14), the differentiation step, (C.21), the substitution, the bracket at $u=1$ and the equivalence with (C.15), (C.22), and the final $\phi$ in (C.16) — all re-derived symbolically and confirmed. The economics of $\phi$ is the problem (MED-2), not the algebra.
- **Bibliography.** All 64 `\cite*` keys used in the body and both appendices resolve in `rubin.bib`. The only citation problem found is MED-9.
- **Cross-references and exhibits.** Every `\ref`/`\eqref` in the body and OA.A–OA.C resolves to an object of the right kind; the build reports no undefined references. Figures 1, 2, 6 and 8 were checked against their TikZ/PNG sources and match the text and notes (step modes, task groupings, job boundaries, hand-off positions, the red optimal-design outline).

### Candidates examined and rejected

Twenty-six candidate defects were raised in the first pass and did not survive re-derivation. The most substantial:

- **“The chain-length placebos cannot separate chaining from clustered exposure.”** Rejected: §7.1 draws only the weaker conclusions the objection would allow (`:54` “AI-executed tasks tend to cluster … forming longer AI chains than would arise under either type of randomization”; `:85` “systematic rather than chance”, framed at `:83` as mitigating data concerns), and the sharper benchmark is the design of Prediction #2.
- **“The long-run problem does not nest the short-run problem.”** Rejected: the sentence's own second clause (“with wage being fixed rather than chosen”) is the needed restriction. With $w=1$ and $t^{H}_{m}=0$ the single job's wage bill collapses to $\sum_bt_{b}$, i.e. Problem (1) verbatim.
- **“The Leontief separability argument is meaningless because marginal products are undefined.”** Rejected: the display uses $\Delta y/\Delta l$ precisely because the min is kinked, and it is evaluated at the efficient allocation, where the ratio is $\tau_b/\tau_a$. The conclusion is independently correct: allocating $l_{A}$ optimally within the AI group gives exactly Equation (C.4) with $\bar\alpha$ as defined.
- **“The EFI's $q_i\in\{0,1\}$ parameterization shuts off the mechanism it is used to detect.”** Rejected: with ties to manual, an isolated AI-able step is manual while a run of length $L\ge2$ is chained at cost $1<L$; solving all 6-step workflows exactly, the execution share is strictly decreasing in the EFI (0.00, 0.33, 0.50 as EFI falls 1.00, 0.833, 0.667).
- **“$g^*$ is undefined at a threshold.”** Rejected: `OA_B:447` defines $g^*$ piecewise as $g_{\mathcal T}$ on each regime, and the proof differentiates the smooth $\phi$, never $C^*$ at the kink.
- **“Reduction 1's identity is asserted on all of $Q$.”** Rejected: the paragraph at `OA_B:75–83` scopes it explicitly, and a numerical check found zero mismatches inside the comparative-advantage region.
- **“Describing $FI$ as *approximating* $OPT$ hides factors of 8 and 5/4.”** Rejected: Proposition 2 states both bands where the result is stated, `:171` says “up to constant factors”, the only direction used ($FI\le\tfrac54OPT\Rightarrow OPT\ge\tfrac45FI$) is the tight one, and OA.B is candid about the loose side.
- Also rejected: “in the absence of AI tasks collapse” is an optimality claim (it is true under both readings); the introduction overstates the chaining evidence relative to mean chain length 1.45; the conclusion's “why firms invest” sentence; $[1/B,B]$ needing $B\ge1$; the “Therefore” at `3_shortrun.tex:60`; the per-use-AI-cost footnote; the perfect-verification and retry-protocol assumptions; hand-off borne by the sender only; §5.4 treating $t^{H}_{i}$ as a primitive while discussing AI lowering it; $\tau_{A},\tau_{M}$ held constant while wages vary; “no firm substitutes” in §6.1; Table 3 columns (1)–(3) not implementing the within-step comparison; column (1) described as having “no additional controls”; and the two OA.C items on “provided that (C.15)” and the $\Gamma(1)=0$ justification.

---

## Where to start

If only three things get fixed: **MAJ-1** (the lemma is false as universally stated — add the transversality condition), **MAJ-3** (add the AI-able share to Equation (12); it is a one-line change that decides whether Prediction #2 identifies anything), and **MED-1** (bring §6.1 into line with OA.C's own caveats). **MAJ-2** is a two-line footnote repair. Of the mediums, MED-3 and MED-6 are pure exposition and cheap; MED-4 needs one extra regression you already have the data for.
