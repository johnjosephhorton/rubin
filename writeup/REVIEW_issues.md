# Review: *Chaining Tasks, Redefining Work: A Theory of AI Automation*

**Draft reviewed:** version of 2026-08-29 (files as of `0_main.tex` and its inputs).
**Reviewed:** `0_main.tex` (title/abstract), `1_introduction.tex`–`8_conclusion.tex`, `OA_A_tables_and_figures.tex`, `OA_B_omitted_proofs.tex`, `OA_C_CES_representation.tex`, `preamble.tex`, `tables/*.tex`, `plots/TikZ_visualization/*.tex`.
**Excluded at the authors' instruction:** `SA_A`–`SA_F`. Nothing below depends on their contents; where a claim could only be checked inside them, it is not reported.
**Out of scope by instruction:** grammar, spelling, wording, style, typography, citation practice.

**Page numbers** are as printed in the compiled draft (a clean `latexmk` build: 128 pages — body pp. 1–42, Online Appendix `OA - 1` to `OA - 33`, Supplementary Appendix `SA - 1` to `SA - 48`). Body pages are bare numbers; Online Appendix pages appear as `OA - n`.

**Method.** Every numeric claim, threshold, derivative, and closed form was recomputed independently (exact rational arithmetic or exhaustive dynamic programming where possible); every proof was re-derived line by line; and every finding was put through adversarial verification before being listed here. Counterexamples are reproducible from the parameters given.

| Severity | Count | Meaning |
|---|---|---|
| **Major** | 6 | A stated result, proof, or headline claim is wrong or unsupported as written |
| **Medium** | 43 | A real error a referee would flag and that must be fixed, but the surrounding result survives |
| **Minor** | 57 | A local notation, indexing, or precision slip with no consequence for the argument |
| **Total** | **106** | |

---

## Reconciliation status (second pass, 2026-09-01)

Each item below now carries a colored **Status** line:

| | Status | Meaning |
|---|---|---|
| 🟢 | <span style="color:#1a7f37">**RESOLVED**</span> | The fix has landed in the draft |
| 🟡 | <span style="color:#9a6700">**PARTIAL**</span> | Some of the fix landed; the remainder is named in the status note |
| 🔴 | <span style="color:#cf222e">**OPEN**</span> | Not yet addressed, or the source file is untouched |
| 🔵 | <span style="color:#0969da">**IGNORED**</span> | Reviewed and deliberately not actioned |

Statuses were assigned by checking every item against the draft as of this pass, diffing against commit `abaa7c3` (the commit that added this file). The second pass re-checked every item against `HEAD` (`e71b3a1`), which picked up fixes landed in `2ae61b9`, `677089c`, `a2038a0` and `e71b3a1`.

| Severity | 🟢 <span style="color:#1a7f37">Resolved</span> | 🟡 <span style="color:#9a6700">Partial</span> | 🔵 <span style="color:#0969da">Ignored</span> | 🔴 <span style="color:#cf222e">Open</span> | Total |
|---|---|---|---|---|---|
| **Major** | 5 | 1 | 0 | 0 | 6 |
| **Medium** | 18 | 2 | 5 | 18 | 43 |
| **Minor** | 3 | 2 | 0 | 52 | 57 |
| **Total** | **26** | **5** | **5** | **70** | **106** |

🟢 **<span style="color:#1a7f37">Resolved.</span>** M1--M5; D1, D3, D5, D6, D7, D8, D15, D16, D17, D19, D22, D24, D25, D26, D28, D30, D32, D41; N28, N29, N47.

🟡 **<span style="color:#9a6700">Partial.</span>** M6; D4, D23; N23, N38.

🔵 **<span style="color:#0969da">Ignored.</span>** D2, D9, D10, D14, D21. Reviewed and deliberately not actioned; they are excluded from the open worklist.

🔴 **<span style="color:#cf222e">Untouched source files.</span>** `2_literature.tex`, `3_shortrun.tex`, `5_longrun.tex`, `8_conclusion.tex`, `OA_A_tables_and_figures.tex` and `tables/` have not been edited since the review, so every item located only in them is open by construction: D11 (in part), D12, D13, D43; N1--N5, N12--N15, N50, N56.

---

# MAJOR

### M1 — Example OA.B.2 substitutes the greedy algorithm's cost for `OPT`; the claimed bound 2.276 is wrong

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>

**Where:** `OA_B_omitted_proofs.tex:328–337` · **p. OA - 15** · the remark "The approximation factor in Lemma OA.B.2 cannot be improved below $\tfrac23(2+\sqrt2)\approx 2.276$" and Example OA.B.2 (`ex:FI.lower.gap`).

**Issue.** For the instance $(t^M_i,t^A_i,q_i)=(\sqrt2,\,1,\,1/\sqrt2)$ for all $i$, the example writes: *"The task sequence described in the proof … handles each task separately, for a total cost of $m\sqrt2$, so the optimal task sequence performs at least this well"*, and then divides $m\sqrt2$ by $FI$. That sentence establishes only $OPT \le m\sqrt2$ — an **upper** bound. A lower bound on the achievable constant requires a **lower** bound on $OPT$, so the ratio computed is $ALG/FI$, not $OPT/FI$. Two consequences:

1. $OPT$ is not $m\sqrt2$. Chaining the identical steps **three at a time** costs $2^{3/2}=2.8284$ per three steps, i.e. $OPT = \tfrac{2\sqrt2}{3}m = 0.94281\,m$. (Per-step cost of a length-$k$ chain is $2^{k/2}/k$: $k{=}1\!:\!1.4142$, $k{=}2\!:\!1.0$, $k{=}3\!:\!0.9428$, $k{=}4\!:\!1.0$.) Hence $OPT/FI \to 0.94281/0.62132 = \mathbf{1.5174}$, not $2.2761$.
2. Even $ALG$ is misstated. The greedy of Lemma OA.B.2 takes the maximal contiguous run with success probability **"greater than or equal to $1/2$"**, and here a two-step run has probability exactly $q^2 = 1/2$. The construction therefore forms **pairs**, at cost $1/(1/2)=2$ per pair, so $ALG = m$ — it does not "handle each task separately". The example silently assumes a strict `>`.

**Why it matters.** The remark this example supports is the paper's only statement about how loose the constant $4$ in Lemma OA.B.2 is. As written the reader is given a false numerical fact about an explicitly constructed instance, and the true gap the instance exhibits is about a third smaller.

**Evidence.** Exact DP over all AI strategies: $OPT/m = 0.94510$ ($m{=}100$), $0.94338$ ($m{=}400$), $0.94304$ ($m{=}1000$) → $0.94281$. $FI/m \to (1-1/\sqrt2)\sqrt2 + (1/\sqrt2)(1-1/\sqrt2) = 0.62132$ (matches the paper's own $0.6213$). $OPT/FI = 1.5019,\,1.5135,\,1.5159 \to 1.5174$.

**Fix.** Recompute with the true $OPT=\tfrac{2\sqrt2}{3}m$ and restate the remark as *"cannot be improved below $\tfrac{4\sqrt2}{9(\sqrt2-1)}\approx 1.52$"*; and perturb the instance to $q_i = 1/\sqrt2-\epsilon$ so the greedy really does produce singletons (or state the greedy's test with a strict inequality). Alternatively drop this example and use Example OA.B.3, which — once corrected (M2) — does deliver exactly $\tfrac23(2+\sqrt2)\approx 2.276$.

---

### M2 — Example OA.B.3 makes the same substitution; the stated **necessity** of $t^M_i \ge 1$ is not established

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>

**Where:** `OA_B_omitted_proofs.tex:375–384` · **p. OA - 16** · Example OA.B.3 (`ex:FI.necessity`) and the sentence introducing it.

**Issue.** The instance is $2K$ alternating steps: $(t^M,t^A,q)=(0,1,1/\sqrt2-\epsilon)$ for odd $i$ and $(1,1,1)$ for even $i$. The paper takes the greedy cost $K\sqrt2$ as $OPT$ and concludes the ratio tends to $2(\sqrt2+1)\approx 4.828 > 4$, hence *"the assumption that $t^M_i \ge 1$ for all $s_i$ is necessary for Lemma OA.B.2 to hold."*

$OPT$ here is far below $K\sqrt2$. The optimum leaves one free odd step manual (it costs $0$) and chains blocks of the form $(E,O,E,O,E)$ — two odd steps, success probability $(1/\sqrt2)^2 = 1/2$, cost $2$ — covering three unit-cost even steps. That gives $OPT = \tfrac23 K$, and
$$OPT/FI \;\to\; \frac{2/3}{1-1/\sqrt2} \;=\; \tfrac23(2+\sqrt2) \;=\; \mathbf{2.2761},$$
**not** $4.828$. Since $2.2761 < 4$, this instance satisfies $FI \ge OPT/4$ comfortably ($FI/OPT \to 0.439$) and therefore **does not show** that the $t^M_i\ge 1$ hypothesis is needed.

**Why it matters.** This is the sole justification for carrying two separate constants ($1/4$ under $t^M_i\ge1$, $1/8$ in general) in Proposition 2. As written that justification is unsupported, and a random search over instances with $t^M_i<1$ did not produce $OPT/FI$ above $\approx 1.6$, so the necessity claim may well be false rather than merely unproved.

Note also that the value the paper attributes to Example OA.B.2 — $\tfrac23(2+\sqrt2)\approx2.276$ — is exactly the *true* value of **this** example. The two constants look to have been crossed.

**Evidence.** Exact DP: $OPT/K = 0.66828,\,0.66707,\,0.66683$ for $K = 100,\,400,\,1000$ (converging to $2/3$); $FI = 1+(K-1)(1-1/\sqrt2)$; $OPT/FI = 2.206,\,2.258,\,2.269 \to 2.2761$. The paper's printed $K\sqrt2/FI$ does converge to $4.828$, which confirms the numerator is $ALG$.

**Fix.** Either exhibit an instance whose true $OPT$ exceeds $4\,FI$, or delete the necessity sentence. Do **not** merely rephrase it as a statement about the greedy: what the paper needs, and does not have, is an instance with $OPT/FI>4$ in the absence of $t^M_i\ge1$. (Separately, the FI count in the same example is one term too large — see N29.)

---

### M3 — Proposition 4's discretization error compounds; the PTAS guarantee and the running time are both wrong as stated

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>

**Where:** `6_extensions.tex:89` and Proposition 4 at `6_extensions.tex:93–94` · **p. 31** · proof at `OA_B_omitted_proofs.tex:571–574` · **p. OA - 22**.

**Issue.** The load-bearing sentence is:

> "Skill and time costs accumulate additively along the manual and chaining options, so rounding the running totals down to powers of $(1+\epsilon)$ maintains a multiplicative error of at most $(1+\epsilon)$ in each coordinate."

This is false. The table is indexed by the **rounded** skill and time levels, so every transition adds an increment to an already-rounded total and rounds again. Along a job of length $L$ the same coordinate is re-rounded $L$ times and the distortions compose: induction gives only $\hat{c}_L \ge c_L/(1+\epsilon)^L$, with $L$ up to $m$. Worse, with round-**down** the state saturates: once $\hat c > a/\epsilon$, adding an increment $a$ leaves $\rho(\hat c + a)=\hat c$, so the recorded total stops growing altogether.

Consequently the next two sentences — the $(1+\epsilon)^2$ bound on the wage bill and *"Rescaling $\epsilon$ by a constant therefore yields the claimed $(1+\epsilon)$ approximation"* — do not follow. A constant rescaling is not enough; one needs $\epsilon' = \Theta(\epsilon/m)$.

**The running time in Proposition 4 is then also wrong.** With $\epsilon' = \Theta(\epsilon/m)$, each coordinate carries $O(m\epsilon^{-1}\log(mB))$ levels, the table has $O(m^3\epsilon^{-2}\log^2(mB))$ entries, and each is filled in $O(m)$:
$$O\!\left(m^4\,\epsilon^{-2}\log^2(mB)\right)\quad\text{not}\quad O\!\left(m^2\,\epsilon^{-2}\log^2(mB)\right).$$

**Why it matters.** Proposition 4 is the paper's PTAS claim for the long-run problem. As written the algorithm can return a job design whose true cost is $\Theta(m\epsilon)$ times the optimum while reporting a value **below** the optimum, so the proposition is not proved and is false at the stated $\epsilon$.

**Evidence (counterexample satisfying the proposition's own hypotheses).** $B=1$, $m=10$, $c^M_i=t^M_i=1$ for all $i$, $t^H_i=1$ for $i<10$, $t^H_{10}=0$, AI never used. With $\epsilon=0.5$ the powers of $1.5$ are $1,\,1.5,\,2.25,\,3.375,\dots$; rounding the running total down after each unit addition gives $1,\,1.5,\,2.25,\,2.25,\,2.25,\dots$ — it sticks at $2.25$, because $\rho(2.25+1)=\rho(3.25)=2.25$. The DP therefore records a 10-step job as $(2.25,2.25)$ instead of $(10,10)$: a per-coordinate error of $4.44$ against the claimed $1.5$, and a wage bill of $5.06$ against the true $100$. A second instance with $B=5$, $m=30$: exact optimum $171$; the discretized recursion at $\epsilon=0.1$ values the single 30-step job at $117.39$ (true cost $900$), prefers it to every split, and returns a design $5.3\times$ the optimum.

**Fix.** Run the DP with $\epsilon' = \ln(1+\epsilon)/(2m+1)=\Theta(\epsilon/m)$ and replace the sentence with the correct induction: after at most $m$ accumulations each coordinate is within $(1+\epsilon')^m$, so each job's wage bill is within $(1+\epsilon')^{2m+1}\le 1+\epsilon$. (Count the rounding that creates the state $(i,0,t^H_i)$ as well as one per task added.) Correct the same sentence in the body (`6_extensions.tex:89`, p. 31), which also omits the squaring, and replace Proposition 4's running time by $O(m^4\epsilon^{-2}\log^2(mB))$ here and in the two displays at `OA_B:560–567`. Also state the direction of the guarantee explicitly: since rounding is downward, the DP value is a *lower* bound on the cost of the design returned, so either round up or re-evaluate the returned $(\mathcal{T},\mathcal{J})$ exactly.

---

### M4 — The CES derivation drops the boundary condition $\Gamma(1)=0$; the closed-form $\phi$ does not solve the aggregation equation except on a knife-edge

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>

**Where:** `OA_C_CES_representation.tex:296–323` (the solution for $\Gamma(u)$ and Eq. (OA.C.15)/(OA.C.21)) · **p. OA - 32** · claim carried in `6_extensions.tex:32` · **p. 29**.

**Issue.** The appendix defines $\Gamma(u)=\int_u^1\phi(\bar\alpha)\,d\bar\alpha$ and $\Psi(u)=\int_u^1 \phi(\bar\alpha)/\bar\alpha\,d\bar\alpha$, so by construction $\Gamma(1)=\Psi(1)=0$. It then differentiates the aggregation identity, solves the resulting algebraic system for $\Gamma(u)$, and recovers $\phi=-\Gamma'$. But $\Gamma' = -\phi$ determines $\Gamma$ only up to an additive constant, and the boundary condition $\Gamma(1)=0$ is never checked. Writing $B = 1-\theta_M\tau_M^{\rho}$ and $D = \theta_A\tau_A^{\rho}$, the bracket at $u=1$ equals $B - B^{\rho/(\rho-1)}D^{1/(1-\rho)}$, which vanishes **iff $B=D$**, i.e. iff
$$\theta_A\,\tau_A^{\rho} + \theta_M\,\tau_M^{\rho} = 1 .$$
Nothing in the appendix or in Section 6.1 imposes this (nothing normalizes $\tau_A$ or $\tau_M$), so for generic parameters $\int_u^1\phi = \Gamma(u)-\Gamma(1) \ne \Gamma(u)$ and the derived density does **not** satisfy the identity it was constructed to satisfy.

**Why it matters.** This is the whole content of Section 6.1: the body asserts that Appendix OA.C *"derives in closed form the distribution of realized AI effectiveness under which Equation (8) holds."* Without the missing restriction it does not, so the CES representation — cited in the abstract, the introduction (p. 6) and Section 6.1 — is unproved for the parameters the appendix allows.

**Evidence.** Analytically, $B=C \iff B^{1/(1-\rho)}=D^{1/(1-\rho)} \iff B=D$. Numerically (quadrature against the appendix's own $\phi$): with $\rho=-1,\theta_A=0.3,\theta_M=0.4,\tau_A=1.2,\tau_M=0.8$ (so $\theta_A\tau_A^\rho+\theta_M\tau_M^\rho=0.75\ne1$), $\Gamma(1)=0.488155\ne0$ and at $u=0.5$ the paper's $\Gamma(0.5)=0.833333$ while $\int_{0.5}^1\phi = 0.345178$; substituting the true values into the aggregation identity gives LHS $2.897$ vs RHS $2.261$. Imposing $\theta_A\tau_A^\rho+\theta_M\tau_M^\rho=1$ (e.g. $\rho=-1,\tau_A=\tau_M=0.5,\theta_A=0.3,\theta_M=0.2$) restores $\Gamma(1)=0$ and the identity holds to machine precision.

**Fix.** State and impose $\theta_A\tau_A^{\rho}+\theta_M\tau_M^{\rho}=1$ (equivalently $1-\theta_M\tau_M^\rho=\theta_A\tau_A^\rho$, under which the bracket collapses to $\theta_A\tau_A^\rho[1-u^{\rho/(\rho-1)}]$), note that it also delivers $1-\theta_M\tau_M^\rho>0$ (needed for the real powers), and add it to the condition list in Section 6.1's footnote 18. Say plainly that this ties the CES share parameters to the firm-level requirements $(\tau_A,\tau_M)$ rather than leaving them free — which is directly relevant to the "two economies" claim (D24).

---

### M5 — The aggregation identity is verified only along a one-dimensional locus, so it does not pin down a three-input CES

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>

**Where:** `OA_C_CES_representation.tex:190–219`, Eqs. (OA.C.11)–(OA.C.14) · **p. OA - 30**.

**Issue.** Because $\tau_M$ is a constant common to all firms (all firms choose the same $\mathcal{T},\mathcal{J}$ by the appendix's own assumption at line 145), Eq. (OA.C.13) reads
$$\mathcal{M} \;=\; \tau_M\!\int_u^1\!\phi \;=\; \tau_M\,Y ,$$
so aggregate manual labour is **proportional to aggregate output at every wage vector**, while $K$ is held at $1$ throughout. As $u$ varies, the economy therefore traces out only the curve $\{(\mathcal{A},\mathcal{M},K,Y):\mathcal{M}=\tau_M Y,\;K=1\}$. Substituting $\mathcal{M}=\tau_M Y$ and $K=1$ into the CES collapses it identically to $(1-\theta_M\tau_M^{\rho})Y^{\rho}=\theta_A\mathcal{A}^{\rho}+(1-\theta_A-\theta_M)$ — a relation between $Y$ and $\mathcal{A}$ alone.

**Why it matters.** A production function of three arguments is not identified by an identity holding on a one-dimensional curve: infinitely many non-CES functions agree with Eq. (8) there. In particular the elasticity of substitution $\sigma=1/(1-\rho)$ that Section 6.1 advertises has no behavioural content in the constructed economy, because $\mathcal{M}/Y$ never moves. This undercuts *"the implications people draw from improving AI quality in a CES economy carry over intact"* (`6_extensions.tex:37`, p. 29).

**Evidence.** With parameters satisfying the M4 restriction ($\rho=-1,\tau_A=\tau_M=0.5,\theta_A=0.3,\theta_M=0.2$), numerically integrating (OA.C.11)–(OA.C.13): $u=0.3$: $Y=0.5427$, $\mathcal{A}=0.4954$, $\mathcal{M}=0.2714$; $u=0.5$: $0.3515,\,0.2485,\,0.1757$; $u=0.8$: $0.1267,\,0.0708,\,0.0633$. $\mathcal{M}/Y = 0.5 = \tau_M$ at every $u$, while $\mathcal{A}/Y$ moves from $0.913$ to $0.559$.

**Fix.** State that the aggregation is verified along the equilibrium locus generated by the one-dimensional heterogeneity, and that Eq. (8) should be read as a representation valid on that locus rather than as the economy's technology over $(\mathcal{A},\mathcal{M},K)$. Recovering a genuine three-argument CES requires heterogeneity in at least two input coefficients (as in Levhari 1968), so that $\mathcal{A}$ and $\mathcal{M}$ can move independently.

---

### M6 — The central clustering claim is false in the paper's own model: dispersing the AI-able steps can strictly *lower* optimal cost

🟡 <span style="color:#9a6700">**Status: PARTIAL.**</span> Section 4.2 states the condition (footnote after Example 1, plus the closed form in Eq. (4)); the abstract and Section 7.2 are softened. `8_conclusion.tex:7` still carries the unconditional claim.

**Where:** `4_implications.tex:105` and `:169`, `:175` · **pp. 16, 18** · also the abstract (`0_main.tex:75`, p. 1), the introduction (`1_introduction.tex:89`, p. 5), the conclusion (`8_conclusion.tex:7`, p. 41), and the sign prediction $\beta_2<0$ in Section 7.2 (p. 35).

**Issue.** The paper asserts as a general property that *"the more fragmented the AI-able steps are across the production sequence, the fewer of them can share a verification, and the more costly deploying AI becomes."* Holding the multiset of steps fixed, this is not true in the model.

A chain's benefit is that **one** verification (cost $t^A_r$) replaces the standalone cost $t^{*}_i=\min\{t^M_i,\,t^A_i/q_i\}$ of every step it covers, at the price of multiplying the chain's expected cost by $1/q_i$ for each. That trade is *most* favourable for a step whose standalone cost is high relative to $1/q_i$ — i.e. exactly for an **AI-hard, human-advantaged** step. The firm therefore wants each chain to *mix* a reliable step with an expensive one; and because chains must be contiguous, the **alternating** arrangement can be strictly cheapest.

**Counterexample (exact rational arithmetic; $t^A_i=1$ for all $i$, as in Proposition 2).**
Two AI-easy steps $E=(t^M,q)=(2,\,3/4)$ — standalone $\min\{2,4/3\}=4/3$, so AI holds the advantage.
Two AI-hard steps $H=(t^M,q)=(9/5,\,11/20)$ — standalone $\min\{1.8,\,20/11=1.8182\}=1.8$, so the **human** holds the advantage, exactly as the paper's AI-hard steps require.

| Arrangement | $OPT$ | attained by | $FI$ |
|---|---|---|---|
| Clustered $E,E,H,H$ | $2491/495 = 5.03232$ | chain over steps 1–3, step 4 manual | $4331/1200 = 3.60917$ |
| Dispersed $E,H,E,H$ | $160/33 = 4.84848$ | two chains $\{E,H\}$, each $1/(0.75\cdot0.55)=2.42424$ | $4379/1200 = 3.64917$ |

Dispersion is **3.65 % cheaper**, and the fragmentation index ranks the two arrangements in the **opposite** order from $OPT$. (Both values confirmed twice: exhaustive enumeration of all 8 contiguous partitions, and the Proposition 3 DP, in exact fractions. A random search over two-type workflows with an AI-advantaged easy type and a human-advantaged hard type found the dispersed order strictly cheaper in 2,821 of 200,000 draws, with cost reductions up to 5.6 %.)

**Why it matters.** This is the mechanism the whole of Section 4.2 rests on, the motivation for the fragmentation index, and the source of the sign $\beta_2<0$ that Section 7.2 takes to the data. Proposition 2 itself survives — a two-sided constant-factor sandwich cannot order two workflows, and both bounds hold in the counterexample ($FI/OPT = 0.717$ and $0.753$) — but the verbal claim attached to it and the empirical prediction derived from it do not follow from the model.

**Fix.** State the claim conditionally. It holds when the AI-hard steps are cheap enough to execute alone that folding them into a chain never pays — which is exactly the regime of Example 1, where $t^M=6 < t^A/q=11.67$ for the hard step and the hard steps are never chained. Add that hypothesis explicitly at `4_implications.tex:105`, `:169` and `:175`, in the introduction and conclusion, and in the derivation of Prediction #2; or restrict the claim to workflows in which no chain ever contains an AI-hard step. Relatedly, see D8: the index's own monotonicity in clustering needs $t^M_i\ge1$.

---

# MEDIUM

## Model and Sections 3–5

### D1 — Proposition 1 concludes "AI executes step $k$", but everything around it promises "automated"

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>
**Where:** `4_implications.tex:53` (Proposition 1(i)) · **p. 14**; prose at `:2`, `:13`, `:44`.
**Issue.** Definition 3 makes *automated* strictly stronger than *AI-executed*: an augmented step is AI-executed but verified. Proposition 1(i) concludes only that "the cost-minimizing arrangement has AI execute step $k$", and one of the three configurations the proof exhibits — the chain $\{k-1,k\}$, in which $k$ is the augmented endpoint — leaves step $k$ **augmented**, not automated. But the section roadmap ("may nonetheless be **automated** in the optimal configuration"), the subsection opening, and the summary at line 44 all promise automation.
**Why it matters.** The headline "comparative advantage is overturned" is stronger if the human-advantaged step is executed with no verification at all; as stated the proposition permits the weaker outcome.
**Fix.** Either strengthen part (i) to "AI automates step $k$" and cite only the two configurations ($\{k,k{+}1\}$ and $\{k{-}1,k,k{+}1\}$) that deliver it, or align the prose with the proposition and say "executed by AI (automated or augmented)".

### D2 — The grey (manual) region of Figure 3 is misdescribed

🔵 <span style="color:#0969da">**Status: IGNORED.**</span> Reviewed and deliberately not actioned.
**Where:** `4_implications.tex:92` · **p. 16** · "Step~$k$ is performed manually only in the lower left where AI is unreliable on both steps $k$ and $k-1$."
**Issue.** The manual region is not confined to low $q_{k-1}$: it runs the full height of the left edge. At the stated parameters, with a perfectly reliable predecessor ($q_{k-1}=1$) step $k$ is still manual for every $q_k \lesssim 0.11$. Low $q_k$ is necessary; low $q_{k-1}$ is neither necessary nor sufficient.
**Fix.** Describe the region as bounded by the vertical frontier near $q_k\approx0.43$ together with the hyperbola-like frontier in $q_{k-1}q_k$, and start the "moving right / moving up" sentence explicitly from a point low in the grey region.

### D3 — Example 1's "less reliable" AI-easy step also changes its verification cost

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>
**Where:** `4_implications.tex:111` · **p. 16** (found independently by four of the audit slices).
**Issue.** Section 4.1's AI-easy step is $(t^M,t^A,q)=(8,\,4,\,0.9)$. Example 1 says it uses *"a version of its AI-easy step that AI completes less reliably"*, but the step actually used is $(8,\,6,\,0.7)$ — the verification cost also rises from $4$ to $6$. Two parameters change; the prose describes one.
**Why it matters.** A reader tracking the running example cannot reproduce the numbers, and the change in $t^A$ is not innocuous: it is part of what makes chaining marginal in panel (a).
**Fix.** "…and a version of its AI-easy step that is both costlier to verify and less reliable, $(8,\,6,\,0.7)$."

### D4 — "no chain can form" in Example 1 is a feasibility claim where an optimality claim is meant

🟡 <span style="color:#9a6700">**Status: PARTIAL.**</span> Body text fixed (`4_implications.tex:136`); the Figure 4 notes at `:131` still say “no chain can form”.
**Where:** `4_implications.tex:137` and the Figure 4 notes at `:131` · **p. 16–17**.
**Issue.** "Because every easy step is flanked by hard ones, no two AI-able steps are ever adjacent, so no AI chain can form." Nothing in the model restricts chains to AI-able steps: Definition 4 admits any contiguous block, every singleton is a feasible length-one chain, and every adjacent pair is a feasible length-two chain. What is true is only that none of them is cost-minimizing.
**Fix.** "…so no chain over two or more steps is worth forming, and neither kind of step is worth augmenting on its own; every step is performed manually."

### D5 — Proposition 2 bounds the *cost* of AI deployment, not the *benefit* from AI

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>
**Where:** `4_implications.tex:169` · **p. 18** · "so a workflow with a high fragmentation index yields less benefit from AI however cleverly the firm deploys it." Same slip in the introduction (`1_introduction.tex:91`, p. 5).
**Issue.** Proposition 2 sandwiches $FI$ against $OPT$, the *level* of minimized production cost. The benefit from AI is $\sum_i t^M_i - OPT$, and nothing in the proposition bounds it. A high $FI$ implies a high $OPT$, which is consistent with either a large or a small saving relative to all-manual execution.
**Fix.** "…so a workflow with a high fragmentation index is costly to produce even under the best AI deployment." If a statement about gains is wanted, restrict it to comparisons across re-orderings of the *same* set of steps, where $\sum_i t^M_i$ is fixed and cost and gain move one-for-one.

### D6 — The "prophet" description does not reproduce Equation (3)

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>
**Where:** `4_implications.tex:170–171` · **p. 18**.
**Issue.** The prophet is said to "chain each such run and execute every remaining step on its own, in whichever of the two modes is cheaper for it." The "whichever is cheaper" clause is attached only to the failed steps, giving the first term $\min\{t^M_i,1/q_i\}$. But the component term in Eq. (3) is $\omega(C)=\min\{1,\sum_{s_i\in C}t^M_i\}$ — i.e. the prophet *also* runs a whole success-run manually when that beats the single verification. Under the description as written each success-run would be charged exactly $1$ and the index would be strictly larger.
**Fix.** "…chains each such run, unless performing that whole run manually is cheaper, and executes every remaining step on its own in whichever of the two modes is cheaper."

### D7 — The advertised greedy algorithm is not the one that carries the guarantee

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>
**Where:** `4_implications.tex:176` · **p. 19**.
**Issue.** The body describes the greedy as grouping steps into a chain while the success probability stays high, "and converting **chains of length one** to manual execution where that is cheaper." The algorithm actually analysed in Lemma OA.B.3 (`OA_B:349`) switches an **entire constructed task of any length** to manual whenever $\sum_{s_i\in T_b}t^M_i<1$. As described in the body the algorithm has no constant-factor guarantee.
**Fix.** "…terminating the chain and starting a new one when it falls too low, and running an entire chain manually whenever its steps' total manual time is below the cost of one verification."

### D8 — "the index is lower exactly when the reliable steps are clustered" needs the $t^M_i\ge1$ hypothesis

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>
**Where:** `4_implications.tex:175` · **p. 18**.
**Issue.** The mechanism given ("a single component is charged only once") operates only when a merged component's manual costs exceed the normalized verification cost, because $\omega(C)=\min\{1,\sum_{s_i\in C}t^M_i\}$. When the reliable steps are cheap by hand ($\sum t^M_i<1$ over a run), merging them saves nothing. "Exactly when" is false in both directions; see also M6.
**Fix.** Replace "exactly when" by a conditional statement and state the hypothesis: the charge-once saving needs $\sum_{s_i\in C}t^M_i\ge1$ for each merged run, which is precisely the extra assumption Lemma OA.B.2 imposes.

### D9 — Section 4.3's stated mechanism fails at the first threshold of the paper's own example

🔵 <span style="color:#0969da">**Status: IGNORED.**</span> Reviewed and deliberately not actioned.
**Where:** `4_implications.tex:190–192` · **p. 19** · "at each of them those returns jump upward **as longer AI chains become worth deploying**."
**Issue.** At $\alpha=0.50$ the optimum switches from "both steps manual" to "step 2 augmented alone". No chain is extended, no previously-manual step is absorbed into an existing chain, and nothing compounds over a longer stretch — the chain goes from *absent* to length one. Lemma OA.B.4 proves nothing about chain length; it is purely about the sign of the jump in $g^*$.
**Fix.** "…and at each of them those returns do not fall, typically jumping upward as the firm redeploys AI over a larger or more chained portion of the workflow." Drop the universal "as longer AI chains become worth deploying", or restrict it to thresholds at which a chain is genuinely extended.

### D10 — The productivity J-curve attribution is not what the model delivers

🔵 <span style="color:#0969da">**Status: IGNORED.**</span> Reviewed and deliberately not actioned.
**Where:** `4_implications.tex:237–239` · **p. 20**.
**Issue.** The J-curve of Brynjolfsson et al. (2021) is an initial *decline* in measured productivity below trend, driven by unmeasured intangible investment, followed by a later rise. In this model $C^*(\alpha)=\min_{\mathcal{T}}C_{\mathcal{T}}(\alpha)$ is weakly decreasing and $g^*(\alpha)\ge0$ everywhere: cost never rises, so there is no initial dip and no "J". What the model delivers is a flat-then-lumpy marginal-return profile.
**Fix.** Say what the model gives — "the same *lumpiness* arises within a single workflow, because the returns to better AI wait on discrete reorganizations of it" — and drop the identification with the J-curve shape, or state explicitly that the model captures the delayed-payoff half of the pattern and not the measured-output decline.

### D11 — "the firm cannot solve the problem step by step and must compare all feasible arrangements" contradicts Proposition 3

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** `3_shortrun.tex:125` · **p. 12**; repeated at `6_extensions.tex:47–48` · **p. 30**.
**Issue.** Proposition 3 states that Problem (1) is solved exactly in $O(m^2)$ by a dynamic program that does proceed step by step — Section 6.2's own words, four sentences after repeating the claim, are "building the workflow up from the beginning one step at a time". The intro repeats the claim at `1_introduction.tex:94` (p. 5).
**Fix.** "…the firm cannot decide each step's mode in isolation; cost minimization compares arrangements of contiguous blocks rather than steps. Section 6.2 shows this can nonetheless be done exactly in $O(m^2)$ time."

### D12 — "Both components of a chain's pair are inherited from its augmented endpoint" is false for the time component

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** `5_longrun.tex:47` · **p. 22**.
**Issue.** The pair printed in the immediately preceding sentence is $\bigl(c^A_r,\; t^A_r/\prod_{i=\ell}^{r} q_i\bigr)$. Only the first argument is a function of $r$ alone; the second depends on the success probability of every step in the chain. The skill claim is right, the time claim is not.
**Fix.** "The chain's *skill* is inherited from its augmented endpoint — it demands only the skill required to verify that step's output — while its *time* is the endpoint's verification cost inflated by the failure risk of the whole chain."

### D13 — Footnote 13's "considerably weaker version" does not support Section 5.4

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** `5_longrun.tex:52–56` · **p. 22**.
**Issue.** The footnote claims the results rest only on "each step demanding some increment of skill no other step supplies". That monotonicity is enough for the frictionless benchmark of Section 5.2 (if $c(J)\ge\max_{b\in J}c_b$ then bundling never helps), but not for Section 5.4's comparative statics: with merely monotone, non-additive skill the "idle skill a bundled job pays for" need not fall when AI strips skill out of one task, so the skill channel is not signed.
**Fix.** "That said, the frictionless benchmark of Section 5.2 requires only that a job's skill requirement be strictly increasing in the set of steps it contains; the comparative statics of Section 5.4 use additivity."

### D14 — The long-run recursion is self-referential and does not determine $V(i)$

🔵 <span style="color:#0969da">**Status: IGNORED.**</span> Reviewed and deliberately not actioned.
**Where:** `6_extensions.tex:78–84` · **p. 31** · same recursion at `OA_B:512–527` · **p. OA - 20**.
**Issue.** Every branch of Eq. (10) except the first moves to a strictly smaller step index, but "close the job" calls $V(i)=R(i,0,t^H_i)$ at the *same* $i$. Evaluated at the state $(i,0,t^H_i)$ that branch reads $0\cdot t^H_i + V(i) = V(i)$, so the recursion says $V(i)=\min\{V(i),\,A_i\}$, satisfied by every $V(i)\le A_i$. The same self-loop occurs at the root $(m,0,0)$. Substantively it corresponds to closing a job holding no tasks, which Definition 6 does not admit.
**Fix.** Define $V(i)$ as the minimum over the manual and chain branches only, evaluated at $(i,0,t^H_i)$ — the closure branch is vacuous there because a job with zero skill has zero wage bill — and state the fill order: complete all layers $r<i$, then the entry $(i,0,t^H_i)$, then the remaining layer-$i$ entries.

## Appendix OA.B (proofs)

### D15 — The realized-fragmentation identity in Lemma OA.B.2 over-counts by 1 whenever the last step fails

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>
**Where:** `OA_B_omitted_proofs.tex:309–311` · **p. OA - 14**.
**Issue.** The proof states that realized fragmentation equals "$1$ plus, for each $s_i\in F$, a contribution of $t^{*}_i$ plus an additional $1$ in the event that $i>1$ and the step immediately preceding $s_i$ did not fail." Under $t^M_i\ge1$ every $\omega(C)=1$, so realized fragmentation $=\sum_{s_i\in F}t^{*}_i + \#\mathcal{C}$, and $\#\mathcal{C}=\#\{i\in F: i>1,\ s_{i-1}\notin F\}+\mathbf{1}\{s_m\notin F\}$ — not $\#\{\cdot\}+1$. Check $m=1$ with the single step failing: the true value is $t^{*}_1$; the displayed rule gives $1+t^{*}_1$.
**Why it matters.** Lemma OA.B.2 needs a **lower** bound on $FI$, so an over-estimate is not a harmless slip.
**Fix.** Replace "equal to $1$ plus" by "equal to $\mathbf{1}\{s_m\notin F\}$ plus", and add one line to the charging argument: the terminal task's baseline $1$ is charged in the event $s_m\notin F$, which has probability $q_m\ge\alpha^{d(T_b)}\ge1/2$, so the $\ge 1/2$ charging still goes through and the constant $4$ survives.

### D16 — Lemma OA.B.3's charging argument never charges the terminal task

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>
**Where:** `OA_B_omitted_proofs.tex:360–368` · **p. OA - 16**.
**Issue.** The charging is defined only for **non-terminal** $T_b\in\mathcal{T}_{NI}$ (it runs off the event $E_b$, which involves the step following $T_b$), but the bound it proves sums over **all** of $\mathcal{T}_{NI}$, terminal task included. The device inherited from Lemma OA.B.2 ("charge the baseline 1 to the terminal task") does not transfer, because under general $t^M_i$ the baseline component weight is $\min\{1,\sum t^M_i\}$, not $1$.
**Fix.** Add the terminal case explicitly: either some step of the terminal $T_b$ fails (charge $\sum_{i\in F\cap T_b}t^{*}_i$ plus the $\omega$ of the components meeting $T_b$), or no step of $T_b$ fails, in which case $T_b\subseteq C$ for some component and $\omega(C)\ge\min\{1,\sum_{s_i\in T_b}t^M_i\}$. Either way the terminal task is charged at least $\min\{1,\sum_{s_i\in T_b}t^M_i\}$ with probability $1$.

### D17 — The "charged to at most two tasks" justification in Lemma OA.B.3 is invalid as stated

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>
**Where:** `OA_B_omitted_proofs.tex:363–366` · **p. OA - 16**.
**Issue.** The parenthetical reads: *"if some $T_b$ is a subset of $C$, then by definition no step of $T_b$ fails and hence $E_b$ did not occur."* But $E_b$ is the event that some step of $\overline{T}_b = T_b\cup\{\text{next step}\}$ fails, not of $T_b$. So $T_b\subseteq C$ is perfectly compatible with $E_b$ occurring — namely when the step immediately following $T_b$ fails.
**Why it matters.** This parenthetical is the entire justification of the "at most two charges" step, which is what produces the factor $4$ in $1/4\cdot\sum(\cdot)$ and hence the constant $8$.
**Fix.** Argue directly: among the tasks intersecting $C$, every task strictly interior to $C$ has its successor step inside $C$ (hence non-failed) and so does not trigger $E_b$; only the leftmost intersecting task and the rightmost one can. That still gives at most two charges, so the constant survives, but the stated reason must be replaced.

### D18 — Lemma OA.B.4 asserts a strict upward jump; the proof gives only a weak inequality

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** `OA_B_omitted_proofs.tex:414` (lemma) and `:440–454` (proof) · **p. OA - 17/18**.
**Issue.** From $\phi(\alpha_0^-)\ge0$, $\phi(\alpha_0)=0$, $\phi(\alpha_0^+)\le0$ the proof concludes $\phi'(\alpha_0)\le0$, hence $g_{\mathcal{T}}(\alpha_0)\le g_{\mathcal{T}'}(\alpha_0)$ — written with a weak $\le$ in the display. Nothing rules out $\phi'(\alpha_0)=0$, i.e. a tangential crossing at which $g^*$ is continuous and there is no jump.
**Fix.** Either restate as "$g^*$ does not jump downward: $\lim_{\alpha\uparrow\alpha_0}g^*\le\lim_{\alpha\downarrow\alpha_0}g^*$", or add the transversality hypothesis $\phi'(\alpha_0)\ne0$ (equivalently, the two cost curves cross rather than touch), noting it holds generically and at both thresholds of Example 2.

## Appendix OA.C (CES) and Section 6.1

### D19 — The AI failure inflation $\alpha^{-d_b}$ is counted twice

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span> Fixed in `677089c`. `\tau_b` is now built from the single-attempt time `t^{E(b)}_b`, with `t_b = t^{E(b)}_b \alpha^{-d_b}` stated explicitly, so the inflation is counted once in Eqs. (OA.C.2), (OA.C.7) and the wage-bill sentence.
**Where:** `OA_C_CES_representation.tex:44` (definition of $\tau_b$) and `:48`, `:80` (Eq. OA.C.2), `:111` (Eq. OA.C.7) · **pp. OA - 24/25/26/27**.
**Issue.** $\tau_b$ is built from $t_b$, and Definition 5 (p. 12) and Table OA.A.1 (p. OA - 1) define $t_b$ for an AI chain over $(s_\ell,\dots,s_r)$ as $t^A_r/\prod_{i=\ell}^{r} q_i = t^A_r\,\alpha^{-\sum_i d_i}$ — the *expected* time, already inclusive of the success probability. The appendix then multiplies by $\alpha^{-d_b}$ **again**: p. OA-25 writes the expected wage bill of an AI task as $w_A\tau_b\alpha^{-d_b}$, Eq. (OA.C.2) divides labour by $\tau^A_b\alpha^{-d_b}$, and Eq. (OA.C.7) defines $\bar\alpha$ the same way.
**Why it matters.** $\bar\alpha$ — the single dimension of firm heterogeneity in the whole aggregation — is defined by this expression, so the double count propagates through everything downstream.
**Fix.** State at the top of Appendix OA.C that $t_b$ there denotes the **single-attempt** time requirement ($t^M_i$ for a manual step, $t^A_r$ for a chain augmented at $r$), so that the expected time is $t_b\,\alpha^{-d_b}$; or keep the body's expected-cost $t_b$ and delete every $\alpha^{-d_b}$ factor in Eqs. (OA.C.2), (OA.C.7) and the p. OA-25 sentence.

### D20 — The source of firm heterogeneity in $\bar\alpha$ is never reconciled with its definition

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** `OA_C_CES_representation.tex:141–147` · **p. OA - 28**; claim carried at `6_extensions.tex:31–32` · **p. 29**.
**Issue.** Eq. (OA.C.7) makes $\bar\alpha$ a deterministic function of $\alpha$ and of $\{\tau^A_b,d_b\}$. Section 3 and Table OA.A.1 define $\alpha$ as a *single economy-wide* object, and the appendix assumes at line 145 that "firms hold the same expectations … [and] choose identical AI strategies and job designs", which makes $\{\tau^A_b,d_b\}$ common too. Under those assumptions $\bar\alpha$ is identical across firms and the distribution $\phi$ has nothing to be a distribution over. Line 141 simply posits the dispersion instead.
**Why it matters.** Section 6.1 says the appendix "obtains that dispersion from the order in which a firm has to commit" — but commitment timing only prevents re-optimization; it generates no dispersion. The dispersion is an exogenous assumption.
**Fix.** Introduce an explicit firm-level primitive that $\bar\alpha$ inherits (e.g. a firm-specific deployment effectiveness $\alpha_f = \zeta_f\alpha$, or firm-specific difficulties $d_{f,b}$), say that firms know its distribution but not their draw, and correct the Section 6.1 sentence to say the timing makes realized effective AI quality the *only operative* dimension of dispersion rather than its source.

### D21 — "No individual firm substitutes … since production is Leontief" is a non sequitur

🔵 <span style="color:#0969da">**Status: IGNORED.**</span> Reviewed and deliberately not actioned.
**Where:** `6_extensions.tex:30` · **p. 29**.
**Issue.** Leontief-ness is over *steps* — every step must be done — but which labour type performs a step is exactly the firm's choice (Definition 5's AI strategy; the whole of Section 4 is about steps moving onto AI). Under the appendix's own wage formulation, Eq. (OA.C.1), the firm's cost depends on $w_A$ and $w_M$ separately, so its cost-minimizing AI strategy does respond to relative wages. What actually shuts down firm-level substitution is the commitment timing.
**Fix.** "No firm in the aggregation substitutes AI management labour for manual labour, because it commits to its AI strategy and job design before it hires; conditional on that commitment its technology is Leontief in tasks."

### D22 — Equation (8)'s capital term is assumed, not derived

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span> Fixed in `e71b3a1`, though in Appendix OA.C rather than in footnote 18: “its exponent is part of the CES form we posit rather than something the aggregation derives.”
**Where:** `6_extensions.tex:22–26` · **p. 29**; capital held at $K=1$ throughout `OA_C:138–160`.
**Issue.** Eq. (8) is presented as a three-input aggregate production function obtained from the firm-level technology, but capital never varies in the derivation: the functional equation the appendix solves, (OA.C.16), carries the capital term as the constant $(1-\theta_A-\theta_M)$. Nothing in the argument identifies the exponent $\rho$ on $K$, or indeed the functional form of the capital term at all.
**Fix.** Either let $K$ vary in the aggregation and derive the term, or state in Section 6.1 and in footnote 18 that capital is held at its normalized level throughout and that its CES exponent is an assumption rather than a result.

### D23 — "the implications people draw from improving AI quality in a CES economy carry over intact" is unsupported

🟡 <span style="color:#9a6700">**Status: PARTIAL.**</span> “carry over intact” softened to “largely carry over”; no mapping from $\alpha$ to $(\theta_A,\theta_M,\rho)$ supplied.
**Where:** `6_extensions.tex:37` · **p. 29**.
**Issue.** AI quality $\alpha$ appears nowhere in Eq. (8); its only parameters are $\theta_A,\theta_M,\rho$. The appendix runs the mapping in the opposite direction — it *fixes* $(\theta_A,\theta_M,\rho)$ and solves for the heterogeneity distribution that makes the identity hold. The paper supplies no map from $\alpha$ to the CES parameters, and the next two sentences (parameters absorb the organization of work) point the other way.
**Fix.** Either supply the mapping from $\alpha$ to $(\theta_A,\theta_M,\rho)$, or weaken to: "the aggregate economy inherits the CES form that literature relies on, though the CES parameters themselves depend on how work is organized and therefore on AI quality."

### D24 — "Two economies … aggregate to the same CES form but with different parameters" is not established

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span> Fixed in `a2038a0`: the two-economies sentence is deleted, leaving only the weaker point that aggregate data cannot reveal workflow structure.
**Where:** `6_extensions.tex:39` · **p. 29**.
**Issue.** Appendix OA.C never compares two arrangements and never derives $\theta_A,\theta_M,\rho$ from a workflow; it fixes them and solves backwards for $\phi$ (lines 163–164). Under M4's missing restriction $\theta_A\tau_A^\rho+\theta_M\tau_M^\rho=1$, the share parameters are in fact *tied* to $(\tau_A,\tau_M)$, so two arrangements with different $\tau$'s cannot freely share CES parameters — which makes the sentence's independence claim harder, not easier, to sustain.
**Fix.** Either establish the comparison (fix $\phi$, vary the workflow arrangement, and show the resulting CES parameters differ), or delete the sentence and keep only the weaker point that follows it — that aggregate data cannot reveal workflow structure.

### D25 — Footnote 18's list of conditions for Equation (8) is incomplete

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span> Fixed in `a2038a0`: $\rho<0$, hence $\sigma<1$, is now stated next to Eq. (8) itself, which is the review's preferred remedy.
**Where:** `6_extensions.tex:33` · **p. 29**.
**Issue.** The footnote lists three conditions (common capital productivity, identical organization, one dimension of heterogeneity). The appendix additionally assumes $\rho<0$, hence $\sigma<1$ (`OA_C:165`, p. OA - 29); assumes $0<w_A\tau_A/(1-w_M\tau_M)<1$ (`OA_C:176` footnote); needs $1-\theta_M\tau_M^\rho>0$ and the bracket in (OA.C.21) positive for the real powers to be defined; and needs the normalization identified in M4.
**Fix.** Extend footnote 18 to say the aggregation is obtained only for $\rho<0$ — an elasticity of substitution below one — and that it also requires the parameter restrictions in Appendix OA.C. Better still, state $\rho<0$ next to Eq. (8) itself, since it is a substantive restriction on the aggregate technology.

### D26 — The capital normalization is inconsistent with treating $K$ as fixed at 1

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span> Fixed in `e71b3a1`: the per-output capital normalization is gone; aggregate capital is normalized to 1 once, and capital is stated to play no part in the allocation.
**Where:** `OA_C_CES_representation.tex:138–140` and `:160` · **pp. OA - 28/29**.
**Issue.** Line 140 normalizes "exactly one unit of capital … per unit of **output** produced", which makes a firm's capital equal to its output $y$ and hence aggregate capital equal to $Y$. Line 160 instead argues that because the measure of firms is $1$, aggregate capital is $1$ — a per-*firm* normalization. The two cannot both hold unless $Y=1$. Relatedly, the profitability condition at `:170` is $w_A l_A + w_M l_M \le y$ with no capital cost, and the firm's technology (OA.C.4) has no capital argument, so nothing caps firm scale.
**Fix.** State that $K$ in Eq. (OA.C.9) is the aggregate capital *stock*, fixed at 1 and independent of the entry margin, and that this is an additional assumption; or carry the capacity constraint into the firm problem as $y=\min\{\bar\alpha l_A/\tau_A,\ l_M/\tau_M,\ k\}$ with $k$ the firm's given stock, and re-derive.

## Section 7 (Empirical Evaluation)

### D27 — "AI-exposed" is defined three inconsistent ways

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** `7_empirics.tex:19` (p. 32), `:116` footnote 21 (p. 36), and the Table 2 notes at `:132–134` (p. 36).
**Issue.** p. 32: *"we … treat their E1 category as exposed to AI and the remaining categories as unexposed, which yields a conservative measure of exposure."* p. 36 footnote: *"We count as AI-exposed both E1- and E2-exposed tasks, which together account for 44 % of the tasks; restricting attention to E1-exposed tasks alone (14 %) would leave potential AI chains too sparse."* The Table 2 notes carry both conventions within one paragraph: the regressor "AI Exposure" is *"the share of AI-exposed (E1) steps"* while the control is *"the number of AI-exposed (E1 or E2) steps."*
**Why it matters.** A reader cannot tell which exposure set produced the headline coefficients of Table 2, and the two sets differ by a factor of three in coverage.
**Fix.** Define the two concepts once, in the data section, with distinct names — e.g. **narrow exposure** (E1, 14 %) and **AI-able** (E1 ∪ E2, 44 %) — say which is used for the regressor, which for the EFI and its count control, and then never use the bare phrase "exposed to AI" again (lines 92–93, 102, 109–117, 132–134, 140).

### D28 — The EFI is a different object from the fragmentation index of Equation (3), and the gap is not stated

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>
**Where:** `7_empirics.tex:109–115` · **p. 35**, against Eq. (3) on p. 18.
**Issue.** Eq. (3) is an expected **cost**, $\sum_{s_i\in F}\min\{t^M_i,1/q_i\}+\sum_{C}\min\{1,\sum_{s_i\in C}t^M_i\}$, depending on manual times and on the success probabilities through the random failure set. The EFI is a normalized block count, $(m-k+r)/m$ where $k$ is the number of AI-able steps and $r$ the number of maximal runs of them. The two coincide only in the degenerate special case $t^A_i=t^M_i=1$ with $q_i\in\{0,1\}$; the paper says it "adapts" the index but never says what is lost.
**Why it matters.** Proposition 2 is a statement about Eq. (3), and Section 7.2 uses it to motivate the sign of $\beta_2$ on the EFI. The link needs stating.
**Fix.** State the special case under which EFI $=FI/m$ ($t^A_i=t^M_i=1$, $q_i\in\{0,1\}$), and say that the empirical measure captures only the arrangement channel of the index, holding the cost and reliability channels at their degenerate values.

### D29 — The EFI still varies with workflow length, which Equation (11) does not control

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** `7_empirics.tex:98–102` · **p. 35**.
**Issue.** With $m$ steps, $k$ AI-able steps and $r$ maximal runs, EFI $=1-(k-r)/m$. Conditional on the control $k$ (the count of AI-able steps), the EFI still moves with $m$ as well as with the arrangement $r$, and $m$ is not in Eq. (11). Since the dependent variable is also a share with $m$ in the denominator, occupation length is a live confound, so "this control **ensures** that $\beta_2$ is identified from how those steps are arranged rather than how many there are" is too strong.
**Fix.** Add $m$ (or $\log m$) to Eq. (11), or replace the EFI with the scale-free statistic $r/k$ (the share of AI-able steps that start a run); and soften "ensures" to "is intended to absorb".

### D30 — The predicted signs $\beta_1>0$ and $\beta_2<0$ are not derived from any result in the paper

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>
**Where:** `7_empirics.tex:104–105` · **p. 35**.
**Issue.** The only formal result about fragmentation is Proposition 2, a two-sided constant-factor sandwich, which cannot ordinally rank two workflows' $OPT$ (a factor-8 gap on one side and 5/4 on the other). And $OPT$ is a **cost**, whereas $\beta_2$ concerns an execution **share**. Proposition 1 is flagged by the paper itself as local. The signs are motivated by Example 1, not derived — and M6 shows the underlying comparative static is not general.
**Fix.** Present the signs as what they are: the empirical content of the mechanism illustrated in Example 1, under the conditions M6 identifies. Do not attribute them to Proposition 2.

### D31 — The execution-based EFI is an exact algebraic function of the dependent variable

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** `7_empirics.tex:141–145` · **p. 37**.
**Issue.** With AI-able $:=$ AI-executed, $\text{EFI}_{\text{exec}} = (m-k_{\text{exec}}+r_{\text{exec}})/m = 1 - \text{ai\_execution} + r_{\text{exec}}/m$ identically. Footnote 22 acknowledges the mechanical relation, but the very next sentence treats the exposure-based and execution-based measures as two sources pointing to "a common pattern", and the abstract-level summary follows.
**Fix.** Either drop the execution-based specification from the evidentiary summary, or state the identity explicitly and reframe the exercise as a description of how AI-executed steps are arranged (a contiguity statistic) rather than as corroborating evidence for the fragmentation mechanism.

### D32 — "occurs through contiguous chains rather than as isolated step substitutions" is not identified by Equation (11)

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>
**Where:** `7_empirics.tex:145` · **p. 37**.
**Issue.** Eq. (11) is an occupation-level regression whose only outcome is the *share* of tasks executed by AI. It contains no variable separating chained execution from isolated substitution, so the composition clause does not follow from Table 2.
**Fix.** Delete the clause, or support it with the Prediction #1 chain-length evidence and say so.

### D33 — The measured "AI chain" is not the model's AI chain

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** `7_empirics.tex:36` (footnote 20) and `:42–55` · **pp. 33–34**.
**Issue.** Definition 4 requires all steps but the last to be automated and the last to be augmented. Footnote 20 discards exactly that information — "we therefore do not distinguish the two labels … and treat both as indicating AI execution" — because the Anthropic labels are position-agnostic. The statistic reported as "average AI chain length" is therefore the mean length of a maximal run of AI-*executed* steps, which is a weaker object: a run of four augmented steps counts as a chain of length four in the data and as four length-one chains in the model.
**Fix.** Rename the statistic (e.g. "average run length of AI-executed steps"), state the discrepancy where the statistic is introduced rather than only in a footnote a page earlier, and say that Prediction #1 is a test of co-occurrence rather than of chain structure.

### D34 — "our model predicts chains to lengthen as AI quality improves" does not follow

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** `7_empirics.tex:55` · **p. 34**; repeated at `:227` (p. 39) and `1_introduction.tex:144` (p. 7).
**Issue.** The measured object is the *average* chain length, (AI-executed steps)/(number of chains). Rising $\alpha$ raises the numerator but can raise the denominator faster, because a step that becomes worth augmenting on its own opens a **new** length-one chain. Average chain length is therefore not monotone in $\alpha$ — as the paper's own Example 2 illustrates, where the first threshold creates a chain of length one.
**Fix.** State the claim the model supports — that any given chain weakly beats its splits as $\alpha$ rises, and that the set of AI-executed steps is monotone (Proposition 1(ii)) — and drop the implication for the average.

### D35 — Equation (12) puts an additive error term inside the logistic CDF

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** `7_empirics.tex:173–188` · **p. 38**.
**Issue.** The displayed equation is $\Pr(\text{is\_ai}_k=1\mid X_k)=\Lambda(\beta_0+\dots+\beta_4\,\text{next2\_is\_ai}_k+\varepsilon_k)$. The left side is a deterministic function of $X_k$; the right side with a random $\varepsilon_k$ is a random variable, so the equation cannot hold. In a logit the disturbance is the latent-index shock that $\Lambda$ has already integrated out.
**Fix.** Either drop $\varepsilon_k$ and write $\Pr(\cdot\mid X_k)=\Lambda(\beta_0+\dots+\beta_4\,\text{next2\_is\_ai}_k+\gamma'Z_k)$, or give the latent form explicitly: $\text{is\_ai}_k=\mathbf{1}\{y^*_k>0\}$ with $y^*_k=\beta_0+\dots+\gamma'Z_k+\varepsilon_k$ and $\varepsilon_k\mid X_k$ logistic.

### D36 — Only columns (4) and (6) implement the comparison Prediction #3 states

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** `7_empirics.tex:153`, `:212–217` · **p. 38–39**.
**Issue.** Prediction #3 is explicitly a within-step comparison — "when the same step appears in two occupations…" — and DWAs are the paper's operationalization of "the same step". Only the DWA-fixed-effects columns hold the step fixed. Columns (1)–(3) and (5) compare *different* DWAs and are consistent with a pure composition story (AI-able DWAs cluster in AI-intensive occupations). The text presents column (1) as the headline result and describes the rest as "progressively demanding comparisons", which understates the difference in kind.
**Fix.** Make column (4) the headline for Prediction #3, present (1)–(3) as descriptive, and say explicitly that only the DWA-FE columns implement the stated comparison.

### D37 — Causal language for what the design identifies as equilibrium co-occurrence

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** Table 3 caption `7_empirics.tex:196`, text at `:212` and `:225`, Figure OA.A.1 caption/notes `OA_A:160`, `:198`, abstract `0_main.tex:75` · **pp. 38–39, OA - 4, 1**.
**Issue.** Eq. (12) regresses one equilibrium outcome on four others. In the model the entire partition is chosen simultaneously by one cost minimization; indeed if step $k$ is chained with $k-1$ then both are AI-executed **by construction**, so the regressand and the regressor are partly the same event. "Raises the probability", "increases the likelihood", and the abstract's "adjacency … increases the likelihood" assert a direction the design cannot deliver. Note that Prediction #2 is written correctly ("is associated with", "predict"); the inconsistency is internal.
**Fix.** Use the Prediction #2 register throughout: "is associated with", "co-occurs with", "predicts". Add one sentence stating that neighbours' execution status is jointly determined with the focal step's, so the coefficients are equilibrium co-occurrence patterns consistent with chaining rather than causal effects.

### D38 — Column (5) contradicts the sentence summarizing Table 3

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** `7_empirics.tex:216` · **p. 39**, against `tables/noTasksWithRepetitiveDWAs/allTasks_ai.tex`.
**Issue.** "The pattern is the same throughout: the immediate neighbour effect attenuates … while the distant neighbour effects shrink toward zero and lose significance." In column (5) the $k-1$ AME is **0.13**, *larger* than the 0.12 baseline, and the $k-2$ AME is 0.06\* — still significant at 10 % and essentially unchanged from the 0.07 baseline.
**Fix.** "The pattern is the same in every specification that absorbs occupation-family or DWA heterogeneity (columns (2)–(4) and (6)): the immediate-neighbour effect attenuates from 0.12 to 0.04–0.06 but stays significant, while the distant-neighbour effects fall to between −0.01 and 0.01 and lose significance. Column (5), which adds only the same-DWA task count and no fixed effects, leaves the baseline pattern intact."

### D39 — The sample drop in columns (4)–(6) is unexplained

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** `tables/noTasksWithRepetitiveDWAs/allTasks_ai.tex` (Observations row) and `7_empirics.tex:215` · **p. 38**.
**Issue.** Observations fall from 10,708 (cols 1–2) to 9,861 (col 3) to 4,096 (cols 4–6). Column (5) carries **no** fixed effects and differs from column (1) only by one control, yet loses 62 % of the sample — so it has evidently been held to the column-(4) DWA-FE estimating sample, which the text never says. The number of DWA clusters is not reported anywhere.
**Fix.** State that columns (4)–(6) are estimated on the sample of DWAs with within-DWA outcome variation (4,096 tasks), report the number of clusters per column, and either run column (5) on the full sample or say it is held to the column-(4) sample for comparability. Qualify "relative to the baseline" at line 216 accordingly.

### D40 — The distant-neighbour result is reported as positive evidence on p. 38 and as *below* the placebo on p. 39

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** `7_empirics.tex:212` vs `:222–223` · **pp. 38–39**.
**Issue.** Line 212 reports that "AI execution of both immediate and **more distant** neighbours raises the probability that the focal task is AI-executed" (col. 1: 0.07\*\*\* and 0.05\*\*\*, significant against a null of zero). Ten lines later the paper says the placebo distributions show "what these coefficients would look like when the position of a task in the workflow does not matter", and that the actual orderings deliver "**weaker** distant-neighbour effects than the placebos". Relative to the paper's own stated benchmark, then, the distant-neighbour coefficients are evidence *against* a positional effect, not for one. The blanket claim that the placebo comparison shows what "the reshuffled orderings cannot reproduce by chance" also does not hold uniformly across the four panels.
**Fix.** Reconcile the two passages: report the distant-neighbour coefficients against the placebo benchmark from the outset, and grade the placebo claim by panel rather than asserting it for all four.

## Framing (abstract, introduction, conclusion)

### D41 — The abstract states the CES result unconditionally

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span>
**Where:** `0_main.tex:76` · **p. 1**.
**Issue.** "…and show that firm-level production aggregates to a macroeconomic CES production function." Every other statement of the result is qualified — the introduction (p. 6) and Section 6.1 (p. 28) both say "under additional assumptions" — and the appendix does not derive CES from the firm-level model: it assumes the CES form and solves backwards for the firm-heterogeneity distribution that would rationalize it (and see M4, M5).
**Fix.** "…and show that, under additional assumptions on how firms differ in their effective AI quality, firm-level production aggregates to a macroeconomic CES production function."

### D42 — The external-validation summary overstates Section 7

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** `1_introduction.tex:138–139` · **p. 7**.
**Issue.** Two overstatements. The predictions are re-estimated only on the APQC/PCF corpus, not on both external sources — the event logs are used solely for the ordering-recovery exercise. And only Predictions #1 and #2 are re-tested; Section 7.4 says explicitly that Prediction #3 cannot be re-run.
**Fix.** "Our sequencing approach recovers the orderings in both external sources to a substantial degree, and Predictions #1 and #2 continue to hold when we re-estimate them on the practitioner-ordered PCF corpus."

### D43 — The conclusion claims the model explains AI investment; the model has no investment margin

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>
**Where:** `8_conclusion.tex:14` · **p. 42**.
**Issue.** "Our framework thus helps explain why firms invest so heavily in AI capabilities even when short-run returns appear limited." $\alpha$ is an exogenous parameter of the general-purpose technology that no agent in the model chooses; the firm's entire choice set is $(\mathcal{T},\mathcal{J})$; Definition 3's footnote makes AI free at the margin; and the model is static, so there is no channel through which anticipated future thresholds affect today's spending.
**Fix.** "Our framework thus helps explain why the returns to advances in AI capability can be lumpy — small quality gains buying little until a threshold triggers a reorganization." If the investment reading is wanted it needs an explicit investment margin.

---

# MINOR

Local slips: notation, indexing, domains, and precision. None changes a result, but each is the kind of thing a careful referee marks. Format: **location · page** — issue → fix.

## Section 3 (model)

- 🔴 **N1** · <span style="color:#cf222e">**OPEN**</span> `3_shortrun.tex:100` · **p. 12** — Definition 5 defines an AI strategy as a *partition alone*, but a singleton block satisfies both alternatives: it is "a single manual step" and, by the convention at line 68, also an AI chain of length one. The partition therefore does not determine the block's mode, yet the cost rule in the same definition assigns a different cost to each. → Define a strategy as a partition **together with a mode label for each singleton block**.
- 🔴 **N2** · <span style="color:#cf222e">**OPEN**</span> `3_shortrun.tex:100` · **p. 12** — the parenthetical "an AI chain (automated or augmented)" does not partition anything: by Definition 4 *every* chain has automated steps before an augmented endpoint, so there is no "automated chain". → "an AI chain (a lone augmented step, or automated steps ending in an augmented one)".
- 🔴 **N3** · <span style="color:#cf222e">**OPEN**</span> `3_shortrun.tex:21`, `:60`, `:82`, `:92` · **pp. 10–11** — $t^A_i$ is defined as the time *"prompting **and** verifying the output of one AI attempt"*, but is called a pure verification cost thereafter (Definition 2, Table OA.A.1, Section 4.1's "cost of verifying step $k+1$ on its own"). Line 82 then says the human "prompts the AI for the entire chain", which sits badly against Definition 3's "direct human time cost is zero" and line 92's "entirely beneath their awareness". → Either rename $t^A_i$ the *AI-interaction* cost and say once that prompting a chain costs the same as prompting one step, or fold the prompting time into the chain's fixed cost explicitly.
- 🔴 **N4** · <span style="color:#cf222e">**OPEN**</span> `3_shortrun.tex:22`, `:35` · **p. 10** — neither $d_i$ nor $\alpha$ is given a domain in the body; $\alpha\in(0,1]$ appears only in Table OA.A.1, and $d_i$ only verbally. $q_i=\alpha^{d_i}\in(0,1]$ needs $d_i\ge0$, and the monotonicity claims need $\alpha\in(0,1)$, $d_i>0$ for strictness. → State $\alpha\in(0,1]$ and $d_i\ge0$ (or $d_i>0$, which Appendix OA.B.3 in fact uses — see N19) where they are introduced.
- 🔴 **N5** · <span style="color:#cf222e">**OPEN**</span> `3_shortrun.tex:9` vs `:114` · **pp. 9, 12** — the short-run environment is set up with several workers ("each worker's job covers a fixed block of steps") and then solved for one ("The worker carries out the entire step sequence"). Section 5 (p. 25), Table OA.A.1's note (p. OA - 1) and `OA_B:487` (p. OA - 19) all describe the short run as a *single* job spanning the workflow. → Adopt the single-job description in Section 3, or add the argument that with one worker type at $w=1$ the multi-worker objective has the same argmin.

## Section 4

- 🔴 **N6** · <span style="color:#cf222e">**OPEN**</span> `4_implications.tex:15` · **p. 13** — footnote 4's composite-step device ("a longer chain … can be treated as a single augmented step") is valid for chain costs but not for the standalone price $t^{*}_{k+1}$ that the Proposition 1 proof uses in $V_0$ and $V_3$: the manual alternative to a *block* is not a single manual step. → Add "…so what follows applies unchanged to the chain costs; the standalone price of the block is its own optimal cost."
- 🔴 **N7** · <span style="color:#cf222e">**OPEN**</span> `4_implications.tex:58–61` · **p. 15** — the prose attributes a two-sided claim to part (i) ("being human-advantaged carries no guarantee of being executed one way or the other"), but part (i) is a pure existence statement in one direction. → Say that the converse — parameters at which the step stays manual — is immediate from the pricing, or add it to the proposition.
- 🔴 **N8** · <span style="color:#cf222e">**OPEN**</span> `4_implications.tex:131` · **p. 17** — the Figure 4 notes say "every AI-easy step is flanked by AI-hard ones", but in panel (a) the first step is AI-easy and has no predecessor. The operative claim ("no two AI-able steps are adjacent") is correct. → "no AI-easy step is adjacent to another".
- 🔴 **N9** · <span style="color:#cf222e">**OPEN**</span> `4_implications.tex:220` · **p. 20** — the Figure 5 notes say panel (a) shows "the cost of each AI strategy", but only four of the five configurations are plotted: "Step 1 augmented, Step 2 manual" ($C=3.5\alpha^{-11}+8$) is missing. → Add the curve, or say the panel shows only the strategies relevant to the envelope.
- 🔴 **N10** · <span style="color:#cf222e">**OPEN**</span> `4_implications.tex:220` · **p. 20** — "the table below gives each strategy's cost" — there is no table below Figure 5; the table meant is Table OA.A.3 on p. OA - 3. → Cite it by reference.
- 🔴 **N11** · <span style="color:#cf222e">**OPEN**</span> `4_implications.tex:229` · **p. 20** — "the marginal return jumps up at each threshold, **since a different strategy governs cost on either side**". A different strategy implies at most that the slope may change; it gives neither discontinuity nor direction. → "…because the strategy that takes over was more expensive just below the threshold and cheaper just above, so its cost is falling at least as fast there (Lemma OA.B.4)."

## Section 5

- 🔴 **N12** · <span style="color:#cf222e">**OPEN**</span> `5_longrun.tex:145` · **p. 25** — Eq. (7) as printed equates a minimized value with an unminimized sum, and leaves $\mathcal{J}$ free on the right while it is bound by $\min_{\mathcal{J}}$ on the left. → Split the definition from the minimization: define $\text{TotalCost}(\mathcal{J};\mathcal{T})\equiv\sum_{J_j}\text{WageBill}_j$, then write the double minimum.
- 🔴 **N13** · <span style="color:#cf222e">**OPEN**</span> `5_longrun.tex:151` · **p. 25** — "It nests the short-run problem … as the case of a single job". With one job Eq. (7) evaluates to $(\sum_b c_b)(\sum_b t_b)$, not $\min\sum_b t_b$; the two agree only after the endogenous wage $\sum_b c_b$ is replaced by the fixed normalized wage. That is a change of objective, not a restriction of the feasible set. → "It reduces to the short-run problem when the workflow is a single job **and** the wage is held at its normalized value rather than chosen."
- 🔴 **N14** · <span style="color:#cf222e">**OPEN**</span> `5_longrun.tex:189` · **p. 26** — the three-task example writes $(c_b,t_b,t^H_b)_{b=1,2,3}$, indexing the hand-off by **task**, while `:109`, `:226` and Table OA.A.1 define $t^H_i$ over **steps** (with $t^H_m=0$). → Define $t^H(T_b)\equiv t^H_i$ for the last step $s_i$ of $T_b$ and use that in the example.
- 🔴 **N15** · <span style="color:#cf222e">**OPEN**</span> `5_longrun.tex:171` and Figure 8(b) · **p. 26** — both figures label the pink hand-off rectangle $h_1$ (and $h_2$), a symbol defined nowhere; Figure 7 annotates the horizontal extent of the *same* box $t^H_1$. → Relabel the boxes $t^H_1$, $t^H_2$ in both PNGs.

## Section 6 and the dynamic programs

- 🔴 **N16** · <span style="color:#cf222e">**OPEN**</span> `6_extensions.tex:64` · **p. 30**, same at `OA_B:471` — Eq. (9)'s second branch is written $\min_{\ell<k}$ with no lower limit; $C[\ell]$ is undefined for $\ell<0$, and $\ell=0$ (a chain beginning at step 1) is exactly the case needed. → Write $\min_{0\le\ell<k}$, matching Eq. (10), and define $C[k]$ for $0\le k\le m$.
- 🔴 **N17** · <span style="color:#cf222e">**OPEN**</span> `6_extensions.tex:61` · **p. 30** — the prose says the chain reaches "back to some earlier step $\ell$", while the display's own underbrace says "a chain begun at $\ell+1$". → "…reaching back to some earlier step $\ell+1$; in either case steps 1 through $\ell$ are optimized separately."
- 🔴 **N18** · <span style="color:#cf222e">**OPEN**</span> `6_extensions.tex:64`, `:82` · **pp. 30–31**, and the same recursions in OA.B — the paper fixes $\ell$ as a chain's *first* step and $r$ as its augmented *last* step (Definition 4, Table OA.A.1), but both DP recursions use $\ell$ and $r$ as the *cut* — the last step **not** in the chain. → Rename the cut variable in both recursions, or index by the chain's own first step.
- 🔴 **N19** · <span style="color:#cf222e">**OPEN**</span> `6_extensions.tex:93` · **p. 31**, and `OA_B:553` — "whose … costs … lie in $[1/B,B]$ for some $B>0$" is an empty interval for $B<1$. → "for some $B\ge1$".
- 🔴 **N20** · <span style="color:#cf222e">**OPEN**</span> `OA_B_omitted_proofs.tex:556` · **p. OA - 21** — the tabulated ranges are $[1/B,mB]$ and $[1/B,2mB^3]$, which exclude $0$; but every state the algorithm reads its answer from has a zero coordinate ($V(i)=R(i,0,t^H_i)$, $V(m)=R(m,0,0)$), and $0$ is not a power of $(1+\epsilon)$. → Add a distinguished level $0$ to each coordinate; this adds $O(1)$ levels and leaves the entry count unchanged.
- 🔴 **N21** · <span style="color:#cf222e">**OPEN**</span> `OA_B_omitted_proofs.tex:512` · **p. OA - 20** — the identity $V(i)=R(i,0,t^H_i)$ is asserted "for all $0\le i\le m$", but $t^H_0$ is undefined ($t^H_i$ is introduced only "given step $s_i$"). → Restrict to $1\le i\le m$, or adopt the convention $t^H_0=0$.
- 🔴 **N22** · <span style="color:#cf222e">**OPEN**</span> `OA_B_omitted_proofs.tex:546` · **p. OA - 21** — "$R(r,\,c,\,t)$ with $r>i$" reuses $r$, bound two sentences earlier as the chain cut with range $0\le r<i$. → Use a fresh letter.
- 🟡 **N23** · <span style="color:#9a6700">**PARTIAL**</span> `OA_B_omitted_proofs.tex:569` · **p. OA - 22** — "lets the firm read **the optimal** AI strategy and job design off $V(m)$": the table back-traced is the rounded one, so what is recovered is an *approximately* optimal pair. The proof also never argues that the true cost of the recovered pair is within the factor claimed. → "an approximately optimal pair", plus the missing line: rounding down gives $\text{DPval}(\hat\sigma)\le\text{DPval}(\sigma^*)\le OPT$, hence $\text{truecost}(\hat\sigma)\le(1+\epsilon')^{2m}\,\text{DPval}(\hat\sigma)$. **Status note:** The missing guarantee line is supplied (rounding up bounds the returned pair's true cost); `OA_B:611` still says “read the optimal AI strategy”.

## Appendix OA.B (proofs)

- 🔴 **N24** · <span style="color:#cf222e">**OPEN**</span> `OA_B_omitted_proofs.tex:10`, `:69` · **pp. OA - 5, OA - 8** — the proof works on $(0,1)^3$, but Definition 2 declares $q_i\in(0,1]$ and Proposition 1(ii) quantifies over "every $(q'_{k-1},q'_k,q'_{k+1})$ with $q'_i\ge q_i$", which includes coordinates equal to 1 — exactly the "perfectly reliable neighbour" case the intuition emphasises. → Take $Q=(0,1]^3$; the single-variable claim goes through with the second branch read as $q^{*}<q\le1$.
- 🔴 **N25** · <span style="color:#cf222e">**OPEN**</span> `OA_B_omitted_proofs.tex:33`, `:62` · **p. OA - 7** — part (i) uses $t^M_k>0$ (e.g. $V_0-V_3=t^M_k>0$), but the proposition's hypothesis $t^M_k<t^A_k/q_k$ admits $t^M_k=0$, and the appendix's own Example OA.B.3 sets $t^M_i=0$. → Add $t^M_i>0$, $t^A_i>0$ to the model's primitives, or state the positivity locally.
- 🔴 **N26** · <span style="color:#cf222e">**OPEN**</span> `OA_B_omitted_proofs.tex:95` · **p. OA - 8** — $\mathcal{A}=\mathcal{A}_1\cup\mathcal{A}_2\cup\mathcal{A}_3$ is displayed on all of $Q=(0,1)^3$, but the pricing $V_0,\dots,V_3$ behind it is valid only where the standing hypothesis holds (the paragraph at `:75–78` disposes of the rest separately). → State the identity on the sub-region where $t^M_k<t^A_k/q_k$.
- 🔴 **N27** · <span style="color:#cf222e">**OPEN**</span> `OA_B_omitted_proofs.tex:244` · **p. OA - 13** — "The ratio … is maximized at $\alpha^{-d(T_b)}=1/2$". With $u=\alpha^{d(T_b)}$ the ratio is $1+u-u^2$, maximized at $u=1/2$, i.e. at $\alpha^{-d(T_b)}=\mathbf{2}$. As printed the value is infeasible, since $\alpha\le1$ and $d\ge0$ force $\alpha^{-d}\ge1$. The constant $5/4$ is right. → "maximized at $\alpha^{d(T_b)}=1/2$ (equivalently $\alpha^{-d(T_b)}=2$)".
- 🟢 **N28** · <span style="color:#1a7f37">**RESOLVED**</span> `OA_B_omitted_proofs.tex:228–231`, `:309–311`, `:334` · **pp. OA - 12 to OA - 15** — "task" is used repeatedly where "step" is meant ("connected component of non-failed **tasks**", "each failed **task** $s_i\in T_b$", "the **task** immediately preceding $s_i$"). Task is a defined object in this paper (a block of the partition), and the objects that fail and form components are steps. → Replace throughout.
- 🟢 **N29** · <span style="color:#1a7f37">**RESOLVED**</span> `OA_B_omitted_proofs.tex:382` · **p. OA - 16** — Example OA.B.3's fragmentation index is given as $1+K(1-1/\sqrt2)$. A failed step opens a new component only if it is not the first and its predecessor did not fail, so the correct count is $1+(K-1)(1-1/\sqrt2)$. → Correct both the formula and the sentence "1 plus 1 for each task that fails".
- 🔴 **N30** · <span style="color:#cf222e">**OPEN**</span> `OA_B_omitted_proofs.tex:401` · **p. OA - 17** — "$D_c=\sum_{i\in c}d_i\ge1$" is asserted, not derived; the model imposes only $d_i\ge0$. The proof needs $D_c>0$. → Add $d_i>0$ as a standing assumption in Section 3 and write $D_c>0$.
- 🔴 **N31** · <span style="color:#cf222e">**OPEN**</span> `OA_B_omitted_proofs.tex:407` · **p. OA - 17** — "the optimal strategy changes at finitely many values of $\alpha$" is asserted with no argument, yet it is what makes "regime" well defined. Finitely many strategies does not by itself bound the number of crossings. → One line suffices: each $C_{\mathcal{T}}$ is a polynomial in $1/\alpha$, so pairwise differences have finitely many roots unless identically zero.
- 🔴 **N32** · <span style="color:#cf222e">**OPEN**</span> `OA_B_omitted_proofs.tex:412`, `:454` · **p. OA - 17/18** — $g^*=-\mathrm{d}C^*/\mathrm{d}\alpha$ does not exist at a reorganization threshold, which is precisely where the lemma's second sentence quantifies. → State the lemma in one-sided limits.
- 🔴 **N33** · <span style="color:#cf222e">**OPEN**</span> `OA_B_omitted_proofs.tex:440` · **p. OA - 18** — $\phi$ was just asserted continuous, so $\phi(\alpha_0^{-})=\phi(\alpha_0)=\phi(\alpha_0^{+})=0$ and the displayed sign conditions read $0\ge0$, $0=0$, $0\le0$ — vacuous, and "a differentiable function with such a sign change" then has no sign change to work with. → "there is $\varepsilon>0$ with $\phi\ge0$ on $(\alpha_0-\varepsilon,\alpha_0)$, $\phi(\alpha_0)=0$, and $\phi\le0$ on $(\alpha_0,\alpha_0+\varepsilon)$".
- 🔴 **N34** · <span style="color:#cf222e">**OPEN**</span> `OA_B_omitted_proofs.tex:431` · **p. OA - 18** — the footnote's claim that "the only scenario where no such threshold exists is the case in which all steps are executed manually" is false in the paper's own model: $d_i=0$ gives $q_i\equiv1$, and both Example OA.B.1 and Example OA.B.3 use such steps, for which one strategy can be optimal on all of $(0,1)$ without being all-manual. → "No threshold exists precisely when a single strategy is optimal on all of $(0,1)$, which happens iff every chain used by the optimum has $D_c=0$."
- 🔴 **N35** · <span style="color:#cf222e">**OPEN**</span> `OA_B_omitted_proofs.tex:125` · **p. OA - 9** — the auxiliary claim rebinds $m$, $A$, $B$ and $c$, all of which denote something else in the paper ($m$ = number of steps in the same proof; $A$ = AI mode / aggregate AI labour; $B$ = the cost bound of Proposition 4; $c$ = skill cost). The binding is local and explicit, but $c$ is instantiated as a *time*. → Use fresh letters, or note the local scope.
- 🔴 **N36** · <span style="color:#cf222e">**OPEN**</span> `OA_B_omitted_proofs.tex:10` and throughout — $t^{*}_i$ is used throughout the appendix but appears in no notation table. → Add it to Table OA.A.1.

## Appendix OA.C (CES)

- 🔴 **N37** · <span style="color:#cf222e">**OPEN**</span> `OA_C_CES_representation.tex:26`, `:33`, `:44` · **p. OA - 24** — Eq. (OA.C.1) and the $\tau_b$ display sum $c^M_b$ and $c^A_b$ over **tasks**, but Table OA.A.1 and Section 5 define $c^M_i$, $c^A_i$ at the **step** level; the task-level skill is $c_b$. → Write the sums with $c_b$.
- 🟡 **N38** · <span style="color:#9a6700">**PARTIAL**</span> `OA_C_CES_representation.tex:48`, `:80`, `:111`, `:181` · **pp. OA - 25 to OA - 29** — $d_b$, the total AI difficulty of task $b$, is used four times and defined nowhere; the paper already has two other symbols for the same object, $d(T_b)$ (`OA_B:234`) and $D_c$ (`OA_B:401`). → Define it once and use a single symbol across both appendices. **Status note:** `d_b` is now defined at `OA_C:41`; the symbol is still not unified with `d(T_b)` and `D_c` in Appendix OA.B.
- 🔴 **N39** · <span style="color:#cf222e">**OPEN**</span> `OA_C_CES_representation.tex:167` · **p. OA - 29** — "Normalize the output price to $p=1$" collides with $p$, the number of jobs (Definition 6, Table OA.A.1, and this appendix four pages earlier). → Use $P$, or drop the symbol.
- 🔴 **N40** · <span style="color:#cf222e">**OPEN**</span> `OA_C_CES_representation.tex:188` · **p. OA - 30** vs `OA_B:434` · **p. OA - 18** — $\phi$ denotes the cost gap $C_{\mathcal{T}'}-C_{\mathcal{T}}$ in OA.B and an output density in OA.C, eleven pages apart, both as functions of an AI-quality argument. → Rename one.
- 🔴 **N41** · <span style="color:#cf222e">**OPEN**</span> `OA_C_CES_representation.tex:189` · **p. OA - 30** — "A firm with effective AI quality $\bar\alpha$ thus produces output $y=\phi(\bar\alpha)$ **by definition**" conflates a density (which is how $\phi$ is used in (OA.C.11)–(OA.C.13), integrated $d\bar\alpha$) with a firm's output level; and line 164 calls $\phi$ a "probability density", which would force $Y=1$. → "$\phi(\bar\alpha)\,d\bar\alpha$ is the output produced by firms with effectiveness in $[\bar\alpha,\bar\alpha+d\bar\alpha]$", and drop "probability".
- 🔴 **N42** · <span style="color:#cf222e">**OPEN**</span> `OA_C_CES_representation.tex:85–98` · **p. OA - 26** — Eq. (OA.C.3) is a separability property of the *production function*, but it is inferred from the **equilibrium allocation** displayed just above it. The Leontief/Fisher aggregation theorems invoked at `:95` are conditions on the function. → Derive the separability from the technology (the max-over-splits characterization of the Leontief), or say the condition is verified at the efficient point and that this suffices here because the firm always operates there.
- 🔴 **N43** · <span style="color:#cf222e">**OPEN**</span> `OA_C_CES_representation.tex:252`, `:282` · **p. OA - 31** — $u=(w_A\tau_A)/(1-w_M\tau_M)$ is introduced as a *constant* given wages, then the identity is differentiated with respect to it. That is legitimate only if the identity is required to hold for a continuum of $u$, i.e. of wage vectors, which is never stated. → Add the quantifier: "since Eq. (OA.C.9) must hold at every wage pair, the identity holds for $u$ in an interval, and we may differentiate."

## Sections 1, 2, 7

- 🔴 **N44** · <span style="color:#cf222e">**OPEN**</span> `1_introduction.tex:11` · **p. 2** — "In the absence of AI, tasks **optimally** collapse to single-step blocks". Definition 5 admits only "a single manual step or an AI chain" as blocks, so with no AI the singleton partition is the *only* feasible one; the statement is definitional, not an optimization result. → "…tasks coincide with steps by construction".
- 🔴 **N45** · <span style="color:#cf222e">**OPEN**</span> `1_introduction.tex:45` · **p. 3** — "*Augmented* completion involves a human performing the step with the use of AI" conflicts with Definition 2, in which the AI executes and the human only verifies (the very next sentence gets it right). → "…involves the AI executing the step and a human verifying its output".
- 🔴 **N46** · <span style="color:#cf222e">**OPEN**</span> `1_introduction.tex:75` · **p. 4** — "The resulting worker tasks in this job thus become $1$, $4$, and $5$." Under the model's definition the tasks are $\{1\}$, $\{2,3,4\}$, $\{5\}$; naming them by their endpoints drops Steps 2–3 from the task list and collides with the sequential task indexing $T_1,\dots,T_n$ used from Section 3 on. → "…are therefore $\{1\}$, $\{2,3,4\}$ and $\{5\}$: two manual tasks and one AI-chain task."
- 🟢 **N47** · <span style="color:#1a7f37">**RESOLVED**</span> `1_introduction.tex:91` · **p. 5** — "show analytically that it approximately tracks the **impact** of optimal AI deployment": Proposition 2 bounds the *cost*, and "approximately tracks" is generous for bounds spanning a factor of ten. → "…approximates, up to constant factors, the cost of the firm's optimal AI deployment" (the wording already used in Section 4).
- 🔴 **N48** · <span style="color:#cf222e">**OPEN**</span> `1_introduction.tex:92` · **p. 5** — "jobs with higher fragmentation see a weaker **translation** from AI exposure to AI execution" reads as an interaction (a flatter exposure slope), but Eq. (11) is additive and $\beta_1$ is identical at every fragmentation level. → "…occupations with higher fragmentation realize less AI execution at any given level of exposure."
- 🔴 **N49** · <span style="color:#cf222e">**OPEN**</span> `1_introduction.tex:129` · **p. 6** — "AI execution in the data operates over consecutive steps" sits against Section 7's own "the modest magnitude of 1.45 indicates that long AI chains remain rare" (p. 34). → Add the magnitude, or soften to "AI-executed steps cluster rather than scatter".
- 🔴 **N50** · <span style="color:#cf222e">**OPEN**</span> `2_literature.tex:9`, `:11` · **p. 8** — "the returns to improving any one step … **arrive only once** AI quality crosses a threshold". Within a regime the marginal benefit is strictly positive and merely diminishing; returns do not wait for the threshold. → "…arrive unevenly, jumping when AI quality crosses a threshold that makes longer chains worthwhile".
- 🔴 **N51** · <span style="color:#cf222e">**OPEN**</span> `7_empirics.tex:50` · **p. 34** — "The two placebos isolate different margins": the second placebo *nests* the first (line 47 says it also randomizes positions within each occupation), and the description at line 50 mentions only the reassignment. → "The second placebo nests the first: it randomizes both the ordering within each occupation and which tasks each occupation contains."
- 🔴 **N52** · <span style="color:#cf222e">**OPEN**</span> `7_empirics.tex:141` · **p. 37** — "find the same pattern with **larger coefficients**": only the fragmentation coefficient is larger; the exposure coefficient is uniformly smaller (less than half as large without fixed effects). → "…with a substantially larger fragmentation coefficient (the exposure coefficient is smaller)."
- 🔴 **N53** · <span style="color:#cf222e">**OPEN**</span> `7_empirics.tex:191` · **p. 38** — the displayed Eq. (12) is not the estimated equation in any column: it omits the focal task's exposure status and the occupation's task count, both of which the prose and the table notes say are always included, and the fixed effects of columns (2)–(6). → Define $X_k$ explicitly and put the always-included controls in the display.
- 🔴 **N54** · <span style="color:#cf222e">**OPEN**</span> `7_empirics.tex:215` · **p. 39** — "controlling for the number of same-DWA tasks …, **which rules out** mechanical proximity inflating the estimates". A count control cannot rule that out: the concern is that near-duplicate same-DWA tasks are placed adjacently and carry the same label, which conditioning on the *number* of them does not address. → Weaken to "which absorbs occupation-level differences in same-DWA task counts", and, to address the concern directly, drop focal observations whose neighbours share their DWA.
- 🔴 **N55** · <span style="color:#cf222e">**OPEN**</span> `7_empirics.tex:226` · **p. 39** — "**implying** that tasks two positions away rarely fall in the same chain": a mean does not pin down the tail of the chain-length distribution. → "consistent with".
- 🔴 **N56** · <span style="color:#cf222e">**OPEN**</span> `tables/fragmentation_index_regression_exposure.tex:12` · **p. 36** — the row label "Empirical Fragmentation Index **(Definition 1)**" has no referent in the paper proper; the text names the variants "exposure-based" and "execution-based", and the paper's numbered Definition 1 is "Manual Step" (p. 10). → Relabel the row "Empirical Fragmentation Index (exposure-based)".
- 🔴 **N57** · <span style="color:#cf222e">**OPEN**</span> `preamble.tex:370–378` — `\aggregateManualLabor` $=M$ and `\aggregateAIlabor` $=A$ are the same letters as the mode superscripts `\manualLetter` $=M$ and `\AIletter` $=A$, so in Eq. (8) $M$ and $A$ denote both aggregate labour and execution mode. Likewise $C$ denotes a connected component in Section 4.2 and Appendix OA.B.2 and a cost ($C[k]$, $C_{\mathcal{T}}(\alpha)$, the "Time cost ($C$)" column of Table OA.A.3) everywhere else. → Use $L_M$, $L_A$ for the aggregates, and a distinct letter for components.

---

# What was checked and found correct

Recorded so the authors know where the review did *not* find problems, and so these are not re-derived on the next pass.

- **Proposition 1, part (ii)** — upward closure of "AI executes step $k$" holds. Verified by brute force: 300,000 random parameter draws with three random upward perturbations each, zero violations. The single-variable claim ($D(q)=A/q-\min\{m,B/q\}-c$), its branch analysis at $q^{*}=B/m$, both the $A\ge B$ and $A<B$ cases, and the three applications (closure in $q_{k-1}$, $q_{k+1}$, $q_k$) are all correct.
- **Proposition 1, part (i)** — all three exhibited configurations check out: every stated parameter interval is nonempty under the hypotheses, and each configuration is strictly cost-minimizing there. The four-partition enumeration $V_0,\dots,V_3$ is exhaustive.
- **Proposition 2's bounds themselves** — $\tfrac18 OPT\le FI\le\tfrac54 OPT$, and $\tfrac14 OPT\le FI$ under $t^M_i\ge1$, hold in 6,000 random instances (observed range $FI/OPT\in[0.617,\,1.134]$), including the M6 counterexample. Lemma OA.B.1's upper-bound argument is sound, including the superadditivity fact $(1-xy)(1+\tfrac1{xy})\ge(1-x)(1+\tfrac1x)+(1-y)(1+\tfrac1y)$ (equivalent to superadditivity of $2\sinh$), modulo the maximizer slip N27. Example OA.B.1 is correct: $OPT=2$, $FI=5/2$, ratio exactly $5/4$.
- **Lemma OA.B.4's first claim** — $g^*$ is non-increasing within a regime: $C_{\mathcal{T}}''\ge0$ is right, so the marginal benefit does diminish. The sign of the jump argument ($\phi$ crossing zero downward $\Rightarrow\phi'(\alpha_0)\le0$) is right up to strictness (D18).
- **Section 4.1's numeric example** — $(25,25,0.95)$, $(6,3.5,0.3)$, $(8,4,0.9)$: comparative advantage does assign $k-1$ and $k$ to the human and $k+1$ to AI; the three-step chain costs $4/(0.95\cdot0.3\cdot0.9)=15.594\approx15.6$ against $25+6+4/0.9=35.44\approx35.4$. Eq. (2)'s decomposition is algebraically exact.
- **Figure 3's thresholds** — the horizontal axis stops at $q_k=3.5/6=0.5833$, exactly the break-even for augmenting step $k$ alone; the chain $\{k,k+1\}$ overtakes manual-$k$ at $q_k=4/(0.9\cdot10.444)=0.4255\approx0.43$; and all four regions (including the orange band where $k$ is augmented) occur in the plotted range, in the order the text describes.
- **Example 1's costs** — dispersed arrangement 28, clustered 24.245, matching the two TikZ figures; the mode assignments in both panels are optimal.
- **Table OA.A.3** — all five cost expressions, all five derivatives $-\mathrm{d}C/\mathrm{d}\alpha$, and the optimal ranges are correct; the crossings are at $\alpha=0.500$ and $\alpha\approx0.925$; the two step-1-augmented configurations are indeed never optimal; and the jump at the second threshold is larger than at the first.
- **Section 5's three-task example** — all eight job costs reproduce exactly: $9/16/15/30$ without hand-offs (optimum $[1][2][3]$) and $18.5/18/24/30$ with them (optimum $[1,2][3]$), matching `tables/job_design_example_*.tex`; and $2^{n-1}$ job designs for $n$ tasks.
- **Proposition 3 and its proof** — the short-run recursion is correct and complete, and $O(m^2)$ is right.
- **Proposition 4's state-space count**, taken on its own terms — $[1/B,mB]$ for skill and $[1/B,2mB^3]$ for time, $O(\epsilon^{-1}\log(mB))$ levels each, are correctly derived from the $2mB^2$ cost bound. The error is in the $\epsilon$ that must be fed to it (M3).
- **Appendix OA.C's algebra**, taken from Eq. (OA.C.16) onward — the differentiation of the aggregation identity, the solution for $\Psi$ in terms of $\Gamma$, the substitution, the solution for $\Gamma$, and the final differentiation to $\phi$ are all correct step by step. The problems are the missing boundary condition (M4) and what the identity is verified on (M5).
- **Table 2 and Table 3** — every number quoted in the text matches the table fragments (exposure 0.09–0.24; EFI $-0.26$, $-0.38$, $-0.28$, all at 1 %; immediate-neighbour effects roughly twice the distant ones in column (1)), with the exceptions recorded as D38 and N52.

---

## Reproducing the checks

The counterexamples above were computed with exact rational arithmetic (`fractions.Fraction`) or exhaustive enumeration:

- **Optimal cost** for a step sequence: the Proposition 3 recursion $C[k]=\min\{C[k-1]+t^M_k,\ \min_{0\le\ell<k}(C[\ell]+t^A_k/\prod_{i=\ell+1}^{k}q_i)\}$, cross-checked against brute-force enumeration of all $2^{m-1}$ contiguous partitions with singletons priced at $\min\{t^M_i,t^A_i/q_i\}$.
- **Fragmentation index**: exact expectation by enumeration over all $2^m$ failure realizations, using Eq. (3) verbatim.
- **CES**: `scipy.integrate.quad` on the appendix's own closed-form $\phi$, compared against the appendix's $\Gamma$ and against the aggregation identity.
- **Page numbers**: `latexmk -pdf 0_main.tex` (128 pages, no undefined references), then a source-line-to-printed-page map built from `pdftotext -layout` output and cross-validated against the `\newlabel` entries in `0_main.aux`.
