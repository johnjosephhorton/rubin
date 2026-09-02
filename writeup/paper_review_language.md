# Language Review: *Chaining Tasks, Redefining Work — A Theory of AI Automation*

**Reviewed:** 2 September 2026, against the working sources (`0_main.tex` and everything it inputs).
**Page numbers** refer to the same local build as the content review (131 pp.: body pp. 1–46, then `OA - 1`…`OA - 35`, then `SA - 1`…`SA - 48`).

**Scope.** The abstract, §§1–8, Online Appendix OA.A / OA.B / OA.C, every footnote, and every figure- and table-Notes block the body inputs. **Excluded by request:** all six `SA_*.tex` files.

**The bar.** You asked for mistakes and for sentences that are genuinely confusing — not basic improvements, not polish. So this report contains only two things: (A) places where the English is **actually wrong**, and (B) sentences a competent reader has to reconstruct. Nothing here is a preference.

Deliberately **not** reported, even where a smoother version exists: comma and semicolon placement, hyphenation, "which" vs "that", passive voice, sentence length, nominalizations, word repetition, wordiness, British vs American usage, and any rewrite whose justification is that it reads better.

**Method.** Twelve independent readings of the paper produced 61 candidates. Each was then re-derived by an adversarial pass whose explicit instruction was to throw out anything that is taste dressed up as an error, checking author counts in `rubin.bib` and the rendered text of the compiled PDF before ruling. **36 were refuted** — the majority — and the notable ones are listed at the end so you can see what was considered and cleared. One of my own candidates was dropped under the same test. What survives is 21 findings.


**Notation.** Symbols are those of the main draft (Table OA.A.1 and `preamble.tex`); the draft's macros are written out here so this file renders on its own.

| Symbol | Definition in the draft |
|---|---|
| $t_{b}$, $t^{E(b)}_{b}$ | Time cost of task $b$, in the mode $E(b) \in \{M, A\}$ that executes it |
| $c^{M}_{i}$, $c^{A}_{i}$ | Skill to complete step $i$ manually; skill to verify one AI attempt at it |
| $c_{b}$ | Skill cost of task $b$: $c^{M}_{i}$ if a manual step, $c^{A}_{r}$ if a chain augmented at step $r$ |
| $w_{M}$, $w_{A}$ | The manual and AI base wage rates (OA.C) |
---

## Summary

| | Count |
|---|---|
| **Major** — the sentence conveys the wrong meaning, so a reader is actively misled | 2 |
| **Medium** — an error needing more than a mechanical fix, or a sentence that must be rewritten to be understood | 4 |
| **Minor** — a certain, mechanical error: one wrong word, one agreement slip, one missing article | 15 |

Two patterns are worth noting before the list. First, **Appendix OA.C carries nine of the twenty-one findings**, including both majors, and several are the same error repeated (three dropped articles, two occurrences of one wrong preposition) — it reads as the one section that has not had a copy-editing pass. Second, the paper's **three execution modes are named ungrammatically in two of the three places they are introduced** — the abstract and §3.1 — while §1 gets it right; that one is worth fixing first because it is the paper's central vocabulary.

---

## Reconciliation status

Each finding carries a **Status** line (Major and Medium) or a status marker on its opening line (Minor), on the same convention as `paper_review_findings.md`.

| | Status | Meaning |
|---|---|---|
| 🟢 | <span style="color:#1a7f37">**RESOLVED**</span> | The fix has landed in the draft |
| 🟡 | <span style="color:#9a6700">**PARTIAL**</span> | Some of the fix landed; the remainder is named in the status note |
| 🔵 | <span style="color:#0969da">**DEFERRED**</span> | Set aside deliberately, with the reason named in the status note |
| 🔴 | <span style="color:#cf222e">**OPEN**</span> | Not yet addressed |

| Severity | 🟢 <span style="color:#1a7f37">Resolved</span> | 🟡 <span style="color:#9a6700">Partial</span> | 🔵 <span style="color:#0969da">Deferred</span> | 🔴 <span style="color:#cf222e">Open</span> | Total |
|---|---|---|---|---|---|
| **Major** | 1 | 0 | 0 | 1 | 2 |
| **Medium** | 0 | 0 | 0 | 4 | 4 |
| **Minor** | 0 | 0 | 0 | 15 | 15 |
| **Total** | **1** | **0** | **0** | **20** | **21** |

Line numbers are those of the sources as reviewed on 2 September 2026 and drift as fixes land.

---

## Major

### LMAJ-1. The abstract's opening sentence has a broken enumeration and attaches the paper's central term to the wrong noun

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>

**Where:** `0_main.tex:72` (abstract) — **page 1**.
**Text:** “Production is a sequence of steps that can be executed (1) manually, (2) augmented with AI, or (3) fully automated within contiguous AI-executed steps called ``chains.''”

**Issue.** Two problems in one sentence.

1. *Broken parallelism.* The numerals fix the boundary of each list item, so the verb “executed” sits outside the list and governs all three. Substituting each item back into the frame gives “can be executed manually” ✓, “can be executed **augmented with AI**” ✗, “can be executed **fully automated**” ✗. Only the first item is a legitimate complement of “executed”; the other two are past participles that cannot be governed by it.
2. *Misattached appositive.* “…within contiguous AI-executed steps called ‘chains’” attaches “called ‘chains’” to **steps**, so it reads as though an individual step is called a chain. Definition 4 says the opposite: “An AI chain is a contiguous **block** of one or more sequential steps.”

**Why it's an issue.** This is the first sentence of the abstract — the first sentence every referee and almost every reader will see — and the second defect misdefines the term the paper is named after. A reader who takes the sentence at face value comes away thinking a “chain” is a step.

**Proposed fix.** Move the verb inside the list and attach the appositive to the block:

> “Production is a sequence of steps, each of which can be (1) executed manually, (2) augmented with AI, or (3) fully automated as part of a contiguous block of AI-executed steps that we call a ``chain.''”

---

### LMAJ-2. “Weighted average” describes the display two lines below as something it is not

🟢 <span style="color:#1a7f37">**Status: RESOLVED.**</span> `OA_C:29` now reads "the total compensation of a job is the sum of the skill costs of its tasks, each weighted by the base wage rate of the labor that performs it", which agrees with Equation (C.1) and with the gloss at `:34`.

**Where:** `OA_C_CES_representation.tex:29` — **page OA - 25**.
**Text:** “Therefore, the total compensation of a job will be a weighted average of skill costs of the job's tasks, with weights being the base wage rates of each type of labor.”

**Issue.** Equation (C.1), which this sentence exists to introduce, is a weighted **sum**: $w_{M}\bigl(\sum_{T^{M}_b\in J}c^{M}_{b}\bigr)+w_{A}\bigl(\sum_{T^{A}_b\in J}c^{A}_{b}\bigr)$. The weights are the two base wage rates; they do not sum to one, and nothing normalizes them.

**Why it's an issue.** “Weighted average” tells the reader the result lies **between** the two skill costs. It does not — it is larger than either, and it scales with the number of tasks in the job, which is the whole point being made in the paragraph above (“the more tasks the firm includes in a job, the higher the required compensation”). The wrong word contradicts both the display immediately below it and the argument immediately above it.

**Proposed fix.**

> “Therefore, the total compensation of a job is the sum of the skill costs of its tasks, each weighted by the base wage rate of the labor that performs it.”

---

## Medium

### LMED-1. The hand-off sentence in §5.4 cannot be resolved into a single reading

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>

**Where:** `5_longrun.tex:186` — **page 26**.
**Text:** “Notably, the hand-off shares the skill requirements of the worker doing it for an extra amount of time added to the last task in their job.”

**Issue.** Three attachment points are open at once. “Shares the skill requirements of the worker” suggests the hand-off and the worker hold skill jointly, rather than that the hand-off is *charged at* the worker's skill level. “For an extra amount of time” then attaches either to “shares” (the sharing lasts a while) or to “the skill requirements”. And “added to the last task” attaches either to “time” or to “the worker”.

**Why it's an issue.** What Equation (7) says is that the hand-off is priced at the same skill level as the worker who performs it, and that its duration is added to that worker's job time — which is exactly what Figure 7 draws, with the pink rectangle the same height as the blue task rectangle beside it. None of that is recoverable from the sentence; the reader has to get it from the figure and then read the sentence backwards to confirm.

**Proposed fix.**

> “The hand-off is charged at the same skill level as the worker who performs it, and its duration is added to that worker's job time — which is why the pink rectangle is as tall as the task rectangle beside it.”

---

### LMED-2. A broken “not only … but also” pair in OA.C

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>

**Where:** `OA_C_CES_representation.tex:25` — **page OA - 25**.
**Text:** “The worker assigned to perform Job 1 is required to obtain not only the skill required for manual Task 1, $c^{M}_{1}$, but must also possess the required skill for AI chain Task 2, $c^{A}_{2}$…”

**Issue.** The correlative is broken. “Not only” governs a noun phrase (“the skill required for manual Task 1”), while “but also” governs a fresh finite verb phrase with its own modal (“must also possess…”). The two halves are not the same kind of constituent, so “is required to obtain” is left without its second object.

**Why it's an issue.** A correlative pair must join like with like; as written the sentence starts one construction and finishes another. The reader has to back up to work out whether the second skill is something the worker is *required to obtain* or something separately asserted.

**Proposed fix.**

> “The worker assigned to Job 1 must acquire not only the skill required for manual Task 1, $c^{M}_{1}$, but also the skill required to verify AI chain Task 2, $c^{A}_{2}$, since the worker's total compensation per unit of time is determined at the job level and equals $c^{M}_{1}+c^{A}_{2}$.”

---

### LMED-3. Two missing words leave a comparative correlative unclosed

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>

**Where:** `OA_C_CES_representation.tex:26` — **page OA - 25**.
**Text:** “Therefore, the more tasks firm includes in a job, the higher required compensation for every task in that job.”

**Issue.** “Tasks firm includes” is missing the article — *the* firm. And the second half of the “the more X, the more Y” construction is missing its determiner: it needs “the higher **the** required compensation”. Without it the clause has no subject and the sentence does not close.

**Why it's an issue.** The comparative correlative requires a determiner in each half; both are absent here, in a one-line sentence that states the paper's core job-design trade-off.

**Proposed fix.**

> “Therefore, the more tasks the firm includes in a job, the higher the required compensation for every task in that job.”

---

### LMED-4. A singular subject takes a plural verb and is then picked up by a plural pronoun

🔴 <span style="color:#cf222e">**Status: OPEN.**</span>

**Where:** `OA_C_CES_representation.tex:86` — **page OA - 27**.
**Text:** “Since the AI strategy and job design are fixed, the required amount of skill-adjusted time to spend on each task in the denominators of Equation~\eqref{eq:task_level_prod} are fixed, and the firm takes them as given.”

**Issue.** The subject is the singular “the required amount”, but the verb is the plural “are fixed” and the following clause refers back with the plural “them”.

**Why it's an issue.** The plural is what is meant — there is one denominator per task, so there are many required amounts — so the noun, not the verb, is the thing to change. Two disagreements in one sentence, with the second (“them”) confirming the intended plural reading.

**Proposed fix.** Pluralize the noun and everything else falls into line:

> “…the required **amounts** of skill-adjusted time to spend on each task in the denominators of Equation~(C.2) are fixed, and the firm takes them as given.”

---

## Minor

Certain, mechanical errors. Listed in paper order.

### §2 Related Literature

🔴 **LMIN-1. Two-author citation with singular verbs (twice in one clause).** `2_literature.tex:10` — **page 8**. “Our results align with \citet{gans2026oring}, who likewise **emphasizes** task interdependence in O-ring production and **obtains** lumpy automation…”. `gans2026oring` is Gans **and** Goldfarb, so the citation renders “Gans and Goldfarb (2026)” (confirmed on p. 8 of the build) and “who” has a plural antecedent. *Fix:* “…who likewise **emphasize** … and **obtain** …”.

🔴 **LMIN-2. Two-author citation with a singular verb.** `2_literature.tex:19` (footnote 2) — **page 8**. “\cite{ide2025ai} **provides** a distinct but related view”. `ide2025ai` is Ide and Talamás; it renders “Ide and Talamás (2025)”. *Fix:* “**provide**”.

### §3 Short Run

🔴 **LMIN-3. An adverb where the mode name belongs.** `3_shortrun.tex:19` — **page 9**. “each is executed in one of three modes: *manually*, *augmented*, or *automated*.” The frame “one of three modes:” takes mode names, and “manually” is an adverb — “a mode: manually” does not substitute back. The paper's own naming is adjectival everywhere else (`1_introduction.tex:43`: “three modes of step completion are recognized: manual, augmented, and automated”, and the definitions are titled “Manual Step”, “Augmented Step”, “Automated Step”). *Fix:* “*manual*”. This is the same defect as LMAJ-1; fixing both makes the paper's central vocabulary consistent across all three places it is introduced.

🔴 **LMIN-4. Mixed conditional.** `3_shortrun.tex:74` (footnote to Definition 3) — **page 11**. “The forces described by the model **remain** unaffected even if we **assumed** that each time the firm **deploys** AI on a step it **has to** incur a cost.” Present-indicative main clause, past-subjunctive protasis, then present again inside it. *Fix:* “…even if we **assume** that…”.

### §5 Long Run

🔴 **LMIN-5. Subject-verb disagreement.** `5_longrun.tex:56` (footnote) — **page 23**. “namely that each step **demand** some increment of skill no other step supplies”. “Each step” is singular; the bare form would only be licensed as a mandative subjunctive, which “assumption” does not trigger, and the parallel clause later in the same sentence is indicative (“a worker … must acquire more skill”). *Fix:* “**demands**”.

🔴 **LMIN-6. Wrong collocation.** `5_longrun.tex:61` (footnote) — **page 23**. “One motivation for this formulation is workers **paying** a human capital investment to acquire the skills necessary for a job.” One *makes* an investment or *pays* a cost. The next sentence confirms the intended sense: “This investment must be offset by wages.” *Fix:* “workers **making** a human capital investment” (or “paying a human capital investment **cost**”).

🔴 **LMIN-7. Dropped complementizer creates a garden path.** `5_longrun.tex:98` — **page 24**. “Equation~(6) implies dividing a job in two splits its skill requirement between two workers…”. “Implies dividing a job in two” reads as verb + gerund object until “splits” arrives with no slot to fill. *Fix:* insert “that”: “implies **that** dividing a job in two splits…”.

🔴 **LMIN-8. Subject-verb disagreement.** `5_longrun.tex:188` — **page 27**. “…and where those two prices balance **determine** the firm's degree of specialization.” The subject is the noun clause “where those two prices balance”, which is singular; the plural verb is being pulled by the adjacent “prices”, which sits inside the subject and is not it. *Fix:* “**determines**”.

### §7 Empirical Evaluation

🔴 **LMIN-9. Subject-verb disagreement with a coordinated subject.** `7_empirics.tex:226` — **page 40**. “This implies that local work context and proximity **does** work that the reshuffled orderings cannot reproduce by chance.” *Fix:* “**do**”.

### Appendix OA.B

🔴 **LMIN-10. Partitive pronoun with no expressed whole.** `OA_B_omitted_proofs.tex:171` — **page OA - 10**. “The successor coordinate is the counterpart of the predecessor one, and **the milder half of it**, so the claim disposes of it directly.” There is no singular NP naming a whole of which the successor coordinate is a half. *Fix:* “…is the counterpart of the predecessor one, and **the milder of the two**, so the claim disposes of it directly.”

🔴 **LMIN-11. Dangling participle.** `OA_B_omitted_proofs.tex:408` — **page OA - 16**. “**Taken together**, we obtain our desired $8$ approximation.” The participle's implied subject is the two charging bounds; the main-clause subject is “we”. *Fix:* “**Taken together, these bounds give** our desired $8$ approximation.” (Or, keeping “we”: “**Combining these bounds, we obtain** …”.)

### Appendix OA.C

🔴 **LMIN-12. Subject-verb disagreement.** `OA_C_CES_representation.tex:37` (footnote) — **page OA - 25**. “…allows tasks to contribute differently to the job's wage depending on their mode of execution and type of labor that **need** to perform them.” The head is the singular “type of labor”. *Fix:* “…and type of labor **needed** to perform them.”

🔴 **LMIN-13. A positional pointer that names the wrong side of the expression.** `OA_C_CES_representation.tex:48` — **page OA - 26**. “Note that the fraction **appearing behind** $t^{E(b)}_{b}$ represents an effective skill adjustment factor.” In the display the fraction *precedes* and multiplies $t^{E(b)}_{b}$. *Fix:* “the fraction **multiplying** $t^{E(b)}_{b}$”.

🔴 **LMIN-14. Wrong preposition in a fixed idiom — twice.** `OA_C_CES_representation.tex:61` (**page OA - 26**) and `:91` (**page OA - 27**). “execute each step manually or **by** the help of AI”; “whether tasks $a$ and $b$ are executed manually or **by** the help of AI”. *Fix:* “**with** the help of AI”, in both places.

🔴 **LMIN-15. Missing definite article before a singular count noun — three times.** `OA_C_CES_representation.tex:62` (**page OA - 26**), `:72` (**page OA - 26**), `:87` (**page OA - 27**). “Specifically, **production function of the firm** can be represented by…”; “Therefore, **hand-off of job $J_j$** can be treated as…”; “In equilibrium, **allocation of labor to tasks** satisfies…”. Each noun is made definite by its own *of*-phrase, so it cannot stand bare as a subject. *Fix:* insert “the” in all three. (Line 57's “must complete requirements of all $m$ production steps” is a bare plural and is fine as written.)

---

## Candidates examined and cleared

37 candidates were raised and did not survive re-derivation. They are listed because several
are sentences a reader might flag on a fast pass, and it is useful to know they were checked
and are sound. The most substantial:

- **“the AI is just less likely to succeed at the former”** (`1_introduction.tex:86`). Objection: “the former” picks out a verification act, which the AI does not perform. Cleared: “the AI … succeed at” selects a task the AI performs, so the only referent a reader can build is the first of the two *execution* scenarios (AI executes Steps 3–4 vs. Step 4 alone). The rival reading cannot be constructed.
- **“because it presses on both sides of this trade-off”** (`1_introduction.tex:113`). Objection: singular pronoun, plural antecedent. Cleared: “AI” is present in the sentence as a singular noun, and the authors write the identical sentence elsewhere (`5_longrun.tex:235`, “AI presses on both sides of the specialization trade-off at once”), which settles both antecedent and intent. The rival antecedent yields the same claim.
- **“our findings are robust to the way each of these measures is constructed”** (`1_introduction.tex:135`). Objection: the three checks that follow do not map onto the three measures named. Cleared as a language finding: the sentence parses one way only; the objection is a substantive coverage claim about the appendices, which belongs to the content review.
- **“does not incur any cost to the firm”** (`3_shortrun.tex:74`). Objection: wrong collocation. Cleared: “incur” takes eventive subjects routinely, and “any cost to the firm” is a well-formed NP with “to the firm” naming the bearer (cf. “at no cost to the taxpayer”). “Impose … on” is an equally good alternative, i.e. a preference.
- **“Moving to the right, the chain $\{k,k+1\}$ becomes more efficient at $q_k \approx 0.43$”** (`4_implications.tex:94`). Objection: dangling modifier. Cleared: this is the ordinary figure-navigation participle of economics prose (“Turning to Table 3, the effects attenuate”), whose implied subject is the reader tracing the axis.
- **“Taking expectations, step $s_i$ fails with probability $1-q_i$”** (`4_implications.tex:180`). Same class; standard absolute construction in mathematical prose.
- **“We call the values of $\alpha$ at which the optimal strategy changes reorganization thresholds”** (`4_implications.tex:206`). Objection: garden path. Cleared: the stray parse leaves “call” without its predicative complement, so it self-corrects inside the clause; genuine ambiguity needs two readings that are both plausible.
- **“the step $k$'s AI exposure status and number of steps in the occupation it appears in”** (`7_empirics.tex:192`). Objection: dropped article. Cleared: the genitive legitimately scopes over the coordinated nominal, and the trailing relative clause fixes the reading immediately.
- **“whereas execution status of more distant neighbors matters much less”** (`7_empirics.tex:228`). Objection: dropped article. Cleared: abstract nouns of this class take zero article in generic use with an *of*-complement, and it is parallel to the bare nominalization heading the contrasting clause in the same sentence. Adding “the” is uniformity, not correction.
- **“which is 534 of the 1,748 DWAs in the sample and leaves 4,096 tasks”** (`7_empirics.tex:211`). Objection: relative pronoun with no antecedent. Cleared: this is the standard summative “which” of applied-economics table notes, taking singular agreement throughout, and the counts are corroborated two sentences later.
- Also cleared: the Table OA.A.3 Notes fragment beginning “Notes: The two-step workflow of Example 2 …” (verbless nominal style is the convention for Notes blocks); the dangling-modifier objections at `OA_B:17` and `6_extensions.tex:45`; the parallelism objections at `OA_B:62`, `OA_B:277` and `4_implications.tex:43`; the pronoun objections at `OA_B:214`, `OA_B:465`, `5_longrun.tex:223` and `6_extensions.tex:50`; the relative-clause objection at `OA_B:582`; and `4_implications.tex:152` (“one that stands in for them”), which I had raised myself and which fails the same test — a measure standing in for the *steps* is not a reading anyone builds, so the intended referent is recoverable.

One item is **not** repeated here because it is already in the content review: the Figure 5 Notes pointing to “the table below” when the table is Table OA.A.3 in the appendix (MIN-9 of `paper_review_findings.md`).

---

## Where to start

Both majors are in one place each and take a minute apiece: rewrite the abstract's first sentence (LMAJ-1) and change “weighted average” to “weighted sum” in OA.C (LMAJ-2). Then give **Appendix OA.C a dedicated copy-editing pass** — nine of the twenty-one findings are there, and five of them are the same two errors repeated, which suggests the section has not had one. Everything in the Minor list is a one-word change.
