# Mert's Comments on `0_main.pdf` (received 2026-08-26)

Source: `~/Downloads/0_main_mert_comments.pdf`, 43 iPad annotations (37 highlights, 4 sticky notes, 2 ink marks).
Substantive items: 30. Comments span **pp. 2-30 only** (intro through start of §6). Nothing on §7 empirics or the appendices.

Status key: ` ` = open, `x` = done, `-` = declined/deferred.

---

## §1 Introduction (`1_introduction.tex`)

- [ ] **M1** (p.2, L6) "should be considered as it changes equilibrium predictions..."
  > *This might be too brash given that they will be our referee*
  Soften the claim about changing equilibrium predictions.

- [ ] **M2** (p.2, L12, footnote) "with highly capable AIs the human role shrinks to verifying AI's outputs, consistent with \cite{agrawal2019exploring}"
  > *Do we need this?*
  Candidate cut.

- [ ] **M3** (p.2, L1) "The proliferation of AI tools **is likely to raise** productivity in the near term..."
  > *We should write it as existing evidence suggests*
  Rephrase from prediction to evidence-reporting.

- [ ] **M4** (p.5, L94-98) "Because chaining makes the cost of automating a step depend on the arrangement... exponentially in m... We show in extensions that it can, exactly in the short run..."
  > *I think we should delete this from here. We can think about adding CES and this later in the intro*
  Cut the computability passage from where it sits; consider a later intro slot alongside CES.

- [ ] **M5** (p.6, L110) `\paragraph{Job Design and Aggregate Production.}`
  > *We should probably mention hand-off idea*
  Add the hand-off margin to this paragraph.

- [ ] **M6** (p.7, L137ff) "A natural caveat in this empirical exercise... Appendix F... Appendix G..."
  > *This should be 4-5 lines at most, the reader doesn't need the details*
  Compress the robustness caveat.

## §2 Literature (`2_literature.tex`)

- [ ] **M7** (p.8, L18) McElheran J-curve sentence — highlighted, no note. Likely flagged in connection with M22/M23 (J-curve placement).

- [ ] **M8** (p.8, L11) "...not merely how many there are, which is what the fragmentation index of Section 4.2 captures."
  > *We can delete this I think*

## §3 Short Run (`3_shortrun.tex`)

- [ ] **M9** (p.9, L3) "We hold the set of production steps fixed and abstract from the creation or disappearance of steps" — highlighted, no note.

- [ ] **M10** (p.9, sticky note at section opening) 
  > *Mention everything here. All proofs are in the appendix and we provide the notation in Table XX. Otherwise it breaks the flow*
  Add one upfront pointer at the start of §3 covering both proofs and notation. **Pairs with M13 and M27** (delete the two in-body notation pointers).

- [ ] **M11** (p.10, L45) "In the top layer of Figure 2, Step 1, for example, is executed manually and hence is a manual step."
  > *Mention this after Def 1-2*
  Move the figure pointer so it comes after the definitions rather than interleaved.

- [ ] **M12** (p.10, L37) "$q_i=\alpha^{d_i}$... must be repeated until it succeeds, for a total expected time cost of $t^A_i/q_i$"
  > *Since it must be repeated until it succeeds, the expected time cost of verification across trials are xxx*
  Spell out that $t^A_i/q_i$ is the expected cost **across trials**.
  > *(sticky, same spot)* **Delete "general purpose"** [describing the AI technology]

- [ ] **M13** (p.12, L102) "Table 1 in Appendix D.1 collects the notation... Panel A covers the steps, tasks, and AI deployment..."
  > *Delete*
  See M10.

## §4 Implications (`4_implications.tex`)

- [ ] **M14** (p.14, L27) "since an automated step's output is never verified on its own, step k's own verification cost never enters. The only additional expense is that the chain now also fails whenever AI fails at step k."
  > *Say now the [chain] is more likely to fail because it needs to be successful in two consecutive tasks*

- [ ] **M15** (p.18, L133) "Whether, and how much, a workflow benefits from AI depends not only on how many of its steps..."
  > *Write this paragraph shorter*

- [ ] **M16** (p.19, L168) Example 3 "Consider the same four steps, now clustered rather than interleaved..."
  > *I think these two should be part of a single example as we compare them. Why don't we make these Panel (a) and Panel (b) and have them under a single example*
  **Merge Examples 2 and 3 into one example with two panels.**

- [ ] **M17** (p.19, L189) "To formalize this, return to the firm's AI deployment problem in (1) and consider the special case in which AI verification costs are normalized..."
  > *This paragraph is a bit confusing because we start [with the] normalization all of a sudden and mention this only at the end. Does it make sense to say this earlier?*

### §4.3 Non-monotone gains (heaviest cluster)

- [ ] **M18** (p.21, L223) "Recall that $\alpha$ measures the quality of the general-purpose AI technology..."
  > *Start a new paragraph here* (ink arrow confirms: break before "Recall", joining it to "When AI quality is so low...")

- [ ] **M19** (p.21, L227-231) The whole "Seeing what this implies requires one ingredient the model leaves out... productivity J-curve" paragraph.
  > *We should delete this. It breaks the flow. We can mention J-curve at the end of this section, see my comment below*
  **Note: this is the paragraph I wrote last session at your request.** Mert wants the J-curve moved to the section's close (see M23).

- [ ] **M20** (p.21, L267) "Table OA-4 in Appendix D.1 collects them, together with the cost expression of each..."
  > *Make this a footnote*

- [ ] **M21** (p.21, end of Example 4) "Figure 7 traces the optimal cost and its marginal benefit as AI quality $\alpha$ rises."
  > *Start a new paragraph here*

- [ ] **M22** (p.21, L268) "...which is what the top row of Figure 3 showed at $\alpha=0.9$; above that it chains the two."
  > *Say Panel (a) here and describe the figure a bit more. Mention the lines and say the lower envelope is the minimized cost*

- [ ] **M23** (p.22, L270) "Panel (b) shows that the marginal return to improving AI quality jumps up at each threshold..."
  > *If it doesn't [get] too long this could be in the same paragraph with above*

- [ ] **M24** (p.22, L276) "...so that the firm's optimal cost is the lower envelope of finitely many smooth cost curves,"
  > *Since this [is] mentioned above delete it. This is not really a result*
  (Depends on M22 adding the envelope description to the figure discussion.)

- [ ] **M25** (p.22, L278ff) The entire closing paragraph "The reorganizations these thresholds trigger are, so far, confined to the short run... steam engine, electricity, and the computer..."
  > *Mention J-curve here. Delete this whole paragraph, make a shorter version with J-curve mentioned*
  **This is where M19's material goes.**

## §5 Long Run (`5_longrun.tex`)

- [ ] **M26** (p.23, L5) "Moreover, free entry in the labor market lets the firm pay workers according to the actual skill requirements..."
  > *Say adjustment in the labor market*

- [ ] **M27** (p.23, L42) "we extend Definitions 1-4 to incorporate the **skill cost** of steps as well"
  > *"Skill cost" sounds weird. We should say "skill required". We later say that wage is proportional to skill*
  **Terminology change, likely repo-wide.**

- [ ] **M28** (p.24, L45) "each task $T_b$ ... carries a pair of skill and time costs $(c_b, t_b)$"
  > *Do we say what this is? I'm confused. Is this the time and skill of [the] last step?*
  Define $(c_b,t_b)$ explicitly at first use.

- [ ] **M29** (p.24, L51) "The total skill required to complete the tasks in job $J_j$ is $\sum_{T_b\in J_j} c_b$"
  > *Should we add a footnote here, as the additivity is a strong assumption*

- [ ] **M30** (p.25, L77) "Note what (5) implies for a job. The worker is compensated for every skill the job demands over its whole duration..."
  > *Should we give an example here? Faculty and cleaning [the] chalkboard are good ones*

- [ ] **M31** (p.25, L91) "Because production in our model exhibits an order, we can be more specific than a cost that depends only on how many workers there are."
  > *Can we write this simpler? It is not clear to me why the number of workers [is] emphasized here. I think this sentence: "We therefore attach the hand-off cost to the step at which the break occurs rather than to the worker or to the number of jobs, so that where the firm divides the workflow matters and not merely how often it divides it" makes a really good point but we got there in a convoluted way*

- [ ] **M32** (p.26, L116) "Panel B of Table 1 in Appendix D.1 collects the skill, job, and wage notation this section adds."
  > *Delete this sentence*
  See M10/M13.

- [ ] **M33** (p.26, L137) "We refer to (7) as the firm's long-run optimization problem, as it anticipates the adjustment of worker wages..."
  > *Doesn't it make sense to mention here that the boxes are [a] representation of these costs? I guess we do it below so maybe that is fine, but right now at the equation [it] made a bit more sense*

## §6 Extensions (`6_extensions.tex`)

- [ ] **M34** (p.29, L2) "...our firm-level, sequential model of production **where tasks and jobs are determined endogenously** with models that study..."
  > *Delete* [the highlighted clause]

- [ ] **M35** (p.30, L10) "**What ultimately matters, however,** is how these firm-level adjustments translate into the broader labor market"
  > *Maybe something weaker here. This attacks our model; firm-level stuff still matters*
