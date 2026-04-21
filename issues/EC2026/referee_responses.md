# Chaining Tasks, Redefining Work — EC'26 Revision Response Tracker

*Running document mapping every EC'26 referee comment to our response status.*

*"Chaining Tasks, Redefining Work: A Theory of AI Automation"*
Demirer, Horton, Immorlica, Lucier & Shahidi — paper #2016

Under each referee comment is a response callout with three fields:

- **What we did** — plain-English description of the response, suitable for any reader.
- **Where in the paper** — the section or paragraph where the fix lives (or will live).
- **Internal tracking** — links to the detailed issue file under `issues/EC2026/`, commit hashes when the change lands, and open questions for coauthors.

**Status legend:**

- ✅ addressed / verified in the paper
- 🟡 response drafted / partially addressed — paper update still pending
- 🔴 open — no response drafted yet
- 🟣 declined per referee's own suggestion

---

## Review #2016A

### Paper summary

This paper develops a model of how jobs are done to study how AI affects production and how firms optimally choose jobs and AI usage to perform them.

Specifically, a firm's production process requires performing a sequence of steps. The firm can partition contiguous sequences of steps into tasks. Furthermore, the firm can partition contiguous task sequences into jobs.

A task can take one of three forms: a step by a human, a step with AI and human (AI-assisted or augmented), or an AI Chain. More precisely, a single step can be performed in three modes: either by a human, AI-assisted (augmented), or automated. The first two require a human with a cost per time and a completion time, and the AI-assisted (augmented) choice also has a probability of success. The authors model this by scaling up the human time (multiplied by 1 over the AI's success probability). The automated step does not require any human skill or time cost.

Here is an important definition: the steps can be chained into an AI chain, which means a series of steps all automated except the last one, which is augmented, and the firm has only human skill cost for the last augmented step and the time cost of the human is its time cost divided by the product of the success probability of all automated (and also the last augmented) steps.

(The success probability is also modeled to be decreasing in the difficulty of a step.)

An AI strategy is a partitioning of a sequence of steps into different tasks (either performed by AI, humans, or augmented humans).

A job is a contiguous sequence of tasks. Finally, wages are the total skills needed to perform a job multiplied by the time required to finish that job (including a hand-off time).

The firm faces an optimization problem: partition a sequence of steps into tasks, and then tasks into jobs. This will determine the overall wages the firm needs to pay. The authors refer to this as the long-term optimization problem. Solving this optimization problem requires solving a set of smaller problems for each job: the short-term optimization problem, which decides how to partition a sequence of steps for a given worker.

The authors, in Proposition 1, develop a dynamic programming approach to solve the short-term problem in time $O(m^2)$, where $m$ is the number of steps. For the long-term problem that jointly optimizes jobs and AI deployment, the authors again propose a dynamic programming approach to obtain an approximately optimal solution with polynomial running time.

The authors then define a fragmentation index that essentially captures the gains from reordering AI steps, taking into account whether each step is successful. In Proposition 3, the authors show that the fragmentation index approximates the true optimal cost within constant factors.

Finally, the authors complement their theory with empirical findings.

### Evaluation

The paper studies a very interesting, important, and timely topic. The model could benefit from stronger justification for some of its assumptions (please see my questions below).

### Feedback for Authors

- The paper benefits from discussing why only the steps performed by humans incur a cost, since AI use can be expensive for firms as well.

> 🟡 **RESPONSE DRAFTED — paper update pending**
>
> **What we did:** We assume the marginal cost of deploying AI is negligible relative to the cost of human time. One concrete microfoundation: the firm has paid a subscription fee to access AI, and from there the marginal cost of using the technology is effectively zero. We will state this assumption explicitly in the model setup and note that extending to a positive marginal AI cost is a natural follow-on.
>
> **Where in the paper:** Model setup (Section 3 area of `writeup/model.tex`), with a sentence in the introduction signaling the assumption.
>
> **Internal tracking:** Full analysis in [01-ai-usage-cost.md](01-ai-usage-cost.md). Not yet committed to the paper.

- The paper also benefits from a discussion of the worker side of the market. For instance, what happens if workers are strategic and charge based on the marginal value of their work to the firm, or if there is a shortage or abundance of skilled workers?

> 🟡 **RESPONSE DRAFTED — paper update pending**
>
> **What we did:** We assume workers do not have bargaining power and that workflows and operations are designed entirely by the firm. The firm decides *how* work is conducted and hires workers to execute those tasks — there is no bargaining in the short run. Workers are indeed paid their marginal value of work: this is the content of the wage formulation, where wages are the per-unit-time flow value of a worker's marginal product and the skill requirements of the job pin down that flow value. We are implicitly assuming (i) the supply of workers is inelastic, and (ii) there is a representative worker employed at the firm. Extending to an elastic labor supply with strategic wage-setting or to a GE with heterogeneous workers and scarcity/abundance of skilled labor is a natural next step.
>
> **Where in the paper:** Model setup; wage-equation paragraph in Section 3 / Section 6 (`writeup/model.tex`, `writeup/aggregation.tex`).
>
> **Internal tracking:** Full analysis in [02-worker-market-conditions.md](02-worker-market-conditions.md). Not yet committed to the paper.

- Does the firm pay the human worker while she is waiting for the AI tasks to be completed? If yes, the paper benefits from justifying this.

> 🟡 **RESPONSE DRAFTED — paper update pending**
>
> **What we did:** We propose a structural modification to the model that jointly addresses this comment and R#2016D's main critique on verification-cost scaling. The per-step AI-management cost becomes chain-length-dependent:
>
> ```
> t^{AI, new}_i  =  t^{AI, old}_i  +  (n − 1) · t^{Chain}
> ```
>
> where `t^{AI, old}_i` is the current step-specific AI-management cost (characteristic of step `i`), `t^{Chain}` is a constant per-additional-chained-step overhead, and `n` is the length of the chain ending at step `i`. The `(n − 1) · t^{Chain}` term captures the human time associated with overseeing / engaging with a longer AI chain — which can be read two ways: (a) the human's engaged time while the AI executes the chain's intermediate steps (the wage-during-wait interpretation that addresses this comment), and (b) the additional per-step verification effort required for longer AI outputs (the interpretation that addresses R#2016D). The firm does pay for this time in the wage equation, and it now scales explicitly with chain length.
>
> The earlier draft under this comment — that AI execution time is second-order relative to human time — remains true as a special case (`t^{Chain} = 0`). The structural modification generalises it.
>
> **Where in the paper:** Model setup (Section 3 area of `writeup/model.tex`), adjacent to the chain definition. Full specification of the modification and its implications for the propositions and the fragmentation-index bound is drafted in [10-verification-cost-chain-length.md](10-verification-cost-chain-length.md).
>
> **Internal tracking:** Full analysis in [03-wage-during-ai-wait.md](03-wage-during-ai-wait.md), with the structural modification specified in [10-verification-cost-chain-length.md](10-verification-cost-chain-length.md). Not yet committed to the paper. **Cross-reference**: this single modification jointly addresses the R#2016D verification-cost-scaling critique below.

- The paper also benefits from clarifying the main takeaway from Section 5: Does this say one cannot benefit much from knowing the successes and reordering the steps?

> 🟡 **RESPONSE DRAFTED — paper update pending**
>
> **What we did:** We assume workflows are fixed and reorganizing the order of work is not allowed. What Section 5 establishes, under that assumption, is that given a workflow, occupations whose AI-able steps are less fragmented stand to gain more from AI. The high-level takeaway is that **individual exposure to AI is not the only determinant of AI's gains**. Local complementarities — whether a step is adjacent to other AI-friendly steps — also determine how much an occupation benefits from AI. The fragmentation index is the observable summary statistic that captures this adjacency structure. We will add a framing paragraph at the start of Section 5 and a closing takeaway that makes these points explicit.
>
> **Where in the paper:** Section 5 of the theory (`writeup/discussion.tex` / `writeup/appendix-FI.tex` framing).
>
> **Internal tracking:** Full analysis in [09-fragmentation-index-role.md](09-fragmentation-index-role.md). Not yet committed to the paper. This response also partially addresses R#2016C's related concern about the fragmentation index's role.

---

## Review #2016B

### Paper summary

This paper develops a task-based production model to analyze how AI automation reshapes the division of labor within firms. The authors model production as an ordered sequence of exogenously-specified steps that firms aggregate into tasks (contiguous blocks of steps) and then partition into jobs assigned to workers. Each step can be executed manually, augmented (AI-assisted with human verification), or automated (fully delegated to AI in contiguous "AI chains"). The framework's core theoretical insight is that AI deployment generates local complementarities: when AI can effectively execute a step, it becomes more valuable to automate adjacent steps in the workflow, creating the claimed "chaining" effects that fundamentally alter optimal job boundaries.

Results show optimal AI deployment strategies in both short-run (fixed wages and job design) and long-run (joint optimization over job design, worker skill requirements, and AI strategy) settings. They introduce a fragmentation index that measures the dispersion of AI-exposed steps within a job's production sequence. Jobs with highly fragmented AI-exposed tasks yield lower returns to AI automation in the short run, as the gains from chaining are foregone when automatable steps are dispersed across workers or separated by non-automatable tasks. Empirical estimations link O\*NET occupational tasks to AI execution modes using Anthropic's Economic Index and GPT-generated workflow orderings.

I like this paper for the novel framework (in an effort to set microfoundations…) for why AI adoption may follow a productivity J-curve—initially yielding modest gains until workflow reorganization becomes optimal, at which point returns increase sharply. The framework also reconciles seemingly contradictory empirical patterns: AI can simultaneously drive worker upskilling (when automation reduces low-skill task requirements) and deskilling (when AI-augmented steps require less expertise), with outcomes depending on how tasks aggregate into jobs. Finally, by explicitly modeling task sequencing and hand-off costs, the paper offers a formal treatment of how AI reshapes firm boundaries and occupational structure.

### Evaluation

This paper makes important theoretical and empirical contributions to our understanding of AI-driven labor market restructuring. The central insight claims that AI deployment creates local complementarities through step chaining for analyzing how automation technologies reshape the occupational division of labor rather than simply substituting for tasks in isolation.

The task-based framework with explicit sequencing is innovative, and the fragmentation index provides a useful conceptual tool for understanding why AI's impact varies across occupations. My central reservation concerns the framework's long-run relevance. The model is well-suited to analyze AI deployment within existing occupational structures, but I suspect the more consequential effects operate through wholesale workflow redesign and firm boundary shifts that the exogenous task sequence assumption precludes.

As AI capabilities approach human-level reliability on cognitive tasks, the binding constraint shifts from "which tasks to automate" to "what fundamentally irreplaceable human capabilities justify continued employment." The current model provides limited guidance on this question, as well as endogenous workflow design.

I couldn't help but think from the perspective of technology transfer and modular systems design (Baldwin & Clark, 2000). The model's treatment of task sequencing and interface costs (hand-offs) offers a natural parallel to product architecture: just as modular decomposition enables parallel development while creating integration costs, task decomposition enables specialization while creating coordination costs. The optimal AI strategy trades off these forces, and the fragmentation index elegantly captures when these costs dominate.

However, I have several substantive concerns about the framework's assumptions, particularly regarding its implications for the future organization of work (as alluded earlier) and the boundaries of human expertise in an AI-saturated economy. My comments are separated into major and minor, listed below.

### MAJOR COMMENTS

**1. The Primacy of Exogenous Task Sequences and the Neglect of Endogenous Workflow Redesign.** The model assumes production follows a fixed, exogenously specified sequence of steps. While analytically tractable, this severely limits the framework's ability to capture how firms actively redesign workflows in response to AI capabilities, arguably the most consequential margin of adjustment. Real-world AI deployment frequently involves workflow reengineering: reordering tasks to enable earlier validation, parallelizing previously sequential activities, or restructuring production to exploit AI's comparative advantage in specific cognitive operations.

The empirical reliance on GPT-generated task orderings (Appendix H) partially addresses this by producing "typical" workflows, but the validation is limited. The authors show task orderings are robust to alternative prompts (Appendix I, Kendall's $\tau \approx 0.6$), but this measures consistency across prompts, not accuracy relative to actual production processes. Given that optimal AI strategies depend critically on which steps are adjacent (Proposition 3, Table 2), measurement error in task ordering could substantially bias the fragmentation index and weaken tests of local complementarities.

As a piece of recommendation, the authors might provide additional validation of GPT-generated sequences against direct observation where possible (e.g., detailed work studies, time-motion data, or interviews with practitioners in selected occupations). More fundamentally, the long-run model (Section 6.1) should be extended to allow firms to choose task orderings jointly with AI deployment and job design. Even a simplified version, say, allowing permutations of a subset of tasks, might be helpful.

> 🔴 **OPEN — no response drafted yet**
>
> **What we did:** Not yet addressed.
>
> **Where in the paper:** TBD.
>
> **Internal tracking:** Full analysis and options in [04-endogenous-workflow-redesign.md](04-endogenous-workflow-redesign.md).

**2. Firm Boundaries, Vertical Integration, and the "Make-or-Buy" Decision for AI Services.** One of the paper's most intriguing implications but largely underdeveloped concerns how AI reshapes firm boundaries. The model implicitly treats AI deployment as an internal capability choice, but in practice, firms increasingly access AI through markets (API calls, SaaS platforms, outsourced AI services). The hand-off cost $t_i^H$ in the model captures coordination frictions within jobs, but there is no analogous friction for task handoffs across firm boundaries.

Unlike previous automation waves (robotics, ERP systems) that required firm-specific customization and capital investment, modern AI increasingly operates as a platform technology with non-negligible inference costs. AI deployment, thus, may favor vertical disintegration: firms could specialize in supervising/validating AI outputs (augmented steps) while outsourcing the AI execution itself.

The model's job design problem (Equation 2) highlights that bundling tasks into jobs trades off skill-based wage premiums ($\sum c_b$) against hand-off costs ($t^H$). But AI fundamentally alters this tradeoff in ways the current framework cannot capture. If AI chains can be outsourced with low transaction costs, the optimal organizational structure may shift from "integrated jobs with long AI chains" to "specialized evaluation roles coordinating multiple external AI providers." This parallels the modularity literature's prediction that standardized interfaces enable vertical disintegration.

The authors might benefit from explicitly modeling the make-or-buy decision for AI execution. Formally, allow the firm to choose whether AI chains are executed internally (incurring fixed AI management costs but zero marginal hand-offs) or externally (incurring per-transaction costs but enabling finer task decomposition).

> 🔴 **OPEN — no response drafted yet**
>
> **What we did:** Not yet addressed.
>
> **Where in the paper:** TBD.
>
> **Internal tracking:** Full analysis and options in [05-firm-boundaries-make-or-buy-ai.md](05-firm-boundaries-make-or-buy-ai.md). Notably this reviewer explicitly invokes Baldwin & Clark (2000), which also connects to the strategy/innovation references suggested by Ezra Zuckerman in the QJE round (see `../QJE/26-strategy-innovation-literature.md`).

**3. The Trajectory of AI Capabilities and the Inevitability of Full Automation.** The model treats AI quality ($\alpha$, governing success probability $q_i = \alpha^{d_i}$) as a fixed parameter, with Section 6.2's non-linearity analysis showing that marginal returns to $\alpha$ increase sharply once reorganization becomes optimal (Figure 2). Long-run implication: as $\alpha \to 1$, the model predicts near-complete automation of all but the final verification step in each job.

This aligns with my view, and growing empirical evidence (Dell'Acqua et al., 2023; Eloundou et al., 2024), that AI's trajectory is toward autonomous task execution with minimal human oversight. The paper's distinction between augmentation and automation is analytically useful but may be transitionally relevant: today's "augmented" tasks (human verification of AI output) increasingly become tomorrow's "automated" tasks as AI reliability improves and human verification costs exceed error costs. The more profound question, IMHO, which the model hints at but does not fully explore, concerns what work remains for humans in the limit.

The current framework assumes humans provide two irreplaceable functions: (1) verification/evaluation of AI outputs (augmented steps), and (2) hand-offs between jobs (coordination). But on (1): As AI error rates fall and outputs become more reliable, the marginal value of human verification declines. Firms may rationally reduce verification intensity or automate verification itself (AI evaluating AI). On (2): The model's hand-off costs ($t_i^H$) implicitly assume human-to-human coordination, but AI increasingly mediates these interfaces (workflow management systems, automated handoffs between AI agents).

I think the long-run equilibrium of full automation is worthy of discussion.

> 🔴 **OPEN — no response drafted yet**
>
> **What we did:** Not yet addressed.
>
> **Where in the paper:** TBD.
>
> **Internal tracking:** Full analysis and options in [06-full-automation-limit-trajectory.md](06-full-automation-limit-trajectory.md).

### Minor Concerns

**4. Exposition and Accessibility.** The paper is technically rigorous but dense, particularly Sections 3–5. The model introduces considerable notation (steps $s_i$, tasks $T_b$, jobs $J_j$, execution modes, skill costs $c_i^M$, time costs $t_i^M$, hand-off costs $t_i^H$, AI probabilities $q_i$, etc.) that accumulates quickly. Feels a lot of notations floating around and have been having a hard time following the exposition. Maybe, consider:

- Add a notational summary table early in Section 3.
- Include a worked numerical example in the main text (currently relegated to Appendix B) to build intuition before the general analysis.
- Consider moving Proposition 1 (short-run dynamic programming) to an appendix and leading with the fragmentation index intuition (currently Section 5), which is more central to the empirical application.

> 🔴 **OPEN — no response drafted yet**
>
> **What we did:** Not yet addressed.
>
> **Where in the paper:** TBD.
>
> **Internal tracking:** Full analysis and options in [07-exposition-notation-load.md](07-exposition-notation-load.md).

**5. Discussion of Skill Requirements and Worker Outcomes.** Section 6.1's discussion of worker skill effects (Examples E.1–E.3) is interesting but underdeveloped. The model shows AI can increase or decrease skill requirements depending on task structure, but the paper does not connect these predictions to measurable outcomes. Do we expect AI-exposed occupations to exhibit wage polarization? Changes in educational requirements? Skill premium shifts?

> 🔴 **OPEN — no response drafted yet**
>
> **What we did:** Not yet addressed.
>
> **Where in the paper:** TBD.
>
> **Internal tracking:** Full analysis and options in [08-skill-predictions-observables.md](08-skill-predictions-observables.md).

---

## Review #2016C

### Paper summary

This paper studies AI automation when production is an ordered sequence of steps rather than a set of independent tasks. A firm chooses, for each step, whether it is done manually, AI-augmented, or fully automated as part of a contiguous "AI chain," and it also chooses how steps are grouped into tasks and jobs while trading off specialization against hand-off costs. On the theory side, the paper gives a dynamic program for the short-run AI deployment problem for a fixed job, a discretized dynamic program for the long-run joint problem of AI deployment and job design, and an approximation result showing that a "fragmentation index" of AI-suitable steps tracks the short-run optimum up to constants. The empirical section constructs an occupation-level workflow dataset and shows that AI-executed tasks cluster in contiguous runs, that occupations with more dispersed AI-suitable tasks exhibit lower realized AI execution conditional on exposure, and that a task is more likely to be AI-executed when its immediate neighbors are AI-executed. Overall, the evidence is suggestive and broadly consistent with the paper's chaining mechanism.

### Evaluation

I found this paper interesting and thought-provoking. Its main contribution is the modeling idea: instead of treating production as a collection of technologically independent tasks, the paper models production as an ordered sequence of steps and studies how AI can automate not just isolated steps, but contiguous chains of steps. This is a natural and important perspective for thinking about modern AI systems, where the value of automation often depends on whether neighboring steps can also be delegated to AI and verified only at the end. In that sense, the paper gives a clear and intuitive framework for understanding why naively scaling up task-by-task automation can fail, and why the optimal allocation of work between humans and AI depends on workflow structure rather than just local comparative advantage.

I also think the paper does a good job of formalizing these ideas. The distinction between manual execution, augmentation, and automation is useful, and the notion of an AI chain captures an economically meaningful force: verification is borne only at the end of a chain, so adjacent AI-suitable steps can become complements. This creates a nontrivial task-allocation problem and helps explain how improvements in AI may induce reorganization rather than simple step-by-step substitution. The model therefore offers a coherent way to think about why large-scale automation may generate unexpected fallouts if workflows are fragmented, and why optimal deployment requires redesigning task boundaries and job structure jointly.

From a technical perspective, I would not describe the proofs as difficult. The dynamic programming arguments for the short-run and long-run optimization problems are reasonably straightforward once the model is set up, and the main formal results are more valuable for clarifying the structure of the problem than for introducing deep new techniques. However, I do not view this as a major weakness. In my opinion, the strength of the paper lies primarily in the model itself. It is intuitive, economically meaningful, and sheds light on an important aspect of AI deployment that standard task-based models miss. For EC, I think this kind of conceptual modeling contribution can still be valuable even when the proofs are technically simple.

My main reservation is that I did not fully understand the point of the fragmentation index and, by extension, the role of the empirical section. Proposition 5 shows that the fragmentation index approximates the short-run optimum up to constants, but I was left uncertain what exactly this contributes beyond a rough proxy for the intuition that clustering matters. It is not the main algorithmic tool, nor does it seem central to the main task-allocation results. Similarly, although the empirical section documents patterns consistent with chaining, its connection to the paper's main organizational-design contribution feels somewhat indirect. Overall, I think the paper's strongest contribution is the model; the fragmentation index and empirical section are less convincing and less clearly integrated into the central contribution.

> 🟡 **RESPONSE DRAFTED (partial) — paper update pending**
>
> **What we did:** The drafted response to R#2016A's Section-5-takeaway question above also addresses part of this concern: we will add an explicit framing paragraph explaining that the fragmentation index is the *observable* summary statistic that captures workflow adjacency — the theoretical optimum depends on per-step success probabilities `p_i` that the econometrician does not see, while the fragmentation index only needs the binary exposure pattern and is provably within constant factors of the optimum. This justifies its use in the empirical section even though it is not the algorithmic tool. The tighter integration of the empirical section with the paper's central contribution is still open.
>
> **Where in the paper:** Section 5 framing paragraph; introduction paragraph explicitly previewing the index's role.
>
> **Internal tracking:** Full analysis in [09-fragmentation-index-role.md](09-fragmentation-index-role.md). Partially addressed by the draft to R#2016A's Section-5 comment; remaining items (empirical-section integration) still open.

> 📝 **Author note on R#2016C overall**
>
> Reviewer #2016C's main takeaways: they were a big fan of the way the model is set up; they found the optimization proofs straightforward; and their reservation is concentrated on the role of the fragmentation index and — by extension — the relevance of the empirical section. We have no additional comments on this review beyond the note above.

---

## Review #2016D

### Paper summary

This paper studies how AI relates to production chains. It models production as a sequence of exogenous steps for which firms can bundle contiguous steps. It assumes steps can be performed manually, automated, or "augmented" (reviewed by humans). In an AI chain, success is stochastic and independent across steps. Regardless of the length of the AI chain, verification costs are modeled as fixed in the "augmented" step. Job boundaries are determined by trading off skill costs and handoff costs. The central idea is that AI is more useful when it applies to adjacent steps in a chain, and firms reorganize jobs around this application. The paper has an empirical exercise which includes the finding that steps with high AI execution in the Anthropic index tend to be contiguous according to GPT-generated step sequences.

### Evaluation

This is an interesting paper, and the sequence-based approach is a useful microfoundation. The insight that AI is more useful when it is adjacent to other AI applications seems to hold in practice.

The strongest challenge to the paper is the assumption that "appending a neighboring step to an existing AI chain adds no additional verification burden, while it may reduce the probability of AI's end-to-end successful completion of the extended chain." Consider AI for legal drafting. Suppose in Firm A, AI is used to gather cases, form arguments, and draft a document. Suppose in Firm B, a human gathers cases, and AI is used to form arguments and draft a document. In Firm A, verification is more costly, because the human has to ensure the cited cases exist and are appropriately used, which involves more checking.

Many of the paper's key results hinge on this assumption. If verification costs increase in the size of the AI chain, the claims with respect to comparative advantage and fragmentation weaken, as well as the J-curve microfoundation.

> 🟡 **RESPONSE DRAFTED — paper update pending**
>
> **What we did:** We thank the reviewer for pressing on this assumption. The legal-drafting example is a sharp way to frame the concern. Our framework abstracts from the *quality* of AI output and treats execution as binary (success/failure) at each attempt — the Firm A vs. Firm B verification-difficulty gradient is a real-world feature our model does not attempt to capture as a continuum. That is an explicit simplifying assumption we now state prominently alongside the chain definition.
>
> We agree with the reviewer that the central economic force — verification being more costly for longer chains — should be present in the model rather than sitting entirely outside it. We propose the following structural modification. The per-step AI-management (verification) cost becomes chain-length-dependent:
>
> ```
> t^{AI, new}_i  =  t^{AI, old}_i  +  (n − 1) · t^{Chain}
> ```
>
> where `t^{AI, old}_i` is the current step-specific verification cost (a characteristic of step `i`), `t^{Chain}` is a constant per-additional-chained-step overhead, and `n` is the length of the chain ending at step `i`. The `(n − 1) · t^{Chain}` term is exactly the "additional verification burden" the reviewer is describing: the more steps have been chained together, the more verification effort the human must expend at the end of the chain. The model reduces to the current setup when `t^{Chain} = 0`.
>
> Implications we expect (and will verify as part of the revision):
>
> 1. *Short-run dynamic program (Proposition 1).* The current O(m²) DP already iterates over chain start and end positions, so the chain length at the endpoint is implicit in the DP state. The chain-length-dependent cost term `(n − 1) · t^{Chain}` can be computed in constant time per `(start, end)` pair. We expect the DP to remain O(m²).
>
> 2. *Long-run joint optimization (Proposition 2).* The existing polynomial-time approximation argument rests on the same primitive operation — computing the cost of a job given a chain structure — which remains polynomial under the new cost. We expect the approximation guarantee to carry over.
>
> 3. *Fragmentation-index bound.* This is the bound we expect to move. Under the current model, `FI ≤ (5/4) · OPT` (Proposition 6). Under the modified cost, OPT rises relative to a fragmented baseline, so the ratio changes. Our conjecture is that the bound takes the form
>
> ```
> FI  ≤  Const · f(t^{Chain} / t^{AI, old}) · OPT
> ```
>
> with `f(0) = 1` (recovering the current bound at zero overhead) and `f` increasing in the overhead ratio. The proportionality structure survives; the specific constants depend on how large `t^{Chain}` is relative to the baseline verification cost. We will re-derive the bound and discuss it.
>
> **Scope of the modification.** This single structural change addresses R#2016D's critique *and* R#2016A's related question about whether the firm pays the worker while the AI executes (see the response to R#2016A above). In both readings, `t^{Chain}` captures real human time that the firm compensates — whether that is time spent overseeing the AI during its chain execution, time spent verifying a longer output, or both.
>
> **Where in the paper:** Model setup (near the AI-chain definition); Proposition 1 and Proposition 6 (revised statements and proofs in the appendix); a short remark on why the bound form generalizes cleanly.
>
> **Internal tracking:** Concrete formulation and expected implications drafted in [10-verification-cost-chain-length.md](10-verification-cost-chain-length.md). The prior "Candidate draft remark" in that file (which argued the success-probability channel alone was sufficient) is now superseded by this structural modification.

The GPT-generated workflows in the empirical application are interesting, though the validation is focused on internal consistency and not external validity. It would have been helpful to have a benchmark.

> 🔴 **OPEN — no response drafted yet**
>
> **What we did:** Not yet addressed.
>
> **Where in the paper:** TBD.
>
> **Internal tracking:** Full analysis and options in [11-gpt-workflow-external-validation.md](11-gpt-workflow-external-validation.md).

That being said, the overall argument in the paper is fresh and important. It helps to move forward the task-based discourse with a realistic view of production.
