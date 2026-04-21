# EC 2026 Referee Reports — Paper #2016

*"Chaining Tasks, Redefining Work: A Theory of AI Automation"*
Demirer, Horton, Immorlica, Lucier & Shahidi

Four reviews received from EC'26 program committee.

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

### [Optional] Feedback for Authors

- The paper benefits from discussing why only the steps performed by humans incur a cost, since AI use can be expensive for firms as well.
- The paper also benefits from a discussion of the worker side of the market. For instance, what happens if workers are strategic and charge based on the marginal value of their work to the firm, or if there is a shortage or abundance of skilled workers?
- Does the firm pay the human worker while she is waiting for the AI tasks to be completed? If yes, the paper benefits from justifying this.
- The paper also benefits from clarifying the main takeaway from Section 5: Does this say one cannot benefit much from knowing the successes and reordering the steps?

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

**2. Firm Boundaries, Vertical Integration, and the "Make-or-Buy" Decision for AI Services.** One of the paper's most intriguing implications but largely underdeveloped concerns how AI reshapes firm boundaries. The model implicitly treats AI deployment as an internal capability choice, but in practice, firms increasingly access AI through markets (API calls, SaaS platforms, outsourced AI services). The hand-off cost $t_i^H$ in the model captures coordination frictions within jobs, but there is no analogous friction for task handoffs across firm boundaries.

Unlike previous automation waves (robotics, ERP systems) that required firm-specific customization and capital investment, modern AI increasingly operates as a platform technology with non-negligible inference costs. AI deployment, thus, may favor vertical disintegration: firms could specialize in supervising/validating AI outputs (augmented steps) while outsourcing the AI execution itself.

The model's job design problem (Equation 2) highlights that bundling tasks into jobs trades off skill-based wage premiums ($\sum c_b$) against hand-off costs ($t^H$). But AI fundamentally alters this tradeoff in ways the current framework cannot capture. If AI chains can be outsourced with low transaction costs, the optimal organizational structure may shift from "integrated jobs with long AI chains" to "specialized evaluation roles coordinating multiple external AI providers." This parallels the modularity literature's prediction that standardized interfaces enable vertical disintegration.

The authors might benefit from explicitly modeling the make-or-buy decision for AI execution. Formally, allow the firm to choose whether AI chains are executed internally (incurring fixed AI management costs but zero marginal hand-offs) or externally (incurring per-transaction costs but enabling finer task decomposition).

**3. The Trajectory of AI Capabilities and the Inevitability of Full Automation.** The model treats AI quality ($\alpha$, governing success probability $q_i = \alpha^{d_i}$) as a fixed parameter, with Section 6.2's non-linearity analysis showing that marginal returns to $\alpha$ increase sharply once reorganization becomes optimal (Figure 2). Long-run implication: as $\alpha \to 1$, the model predicts near-complete automation of all but the final verification step in each job.

This aligns with my view, and growing empirical evidence (Dell'Acqua et al., 2023; Eloundou et al., 2024), that AI's trajectory is toward autonomous task execution with minimal human oversight. The paper's distinction between augmentation and automation is analytically useful but may be transitionally relevant: today's "augmented" tasks (human verification of AI output) increasingly become tomorrow's "automated" tasks as AI reliability improves and human verification costs exceed error costs. The more profound question, IMHO, which the model hints at but does not fully explore, concerns what work remains for humans in the limit.

The current framework assumes humans provide two irreplaceable functions: (1) verification/evaluation of AI outputs (augmented steps), and (2) hand-offs between jobs (coordination). But on (1): As AI error rates fall and outputs become more reliable, the marginal value of human verification declines. Firms may rationally reduce verification intensity or automate verification itself (AI evaluating AI). On (2): The model's hand-off costs ($t_i^H$) implicitly assume human-to-human coordination, but AI increasingly mediates these interfaces (workflow management systems, automated handoffs between AI agents).

I think the long-run equilibrium of full automation is worthy of discussion.

### Minor Concerns

**4. Exposition and Accessibility.** The paper is technically rigorous but dense, particularly Sections 3–5. The model introduces considerable notation (steps $s_i$, tasks $T_b$, jobs $J_j$, execution modes, skill costs $c_i^M$, time costs $t_i^M$, hand-off costs $t_i^H$, AI probabilities $q_i$, etc.) that accumulates quickly. Feels a lot of notations floating around and have been having a hard time following the exposition. Maybe, consider:

- Add a notational summary table early in Section 3.
- Include a worked numerical example in the main text (currently relegated to Appendix B) to build intuition before the general analysis.
- Consider moving Proposition 1 (short-run dynamic programming) to an appendix and leading with the fragmentation index intuition (currently Section 5), which is more central to the empirical application.

**5. Discussion of Skill Requirements and Worker Outcomes.** Section 6.1's discussion of worker skill effects (Examples E.1–E.3) is interesting but underdeveloped. The model shows AI can increase or decrease skill requirements depending on task structure, but the paper does not connect these predictions to measurable outcomes. Do we expect AI-exposed occupations to exhibit wage polarization? Changes in educational requirements? Skill premium shifts?

---

## Review #2016C

### Paper summary

This paper studies AI automation when production is an ordered sequence of steps rather than a set of independent tasks. A firm chooses, for each step, whether it is done manually, AI-augmented, or fully automated as part of a contiguous "AI chain," and it also chooses how steps are grouped into tasks and jobs while trading off specialization against hand-off costs. On the theory side, the paper gives a dynamic program for the short-run AI deployment problem for a fixed job, a discretized dynamic program for the long-run joint problem of AI deployment and job design, and an approximation result showing that a "fragmentation index" of AI-suitable steps tracks the short-run optimum up to constants. The empirical section constructs an occupation-level workflow dataset and shows that AI-executed tasks cluster in contiguous runs, that occupations with more dispersed AI-suitable tasks exhibit lower realized AI execution conditional on exposure, and that a task is more likely to be AI-executed when its immediate neighbors are AI-executed. Overall, the evidence is suggestive and broadly consistent with the paper's chaining mechanism.

### Evaluation

I found this paper interesting and thought-provoking. Its main contribution is the modeling idea: instead of treating production as a collection of technologically independent tasks, the paper models production as an ordered sequence of steps and studies how AI can automate not just isolated steps, but contiguous chains of steps. This is a natural and important perspective for thinking about modern AI systems, where the value of automation often depends on whether neighboring steps can also be delegated to AI and verified only at the end. In that sense, the paper gives a clear and intuitive framework for understanding why naively scaling up task-by-task automation can fail, and why the optimal allocation of work between humans and AI depends on workflow structure rather than just local comparative advantage.

I also think the paper does a good job of formalizing these ideas. The distinction between manual execution, augmentation, and automation is useful, and the notion of an AI chain captures an economically meaningful force: verification is borne only at the end of a chain, so adjacent AI-suitable steps can become complements. This creates a nontrivial task-allocation problem and helps explain how improvements in AI may induce reorganization rather than simple step-by-step substitution. The model therefore offers a coherent way to think about why large-scale automation may generate unexpected fallouts if workflows are fragmented, and why optimal deployment requires redesigning task boundaries and job structure jointly.

From a technical perspective, I would not describe the proofs as difficult. The dynamic programming arguments for the short-run and long-run optimization problems are reasonably straightforward once the model is set up, and the main formal results are more valuable for clarifying the structure of the problem than for introducing deep new techniques. However, I do not view this as a major weakness. In my opinion, the strength of the paper lies primarily in the model itself. It is intuitive, economically meaningful, and sheds light on an important aspect of AI deployment that standard task-based models miss. For EC, I think this kind of conceptual modeling contribution can still be valuable even when the proofs are technically simple.

My main reservation is that I did not fully understand the point of the fragmentation index and, by extension, the role of the empirical section. Proposition 5 shows that the fragmentation index approximates the short-run optimum up to constants, but I was left uncertain what exactly this contributes beyond a rough proxy for the intuition that clustering matters. It is not the main algorithmic tool, nor does it seem central to the main task-allocation results. Similarly, although the empirical section documents patterns consistent with chaining, its connection to the paper's main organizational-design contribution feels somewhat indirect. Overall, I think the paper's strongest contribution is the model; the fragmentation index and empirical section are less convincing and less clearly integrated into the central contribution.

---

## Review #2016D

### Paper summary

This paper studies how AI relates to production chains. It models production as a sequence of exogenous steps for which firms can bundle contiguous steps. It assumes steps can be performed manually, automated, or "augmented" (reviewed by humans). In an AI chain, success is stochastic and independent across steps. Regardless of the length of the AI chain, verification costs are modeled as fixed in the "augmented" step. Job boundaries are determined by trading off skill costs and handoff costs. The central idea is that AI is more useful when it applies to adjacent steps in a chain, and firms reorganize jobs around this application. The paper has an empirical exercise which includes the finding that steps with high AI execution in the Anthropic index tend to be contiguous according to GPT-generated step sequences.

### Evaluation

This is an interesting paper, and the sequence-based approach is a useful microfoundation. The insight that AI is more useful when it is adjacent to other AI applications seems to hold in practice.

The strongest challenge to the paper is the assumption that "appending a neighboring step to an existing AI chain adds no additional verification burden, while it may reduce the probability of AI's end-to-end successful completion of the extended chain." Consider AI for legal drafting. Suppose in Firm A, AI is used to gather cases, form arguments, and draft a document. Suppose in Firm B, a human gathers cases, and AI is used to form arguments and draft a document. In Firm A, verification is more costly, because the human has to ensure the cited cases exist and are appropriately used, which involves more checking.

Many of the paper's key results hinge on this assumption. If verification costs increase in the size of the AI chain, the claims with respect to comparative advantage and fragmentation weaken, as well as the J-curve microfoundation.

The GPT-generated workflows in the empirical application are interesting, though the validation is focused on internal consistency and not external validity. It would have been helpful to have a benchmark.

That being said, the overall argument in the paper is fresh and important. It helps to move forward the task-based discourse with a realistic view of production.
