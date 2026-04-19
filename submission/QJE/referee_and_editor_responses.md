# QJE MS 45218 — Editor Letter and Referee Reports

*"Chaining Tasks, Redefining Work: A Theory of AI Automation"*
Demirer, Horton, Immorlica, Lucier & Shahidi

---

## Editor Letter (Lawrence Katz)

Dear Mert:

I attach reports from four expert referees on your submission (with John Horton, Nicole Immorlica, Brendan Lucier, and Peyman Shahidi), MS 45218, entitled

"CHAINING TASKS, REDEFINING WORK: A THEORY OF AI AUTOMATION"

for possible publication in the Quarterly Journal of Economics. I also have read with great interest your impressive and creative development of a theory of AI-based automation and work reorganization in a model in which production requires sequential discrete steps that can be automated or augmented by AI or done manually by humans. I greatly appreciate your insights and attempt to provide some initial evidence related to AI-based "chaining" (AI being clustered in contiguous steps) and how "fragmentation" (more disperse-steps susceptible to AI automation) may limit AI adoption.

The referees and I think you are tacking timely and crucial issues concerning how improvements in AI are likely to impact the organization of production, job development, and the division of labor. We all are fans of the core insights related to the trade-offs of longer AI chains leading to higher error rates but also conserving on human supervision/verification costs. But most of the referees (R1, R2, and R4) and I find the model to be more complex and difficult to follow than necessary. We (especially R2, R4, and I) have concerns with the plausibility and durability of your core empirical work imputing workflow adjacency of tasks in occupations. Three of the referees (R1, R2, and R4) don't see a large enough overall advance for the QJE and recommend rejection. The other referee (R3) is more enthusiastic and favorable on your paper's potential for the QJE.

We need to make some very tough calls. I reluctantly share the conclusion of the majority of the reviewers (R1, R2, and R4) that your paper at present is not a good fit for the QJE. And I don't see a clear path to addressing the key issues raised by the referees. Thus, we will need to pass on your submission. I think a somewhat streamlined and more focused version of your paper likely would be a better fit for an outlet such as AEJ: Applied Economics, Quantitative Economics, or possibly RESTUD.

I am sorry that we can't provide you and your collaborators with more positive editorial news for your thought-provoking study at the QJE. We certainly remain keenly interested in research in this area. Thank you for giving us the chance to consider your insightful work.

Sincerely yours,

Larry

---

Lawrence Katz
Editor
lkatz@harvard.edu
The Quarterly Journal of Economics

Attachments:

R1's report
R2's report
R3's report
R4's report

---

## Referee Report R1 (MS45218-1-0.pdf)

**Referee Report – QJE-45218-1**
*"Chaining Tasks, Redefining Work: A Theory of AI Automation"*
(Demirer, Horton, Immorlica, Lucier & Shahidi)

### Summary

This paper develops a model to study how artificial intelligence (AI) reshapes the organization of production and the division of labor.

Production is modelled as a sequence of interdependent steps. In the absence of AI, these steps map one-to-one to tasks that firms bundle into discrete jobs assigned to workers. This endogenous job design trades off worker specialization against coordination costs.

Specifically, a worker's total wage bill is the product of the cumulative skill requirement of all tasks assigned to the job and the total time required to execute the job. Unbundling a task from a job lowers the worker's per-unit-time wage rate, yielding cost savings that scale across the entire duration of the remaining work. However, this unbundling fragments the workflow and introduces time-consuming hand-off costs whenever work passes between individuals. The optimal workflow structure must therefore balance the per-unit-time wage-reduction benefits of worker specialization against the time costs of handoffs.

Now introduce AI. Firms can deploy the technology to execute steps in two distinct modes: augmentation and automation. Under augmentation, the AI performs the step subject to direct human oversight and validation. Under automation, the AI completes the step end-to-end without human intervention, feeding its output directly into the subsequent step. AI capabilities are heterogeneous across the workflow, meaning the probability of successful AI execution is a step-specific parameter.

This capability decouples steps from tasks. Because automated steps require no direct human intervention, firms can delegate a contiguous sequence of steps to the AI as a single composite task, defined as an AI chain. In these chains, all intermediate steps are fully automated, and only the final step requires augmented human validation. Consequently, human oversight acts as a fixed cost for the entire chain rather than a marginal cost incurred per step.

However, the optimal length of an AI chain is constrained by the compounding probability of failure. Because end-to-end success requires the AI to execute every step correctly, appending a step strictly reduces the overall success rate and raises the expected time cost of human management. Firms must therefore trade off verification savings against the declining reliability of longer chains.

The model yields four main theoretical results:

**First**, it overturns standard comparative advantage logic in task assignment. Because an AI chain must end with an augmented step, appending a neighboring step simply shifts the validation checkpoint, imposing no new verification burden. By contrast, assigning that marginal step to a human terminates the current chain, forces an immediate validation checkpoint, and incurs a separate execution cost for the subsequent step. When the AI's success probability on the marginal step is sufficiently high, the savings from avoiding this extra human checkpoint dominate. Consequently, the firm optimally assigns a step to an AI chain even if human execution holds the comparative advantage for that step in isolation.

**Second**, the aggregate gains from automation depend critically on the fragmentation of AI-capable steps within the baseline workflow. Clustered AI-capable steps facilitate long AI chains and end-to-end automation, maximizing the fixed-cost verification savings. Conversely, dispersed steps severely constrain automation because human checkpoints must repeatedly intervene to bridge gaps between isolated AI tasks.

**Third**, the model also has implications for the skill composition of jobs and the degree of worker specialization. Because firms endogenously determine how tasks are assigned to jobs, AI adoption can alter both the time and skill requirements associated with different parts of the workflow. When AI reduces the skill required to complete a sequence of steps, a job's skill requirements may decline, leading to deskilling. In other cases, however, AI augmentation may increase the skill required to supervise AI outputs, raising the skill content of work. By altering the time and skill profiles of tasks, AI can also shift the balance between narrower specialized jobs and broader generalist roles.

**Fourth**, because firms optimize over a discrete set of job arrangements, marginal improvements in AI quality can trigger discontinuous reorganizations of production. This discrete structural shift provides a theoretical microfoundation for the "productivity J-curves."

**A preliminary empirical evaluation** tests the model's implications using a dataset that merges ONET tasks, AI exposure metrics, realized AI execution outcomes, and GPT-generated workflow sequences. The evidence supports three core predictions. First, AI-executed tasks cluster in contiguous sequences rather than appearing independently across workflows. Second, conditional on aggregate AI exposure, occupations with highly dispersed AI-capable tasks exhibit strictly lower rates of realized AI execution. Third, a given task is significantly more likely to be executed by AI if its immediate workflow neighbors are also executed by AI.

**Finally**, the model provides a formal two-step aggregation from discrete micro-level task sequences to a macroeconomic production function. Conditional on a chosen job design and AI deployment strategy, the firm's cost function simplifies to a two-input, firm-level Leontief function operating on skill-adjusted manual labor and skill-adjusted AI management labor. The authors then characterize the distribution of effective AI quality across firms, which allows these firm-level technologies to aggregate into an economy-wide constant-elasticity-of-substitution (CES) production function defined over aggregate capital, manual labor, and AI-assisted labor.

### Assessment

This is a highly ambitious paper that tackles an issue of first-order importance. Since Autor and Thompson (2025), an important open question in the literature has been how to formally microfound the idea that "jobs are bundles of tasks," as this appears to have important implications for the labor-market effects of new technologies. In this regard, the paper's approach to endogenous job design and coordination frictions is promising and represents a meaningful theoretical step forward.

However, in its current form, the paper lacks a clear organizing focus, obscuring its primary contributions and making the exposition difficult to follow.

**First**, the results appear to be driven by two distinct mechanisms that operate in parallel. Some results arise from one mechanism, while others rely primarily on the other. This does more than create a somewhat fragmented narrative; it also makes it difficult to determine which assumptions are responsible for which results at any given point in the paper. As a result, considerable effort is required to disentangle the underlying logic of the analysis. Moreover, the empirical evidence presented later in the paper speaks primarily to only one of these mechanisms, leaving the other largely untested.

**Second**, the baseline model is extremely parameter-heavy. Each step is endowed with six distinct exogenous parameters, in addition to the global AI quality parameter. As a result, even very small examples quickly involve a high-dimensional parameter space (for instance, with just three steps, the reader must keep track of nineteen parameters). Moreover, the analysis distinguishes between a short-run optimization problem and a long-run job-design optimization, further increasing the model's conceptual complexity. This has two drawbacks. First, the combination of extensive parameterization and layered optimization makes it difficult to develop a clear intuition about how the model behaves or which forces drive the main results. Second, it forces the authors to rely heavily on carefully selected examples to convey their insights, making it challenging to assess the broader applicability of the results.

**Third**, compounding this complexity, the paper devotes substantial space to developing the algorithmic machinery needed to demonstrate that the model is computationally tractable. While this exercise is valuable in showing that the firm's optimization problem can be solved, the resulting procedure largely functions as a mathematical black box from the reader's perspective, revealing relatively little about the economic structure of the solutions. As a result, the delivery of clear economic insights is substantially delayed.

**Fourth**, the macroeconomic extension imposes a heavy burden of additional assumptions without delivering a commensurate theoretical or empirical payoff. Consequently, this section places additional cognitive demands on the reader without strengthening the paper's central contribution or clarifying its central mechanism.

**Taken together,** these issues make the paper difficult to follow and make it hard to assess its main contribution. I found it challenging to disentangle the different mechanisms and see how the various parts of the analysis fit together. I elaborate on these and several related issues in the comments below.

### Detailed Comments and Suggestions

#### 1. Clarifying and Separating the Two Mechanisms

A central difficulty in evaluating the model is that it appears to combine two distinct mechanisms that operate largely in parallel, yet the exposition does not clearly separate their roles. As a result, it is often difficult to determine which mechanism is responsible for the paper's various results.

The first mechanism arises from the AI chaining. Because only the final step of a chain requires human verification, verification acts as a fixed cost rather than a marginal cost incurred at each step. This creates a tradeoff between extending the chain (which reduces skill verification costs) and the declining probability that AI successfully completes the entire sequence (which increases human supervision time). This mechanism generates several of the paper's central results, including the reversal of comparative advantage in step assignment, the importance of fragmentation in the distribution of AI-capable steps within the workflow, and the non-linear adoption dynamics that produce the J-curve effects. The empirical section appears to test precisely these predictions, focusing on the clustering of AI-executed tasks, fragmentation, and local complementarities in AI deployment.

The second mechanism is conceptually distinct and arises from the tradeoff between worker specialization and hand-off costs in the endogenous job design problem. Here, the firm determines how tasks are bundled into jobs while balancing the gains from specialization against the coordination costs of transferring work between workers. Changes in the time and skill requirements of tasks, such as those induced by AI's introduction, can therefore alter the optimal degree of worker specialization and the skill composition of jobs. This mechanism underlies the paper's results on upskilling, deskilling, and the shift between specialized and generalist roles.

Crucially, this second mechanism does not appear to rely on AI chaining. It arises even in environments where AI only augments human work rather than automating chains of steps. This is particularly clear in Examples III and IV, where the results on skill composition and specialization emerge under AI augmentation alone. As a result, these implications do not seem to depend directly on the paper's central chaining mechanism.

This distinction matters because the paper frames itself primarily as a theory of AI chaining and automation. The introduction, conceptual discussion, and empirical analysis all emphasize the role of chaining in shaping decisions about task assignment and AI deployment. The reader, therefore, expects the model's broader labor-market implications to follow from this mechanism. However, the results on upskilling, deskilling, and the shift between specialist and generalist roles appear to arise from the separate job-design mechanism described above. This creates confusion and makes it difficult to determine which assumptions are responsible for the various results at different points in the paper.

Of course, it is possible that these two forces interact in meaningful ways. However, the firm's optimization problem is so complex and heavily parameterized that this interaction is difficult to discern at first glance. The authors themselves seem to implicitly recognize this in Examples III and IV, where the AI-chaining element is effectively shut down to illustrate the implications of AI for job design.

More broadly, the paper would benefit from clarifying whether its primary contribution lies in the AI-chaining mechanism, the implications of AI for the specialization–coordination tradeoff in job design, or the interaction between the two.

If the paper aims to develop both mechanisms, two things become particularly important. First, the mechanisms should be clearly and explicitly delineated so that the reader can understand which results are driven by which force. Second, the paper should make clear how the interaction between the two mechanisms generates novel and interesting effects.

#### 2. Model Complexity and Economic Intuition

A second difficulty is the model's complexity. As noted above, each step is associated with a large number of parameters, and the firm's problem combines short-run optimization of AI deployment with long-run optimization of job design. Taken together, this creates a highly parameterized environment in which it is difficult for the reader to develop a clear intuition about how the model behaves or which forces drive the results.

I do not have a specific recommendation for how the model should be simplified. However, clarifying the paper's primary focus (whether it is the AI-chaining mechanism, the implications of AI for the specialization–coordination tradeoff, or the interaction between the two) would likely provide guidance. Once the central mechanism is clearly established, it may become easier to identify which elements of the model are essential to delivering the paper's main insights and which could be simplified or abstracted.

A related point concerns Section 4, which establishes the algorithmic machinery required to demonstrate that the model is computationally tractable. This result is certainly valuable. At the same time, devoting substantial space to this material comes at a cost. It increases the reader's cognitive load while offering relatively limited insight into the model's underlying economic trade-offs and delays the presentation of the paper's main economic insights. Put differently, Section 4 occupies prime real estate in the paper, and it may be worth considering carefully whether this is the best use of that space.

#### 3. The Macroeconomic Extension

The macroeconomic extension in Section 6 adds an additional layer of structure to the model by deriving an aggregate production function from the firm-level environment. The central result is that, under an appropriate distribution of heterogeneous AI capabilities across firms, the micro-level Leontief production technologies implied by the model can be aggregated into an economy-wide CES production function defined over capital, manual labor, and AI-assisted labor.

While this is an interesting theoretical exercise, the result is essentially a **rationalization exercise**. It shows that there exists a distribution of heterogeneous AI capabilities that allows the aggregation from firm-level Leontief technologies to a macro-level CES production function. As such, the result does not appear to deliver strong testable implications or sharp restrictions beyond demonstrating that such an aggregation is possible.

At the same time, presenting this result requires introducing a substantial amount of additional notation and assumptions. This adds further complexity to a paper that is already heavily parameterized, while the payoff of the extension remains somewhat unclear. In its current form, it is difficult to see how the macro section materially advances the paper's central arguments about AI chaining and the organization of production.

For this reason, I would consider moving the macroeconomic extension to an appendix or an online appendix. If the authors prefer to keep it in the main text, it would be helpful to make the value of the result more explicit. For example, the paper could compare the macroeconomic implications of its framework with those obtained in recent work, such as Acemoglu (2025) or Aghion and Bunel (2024), which derive aggregate productivity effects of AI adoption using the canonical task-based approach. Such a comparison could clarify whether the chaining mechanism leads to distinct macroeconomic predictions.

#### 4. Clarifying the J-Curve Connection

The paper argues that the model provides a microfoundation for the "productivity J-curve" phenomenon. However, the connection between the model and this empirical concept could be clarified more precisely.

In the empirical literature, a productivity J-curve refers to the time path of observed productivity following the adoption of a general-purpose technology: productivity growth is initially weak and only later accelerates as complementary organizational and technological adjustments take effect (Brynjolfsson, Rock, and Syverson, 2021). By contrast, the model appears to generate discontinuities in organizational choices and in the marginal value of the technology when AI quality crosses certain thresholds.

It would therefore be helpful for the paper to clarify how these results map into the empirical notion of a J-curve. If the claim is that the model speaks to productivity J-curves in the sense used in the empirical literature, it would be useful to explain more explicitly how the threshold effects in organizational choices translate into a J-shaped path of measured productivity over time, assuming that the technology improves smoothly. As written, the model clearly produces threshold effects in organizational choices, but it is less clear how these generate the empirical productivity dynamics typically associated with J-curves.

#### 5. Improving the Introduction and Framing

The introduction is "locally" well written (individual passages are clear) and contains many helpful examples. However, the introduction spends considerable time developing the conceptual distinctions between steps, tasks, and jobs. As a result, the model and its key trade-offs are introduced gradually and in a somewhat fragmented way, with the main idea only becoming fully clear after several pages of exposition. This makes it difficult for the reader to quickly grasp the paper's core contribution.

A more streamlined structure could help sharpen the paper's message. In particular, the introduction could first explain what is missing in the existing literature and how the paper addresses that gap. It could then present the central mechanism of the model and the key trade-offs it introduces. Finally, it could summarize the main results and explain how they change our understanding of the conventional wisdom about automation and task allocation. Organizing the introduction around this sequence would likely make the paper's main contribution and mechanism much clearer.

### References

Acemoglu, D. (2025). The simple macroeconomics of AI. *Economic Policy*, 40(121), 13-58.

Aghion, P., & Bunel, S. (2024). AI and growth: Where do we stand?. *Policy Note.*

Autor, D., & Thompson, N. (2025). Expertise. *Journal of the European Economic Association*, 23(4), 1203-1271.

Brynjolfsson, E., Rock, D., & Syverson, C. (2021). The productivity J-curve: How intangibles complement general purpose technologies. *American Economic Journal: Macroeconomics*, 13(1), 333-372.

---

## Referee Report R2 (MS45218-1-1.pdf)

**Referee Report on "Chaining Tasks, Redefining Work: A Theory of AI Automation"**
by Demirer, Horton, Immorlica, Lucier and Shahidi

*February 28, 2026*

**Before I begin … a rant** Who submits a paper that is single spaced, maybe 11 point font and that is 83 pages long? I mean, come on. I know you want to "format for the journal you want be in," but I have got to read this thing. And if things go well for you, I may be one of the last four people in the world to do so because no one is going to read that paper once it's published. Instead, for normal situations, we just go for the large font, readable pre-print or better still, just use AI to tell us what all them pages is about.

OK on to the main report ...

### 1 Summary

This paper offers a new theory of AI automation. The theory is new in its way, especially, its application to the present moment. But my bottom line is that I worry about the half-life of the theory as I will explain below.

But first, what does the paper do? The paper proceeds as a linked chain of construction steps in which later claims inherit the assumptions and measurement choices made earlier. It begins by modeling production as an ordered sequence of steps and by defining the central objects that make the mechanism work: AI "augmentation" versus "automation," and an "AI chain" in which multiple consecutive steps are executed by AI with only the final output verified, a setup the authors explicitly connect to O-ring logic where failure in one component degrades the value of the whole chain. It then turns that conceptual setup into a tractable optimisation problem, showing how to compute the time-optimal AI strategy for an *m*-step job via dynamic programming in polynomial time, thereby doing much of the load-bearing work for the theory-to-empirics bridge. Next, it compresses the model's "clustering matters" intuition into summary statistics like the fragmentation index, with formal results tying fragmentation to the cost of optimal chaining and thus motivating why dispersion of AI-suitable steps should predict weaker AI deployment in practice. Finally, it operationalises the theory by constructing an occupation-by-task dataset (combining O*NET with AI exposure/execution labels and task position in a production sequence) and uses those constructed variables to test three empirical predictions about clustering into AI chains, fragmentation weakening the exposure-to-execution mapping, and neighbour spillovers in execution status, plus robustness checks to alternative GPT prompts for the ordering step. Read this way, any fragility in the early links (how "steps" are sequenced, how exposure/execution are labelled, or how fragmentation is measured) propagates mechanically into the interpretation of the empirical coefficients, so the project can look statistically tidy while still being conceptually undermined by a single weak upstream link.

See what I did there? The paper itself is an example of the very thing that it is modelling, and the issues that the model anticipates are also issues with this paper. Now I don't know if we can blame AI or any automation for that, as I couldn't see a reference to that beyond the use of AI in the empirical methodology, but I can tell you that I was very pleased when I made the connection.

### 2 What is strong?

To start, the model is interesting. It takes the task-based approach to task (yes, I am having one of those days), but that is the most recent arrival in what is a long line of detractors who have realised that approach isn't really doing it for AI as it did for straight out automation. The twist in this paper is basically an O-Ring or, in places, weakest link approach, which means that optimal adoption of AI cannot be conceived of as task-by-task but instead in groups of tasks. The paper then notes that when hand-offs are costly, it is natural to think of those groups as adjacent tasks that can be automated as a chain. This is a good insight, but it is not earthshattering. As the paper notes, others have contemporaneously (and independently) identified the same type of thing. The paper then places that within an organisational context, which is useful. It notes, as economists have done for centuries, that this naturally leads to a short-run adoption situation (where lots of things are fixed) to a long-run adoption situation (where many of those are variable), and that will impact what we observe today. I feel that the Le Chatelier principle is missing from this analysis, but that's not a big deal.

The paper then veers into computer science and computabilty that is kind of abrupt for this economist but I have no real basis to work out whether it is insightful or not here. I guess if it wasn't there I wouldn't have noticed it being gone.

All that said, the meat of the paper is to take the model and use it to derive a fragmentation index, which potentially can do some interesting empirical work. So at this point, the tasks in the paper are progressing well.

### 3 What is the weak link?

You know what is happening next. Then comes the weak links. First, we jump to provide a microfoundation for a macroeconomic analysis. Basically, it says if we assume heterogeneity is shaped *just so*, we can recover CES. That's nice, but I had trouble seeing it as an essential task. Maybe an appendix point. I may be missing something here. Put simply, sometimes the tasks you think you are doing because they are O-Ring essential are not that at all.

But it is the empirical section that I think, in the end, comes up short. To be sure, the authors make a somewhat heroic attempt to draw something empirical from the apparatus they have spent the first 35 pages of the paper building up, and there are lots of clever bits. But is it convincing or telling me something that might stand the test of time? There, I don't think the case is made.

**Empirical construction and main tests.** The empirical section operationalises the model by building a task–occupation panel that assigns three attributes to each O*NET task (treated as a production "step"): (i) an exposure label from Eloundou et al. (2024), (ii) an execution label from Anthropic's Economic Index (AEI) mapping of Claude conversations to task categories, and (iii) a position in an occupation-specific workflow sequence generated by an LLM. Specifically, the authors combine the May 2023 O*NET task lists with human exposure labels (treating E1 as exposed by default) and then infer realised execution mode (manual vs. AI-augmented vs. AI-automated) by mapping AEI's conversation taxonomy to tasks and classifying tasks by the majority share of non-filtered conversations, while treating tasks that are unmapped or 100% filtered as manual. They then prompt `GPT-5-mini` with the full set of tasks for each occupation and ask it to return the "typical sequential order" in which tasks are performed, yielding a ranked JSON list that encodes adjacency and "runs" of tasks within occupations. This constructed ordering is used to test three qualitative predictions: (1) AI-executed tasks cluster into contiguous "chains" more than in placebo datasets formed by within-occupation reshuffling; (2) conditional on exposure, occupations whose AI-able tasks are more dispersed in the LLM-imputed workflow exhibit lower realized AI execution; and (3) among conceptually similar tasks (proxied using Detailed Work Activities), adjacency to AI-executed neighbors predicts a higher probability of AI execution, with placebo reshuffles used to benchmark magnitudes and fixed effects used to limit comparisons within occupation families and within DWAs.

**Why the claims are limited: the workflow ordering is an LLM-imputed latent object.** The central empirical move—introducing workflow adjacency—rests on a generated ordering of O*NET tasks that does not exist in the underlying data. O*NET task lists do not observe production sequences; they are catalogues of job activities. The paper, therefore, substitutes an LLM's narrative reconstruction of "typical" task order for an observed workflow, and all subsequent adjacency-based objects (chain length, chain count, the empirical fragmentation index, and neighbour indicators in the spillover regressions) are functions of that imputation. Although the authors compare outcomes to placebo datasets that reshuffle positions and show robustness to alternative prompt formulations, these checks primarily establish that the LLM is not generating purely random permutations, not that it is recovering a correct or policy-invariant production order. Moreover, the LLM may systematically cluster tasks by semantic similarity (e.g. writing-related tasks adjacent to one another) in ways that are mechanically correlated with AI executability, which can inflate "chain" patterns even when true operational workflows are more parallel, conditional, or intermittent than the imposed linear ordering. The paper itself notes definitional mismatches between model-implied chains and observed task labels, and therefore collapses automation and augmentation into a single "AI execution" indicator for several analyses, underscoring that the empirical chain concept is a reduced-form proxy for a theoretical object rather than a direct measurement.

**Why the findings may be ephemeral as AI and measurement evolve.** Because workflow position is LLM-generated and AI execution is proxied using a particular platform's contemporary usage taxonomy, the empirical relationships should be interpreted as contingent snapshots of a rapidly moving target. First, as frontier models improve, the economically relevant notion of "chaining" is likely to expand from short local runs to longer multi-step pipelines and end-to-end systems, weakening the paper's reliance on immediate adjacency (indeed, the paper notes that distant neighbours matter little partly because observed chains are short, and suggests this could change as AI quality rises). Second, the mapping from observed conversations to "automation" versus "augmentation" is platform- and period-specific; tool specialisation (e.g. different models for coding versus writing) and changes in privacy filtering or usage composition can shift which tasks appear AI-executed without any underlying change in production technology. Third, if firms reorganise work in response to AI (the paper's long-run mechanism), then both the underlying task composition and the true workflow ordering may change endogenously, so that an ordering inferred from a static taxonomy of tasks becomes increasingly stale. In short, the empirical exercises most credibly support a limited claim: conditional on the paper's constructed workflow proxy and its contemporary execution labels, AI involvement appears to cluster locally, and fragmentation correlates with a weaker exposure-to-execution mapping. Stronger claims about stable complementarities or enduring occupational "chain structure" are likely to be temporary as model capabilities, workflow tooling, and organisational responses evolve.

### 4 Conclusion

The bottom line is that (a) there is nothing wrong with this paper and (b) it makes some useful points. The issue is whether it is interesting for a general economics audience, and I think that the world is not in a place where it is possible to conduct this type of empirical exercise to give rise to long-lasting results or insights. A good field journal is a better match for this.

---

## Referee Report R3 (MS45218-1-2.pdf)

Excellent paper, I wish I'd been a part of it!

Just a few small revisions I'd like to request:

### Substantive

- On p.34 you say you use "human-generated labels for AI exposure of O*NET tasks from Eloundou et al. (2024)". But this confuses me, since almost all of the AI exposure labels from Eloundou et al. are AI-generated; they just validate them against a small set of human-generated labels.
    - Are you saying you managed to get statistical power even after restricting your analysis to this small subset? If so, I'm afraid the analysis is suspect, since I don't believe the set of tasks for which they got human labeling was random.
    - If you used the whole dataset, don't refer to the labels as human-generated.
- Figure IX: This figure seems incomplete with a panel (c): *Share of Exposed Tasks Executed* on *Fragmentation Index*. Indeed to my mind this would be much more relevant than (a) or (b).
- On p.41, you say "the coefficient on the empirical fragmentation index is negative in nearly all specifications, indicating that greater dispersion of AI-able steps is associated with lower realized AI execution at the job level after controlling for AI exposure. The only exception arises when we use the stricter fragmentation measure in Definition 1 without SOC controls in column (1). Under this conservative measurement of the hypothetical exposure-to-execution mapping, the fragmentation estimates are imprecise and statistically insignificant…"
    - From Table III, I think it would be more appropriate to just say that the fragmentation index is always statistically insignificant under Definition 1 (but always highly significant under Defn. 2). Certainly the -0.01 of regression 3 can hardly count toward the "nearly all" (5 of 6) specifications.
- The notes under table C.III run off the page, so I couldn't evaluate them. I don't doubt they're fine, of course, but please do make them visible in a revision!
- Since the outputs of ChatGPT are stochastic, please make public the occupation-level task orderings by prompt, so that the results are replicable. (Apologies if they're available somewhere, but I wasn't able to find them.)

### Nitpicks

- On p.17: instead of "both… along with", either cut "both" or use "both… and"
- On p.22: I don't understand the phrasing "Write *F* for the set of steps that fail, and *C*… for the random variable…". They're both random variables, no? One is fully dependent on the other, so maybe say "Write *F* as the random variable representing the set of steps that fail, and *C* = {*C*₁, …, *C*ₖ} as the corresponding collection…"?
    - Likewise, "Given a realization of *C* and *F*" makes them sound independent; I would just say "Given a realization of *F*".
    - Since we've already defined "failed" as "not successful", I would say "successful" rather than "non-failed".
- On p.28: The notation/terminology in (and surrounding) expression (6) is confusing to me.
    - We are summing across the tasks *Tb* in the manual and AI-assisted subsets of *J*, no? Wouldn't it be more correct to define these subsets--say, denoting them *J*ᴹ and *J*ᴬ--and then sum across *T*ᵦ ∈ *J*ᴹ and *T*ᵦ ∈ *J*ᴬ?
    - Total compensation is a weighted *sum* of *w*ᴹ and *w*ᴬ, not really an average, no?
- p.32: you say that *ρ* < 0 indicates "some degree of complementarity", but this would be true of any *ρ* < 1. What *ρ* < 0 entails is *gross* complementarity.
    - But do you use *ρ* < 0? It seems to me that you only use *ρ* < 1.
- Just a formatting issue, but on six occasions the phrase "Appendix Appendix [letter]" appears.

The rest of my comments are only suggestions.

### Framing

- The results are framed as being about "AI automation". But for the most part, I think they apply to automation and the structure of work in general. So it might make more sense to call it "...A Theory of Automation, with Applications to AI".
- Likewise, the abstract says that one of the insights is that "comparative advantage logic can fail with AI chaining". But handoff costs between people can make it fail even without chaining, as in Becker and Murphy (as well handoff costs between people and non-AI machines).

### Other comments

- Do we need to have Section IV, on how to compute the optimization, in the body of the paper? I see the value of stating the O(.) results somewhere--what makes them economically interesting is that we should expect firms to find an ~optimum--but at minimum I would have thought it made more sense to put the proofs in the appendix.
- On p.25, it's not clear to me why the possibility that AI leads to more specialization is presented as only "technically possible". It seems intuitive to me that this would happen sometimes, and as Example V shows, AI could make it optimal to combine more tasks into a single job under conditions that don't seem outlandish. I would cut "technically" or spend longer explaining why you think this is extremely unlikely in practice.
- I'll confess I don't see a lot of value in Section VI. The mechanism used to produce a CES aggregator--in which firms first form expectations of AI quality, then design hard and fast task and job definitions, then deploy, never reorganizing their work in light of what their AI turns out capable of--presumably does happen sometimes, but of course the main thing keeping substitutability in aggregate production positive is that there are different kinds of firms making different, partly substitutable outputs. As you cite, we already know e.g. from Sato (1975) that firm-level Leontief production functions can aggregate to CES, so is there much added by exploring one narrow way this could occur, via uncertainty about AI quality? To my mind this would be a better fit for a supplemental appendix.
- Maybe this is for another paper, but in Section V.III on the non-linear impacts of AI improvements, I was hoping to see more than the short commentary and one example. After all, AI is improving rapidly, and non-linear impacts are the ones we most desperately need help predicting…! E.g.:
    - Quantitatively, to what extent do you think these "machine handoff costs" (/need for time and skill verifying AI outputs) explain the "puzzle" that a large share of tasks seem cheaply automatable but there have been only minor economic impacts so far?
    - On some extrapolation of AI capabilities, when should we start seeing the effects arrive "steeply"?
    - To what extent do you think that sufficiently advanced AI/automation could have even more of an impact than automation in an atomistic task model, by saving on handoff costs between jobs?

---

## Referee Report R4 (MS45218-1-3.pdf)

**Referee report: Chaining Tasks, Redefining Work: A Theory of AI automation**

**Authors:** Mert Demirer, John J. Horton, Nicole Immorlica, Brendan Lucier, Peyman Shahidi

**Summary:** The paper develops a model of AI-automation in which production consists of sequential steps that can be automated or augmented by AI, and jobs may be reorganized in response to AI adoption. When adjacent steps can be delegated jointly to AI, the value of utilizing AI increases. Empirically, the paper presents evidence consistent with three model predictions: AI-executed steps cluster in contiguous blocks, occupations with more dispersed AI-exposed steps exhibit less realized AI execution conditional on exposure, and AI use of adjacent steps increases the likelihood that a given step is executed by AI.

**Overall assessment:** I found the model novel and interesting, and I think the paper could eventually have strong potential. However, the current version falls short of that potential. I think the current draft could present the theory in a more efficient way and engage more with the empirical content of the theory by confronting it with data in a more significant and meaningful way. In my view, the most interesting aspect of the model, which is underexplored in the paper, is that it allows us to think about how jobs are reorganized in response to automation shocks, something existing models are largely silent on. I list my comments below, starting with the big picture and then narrowing the focus.

### Comments:

- **Why is this theory needed?** The paper spends most of its time outlining a new theory and develops and presents a large number of theory-motivated predictions, illustrated by examples. My main concern with this approach is that it largely misses what traditionally *motivates* the development of new theories in Economics. After reading the paper, I was left wondering about a couple of core motivational questions: What question can we answer or answer better by viewing the world through this lens? Where do existing theories fall short? What features of the data can the theory speak to but others cannot? Why do these data features plausibly matter?

    A lot of these questions could be answered better by focusing more on the empirical content of the new theory, and I list some ideas in my next comment below. Unfortunately, the current version of the paper, which starts talking about any data only on page 34, treats the empirical content of its theory, and to what extent it is backed up by data, as a bit of an afterthought.

- **What empirics validate the model?** Right now, the paper focuses on three predictions of its theory. AI-automated steps are adjacent, adjacency of automatable steps leads to more automation at the occupation level, and a given step is more likely to be automated if its neighbors are automated. Really, the last two can be merged into one idea: Adjacency fuels automation.

    I am sympathetic to the approach the authors take to test these predictions in the data.[^1] However, I think the data section should be expanded in a major way.

    My main concern is that this validation does little to signify the relevance of this particular theory for thinking about important empirical patterns of AI-driven automation. I think that a major reframing of the paper is required to alleviate this concern. Two possible options come to mind.

    **Option 1: framing the paper as a *theory of AI adoption*.** This is closer to what the draft currently offers. One unique aspect of the theory is that it is able to speak to the discrepancy between exposure to automation and actual automation decisions. Currently, the data that is used to validate the adoption aspects of the theory utilize the gap between Anthropic's usage data and Eloundou et al.'s exposure data. The validation approach is limited to one regression (equation 21) which does not have significant results for one of its two definitions of fragmentation (it also runs into applied concerns, e.g. that workers may be using Claude for only those kinds of tasks where it has a competitive advantage vis-a-vis its competitors). Complementing this analysis with other data points would therefore be valuable to convince the reader that the theory does well in predicting AI adoption. For instance, perhaps there is a way to compare automation decisions of *the same task across occupations* where some occupations have adjacent automateable steps and other do not. It is also worth thinking about whether the theory has anything to say about the adoption of non-AI technologies during historical automation episodes.

    **Option 2: framing the paper as a *theory of job design*.** In my opinion, this would be a more interesting, but also a more challenging approach. Here, the paper would benefit from a juxtaposition of the model's predictions for job design with data on job descriptions, e.g. from vacancy data (Lightcast). In the current version of the draft, this aspect of the theory is underexplored. The closest the authors come to making testable predictions is the following paragraph on page 24:

    > "One hypothesis is that AI deployment will tend to have a normalizing effect on tasks, reverting both skill requirements and time requirements toward the mean. [...] This suggests that AI deployment may reduce worker specialization."

    It would be very interesting to formalize and then test these types of predictions, as they would contribute to our understanding of how the boundaries of a job are determined and can change in response to technological advancement. I see big potential in this direction, but the current draft does not offer a satisfactory analysis even of the model's theoretical predictions in this domain.

- **Verbosity.** The first two thirds of the paper are characterized by what to me feels like quite a verbose theory section with overly long/numerous examples that don't efficiently make succinct points. Much of it felt like it could be cut or significantly compressed without much loss, or at least moved to an appendix. The problem is compounded by the fact that the paper at first explores various, seemingly unrelated, predictions of the theory, all without striving towards an apparent destination. The paper would be far easier to read if it focused on a more narrow set of predictions that are testable in the data, and then moved more swiftly to execute such tests. Relatedly, the quality and clarity of writing could also be significantly improved in various places.

- **Micro-foundation for wage equation.** I was left unsure about what micro-foundation would deliver the wage formulation in equation (1). I think it would be very helpful to spell this out much more clearly and explicitly. Otherwise, this equation, which is at the heart of the theory's prediction for wages, seems to be too ad-hoc to be believable.

- **Do specialists earn low wages?** A related point: The model seems to predict/assume that specialists should earn a lower wage rate than generalists. In fact, this is a key part of the tradeoff firms face. Does that not contradict how we would typically think of wages vary with specialization? At least it seems to require significant discussion.

- **General equilibrium.** The generalization of the theory, which is at first introduced as a firm optimization problem, to general equilibrium, seems to lack a formal treatment of the model environment, which makes it harder to comprehend. Right now, its sole purpose is the formalization of the aggregation result. It would arguably be more interesting to expand the GE setting, formalizing it carefully, and then use it to think about economy-wide changes in job design as a direct effect of automation.

[^1]: Using an LLM to order ONET tasks within an occupation seems like an innovative and correct use case for LLMs in Economics research. However, in how the model is taken to the data, I do have mild reservations about the idea that *every* step within an occupation must belong to a single chain, rather than having some separate, independent step chains within an occupation (teaching and research may both be steps an economics professor must complete within their job, but which comes first?).
