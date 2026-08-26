# Pre-submission review: findings verified against our own code and data

Status key: **VERIFIED** = reproduced from `data/computed_objects/` with the paper's own
construction. **UNVERIFIED** = raised by a referee-lens agent, not yet checked.

---

## 1. VERIFIED (fatal). The fragmentation result is an artifact of mismatched exposure definitions.

`analysis/onet_fragmentationIndex.ipynb` cell 2 sets `ai_exposure_var = 'human_E1_fraction'`,
so the exposure regressor is the **E1-only** share. The EFI is built from **E1 or E2** tasks.

The EFI satisfies an exact identity (max deviation 1.1e-16 across all 872 occupations):

    EFI  =  1  -  share_E1E2  +  (runs of exposed steps / steps)

So EFI is *one minus the E1|E2 exposure share*, plus a small arrangement term. Its correlation
with the E1|E2 share is **-0.954**; with the E1 share only **-0.704**. Controlling for the E1
share therefore leaves the EFI coefficient loaded with the E1|E2 exposure *level*, which is not
arrangement at all.

Reproducing Table `tab:fragmentation_index_regression_exposure` exactly, then swapping in the
matched exposure share:

| specification (major SOC FE)                  | EFI coef | exposure coef |
|-----------------------------------------------|----------|---------------|
| E1 share + EFI + count  (**the paper**)       | **-0.380** (se 0.050) | +0.109 |
| E1 share + **E1E2 share** + EFI + count       | **-0.080** (se 0.080) | +0.082 / +0.438 |

Entering the arrangement term directly, which is the only part of EFI that is about arrangement:

| specification                                   | runs/step coef |
|-------------------------------------------------|----------------|
| y ~ E1E2 share + runs/step + count, no FE        | **-0.002** (se 0.031) |
| y ~ E1E2 share + runs/step + count, major FE     | **-0.029** (se 0.027) |

**Arrangement has no explanatory power.** The published -0.26 / -0.38 is the exposure-share
level effect entering through the half of EFI that the E1-only control does not absorb.
The footnoted rationale in Sec 7.2 ("conditioning on that count ensures beta_2 is identified
from how the AI-able steps are arranged") does not hold: the *count* of E1|E2 steps does not
absorb the *share* when workflow length varies.

FIX: either (a) report the specification controlling for the E1|E2 share and reframe
Prediction 2 around whatever survives, or (b) find an arrangement measure orthogonal to the
exposure share by construction. Do not submit the current table.

## 2. VERIFIED (major). Prediction 1's effect size is much smaller than the text conveys.

Reproduced mean AI chain length **1.447** (paper: 1.45). The within-occupation permutation
placebo mean is **1.383** (sd 0.012). The excess is **+0.064 steps, 4.6%**, at 5.2 placebo SDs.

Run-length distribution: 1154 chains of length 1, 308 of length 2, 96 of 3, 43 of 4, 16 of 5,
2 of 6, 2 of 7, 1 of 11. **71% of chains are singletons; 49% of AI-executed tasks are isolated.**

The text says the observed value is "noticeably larger" and "lies at the extreme tail." Both are
true of the *test statistic* and misleading about the *magnitude*. A referee who computes 1.45
vs 1.38 will conclude the model's central object is nearly absent from the data.

FIX: report the placebo mean next to the observed value, and report the share of AI tasks in
chains of length >= 2. Frame Prediction 1 as detectable rather than large.

## 3. VERIFIED (favorable). The obvious "semantic clustering" confound does NOT explain Prediction 1.

Several lenses flagged that GPT orders semantically similar tasks adjacently, and similar tasks
share AI-ability, so clustering could be mechanical. This is testable and the answer is good news.

Clustering of each label under the same within-occupation permutation null:

| label                | base rate | observed | placebo | excess |
|----------------------|-----------|----------|---------|--------|
| AI execution (is_ai) | 0.131     | 1.447    | 1.383   | +4.6%  |
| Exposure E1          | 0.141     | 1.389    | 1.291   | +7.6%  |
| Exposure E1 or E2    | 0.436     | 2.608    | 2.342   | +11.3% |

Exposure labels cluster *more* than execution labels, and exposure is a property of the task in
isolation with no chaining mechanism behind it. That is the confound, visible.

But an **exposure-stratified permutation** (permuting execution labels within occupation x
exposure-status cells, so the exposure profile is held in place) gives a null of **1.3868**
(sd 0.013) against observed 1.447: still **+4.3%, 4.5 SD**.

The clustering of execution survives conditioning on the clustering of exposure. This placebo
is not in the paper and it should be, because it is the first thing a referee will attack.

## 4. VERIFIED (major, and the biggest opportunity). The overturning result is never tested, and it works.

Prop 1 is about steps where *the human holds the advantage*. Sec 7.3 controls for focal exposure
but never splits on it, so the paper's signature theoretical claim has no direct test.
`onet_neighborAI_E1E2exposureControls.ipynb` adds exposure *controls*, not a split.

Running the split on `similarTasks_allEligibleTasks.csv` (paper's estimation sample, AMEs,
DWA-clustered SEs). Column (1) replicates exactly (+0.07/+0.12/+0.12/+0.05):

| sample                          | N     | ybar  | t-2    | t-1    | t+1    | t+2    |
|---------------------------------|-------|-------|--------|--------|--------|--------|
| all tasks (paper baseline)      | 10708 | 0.129 | +0.065 | +0.124 | +0.117 | +0.050 |
| **focal NOT exposed (E0)**      | 6076  | 0.064 | +0.064 | **+0.106** | **+0.077** | +0.061 |
| focal exposed (E1 or E2)        | 4632  | 0.214 | +0.052 | +0.125 | +0.138 | +0.031 |

Among steps human annotators judged AI cannot do at human quality, an AI-executed immediate
predecessor raises AI execution by **10.6pp on a 6.4% base rate: a 165% relative increase.**
That *is* overturning, measured directly, and it is a stronger result than what is in the paper.

With SOC major-group FE it survives: focal-not-exposed gives t-1 +0.039, t+1 +0.028 on the same
6.4% base (+62% / +44%). Honest caveat: in this subsample the distant neighbours (+0.018 each)
do not collapse the way they do in the pooled sample, which slightly weakens the
"chains are short" reading. SOC-minor FE is singular on this subsample.

FIX: make this the headline of Sec 7.3. It maps one-to-one onto Prop 1 and it is already in hand.

## 5. VERIFIED (moderate). The AI-execution indicator is a coverage indicator.

Per Appendix C: tasks not in the Anthropic data, and tasks with 100% filtered conversations, are
coded manual; every one of the 2,347 matched-and-unfiltered tasks receives Augmentation or
Automation. Confirmed in the data: 1,626 + 721 = 2,347 exactly, with no matched task coded manual.

So `is_ai = 1` if and only if the task was matched to at least one unfiltered Claude conversation.
There is no observation anywhere of a task that Claude users touched and that is nonetheless
executed manually. 15,578 of 17,925 tasks (87%) are manual by imputation, not by observation, and
317 of 872 occupations (36%) have zero AI tasks.

The Appendix C footnote answers a strawman ("for our results to be *entirely spurious*..."). The
concern is not that the results are entirely spurious; it is that the extensive margin of the
dependent variable is Claude coverage. Finding 3 above is the defence and it should be on the page.

---

## Modularization note

The referee panel spawned 55 agents (7 finders + 47 per-finding verifiers + 1 synthesis) and hit
the session limit with 49 unfinished. Rebuilt as one batched verifier per lens, run one lens at a
time: 7 agents total, at most 1 in flight. See `verify_lens.js` in this directory.
