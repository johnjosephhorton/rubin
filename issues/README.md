# QJE Revision Issues — Index

Triage of referee and editor feedback on MS 45218 "Chaining Tasks, Redefining Work: A Theory of AI Automation". Source feedback: [../submission/QJE/referee_and_editor_responses.md](../submission/QJE/referee_and_editor_responses.md) and GitHub Issue #3.

Each issue file below:
- Quotes the referee(s) raising the concern
- Reports verification against the current `writeup/` source (file:line where applicable)
- Lists 2–4 ways to address it
- Gives an effort vs. payoff estimate and a recommendation

## Priority map

A rough placement on (effort to do, payoff in a resubmission). Numbers correspond to the issue files in this folder.

```
Payoff
  ^
H |  02 03 08 10                      01 16
  |  20 19 11                         21 03-opt2
  |  06 07 17
  |            14 15 12
M |  04 18 22                         05
  |     09
  |
L |  23 24 25 13
  +---------------------------------> Effort
       Low              Med          High
```

## Issues by theme

### A. Big-picture restructuring

| # | Issue | Effort | Payoff |
|---|---|---|---|
| [01](01-two-mechanisms-separation.md) | Two mechanisms (chaining vs. job-design) conflated | M–H | **H** |
| [02](02-motivation-why-theory-needed.md) | Motivation gap — "why is this theory needed?" | M | **H** |
| [03](03-empirical-reframe-adoption-or-jobdesign.md) | Reframe as theory of AI adoption OR job design | M–H | **H–VH** |
| [04](04-title-framing-automation-with-ai.md) | Title: "automation, with applications to AI" | L | M |
| [08](08-introduction-streamline.md) | Introduction needs streamlining | M | **H** |

### B. Paper length / structure

| # | Issue | Effort | Payoff |
|---|---|---|---|
| [05](05-model-complexity-parameters.md) | Model is parameter-heavy (19 params for 3 steps) | M | **H** |
| [06](06-section-iv-optimization-to-appendix.md) | Section IV (DP machinery) to appendix | L | **H** |
| [07](07-section-vi-macro-to-appendix.md) | Section VI (macro CES aggregation) to appendix | L | **H** |
| [09](09-paper-length-formatting.md) | Paper too long / dense formatting | L–M | M |

### C. Empirical durability & expansion

| # | Issue | Effort | Payoff |
|---|---|---|---|
| [10](10-llm-workflow-ordering-fragility.md) | LLM-imputed workflow ordering is fragile | M–H | **H** |
| [11](11-empirical-expansion-new-tests.md) | Empirics need expansion beyond one regression | M | **H** |
| [12](12-single-chain-per-occupation.md) | Single-chain-per-occupation assumption | M | H |
| [13](13-publish-task-orderings.md) | Publish GPT task orderings for replicability | L | H |

### D. Model specification

| # | Issue | Effort | Payoff |
|---|---|---|---|
| [14](14-wage-equation-microfoundation.md) | Wage equation (1) lacks micro-foundation | M | H |
| [15](15-specialist-generalist-wages.md) | Specialists-earn-less contradicts intuition | M–H | H |
| [16](16-general-equilibrium-formal-setup.md) | GE setup lacks formal treatment | L (if moved) – H (if formalized) | H |
| [17](17-j-curve-connection.md) | J-curve connection needs clarification | M | H |

### E. Content-specific fixes

| # | Issue | Effort | Payoff |
|---|---|---|---|
| [18](18-eloundou-labels-terminology.md) | "Human-generated" Eloundou label terminology | L | H |
| [19](19-fragmentation-regression-framing.md) | p.41 "nearly all specifications" framing dishonest | L | **H** |
| [20](20-figure-ix-add-panel-c.md) | Figure IX panel (c) missing | L | **H** |
| [21](21-section-v-iii-expand-nonlinear.md) | Section V.III non-linear impacts — expand | M–H | **VH** |
| [22](22-technically-possible-rephrase.md) | "Technically possible" specialization rephrase | L | M–H |

### F. Typography / copy-editing

| # | Issue | Effort | Payoff |
|---|---|---|---|
| [23](23-appendix-appendix-typo.md) | "Appendix Appendix" artifact (6 occurrences) | L | H |
| [24](24-table-notes-page-overflow.md) | Table C.III notes run off page | L | H |
| [25](25-notation-nitpicks.md) | Notation nitpicks (p.17, 22, 28, 32) | L | M–H |

## Suggested meeting agenda

Before the meeting, decide the **top-level framing question** that governs almost everything else:

> **Is this a theory of AI chaining that happens to have labor-market implications, or a theory of job design with AI as the driving shock?**
>
> (Issue 03 — R4 Option 1 vs. Option 2)

Once that's settled:
- Several issues collapse automatically (e.g. choosing R4 Option 2 elevates Issues 15 and 16 to centerpieces; choosing Option 1 lets you shrink Sections 4 and 6).
- A clear answer unlocks the intro rewrite (Issue 08) and the motivation fix (Issue 02).

After the framing question, the next-biggest decisions are:
- Commit to an empirical expansion (Issues 10, 11, 21) — which tests are we running?
- What goes to the appendix (Issues 06, 07) and what stays?

Low-effort items (Issues 13, 18, 19, 20, 22, 23, 24, 25) can be bundled into a single editing sprint and largely don't need meeting discussion.

## Next journal

Editor suggested AEJ:Applied, Quantitative Economics, or RESTUD. Framing choice (Issue 03) should weigh where each option lands best — a job-design reframe plays to QE/RESTUD; an AI-adoption reframe plays to AEJ:Applied.
