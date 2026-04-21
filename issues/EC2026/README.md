# EC 2026 Revision Issues — Index

Triage of referee feedback from the four EC'26 program-committee reviewers on paper #2016 *"Chaining Tasks, Redefining Work: A Theory of AI Automation"*. Source: [referee_responses.md](referee_responses.md).

Each issue file below:
- Quotes the reviewer raising the concern (verbatim)
- Reports verification against the current `writeup/` source (file:line where applicable)
- Lists 2–5 ways to address it
- Gives an effort vs. payoff estimate and a recommendation

## Tone summary

EC reviewers were more favorable than QJE overall:
- **R#2016A** — positive, with specific clarification questions about cost structure and market conditions
- **R#2016B** — supportive of the framework, detailed and constructive; most substantial reviewer; brings in Baldwin & Clark (2000) modularity parallels that connect naturally to existing QJE Issue 26
- **R#2016C** — enthusiastically favorable ("interesting and thought-provoking… intuitive, economically meaningful, sheds light"), one reservation about the fragmentation index's role
- **R#2016D** — supportive of the framing, one sharp foundational critique (verification cost scaling) and a call for external validation of GPT orderings

## Issues by theme

### A. Cost structure and accounting (R#2016A)

| # | Issue | Effort | Payoff |
|---|---|---|---|
| [01](01-ai-usage-cost.md) | AI usage cost is zero in the model | L–M | H |
| [02](02-worker-market-conditions.md) | Worker side of the market is unmodeled | M | H |
| [03](03-wage-during-ai-wait.md) | Does the firm pay wages while waiting for AI? | L–M | M |

### B. Scope of the framework (R#2016B major comments)

| # | Issue | Effort | Payoff |
|---|---|---|---|
| [04](04-endogenous-workflow-redesign.md) | Task sequences are exogenous; should allow workflow redesign | M–H | **H** |
| [05](05-firm-boundaries-make-or-buy-ai.md) | Firm boundaries / make-or-buy for AI services (Baldwin & Clark parallel) | L–M | **H** |
| [06](06-full-automation-limit-trajectory.md) | AI trajectory and the full-automation limit as α→1 | L–M | H |

### C. Exposition & empirical connection (R#2016B minor, R#2016A, R#2016C)

| # | Issue | Effort | Payoff |
|---|---|---|---|
| [07](07-exposition-notation-load.md) | Notation load; suggest summary table, worked example, move Prop 1 | L | H |
| [08](08-skill-predictions-observables.md) | Connect skill predictions to measurable outcomes (wages, education) | M–H | H |
| [09](09-fragmentation-index-role.md) | Fragmentation index's role is unclear; empirics feel indirect | L–M | **H** |

### D. Foundational modelling (R#2016D)

| # | Issue | Effort | Payoff |
|---|---|---|---|
| [10](10-verification-cost-chain-length.md) | **Verification cost should scale with chain length** (foundational critique) | M–H | **VH** |
| [11](11-gpt-workflow-external-validation.md) | GPT workflows need external validation beyond prompt-robustness | M | H |

## Reviewer → issue coverage map

| Reviewer | Issues raised |
|---|---|
| R#2016A | 01, 02, 03, 09 |
| R#2016B | 04, 05, 06, 07, 08 |
| R#2016C | 09 |
| R#2016D | 10, 11 |

## Priority map (effort vs payoff)

```
Payoff
  ^
VH|                         10
H |  05 07 09              04 08 11
  |  01 06                  02
M |  03
  |
L |
  +------------------------> Effort
      Low          Med      High
```

## Suggested ordering for the revision

**Highest priority (address first):**
- **Issue 10** (verification cost scaling) — most pointed foundational critique; engagement signals scientific integrity. Either concede + generalize or offer a two-regime model.
- **Issue 05** (firm boundaries / make-or-buy) — rich, constructive suggestion that connects to the Baldwin/Clark modularity literature already on the table from earlier reader feedback. Low-cost modularity discussion + optional formal extension.

**Medium priority (bundle into one editing pass):**
- Issues 07 (notation table, worked example, Prop 1 to appendix), 09 (fragmentation-index bridging), 01 (AI cost parameter), 06 (limiting-case proposition). These are all low-to-medium effort and collectively satisfy ~half the reviewer concerns.

**Higher-effort items to scope carefully:**
- Issue 04 (endogenous workflow redesign) — the ambitious move that most strengthens the paper's long-run relevance.
- Issue 02 (worker market) — bundle with any GE-related revision.
- Issue 08 (skill predictions to observables) — depends on whether the reframe pushes the paper toward labor-economics observables or stays conceptual.
- Issue 11 (external validation of GPT orderings) — can be partly addressed via ATUS; full treatment is a paper on its own.

**Low priority:**
- Issue 03 (wage during AI wait) — one-paragraph justification likely sufficient.

## Relationship to QJE triage

This triage is deliberately standalone (no cross-references to QJE issues). Many EC concerns have thematic overlap with QJE — notably Issues 04, 07, 09 — but the decision to merge or keep separate can be made at the meeting stage. See `../QJE/README.md` for the QJE-side index.
