# Revision strategy — meeting prep

*Chaining Tasks, Redefining Work: A Theory of AI Automation*
Demirer, Horton, Immorlica, Lucier & Shahidi

Cross-venue synthesis based on the QJE reject-with-suggestions and the four EC'26 reviews. Organized by **decisions to resolve** and **handoffs** — rather than by referee — so the meeting can move fast.

---

## 1. The four decisions that cascade to everything else

These gate many of the smaller issues. Nothing downstream is worth committing to until these are settled.

### D1 — Primary framing of the paper

> *Theory of AI chaining with labor-market implications, or theory of job design with AI as the driving shock?*

QJE R4 explicitly framed this as a choice (QJE [Issue 03](QJE/03-empirical-reframe-adoption-or-jobdesign.md)):
- **Option A — AI adoption.** Builds on current empirical setup (fragmentation regression, adjacency regressions). Low-risk path.
- **Option B — Job design.** More novel, bigger payoff, requires new data (e.g. Lightcast vacancies).

EC R#2016B implicitly echoes this: their "endogenous workflow redesign" critique (Major 1) and their Baldwin-Clark modularity framing would be far more productive under Option B.

**Why this matters for the meeting:** Once we pick, ~10 QJE issues and 3-4 EC issues collapse into a coherent agenda. E.g., Option A makes QJE §VI (CES aggregation) unnecessary in the main text; Option B turns it into centerpiece material.

### D2 — Model simplification path

> *Do we keep the current six-per-step parameterization, or move to the atomic decomposition we've been drafting?*

Three reviewers (QJE R1, QJE R4, EC R#2016B) independently flagged parameter load. We've sketched an atomic-decomposition approach: normalize time, keep skill heterogeneity, binary or continuous atom skill types (Options A/B/C under time normalization).

Open sub-question: **normalize time only, or also restrict skill to binary atom types?** I have the analysis ready but the trade-off between teaching simplicity (binary) and standard economic form (continuous) is a judgment call that needs John's read.

### D3 — The chain-length verification cost modification

> *Do we adopt `t^{AI, new}_i = t^{AI, old}_i + (n − 1) · t^{Chain}` as proposed?*

This jointly answers EC R#2016A's wage-during-AI-wait question and EC R#2016D's verification-scaling critique (see EC [Issue 10](EC2026/10-verification-cost-chain-length.md)). DP complexity expected to survive (O(m²)); fragmentation-index bound changes form but retains proportional structure.

**Why it matters:** if yes, we need to re-derive Proposition 6's FI bound, update Section 3 (chain definition), and possibly revise Example 5. Non-trivial but self-contained. If no, we fall back to the "success-probability channel already captures the concern" response, which is analytically cleaner but may not fully satisfy the reviewer.

### D4 — Empirical expansion scope

> *Do we add new empirical tests, and if so which?*

Candidates flagged across both venues:
- **Same-task across occupations** (QJE R4) — cheap, within existing data. **Likely yes.**
- **External validation of GPT orderings against ATUS time-use** (EC R#2016D, QJE R2). Medium cost, high signal.
- **Historical automation analogs** — probably not this round.
- **Lightcast vacancy data for job-design predictions** — only if D1 resolves toward Option B.

---

## 2. Cross-venue themes (both review panels agree)

These are low-risk wins: if two independent panels flag the same thing, addressing it cheaply signals responsiveness and lands real content.

### Theme A — Model notation load / exposition density
- QJE R1 ([Issue 05](QJE/05-model-complexity-parameters.md)): 19 params for 3 steps
- EC R#2016B minor 4: "considerable notation accumulates quickly"
- **Fix bundle:** notation summary table at top of §3; move Proposition 1's proof to appendix; bring a worked example into the body. 1-2 day job.

### Theme B — Exogenous workflow / GPT ordering fragility
- QJE R2 ([Issue 10](QJE/10-llm-workflow-ordering-fragility.md)): LLM-imputed ordering is a latent object
- EC R#2016B major 1: should allow endogenous workflow redesign
- EC R#2016D: GPT orderings need external validation benchmarks, not just prompt-robustness
- **Fix bundle:** (i) scope-of-claim paragraph acknowledging what the ordering is and isn't; (ii) external validation against a subset of occupations (ATUS or practitioner interviews for 3–5 occupations); (iii) optional small-permutation extension to the long-run model.

### Theme C — Fragmentation index role / empirical connection
- QJE R1 ([Issue 01](QJE/01-two-mechanisms-separation.md)): two mechanisms conflated
- EC R#2016A: "main takeaway from Section 5?"
- EC R#2016C: "did not fully understand the point of the fragmentation index"
- **Fix bundle:** a "bridging" paragraph making explicit that the index is the observable proxy for a theoretical object the econometrician can't observe directly, provably within constant factors of OPT. Half-day job. We've already drafted part of this in the EC tracker.

### Theme D — Worker-side / institutional assumptions
- QJE R4 ([Issue 14](QJE/14-wage-equation-microfoundation.md)) + ([Issue 16](QJE/16-general-equilibrium-formal-setup.md)): wage microfoundation, GE
- EC R#2016A: worker market, AI cost, wait-wages
- **Fix bundle:** already drafted the EC-side responses (inelastic supply, representative worker, subscription microfoundation for AI). QJE-side (specialist-wages paradox, GE formalization) is meatier. Could be addressed with a combined "institutional scope" paragraph near the model's wage equation.

---

## 3. Venue-specific heavyweights

### EC-only — no QJE counterpart

- **Firm boundaries / make-or-buy for AI** (R#2016B major 2) — explicit Baldwin-Clark parallel; also echoes Ezra Zuckerman's suggestion on QJE ([Issue 26](QJE/26-strategy-innovation-literature.md)). **Potential leverage point**: a single "modularity" section could satisfy both an EC reviewer and a friendly external reader, and reposition the paper against a more distinctive literature.

- **AI trajectory / full-automation limit** (R#2016B major 3) — what remains for humans as α→1? A limiting-case proposition is cheap (2–3 days) and answers something the reviewer specifically asked about. Worth doing.

### QJE-only — concerns EC panels did not raise

- **Two-mechanisms separation** (Issue 01) — QJE R1's central critique. Interacts directly with D1.
- **§VI CES aggregation** (Issue 07) — three QJE reviewers said "move to appendix." EC reviewers didn't comment. **Should do regardless** — frees up ~8 pages of body real estate.
- **§IV DP machinery** (Issue 06) — similar, QJE consensus. Move proofs to appendix.
- **Specialist-wages paradox** (Issue 15) — QJE R4's observation that model implies specialists earn less. Needs either an explicit depth-vs-breadth reinterpretation or a scarcity premium.

---

## 4. What's already drafted (don't rehash)

From the EC tracker, responses are drafted (🟡) for:
- R#2016A: all four feedback points (AI cost, worker market, wait-wages, Section 5 takeaway)
- R#2016C: partial, sharing the Section-5 response above; plus your author note on tone
- R#2016D: the chain-length verification-cost structural proposal

From the QJE triage, deep-dive analyses exist for all 27 issues but responses are not drafted yet — that's the meeting's QJE task.

---

## 5. Key handoffs to raise with collaborator

### Questions that specifically need John's read

1. **D1 framing choice** — the single highest-leverage decision. Currently undecided.
2. **D2 atomic decomposition** — binary vs continuous skill types; my analysis leans binary for teaching clarity, but wants John's judgment.
3. **D3 chain-length verification cost** — the structural modification requires re-deriving Proposition 6's bound; we should decide before investing the time.
4. **Same-task-across-occupations test** — is the data structured to make this a clean test? (needs the empirical pipeline walkthrough)
5. **Next venue** — QJE editor recommended AEJ:Applied, QE, or RESTUD. Choice is downstream of D1 (Option A → AEJ:Applied; Option B → QE/RESTUD).

### Issues to flag even if we don't resolve them this meeting

- **QJE R4 Issue 02** (motivation / "why is this theory needed"): needs a sharp opening puzzle before anything else works.
- **QJE R1 Issue 08** (intro streamline): can't rewrite until D1 is settled.
- **EC R#2016B major 3** (full-automation limit): worth a 2–3 day theoretical exercise but only after simpler items land.

---

## 6. Low-effort bundle to ship regardless

These cost very little, signal responsiveness, and don't depend on any strategic decision:

| Item | Source | Effort |
|---|---|---|
| Eloundou labels terminology fix | QJE R3 [Issue 18](QJE/18-eloundou-labels-terminology.md) | 15 min |
| p.41 fragmentation framing rewrite | QJE R3 [Issue 19](QJE/19-fragmentation-regression-framing.md) | 30 min |
| Figure IX add panel (c) | QJE R3 [Issue 20](QJE/20-figure-ix-add-panel-c.md) | 1 day |
| Notation table at top of §3 | QJE Issue 05 + EC R#2016B | ½ day |
| "Technically" hedge rephrase (p.25) | QJE [Issue 22](QJE/22-technically-possible-rephrase.md) | 15 min |
| Publish GPT task orderings | QJE [Issue 13](QJE/13-publish-task-orderings.md) + EC R#2016D | 1 day |
| Table notes overflow audit | QJE [Issue 24](QJE/24-table-notes-page-overflow.md) | ½ day |
| Notation nitpicks | QJE [Issue 25](QJE/25-notation-nitpicks.md) | 2 hours |
| Section 5 takeaway paragraph | EC R#2016A/C + QJE R1 | 1 hour |

Total: roughly 3–4 days of work, addresses ~10 reviewer items across both venues.

---

## 7. Proposed meeting agenda (60-90 min)

1. **D1 framing** — 20 min. Decide AI-adoption vs job-design. Everything else is downstream.
2. **D2 model simplification** — 15 min. Confirm atomic decomposition and decide Option A/B/C.
3. **D3 chain-length verification cost** — 10 min. Accept/reject the structural proposal.
4. **D4 empirical scope** — 15 min. Pick 1–2 new tests.
5. **Venue** — 10 min. Where next, and what's the timeline?
6. **Low-effort bundle sign-off** — 5 min. Confirm the ship-regardless list.

Skip section-by-section review of individual issues unless a specific one comes up during the four decisions. The issue files are the scratch pad; the decisions drive the sequence.
