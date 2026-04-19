# Issue 01 — Two mechanisms are conflated, not cleanly separated

## Raised by
**R1, §1 "Clarifying and Separating the Two Mechanisms"** (most forcefully). Also implicit in the editor's letter ("model to be more complex and difficult to follow than necessary").

> "The results appear to be driven by two distinct mechanisms that operate in parallel. Some results arise from one mechanism, while others rely primarily on the other… considerable effort is required to disentangle the underlying logic of the analysis."
>
> Mechanism 1: AI chaining → reverses comparative advantage, drives fragmentation results, produces J-curve threshold effects. **This is what the empirics test.**
>
> Mechanism 2: specialization–coordination tradeoff in job design → drives upskilling/deskilling and generalist-vs-specialist results (visible in Examples III and IV even with chaining *shut down*). **This is not empirically tested.**

## Verification against source
- `discussion.tex:131` — "technically possible in our model for AI to increase the amount of specialization" — language suggests mechanism 2 is treated as a corollary, not as a pillar.
- `model.tex` (Section 3) and `optimization.tex` (Section 4) interleave both mechanisms without a decomposition result that isolates them.
- Examples III and IV (referenced by R1) do shut down chaining and nonetheless generate job-design results, which confirms R1's observation.

## Options to address

1. **Full separation** — rewrite the theory as two nested models: (M1) chaining-only (no endogenous job design), (M2) chaining + job design. State each result under the minimal mechanism that produces it. Would require restructuring Sections 3–5.
2. **Explicit tagging** — keep the unified model but tag every proposition/example with "(uses chaining)" / "(uses job-design tradeoff)" / "(uses both)". Add a summary box at the start of Section 3 listing the two mechanisms and which results depend on which.
3. **Choose a side** — following R4's advice, pick one mechanism as the primary contribution and move the other to an appendix subsection. R1 and R4 both hint this is the cleanest path (and it naturally slims the paper, addressing R1.2, R2.1, R4.4).

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Full separation | High (6–10 days rewrite) | High — restores clarity, makes model decomposable |
| 2. Explicit tagging | Low (1–2 days) | Medium — cosmetic clarity, doesn't address the deeper "which one matters?" question |
| 3. Pick a side | Medium (3–5 days) | **Highest** — also resolves complexity and empirical-testing-gap concerns in one move |

**Recommendation:** Option 3 is the natural centerpiece of the revision. It connects directly to Issue 03 (R4's adoption-vs-job-design reframe) and automatically dampens complaints about length (Issues 05, 09) and the macro extension (Issue 07).
