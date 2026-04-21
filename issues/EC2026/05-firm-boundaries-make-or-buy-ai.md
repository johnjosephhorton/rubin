# Issue EC 05 — Firm boundaries and the make-or-buy decision for AI services

## Raised by
**Review #2016B (Major Comment 2).**

> "One of the paper's most intriguing implications but largely underdeveloped concerns how AI reshapes firm boundaries. The model implicitly treats AI deployment as an internal capability choice, but in practice, firms increasingly access AI through markets (API calls, SaaS platforms, outsourced AI services)."
>
> "Unlike previous automation waves (robotics, ERP systems) that required firm-specific customization and capital investment, modern AI increasingly operates as a platform technology with non-negligible inference costs. AI deployment, thus, may favor vertical disintegration: firms could specialize in supervising/validating AI outputs (augmented steps) while outsourcing the AI execution itself."
>
> "This parallels the modularity literature's prediction that standardized interfaces enable vertical disintegration."
>
> Recommendation: "The authors might benefit from explicitly modeling the make-or-buy decision for AI execution. Formally, allow the firm to choose whether AI chains are executed internally (incurring fixed AI management costs but zero marginal hand-offs) or externally (incurring per-transaction costs but enabling finer task decomposition)."

## Why this matters
The firm-boundary implications of AI are distinctive: unlike previous general-purpose technologies, AI is primarily consumed as an API. A firm running 5 different AI-augmented workflows likely doesn't build its own AI — it subscribes. This has direct theoretical consequences:

- **Outsourced AI chains eliminate internal handoff costs** but introduce per-call/per-transaction fees.
- **Verification becomes the new internal competence**: firms specialize in "evaluating AI outputs" rather than in "doing the work the AI does."
- **The specialization–coordination tradeoff shifts**: narrower specialist jobs (reviewers of external AI output) become more viable.

The reviewer explicitly invokes Baldwin & Clark (2000) — which connects naturally to the modularity literature references already suggested by Ezra Zuckerman (see QJE Issue 26 / Baldwin, Henderson-Clark, Novak-Wernerfelt).

## Verification against source
- The model's handoff cost `\handofftime{i}` is defined only between tasks *within* a job, i.e., internal handoffs.
- No firm-boundary friction in the model.
- The firm is a monolith: all steps either go through a human employee or the firm's own AI.

## Options to address

1. **Binary outsourcing choice.** Each AI chain can be executed (i) internally, incurring zero per-atom cost but a fixed "AI management" cost per chain; or (ii) externally via a provider, incurring a per-atom transaction cost `κ_ext` and enabling arbitrary task decomposition. Firms optimize over this plus deployment and job design. Tractable extension.
2. **Platform-vs-integrated firm typology.** Define two firm archetypes: "integrated firms" (own AI, long internal chains) and "platform-consuming firms" (outsource AI, specialize in verification). Derive conditions under which each dominates. Connects to empirical evidence on enterprise AI adoption.
3. **Explicit modularity section.** Add a subsection drawing the formal parallel to Baldwin-Clark and Novak-Wernerfelt. Position the paper as the labor-side complement to the product-architecture side of the modularity literature. Engages the reviewer's explicit suggestion and the Ezra-suggested literature in one move.
4. **Acknowledge + future work.** Add a paragraph flagging the firm-boundary implications as a natural extension and cite Baldwin-Clark. Weakest response.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Binary outsource choice | Medium (2–3 weeks; new proposition + discussion) | **High** — directly implements what reviewer asked for |
| 2. Platform-vs-integrated | Medium-high (3–4 weeks; typology + derivations) | High — clean predictions for empirical testing in follow-up |
| 3. Modularity parallel section | Low (2–3 days + careful writing) | **Very high** per-cost — engages reviewer, Ezra's refs, and broader literature simultaneously |
| 4. Acknowledge in discussion | Low (½ day) | Low for revision but possibly sufficient for EC's conference format |

**Recommendation:** Option 3 is the cheapest high-impact move and pairs directly with the strategy/innovation references already flagged. Option 1 if there's bandwidth to extend the formal model; the reviewer's proposed structure is tractable.
