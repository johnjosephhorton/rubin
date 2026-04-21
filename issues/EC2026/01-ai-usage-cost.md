# Issue EC 01 — AI use can be expensive; model treats it as free

## Raised by
**Review #2016A.**

> "The paper benefits from discussing why only the steps performed by humans incur a cost, since AI use can be expensive for firms as well."

## Why this matters
The current cost accounting attributes zero direct cost to AI execution — the only costs are human skill × time plus handoffs. In reality, frontier-model inference has non-trivial per-call or per-token pricing (especially for long-context or agentic use). If AI cost scales with chain length (more atoms → more tokens) or with task complexity, it directly competes with the verification-savings benefit that drives the chaining mechanism. Ignoring it is a modelling simplification that needs a defense.

## Verification against source
- `aggregation.tex:32` wage equation: `(v_L + Σ skill_i) × (Σ time_i)` — only human-side costs enter.
- No `\aiCost`-style term in the model macros ([main.tex:104–128](writeup/main.tex:104)).
- AI quality is a parameter (via `p_i`), but AI compute has no price.

## Options to address

1. **Justify in-text and move on.** Argue that AI inference cost is small relative to human wages at current price points and will fall further, so treating it as zero is a reasonable approximation for the theoretical results. Cite relevant pricing data.
2. **Add a per-atom AI cost parameter `κ`.** The firm's objective becomes `wage_bill + κ × (atoms executed by AI)`. Re-derive the optimal chain length: the extra marginal cost `κ` pushes against chain extension in the same direction as failure probability, but linearly rather than multiplicatively. Most propositions survive with minor adjustments.
3. **Two-cost structure.** Augmented steps (AI with verification) cost `κ_aug` per call; automated steps (no verification) cost `κ_auto`. If `κ_aug > κ_auto` (because verification consumes more AI calls or human time), this shifts the chain-length calculus. Useful for tying back to firm-boundary arguments (see EC Issue 05).
4. **AI cost proportional to human-time-equivalent.** `κ_i = γ × τ_i` where `γ` is a conversion factor. Economically interpretable as "AI charges a fraction of the human wage for the same work" and admits a clean comparative static in `γ`.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Justify only | Trivial (½ day) | Low — doesn't answer the reviewer's theoretical concern |
| 2. Per-atom κ | Medium (2–3 days; reproves main results) | **High** — directly addresses the reviewer; makes the wage equation more defensible |
| 3. Two-cost structure | Medium-high (1 week) | High — connects to make-or-buy (EC Issue 05) |
| 4. Proportional | Medium (2–3 days) | High — cleanest comparative statics |

**Recommendation:** Option 2 as the minimum response. Option 3 if the revision also tackles firm boundaries (EC Issue 05).
