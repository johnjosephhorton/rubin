# Issue 20 — Figure IX is missing panel (c) "Share of Exposed Tasks Executed on Fragmentation Index"

## Raised by
**R3 (second substantive bullet).**

> "Figure IX: This figure seems incomplete with a panel (c): *Share of Exposed Tasks Executed* on *Fragmentation Index*. Indeed to my mind this would be much more relevant than (a) or (b)."

## Verification against source
- Figure IX (9th figure in main body) = `empirics.tex:233`, label `fig:fragmentation_index_regression`.
- Verified: only panels (a) and (b) currently exist. Panel (c) is absent. **R3 is correct.**
- The two existing panels likely plot exposure vs. fragmentation and execution vs. fragmentation separately. Panel (c) would show the *ratio* (share of exposed tasks that are executed) vs. fragmentation — directly the mechanism the paper predicts.

## Options to address

1. **Add panel (c).** Compute Share_Executed = (tasks with AI execution) / (tasks with AI exposure) per occupation; plot against fragmentation index. If the fragmentation-mechanism holds, slope should be negative. This is the mechanism test R3 is explicitly asking for.
2. **Minimal analog** — instead of adding a third panel, add the Share_Executed vs. fragmentation scatter as a small inset or supplementary figure if the main figure is space-constrained.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Add panel (c) | Low (1 day: data aggregation + plot + caption) | **High** — direct reviewer ask; panel shows the mechanism visually |
| 2. Supplementary only | Low (½ day) | Medium-high — still satisfies R3 but less prominent |

**Recommendation:** Option 1. R3 explicitly said this panel "would be much more relevant than (a) or (b)" — taking the suggestion seriously signals engagement. The computation is cheap.
