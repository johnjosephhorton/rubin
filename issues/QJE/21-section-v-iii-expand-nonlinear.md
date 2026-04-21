# Issue 21 — Section V.III: expand on non-linear impacts of AI improvements

## Raised by
**R3 (last "other comments" bullet).**

> "In Section V.III on the non-linear impacts of AI improvements, I was hoping to see more than the short commentary and one example. After all, AI is improving rapidly, and non-linear impacts are the ones we most desperately need help predicting…! E.g.:
> - Quantitatively, to what extent do you think these 'machine handoff costs' explain the 'puzzle' that a large share of tasks seem cheaply automatable but there have been only minor economic impacts so far?
> - On some extrapolation of AI capabilities, when should we start seeing the effects arrive 'steeply'?
> - To what extent do you think that sufficiently advanced AI/automation could have even more of an impact than automation in an atomistic task model, by saving on handoff costs between jobs?"

## Verification against source
- Section V.III on non-linear impacts is in `discussion.tex` (specific line to be confirmed). Currently contains one example (Example V per R3) and short commentary.
- The three sub-questions R3 raises are genuine open problems the model is uniquely positioned to address.

## Options to address

1. **Add a quantitative calibration exercise.** Parametrize verification/handoff costs from data proxies (e.g. time spent reviewing AI output in usage logs, or surveys of AI users). Predict how much output gain has been "left on the table" due to these costs. Maps directly to R3's puzzle question.
2. **Extrapolation / forecasting section.** Under an AI capability trajectory (say, Moore's-law-style success-probability growth), trace out the predicted time path of chain length, execution share, job reorganization. Gives concrete predictions for "when does adoption accelerate".
3. **Handoff-cost-saving mechanism.** Explicitly model the case where AI saves handoff costs between jobs (not just within). Show this as an independent mechanism that can amplify adoption effects at the frontier.
4. **All three as a new subsection** titled "Implications for the diffusion of AI".

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Calibration | High (2–4 weeks; needs cost estimates) | **Very high** — deliverable R3 explicitly wants; publishable headline |
| 2. Extrapolation | Medium (1–2 weeks; numerical model exercise) | High — policy-relevant predictions |
| 3. Between-jobs handoff | Medium (1 week theory) | High — extends model's reach |
| 4. Bundle | High | Very high — single big contribution for the revision |

**Recommendation:** Option 4 is the ambitious play — it directly targets what R3 thinks is most interesting ("non-linear impacts are the ones we most desperately need help predicting"). If bandwidth is constrained, Option 2 alone is the best ROI (it leverages the existing model without new data).
