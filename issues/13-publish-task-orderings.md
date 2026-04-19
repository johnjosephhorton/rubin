# Issue 13 — Publish occupation-level GPT task orderings for replicability

## Raised by
**R3 (substantive bullet 5).**

> "Since the outputs of ChatGPT are stochastic, please make public the occupation-level task orderings by prompt, so that the results are replicable."

## Verification against source
- The paper calls GPT-5-mini with a prompt to produce orderings; prompt variants are in `writeup/appendix-GPTprompts_robustness.tex`.
- The repo does not currently check in the raw GPT responses or the resulting ordered task lists per occupation.
- Plot outputs (`writeup/plots/GPT_task_sequences_overlap_analysis/`) suggest the orderings were generated and used, but the source artifacts are not archived in the repo.

## Options to address

1. **Check in a `data/task_orderings/` directory** with one JSON per (prompt-variant × occupation). Version under git or Zenodo.
2. **Add a reproducibility appendix section** listing the exact API calls, seeds (if any), model version, timestamp, and a link to the dataset.
3. **Publish the prompt + code** to regenerate the orderings. Stochastic outputs mean re-running won't reproduce byte-for-byte, so this is secondary to Option 1.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Check in orderings | Low (½–1 day, depending on file size) | **High** — direct response to the reviewer ask; any future referee will also want this |
| 2. Repro appendix | Low (½ day) | Medium — necessary but not sufficient |
| 3. Publish code | Low if code already exists (1 day) | Medium — standard expectation for this type of paper |

**Recommendation:** Do all three; together they take ≤2 days. This is pure gain for reviewer trust. Upload large JSON blobs to Zenodo if they exceed reasonable git size.
