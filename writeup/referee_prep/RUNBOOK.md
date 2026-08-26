# Referee-prep runbook (modular, one agent per run)

## Why the first attempt died
The panel fanned out to **55 agents**: 7 finders, then **one verifier per finding (47)**, then a
synthesizer. All 47 verifiers launched at once and the session limit hit with 49 unfinished.
6 of 7 finders had already returned, so their output was recoverable; nothing else was.

## The rebuilt shape
`verify_lens.js` verifies **one lens per invocation using one agent**, batching that lens's
findings into a single prompt. Ceiling: 1 agent, ~1 lens of context, per run. Nothing fans out.

## Files here
| file | contents |
|---|---|
| `VERIFIED_FINDINGS.md` | the 5 findings I checked myself against the code and data. Read this first. |
| `findings_identification.json` | 7 unverified findings, empirical identification lens |
| `findings_measurement.json` | 7 unverified, data credibility lens |
| `findings_mapping.json` | 7 unverified, theory-to-empirics lens |
| `findings_contribution.json` | 10 unverified, contribution and positioning lens |
| `findings_theory-long.json` | 8 unverified, long-run theory and aggregation lens |
| `findings_exposition.json` | 8 unverified, structure and journal-fit lens |
| `verify_lens.js` | the one-lens-at-a-time verifier |

The **theory-short** lens (short-run model rigour, Props 1-2, Appendix A proofs) never returned
and still needs a finder run.

## Suggested order, one per sitting
1. `identification` — overlaps my verified finding 1; highest stakes
2. `mapping` — overlaps my verified finding 4
3. `measurement` — overlaps my verified findings 3 and 5
4. `theory-short` — needs a fresh finder first, nothing recovered
5. `theory-long`
6. `contribution`
7. `exposition`

## To run one
Ask for it by name ("verify the identification lens"). It becomes:

    Workflow({ scriptPath: ".../writeup/referee_prep/verify_lens.js",
               args: { lens: "identification", findings: <contents of findings_identification.json> } })

Save each result next to its input as `verdicts_<lens>.json` before starting the next lens, so a
limit mid-run costs at most one lens.

## Synthesis
Only after the lenses you care about are verified. It reads the `verdicts_*.json` files plus
`VERIFIED_FINDINGS.md` and needs no agents at all.
