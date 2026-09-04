# E1-or-E2 exposure switch

Regeneration of every paper exhibit that had implemented AI exposure as **E1 only**,
under the definition the paper actually declares, **E1 or E2**.

Landed in `b2d11a8`. Recorded as **M2** in `writeup/REVIEW_issues_2026-09-04.md`.

## The problem

The paper states its exposure rule twice, at `7_empirics.tex:25` and
`SA_A_sample_construction.tex:16`: a task counts as AI-exposed when it carries an
Eloundou **E1 or E2** label. Table 3 followed that rule. The Prediction #2 pipeline
did not: `onet_antrhopicIndex_execTypeVaryingDWA.ipynb` set

```python
merged_data['is_exposed'] = merged_data['human_labels'].isin(['E1']).astype(int)
```

which coded **3,109 of the 4,632 E1-or-E2 tasks (67.1%)** as unexposed in a control the
table note describes as "the AI exposure status of task k".

## What changed

| Exhibit | Result |
|---|---|
| Table 2 (p. 38) | 8 of 24 coefficient cells; 3 stars; N identical in all six columns |
| Figure OA.A.1 (a)-(d) | all 16 observed labels shift; placebo nulls redrawn |
| Tables SA.B.1-SA.B.3 | 0 sign flips over 72 coefficients, 1 star, largest move 0.0064 |
| Figure SA.D.1(c) | split moves onto `human_aiExposure_fraction` |
| SA.D.4, SA.E.3, SA.E.4 | regenerated; SA.E.4 placebo nulls redrawn |
| SA.A statistic | 605 (69%) -> **809 (93%)** |
| **Table 3** | **byte-identical (control)** |

Table 2's immediate-neighbour effects **fall** from 0.12 to 0.11. The E1-only control had
been leaving AI-execution level effects in the residual for the neighbour dummies to
absorb, so the corrected estimates are more conservative, not less.

## Why Table 3 is the control

Table 3 was already estimated on E1-or-E2, so a faithful pipeline had to return it
unchanged. It did: md5 identical, `diff` exit 0, 0 of 42 cells. That is the evidence the
regeneration machinery is sound and the movement in the other exhibits is real rather
than pipeline drift.

## Verification

Every regenerated number was checked against two independent prior recomputations at full
double precision (max |dAME| = 0.0, max |dSE| = 0.0). Each E1-only rerun reproduced its
published `.tex` byte-for-byte before any E1-or-E2 number was reported, so the "before"
side is the published file rather than a reconstruction. `BUILD_LOG.txt` carries the
per-artifact provenance; `diffs/*.txt` carry the per-cell changes.

## Deviations from the published pipeline

1. **A pre-existing quirk was reproduced, not fixed.** Draw `i = 0` of the 1,000-value
   placebo histogram is the *observed* estimate, not a reshuffle, because cell 19 loads the
   file cell 18 wrote from unshuffled data. Fixing it would have been a second change.
   Consequence: tail shares sit 0.001 above a version that drops `i = 0`. **Still open.**
2. **Bootstrap speed only.** Frames are assembled with a pre-indexed `.take()` instead of a
   per-cluster `concat`. The RNG stream is untouched, validated by the byte-identical control.
3. **Figure x-limits** move from +/-0.1587 to +/-0.1592. Data-derived from `max|value|`, not
   a styling choice; nothing is clipped under either mask.

## Deliberately not switched

- `onet_neighborAI_E1E2exposureControls.ipynb` keeps its E1-only mask. The E1 versus
  E1-or-E2 contrast is the purpose of that notebook.
- `onet_neighborAI_excludeSOC25.ipynb` and `onet_neighborAI_sequenceability.ipynb` write to
  plot directories the paper does not reference. Flipping a mask without regenerating its
  output would recreate exactly the code-versus-exhibit mismatch this work removed.

## Files

- `scripts/` estimation and rendering, transcribed out of the notebooks so nothing ran in
  place. `scripts/repro_E1only/` reproduces the published E1-only side as a fixture.
- `E1E2_PREVIEW.tex` source of the 44-page side-by-side review document.

Present in this folder but **not tracked**, because the repo ignores `*txt` globally
(`.gitignore:35`):

- `diffs/*.txt` per-cell before/after for each exhibit group.
- `BUILD_LOG.txt` per-artifact log, including the checks that the repo stayed untouched.

The numbers that matter are summarised above and in the review file, so the untracked logs
are a convenience rather than the record. Track them with `git add -f` if that changes.

The rendered PDF, the before/after PNG pairs and the intermediate CSVs were left out of
version control: 69 MB, and the repo does not track figures or data. They stayed in the
untracked staging folder `writeup/_e1e2_preview/`, which is also the rollback path for the
figures, since `writeup/plots/` is gitignored and git cannot restore them.
