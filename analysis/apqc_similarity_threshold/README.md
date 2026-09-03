# Sensitivity of the PCF results to the label-transfer similarity floor

AI exposure and execution are measured on O\*NET tasks and have no counterpart on PCF elements, so
each PCF step is matched to its nearest O\*NET task by embedding similarity and a label is carried
across only when the match clears a floor. The draft sets that floor at **0.71**. This sweeps it over 0.65 to 0.75 in steps of 0.01 and reports what moves.

**The 525 process groups are fixed across every threshold.** The five-step minimum is applied to the
raw step count, not to the labels, so raising the floor only relabels steps as unexposed and
unexecuted. The sweep is therefore a clean check on the labelling, not on sample selection.

## What holds

| | |
|---|---|
| Average AI chain length exceeds its within-group reshuffle null | **11 of 11** thresholds |
| Fragmentation coefficient carries the predicted negative sign | **11 of 11**, under all three fixed-effect specifications |
| Fragmentation coefficient significant at 5% | 8 of 11 without fixed effects |

## What to be aware of

The fragmentation coefficient is **not monotone** in the floor. It runs from -0.26 at 0.65 down to
-0.63 at 0.69, back to -0.11 at 0.72, and is -0.35 at the chosen 0.71. Significance therefore
switches on and off across neighbouring thresholds (0.72 null, 0.73 significant, 0.74 null).

That pattern is noise crossing a line rather than a discontinuity: the 0.72 and 0.73 point estimates
each sit inside the other's 95% confidence interval. Standard errors widen from 0.10 to 0.21 as the
floor rises and the labels thin out, so the high-floor cells lose significance through power rather
than through sign.

**The two coefficients move against each other**, which is what picks the floor. Where the
fragmentation coefficient is deepest (0.68 to 0.70) the exposure coefficient is at its weakest, and
it is essentially zero at 0.69 (+0.007). Where exposure is strongest (0.72 and above) fragmentation
weakens. **The floor is chosen on label density, not on the estimates.** Raising it thins both the exposure
and the execution content of the corpus, and exposure is the binding one since it is the more common
of the two labels. At 0.71 the corpus retains 12.2% of steps as AI-exposed and 4.0% as AI-executed;
at 0.72 exposure falls to 9.9%, and it keeps falling above that. 0.71 is the highest floor that
holds exposure at or above 10%.

Worth knowing, though it is not the selection criterion: 0.65, 0.66 and 0.71 are the only floors
where both coefficients clear 5% in all three fixed-effect specifications, and 0.71's fragmentation
estimate is the fifth largest of eleven against a median of -0.33.

The mechanism is the decomposition: the index is 1 - k/m + r/m, so raising the floor thins the
exposure share and shifts weight between the level term and the arrangement term. The two
coefficients are reading the same variation from opposite ends.

Chain length weakens at the top of the range: z falls from about 6 at 0.71 to 1.9 and 2.0 at 0.74
and 0.75, where only 5% to 6% of steps retain any label.

## Files

| | |
|---|---|
| `apqc_similarity_threshold.py` | runs the sweep, prints both tables, writes the CSV |
| `figure.py` | the two-panel sensitivity plot |
| `data/computed_objects/apqc_similarity_threshold/apqc_similarity_threshold.csv` | one row per threshold |
| `writeup/plots/apqc_similarity_threshold/apqc_similarity_threshold.png` | the figure, 300 dpi |

## Calibration

At 0.73, the floor the draft used before this sweep, the script reproduces the previously published
figures exactly: 1,364 steps clearing the floor (10.1% of the corpus), 8.1% of steps AI-exposed,
2.6% AI-executed, and an average chain of 1.14 against a reshuffle null of 1.07.

## Running

```bash
python analysis/apqc_similarity_threshold/apqc_similarity_threshold.py
python analysis/apqc_similarity_threshold/figure.py
```

## Both coefficients across the grid

`both_coefficients_by_floor.csv` in the computed-objects folder carries the exposure and
fragmentation coefficients with standard errors for all eleven floors and all three fixed-effect
specifications. That is the table to consult before moving the floor again.
