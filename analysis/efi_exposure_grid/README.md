# Fragmentation regression: the exposure-definition / control-set / fixed-effect grid

Answers one question: **does the fragmentation result depend on how AI-ability is defined, on
which count is controlled for, or on the corpus?** 54 estimated cells.

| dimension | levels |
|---|---|
| corpus | O\*NET main prompt (N=872) · O\*NET 10 alternative prompt orderings (N=785 to 868) · APQC PCF process groups (N=525) |
| exposure and EFI | **E1 only** · **E1 or E2**. Both the regressor and the index are built on the same label set in each case, so no exposure level leaks into the fragmentation coefficient |
| controls | `+ number of AI-able steps` (the draft) · `+ number of steps in the workflow` · none |
| fixed effects | none · SOC major · SOC minor  (O\*NET) — none · PCF category · framework (APQC) |

Dependent variable is the share of steps AI-executed. All variables z-scored within the
estimation sample. Standard errors clustered on O\*NET-SOC code for O\*NET, HC1 for APQC,
matching the paper.

**Unlabelled O\*NET tasks (789 rows, 4.4%) are kept in place and coded not-AI-able.** They are
part of the sequence the GPT orderings were generated over, so removing them would change the
adjacency structure the index is computed on.

## What the grid shows

1. **The control set does not matter.** Holding corpus, exposure definition and fixed effects
   fixed, the fragmentation coefficient moves by at most 0.040 across the three control sets and
   by 0.013 at the median. Swapping the AI-able-step count for workflow length, or dropping the
   control entirely, changes no verdict anywhere.
2. **The exposure definition is what matters, and only on APQC.** On the PCF corpus the
   coefficient is -0.24 to -0.30 and significant under E1-or-E2, but -0.05 to -0.08 and null
   under E1 only. On O\*NET both definitions are null.
3. **O\*NET is null in all 9 combinations under both definitions**, on the main prompt and on all
   10 alternative orderings (0 of 10 significant with the predicted sign in every one of the 18
   alternative-prompt cells). Under E1 only and no fixed effects the O\*NET point estimates are
   positive, the sign the model rules out, though never significantly.
4. **One caveat on the APQC result.** Under PCF category fixed effects it clears 10% only with
   the AI-able-step control (-0.250) and falls just short without it (-0.237, -0.240). The other
   two APQC fixed-effect columns hold up under all three control sets.

## Files

| | |
|---|---|
| `efi_exposure_grid.py` | estimates every cell, writes the tidy CSV |
| `report.py` | renders the readable tables, and validates against the draft's Table 3 |
| `figure.py` | the coefficient plot |
| `data/computed_objects/efi_exposure_grid/efi_exposure_grid.csv` | one row per cell |
| `data/computed_objects/efi_exposure_grid/grid_tables.txt` | the rendered tables |
| `writeup/plots/efi_exposure_grid/efi_exposure_grid.png` | the coefficient plot, 300 dpi |

## Validation

`report.py` prints the E1-or-E2 / `+ # AI-able steps` / main-prompt row first. It reproduces the
table now in the draft exactly: EFI -0.007 / -0.086 / -0.040, exposure +0.494 / +0.479 / +0.390,
N=872.

## Running

```bash
python analysis/efi_exposure_grid/efi_exposure_grid.py
python analysis/efi_exposure_grid/report.py
python analysis/efi_exposure_grid/figure.py
```
