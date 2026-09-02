# The empirical fragmentation index under a matched exposure definition

Status as of 2026-09-01. **Nothing in the draft has been changed.** Every published table
and figure is still the old specification. This file records what the matched specification
does to each result, what prose it invalidates, and what is still undecided.

Companion material:

| What | Where |
|---|---|
| Scripts (18, all runnable) | `analysis/efi_matched_exposure/` and its `README.md` |
| Generated data and matched `.tex` tables | `data/computed_objects/efi_matched_exposure/` |
| Matched versions of the two figures | `writeup/plots/efi_matched_exposure/` |
| Report, PDF | `writeup/EFI_matched_specification_report.pdf` |
| Report, web version | Artifact "The Fragmentation Null" |

---

## 1. The problem

EFI Definition 1 counts a step as AI-able when its Eloundou label is E1 **or** E2, and merges
consecutive AI-able steps into one task. The regression of Equation (11) controls for the E1
share **alone**.

For a workflow of `m` steps with `k` AI-able steps forming `r` maximal runs:

```
EFI = (m - k + r)/m = 1 - k/m + r/m
```

an exposure **level** term `-k/m` entering with coefficient exactly `-1`, plus an
**arrangement** term `r/m`. This is an identity, not an approximation: it reproduces the
notebook-constructed index to exact float equality (max deviation `0.000e+00`) on all 28
estimation samples in the paper, and regressing the EFI on `k/m` and `r/m` returns
R-squared = 1.0000000000 with slopes of exactly -1 and +1, so there is no third component.

Because the E1∪E2 level is never absorbed, the EFI coefficient loads on `-k/m` rather than on
`r/m`. On the main sample `corr(EFI, E1∪E2 share) = -0.954` and R-squared is 0.909, while the
E1 share the paper controls for absorbs only R-squared 0.496 of it.

**The matched specification** puts the E1∪E2 share on the right-hand side, so exposure and the
index are measured on the same steps. Everything else, including the `num_E1E2_tasks`
step-count control, is unchanged.

**Decision taken 2026-09-01 (Peyman): the step-count control stays in the headline.** The
no-control variant is reported only as a labelled robustness row. As it turns out this choice
is not load-bearing; see section 6.

## 2. Main text, Table 2

Standardized coefficients, SE clustered on O\*NET-SOC code, N = 872. Produced by
`analysis/efi_matched_exposure/table2_and_SAB.py`, which regenerates the published `.tex`
byte-for-byte before changing anything.

| | (1) no FE | (2) Major FE | (3) Minor FE |
|---|---|---|---|
| EFI, published | -0.261 (0.073)\*\*\* | -0.380 (0.064)\*\*\* | -0.283 (0.071)\*\*\* |
| **EFI, matched** | **-0.007 (0.101)** | **-0.086 (0.086)** | **-0.040 (0.093)** |
| matched 95% CI | [-0.204, +0.191] | [-0.254, +0.082] | [-0.222, +0.143] |
| Exposure, published (E1) | +0.241 (0.046)\*\*\* | +0.109 (0.042)\*\*\* | +0.092 (0.047)\* |
| **Exposure, matched (E1∪E2)** | **+0.494 (0.102)\*\*\*** | **+0.479 (0.094)\*\*\*** | **+0.390 (0.093)\*\*\*** |
| R-squared, published | 0.357 | 0.629 | 0.713 |
| R-squared, matched | 0.347 | 0.636 | 0.717 |

In levels the fragmentation effect moves from -4.13 / -6.01 / -4.48 percentage points of the
AI-execution share per SD to -0.11 / -1.36 / -0.63.

### Where the published coefficient came from

Entering the EFI, the E1 share and the E1∪E2 share together: the EFI collapses to
-0.010 / -0.080 / -0.044 (p = 0.92 / 0.35 / 0.63) while the E1∪E2 share is
+0.360 / +0.438 / +0.355, significant at 1% in every column. An exact omitted-variable
identity attributes **96.3% / 79.0% / 84.4%** of the published EFI coefficient to the omitted
E1∪E2 level.

### What the matched design can and cannot rule out

80%-power MDE is 0.282 / 0.240 / 0.261 SD units, i.e. 4.47 / 3.80 / 4.13 percentage points,
confirmed by a 4,000-draw wild residual bootstrap at 0.284 / 0.239 / 0.247 with correct size.

- Underpowered for a modest effect: anything up to about a 3-point true reduction would
  routinely go undetected.
- But the published magnitude is excluded in all three columns (two-sided p of
  0.012 / 0.001 / 0.009 treating the published point as fixed). **State this carefully.** The
  exclusion is comfortable under major-group FE (gap 0.126 SD) and marginal in the other two
  columns (gaps 0.057 and 0.061, both under one percentage point).

The cost of the fix is a 1.7x to 1.9x variance inflation. After conditioning on the E1∪E2
share and the step count, only 30.0% of the index's SD survives under no FE, falling to 23.4%
under minor-group FE. That residual is the entire identifying variation the arrangement
channel has left.

## 3. Appendices

### SA.B, execution-based EFI: not evidence, and not repairable here

The level term of the execution-based index is **bitwise identical** to the dependent
variable (`max |k2/m - ai_fraction| = 0.000e+00`), so the table regresses `y` on
`(1 - y + r2/m)`. Substituting that algebraic equivalent leaves coefficients and R-squared
unchanged to the digit.

Matching the exposure regressor barely moves it: -0.780 / -0.703 / -0.679 becomes
-0.777 / -0.682 / -0.666. Isolating the arrangement term alone **flips the sign to
+0.871 / +0.748 / +0.715**. And 317 of 872 occupations (36.4%) have zero AI-executed steps,
so their index is exactly 1 and their outcome exactly 0 by construction.

This is review issue D31 with a proof attached. It should be raised to Major, because once
the exposure-based leg is null this is the only leg still standing under `7_empirics.tex:144`.

### SA.D, eleven GPT prompt orderings: stable null

| | no FE | Major FE | Minor FE | matched range |
|---|---|---|---|---|
| EFI significant and negative, published | 11 / 11 | 11 / 11 | 11 / 11 | -0.41 to -0.24 |
| EFI significant and negative, matched | 0 / 11 | 0 / 11 | 0 / 11 | -0.15 to +0.06 |
| Exposure significant, published | 11 / 11 | 11 / 11 | **0 / 11** | +0.07 to +0.25 |
| Exposure significant, matched | 11 / 11 | 11 / 11 | **11 / 11** | +0.30 to +0.56 |

Only 2 of 33 matched cells reach even 10%. The null is stable rather than fragile:
between-prompt dispersion of the matched coefficient (SD 0.044 / 0.023 / 0.040) is two to four
times **smaller** than the sampling noise in a single estimate, and all 33 estimates lie inside
the main-prompt confidence interval. 30 of 33 point estimates keep the predicted negative sign.

Note the exposure rows. Under the published spec, minor-FE exposure cleared 5% in **zero** of
eleven orderings and the appendix's "positive and statistically different from zero
throughout" was resting on 90% bars. Matched, it clears 1% in all eleven.

### SA.E, twelve frequency cuts: stable null

The unpruned row goes from -0.260 / -0.378 / -0.282, all at 1%, to -0.013 / -0.090 / -0.048.
Across all 39 cells of the logic x threshold x FE grid the matched coefficient is **never
significant at 5% in either direction**. Median absolute standardized coefficient 0.055 over the 39 cells, 0.059 over the 36 pruned ones.

The three largest cuts (all tasks N=871, Daily+ >=20% N=832, Daily+ >=35% N=725) have SE of
0.085 to 0.101, so these are precise zeros, not underpowered ones. The sparser cuts genuinely
cannot rule out a moderate effect and the appendix should not claim they do: 23 of the 36
pruned cells have 95% intervals that still admit -0.20 (26 of 39 including the unpruned row).

Sign pattern goes from 33 of 39 negative to 16 of 39, a coin flip. The exposure coefficient
strengthens throughout, roughly doubling in every cut.

### SA.F, APQC documented sequences: survives

N = 525 pooled process groups. Three columns: no FE, PCF Category FE, Framework FE.

| | (1) | (2) | (3) |
|---|---|---|---|
| EFI, published | -0.375 (0.150)\*\* | -0.370 (0.154)\*\* | -0.376 (0.152)\*\* |
| **EFI, matched** | **-0.295 (0.141)\*\*** | **-0.250 (0.142)\*** | **-0.290 (0.144)\*\*** |
| Exposure, published | +0.139 (0.100) | +0.185 (0.094)\*\* | +0.145 (0.101) |
| **Exposure, matched** | **+0.261 (0.110)\*\*** | **+0.356 (0.107)\*\*\*** | **+0.280 (0.112)\*\*** |

Why it survives, and it is not luck. The 0.73 similarity floor leaves the PCF corpus at 8.6%
E1∪E2 density against 43.2% in O\*NET, so the level term stops dominating the index:

| | APQC pooled | O\*NET main |
|---|---|---|
| mean E1∪E2 share | 0.086 | 0.432 |
| R-squared of EFI on the exposure share | 0.58 | 0.91 |
| SD of EFI surviving conditioning | 65% | 30% |
| VIF of the EFI | 2.4 | 11.1 |

The observed coefficient still sits outside the within-group reshuffle null. Note the placebo
z is **seed-dependent** and no seed is set: across three seeds it runs -3.05 to -3.19, with 0
to 3 of 1000 draws below the observed value, so "below all 1000 draws" holds for some seeds
only.

**This inverts the appendix's rhetorical position.** SA.F currently reads its -0.37 against a
main-text range of -0.26 to -0.38. Under the matched specification the external validation
reports a larger and better-identified fragmentation effect than the specification it is
validating. That has to be said out loud.

## 4. No better arrangement statistic recovers the effect

### The definition grid

Crossing three exposure definitions with three EFI definitions and three FE specifications
gives 27 cells, each with its matching step-count control. Fifteen are significantly negative
at 5% (twelve at 1%), and **every one is off-diagonal**, meaning the exposure label set and the EFI label set
disagree. Every matched (diagonal) cell is null or wrong-signed; the E2-on-E2 cell is
significantly **positive** without fixed effects.

### Six alternative statistics, matched, headline control

| Statistic | no FE | Major FE | Minor FE | 80% MDE |
|---|---|---|---|---|
| EFI (reference) | -0.007 | -0.086 | -0.040 | 0.24 - 0.28 |
| `r/m`, runs per step | -0.002 | -0.029 | -0.014 | identical by algebra |
| `r/k`, runs per exposed step | +0.025 | +0.056 | +0.014 | 0.12 - 0.13 |
| adjacent-pair share | -0.022 | -0.043 | -0.018 | 0.08 - 0.09 |
| longest run / `k` | -0.029 | -0.017 | -0.029 | 0.06 - 0.08 |
| permutation z-score | +0.032 | +0.027 | +0.004 | 0.06 - 0.08 |

Not one cell reaches even 10%. The permutation z-score compares each occupation's observed run
count against its own null holding `m` and `k` fixed. It is the cleanest scale-free
arrangement measure available and is **four times better powered** than the EFI, and it comes
out wrong-signed and insignificant. So the matched null is not an artifact of the EFI being a
poor arrangement measure; it is the opposite. The EFI is the only statistic in the set with
enough residual level content to be moved at all.

`r/m` reproduces the EFI's t-statistic and R-squared exactly, because
`span{EFI, k/m} = span{r/m, k/m}`. Entering the level and arrangement terms separately gives
the same arrangement coefficient with a level coefficient of about -0.50.

## 5. What becomes false in the draft

Exhibits regenerate mechanically. These sentences need an authorial decision.

### Sign or significance claims that become false

| Where | Sentence |
|---|---|
| `0_main.tex:75` | abstract clause (2), "dispersion of AI-exposed steps predicts lower AI execution at the workflow level" |
| `1_introduction.tex:92` | "jobs with higher fragmentation see a weaker translation from AI exposure to AI execution" |
| `1_introduction.tex:131` | "exhibit a substantially lower share of their steps executed by AI"; *substantially* is exactly what dies |
| `1_introduction.tex:136-137` | the prompt-robustness and task-listing-robustness claims, for Prediction #2 only |
| `7_empirics.tex:141` | "negative and statistically significant at the 1% level in all specifications"; p = 0.945, 0.317, 0.670 |
| `7_empirics.tex:144-146` | the "common pattern" paragraph; its only surviving support is SA.B, which is mechanical |
| `SA_D:146`, `:156` | "the EFI coefficients are consistently negative and statistically different from zero" |
| `SA_E:133-136` | "The fragmentation relationship therefore survives the restriction to frequent tasks" |
| `SA_E:126`, `:132` | "and found in the full-sample estimates", in the blue/red colour legend |
| `8_conclusion.tex:17-18` | the dispersion clause of the closing summary |

`SA_F:46-48`, "the fragmentation channel of Prediction #2 continues to hold", **survives**.

### Definitional statements about what "AI Exposure" is measured on

| Where | Change |
|---|---|
| `7_empirics.tex:19` | "treat their E1 category as exposed to AI"; correct today, becomes incorrect |
| `7_empirics.tex:133`, `SA_B:38` | Table notes, "the share of AI-exposed (E1) steps"; same fix |
| `7_empirics.tex:135`, `SA_B:40` | the step-count control sentence becomes internally consistent for the first time |
| `7_empirics.tex:102` | "this control **ensures** that beta2 is identified from how those steps are arranged"; soften, per D29 |
| `SA_E:130-131` | correct about the EFI, silent about the regressor; add it |
| `SA_D:48`, `:68` | the median splits are on E1; move them to E1∪E2 for consistency. `GPT_task_sequences_overlap.ipynb` already computes the E1∪E2 split and never plots it, so this is nearly free |

### Numbers that simply change

`7_empirics.tex:140` (exposure range 0.09 to 0.24 becomes about 0.39 to 0.49),
`SA_E:133-134`, `SA_F:393`, `SA_F:397-399`, `SA_F:400-401`, and the two tables and two figures.

### Against the review log

- **D27** (AI-exposed defined three inconsistent ways) is directly resolved, and should be
  upgraded from a presentational inconsistency to a specification error.
- **D29** (EFI varies with workflow length) is superseded in emphasis: the dominant
  uncontrolled term is the level share, not length. Adding `m` or `log m` changes nothing.
- **D31** (execution-based EFI is a function of the outcome) escalates to load-bearing.
- **D30** (predicted signs not derived) reopens in substance: the paper still predicts
  beta2 < 0 and now needs a sentence about what [-0.20, +0.19] rules out.
- **D42** (external validation overstates Section 7) needs rebalancing in the opposite
  direction from what it anticipated.
- Newly raised, not covered by any existing item: the regressor and the index are built on
  different exposure sets (the specification error itself), and the robustness architecture
  inverts, with the two appendices advertised as confirming robustness now contradicting the
  headline while the external benchmark supports it.

## 5b. Does the count control matter? (added 2026-09-02)

Two different variables get called "the step-count control" and they are not the same thing:
`k`, the count of AI-able (E1 or E2) steps, which is what Equation (11) controls for and
prints as "Number of AI-able Steps Control"; and `m`, total workflow length, which is not in
the regression at all and which D29 argues should be. `step_count_controls.py` runs the full
grid: 2 samples x 2 exposure regressors x 5 control sets x 3 FE, so 60 cells. The 45 of them
that the summary below rests on were independently reproduced from the raw data at three
decimals by a separate script.

**On the matched specification, 872 sample, the control is close to irrelevant.** EFI
coefficient, clustered SE, p:

| control | no FE | Major FE | Minor FE |
|---|---|---|---|
| none | -0.019 (0.098) .84 | -0.057 (0.087) .51 | -0.008 (0.093) .93 |
| `k` (the paper's) | -0.007 (0.101) .95 | -0.086 (0.086) .32 | -0.040 (0.093) .67 |
| `m` (D29's proposal) | -0.011 (0.099) .91 | -0.072 (0.086) .41 | -0.020 (0.093) .83 |
| log `m` | -0.014 (0.099) .89 | -0.069 (0.086) .42 | -0.020 (0.092) .83 |
| `k` + `m` | -0.014 (0.101) .89 | -0.086 (0.086) .31 | -0.041 (0.093) .66 |

Within an FE column the spread across control sets is at most 0.033, about a third of one
standard error. Adding `m`, which is what D29 asks for, moves the coefficient by 0.004 to
0.015 and changes no verdict, so **D29 can be closed cheaply**: put `m` or `log m` in and
report that nothing moves. The coefficient on `m` itself is +0.064 (p .02) with no FE and
-0.048 (p .03) with major FE, so workflow length does predict the execution share; it just
does not run through the fragmentation coefficient.

An adversarial search over 2,229 matched-872 specifications, including polynomials in `k` and
`m`, bins and deciles, `k` x `m` interactions, `1/m`, and a full set of workflow-length
dummies, found **zero fixed-effects cells significant at 5%** and a minimum p anywhere of
0.025. Exactly two cells clear 5%, both without fixed effects, and both need a
non-hierarchical polynomial plus singleton length-dummy cells; each dies four separate ways
(influence trimming, sparse-cell trimming, HC3, and any fixed effects). Under the null one
would expect about 5% of cells to clear 5% by chance; 2 of 2,229, or 0.09%, is observed.

**Three things this changes in how the earlier sections should be read.**

1. *The reason the controls agree is empirical, not algebraic.* It is tempting to say that
   `k/m` together with `k` pins down `m`. That is true only nonlinearly: the R-squared of `m`
   on `(1, k/m, k)` is 0.659, so a third of `m`'s variation is orthogonal to that conditioning
   set. The five control sets are genuinely distinct linear conditioning sets, and they agree
   because the residual EFI happens to be nearly orthogonal to residual `k` and `m` once `k/m`
   is held. That is a fact about this dataset, not a theorem, and it could fail on another.
2. *It is the 871 result that is control-fragile, not the 872 one.* Six of the fifteen
   matched-871 cells are insignificant at 5%, and the pattern is structural rather than
   random: every minor-FE cell is insignificant while every no-FE and major-FE cell is
   significant. The flip also happens inside the paper's own five choices: 871, major FE, no
   control gives p = 0.056 while `k` gives p = 0.024. Anyone tempted to treat the 871 estimate as the "real" matched result is leaning
   on something the control choice can dissolve. This strengthens section 6's conclusion.
3. *The published specification is more control-fragile than the five-control grid shows.*
   Within those five the largest swing is 0.137 (no FE, `m` at -0.376 against `k + m` at
   -0.239). But adding a `k` x `m` interaction moves it to -0.131 (0.100), p = 0.191, killing
   significance outright, and length dummies plus `1/m` plus `k` x `m` gives -0.061 (0.100),
   p = 0.543. So the published result is not merely mis-specified, it is unstable to
   innocuous-looking changes in how the count enters.

**A separate finding, flagged rather than relied on.** The matched null is scale-dependent.
Regressing log of the AI-execution share on the EFI, among the 555 occupations with any AI
execution, gives +0.372 (0.100), p = 0.0002 with no FE and +0.200 (0.097), p = 0.039 under
major-group FE. Two reasons this is not a rescue of Prediction #2 and should not be presented
as one. The sign is **positive**, the opposite of what the model predicts. And the subsample is
selected on the outcome in a way that is strongly correlated with the regressor: the 317
dropped occupations have mean EFI 0.912 against 0.635 among those kept, because an occupation
with no exposed steps has EFI 1 by construction and zero execution. It is worth knowing that
the level and proportional specifications disagree, but this particular contrast is
confounded.

## 6. Two traps worth remembering

**The -0.211 is not about the step-count control.** `exposure_definition_grid.csv` in
`data/computed_objects/apqc_pcf_fragmentation/` reports the matched O\*NET spec at -0.211
(p = 0.026) without the count control, and -0.209 (p = 0.031) with it. Dropping the control on
the 872 sample moves the coefficient only from -0.007 to -0.019, both null. The -0.211 comes from the **871-occupation sample**, which drops the 789
task rows with missing `human_labels` and recomputes `m`. The mechanism: 183 of the 2,998
maximal runs in the 872 panel, 6.1%, exist only because an unlabelled task sitting inside a
block of E1∪E2 tasks is coded "not AI-able" and splits the block.

Reaching that significant result takes three stacked choices, and dropping any one kills it:
drop the unlabelled rows; keep the 62 zero-exposure occupations whose EFI is 1.0 by
construction (restricting to `k >= 1` gives -0.173, p = 0.106); and avoid minor-group fixed
effects (-0.105, p = 0.291). On that same 871 sample all four scale-free arrangement
statistics are indistinguishable from zero. A census of 192 matched specifications finds 21
significant with the predicted sign, all 21 on the 871 sample and all 21 the same statistic.

**Whether unlabelled tasks belong in `m` is a real modelling question**, not a bug. It is
defensible to say unknown exposure should not count as unexposed. It is equally defensible to
keep the task in the workflow, which is what the paper's own headline sample does. Whichever
is chosen, it should be chosen explicitly and applied everywhere, not left to differ between
two files.

## 7. Errors found along the way, unrelated to this change

- `SA_F:339` and `:388` say "sixteen industry-specific frameworks". The estimation sample
  carries 18 framework labels: Cross-Industry plus **seventeen** industry frameworks.
  `apqc_industry_leaf_matching.py` globs 18 `K*.xlsx` workbooks, skips the Cross-Industry
  duplicate, and prepends the explicit entry; its own docstring repeats the miscount. Column
  (3)'s framework fixed effect absorbs 18 levels, not 17. No estimate moves.
- `SA_E:134` says the positive cells are "+0.05 or smaller in magnitude". Two of the six
  exceed that, at +0.0705 and +0.1956.
- `7_empirics.tex:117` says 44% and 14%; `SA_F:355` says 46% and 13%.
- The APQC reshuffle placebo sets no random seed.
- `analysis/apqc_pooled_predictions.py:18` hardcodes an absolute home directory, against the
  repo rule. Not changed, since that file is untouched by this pass.
- Re-running `analysis/apqc_pooled_predictions.py` always leaves
  `writeup/tables/apqc_fragmentation_index_regression.tex` dirty in git. The numbers it emits
  are identical; what differs is a trailing `\multicolumn{4}` significance-footnote row that
  the emitter writes and that was deleted by hand in the committed file, because the paper
  supplies notes in the float's minipage instead. So the diff after any re-run is one added
  line and nothing else. Either drop that row from the emitter or stop hand-editing the
  output, otherwise every re-run looks like a change when it is not.
- `analysis/onet_fragmentationIndex_weeklyTasks_placebo.ipynb` carries the same E1-vs-E1∪E2
  mismatch **and** omits the step-count control. It is unpublished, so it is low priority, but
  it would need re-running before any randomization p-value from it is cited.

## 8. Still undecided

The two options from the original note stand, and the evidence has shifted the balance:

1. **Drop the Section 7 test of Prediction #2**, keep the fragmentation index as a theoretical
   object in Section 4, and say the data do not identify the arrangement channel.
2. **Keep the table on the matched specification and report the null honestly.**

Option 2 is better supported than it looked, but not for the expected reason. The fear was
that the robustness exercises would collapse; two did, but into stable nulls that are
publishable statements in their own right, and SA.F holds up with a mechanism that can be
explained rather than explained away. The exposure result also gets materially cleaner.

The real cost of option 2 is not the fragmentation table. It is that once the exposure-based
leg is null, `7_empirics.tex:144` rests entirely on SA.B, which is algebraically the outcome.
That paragraph and abstract clause (2) need rewriting under either option.

## 9. Reproducing

```bash
cd "$(git rev-parse --show-toplevel)"
for f in analysis/efi_matched_exposure/*.py analysis/efi_matched_exposure/verify/*.py; do
  /opt/anaconda3/bin/python "$f" || echo "FAILED: $f"
done
```

All 18 scripts run clean, take repo-relative paths, and write only into
`data/computed_objects/efi_matched_exposure/`. Each reproduces the published numbers before
changing anything. `table2_and_SAB.py` regenerates the two O\*NET tables byte-for-byte. For
the APQC table `verify/verify_SAF_bytecheck.py` matches every number and every line except
one: the emitter writes a trailing significance-footnote row that was deleted by hand in the
committed file, so the replication is 21 lines against the published 20 and is identical once
that row is dropped. See section 7. The `verify/` scripts are independent reimplementations
written without reference to the others, and both numeric verifiers returned no disagreement
in any coefficient.
