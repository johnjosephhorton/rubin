## Pre-submission fixes (ReStud draft)

### Submission compliance
- [ ] **Abstract is 151 words (limit 150).** Commit `5da9f6a` had trimmed it to 147 and `df31bc2` pushed it back over. Delete "just" in "beyond just how many there are" to reach 150.

### Major
- [ ] **Prediction #3 is stated as a general finding (p. 5, p. 7, p. 42, p. SA-33).** Only the PCF result is significant (-0.35, -0.26, -0.34). The O*NET estimates (-0.01, -0.09, -0.04) are not. Name the sample in the intro, the conclusion and the SA.E closing sentence. The Sept 4 fix to the intro has been undone.
- [ ] **PCF units are called "jobs" (p. 5).** They are process groups.

### Medium
- [ ] **SA.F contradicts Section 7 on the PCF.**
  - [ ] p. SA-47 still calls the PCF "a validation sample rather than ... the main sample".
  - [ ] p. SA-46 says no labels exist for PCF steps and that the PCF is not crosswalked to O*NET tasks. SA.F.2 builds exactly that crosswalk, and Section 7 treats the two samples equally.
- [ ] **SA.E text does not match its heatmaps.**
  - [ ] The baseline is quoted as "+0.12". The figure shows 0.113 and 0.105, and Table 2 shows 0.11.
  - [ ] "Often grows" under frequency filters is false. Every Daily+ cell is below baseline (0.070 to 0.092 against 0.113).
  - [ ] "Loses significance only in the sparsest cells" is reversed. The sparsest cells have the largest, mostly significant estimates (0.279\*\*\*, 0.328\*\*\*).
  - [ ] "The result survives pruning" is overstated for chain length, where 8 of 12 pruned cuts sit inside the placebo band.
- [ ] **SA.F exact-recovery claim is the known Kendall τ float-equality bug (p. SA-39).** "None of the branches with five or more steps" is wrong. The correct figure is 21 of 208 (10.1%).
- [ ] **SA.A defines E1 backwards (p. SA-1).** "In at least half the time required by a human" should be a time saving of at least half, as Section 7 states correctly.
- [ ] **Causal captions survive in the appendices.** Figure A.1 and six SA.B exhibits are titled "Effect of ...", and SA.D says "spillover". Match Table 2's association wording.

### Prose (introduced by the Overleaf sync)
- [ ] p. 6: "improvements in AI ... because it presses" should be "they press".
- [ ] p. 26: "where those two prices balance determine" should be "determines".
- [ ] p. 2: "A step is ..., what classic models call a task" no longer attaches to anything. Restore "corresponding to".
- [ ] p. 6: remove the last em dash ("the same reduction---and any direct saving in coordination---makes").
- [ ] p. OA-24: "Sections 3– 5" has a stray space. Use `--\ref` instead of `--~\ref`.
- [ ] Appendices: replace the 11 remaining "main sample" with "O*NET sample".
- [ ] Table 3 header: change "APQC Process Groups" to "PCF Process Groups".

Full details, page numbers and rewrites are in `REVIEW_issues_2026-09-26.md` and `PROSE_issues.md`.
