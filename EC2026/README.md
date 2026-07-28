# EC 2026 Camera-Ready

LaTeX source of the EC'26 submission (HotCRP paper #2016, "Chaining Tasks,
Redefining Work: A Theory of AI Automation"), imported 2026-06-12 from the
submitted version. This folder is the working copy for the camera-ready and
is kept separate from the main draft in `writeup/`.

The folder holds the complete submission source, including figure PNGs.
Per the repo-wide gitignore policy (`*png`, `*pdf`, aux files), figures and
build artifacts live here locally (synced via Dropbox) but stay out of git.

## Build

```sh
latexmk -f -pdf -interaction=nonstopmode main.tex
```

The `-f` is required: the submitted source carries two benign LaTeX errors
that Overleaf tolerated the same way — a `geometry` option clash (acmart
already loads `geometry`, so the `margin=0.86in` override in `main.tex`
never takes effect, in the submitted PDF either) and a `\Bbbk` redefinition
(`amssymb` after acmart's newtxmath). Last verified build: 58 pages, all
references and citations resolved.

## Files

- `main.tex` — full paper, de-anonymized for the camera-ready (real author
  block with ORCIDs, acknowledgments restored). Builds with the command
  above; de-anonymizing added a third benign error ("No country present"),
  which is spurious — it fires in acmart's PDF-metadata pass even though
  every affiliation carries `\country`.
- `camera_ready/` — self-contained folder for the 1-page ACM Digital
  Library abstract, kept separate so submission and camera-ready outputs
  don't mix. Holds `camera_ready.tex` plus verbatim copies of the official
  proceedings template files (`acmart.cls` v2.12 and `acm-ec-26-proc.sty`
  from `ec26-proceedings-style-files.zip`); build from inside the folder
  with `latexmk -pdf camera_ready.tex` (compiles clean, exactly 1 page).
  Updated 2026-07-25 per the July 15 instructions (proceedings style files
  swapped in, which removes the stray journal footer and page number) and
  2026-07-27 after the e-Rights form: rights block pasted verbatim from the
  ACM Publication Release Confirmation (CC BY 4.0, DOI
  10.1145/3821539.3827858; PDF copies of the release form and confirmation
  sit in this folder), acknowledgments deliberately omitted (not the norm
  for abstract-only entries). Still open: keyword wording sign-off with
  coauthors, John's ORCID in his HotCRP profile, then the HotCRP upload
  (CCS XML and keywords fields, PDF plus source, "Save and resubmit").

## Deadlines (from EC'26 chairs)

Camera-ready instructions arrived by email July 15, 2026; they supersede
the June milestones (title/author lock June 15, talk video June 30, both
past).

- **July 29, 2026** — camera-ready due on HotCRP
  (<https://ec2026.hotcrp.com/paper/2016/edit>), three steps: (1) fill the
  ACM e-Rights form, which generates the license/DOI values by email
  (done 2026-07-27; values are in the tex);
  (2) pick CCS terms at <https://dl.acm.org/ccs>, paste the generated XML
  into HotCRP, and enter keywords there; (3) upload the final PDF plus
  source and click "Save and resubmit". We publish the 1-page abstract
  (`camera_ready/`), an option the chairs' detailed instructions confirm;
  the 18-page body plus 10-page appendix limits apply to full-paper
  uploads only.
