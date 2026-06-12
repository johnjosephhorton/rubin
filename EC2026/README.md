# EC 2026 Camera-Ready

LaTeX source of the EC'26 submission (HotCRP paper #2016, "Chaining Tasks,
Redefining Work: A Theory of AI Automation"), imported 2026-06-12 from the
submitted version. This folder is the working copy for the camera-ready and
is kept separate from the main draft in `writeup/`.

Binary figures (PNGs) are intentionally not tracked, per repo convention.
TikZ figure sources (`plots/*.tex`) and tables (`tables/`) are included.
To compile the full paper locally, copy the PNGs from `writeup/plots/` into
`plots/` or add `\graphicspath{{../writeup/plots/}}` to `main.tex`.

## Deadlines (from EC'26 chairs)

- **June 15, 2026** — last day to edit title/authors on HotCRP; all authors
  need an ORCID iD in their HotCRP profile first.
- **June 22, 2026** — full camera-ready upload on HotCRP
  (<https://ec26.hotcrp.com/u/1/paper/2016/edit>): 1-page abstract with
  pointer to the full version, keywords, at least one ACM CCS code,
  ACM corresponding author, and full LaTeX source.
- **June 30, 2026** — talk video upload (max 20 min,
  `ec2026-video-paper2016.mp4`).
