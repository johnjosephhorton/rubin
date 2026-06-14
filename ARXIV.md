# arXiv submission — full version

This is the **full version** of the paper (the `writeup/` source, i.e. the PDF
hosted at <https://peymanshahidi.github.io/assets/pdf/chaining_tasks_ai_automation.pdf>),
distinct from the shorter EC'26 conference submission in `EC2026/`.

## Build the upload package

```sh
./build_arxiv.sh
```

Regenerates `arxiv/` (staging) and `arxiv_submission.tar.gz` from `writeup/`,
verifying it compiles to a clean 80-page PDF. Both outputs are git-ignored —
edit `writeup/`, re-run, re-upload a new version. **Upload `arxiv_submission.tar.gz`.**

## Form metadata (paste into arXiv)

**Title:** Chaining Tasks, Redefining Work: A Theory of AI Automation

**Authors:** Mert Demirer (MIT); John J. Horton (MIT & NBER); Nicole Immorlica
(Yale & Microsoft); Brendan Lucier (Microsoft); Peyman Shahidi (MIT)

**Abstract** (plain text, matches the `writeup/` full version):

> Production is a sequence of steps that can be executed (1) manually, (2)
> augmented with AI, or (3) fully automated within contiguous AI-executed steps
> called "chains." Firms optimally bundle steps into tasks and then jobs,
> trading off specialization gains against coordination costs. We characterize
> the optimal assignment of humans and AI to steps and the firm's resulting job
> structure, showing that comparative advantage logic can fail with AI chaining.
> The model implies non-linear productivity gains from AI quality improvements
> and admits a CES representation at the macro level. Empirical evidence supports
> the model's key predictions that (1) AI-executed steps co-occur in chains, (2)
> dispersion of AI-exposed steps lowers AI execution at the job level, and (3)
> adjacency to AI-executed steps increases the likelihood that a step is
> AI-executed.

**Categories:** primary `econ.GN` (General Economics); cross-list `cs.GT`
(Computer Science and Game Theory). Flip to `cs.GT` primary if you prefer to lead
with the EC venue. — TODO: confirm with coauthors.

**Comments:** To appear in the 27th ACM Conference on Economics and Computation
(EC '26).

**License:** CC BY 4.0 (matches the ACM Open version). — TODO: confirm.

## Notes / gotchas

- First-time submitters may need **endorsement** for `econ.GN`/`cs.GT`; a coauthor
  (e.g. Immorlica/Lucier for `cs.GT`) can endorse, or submit from their account.
- arXiv announces on business days only; submit early in the week so the
  `arxiv.org/abs/...` URL is live before the EC'26 final upload (June 22).
- Once the abs URL exists, swap it into the EC one-pager link
  (the TODO in `EC2026/camera_ready/camera_ready.tex`).
- The build script strips the title-page "most recent version" website link from
  the arXiv copy (inappropriate on arXiv, which is versioned). The `writeup/`
  source keeps it for the hosted PDF.
