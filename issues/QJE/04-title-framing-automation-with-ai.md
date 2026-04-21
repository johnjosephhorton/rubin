# Issue 04 — Title framing: "automation with applications to AI"

## Raised by
**R3 (framing section).**

> "The results are framed as being about 'AI automation'. But for the most part, I think they apply to automation and the structure of work in general. So it might make more sense to call it '…A Theory of Automation, with Applications to AI'."
>
> "The abstract says that 'comparative advantage logic can fail with AI chaining'. But handoff costs between people can make it fail even without chaining, as in Becker and Murphy."

## Verification against source
- Current title: `main.tex:76` — "Chaining Tasks, Redefining Work: A Theory of AI Automation"
- Abstract CA claim: `main.tex:138` — "comparative advantage logic can fail with AI chaining"
- The model itself (`model.tex`, `optimization.tex`) is indeed agnostic about whether the "machine" is AI or some other automation technology. Nothing structurally depends on AI specifically.

## Options to address

1. **Full rebrand.** Retitle "Chaining Tasks, Redefining Work: A Theory of Automation, with Applications to AI". Generalize abstract and intro language accordingly. Cite Becker-Murphy and non-AI automation literature.
2. **Keep AI framing, justify narrowly.** Argue AI is a distinctive case (chains of many high-capability steps with shared validation) and explicitly scope the theory to it. Address R3's CA-without-chaining concern with a footnote.
3. **Split-framing.** Title stays AI-centric; intro says "the framework applies to automation broadly; we focus on AI because [new data / policy relevance / timeliness]".

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Full rebrand | Low (½ day rewrite) | Medium-high — widens audience, hits R3's point, but weakens AI hook at a moment when AI-specific papers are in vogue |
| 2. Keep AI framing | Low (½ day) | Low-medium — doesn't move the needle for the reviewer |
| 3. Split-framing | Low (½ day) | Medium — best of both, cheapest signal of responsiveness |

**Recommendation:** Option 3 at minimum. Consider Option 1 only if the next journal (AEJ:Applied, QE, RESTUD) has a technology/organization angle that benefits from the broader framing.
