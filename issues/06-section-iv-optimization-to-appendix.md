# Issue 06 — Section IV (algorithmic tractability / DP) occupies prime real estate

## Raised by
**R1 (§2 end) and R3 ("other comments" bullet 1).**

> R1: "Section 4 occupies prime real estate in the paper, and it may be worth considering carefully whether this is the best use of that space."
>
> R3: "Do we need to have Section IV, on how to compute the optimization, in the body of the paper? I see the value of stating the O(·) results somewhere… but at minimum I would have thought it made more sense to put the proofs in the appendix."

## Verification against source
- Section 4 lives at `writeup/optimization.tex` — 187 lines.
- Contains the polynomial-time DP result that translates the firm's problem into a tractable computation.
- Currently inputed from `main.tex:162` into the main body.

## Options to address

1. **Full move to appendix.** Keep a 1-paragraph statement in the main body (Theorem: "The firm's problem can be solved in time O(…)"), banish proof, DP construction, and lemmas to a new Appendix.
2. **Keep statement + intuition, move construction.** Retain a short subsection (½–1 page) explaining why tractability matters economically (firms can find the optimum); move the algorithmic construction and correctness proofs to an appendix.
3. **Status quo** — argue Section IV is integral. Weakest response to the reviewers.

## Effort vs. payoff

| Option | Effort | Payoff |
|---|---|---|
| 1. Full move | Low (½ day of cut-and-paste + rewrites to appendix) | **High** — directly addresses R1 + R3; recovers ~5–7 pages of prime real estate |
| 2. Keep statement + intuition | Low (1 day) | High — similar gains with minimal risk of losing economically-relevant content |
| 3. Status quo | Zero | Negative (reviewer irritation) |

**Recommendation:** Option 2 — lets you keep the economic payoff ("firms can be expected to find near-optimum") visible in the body while shipping the machinery to the appendix. Pairs with Issue 09 (paper length).
