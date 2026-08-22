"""Numerical check of Proposition 1 (Local Overturning of Comparative Advantage).

Companion to comparativeAdvantage_regions_plot.ipynb, which draws the region map
for the same three-step block {k-1, k, k+1}.  Nothing here produces a figure; the
script only verifies the claims of the proposition against brute-force enumeration of
every contiguous arrangement of the block.

    enumeration  a human-advantaged step k is never augmented on its own, so AI
                 reaches it only through one of three chains (a step of the proof,
                 not a part of the statement),
    (i)          each of those three chains is uniquely optimal for some neighbor
                 parameters, whatever step k's own parameters,
    (ii)         whether AI executes step k is monotone in the neighbors' success
                 probabilities q_{k-1} and q_{k+1}, together with the reduction of
                 both coordinates to a single functional form that the proof of
                 (ii) runs on.

Run:  python analysis/comparativeAdvantage_proposition_check.py
"""

import itertools
import random

random.seed(11)


# --------------------------------------------------------------------------- #
# arrangement costs for the block {k-1, k, k+1}
# --------------------------------------------------------------------------- #
def arrangement_costs(tMp, tAp, qp, tMk, tAk, qk, tMn, tAn, qn):
    """V0..V3: cost of the four contiguous arrangements of the block.

    p = predecessor (k-1), k = focal step, n = successor (k+1).  A step executed
    on its own costs min(manual, AI management / success prob); an AI chain costs
    the management time of its augmented endpoint divided by the product of the
    success probabilities of all its steps.  V0 prices a singleton {k} at its
    manual cost, which is the proposition's hypothesis t^M_k < t^A_k / q_k.
    """
    cp = min(tMp, tAp / qp)
    cn = min(tMn, tAn / qn)
    V0 = cp + tMk + cn                    # no chain covers step k
    V1 = cp + tAn / (qk * qn)             # chain {k, k+1}      -> k automated
    V2 = tAn / (qp * qk * qn)             # chain {k-1, k, k+1} -> k automated
    V3 = tAk / (qp * qk) + cn             # chain {k-1, k}      -> k augmented
    return V0, V1, V2, V3


def brute_force(tMp, tAp, qp, tMk, tAk, qk, tMn, tAn, qn):
    """Minimum cost over every contiguous partition of the block, pricing each
    singleton at min(manual, augmented) rather than assuming the hypothesis."""
    tM, tA, q = [tMp, tMk, tMn], [tAp, tAk, tAn], [qp, qk, qn]
    best = (float("inf"), None)
    for cuts in itertools.product([0, 1], repeat=2):
        runs, cur = [], [0]
        for i, c in enumerate(cuts):
            if c:
                runs.append(cur)
                cur = []
            cur.append(i + 1)
        runs.append(cur)
        total = 0.0
        for r in runs:
            if len(r) == 1:
                j = r[0]
                total += min(tM[j], tA[j] / q[j])
            else:
                p = 1.0
                for j in r:
                    p *= q[j]
                total += tA[r[-1]] / p
        if total < best[0]:
            best = (total, tuple(tuple(r) for r in runs))
    return best


def ai_executes(**kw):
    V = arrangement_costs(**kw)
    return min(V[1], V[2], V[3]) < V[0] - 1e-12


def draw_block():
    """Random block with the human holding comparative advantage on step k."""
    lu = lambda a, b: 10 ** random.uniform(a, b)
    tMk = lu(-2, 2)
    qk = random.uniform(0.01, 0.99)
    tAk = tMk * qk * (1 + lu(-3, 2))          # forces tMk < tAk / qk
    return dict(
        tMp=lu(-2, 2), tAp=lu(-2, 2), qp=random.uniform(0.01, 0.99),
        tMk=tMk, tAk=tAk, qk=qk,
        tMn=lu(-2, 2), tAn=lu(-2, 2), qn=random.uniform(0.01, 0.99),
    )


# --------------------------------------------------------------------------- #
# proof step: step k is never augmented on its own
# --------------------------------------------------------------------------- #
def check_enumeration(n=200_000):
    bad = 0
    for _ in range(n):
        d = draw_block()
        cost, runs = brute_force(**d)
        for r in runs:
            if r == (1,):                              # step k as a singleton
                tM, tA, q = d["tMk"], d["tAk"], d["qk"]
                if tA / q < tM - 1e-12:                # would be augmented alone
                    bad += 1
    print(f"enum  singleton step k augmented in an optimum: {bad} / {n}")
    return bad == 0


# --------------------------------------------------------------------------- #
# part (i): each chain is uniquely optimal somewhere
# --------------------------------------------------------------------------- #
def check_part_i(n=2_000):
    """The three constructions of the proof, each with neighbor costs
    t^M_{k-1} = t^A_{k-1} = mu, t^M_{k+1} = 1, t^A_{k+1} = tau."""
    label = {0: "manual", 1: "{k,k+1}", 2: "{k-1,k,k+1}", 3: "{k-1,k}"}
    bad = 0
    for _ in range(n):
        d = draw_block()
        tMk, tAk, qk, qp, qn = d["tMk"], d["tAk"], d["qk"], d["qp"], d["qn"]

        # chain {k, k+1}: predecessor too cheap to be worth absorbing
        tau = qk * qn * tMk / 2
        mu = 0.5 * min(tMk / 2 * (1 - qp) / qp, tAk / (qp * qk) - tMk / 2)
        cases = [(mu, tau, 1)]
        # chain {k-1, k, k+1}: a cheap endpoint makes the long chain dominant
        cases.append((1.0, 0.999 * qn * min(qp * qk, tAk), 2))
        # chain {k-1, k}: an expensive endpoint stops the chain at step k
        mu3 = tAk / (qp * qk)
        cases.append((mu3, 1.001 * max(qn, qk * qn, qp * qk * qn * (mu3 + 1)), 3))

        for mu_, tau_, want in cases:
            kw = dict(tMp=mu_, tAp=mu_, qp=qp, tMk=tMk, tAk=tAk, qk=qk,
                      tMn=1.0, tAn=tau_, qn=qn)
            V = arrangement_costs(**kw)
            j = min(range(4), key=lambda i: V[i])
            if j != want or abs(brute_force(**kw)[0] - V[j]) > 1e-9:
                bad += 1
                print(f"      MISMATCH: wanted {label[want]}, got {label[j]}, {kw}")
    print(f"(i)   constructions failing to be the unique optimum: {bad} / {3 * n}")
    return bad == 0


# --------------------------------------------------------------------------- #
# part (ii): monotonicity in the neighbors' success probabilities
# --------------------------------------------------------------------------- #
def check_part_ii(n=400_000, perturbations=3):
    base = viol_p = viol_n = viol_k = 0
    for _ in range(n):
        d = draw_block()
        if not ai_executes(**d):
            continue
        base += 1
        for _ in range(perturbations):
            up = dict(d, qp=random.uniform(d["qp"], 1.0))
            if not ai_executes(**up):
                viol_p += 1
            un = dict(d, qn=random.uniform(d["qn"], 1.0))
            if not ai_executes(**un):
                viol_n += 1
            # not part of the proposition, which fixes step k's own parameters, but
            # asserted in the text: a uniform rise in AI quality lifts q_k too, and
            # that cannot return the step to the human either.
            uk = dict(d, qk=random.uniform(d["qk"], 1.0))
            if not ai_executes(**uk):
                viol_k += 1
    print(f"(ii)  AI-executing base points: {base}")
    print(f"      violations raising q_(k-1): {viol_p}")
    print(f"      violations raising q_(k+1): {viol_n}")
    print(f"      violations raising q_k:     {viol_k}")
    return viol_p == 0 and viol_n == 0 and viol_k == 0


# --------------------------------------------------------------------------- #
# part (ii), proof step: both coordinates are instances of one functional form
# --------------------------------------------------------------------------- #
def check_common_form(n=200_000):
    """The proof of part (ii) reduces both coordinates to

        D(q) = A/q - min{m, B/q} - c,

    reading q as q_{k-1} for D_2, D_3 and as q_{k+1} for D_1, D_2.  Check the
    four identifications, and that A >= B always holds in the successor case,
    which is what confines it to the easy branch of the claim."""
    def form(A, B, m, c, q):
        return A / q - min(m, B / q) - c

    bad = 0
    for _ in range(n):
        d = draw_block()
        V = arrangement_costs(**d)
        D1, D2, D3 = V[1] - V[0], V[2] - V[0], V[3] - V[0]
        tMp, tAp, qp = d["tMp"], d["tAp"], d["qp"]
        tMk, tAk, qk = d["tMk"], d["tAk"], d["qk"]
        tMn, tAn, qn = d["tMn"], d["tAn"], d["qn"]
        cp, cn = min(tMp, tAp / qp), min(tMn, tAn / qn)

        cases = [                                     # (A, B, m, c, q), target
            ((tAn / (qk * qn), tAp, tMp, tMk + cn,   qp), D2),   # q_(k-1), D_2
            ((tAk / qk,        tAp, tMp, tMk,        qp), D3),   # q_(k-1), D_3
            ((tAn / qk,        tAn, tMn, tMk,        qn), D1),   # q_(k+1), D_1
            ((tAn / (qp * qk), tAn, tMn, cp + tMk,   qn), D2),   # q_(k+1), D_2
        ]
        for args, want in cases:
            if abs(form(*args) - want) > 1e-9 * max(1.0, abs(want)):
                bad += 1
        if tAn / qk < tAn - 1e-12 or tAn / (qp * qk) < tAn - 1e-12:
            bad += 1                                  # successor case needs A >= B
    print(f"form  identifications failing: {bad} / {5 * n}")
    return bad == 0


# --------------------------------------------------------------------------- #
# Example 1 of the paper
# --------------------------------------------------------------------------- #
def check_example():
    kw = dict(tMp=25.0, tAp=25.0, qp=0.95, tMk=6.0, tAk=3.5, qk=0.3,
              tMn=8.0, tAn=4.0, qn=0.9)
    V = arrangement_costs(**kw)
    cost, runs = brute_force(**kw)
    print(f"Example 1: V0={V[0]:.3f} V1={V[1]:.3f} V2={V[2]:.3f} V3={V[3]:.3f}")
    print(f"           optimum {cost:.3f} at {runs} (all three steps chained)")
    return abs(cost - V[2]) < 1e-9


if __name__ == "__main__":
    ok = all([check_example(), check_enumeration(), check_part_i(),
              check_common_form(), check_part_ii()])
    print("\nAll claims of Proposition 1 verified." if ok else "\nFAILURES ABOVE.")
