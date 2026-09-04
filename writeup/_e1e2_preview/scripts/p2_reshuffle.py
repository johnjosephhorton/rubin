"""Figure OA.A.1 placebo null: the paper's position-reshuffle loop, under E1|E2.

Transcribed from cell 19 of analysis/onet_antrhopicIndex_execTypeVaryingDWA.ipynb:
Task Position is permuted within occupation with random_state = 42 + i, the four
neighbour flags are recomputed on the permuted 10,708-row estimation sample, rows with
any missing neighbour are dropped, and the six specifications are refit with
calculate_standard_errors=False.  Exposure mask is the only thing that differs.

Usage:  python3 p2_reshuffle.py <lo> <hi> <E1|E1E2> <out.csv>
"""
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin/"
                   "writeup/_e1e2_preview/scripts")
from p2_estimate import run_regressions_on, load_sample, TARGET_REGS, dependent_var

NEIGHBOR = ["prev_is_ai", "prev2_is_ai", "next_is_ai", "next2_is_ai"]
MASKS = {"E1": ["E1"], "E1E2": ["E1", "E2"]}


def reshuffle_once(md, seed):
    d = md.copy()
    d["Task Position"] = d.groupby("O*NET-SOC Code")["Task Position"].transform(
        lambda x: x.sample(frac=1, random_state=seed).values)
    d = d.sort_values(["O*NET-SOC Code", "Task Position"]).reset_index(drop=True)
    g = d.groupby("O*NET-SOC Code")["is_ai"]
    d["prev_is_ai"] = g.shift(1)
    d["prev2_is_ai"] = g.shift(2)
    d["next_is_ai"] = g.shift(-1)
    d["next2_is_ai"] = g.shift(-2)
    d = d.dropna(subset=NEIGHBOR).reset_index(drop=True)
    for c in NEIGHBOR:
        d[c] = d[c].astype(int)
    return d


def main():
    lo, hi, tag, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    md = load_sample(MASKS[tag])
    rows = []
    t0 = time.time()
    for i in range(lo, hi + 1):
        d = reshuffle_once(md, 42 + i)
        try:
            _, r = run_regressions_on(d, f"full_{i}", dependent_var, TARGET_REGS,
                                      calculate_standard_errors=False, out_dir=None)
        except Exception as e:
            print(f"draw {i} {tag} FAILED: {e}", flush=True)
            continue
        r["exposure"] = tag
        r["draw"] = i
        rows.append(r)
        if (i - lo + 1) % 25 == 0:
            print(f"  {i-lo+1} draws in {time.time()-t0:.0f}s", flush=True)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out, index=False)
    print("wrote", out, res.shape, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
