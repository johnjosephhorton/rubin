"""Control checks feeding section A of diffs/table2-figOAA1.txt.

1. pixel_control.csv     regenerating the four published panels from the repo's own
                         cached draws must reproduce the committed PNGs exactly.
2. reshuffle_control.csv my transcription of the placebo loop, run under E1-only, must
                         reproduce the repo's cached regression_ame_results_full_i.csv.
3. xlim.csv              the x-limits the paper's own plotting code computes on each input.
"""
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

REPO = "/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin"
STAGE = f"{REPO}/writeup/_e1e2_preview"
WORK = f"{STAGE}/work/t2f1"
CACHE = (f"{REPO}/data/computed_objects/execTypeVaryingDWA_anthropicIndex"
         f"_noTasksWithRepetitiveDWAs/regression_summaries_is_ai")
PANELS = ["no_fe_no_dwa", "major_fe_no_dwa", "minor_fe_no_dwa", "no_fe_with_dwa"]

# ---- 1. pixel control -----------------------------------------------------------------
rows = []
for s in PANELS:
    a = Image.open(f"{STAGE}/plots_current/AME_full_is_ai_{s}.png").convert("RGB")
    b = Image.open(f"{WORK}/repro_current/AME_full_is_ai_{s}.png").convert("RGB")
    if a.size != b.size:
        rows.append({"spec": s, "identical": False, "maxdiff": -1,
                     "note": f"size {a.size} vs {b.size}"})
        continue
    d = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16))
    rows.append({"spec": s, "identical": bool(d.max() == 0), "maxdiff": int(d.max()),
                 "note": f"{a.size[0]}x{a.size[1]}"})
pd.DataFrame(rows).to_csv(f"{WORK}/pixel_control.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))

# ---- 2. reshuffle control -------------------------------------------------------------
mine = pd.read_csv(f"{WORK}/resh_test_E1.csv")
lo, hi = int(mine.draw.min()), int(mine.draw.max())
cached = pd.concat([pd.read_csv(f"{CACHE}/regression_ame_results_full_{i}.csv")
                    for i in range(lo, hi + 1)], ignore_index=True)
m = mine.merge(cached, on=["dataset", "model", "term"], suffixes=("_mine", "_pub"))
md = float(np.abs(m.ame_coef_mine - m.ame_coef_pub).max())
pd.DataFrame([{"lo": lo, "hi": hi, "ncells": len(m), "maxdiff": md,
               "nobs_identical": bool((m.nobs_mine == m.nobs_pub).all()),
               "ok": bool(md < 1e-12 and (m.nobs_mine == m.nobs_pub).all())}]) \
    .to_csv(f"{WORK}/reshuffle_control.csv", index=False)
print(f"reshuffle control: draws {lo}-{hi}, {len(m)} cells, max abs diff {md:.2e}")

# ---- 3. xlim --------------------------------------------------------------------------
rows = []
for mode in ["published", "e1e2"]:
    st = pd.read_csv(f"{WORK}/figOAA1_stats_{mode}.csv")
    # the code's bound: symmetric over all reshuffled values AND all observed values
    gmax = max(st.null_max.max(), st.observed.max())
    gmin = min(st.null_min.min(), st.observed.min())
    b = max(abs(gmin), abs(gmax))
    rows.append({"mode": mode, "xmin": -b - 0.02 * b, "xmax": b + 0.02 * b})
pd.DataFrame(rows).to_csv(f"{WORK}/xlim.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
