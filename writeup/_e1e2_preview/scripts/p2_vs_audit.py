"""Cross-check this regeneration against the completed audit (scratchpad m6).

The audit and its independent second agent each rebuilt these numbers from the source
data with their own transcription of the notebook.  Any disagreement is a red flag.
"""
import glob

import numpy as np
import pandas as pd

REPO = "/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin"
WORK = f"{REPO}/writeup/_e1e2_preview/work/t2f1"
M6 = ("/private/tmp/claude-502/-Users-peymansh-MIT-Dropbox-Peyman-Shahidi-GitHub-rubin/"
      "5af7d286-51e7-470a-b57e-4e1373499eea/scratchpad/m6")

out = []
mine = pd.read_csv(f"{WORK}/ame_full_0_E1E2.csv")

# --- Table 2 vs audit agent 1 ---------------------------------------------------------
a = pd.read_csv(f"{M6}/rebuild_full_E1E2.csv")
a = a[a.term != "is_exposed"]
m = mine.merge(a, on=["model", "term"], suffixes=("_m", "_a"))
out.append(("Table 2 E1|E2 vs m6/rebuild_full_E1E2.csv (audit agent 1)", len(m),
            float(np.abs(m.ame_coef_m - m.ame_coef_a).max()),
            float(np.abs(m.ame_se_m - m.ame_se_a).max()),
            bool((m.nobs_m == m.nobs_a).all())))

# --- Table 2 vs audit agent 2 ---------------------------------------------------------
v = pd.read_csv(f"{M6}/verify_pred2-main/verify_table2_E1_vs_E1E2.csv")
v = v[(v.exposure == "E1E2") & (v.term != "is_exposed")]
m = mine.merge(v, on=["model", "term"], suffixes=("_m", "_v"))
out.append(("Table 2 E1|E2 vs m6/verify_pred2-main/... (audit agent 2)", len(m),
            float(np.abs(m.ame_coef_m - m.ame_coef_v).max()),
            float(np.abs(m.ame_se_m - m.ame_se_v).max()),
            bool((m.nobs_m == m.nobs_v).all())))

# --- 999 placebo draws vs audit agent 2 -----------------------------------------------
fs = [f for f in sorted(glob.glob(f"{M6}/verify_pred2-main/vresh_*.csv"))
      + sorted(glob.glob(f"{M6}/verify_pred2-main/vresh2_*.csv")) if "test" not in f]
aud = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
aud = aud[aud.exposure == "E1E2"]
mine_r = pd.read_csv(f"{WORK}/resh_E1E2_all.csv")
m = mine_r.merge(aud, on=["draw", "model", "term"], suffixes=("_m", "_a"))
out.append((f"999 E1|E2 placebo draws vs m6/verify_pred2-main/vresh*.csv", len(m),
            float(np.abs(m.ame_coef_m - m.ame_coef_a).max()), float("nan"),
            bool((m.nobs_m == m.nobs_a).all())))

print(f"{'check':<62}{'cells':>8}{'max|dAME|':>13}{'max|dSE|':>13}{'nobs ok':>9}")
for name, n, dc, ds, ok in out:
    print(f"{name:<62}{n:>8,}{dc:>13.2e}{ds:>13.2e}{str(ok):>9}")

pd.DataFrame(out, columns=["check", "cells", "max_abs_dAME", "max_abs_dSE", "nobs_identical"]) \
    .to_csv(f"{WORK}/audit_crosscheck.csv", index=False)
