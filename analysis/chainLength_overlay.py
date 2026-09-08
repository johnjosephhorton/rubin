"""Prediction #1 as one panel per corpus, with both placebo distributions overlaid.

Replaces the four separate histograms with two. Within a panel the two nulls are drawn as
outlines over a light fill, which keeps both readable where they coincide: in O*NET the two
nulls sit on top of each other (means 1.3823 and 1.3818, a gap of 0.04 standard deviations),
so filled bars there resolve into one muddy shape and the visible colour is bin-by-bin Monte
Carlo noise rather than a difference between the placebos.

Colours follow the rest of the paper. Orange is the position reshuffle, as in the histograms
this replaces, blue is the reassignment placebo, and red marks the observed value.

Inputs are the draws already on disk, so nothing is re-simulated here:
  O*NET  aiChain_length_count/aiChains_task{Position,Assignment}Reshuffle_definition1.csv
         (row 0 of each is the observed run, rows 1.. are the 1,000 draws)
  PCF    apqc_chainLength_placebo/chain_length_placebo_{draws,summary}.csv
         (written by analysis/apqc_chainLength_placebo.py)
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = os.path.splitext(os.path.basename(__file__))[0]
FIG  = os.path.join(REPO, "writeup", "plots", NAME)
os.makedirs(FIG, exist_ok=True)

ORANGE = "orange"              # position reshuffle
BLUE   = plt.cm.tab10(0)       # execution-label reassignment
NBINS  = 34

onet = os.path.join(REPO, "data", "computed_objects", "aiChain_length_count")
pos  = pd.read_csv(f"{onet}/aiChains_taskPositionReshuffle_definition1.csv")
asg  = pd.read_csv(f"{onet}/aiChains_taskAssignmentReshuffle_definition1.csv")

pcf_dir = os.path.join(REPO, "data", "computed_objects", "apqc_chainLength_placebo")
pcf     = pd.read_csv(f"{pcf_dir}/chain_length_placebo_draws.csv")
pcf_sum = pd.read_csv(f"{pcf_dir}/chain_length_placebo_summary.csv")

PANELS = [
    ("onet", float(pos.iloc[0]["mean_chain_length"]),
     pos.iloc[1:]["mean_chain_length"].to_numpy(),
     asg.iloc[1:]["mean_chain_length"].to_numpy(), "Task"),
    ("apqc", float(pcf_sum["observed"].iloc[0]),
     pcf["reshuffle"].to_numpy(), pcf["reassign"].to_numpy(), "Step"),
]

for tag, obs, a, b, unit in PANELS:
    edges = np.histogram_bin_edges(np.concatenate([a, b]), bins=NBINS)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(a, bins=edges, histtype="stepfilled", color=ORANGE, alpha=.45, lw=0, zorder=1)
    ax.hist(b, bins=edges, histtype="stepfilled", color=BLUE,   alpha=.35, lw=0, zorder=1)
    ax.hist(a, bins=edges, histtype="step", color=ORANGE, lw=2.2, zorder=3,
            label=f"Shuffled {unit} Positions")
    ax.hist(b, bins=edges, histtype="step", color=BLUE, lw=2.2, zorder=3,
            label=f"Shuffled {unit} Execution Labels")
    ax.axvline(obs, color="red", linestyle="dashed", linewidth=2, zorder=4,
               label=f"Observed = {obs:.2f}")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("Average AI Chain Length", fontsize=19)
    ax.set_ylabel("Frequency", fontsize=16)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=12.5, loc="upper left")
    lo = min(a.min(), b.min()); pad = (obs - lo) * .10
    ax.set_xlim(lo - pad, obs + pad)          # the observed line anchors the right edge
    fig.tight_layout()
    fig.savefig(f"{FIG}/aiChains_chainLength_overlay_{tag}.png", dpi=300)
    plt.close()
    print(f"{tag:5} observed {obs:.3f} | position null {a.mean():.3f} (sd {a.std(ddof=1):.3f}) "
          f"| label null {b.mean():.3f} (sd {b.std(ddof=1):.3f})")
print("wrote 2 figures to", FIG)
