"""Two-panel sensitivity plot for the PCF similarity floor."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.abspath(os.path.join(_HERE,"..",".."))
NAME=os.path.basename(_HERE)
R=pd.read_csv(os.path.join(REPO,"data","computed_objects",NAME,"apqc_similarity_threshold.csv"))
FIG=os.path.join(REPO,"writeup","plots",NAME); os.makedirs(FIG,exist_ok=True)
CHOSEN=0.71

fig,axes=plt.subplots(1,2,figsize=(13.5,5.0))

ax=axes[0]
ax.plot(R.threshold,R.chain,"o-",color="#1d4ed8",lw=2,ms=6,label="observed")
ax.plot(R.threshold,R.chain_null,"s--",color="0.55",lw=1.6,ms=5,label="within-group reshuffle null")
ax.fill_between(R.threshold,R.chain_null-2*R.chain_null_sd,R.chain_null+2*R.chain_null_sd,
                color="0.75",alpha=.35,lw=0)
for _,r in R.iterrows():
    ax.annotate(f"z={r.chain_z:.1f}",(r.threshold,r.chain),textcoords="offset points",
                xytext=(0,9),ha="center",fontsize=9,color="#1d4ed8")
ax.axvline(CHOSEN,color="#b45309",lw=1.4,ls=":")
ax.text(CHOSEN,ax.get_ylim()[1],"  Chosen Threshold",color="#b45309",fontsize=11,va="top")
ax.set_xlabel("Cosine Similarity Floor",fontsize=13); ax.set_ylabel("Average AI Chain Length",fontsize=13)
ax.tick_params(labelsize=11.5)
ax.set_title("Prediction #1: Chain Length Against Its Reshuffle Null",fontsize=13.5)
ax.legend(fontsize=11,frameon=False); ax.grid(alpha=.25,lw=.6)
for s in ("top","right"): ax.spines[s].set_visible(False)

ax=axes[1]
for tag,lab,col,off in [("noFE","no FE","#1d4ed8",-0.0026),
                        ("cat","PCF category FE","#059669",0.0),
                        ("fw","framework FE","#b45309",0.0026)]:
    x=R.threshold+off
    ax.errorbar(x,R[f"efi_{tag}"],yerr=1.645*R[f"efi_se_{tag}"],fmt="o",ms=5,lw=1.5,
                capsize=2,color=col,label=lab,alpha=.9)
ax.axhline(0,color="0.25",lw=1)
ax.axvline(CHOSEN,color="#b45309",lw=1.4,ls=":")
ax.text(CHOSEN,ax.get_ylim()[1],"  Chosen Threshold",color="#b45309",fontsize=11,va="top")
ax.set_xlabel("Cosine Similarity Floor",fontsize=13)
ax.set_ylabel("Fragmentation Coefficient (SD Units)",fontsize=13)
ax.tick_params(labelsize=11.5)
ax.set_title("Prediction #3: Fragmentation Coefficient, 90% CI",fontsize=13.5)
ax.legend(fontsize=11,frameon=False); ax.grid(alpha=.25,lw=.6)
for s in ("top","right"): ax.spines[s].set_visible(False)

fig.suptitle("PCF results as the label-transfer similarity floor moves; the 525 process groups are fixed throughout",
             fontsize=14,y=0.99)
fig.tight_layout(rect=[0,0,1,0.94])
p=os.path.join(FIG,"apqc_similarity_threshold.png"); fig.savefig(p,dpi=300,bbox_inches="tight")
print("wrote",p)
