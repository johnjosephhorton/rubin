"""Coefficient plot for the whole grid."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
_HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.abspath(os.path.join(_HERE,"..",".."))
NAME=os.path.basename(_HERE)
R=pd.read_csv(os.path.join(REPO,"data","computed_objects",NAME,"efi_exposure_grid.csv"))
FIG=os.path.join(REPO,"writeup","plots",NAME); os.makedirs(FIG,exist_ok=True)

CTRL=["+ # AI-able steps","+ # steps in workflow","no controls"]
CSHORT={"+ # AI-able steps":"# AI-able","+ # steps in workflow":"# steps","no controls":"none"}
ROWS=[("O*NET main prompt",["no FE","SOC major","SOC minor"]),
      ("O*NET alt prompts",["no FE","SOC major","SOC minor"]),
      ("APQC PCF",["no FE","PCF category","framework"])]
EXPO=["E1 only","E1 or E2"]
COL={"E1 only":"#b45309","E1 or E2":"#1d4ed8"}

fig,axes=plt.subplots(3,2,figsize=(13,11),sharex="row")
for i,(corpus,fes) in enumerate(ROWS):
    for j,expo in enumerate(EXPO):
        ax=axes[i,j]; d=R[(R.corpus==corpus)&(R.exposure==expo)]
        labels=[]; y=0; ticks=[]
        for fe in fes:
            for c in CTRL:
                s=d[(d.fe==fe)&(d.controls==c)]
                if corpus=="O*NET alt prompts":
                    ax.scatter(s.efi,[y]*len(s),s=16,color=COL[expo],alpha=.45,zorder=3)
                    ax.scatter([s.efi.median()],[y],s=80,color=COL[expo],marker="D",
                               edgecolor="white",linewidth=.8,zorder=4)
                else:
                    r=s.iloc[0]
                    ax.plot([r.efi_lo,r.efi_hi],[y,y],color=COL[expo],lw=2,alpha=.75,zorder=3)
                    ax.scatter([r.efi],[y],s=70,color=COL[expo],zorder=4,
                               edgecolor="white",linewidth=.8)
                labels.append(f"{fe}  /  {CSHORT[c]}"); ticks.append(y); y-=1
            y-=0.5
        ax.axvline(0,color="0.25",lw=1)
        ax.set_yticks(ticks); ax.set_yticklabels(labels,fontsize=8.5)
        ax.set_ylim(y+0.7,1)
        ax.grid(axis="x",alpha=.25,lw=.6)
        if corpus=="O*NET alt prompts":
            nlab=f"N={int(d.n.min())}-{int(d.n.max())}"; extra=" (10 orderings; diamond = median)"
        else:
            nlab=f"N={int(d.n.iloc[0])}"; extra=""
        ax.set_title(f"{corpus}  |  {expo}  |  {nlab}{extra}",fontsize=10,color=COL[expo])
        for sp in ("top","right"): ax.spines[sp].set_visible(False)
        if i==2: ax.set_xlabel("Empirical fragmentation index coefficient (SD units)",fontsize=9.5)
fig.suptitle("Fragmentation coefficient across corpora, exposure definitions, controls and fixed effects\n"
             "model predicts a negative coefficient; bars are 95% confidence intervals",
             fontsize=12.5,y=0.985)
fig.tight_layout(rect=[0,0,1,0.955])
p=os.path.join(FIG,"efi_exposure_grid.png"); fig.savefig(p,dpi=300,bbox_inches="tight")
print("wrote",p)
