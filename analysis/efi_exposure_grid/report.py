"""Readable rendering of the grid, plus the validation check against the published table."""
import os, numpy as np, pandas as pd
_HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.abspath(os.path.join(_HERE,"..",".."))
R=pd.read_csv(os.path.join(REPO,"data","computed_objects","efi_exposure_grid","efi_exposure_grid.csv"))
star=lambda p:"***" if p<.01 else "**" if p<.05 else "*" if p<.1 else ""
CTRL=["+ # AI-able steps","+ # steps in workflow","no controls"]

print("VALIDATION against the table now in the draft (E1 or E2, + # AI-able steps, main prompt):")
v=R[(R.corpus=="O*NET main prompt")&(R.exposure=="E1 or E2")&(R.controls=="+ # AI-able steps")]
for _,r in v.iterrows():
    print(f"   {r.fe:<12} EFI {r.efi:+.3f} ({r.efi_se:.3f})   exposure {r.exp:+.3f} ({r.exp_se:.3f})   N={r.n}")
print("   draft table says: EFI -0.01/-0.09/-0.04, exposure 0.49/0.48/0.39, N=872\n")

def block(corpus, expo, fes):
    d=R[(R.corpus==corpus)&(R.exposure==expo)]
    n=int(d.n.iloc[0])
    print("="*104)
    print(f"  {corpus}   |   exposure and EFI both measured on {expo}   |   N = {n}")
    print("="*104)
    print(f"  {'controls':<24}{'':<10}"+"".join(f"{f:>22}" for f in fes))
    print("  "+"-"*100)
    for c in CTRL:
        for term,lab in [("efi","EFI"),("exp","AI exposure")]:
            cells=[]
            for f in fes:
                r=d[(d.controls==c)&(d.fe==f)].iloc[0]
                cells.append(f"{r[term]:+.3f}{star(r[term+'_p'])} ({r[term+'_se']:.3f})")
            print(f"  {c if term=='efi' else '':<24}{lab:<10}"+"".join(f"{x:>22}" for x in cells))
        print()

ONET_FE=["no FE","SOC major","SOC minor"]; PCF_FE=["no FE","PCF category","framework"]
for expo in ["E1 only","E1 or E2"]:
    block("O*NET main prompt",expo,ONET_FE)
for expo in ["E1 only","E1 or E2"]:
    block("APQC PCF",expo,PCF_FE)

print("="*104)
print("  O*NET, 10 ALTERNATIVE PROMPT ORDERINGS   |   EFI coefficient across prompts")
print("="*104)
print(f"  {'exposure':<12}{'controls':<24}{'FE':<14}{'median':>9}{'min':>9}{'max':>9}"
      f"{'neg & sig 5%':>14}{'sig either way':>16}")
print("  "+"-"*100)
for expo in ["E1 only","E1 or E2"]:
    for c in CTRL:
        for f in ONET_FE:
            d=R[(R.corpus=="O*NET alt prompts")&(R.exposure==expo)&(R.controls==c)&(R.fe==f)]
            neg=((d.efi<0)&(d.efi_p<.05)).sum(); sig=(d.efi_p<.05).sum()
            print(f"  {expo:<12}{c:<24}{f:<14}{d.efi.median():>+9.3f}{d.efi.min():>+9.3f}"
                  f"{d.efi.max():>+9.3f}{str(neg)+' of 10':>14}{str(sig)+' of 10':>16}")
    print()
