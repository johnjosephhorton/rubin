"""Prediction #3 on APQC sequences, pooling the cross-industry and 16 industry PCFs."""
import os, glob, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
MAIN = "/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin"
PCF  = f"{MAIN}/data/APQC_PCF"
OUT  = f"{MAIN}/data/computed_objects/apqc_pred3_industry"
os.makedirs(OUT, exist_ok=True)
MIN_STEPS = 5          # a unit needs >=5 steps for a focal step to have two neighbours on each side

def levl(h):
    p = h.split('.'); return 1 if (len(p) == 2 and p[-1] == '0') else len(p)

def leaves_of(path, framework):
    d = pd.read_excel(path, sheet_name='Combined', dtype={'Hierarchy ID': str})
    d = d.rename(columns={'Hierarchy ID': 'hid', 'Name': 'name', 'Element Description': 'desc'})
    if 'desc' not in d.columns:
        d['desc'] = ''
    d['hid'] = d['hid'].astype(str).str.strip(); d['name'] = d['name'].astype(str).str.strip()
    d['desc'] = d['desc'].fillna('').astype(str).str.strip()
    d = d[d['hid'].str.match(r'^[\d.]+$')].copy()
    parents = {h.rsplit('.', 1)[0] for h in d['hid'] if levl(h) > 1}
    d = d[~d['hid'].isin(parents)].copy()                       # leaves only
    d = d[d['hid'].map(levl) >= 3].copy()
    d['framework'] = framework
    d['unit'] = d['hid'].map(lambda h: '.'.join(h.split('.')[:2]))
    d['sk'] = d['hid'].map(lambda h: tuple(int(x) for x in h.split('.')))
    return d.sort_values('sk')

files = [('CrossIndustry', f"{PCF}/APQC_PCF_CrossIndustry_v8.0.xlsx")]
for p in sorted(glob.glob(f"{PCF}/K*.xlsx")):
    if 'Cross-In' in p: continue
    m = re.match(r'K\d+_\s*(?:BE_)?(.+?)(?:_v\d|_APQC|_\d{4}|\.xls)', os.path.basename(p))
    files.append(((m.group(1) if m else os.path.basename(p)).replace('_', ' ').strip()[:28], p))

frames = []
for fw, path in files:
    try:
        frames.append(leaves_of(path, fw))
    except Exception as e:
        print(f"  SKIP {fw}: {type(e).__name__}", flush=True)
L = pd.concat(frames, ignore_index=True)
print(f"leaf elements across {L.framework.nunique()} frameworks: {len(L):,}", flush=True)

# Deduplicate units: identical (unit name, ordered child names) contribute the same sequence twice
L['uid'] = L['framework'] + '||' + L['unit']
sig = L.groupby('uid').agg(sig=('name', lambda s: tuple(s)), n=('name', 'size')).reset_index()
sig = sig[sig['n'] >= MIN_STEPS]
sig['key'] = sig['sig'].map(hash)
keep = sig.drop_duplicates('key')['uid']
L = L[L['uid'].isin(set(keep))].reset_index(drop=True)
print(f"after dedup and >= {MIN_STEPS} steps: {len(L):,} steps in {L.uid.nunique():,} units", flush=True)

L['text'] = (L['name'] + ': ' + L['desc']).str.strip(': ')

# ---- O*NET side ----
o = pd.read_csv(f"{MAIN}/data/computed_objects/ONET_Eloundou_Anthropic_GPT/ONET_Eloundou_Anthropic_GPT.csv")
o = o.dropna(subset=['human_labels']).drop_duplicates('Task ID')[
    ['Task ID', 'Task Title', 'human_labels', 'label']].reset_index(drop=True)
print(f"labelled O*NET tasks: {len(o):,}", flush=True)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer(f"{MAIN}/data/sentence-transformers/all-mpnet-base-v2", device='cpu')
Et = model.encode(o['Task Title'].astype(str).tolist(), batch_size=64, normalize_embeddings=True,
                  show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
uniq = L['text'].drop_duplicates().reset_index(drop=True)
print(f"encoding {len(uniq):,} unique PCF step texts ...", flush=True)
El = model.encode(uniq.astype(str).tolist(), batch_size=64, normalize_embeddings=True,
                  show_progress_bar=False, convert_to_numpy=True).astype(np.float32)

best_i, best_s = np.empty(len(uniq), int), np.empty(len(uniq), np.float32)
for a in range(0, len(uniq), 2000):
    S = El[a:a+2000] @ Et.T
    best_i[a:a+2000] = S.argmax(1); best_s[a:a+2000] = S.max(1)
mp = pd.DataFrame({'text': uniq, 'match_task_id': o['Task ID'].values[best_i],
                   'similarity': best_s, 'human_labels': o['human_labels'].values[best_i],
                   'label': o['label'].values[best_i]})
L = L.merge(mp, on='text', how='left')
L.to_csv(f"{OUT}/industry_leaf_matches.csv", index=False)
print(f"matched. mean cosine {L.similarity.mean():.3f} | median {L.similarity.median():.3f}", flush=True)
print("RUN COMPLETE", flush=True)
