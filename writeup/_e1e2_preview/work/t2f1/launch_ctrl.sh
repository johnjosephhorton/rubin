#!/bin/bash
cd "/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin/writeup/_e1e2_preview"
for r in 1:15 16:30 31:45 46:60; do
  lo="${r%%:*}"; hi="${r##*:}"
  python3 scripts/p2_reshuffle.py "$lo" "$hi" E1 "work/t2f1/resh/ctrlE1_${lo}_${hi}.csv" > "work/t2f1/resh/ctrlE1_${lo}_${hi}.log" 2>&1 &
done
wait
python3 - <<'PY'
import pandas as pd, glob
fs=sorted(glob.glob('work/t2f1/resh/ctrlE1_*.csv'))
d=pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
d.to_csv('work/t2f1/resh_test_E1.csv', index=False)
print('E1 control draws', sorted(d.draw.unique())[0], '-', sorted(d.draw.unique())[-1], len(d), 'rows')
PY
