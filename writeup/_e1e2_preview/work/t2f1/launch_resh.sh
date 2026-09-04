#!/bin/bash
cd "/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin/writeup/_e1e2_preview"
for r in 1:125 126:250 251:375 376:500 501:625 626:750 751:875 876:999; do
  lo="${r%%:*}"; hi="${r##*:}"
  nohup python3 scripts/p2_reshuffle.py "$lo" "$hi" E1E2 "work/t2f1/resh/E1E2_${lo}_${hi}.csv" > "work/t2f1/resh/E1E2_${lo}_${hi}.log" 2>&1 &
done
wait
