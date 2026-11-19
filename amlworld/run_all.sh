#!/usr/bin/env bash
# Full AMLworld HI-Small pipeline (run from this folder).
set -euo pipefail
bash 00_download.sh
python 01_build_features.py --data data/HI-Small_Trans.csv --out features --window_hours 24
python 02_eval.py --feat features --out results --seeds 0,1,2
bash 04_windows.sh                                             # 6 h / 72 h / full-history windows
python 03_gfp.py --data data/HI-Small_Trans.csv --feat features --threads 32   # needs: pip install snapml
python 02_eval.py --feat features --out results_gfp --seeds 0,1,2 \
  --sets base+gfp,base+gfp+ctx1,base+gfp+ctx1+ctx2,base+gfp+ctx1+ctx2+ctx3
