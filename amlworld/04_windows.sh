#!/usr/bin/env bash
# Sensitivitas jendela waktu konteks hulu: 6 jam, 72 jam, seluruh riwayat (0).
# Hasil 24 jam sudah ada di results/. Tiap jendela: bangun fitur + XGBoost 3 seed (tanpa GFP).
set -euo pipefail
for W in 6 72 0; do
  echo "=== jendela ${W} jam (0 = seluruh riwayat) ==="
  python 01_build_features.py --data data/HI-Small_Trans.csv --out features_w${W} --window_hours ${W}
  python 02_eval.py --feat features_w${W} --out results_w${W} --seeds 0,1,2 \
    --sets base+ctx1,base+ctx1+ctx2,base+ctx1+ctx2+ctx3
done
for W in 6 72 0; do echo "### jendela ${W} jam"; head -8 results_w${W}/summary.md | tail -5; done
