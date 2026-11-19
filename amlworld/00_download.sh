#!/usr/bin/env bash
# Unduh IBM AMLworld HI-Small dari Kaggle (ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
# Hanya file HI-Small (~0,5 GB) yang diunduh; token Kaggle sama dengan Elliptic2 (~/.kaggle/access_token).
set -euo pipefail
mkdir -p data && cd data
DS=ealtman2019/ibm-transactions-for-anti-money-laundering-aml
kaggle datasets files "$DS" | head -30
for f in HI-Small_Trans.csv HI-Small_Patterns.txt; do
  kaggle datasets download -d "$DS" -f "$f" -p . --force
  [ -f "$f.zip" ] && unzip -o "$f.zip" && rm -f "$f.zip"
done
ls -lh
head -3 HI-Small_Trans.csv
