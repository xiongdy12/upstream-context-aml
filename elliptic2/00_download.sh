#!/usr/bin/env bash
# Download Elliptic2 from the official Elliptic Kaggle account (ellipticco/elliptic2-data-set, ~88 GB).
# Requires a Kaggle API token in ~/.kaggle/access_token (or KAGGLE_API_TOKEN) and >= 200 GB free disk.
set -euo pipefail
mkdir -p dataset
df -h . | tail -1
kaggle datasets download -d ellipticco/elliptic2-data-set -p dataset
cd dataset && unzip -o elliptic2-data-set.zip && rm -f elliptic2-data-set.zip && ls -lh
