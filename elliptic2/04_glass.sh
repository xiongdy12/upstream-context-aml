#!/usr/bin/env bash
# Baseline GLASS mengikuti panduan repo resmi Elliptic2 (MITIBMxGraph/Elliptic2).
# Paper Elliptic2 menjalankan GLASS TANPA fitur node/edge (memori GPU tidak cukup);
# angka acuan: test PR-AUC 0.208, ROC-AUC 0.889.
# Cek ulang langkah ini dengan README repo resmi bila ada perubahan.
set -euo pipefail
GPU=${GPU:-0}
[ -d Elliptic2 ] || git clone https://github.com/MITIBMxGraph/Elliptic2.git
cd Elliptic2
[ -d GLASS ] || git clone https://github.com/Xi-yuanWang/GLASS.git
ln -sfn ../dataset dataset
# preprocess_glass.py membaca ./dataset dan menulis edge_list.txt, subgraphs.pth, n2id.pkl
python preprocess_glass.py
mkdir -p GLASS/dataset/elliptic GLASS/config
mv -f edge_list.txt subgraphs.pth GLASS/dataset/elliptic/
cp -f elliptic.yml GLASS/config/
cd GLASS
python GLASSTest.py --use_deg --use_maxzeroone --repeat 5 --device "$GPU" --dataset elliptic \
  2>&1 | tee ../../results_glass.log
