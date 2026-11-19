#!/usr/bin/env bash
# Full Elliptic2 pipeline (run from this folder; dataset/ must contain the five Elliptic2 CSV files).
set -euo pipefail
python 01_inspect.py --data dataset | tee inspect.log
python 02_build_features.py --data dataset --out features                 # 1-hop context (+ bg_touch.parquet)
python 06_context2hop.py   --data dataset --out features                  # 2-hop upstream context
python 10_context3hop.py   --data dataset --out features                  # 3-hop upstream context
python 03_baselines.py --feat features --seeds 0,1,2,3,4 --out results_baseline      # RF / LightGBM / XGBoost
python 05_validity.py  --feat features --data dataset --out results_validity          # shortcut & leakage checks
python 07_eval.py --data dataset --feat features --out results_phase2 --models XGBoost   # hop curve, 3 protocols
python 08_gnn.py  --data dataset --feat features --out results_gnn --splits group_src,strat --seeds 0,1,2
python 11_analysis.py --data dataset --feat features --out results_analysis            # ablation + SHAP
python 13_tune.py --data dataset --feat features --out results_tuned --variants ds1,ds2  # RevClassify vs ours (tuned)
python 13_tune.py --data dataset --feat features --out results_tuned_rep --tune_from results_tuned \
  --variants ds1,ds2 --eval_splits group_src --group_seeds 0,1,2                      # 15-fold source-disjoint test
python 14_efficiency.py --data dataset --feat features --tuned results_tuned --out results_efficiency
