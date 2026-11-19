# Multi-hop Upstream Context for Robust Money-Laundering Detection

Code accompanying the paper *(under preparation for Knowledge and Information Systems)*.

**Main finding.** Sender-based subgraph classifiers for anti-money laundering (AML), such as
RevClassify (Song et al., ICAIF 2024), look near-perfect under the random splits used in the
literature but become unstable when evaluated on **funding sources never seen during training**.
Adding **multi-hop upstream context** (the senders of the senders) to a Deep Sets classifier
restores robustness on unseen sources and matches the published random-split performance.

## Results

### Elliptic2 (real Bitcoin data, subgraph classification, PR-AUC)

`strat` = random stratified 80/10/10 (protocol used in prior work);
`group_src` = **source-disjoint** split: subgraphs grouped by their dominant funding source,
StratifiedGroupKFold, so test sources never appear in training.

| Model | strat | group_src (5 folds) | group_src (15 folds) |
|---|---|---|---|
| XGBoost, subgraph-internal features | 0.568 | 0.248 | |
| XGBoost + 1-hop context | 0.893 | 0.656 | |
| XGBoost + 1/2-hop context | 0.930 | 0.816 | |
| XGBoost + 1/2/3-hop context | 0.934 | 0.832 | |
| GNN (GINE, full features, 1-hop ego-graph) | 0.934 | 0.621 | |
| RevClassify-DS, our reimplementation, tuned | 0.950 | 0.823 | 0.826 ± 0.137 |
| **Ours: 2-hop upstream Deep Sets, tuned** | **0.973** | **0.908** | **0.928 ± 0.039** |
| RevClassify-DS, as reported by Song et al. (2024) | 0.974 | n/a | n/a |

On the 15 source-disjoint folds (3 shuffles × 5 folds), ours beats RevClassify-DS on 13/15 folds,
mean ΔPR-AUC = +0.102, one-sided Wilcoxon signed-rank p = 0.0016; worst fold 0.864 vs 0.454.
Hyper-parameters of both models were selected on validation data only (`13_tune.py`).
A third hop adds no significant gain over two hops.

### IBM AMLworld HI-Small (synthetic, ground-truth labels, transaction classification)

Temporal 60/20/20 split, minority-class F1, XGBoost, 3 seeds. All context features are causal
(only transactions strictly before the predicted one; verified by brute force in the tests).

| Features | F1 (minority) |
|---|---|
| transaction features only | 0.232 |
| + 1-hop upstream context (24 h window) | 0.434 |
| + 1-hop upstream context (72 h window) | 0.480 |
| GFP pattern features (IBM Graph Feature Preprocessor), our run | 0.550 |
| **GFP + 1-hop upstream context** | **0.607** |

Deeper hops do not help on AMLworld. Our numbers are below the current state of the art on this
benchmark (e.g. Multi-PNA, 68.16 F1); the dataset is used to test whether the upstream-context
effect holds with simulator ground-truth labels and is complementary to pattern features.

### Efficiency (Elliptic2, one NVIDIA B200, group_src fold 0)

Background graph: 49.3M nodes, 196.2M edges; 83,664 / 13,799 / 24,347 train / val / test subgraphs.
Measured from scratch with `elliptic2/14_efficiency.py`.

| Method | Preprocessing (s) | Training (s) | Inference (ms / 1k subgraphs) | Parameters | Peak GPU memory (GB) |
|---|---|---|---|---|---|
| XGBoost + 1/2/3-hop context | 157.7 | 11.1 | 9.6 | — | — |
| GNN (GINE, 1-hop ego-graph) | 32.5 | 30.7 | 7.9 | 0.45 M | 0.18 |
| RevClassify-DS (reimpl., tuned) | 36.7 | 30.6 | 1.9 | 0.18 M | 0.05 |
| **Ours: 2-hop upstream Deep Sets (tuned)** | **71.4** | **45.1** | **4.1** | **1.02 M** | **0.26** |
| 3-hop upstream Deep Sets | 166.2 | 122.8 | 16.6 | 1.36 M | 1.12 |

Preprocessing is an upper bound (one pass over the 83 GB edge file per hop, plus set/graph construction).
Parameter counts differ mainly because validation-based tuning selected hidden size 256 for ours and 128 for RevClassify-DS.

## Repository layout

```
elliptic2/                 Elliptic2 pipeline (run scripts from inside this folder)
  00_download.sh           download from Kaggle (ellipticco/elliptic2-data-set, ~88 GB)
  01_inspect.py            schema check
  02_build_features.py     internal + 1-hop context features (single pass over 196M background edges)
  03_baselines.py          RF / LightGBM / XGBoost baselines
  04_glass.sh              official GLASS baseline (optional)
  05_validity.py           shortcut / leakage checks: shared neighbours, group splits, ablations
  06_context2hop.py        2-hop upstream context features
  07_eval.py               XGBoost / LightGBM under strat, group_src, group_K10
  08_gnn.py                GINE GNN on ego-graphs with full node and edge features
  09_ensemble.py           rank ensembles + paired bootstrap tests
  10_context3hop.py        3-hop upstream context features
  11_analysis.py           block ablation, top-k shortcut test, TreeSHAP
  12_revclassify.py        RevClassify-DS reimplementation (ds1) + multi-hop variants (ds2, ds3)
  13_tune.py               validation-only hyper-parameter tuning + repeated source-disjoint CV + Wilcoxon test
  14_efficiency.py         uniform timing / memory benchmark
  groups.py, common.py     split construction (identical for all models), schema helpers
  run_all.sh               full pipeline in order
amlworld/                  IBM AMLworld HI-Small pipeline
  01_build_features.py     causal 1/2/3-hop upstream context (cumulative sums + as-of joins)
  02_eval.py               temporal split evaluation
  03_gfp.py                IBM GFP baseline (snapml)
  04_windows.sh            window-length sensitivity
tests/ (in each folder)    small synthetic data with the real schema, for pipeline testing only
```

## Reproducing

Hardware used: one NVIDIA B200 GPU, 2 TB RAM, Python 3.12. Elliptic2 needs about 200 GB free disk.

```bash
pip install -r requirements.txt
# Kaggle API token in ~/.kaggle/access_token

cd elliptic2 && bash 00_download.sh && bash run_all.sh
cd ../amlworld && pip install snapml && bash run_all.sh
```

Every model is evaluated on exactly the same splits (`groups.make_splits`), and every script
writes per-fold scores and test predictions so that tables and tests can be recomputed.

## Data

- **Elliptic2**: Bellei et al., 2024, *The Shape of Money Laundering: Subgraph Representation
  Learning on the Blockchain with the Elliptic2 Dataset*. Downloaded from Elliptic's Kaggle account;
  see the Kaggle page for its terms of use. Not redistributed here.
- **IBM AMLworld**: Altman et al., 2023, *Realistic Synthetic Financial Transactions for
  Anti-Money Laundering Models*. Data under CDLA-Sharing-1.0. Not redistributed here.

## Acknowledgements

RevClassify / RevTrack (Song et al., ICAIF 2024; MITIBMxGraph/RevTrack), GLASS (Wang et al.),
IBM Graph Feature Preprocessor (snapml) and the Elliptic2 and AMLworld dataset authors.

## Citation

Citation will be added after publication.

## License

MIT (code only). Datasets keep their own licenses.
