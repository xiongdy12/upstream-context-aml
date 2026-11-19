"""Efficiency benchmark: uniform timing of every method on the same machine and the same split.

Measured
  (a) feature/structure extraction from scratch into a temporary folder:
      1-hop context (02), 2-hop (06), 3-hop (10), GNN ego-graphs (08), Deep Sets sender sets (12)
  (b) training + inference on group_src fold 0 for:
      XGBoost (internal+ctx1+ctx2+ctx3), GNN (GINE), Deep Sets ds1 (RevClassify), ds2 (ours), ds3
      -> wall-clock train time, epochs, inference time per 1,000 subgraphs, #parameters, peak GPU memory

  python 14_efficiency.py --data dataset --feat features --tuned results_tuned --out results_efficiency
  (add --skip_extraction to reuse cached features and time only training/inference)
"""
import argparse
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.metrics import average_precision_score

from groups import build_groups, make_splits

HERE = Path(__file__).parent


def load_mod(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ev, gn, rc = load_mod("ev", "07_eval.py"), load_mod("gn", "08_gnn.py"), load_mod("rc", "12_revclassify.py")


def run(cmd):
    t = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--feat", default="features")
    ap.add_argument("--tuned", default="results_tuned", help="folder with best_configs.csv from 13_tune.py")
    ap.add_argument("--out", default="results_efficiency")
    ap.add_argument("--tmp", default="features_bench")
    ap.add_argument("--skip_extraction", action="store_true")
    ap.add_argument("--keep_tmp", action="store_true")
    ap.add_argument("--gnn_epochs", type=int, default=40)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    py = sys.executable
    prep = {}

    # ------------------------------------------------------------------ (a) extraction from scratch
    fdir = Path(a.feat)
    if not a.skip_extraction:
        fdir = Path(a.tmp)
        if fdir.exists():
            shutil.rmtree(fdir)
        prep["1-hop context (02)"] = run([py, str(HERE / "02_build_features.py"), "--data", a.data, "--out", str(fdir)])
        prep["2-hop context (06)"] = run([py, str(HERE / "06_context2hop.py"), "--data", a.data, "--out", str(fdir)])
        prep["3-hop context (10)"] = run([py, str(HERE / "10_context3hop.py"), "--data", a.data, "--out", str(fdir)])
    t = time.time()
    G = gn.build_graphs(a.data, str(fdir), 32, fdir / "bench_gnn_graphs.npz" if not a.skip_extraction
                        else fdir / "gnn_graphs_M32.npz")
    prep["GNN ego-graphs (08)"] = time.time() - t
    t = time.time()
    S = rc.build_sets(a.data, str(fdir), 128, 16, fdir / "bench_revsets.npz" if not a.skip_extraction
                      else fdir / "revsets_K128_Kh16.npz")
    prep["Deep Sets sender sets (12)"] = time.time() - t
    if a.skip_extraction:
        prep = {k: (v if "(08)" not in k and "(12)" not in k else np.nan) for k, v in prep.items()}
        print("--skip_extraction: preprocessing times are not measured (cached files)", flush=True)
    print("preprocessing (s):", {k: round(v, 1) for k, v in prep.items()}, flush=True)

    # ------------------------------------------------------------------ (b) training + inference, same fold
    df, blocks = ev.load(a.feat)
    y = df["label"].values.astype(int)
    lab = pl.read_parquet(Path(a.feat) / "labels.parquet").sort("row_order")
    groups = pl.DataFrame({"ccId": lab["ccId"]}).join(build_groups(a.data, a.feat, Ks=(10,)), on="ccId", how="left")
    split, fold, tr, va, te = next(make_splits(y, groups, [0], ["group_src"]))
    rows = []

    # XGBoost
    import xgboost as xgb
    cols = ev.feature_sets(blocks).get("internal+ctx1+ctx2+ctx3") or ev.feature_sets(blocks)["internal+ctx1+ctx2"]
    X = df[cols].astype("float32")
    pw = (len(tr) - y[tr].sum()) / max(y[tr].sum(), 1)
    m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.6,
                          tree_method="hist", device="cpu" if a.cpu else "cuda", scale_pos_weight=pw,
                          eval_metric="aucpr", early_stopping_rounds=150, random_state=0)
    t = time.time()
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
    tt = time.time() - t
    t = time.time()
    p = m.predict_proba(X.iloc[te])[:, 1]
    ti = time.time() - t
    rows.append({"method": "XGBoost + 1/2/3-hop context", "preprocessing": "1+2+3-hop context",
                 "train_s": tt, "epochs_or_trees": m.best_iteration + 1, "infer_ms_per_1k": 1000 * ti / len(te) * 1000,
                 "params": np.nan, "peak_gpu_GB": np.nan, "pr_auc": average_precision_score(y[te], p)})
    print(rows[-1], flush=True)

    # GNN
    gargs = SimpleNamespace(hidden=128, layers=3, dropout=0.2, lr=1e-3, bs=512, epochs=a.gnn_epochs, patience=6)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    model = gn.train_one(G, tr, va, gargs, device, 0)
    sync()
    tt = time.time() - t
    t = time.time()
    p = gn.predict(model, G, te, 1024, device)
    sync()
    ti = time.time() - t
    rows.append({"method": "GNN (GINE, 1-hop ego-graph)", "preprocessing": "GNN ego-graphs", "train_s": tt,
                 "epochs_or_trees": np.nan, "infer_ms_per_1k": 1000 * ti / len(te) * 1000, "params": n_params(model),
                 "peak_gpu_GB": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else np.nan,
                 "pr_auc": average_precision_score(y[te], p)})
    print(rows[-1], flush=True)

    # Deep Sets variants with tuned configs
    best = pd.read_csv(Path(a.tuned) / "best_configs.csv").set_index("variant")
    variants = {"ds1": (["S1", "R1"], "RevClassify-DS (1-hop, reimpl.)"),
                "ds2": (["S1", "R1", "S2"], "Ours: 2-hop upstream Deep Sets"),
                "ds3": (["S1", "R1", "S2", "S3"], "Ours: 3-hop upstream Deep Sets")}
    for v, (names, label) in variants.items():
        b = best.loc[v] if v in best.index else best.loc["ds2"]
        cfg = SimpleNamespace(hidden=int(b.hidden), dropout=float(b.dropout), lr=float(b.lr), bs=1024,
                              epochs=200, patience=15)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t = time.time()
        p, best_va, eps = rc.train_eval(S, names, tr, va, te, cfg, device, 0)
        sync()
        tt = time.time() - t
        mdl = rc.DeepSets(names, S["X"].shape[1] + 1, cfg.hidden, cfg.dropout).to(device)
        t = time.time()
        rc.predict(mdl, S, names, te, 4096, device)
        sync()
        ti = time.time() - t
        sz = {n: float(np.diff(S[f"{n}_ptr"]).mean()) for n in names}
        rows.append({"method": label, "preprocessing": "sender sets", "train_s": tt, "epochs_or_trees": eps,
                     "infer_ms_per_1k": 1000 * ti / len(te) * 1000, "params": n_params(mdl),
                     "peak_gpu_GB": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else np.nan,
                     "pr_auc": average_precision_score(y[te], p), "mean_set_sizes": sz})
        print(rows[-1], flush=True)

    res = pd.DataFrame(rows)
    g = lambda k: prep.get(k, np.nan)
    need = {  # extraction steps each method depends on (02 also writes bg_touch used by all graph methods)
        "XGBoost + 1/2/3-hop context": g("1-hop context (02)") + g("2-hop context (06)") + g("3-hop context (10)"),
        "GNN (GINE, 1-hop ego-graph)": g("1-hop context (02)") + g("GNN ego-graphs (08)"),
        "RevClassify-DS (1-hop, reimpl.)": g("1-hop context (02)") + g("Deep Sets sender sets (12)"),
        "Ours: 2-hop upstream Deep Sets": g("1-hop context (02)") + g("2-hop context (06)") + g("Deep Sets sender sets (12)"),
        "Ours: 3-hop upstream Deep Sets": g("1-hop context (02)") + g("2-hop context (06)") + g("3-hop context (10)")
        + g("Deep Sets sender sets (12)"),
    }
    res.insert(2, "preprocess_total_s", res["method"].map(need))
    res.to_csv(out / "efficiency.csv", index=False)
    pre = pd.Series(prep, name="seconds").round(1).to_frame()
    pre.to_csv(out / "preprocessing.csv")
    show = res.drop(columns=[c for c in ("mean_set_sizes",) if c in res.columns]).round(
        {"preprocess_total_s": 1, "train_s": 1, "infer_ms_per_1k": 2, "peak_gpu_GB": 2, "pr_auc": 3})
    md = ["# Efficiency (group_src fold 0, same machine)\n",
          f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}; "
          f"subgraphs: train {len(tr):,} / val {len(va):,} / test {len(te):,}\n",
          "## One-off preprocessing (seconds, from scratch)\n", pre.to_markdown(),
          "\n\n## Training and inference\n", show.to_markdown(index=False),
          "\n\nNote: preprocess_total_s is an upper bound (step 02 also builds features not needed by every method). "
          "XGBoost inference time excludes feature extraction; Deep Sets/GNN inference excludes "
          "set/graph construction (both reported above as preprocessing)."]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    if not a.skip_extraction and not a.keep_tmp:
        shutil.rmtree(fdir, ignore_errors=True)


if __name__ == "__main__":
    main()
