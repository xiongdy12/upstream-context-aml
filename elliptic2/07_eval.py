"""Evaluasi utama Fase 2 (tabular): protokol split acak + group split (sumber dana baru).

Set fitur : internal | +ctx1 (konteks 1-hop) | +ctx1+ctx2 (1-hop & 2-hop) | +ctx2 saja
Model     : XGBoost (GPU), LightGBM (early stopping berbasis PR-AUC -> perbaikan dari Fase 1)
Split     : strat (3 seed), group_src (5 fold; METRIK UTAMA), group_K10 (5 fold)
Simpan    : skor per fold + prediksi test (untuk uji signifikansi & ensemble dengan GNN)

  python 07_eval.py --data dataset --feat features --out results_phase2
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from groups import build_groups, make_splits

warnings.filterwarnings("ignore")


def load(feat_dir):
    f = Path(feat_dir)
    df = pl.read_parquet(f / "labels.parquet")
    blocks = {}
    for name, fn in (("internal", "internal.parquet"), ("ctx1", "context.parquet"),
                     ("ctx2", "context2.parquet"), ("ctx3", "context3.parquet")):
        if (f / fn).exists():
            t = pl.read_parquet(f / fn)
            blocks[name] = [c for c in t.columns if c != "ccId"]
            df = df.join(t, on="ccId", how="left")
    return df.sort("row_order").to_pandas(), blocks


def feature_sets(blocks):
    fs = {"internal": blocks["internal"],
          "internal+ctx1": blocks["internal"] + blocks.get("ctx1", [])}
    if "ctx2" in blocks:
        fs["internal+ctx1+ctx2"] = blocks["internal"] + blocks["ctx1"] + blocks["ctx2"]
        fs["internal+ctx2"] = blocks["internal"] + blocks["ctx2"]
    if "ctx3" in blocks:
        fs["internal+ctx1+ctx2+ctx3"] = blocks["internal"] + blocks["ctx1"] + blocks["ctx2"] + blocks["ctx3"]
    return fs


def select_sets(fsets, spec):
    if spec in ("", "all"):
        return fsets
    keep = spec.split(",")
    missing = [k for k in keep if k not in fsets]
    if missing:
        raise SystemExit(f"set fitur tidak dikenal: {missing}; tersedia: {list(fsets)}")
    return {k: fsets[k] for k in keep}


def fit_predict(model, Xtr, ytr, Xva, yva, Xte, seed, gpu, lgb_threads=32):
    pw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    if model == "XGBoost":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=8, subsample=0.8,
                              colsample_bytree=0.6, tree_method="hist", device="cuda" if gpu else "cpu",
                              scale_pos_weight=pw, eval_metric="aucpr", early_stopping_rounds=150,
                              random_state=seed)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=63, min_child_samples=20,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                               scale_pos_weight=pw, metric="average_precision", random_state=seed,
                               n_jobs=lgb_threads, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])
    return m.predict_proba(Xte)[:, 1]


def rec_at(y, p, frac=0.01):
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-p)[:k]
    return y[top].sum() / max(y.sum(), 1), y[top].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--feat", default="features")
    ap.add_argument("--out", default="results_phase2")
    ap.add_argument("--models", default="XGBoost,LightGBM")
    ap.add_argument("--splits", default="strat,group_src,group_K10")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sets", default="all", help="daftar set fitur dipisah koma, atau 'all'")
    ap.add_argument("--lgb_threads", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df, blocks = load(a.feat)
    y = df["label"].values.astype(int)
    groups = build_groups(a.data, a.feat, Ks=(10,))
    groups = pl.DataFrame({"ccId": df["ccId"].values}).join(groups, on="ccId", how="left")
    fsets = select_sets(feature_sets(blocks), a.sets)
    print("set fitur:", {k: len(v) for k, v in fsets.items()})

    rows, preds, t0 = [], [], time.time()
    for split, fold, tr, va, te in make_splits(y, groups, [int(s) for s in a.seeds.split(",")],
                                               a.splits.split(",")):
        for fs, cols in fsets.items():
            X = df[cols].astype("float32")
            for model in a.models.split(","):
                t1 = time.time()
                p = fit_predict(model, X.iloc[tr], y[tr], X.iloc[va], y[va], X.iloc[te], fold, not a.cpu,
                                a.lgb_threads)
                fit_sec = time.time() - t1
                r1, p1 = rec_at(y[te], p)
                rows.append({"split": split, "fold": fold, "set": fs, "model": model,
                             "pr_auc": average_precision_score(y[te], p), "roc_auc": roc_auc_score(y[te], p),
                             "rec@1%": r1, "prec@1%": p1, "pos_test": int(y[te].sum()),
                             "fit_sec": round(fit_sec, 1), "n_feat": len(cols)})
                preds.append(pd.DataFrame({"split": split, "fold": fold, "set": fs, "model": model,
                                           "idx": te, "y": y[te], "p": p.astype("float32")}))
                print(f"[{time.time() - t0:6.0f}s] {split:<9} f{fold} {fs:<20} {model:<8} "
                      f"PR-AUC={rows[-1]['pr_auc']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)  # simpan bertahap

    res = pd.DataFrame(rows)
    pd.concat(preds).to_parquet(out / "test_preds.parquet", index=False)
    g = res.groupby(["set", "model", "split"])["pr_auc"]
    tab = (g.mean().round(3).astype(str) + " ± " + g.std().fillna(0).round(3).astype(str)).unstack("split")
    tab = tab[[c for c in a.splits.split(",") if c in tab.columns]]
    # selisih berpasangan (fold yang sama) terhadap 'internal'
    base = res[res.set == "internal"].set_index(["split", "fold", "model"])["pr_auc"]
    res["gain_vs_internal"] = res.apply(lambda r: r.pr_auc - base.get((r.split, r.fold, r.model), np.nan), axis=1)
    gain = res[res.set != "internal"].groupby(["set", "model", "split"])["gain_vs_internal"].agg(
        lambda v: f"{v.mean():+.3f} (menang {int((v > 0).sum())}/{len(v)} fold)").unstack("split")
    md = ["# Fase 2: evaluasi tabular\n",
          "Metrik utama: **group_src** (sumber dana yang belum pernah dilihat).\n",
          "## PR-AUC (mean ± std)\n", tab.to_markdown(),
          "\n\n## Selisih PR-AUC berpasangan vs fitur internal\n", gain.to_markdown(),
          "\n\n## Waktu latih rata-rata per fold (detik) & jumlah fitur\n",
          res.groupby(["set", "model"])[["fit_sec", "n_feat"]].mean().round(1).to_markdown()]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
