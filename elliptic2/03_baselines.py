"""Baseline tabular pada Elliptic2 dengan protokol ketat.

Model   : RandomForest, LightGBM, XGBoost (GPU jika ada)
Fitur   : internal  vs  internal+context (graf latar 1-hop)
Split   : 'orig'  -> 80/10/10 urutan baris (meniru repo resmi: indeks mod 10, 8=val, 9=test)
          'strat' -> 80/10/10 stratified acak, beberapa seed
Metrik  : PR-AUC (utama), ROC-AUC, F1 (ambang dipilih di val), Precision/Recall@1% (anggaran
          investigasi: 1% subgraf dengan skor tertinggi di test)

  python 03_baselines.py --feat features --seeds 0,1,2,3,4 --out results_baseline
"""
import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split


def load(feat_dir, use_context):
    f = Path(feat_dir)
    df = pl.read_parquet(f / "labels.parquet").join(pl.read_parquet(f / "internal.parquet"), on="ccId")
    if use_context and (f / "context.parquet").exists():
        df = df.join(pl.read_parquet(f / "context.parquet"), on="ccId", how="left")
    df = df.sort("row_order").to_pandas()
    X = df.drop(columns=["ccId", "label", "row_order"]).astype("float32")
    return X, df["label"].values.astype(int), df["row_order"].values


def splits(y, row_order, seeds):
    r = row_order % 10
    yield "orig", 0, np.where(r <= 7)[0], np.where(r == 8)[0], np.where(r == 9)[0]
    idx = np.arange(len(y))
    for s in seeds:
        tr, tmp = train_test_split(idx, test_size=0.2, stratify=y, random_state=s)
        va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=s)
        yield "strat", s, tr, va, te


def models(pos_weight, seed, gpu):
    import lightgbm as lgb
    import xgboost as xgb
    return {
        "RF": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                                     class_weight="balanced_subsample", random_state=seed),
        "LightGBM": lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                                       scale_pos_weight=pos_weight, random_state=seed,
                                       n_jobs=-1, verbose=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=8,
                                     subsample=0.8, colsample_bytree=0.6, tree_method="hist",
                                     device="cuda" if gpu else "cpu", scale_pos_weight=pos_weight,
                                     eval_metric="aucpr", early_stopping_rounds=100,
                                     random_state=seed),
    }


def fit_predict(name, m, Xtr, ytr, Xva, yva, Xte):
    if name == "LightGBM":
        import lightgbm as lgb
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    elif name == "XGBoost":
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        m.fit(Xtr.fillna(-1), ytr)
        Xva, Xte = Xva.fillna(-1), Xte.fillna(-1)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], m


def metrics(y, p, thr, budget=0.01):
    k = max(1, int(round(budget * len(y))))
    top = np.argsort(-p)[:k]
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "f1": f1_score(y, p >= thr), "prec@1%": y[top].mean(),
            "rec@1%": y[top].sum() / max(y.sum(), 1)}


def best_thr(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    return th[np.nanargmax(f[:-1])] if len(th) else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="features")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--models", default="RF,LightGBM,XGBoost")
    ap.add_argument("--out", default="results_baseline")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    rows, t0 = [], time.time()
    for fs in ("internal", "internal+context"):
        X, y, ro = load(a.feat, fs != "internal")
        if fs != "internal" and X.shape[1] == load(a.feat, False)[0].shape[1]:
            print("context.parquet tidak ada -> lewati internal+context")
            continue
        print(f"\n### fitur={fs}  X={X.shape}  positif={y.sum()} ({y.mean():.2%})")
        for split, seed, tr, va, te in splits(y, ro, seeds):
            pw = (len(tr) - y[tr].sum()) / max(y[tr].sum(), 1)
            for name, m in models(pw, seed, not a.cpu).items():
                if name not in a.models.split(","):
                    continue
                pva, pte, fitted = fit_predict(name, m, X.iloc[tr], y[tr], X.iloc[va], y[va], X.iloc[te])
                r = metrics(y[te], pte, best_thr(y[va], pva))
                rows.append({"features": fs, "split": split, "seed": seed, "model": name, **r})
                print(f"  [{time.time() - t0:6.0f}s] {split:<5} s{seed} {name:<8} "
                      f"PR-AUC={r['pr_auc']:.3f} ROC-AUC={r['roc_auc']:.3f} F1={r['f1']:.3f} "
                      f"P@1%={r['prec@1%']:.3f}", flush=True)
                if split == "orig" and name == "LightGBM":
                    imp = pd.Series(fitted.booster_.feature_importance("gain"), index=X.columns)
                    imp.sort_values(ascending=False).head(40).to_csv(out / f"importance_{fs}.csv")

    res = pd.DataFrame(rows)
    res.to_csv(out / "scores.csv", index=False)
    orig = res[res.split == "orig"].pivot_table(index=["features", "model"],
                                                values=["pr_auc", "roc_auc", "f1", "prec@1%", "rec@1%"])
    st = res[res.split == "strat"].groupby(["features", "model"])[["pr_auc", "roc_auc", "f1", "prec@1%"]]
    strat = st.mean().round(3).astype(str) + " ± " + st.std().fillna(0).round(3).astype(str)
    md = ["# Baseline Elliptic2\n",
          "Referensi: GLASS tanpa fitur (paper Elliptic2) PR-AUC 0.208 / ROC-AUC 0.889; "
          "RF (Akter dkk. 2026) PR-AUC 0.494 / ROC-AUC 0.923.\n",
          "## Split asli (80/10/10 urutan baris)\n", orig.round(3).to_markdown(),
          f"\n\n## Stratified acak, {len(seeds)} seed (mean ± std)\n", strat.to_markdown()]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    json.dump(vars(a), open(out / "args.json", "w"), indent=1)


if __name__ == "__main__":
    main()
