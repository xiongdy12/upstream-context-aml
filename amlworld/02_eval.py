"""Evaluasi AMLworld HI-Small: split TEMPORAL 60/20/20 (urutan waktu), XGBoost GPU.

Set fitur (kurva hop, setara Elliptic2):
  base | base+ctx1 | base+ctx1+ctx2 | base+ctx1+ctx2+ctx3 | ctx1+ctx2+ctx3 (tanpa base)
Metrik: F1 kelas minoritas (ambang dipilih di validasi; metrik standar AMLworld), PR-AUC, ROC-AUC.
Juga: porsi SHAP per blok untuk set terlengkap (seed 0).

  python 02_eval.py --feat features --out results --seeds 0,1,2
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

warnings.filterwarnings("ignore")


def load(feat):
    f = Path(feat)
    d = pl.read_parquet(f / "meta.parquet").select("tx", "t", "label")
    blocks = {}
    for name in ("base", "ctx1", "ctx2", "ctx3", "gfp"):
        if not (f / f"{name}.parquet").exists():
            continue
        t = pl.read_parquet(f / f"{name}.parquet")
        blocks[name] = [c for c in t.columns if c != "tx"]
        d = d.join(t, on="tx", how="left")
    return d.sort("tx").to_pandas(), blocks


def best_thr(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    return th[np.nanargmax(f[:-1])] if len(th) else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="features")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sets", default="all", help="daftar set dipisah koma, atau 'all'")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import xgboost as xgb

    df, bl = load(a.feat)
    y = df["label"].values.astype(int)
    n = len(df)
    tr, va, te = np.arange(0, int(0.6 * n)), np.arange(int(0.6 * n), int(0.8 * n)), np.arange(int(0.8 * n), n)
    print(f"transaksi {n:,}; laundering train/val/test: {y[tr].sum()}/{y[va].sum()}/{y[te].sum()}", flush=True)
    sets = {
        "base": bl["base"],
        "base+ctx1": bl["base"] + bl["ctx1"],
        "base+ctx1+ctx2": bl["base"] + bl["ctx1"] + bl["ctx2"],
        "base+ctx1+ctx2+ctx3": bl["base"] + bl["ctx1"] + bl["ctx2"] + bl["ctx3"],
        "ctx1+ctx2+ctx3": bl["ctx1"] + bl["ctx2"] + bl["ctx3"],
    }
    if "gfp" in bl:  # baseline publikasi GFP, dan GFP + konteks hulu kita
        sets["base+gfp"] = bl["base"] + bl["gfp"]
        sets["base+gfp+ctx1"] = bl["base"] + bl["gfp"] + bl["ctx1"]
        sets["base+gfp+ctx1+ctx2"] = bl["base"] + bl["gfp"] + bl["ctx1"] + bl["ctx2"]
        sets["base+gfp+ctx1+ctx2+ctx3"] = bl["base"] + bl["gfp"] + bl["ctx1"] + bl["ctx2"] + bl["ctx3"]
    if a.sets != "all":
        keep = a.sets.split(",")
        sets = {k: v for k, v in sets.items() if k in keep}
    rows, t0, shap_done = [], time.time(), None
    for name, cols in sets.items():
        X = df[cols].astype("float32")
        for s in [int(x) for x in a.seeds.split(",")]:
            pw = (len(tr) - y[tr].sum()) / max(y[tr].sum(), 1)
            m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=8, subsample=0.8,
                                  colsample_bytree=0.6, min_child_weight=5, tree_method="hist",
                                  device="cpu" if a.cpu else "cuda", scale_pos_weight=pw,
                                  eval_metric="aucpr", early_stopping_rounds=150, random_state=s)
            t1 = time.time()
            m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
            fit_sec = time.time() - t1
            pv, pt = m.predict_proba(X.iloc[va])[:, 1], m.predict_proba(X.iloc[te])[:, 1]
            thr = best_thr(y[va], pv)
            rows.append({"set": name, "seed": s, "f1_minority": f1_score(y[te], pt >= thr),
                         "pr_auc": average_precision_score(y[te], pt), "roc_auc": roc_auc_score(y[te], pt),
                         "fit_sec": round(fit_sec, 1), "n_feat": len(cols)})
            print(f"[{time.time() - t0:6.0f}s] {name:<22} s{s} F1={rows[-1]['f1_minority']:.4f} "
                  f"PR-AUC={rows[-1]['pr_auc']:.4f}", flush=True)
            if name == ("base+gfp+ctx1+ctx2+ctx3" if "gfp" in bl else "base+ctx1+ctx2+ctx3") and s == 0:
                contrib = m.get_booster().predict(xgb.DMatrix(X.iloc[te]), pred_contribs=True)[:, :-1]
                imp = pd.Series(np.abs(contrib).mean(0), index=cols)
                blk = {c: next(k for k in bl if c in bl[k]) for c in cols}
                shap_done = (imp.groupby(blk).sum() / imp.sum()).round(3).sort_values(ascending=False)
                imp.sort_values(ascending=False).head(25).to_csv(out / "shap_top25.csv")
        pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)

    res = pd.DataFrame(rows)
    g = res.groupby("set")
    fmt = lambda c: (g[c].mean().round(4).astype(str) + " ± " + g[c].std().fillna(0).round(4).astype(str))
    tab = pd.DataFrame({"F1 minoritas": fmt("f1_minority"), "PR-AUC": fmt("pr_auc"), "ROC-AUC": fmt("roc_auc"),
                        "detik/fit": g["fit_sec"].mean().round(1), "n_fitur": g["n_feat"].first()}).reindex(list(sets))
    md = ["# AMLworld HI-Small — split temporal 60/20/20, XGBoost\n", tab.to_markdown(),
          "\n\n## Porsi kontribusi SHAP per blok (set terlengkap, test, seed 0)\n",
          shap_done.to_frame("porsi").to_markdown() if shap_done is not None else "-"]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
