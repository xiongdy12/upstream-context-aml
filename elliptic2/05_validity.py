"""Uji validitas: apakah lonjakan PR-AUC dari konteks graf latar itu sinyal asli atau shortcut/kebocoran?

Tiga analisis:
  A. Berbagi tetangga  : berapa subgraf (suspicious/licit) yang berbagi node tetangga eksternal yang sama;
                         apakah subgraf saling bertetangga langsung; kesamaan label antar-tetangga.
  B. Group split       : subgraf yang berbagi tetangga eksternal (dengan derajat-subgraf <= K) atau saling
                         terhubung digabung jadi satu grup -> StratifiedGroupKFold 5 fold. Jika PR-AUC tetap
                         tinggi, model tidak sekadar menghafal sumber dana yang sama.
  C. Ablasi fitur      : XGBoost-GPU pada beberapa set fitur (lihat FEATURE_SETS).

  python 05_validity.py --feat features --data dataset --out results_validity
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

from common import load_schema

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------- data
def load_all(feat_dir):
    f = Path(feat_dir)
    lab = pl.read_parquet(f / "labels.parquet")
    internal = pl.read_parquet(f / "internal.parquet")
    ctx = pl.read_parquet(f / "context.parquet")
    df = lab.join(internal, on="ccId").join(ctx, on="ccId", how="left").sort("row_order").to_pandas()
    int_cols = [c for c in internal.columns if c != "ccId"]
    ctx_cols = [c for c in ctx.columns if c != "ccId"]
    return df, int_cols, ctx_cols


def feature_sets(int_cols, ctx_cols, top1):
    innb = [c for c in ctx_cols if c.startswith("bg_innb")]
    outnb = [c for c in ctx_cols if c.startswith("bg_outnb")]
    deg = [c for c in ctx_cols if "deg" in c or "_nbrs" in c]
    edge = [c for c in ctx_cols if c.startswith("bg_ine") or c.startswith("bg_oute")]
    return {
        "internal": int_cols,
        "internal+context": int_cols + ctx_cols,
        "internal+context-innb": int_cols + [c for c in ctx_cols if c not in innb],
        "internal+context-allnb": int_cols + [c for c in ctx_cols if c not in innb + outnb],
        "internal+deg": int_cols + deg,
        "internal+edgectx": int_cols + edge,
        "internal+innb": int_cols + innb,
        "context_only": ctx_cols,
        f"top1[{top1}]": [top1],
    }


# ----------------------------------------------------------------------------- A. berbagi tetangga
def neighbor_analysis(data_dir, feat_dir, df, out, Ks):
    s = load_schema(data_dir)
    mem = (pl.read_csv(Path(data_dir) / "nodes.csv", columns=[s["n_id"], s["n_cc"]])
           .rename({s["n_id"]: "clId", s["n_cc"]: "ccId"}).unique())
    touch = pl.read_parquet(Path(feat_dir) / "bg_touch.parquet", columns=["src", "dst"])
    lab = pl.from_pandas(df[["ccId", "label"]])
    pairs = []
    for me, other, dirn in (("src", "dst", "out"), ("dst", "src", "in")):
        p = (touch.join(mem.rename({"clId": me}), on=me)
             .join(mem.rename({"clId": other, "ccId": "cc_other"}), on=other, how="left")
             .filter(pl.col("cc_other").is_null() | (pl.col("cc_other") != pl.col("ccId")))
             .group_by(pl.col("ccId"), pl.col(other).alias("nbr"), "cc_other")
             .agg(pl.len().alias("w")).with_columns(dir=pl.lit(dirn)))
        pairs.append(p)
    pairs = pl.concat(pairs).join(lab, on="ccId")
    lines = ["# Uji validitas Elliptic2\n", "## A. Berbagi tetangga eksternal\n"]

    # A1. subgraf yang saling bertetangga langsung
    adj = pairs.filter(pl.col("cc_other").is_not_null()).select("ccId", "cc_other").unique()
    adj = adj.join(lab.rename({"ccId": "cc_other", "label": "label_other"}), on="cc_other").join(lab, on="ccId")
    n_adj_sub = adj["ccId"].n_unique()
    conc = (adj["label"] == adj["label_other"]).mean() if len(adj) else float("nan")
    ss = adj.filter((pl.col("label") == 1) & (pl.col("label_other") == 1))["ccId"].n_unique()
    lines.append(f"- Subgraf yang bertetangga langsung dengan subgraf lain: {n_adj_sub:,} "
                 f"(kesamaan label antar-pasangan: {conc:.3f}); suspicious yang bertetangga dengan "
                 f"suspicious lain: {ss:,} dari {int(df.label.sum()):,}")

    # A2. tetangga eksternal bersama
    for dirn in ("in", "out"):
        p = pairs.filter(pl.col("dir") == dirn)
        nstat = p.group_by("nbr").agg(pl.col("ccId").n_unique().alias("n_sub"),
                                      pl.col("label").sum().alias("n_susp"))
        top = nstat.sort("n_sub", descending=True).head(15)
        susp_p = p.filter(pl.col("label") == 1).join(nstat, on="nbr")
        # suspicious yang punya tetangga yang juga dipakai suspicious lain
        shared = susp_p.filter(pl.col("n_susp") >= 2)["ccId"].n_unique()
        lines += [f"\n### Tetangga arah `{dirn}` "
                  f"({'pengirim dana ke subgraf' if dirn == 'in' else 'penerima dana dari subgraf'})",
                  f"- node tetangga unik: {len(nstat):,}",
                  f"- suspicious yang berbagi >=1 tetangga `{dirn}` dengan suspicious lain: "
                  f"{shared:,} / {int(df.label.sum()):,} ({shared / df.label.sum():.1%})",
                  "- 15 tetangga dengan subgraf terbanyak (n_sub = jumlah subgraf, n_susp = yang suspicious):\n",
                  top.to_pandas().to_markdown(index=False)]

    # B-prep. grup "sumber dana dominan": tiap subgraf dikelompokkan menurut tetangga pengirim (in)
    # dengan edge terbanyak; subgraf tanpa pengirim eksternal jadi grup sendiri.
    idx = pl.from_pandas(df[["ccId"]].reset_index()).rename({"index": "i"})
    groups = {}
    dom = (pairs.filter(pl.col("dir") == "in").sort(["ccId", "w", "nbr"], descending=[False, True, False])
           .group_by("ccId", maintain_order=True).first().select("ccId", pl.col("nbr").alias("dom")))
    dd = idx.join(dom, on="ccId", how="left").sort("i")
    dom_arr = dd["dom"].to_numpy()
    n = len(df)
    codes = pd.Series(dom_arr).astype("object")
    codes[codes.isna()] = ["solo_" + str(i) for i in np.where(codes.isna())[0]]
    comp = pd.factorize(codes)[0]
    sizes = np.bincount(comp)
    groups["src"] = comp
    lines.append(f"\n- Grup sumber dominan: {len(sizes):,} grup, terbesar {sizes.max():,} subgraf "
                 f"({sizes.max() / n:.1%}), suspicious di grup terbesar: "
                 f"{int(df.label.values[comp == sizes.argmax()].sum()):,}")
    nsub = pairs.group_by("nbr").agg(pl.col("ccId").n_unique().alias("n_sub"))
    for K in Ks:
        keep = pairs.join(nsub.filter(pl.col("n_sub") <= K), on="nbr").select("nbr", "ccId").unique()
        e1 = (keep.sort("nbr", "ccId").with_columns(prev=pl.col("ccId").shift(1).over("nbr"))
              .drop_nulls("prev").select(pl.col("prev").alias("a"), pl.col("ccId").alias("b")))
        e2 = adj.select(pl.col("ccId").alias("a"), pl.col("cc_other").alias("b"))
        e = (pl.concat([e1, e2]).join(idx.rename({"ccId": "a", "i": "ia"}), on="a")
             .join(idx.rename({"ccId": "b", "i": "ib"}), on="b"))
        g = coo_matrix((np.ones(len(e)), (e["ia"].to_numpy(), e["ib"].to_numpy())), shape=(n, n))
        _, comp = connected_components(g, directed=False)
        sizes = np.bincount(comp)
        usable = len(sizes) >= 10 and sizes.max() / n <= 0.3
        if usable:
            groups[f"K{K}"] = comp
        lines.append(f"\n- {'' if usable else '(DILEWATI: grup terlalu besar) '}Grup K={K}: {len(sizes):,} grup, terbesar {sizes.max():,} subgraf "
                     f"({sizes.max() / n:.1%}), suspicious di grup terbesar: "
                     f"{int(df.label.values[comp == sizes.argmax()].sum()):,}")
    return lines, groups


# ----------------------------------------------------------------------------- C. model
def xgb_fit_predict(Xtr, ytr, Xva, yva, Xte, seed, gpu):
    import xgboost as xgb
    pw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=8, subsample=0.8,
                          colsample_bytree=0.6, tree_method="hist", device="cuda" if gpu else "cpu",
                          scale_pos_weight=pw, eval_metric="aucpr", early_stopping_rounds=100,
                          random_state=seed)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m.predict_proba(Xte)[:, 1]


def run_eval(df, fsets, groups, seeds, gpu, group_sets):
    y = df["label"].values.astype(int)
    rows, t0 = [], time.time()
    for name, cols in fsets.items():
        X = df[cols].astype("float32")
        for s in seeds:  # stratified acak
            idx = np.arange(len(y))
            tr, tmp = train_test_split(idx, test_size=0.2, stratify=y, random_state=s)
            va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=s)
            p = xgb_fit_predict(X.iloc[tr], y[tr], X.iloc[va], y[va], X.iloc[te], s, gpu)
            rows.append({"set": name, "split": "strat", "fold": s,
                         "pr_auc": average_precision_score(y[te], p), "roc_auc": roc_auc_score(y[te], p)})
        if name not in group_sets:
            print(f"[{time.time() - t0:6.0f}s] {name} selesai", flush=True)
            continue
        for K, comp in groups.items():
            sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
            for k, (trv, te) in enumerate(sgk.split(X, y, comp)):
                gss = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=k)
                tr_i, va_i = next(gss.split(trv, y[trv], comp[trv]))
                tr, va = trv[tr_i], trv[va_i]
                if y[te].sum() == 0 or y[va].sum() == 0:
                    continue
                p = xgb_fit_predict(X.iloc[tr], y[tr], X.iloc[va], y[va], X.iloc[te], k, gpu)
                rows.append({"set": name, "split": f"group_{K}", "fold": k,
                             "pr_auc": average_precision_score(y[te], p),
                             "roc_auc": roc_auc_score(y[te], p), "pos_test": int(y[te].sum())})
        print(f"[{time.time() - t0:6.0f}s] {name} selesai", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="features")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="results_validity")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--K", default="10,100", help="batas derajat-subgraf tetangga untuk membentuk grup")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df, int_cols, ctx_cols = load_all(a.feat)
    imp = Path("results_baseline/importance_internal+context.csv")
    top1 = pd.read_csv(imp, index_col=0).index[0] if imp.exists() else "bg_innb_mean_feat#43"
    if top1 not in df.columns:
        top1 = ctx_cols[0]
    fsets = feature_sets(int_cols, ctx_cols, top1)

    lines, groups = neighbor_analysis(a.data, a.feat, df, out, [int(k) for k in a.K.split(",")])
    (out / "validity_summary.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)

    group_sets = {"internal", "internal+context", "internal+context-allnb", f"top1[{top1}]"}
    res = run_eval(df, fsets, groups, [int(s) for s in a.seeds.split(",")], not a.cpu, group_sets)
    res.to_csv(out / "validity_scores.csv", index=False)
    g = res.groupby(["set", "split"])["pr_auc"]
    tab = (g.mean().round(3).astype(str) + " ± " + g.std().fillna(0).round(3).astype(str)).unstack("split")
    tab = tab.reindex(list(fsets)).fillna("-")
    lines += ["\n## B & C. PR-AUC XGBoost (mean ± std) per set fitur dan jenis split\n",
              "strat = acak stratified; group_src = grup sumber dana dominan; group_K* = grup tetangga bersama (derajat <= K)\n",
              tab.to_markdown()]
    (out / "validity_summary.md").write_text("\n".join(lines))
    print("\n".join(lines[-3:]))


if __name__ == "__main__":
    main()
