"""Bangun & cache grup untuk group split (dipakai 07_eval.py dan 08_gnn.py).

  group_src : grup "sumber dana dominan" (pengirim eksternal dengan edge terbanyak ke subgraf)
              -> menguji generalisasi ke sumber dana yang belum pernah dilihat (protokol utama)
  group_K10 : subgraf yang berbagi tetangga eksternal dengan derajat-subgraf <= 10 digabung
"""
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from common import load_schema


def build_groups(data_dir, feat_dir, Ks=(10,)):
    feat_dir = Path(feat_dir)
    cache = feat_dir / "groups.parquet"
    if cache.exists():
        g = pl.read_parquet(cache)
        if all(f"group_K{k}" in g.columns for k in Ks):
            return g
    s = load_schema(data_dir)
    mem = (pl.read_csv(Path(data_dir) / "nodes.csv", columns=[s["n_id"], s["n_cc"]])
           .rename({s["n_id"]: "clId", s["n_cc"]: "ccId"}).unique())
    touch = pl.read_parquet(feat_dir / "bg_touch.parquet", columns=["src", "dst"])
    lab = pl.read_parquet(feat_dir / "labels.parquet").sort("row_order")
    pairs = []
    for me, other, dirn in (("src", "dst", "out"), ("dst", "src", "in")):
        p = (touch.join(mem.rename({"clId": me}), on=me)
             .join(mem.rename({"clId": other, "ccId": "cc_other"}), on=other, how="left")
             .filter(pl.col("cc_other").is_null() | (pl.col("cc_other") != pl.col("ccId")))
             .group_by("ccId", pl.col(other).alias("nbr")).agg(pl.len().alias("w"))
             .with_columns(dir=pl.lit(dirn)))
        pairs.append(p)
    pairs = pl.concat(pairs)
    n = len(lab)
    idx = lab.select("ccId").with_row_index("i")

    dom = (pairs.filter(pl.col("dir") == "in").sort(["ccId", "w", "nbr"], descending=[False, True, False])
           .group_by("ccId", maintain_order=True).first().select("ccId", pl.col("nbr").alias("dom")))
    codes = idx.join(dom, on="ccId", how="left").sort("i")["dom"].to_pandas().astype("object")
    na = codes.isna()
    codes[na] = ["solo_" + str(i) for i in np.where(na)[0]]
    out = {"ccId": lab["ccId"].to_numpy(), "group_src": pd.factorize(codes)[0]}

    nsub = pairs.group_by("nbr").agg(pl.col("ccId").n_unique().alias("n_sub"))
    for K in Ks:
        keep = pairs.join(nsub.filter(pl.col("n_sub") <= K), on="nbr").select("nbr", "ccId").unique()
        e = (keep.sort("nbr", "ccId").with_columns(prev=pl.col("ccId").shift(1).over("nbr"))
             .drop_nulls("prev").select(pl.col("prev").alias("a"), pl.col("ccId").alias("b"))
             .join(idx.rename({"ccId": "a", "i": "ia"}), on="a")
             .join(idx.rename({"ccId": "b", "i": "ib"}), on="b"))
        g = coo_matrix((np.ones(len(e)), (e["ia"].to_numpy(), e["ib"].to_numpy())), shape=(n, n))
        out[f"group_K{K}"] = connected_components(g, directed=False)[1]
    g = pl.DataFrame(out)
    g.write_parquet(cache)
    return g


def make_splits(y, groups: pl.DataFrame | None, seeds=(0, 1, 2), split_names=("strat", "group_src", "group_K10"),
                n_folds=5, val_frac=0.12, group_seeds=(0,)):
    """Split deterministik yang sama untuk semua model (tabular & GNN).
    yield (nama_split, fold, idx_train, idx_val, idx_test)."""
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
    y = np.asarray(y)
    idx = np.arange(len(y))
    for name in split_names:
        if name == "strat":
            for s in seeds:
                tr, tmp = train_test_split(idx, test_size=0.2, stratify=y, random_state=s)
                va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=s)
                yield name, s, tr, va, te
            continue
        if groups is None or name not in groups.columns:
            continue
        g = groups[name].to_numpy()
        sizes = np.bincount(pd.factorize(g)[0])
        if len(sizes) < 2 * n_folds or sizes.max() > 0.3 * len(g):
            print(f"[peringatan] split {name} dilewati: grup terlalu sedikit/terlalu besar")
            continue
        for rep in group_seeds:  # rep 0 = fold yang sama dengan semua eksperimen sebelumnya
            sgk = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=rep)
            for k, (trv, te) in enumerate(sgk.split(idx, y, g)):
                gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=k + 1000 * rep)
                a, b = next(gss.split(trv, y[trv], g[trv]))
                if y[te].sum() == 0 or y[trv[b]].sum() == 0:
                    continue
                yield name, k + 100 * rep, trv[a], trv[b], te
