"""Fitur konteks 3-hop hulu: pengirim dari pengirim dari pengirim.

Rantai: N3 -> N2 -> N1 -> subgraf S
  N1 = pengirim langsung ke S (bobot w1 = jumlah edge N1->S)
  N2 = pengirim ke N1 (dari features/bg_hop2_in.parquet; bobot w2 = jumlah edge N2->N1)
  Untuk tiap n2: derajat masuk, pengirim unik, rata-rata fitur edge masuk ke n2,
  rata-rata fitur node pengirim n2 (= node 3-hop).
  Agregasi: n2 -> n1 (rata-rata berbobot w2) -> S (rata-rata berbobot w1).

background_edges.csv dibaca sekali lagi (edge menuju N2) -> features/bg_hop3_in.parquet.

  python 10_context3hop.py --data dataset --out features
"""
import argparse
import time
from pathlib import Path

import polars as pl

from common import collect, load_schema


def wmean(cols, w="w"):
    return [((pl.col(c) * pl.col(w)).sum() / pl.col(w).filter(pl.col(c).is_not_null()).sum()).alias(c)
            for c in cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="features")
    ap.add_argument("--schema", default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no_nb3", action="store_true", help="lewati fitur node 3-hop (lebih ringan)")
    a = ap.parse_args()
    d, out = Path(a.data), Path(a.out)
    s = load_schema(d, a.schema)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)
    be_feats, bn_feats = s["be_feats"], s["bn_feats"]

    mem = (pl.read_csv(d / "nodes.csv", columns=[s["n_id"], s["n_cc"]])
           .rename({s["n_id"]: "clId", s["n_cc"]: "ccId"}).unique())
    touch = pl.read_parquet(out / "bg_touch.parquet", columns=["src", "dst"])
    p1 = (touch.join(mem.rename({"clId": "dst"}), on="dst")
          .join(mem.rename({"clId": "src", "ccId": "cc_src"}), on="src", how="left")
          .filter(pl.col("cc_src").is_null() | (pl.col("cc_src") != pl.col("ccId")))
          .group_by("ccId", pl.col("src").alias("n1")).agg(pl.len().alias("w")))
    p2 = (pl.scan_parquet(out / "bg_hop2_in.parquet").select("src", "dst")
          .group_by(pl.col("dst").alias("n1"), pl.col("src").alias("n2")).agg(pl.len().alias("w")))
    p2 = collect(p2)
    n2 = p2["n2"].unique()
    log(f"pasangan S-N1: {len(p1):,}; pasangan N1-N2: {len(p2):,}; N2 unik: {len(n2):,}")

    hop3_path = out / "bg_hop3_in.parquet"
    if a.rebuild or not hop3_path.exists():
        log("menyaring edge yang menuju N2 dari background_edges.csv (satu kali baca)...")
        lf = (pl.scan_csv(d / "background_edges.csv")
              .select([s["be_src"], s["be_dst"]] + be_feats)
              .rename({s["be_src"]: "src", s["be_dst"]: "dst"})
              .filter(pl.col("dst").is_in(n2.implode()))
              .with_columns([pl.col(c).cast(pl.Float32) for c in be_feats]))
        try:
            lf.sink_parquet(hop3_path)
        except Exception as e:  # noqa: BLE001
            log(f"sink_parquet gagal ({e}); pakai collect")
            collect(lf).write_parquet(hop3_path)
    hop3 = pl.scan_parquet(hop3_path)
    log(f"edge 3-hop (menuju N2): {collect(hop3.select(pl.len())).item():,}")

    st = collect(hop3.group_by("dst").agg(
        [pl.len().alias("h3_indeg"), pl.col("src").n_unique().alias("h3_nsrc")]
        + [pl.col(c).mean().alias(f"h3e_mean_{c}") for c in be_feats])).rename({"dst": "n2"})
    if not a.no_nb3:
        bn = (pl.scan_csv(d / "background_nodes.csv").select([s["bn_id"]] + bn_feats)
              .rename({s["bn_id"]: "clId"}).with_columns([pl.col(c).cast(pl.Float32) for c in bn_feats]))
        nb3 = collect(hop3.select("src", "dst").join(bn, left_on="src", right_on="clId", how="inner")
                      .group_by("dst").agg([pl.col(c).mean().alias(f"h3nb_mean_{c}") for c in bn_feats])
                      ).rename({"dst": "n2"})
        st = st.join(nb3, on="n2", how="left")
    log(f"statistik per N2: {st.shape}")
    cols = [c for c in st.columns if c != "n2"]

    per_n1 = p2.join(st, on="n2", how="left").group_by("n1").agg(
        wmean(cols) + [pl.col("h3_indeg").max().alias("h3_indeg_maxN2")])
    log(f"agregasi ke N1: {per_n1.shape}")
    cols1 = [c for c in per_n1.columns if c != "n1"]
    ctx3 = (p1.join(per_n1, on="n1", how="left").group_by("ccId")
            .agg([c.alias(f"{c.meta.output_name()}_wmean") for c in wmean(cols1)]))
    lab = pl.read_parquet(out / "labels.parquet").select("ccId")
    ctx3 = lab.join(ctx3, on="ccId", how="left")
    ctx3 = ctx3.with_columns([pl.col(c).fill_nan(None) for c in ctx3.columns if c != "ccId"])
    ctx3.write_parquet(out / "context3.parquet")
    log(f"context3: {ctx3.shape} -> {out / 'context3.parquet'}")


if __name__ == "__main__":
    main()
