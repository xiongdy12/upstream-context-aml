"""Fitur konteks 2-hop hulu (upstream): "pengirim dari pengirim".

Untuk tiap subgraf S:
  N1 = node eksternal yang mengirim dana ke anggota S (bobot w = jumlah edge N1 -> S)
  Untuk tiap n1 di N1: derajat masuk n1, jumlah pengirim unik n1, rata-rata fitur edge yang masuk ke n1,
  dan rata-rata fitur node pengirim n1 (2-hop).
  Diagregasi ke S: rata-rata berbobot w, plus max/sum untuk derajat.

background_edges.csv dibaca SEKALI lagi (edge yang menuju N1) -> features/bg_hop2_in.parquet.

  python 06_context2hop.py --data dataset --out features
"""
import argparse
import time
from pathlib import Path

import polars as pl

from common import collect, load_schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="features")
    ap.add_argument("--schema", default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no_nb2", action="store_true", help="lewati fitur node pengirim 2-hop (lebih ringan)")
    a = ap.parse_args()
    d, out = Path(a.data), Path(a.out)
    s = load_schema(d, a.schema)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)

    mem = (pl.read_csv(d / "nodes.csv", columns=[s["n_id"], s["n_cc"]])
           .rename({s["n_id"]: "clId", s["n_cc"]: "ccId"}).unique())
    touch = pl.read_parquet(out / "bg_touch.parquet", columns=["src", "dst"])
    # pasangan (subgraf, pengirim eksternal, bobot)
    in_pairs = (touch.join(mem.rename({"clId": "dst"}), on="dst")
                .join(mem.rename({"clId": "src", "ccId": "cc_src"}), on="src", how="left")
                .filter(pl.col("cc_src").is_null() | (pl.col("cc_src") != pl.col("ccId")))
                .group_by("ccId", pl.col("src").alias("n1")).agg(pl.len().alias("w")))
    n1 = in_pairs["n1"].unique()
    n1_set = n1.implode()
    log(f"pasangan subgraf-pengirim: {len(in_pairs):,}; pengirim unik (N1): {len(n1):,}")

    be_feats, bn_feats = s["be_feats"], s["bn_feats"]
    hop2_path = out / "bg_hop2_in.parquet"
    if a.rebuild or not hop2_path.exists():
        log("menyaring edge yang menuju N1 dari background_edges.csv (satu kali baca)...")
        lf = (pl.scan_csv(d / "background_edges.csv")
              .select([s["be_src"], s["be_dst"]] + be_feats)
              .rename({s["be_src"]: "src", s["be_dst"]: "dst"})
              .filter(pl.col("dst").is_in(n1_set))
              .with_columns([pl.col(c).cast(pl.Float32) for c in be_feats]))
        try:
            lf.sink_parquet(hop2_path)
        except Exception as e:  # noqa: BLE001
            log(f"sink_parquet gagal ({e}); pakai collect")
            collect(lf).write_parquet(hop2_path)
    hop2 = pl.scan_parquet(hop2_path)
    n_rows = collect(hop2.select(pl.len())).item()
    log(f"edge 2-hop (menuju N1): {n_rows:,}")

    # statistik per n1
    st = collect(hop2.group_by("dst").agg(
        [pl.len().alias("h2_indeg"), pl.col("src").n_unique().alias("h2_nsrc")]
        + [pl.col(c).mean().alias(f"h2e_mean_{c}") for c in be_feats])).rename({"dst": "n1"})
    log(f"statistik edge per N1: {st.shape}")
    if not a.no_nb2:
        bn = (pl.scan_csv(d / "background_nodes.csv").select([s["bn_id"]] + bn_feats)
              .rename({s["bn_id"]: "clId"}).with_columns([pl.col(c).cast(pl.Float32) for c in bn_feats]))
        nb2 = collect(hop2.select("src", "dst").join(bn, left_on="src", right_on="clId", how="inner")
                      .group_by("dst").agg([pl.col(c).mean().alias(f"h2nb_mean_{c}") for c in bn_feats])
                      ).rename({"dst": "n1"})
        st = st.join(nb2, on="n1", how="left")
        log(f"fitur node 2-hop per N1: {nb2.shape}")

    # agregasi ke subgraf (rata-rata berbobot w)
    feat_cols = [c for c in st.columns if c != "n1"]
    j = in_pairs.join(st, on="n1", how="left")
    wsum = pl.col("w").sum()
    agg = [((pl.col(c) * pl.col("w")).sum() / wsum).alias(f"{c}_wmean") for c in feat_cols]
    agg += [pl.col("h2_indeg").max().alias("h2_indeg_max"), pl.col("h2_indeg").sum().alias("h2_indeg_sum"),
            pl.col("h2_nsrc").max().alias("h2_nsrc_max"),
            (pl.col("h2_indeg").fill_null(0) == 0).mean().alias("h2_frac_n1_tanpa_pengirim"),
            pl.len().alias("h2_n1_count")]
    ctx2 = j.group_by("ccId").agg(agg)
    lab = pl.read_parquet(out / "labels.parquet").select("ccId")
    ctx2 = lab.join(ctx2, on="ccId", how="left")
    ctx2.write_parquet(out / "context2.parquet")
    log(f"context2: {ctx2.shape} -> {out / 'context2.parquet'}")


if __name__ == "__main__":
    main()
