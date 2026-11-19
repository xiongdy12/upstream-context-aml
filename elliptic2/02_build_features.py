"""Bangun fitur tabular per subgraf (input baseline gradient boosting / RF).

Alur (background_edges.csv 83 GB hanya dibaca SEKALI):
  1. Saring edge latar yang menyentuh node anggota subgraf -> features/bg_touch.parquet
  2. internal : agregasi fitur node anggota + fitur edge internal (edges.csv dicocokkan ke
                bg_touch via clId1, clId2, txId) + statistik struktur
  3. context  : konteks 1-hop dari graf latar: derajat masuk/keluar anggota, rata-rata
                fitur edge yang menyentuh anggota, rata-rata fitur node tetangga

Contoh:
  python 02_build_features.py --data dataset --out features
  python 02_build_features.py --data dataset --out features --no_neighbor_feats   # lebih ringan
"""
import argparse
import time
from pathlib import Path

import polars as pl

from common import collect, load_schema

AGG = ("mean", "max", "min", "std")


def agg_exprs(cols, prefix, aggs=AGG):
    return [getattr(pl.col(c), a)().alias(f"{prefix}{a}_{c}") for c in cols for a in aggs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="features")
    ap.add_argument("--schema", default=None, help="JSON override nama kolom (opsional)")
    ap.add_argument("--no_neighbor_feats", action="store_true")
    ap.add_argument("--rebuild_touch", action="store_true", help="paksa saring ulang edge latar")
    a = ap.parse_args()
    d, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = load_schema(d, a.schema)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)

    # ---------------------------------------------------------------- label & anggota
    cc = (pl.read_csv(d / "connected_components.csv", columns=[s["cc_id"], s["cc_label"]])
          .rename({s["cc_id"]: "ccId", s["cc_label"]: "ccLabel"}).with_row_index("row_order"))
    lab = cc["ccLabel"].cast(pl.Utf8).str.to_lowercase()
    cc = cc.with_columns(label=(lab.str.contains("suspic") | (lab == "1")).cast(pl.Int8))
    cc.select("ccId", "label", "row_order").write_parquet(out / "labels.parquet")
    log(f"subgraf: {len(cc):,}  suspicious: {int(cc['label'].sum()):,}")

    mem = (pl.read_csv(d / "nodes.csv", columns=[s["n_id"], s["n_cc"]])
           .rename({s["n_id"]: "clId", s["n_cc"]: "ccId"}).unique())
    member_ids = mem["clId"].unique()
    member_set = member_ids.implode()  # bentuk yang dianjurkan polars untuk is_in
    log(f"node anggota unik: {len(member_ids):,}")

    # ---------------------------------------------------------------- fitur node (49 jt baris)
    bn_feats = s["bn_feats"]
    bn = collect(pl.scan_csv(d / "background_nodes.csv")
                 .select([s["bn_id"]] + bn_feats).rename({s["bn_id"]: "clId"})
                 .with_columns([pl.col(c).cast(pl.Float32) for c in bn_feats]))
    log(f"background_nodes dimuat: {bn.shape}")

    # ---------------------------------------------------------------- saring edge latar sekali
    be_feats = s["be_feats"]
    tx = s["tx_col"]
    touch_path = out / "bg_touch.parquet"
    if a.rebuild_touch or not touch_path.exists():
        log("menyaring background_edges.csv (satu kali baca, bisa 20-60 menit)...")
        lf = (pl.scan_csv(d / "background_edges.csv")
              .select([s["be_src"], s["be_dst"]] + ([tx] if tx else []) + be_feats)
              .rename({s["be_src"]: "src", s["be_dst"]: "dst"})
              .filter(pl.col("src").is_in(member_set) | pl.col("dst").is_in(member_set))
              .with_columns([pl.col(c).cast(pl.Float32) for c in be_feats]))
        try:
            lf.sink_parquet(touch_path)
        except Exception as e:  # noqa: BLE001  fallback bila sink tidak didukung
            log(f"sink_parquet gagal ({e}); pakai collect")
            collect(lf).write_parquet(touch_path)
    touch = pl.read_parquet(touch_path)
    if tx and tx != "txId" and tx in touch.columns:
        touch = touch.rename({tx: "txId"})
    log(f"edge latar yang menyentuh anggota: {touch.shape}")

    # ---------------------------------------------------------------- fitur internal
    nf = bn.filter(pl.col("clId").is_in(member_set))
    node_agg = (mem.join(nf, on="clId", how="left").group_by("ccId")
                .agg([pl.len().alias("n_nodes")] + agg_exprs(bn_feats, "n_")))

    e = pl.read_csv(d / "edges.csv").rename({s["e_src"]: "src", s["e_dst"]: "dst"})
    if s["e_tx"]:
        e = e.rename({s["e_tx"]: "txId"})
    e_feats = s["e_feats"]
    if not e_feats:  # ambil fitur edge internal dari graf latar
        keys = ["src", "dst"] + (["txId"] if "txId" in e.columns and "txId" in touch.columns else [])
        e = e.select(keys + ([s["e_cc"]] if s["e_cc"] else [])).join(
            touch.select(keys + be_feats).unique(subset=keys), on=keys, how="left")
        e_feats = be_feats
        miss = e[e_feats[0]].null_count()
        log(f"edge internal: {len(e):,}, tanpa padanan di graf latar: {miss:,}")
    if s["e_cc"]:
        e = e.rename({s["e_cc"]: "ccId"})
    else:  # tetapkan subgraf lewat keanggotaan kedua ujung edge
        e = (e.join(mem.rename({"clId": "src", "ccId": "cc1"}), on="src")
              .join(mem.rename({"clId": "dst", "ccId": "ccId"}), on="dst")
              .filter(pl.col("cc1") == pl.col("ccId")).drop("cc1"))
    edge_agg = e.group_by("ccId").agg([pl.len().alias("n_edges")]
                                      + agg_exprs(e_feats, "e_", ("mean", "max", "min")))
    deg_out = e.group_by("ccId", "src").agg(pl.len().alias("d")).group_by("ccId").agg(
        pl.col("d").max().alias("int_outdeg_max"), pl.col("d").mean().alias("int_outdeg_mean"))
    deg_in = e.group_by("ccId", "dst").agg(pl.len().alias("d")).group_by("ccId").agg(
        pl.col("d").max().alias("int_indeg_max"), pl.col("d").mean().alias("int_indeg_mean"))
    roles = e.group_by("ccId").agg(
        pl.col("src").n_unique().alias("n_src_nodes"), pl.col("dst").n_unique().alias("n_dst_nodes"))

    internal = (cc.select("ccId").join(node_agg, on="ccId", how="left")
                .join(edge_agg, on="ccId", how="left").join(deg_out, on="ccId", how="left")
                .join(deg_in, on="ccId", how="left").join(roles, on="ccId", how="left")
                .with_columns(pl.col("n_edges").fill_null(0))
                .with_columns(density=pl.col("n_edges")
                              / (pl.col("n_nodes") * (pl.col("n_nodes") - 1)).clip(lower_bound=1)))
    internal.write_parquet(out / "internal.parquet")
    log(f"internal: {internal.shape} -> {out / 'internal.parquet'}")

    # ---------------------------------------------------------------- konteks graf latar
    per_node = []
    for direction, me, other in (("out", "src", "dst"), ("in", "dst", "src")):
        sel = touch.filter(pl.col(me).is_in(member_set))
        stats = sel.group_by(me).agg(
            [pl.len().alias(f"bg_{direction}deg"), pl.col(other).n_unique().alias(f"bg_{direction}_nbrs")]
            + [pl.col(c).mean().alias(f"bg_{direction}e_mean_{c}") for c in be_feats]
        ).rename({me: "clId"})
        if not a.no_neighbor_feats:
            nb = (sel.select(me, other).join(bn, left_on=other, right_on="clId", how="inner")
                  .group_by(me).agg([pl.col(c).mean().alias(f"bg_{direction}nb_mean_{c}")
                                     for c in bn_feats]).rename({me: "clId"}))
            stats = stats.join(nb, on="clId", how="left")
        per_node.append(stats)
        log(f"konteks {direction}: {stats.shape}")

    node_ctx = (mem.select("clId").unique().join(per_node[0], on="clId", how="left")
                .join(per_node[1], on="clId", how="left"))
    deg_cols = [c for c in node_ctx.columns if c.endswith("deg") or c.endswith("_nbrs")]
    mean_cols = [c for c in node_ctx.columns if c not in deg_cols and c != "clId"]
    ctx = (mem.join(node_ctx, on="clId", how="left").group_by("ccId").agg(
        [pl.col(c).fill_null(0).sum().alias(f"{c}_sum") for c in deg_cols]
        + [pl.col(c).fill_null(0).max().alias(f"{c}_max") for c in deg_cols]
        + [pl.col(c).mean().alias(c) for c in mean_cols]))
    ctx = cc.select("ccId").join(ctx, on="ccId", how="left")
    ctx.write_parquet(out / "context.parquet")
    log(f"context: {ctx.shape} -> {out / 'context.parquet'}")


if __name__ == "__main__":
    main()
