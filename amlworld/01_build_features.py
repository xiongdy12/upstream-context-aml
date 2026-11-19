"""Fitur konteks hulu TEMPORAL (kausal) untuk IBM AMLworld HI-Small (klasifikasi transaksi).

Transaksi t: akun u -> akun v pada waktu tau.  Semua konteks hanya memakai transaksi dengan waktu < tau
(strict; transaksi pada menit yang sama tidak dipakai) dan berada dalam jendela W (default 24 jam).

  base : fitur transaksi itu sendiri (jumlah, mata uang, format pembayaran, bank sama, jam)       [~ "internal"]
  C1   : konteks 1-hop as-of tau: transaksi MASUK ke u (hulu), KELUAR dari u, MASUK ke v, KELUAR dari v
         (jumlah transaksi + rata-rata fitur base)                                                 [~ "hop1"]
  C2   : rata-rata inti-C1 dari transaksi yang masuk ke u (dihitung as-of waktu masing-masing)       [~ "hop2"]
  C3   : rata-rata inti-C2 dari transaksi yang masuk ke u                                          [~ "hop3"]
Rekursi: C_{k+1}(t) = mean_{e masuk ke u, tau-W <= time(e) < tau} inti(C_k(e)).
Implementasi: jumlah kumulatif per akun + join_asof (O(N log N)), tanpa loop per transaksi.

  python 01_build_features.py --data data/HI-Small_Trans.csv --out features --window_hours 24
"""
import argparse
import warnings
import time
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore", message=".*Sortedness.*")

COLS = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amt_recv", "cur_recv",
        "amt_paid", "cur_paid", "fmt", "label"]


def load(path):
    df = pl.read_csv(path, has_header=False, skip_rows=1, new_columns=COLS,
                     schema_overrides={"from_bank": pl.Utf8, "to_bank": pl.Utf8, "from_acct": pl.Utf8,
                                       "to_acct": pl.Utf8, "fmt": pl.Utf8, "cur_recv": pl.Utf8,
                                       "cur_paid": pl.Utf8})
    df = df.with_columns(
        t=pl.col("ts").str.strptime(pl.Datetime, "%Y/%m/%d %H:%M").dt.epoch("s"),
        u=pl.concat_str(["from_bank", "from_acct"], separator="_"),
        v=pl.concat_str(["to_bank", "to_acct"], separator="_"),
    ).sort("t", maintain_order=True).with_row_index("tx")
    return df


def base_features(df):
    fmts = sorted(df["fmt"].unique().to_list())
    curs = sorted(set(df["cur_paid"].unique().to_list()) | set(df["cur_recv"].unique().to_list()))
    hour = (pl.col("t") % 86400) / 3600.0
    exprs = [
        pl.col("amt_paid").log1p().alias("b_logamt_paid"),
        pl.col("amt_recv").log1p().alias("b_logamt_recv"),
        (pl.col("cur_paid") == pl.col("cur_recv")).cast(pl.Float32).alias("b_same_cur"),
        (pl.col("from_bank") == pl.col("to_bank")).cast(pl.Float32).alias("b_same_bank"),
        (pl.col("u") == pl.col("v")).cast(pl.Float32).alias("b_self"),
        (2 * np.pi * hour / 24).sin().alias("b_hour_sin"), (2 * np.pi * hour / 24).cos().alias("b_hour_cos"),
        ((pl.col("t") // 86400 + 4) % 7).alias("b_dow"),
    ]
    exprs += [(pl.col("fmt") == f).cast(pl.Float32).alias(f"b_fmt_{f.replace(' ', '_')}") for f in fmts]
    exprs += [(pl.col("cur_paid") == c).cast(pl.Float32).alias(f"b_curp_{c.replace(' ', '_')}") for c in curs]
    B = df.select(["tx"] + exprs)
    return B, [c for c in B.columns if c != "tx"]


def win_agg(events, ev_key, val_cols, queries, q_key, W, prefix):
    """Agregasi jendela kausal. events: (tx_ev, key, t, vals); queries: (tx, key, t).
    Kembalikan tx + {prefix}n (jumlah event dalam [t-W, t)) + {prefix}mean_* ."""
    ev = (events.select(pl.col(ev_key).alias("k"), "t", *[pl.col(c).cast(pl.Float64).fill_null(0) for c in val_cols])
          .sort("t", maintain_order=True)
          .with_columns([pl.col(c).cum_sum().over("k") for c in val_cols]
                        + [pl.int_range(1, pl.len() + 1, dtype=pl.Int64).over("k").alias("_n")]))
    q = queries.select("tx", pl.col(q_key).alias("k"), (pl.col("t") - 1).alias("t")).sort("t")
    A = q.join_asof(ev, on="t", by="k", strategy="backward")
    cols = val_cols + ["_n"]
    if W:
        qb = q.with_columns(pl.col("t") - W).sort("t")
        Bq = qb.join_asof(ev, on="t", by="k", strategy="backward").select(
            "tx", *[pl.col(c).alias(f"{c}__b") for c in cols])
        A = A.join(Bq, on="tx", how="left").with_columns(
            [(pl.col(c).fill_null(0) - pl.col(f"{c}__b").fill_null(0)).alias(c) for c in cols])
    n = pl.col("_n").fill_null(0)
    out = A.select("tx", n.cast(pl.Float32).alias(f"{prefix}n"),
                   *[pl.when(n > 0).then(pl.col(c) / n).otherwise(None).cast(pl.Float32)
                     .alias(f"{prefix}mean_{c}") for c in val_cols])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/HI-Small_Trans.csv")
    ap.add_argument("--out", default="features")
    ap.add_argument("--window_hours", type=float, default=24)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    W = int(a.window_hours * 3600) if a.window_hours > 0 else 0
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    df = load(a.data)
    log(f"transaksi: {len(df):,}; akun unik: {pl.concat([df['u'], df['v']]).n_unique():,}; "
        f"laundering: {int(df['label'].sum()):,} ({df['label'].mean():.3%})")
    B, bcols = base_features(df)
    tx = df.select("tx", "t", "u", "v", "label")
    E = tx.join(B, on="tx")  # setiap transaksi sebagai "event"

    # ---------------- C1: empat arah relatif terhadap u dan v
    agg_cols = ["b_logamt_paid", "b_same_cur", "b_same_bank", "b_self"] + \
        [c for c in bcols if c.startswith("b_fmt_")]
    parts = []
    for prefix, ev_key, q_key in (("c1_uin_", "v", "u"),    # masuk ke u (hulu)
                                  ("c1_uout_", "u", "u"),   # keluar dari u
                                  ("c1_vin_", "v", "v"),    # masuk ke v
                                  ("c1_vout_", "u", "v")):  # keluar dari v
        parts.append(win_agg(E, ev_key, agg_cols, tx, q_key, W, prefix))
        log(f"C1 {prefix}: selesai")
    C1 = parts[0]
    for p in parts[1:]:
        C1 = C1.join(p, on="tx")
    core1 = ["c1_uin_n", "c1_uin_mean_b_logamt_paid", "c1_uout_n", "c1_uout_mean_b_logamt_paid",
             "c1_uin_mean_b_same_bank"] + [c for c in C1.columns if c.startswith("c1_uin_mean_b_fmt_")]

    # ---------------- C2, C3: rekursi hulu (transaksi yang masuk ke u)
    E2 = tx.join(C1.select(["tx"] + core1), on="tx")
    C2 = win_agg(E2, "v", core1, tx, "u", W, "c2_")
    log(f"C2: {C2.shape}")
    core2 = ["c2_n"] + [f"c2_mean_{c}" for c in core1]
    E3 = tx.join(C2.select(["tx"] + core2), on="tx")
    C3 = win_agg(E3, "v", core2, tx, "u", W, "c3_")
    log(f"C3: {C3.shape}")

    meta = df.select("tx", "t", "label", "u", "v")
    meta.write_parquet(out / "meta.parquet")
    B.write_parquet(out / "base.parquet")
    C1.write_parquet(out / "ctx1.parquet")
    C2.write_parquet(out / "ctx2.parquet")
    C3.write_parquet(out / "ctx3.parquet")
    log(f"tersimpan di {out}: base {B.shape}, ctx1 {C1.shape}, ctx2 {C2.shape}, ctx3 {C3.shape}")


if __name__ == "__main__":
    main()
