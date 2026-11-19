"""Baseline publikasi: fitur pola graf IBM Graph Feature Preprocessor (GFP, pustaka snapml).

GFP menghitung fitur pola pencucian uang per transaksi: fan-in/out, degree, scatter-gather,
temporal cycle, length-constrained cycle, dan statistik vertex. Transaksi diproses BERURUTAN
dalam batch kecil (default 128, sama dengan protokol paper GFP), sehingga fitur hanya memakai
transaksi sebelumnya + transaksi dalam batch yang sama.

Butuh: pip install snapml   (wheel tersedia untuk Python 3.9–3.12, Linux)

  python 03_gfp.py --data data/HI-Small_Trans.csv --feat features --batch 128
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import polars as pl

spec = importlib.util.spec_from_file_location("bf", Path(__file__).with_name("01_build_features.py"))
bf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bf)

DAY = 86400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/HI-Small_Trans.csv")
    ap.add_argument("--feat", default="features")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--no_lc_cycle", action="store_true", help="matikan simple cycle (paling mahal)")
    a = ap.parse_args()
    import snapml

    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
    df = bf.load(a.data)  # urutan & nomor tx identik dengan 01_build_features.py
    ids = pl.concat([df["u"], df["v"]]).unique().to_frame("acct").with_row_index("aid")
    df = (df.join(ids.rename({"acct": "u", "aid": "uid"}), on="u", how="left")
            .join(ids.rename({"acct": "v", "aid": "vid"}), on="v", how="left").sort("tx"))
    enc = lambda c: df[c].cast(pl.Categorical).to_physical().cast(pl.Float64).to_numpy()
    X = np.column_stack([
        df["tx"].to_numpy(), df["uid"].to_numpy(), df["vid"].to_numpy(), df["t"].to_numpy(),
        np.log1p(df["amt_paid"].to_numpy()), np.log1p(df["amt_recv"].to_numpy()),
        enc("fmt"), enc("cur_paid"), enc("cur_recv"),
    ]).astype(np.float64)
    bins = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    params = {
        "num_threads": a.threads, "time_window": -1,
        "vertex_stats": True, "vertex_stats_tw": 5 * DAY, "vertex_stats_cols": [4, 5],
        "vertex_stats_feats": [0, 1, 2, 3, 4, 8, 9, 10],
        "fan": True, "fan_tw": DAY, "fan_bins": bins,
        "degree": True, "degree_tw": DAY, "degree_bins": bins,
        "scatter-gather": True, "scatter-gather_tw": 5 * DAY, "scatter-gather_bins": bins,
        "temp-cycle": True, "temp-cycle_tw": 5 * DAY, "temp-cycle_bins": bins,
        "lc-cycle": not a.no_lc_cycle, "lc-cycle_tw": 2 * DAY, "lc-cycle_len": 10, "lc-cycle_bins": bins,
    }
    g = snapml.GraphFeaturePreprocessor()
    g.set_params(params)
    log(f"GFP params: {params}")
    outs = []
    for i in range(0, len(X), a.batch):
        outs.append(g.transform(X[i:i + a.batch])[:, X.shape[1]:].astype(np.float32))
        if (i // a.batch) % 5000 == 0:
            log(f"  {i:,}/{len(X):,} transaksi")
    F = np.vstack(outs)
    cols = [f"gfp_{k}" for k in range(F.shape[1])]
    out = pl.DataFrame(F, schema=cols).with_columns(pl.Series("tx", df["tx"].to_numpy()))
    out.select(["tx"] + cols).write_parquet(Path(a.feat) / "gfp.parquet")
    log(f"gfp: {out.shape} -> {Path(a.feat) / 'gfp.parquet'}")


if __name__ == "__main__":
    main()
