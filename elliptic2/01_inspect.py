"""Cek isi Elliptic2: header, 3 baris contoh, jumlah baris, distribusi label, skema terdeteksi.
Kirim output skrip ini jika ada kolom yang tidak terdeteksi.

  python 01_inspect.py --data dataset
"""
import argparse
import json
from pathlib import Path

import polars as pl

from common import FILES, collect, detect_schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--count_rows", action="store_true", help="hitung baris file besar (lambat)")
    a = ap.parse_args()
    d = Path(a.data)
    for f in FILES:
        p = d / f"{f}.csv"
        df = pl.read_csv(p, n_rows=3)
        print(f"\n=== {f}.csv  ({p.stat().st_size / 1e9:.2f} GB)  {len(df.columns)} kolom")
        print("kolom:", df.columns[:12], "..." if len(df.columns) > 12 else "")
        print(df.select(df.columns[:8]))
        if a.count_rows or not f.startswith("background"):
            n = collect(pl.scan_csv(p).select(pl.len())).item()
            print(f"baris: {n:,}")
    s = detect_schema(d)
    cc = pl.read_csv(d / "connected_components.csv")
    print("\nDistribusi label:\n", cc[s["cc_label"]].value_counts())
    print("\nSkema terdeteksi:")
    print(json.dumps({k: (v if not isinstance(v, list) else f"{len(v)} kolom: {v[:4]}...")
                      for k, v in s.items()}, indent=1))


if __name__ == "__main__":
    main()
