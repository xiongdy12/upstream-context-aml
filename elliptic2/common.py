"""Utilitas bersama: deteksi skema kolom Elliptic2 & pembacaan dengan polars.

Skema yang diharapkan (berdasarkan repo resmi MITIBMxGraph/Elliptic2):
  background_nodes.csv      : clId + 43 fitur node (ordinal/binned)
  background_edges.csv      : clId1, clId2 + 95 fitur edge
  nodes.csv                 : clId, ccId                 (anggota tiap subgraf)
  edges.csv                 : clId1, clId2 (+ccId?) + fitur edge (edge di dalam subgraf)
  connected_components.csv  : ccId, ccLabel ('licit' / 'suspicious')
Nama kolom dideteksi otomatis; jika berbeda, set lewat --schema JSON (lihat README).
"""
import json
from pathlib import Path

import polars as pl

FILES = ["background_nodes", "background_edges", "nodes", "edges", "connected_components"]


def collect(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:  # polars lama
        return lf.collect(streaming=True)


def header(path: Path) -> list[str]:
    return pl.read_csv(path, n_rows=5).columns


def _pick(cols, names, fallback_pos=None):
    low = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    if fallback_pos is not None:
        real = [c for c in cols if not (c == "" or c.startswith("Unnamed") or c.lower() in ("index",))]
        return real[fallback_pos]
    raise KeyError(f"Kolom {names} tidak ditemukan di {cols}")


def detect_schema(data_dir: Path, override: dict | None = None) -> dict:
    d = Path(data_dir)
    h = {f: header(d / f"{f}.csv") for f in FILES}
    s = {
        "bn_id": _pick(h["background_nodes"], ["clId", "cl_id", "node_id"], 0),
        "be_src": _pick(h["background_edges"], ["clId1", "src", "source"], 0),
        "be_dst": _pick(h["background_edges"], ["clId2", "dst", "target"], 1),
        "n_id": _pick(h["nodes"], ["clId", "cl_id", "node_id"], 0),
        "n_cc": _pick(h["nodes"], ["ccId", "cc_id", "subgraph_id"], 1),
        "e_src": _pick(h["edges"], ["clId1", "src", "source"], 0),
        "e_dst": _pick(h["edges"], ["clId2", "dst", "target"], 1),
        "cc_id": _pick(h["connected_components"], ["ccId", "cc_id", "subgraph_id"], 0),
        "cc_label": _pick(h["connected_components"], ["ccLabel", "label"], 1),
    }
    e_cc = [c for c in h["edges"] if c.lower() in ("ccid", "cc_id")]
    s["e_cc"] = e_cc[0] if e_cc else None
    if override:
        s.update(override)
    ignore = lambda c: (c == "" or c.startswith("Unnamed")
                        or c.lower() in ("index", "txid", "tx_id"))
    s["tx_col"] = next((c for c in h["background_edges"] if c.lower() in ("txid", "tx_id")), None)
    s["e_tx"] = next((c for c in h["edges"] if c.lower() in ("txid", "tx_id")), None)
    s["bn_feats"] = [c for c in h["background_nodes"] if c != s["bn_id"] and not ignore(c)]
    s["be_feats"] = [c for c in h["background_edges"]
                     if c not in (s["be_src"], s["be_dst"]) and not ignore(c)]
    s["e_feats"] = [c for c in h["edges"]
                    if c not in (s["e_src"], s["e_dst"], s["e_cc"]) and not ignore(c)]
    # Elliptic2 asli: edges.csv hanya (clId1, clId2, txId); fitur diambil dari background_edges
    return s


def load_schema(data_dir, schema_json=None):
    override = json.loads(Path(schema_json).read_text()) if schema_json else None
    return detect_schema(Path(data_dir), override)
