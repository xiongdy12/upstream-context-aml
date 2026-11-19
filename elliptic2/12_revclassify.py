"""Reimplementasi RevClassify-DS (Song dkk., ICAIF 2024) + varian multi-hop usulan kita.

Definisi (mengikuti paper RevClassify):
  sources(H) = anggota subgraf dengan derajat-masuk internal 0; sinks(H) = derajat-keluar internal 0
  S1 = pengirim: node di luar H yang punya edge ke salah satu source
  R1 = penerima: node di luar H yang menerima edge dari salah satu sink
  (jika H tidak punya source/sink, mis. siklus, dipakai semua anggota)
Varian multi-hop (usulan):
  S2 = pengirim dari S1 (edge masuk ke node S1), S3 = pengirim dari S2
Setiap elemen himpunan = 43 fitur node (/100) + log(1 + bobot edge). Himpunan dibatasi top-K menurut bobot;
untuk hop >= 2, tiap node dibatasi top-K_hop pengirim terbesarnya (mencegah ledakan pada hub).

Model Deep Sets: phi (MLP) per elemen -> pooling mean+max+log|set| per himpunan -> MLP -> logit.
Varian: ds1 = {S1, R1} (setara RevClassify-DS), ds2 = {S1, R1, S2}, ds3 = {S1, R1, S2, S3}.
Split sama persis dengan 07/08 (groups.make_splits) -> bisa dibandingkan & di-ensemble lewat 09.

  python 12_revclassify.py --data dataset --feat features --out results_revds --variants ds1,ds2,ds3
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from common import load_schema
from groups import build_groups, make_splits


def topk(df, key, k, w="w"):
    return df.sort([key, w], descending=[False, True]).group_by(key, maintain_order=True).head(k)


# ============================================================================ bangun himpunan
def build_sets(data_dir, feat_dir, K, K_hop, cache):
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    log = lambda m: print(f"[set {time.time() - t0:6.0f}s] {m}", flush=True)
    s = load_schema(data_dir)
    d, f = Path(data_dir), Path(feat_dir)
    lab = pl.read_parquet(f / "labels.parquet").sort("row_order").with_row_index("gid")
    mem = (pl.read_csv(d / "nodes.csv", columns=[s["n_id"], s["n_cc"]])
           .rename({s["n_id"]: "clId", s["n_cc"]: "ccId"}).unique())
    e = pl.read_csv(d / "edges.csv").rename({s["e_src"]: "src", s["e_dst"]: "dst"}).select("src", "dst")
    e = (e.join(mem.rename({"clId": "src"}), on="src")
          .join(mem.rename({"clId": "dst", "ccId": "cc2"}), on="dst")
          .filter(pl.col("ccId") == pl.col("cc2")).drop("cc2"))
    has_in = e.select("ccId", pl.col("dst").alias("clId")).unique().with_columns(hi=pl.lit(True))
    has_out = e.select("ccId", pl.col("src").alias("clId")).unique().with_columns(ho=pl.lit(True))
    m2 = (mem.join(has_in, on=["ccId", "clId"], how="left").join(has_out, on=["ccId", "clId"], how="left")
          .with_columns(is_src=pl.col("hi").is_null(), is_sink=pl.col("ho").is_null()))
    # fallback: subgraf tanpa source/sink -> semua anggota
    m2 = m2.with_columns(
        is_src=pl.col("is_src") | (~pl.col("is_src").any().over("ccId")),
        is_sink=pl.col("is_sink") | (~pl.col("is_sink").any().over("ccId")))
    sources = m2.filter("is_src").select("ccId", "clId")
    sinks = m2.filter("is_sink").select("ccId", "clId")
    log(f"source: {len(sources):,}; sink: {len(sinks):,}")

    touch = pl.read_parquet(f / "bg_touch.parquet", columns=["src", "dst"])
    member = mem.rename({"clId": "node", "ccId": "cc_node"})
    S1 = (touch.join(sources.rename({"clId": "dst"}), on="dst")
          .join(member.rename({"node": "src"}), on="src", how="left")
          .filter(pl.col("cc_node").is_null() | (pl.col("cc_node") != pl.col("ccId")))
          .group_by("ccId", pl.col("src").alias("node")).agg(pl.len().alias("w")))
    R1 = (touch.join(sinks.rename({"clId": "src"}), on="src")
          .join(member.rename({"node": "dst"}), on="dst", how="left")
          .filter(pl.col("cc_node").is_null() | (pl.col("cc_node") != pl.col("ccId")))
          .group_by("ccId", pl.col("dst").alias("node")).agg(pl.len().alias("w")))
    S1, R1 = topk(S1, "ccId", K), topk(R1, "ccId", K)
    log(f"S1: {len(S1):,} pasangan; R1: {len(R1):,} pasangan")
    sets = {"S1": S1, "R1": R1}

    def upstream(prev, hop_file, name):
        if not (f / hop_file).exists():
            log(f"{hop_file} tidak ada -> {name} dilewati")
            return None
        nodes = prev["node"].unique().implode()
        per = (pl.scan_parquet(f / hop_file).select("src", "dst").filter(pl.col("dst").is_in(nodes))
               .group_by("dst", "src").agg(pl.len().alias("w")).collect())
        per = topk(per, "dst", K_hop)  # top-K_hop pengirim per node
        nxt = (prev.select("ccId", pl.col("node").alias("dst")).join(per, on="dst")
               .group_by("ccId", pl.col("src").alias("node")).agg(pl.col("w").sum()))
        return topk(nxt, "ccId", K)

    S2 = upstream(S1, "bg_hop2_in.parquet", "S2")
    if S2 is not None:
        sets["S2"] = S2
        log(f"S2: {len(S2):,} pasangan")
        S3 = upstream(S2, "bg_hop3_in.parquet", "S3")
        if S3 is not None:
            sets["S3"] = S3
            log(f"S3: {len(S3):,} pasangan")

    # fitur node untuk semua node yang terlibat
    allnodes = pl.concat([v.select("node") for v in sets.values()]).unique()
    bn_feats = s["bn_feats"]
    bn = (pl.scan_csv(d / "background_nodes.csv").select([s["bn_id"]] + bn_feats)
          .rename({s["bn_id"]: "node"}).filter(pl.col("node").is_in(allnodes["node"].implode())).collect())
    bn = bn.with_row_index("row")
    X = bn.select(bn_feats).fill_null(0).to_numpy().astype(np.float32) / 100.0
    log(f"node unik dalam himpunan: {len(bn):,}")
    out = {"X": X, "y": lab["label"].to_numpy().astype(np.float32)}
    n = len(lab)
    for name, v in sets.items():
        v = (v.join(lab.select("ccId", "gid"), on="ccId").join(bn.select("node", "row"), on="node")
             .sort("gid", "w", descending=[False, True]))
        gid = v["gid"].to_numpy()
        out[f"{name}_row"] = v["row"].to_numpy().astype(np.int64)
        out[f"{name}_w"] = np.log1p(v["w"].to_numpy().astype(np.float32))
        out[f"{name}_ptr"] = np.concatenate([[0], np.cumsum(np.bincount(gid, minlength=n))]).astype(np.int64)
        sz = np.diff(out[f"{name}_ptr"])
        log(f"{name}: rata-rata ukuran himpunan {sz.mean():.1f}, kosong {np.mean(sz == 0):.1%}")
    np.savez(cache, **out)
    return out


# ============================================================================ model
class SetEnc(nn.Module):
    def __init__(self, din, h):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.h = h

    def forward(self, x, seg, B):
        z = self.phi(x)
        cnt = torch.zeros(B, device=z.device).index_add_(0, seg, torch.ones_like(seg, dtype=z.dtype))
        mean = torch.zeros(B, self.h, device=z.device).index_add_(0, seg, z) / cnt.clamp(min=1)[:, None]
        mx = torch.full((B, self.h), -1e4, device=z.device, dtype=z.dtype).scatter_reduce(
            0, seg[:, None].expand(-1, self.h), z, reduce="amax", include_self=True)
        mx = torch.where(cnt[:, None] > 0, mx, torch.zeros_like(mx))
        return torch.cat([mean, mx, torch.log1p(cnt)[:, None]], 1)


class DeepSets(nn.Module):
    def __init__(self, names, din, h=128, dropout=0.2):
        super().__init__()
        self.names = names
        self.enc = nn.ModuleDict({n: SetEnc(din, h) for n in names})
        self.head = nn.Sequential(nn.Linear(len(names) * (2 * h + 1), 2 * h), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(2 * h, 1))

    def forward(self, batch, B):
        return self.head(torch.cat([self.enc[n](*batch[n], B) for n in self.names], 1)).squeeze(-1)


def make_batch(S, names, ids, device):
    out = {}
    for n in names:
        ptr = S[f"{n}_ptr"]
        rng = [np.arange(ptr[i], ptr[i + 1]) for i in ids]
        idx = np.concatenate(rng) if rng else np.zeros(0, np.int64)
        seg = np.repeat(np.arange(len(ids)), [len(r) for r in rng])
        x = np.concatenate([S["X"][S[f"{n}_row"][idx]], S[f"{n}_w"][idx][:, None]], 1)
        out[n] = (torch.as_tensor(x, device=device), torch.as_tensor(seg, device=device))
    return out


@torch.no_grad()
def predict(model, S, names, ids, bs, device):
    model.eval()
    ps = []
    for i in range(0, len(ids), bs):
        b = ids[i:i + bs]
        ps.append(torch.sigmoid(model(make_batch(S, names, b, device), len(b))).cpu().numpy())
    return np.concatenate(ps)


def train_eval(S, names, tr, va, te, a, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    y = S["y"]
    model = DeepSets(names, S["X"].shape[1] + 1, a.hidden, a.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    pw = torch.tensor((len(tr) - y[tr].sum()) / max(y[tr].sum(), 1), device=device)
    best, state, bad = -1.0, None, 0
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), a.bs):
            b = perm[i:i + a.bs]
            logit = model(make_batch(S, names, b, device), len(b))
            loss = F.binary_cross_entropy_with_logits(logit, torch.as_tensor(y[b], device=device), pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        ap = average_precision_score(y[va], predict(model, S, names, va, a.bs * 4, device))
        if ap > best:
            best, bad, state = ap, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= a.patience:
            break
    model.load_state_dict(state)
    return predict(model, S, names, te, a.bs * 4, device), best, ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--feat", default="features")
    ap.add_argument("--out", default="results_revds")
    ap.add_argument("--variants", default="ds1,ds2,ds3")
    ap.add_argument("--splits", default="strat,group_src,group_K10")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--K", type=int, default=128, help="maks elemen per himpunan")
    ap.add_argument("--K_hop", type=int, default=16, help="maks pengirim per node pada hop >= 2")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    S = build_sets(a.data, a.feat, a.K, a.K_hop, Path(a.feat) / f"revsets_K{a.K}_Kh{a.K_hop}.npz")
    y = S["y"].astype(int)
    lab = pl.read_parquet(Path(a.feat) / "labels.parquet").sort("row_order")
    groups = pl.DataFrame({"ccId": lab["ccId"]}).join(build_groups(a.data, a.feat, Ks=(10,)), on="ccId", how="left")
    variants = {"ds1": ["S1", "R1"], "ds2": ["S1", "R1", "S2"], "ds3": ["S1", "R1", "S2", "S3"]}
    variants = {k: v for k, v in variants.items() if k in a.variants.split(",") and all(f"{n}_ptr" in S for n in v)}
    print("device:", device, "| varian:", variants, flush=True)

    rows, preds, t0 = [], [], time.time()
    for split, fold, tr, va, te in make_splits(y, groups, [int(x) for x in a.seeds.split(",")], a.splits.split(",")):
        for vname, names in variants.items():
            p, best_va, eps = train_eval(S, names, tr, va, te, a, device, fold)
            k = max(1, int(round(0.01 * len(te))))
            top = np.argsort(-p)[:k]
            rows.append({"split": split, "fold": fold, "set": vname, "model": "DeepSets",
                         "pr_auc": average_precision_score(y[te], p), "roc_auc": roc_auc_score(y[te], p),
                         "rec@1%": y[te][top].sum() / max(y[te].sum(), 1), "prec@1%": y[te][top].mean(),
                         "pos_test": int(y[te].sum()), "val_pr_auc": best_va, "epochs": eps})
            preds.append(pd.DataFrame({"split": split, "fold": fold, "set": vname, "model": "DeepSets",
                                       "idx": te, "y": y[te], "p": p.astype("float32")}))
            print(f"[{time.time() - t0:6.0f}s] {split:<9} f{fold} {vname} PR-AUC={rows[-1]['pr_auc']:.3f} "
                  f"(val {best_va:.3f}, {eps} epoch)", flush=True)
        pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)

    res = pd.DataFrame(rows)
    pd.concat(preds).to_parquet(out / "test_preds.parquet", index=False)
    g = res.groupby(["set", "split"])["pr_auc"]
    tab = (g.mean().round(3).astype(str) + " ± " + g.std().fillna(0).round(3).astype(str)).unstack("split")
    md = ["# RevClassify-DS (reimplementasi) & varian multi-hop\n",
          "ds1 = {S1,R1} setara RevClassify-DS (paper: PR-AUC 0.974 pada split acak 80/10/10); "
          "ds2/ds3 = + pengirim 2-hop/3-hop.\n", tab.to_markdown()]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
