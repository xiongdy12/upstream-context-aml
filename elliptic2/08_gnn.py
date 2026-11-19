"""GNN pembanding dengan fitur penuh + konteks 1-hop (PyTorch murni, tanpa PyG).

Graf per subgraf (ego-graph):
  node = anggota subgraf (peran 0) + hingga M pengirim eksternal teratas (peran 1)
         + hingga M penerima eksternal teratas (peran 2); peran 3 = pengirim sekaligus penerima
  fitur node = 43 fitur node (/100) + one-hot peran
  edge = edge internal + edge batas (anggota <-> tetangga terpilih), multi-edge digabung
         (rata-rata 95 fitur edge /100 + log jumlah transaksi); ditambah edge balik (flag arah)
Model: encoder MLP -> L lapis GINE (residual + LayerNorm) -> pooling mean & max per kelompok peran
       (anggota / pengirim / penerima) -> MLP -> logit. Loss BCE berbobot kelas, early stopping PR-AUC val.
Split: sama persis dengan 07_eval.py (groups.make_splits) -> hasil bisa dibandingkan & di-ensemble.

  python 08_gnn.py --data dataset --feat features --out results_gnn --splits group_src,strat --seeds 0
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


# ============================================================================ bangun graf
def build_graphs(data_dir, feat_dir, M, cache):
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    log = lambda m: print(f"[graf {time.time() - t0:6.0f}s] {m}", flush=True)
    s = load_schema(data_dir)
    d, f = Path(data_dir), Path(feat_dir)
    bn_feats, be_feats = s["bn_feats"], s["be_feats"]
    lab = pl.read_parquet(f / "labels.parquet").sort("row_order").with_row_index("gid")
    mem = (pl.read_csv(d / "nodes.csv", columns=[s["n_id"], s["n_cc"]])
           .rename({s["n_id"]: "clId", s["n_cc"]: "ccId"}).unique())
    touch = pl.read_parquet(f / "bg_touch.parquet")
    if s["tx_col"] and s["tx_col"] in touch.columns and s["tx_col"] != "txId":
        touch = touch.rename({s["tx_col"]: "txId"})

    # --- node: anggota + tetangga top-M per arah
    nb = []
    for me, other, role in (("dst", "src", 1), ("src", "dst", 2)):
        p = (touch.select(me, other).join(mem.rename({"clId": me}), on=me)
             .join(mem.rename({"clId": other, "ccId": "cc_o"}), on=other, how="left")
             .filter(pl.col("cc_o").is_null() | (pl.col("cc_o") != pl.col("ccId")))
             .group_by("ccId", pl.col(other).alias("clId")).agg(pl.len().alias("w"))
             .sort(["ccId", "w"], descending=[False, True])
             .with_columns(r=pl.col("w").cum_count().over("ccId"))
             .filter(pl.col("r") <= M).select("ccId", "clId", pl.lit(role).alias("role")))
        nb.append(p)
    nbr = (pl.concat(nb).group_by("ccId", "clId").agg(pl.col("role").sum().alias("role")))  # 1+2 -> 3
    nodes = (pl.concat([mem.select("ccId", "clId", pl.lit(0, dtype=pl.Int32).alias("role")),
                        nbr.select("ccId", "clId", pl.col("role").cast(pl.Int32))])
             .join(lab.select("ccId", "gid"), on="ccId").sort("gid", "role", "clId")
             .with_columns(lid=(pl.col("clId").cum_count().over("gid") - 1).cast(pl.Int64)))
    log(f"node total: {len(nodes):,}")
    bn = (pl.scan_csv(d / "background_nodes.csv").select([s["bn_id"]] + bn_feats)
          .rename({s["bn_id"]: "clId"}).filter(pl.col("clId").is_in(nodes["clId"].unique().implode())))
    bn = bn.collect()
    nodes = nodes.join(bn, on="clId", how="left")
    x_feat = nodes.select(bn_feats).fill_null(0).to_numpy().astype(np.float32) / 100.0
    role = nodes["role"].to_numpy()
    x = np.concatenate([x_feat, np.eye(4, dtype=np.float32)[role]], 1)
    node_gid = nodes["gid"].to_numpy()

    # --- edge: sisi anggota menentukan ccId (anggota hanya di satu subgraf -> tanpa ledakan join)
    key = nodes.select("ccId", "clId", "lid")
    parts = []
    for me in ("src", "dst"):
        other = "dst" if me == "src" else "src"
        e = (touch.join(mem.rename({"clId": me}), on=me)
             .join(key.rename({"clId": me, "lid": f"l_{me}"}), on=["ccId", me])
             .join(key.rename({"clId": other, "lid": f"l_{other}"}), on=["ccId", other]))
        parts.append(e)
    keycols = ["ccId", "l_src", "l_dst"] + (["txId"] if "txId" in touch.columns else ["src", "dst"])
    e = pl.concat(parts, how="diagonal_relaxed").unique(subset=keycols)
    e = (e.group_by("ccId", "l_src", "l_dst")
         .agg([pl.len().alias("ntx")] + [pl.col(c).mean() for c in be_feats])
         .join(lab.select("ccId", "gid"), on="ccId").sort("gid", "l_src", "l_dst"))
    log(f"edge total (setelah gabung multi-edge): {len(e):,}")
    ea = np.concatenate([e.select(be_feats).fill_null(0).to_numpy().astype(np.float32) / 100.0,
                         np.log1p(e["ntx"].to_numpy().astype(np.float32))[:, None]], 1)
    ei = np.stack([e["l_src"].to_numpy(), e["l_dst"].to_numpy()]).astype(np.int64)
    edge_gid = e["gid"].to_numpy()
    n = len(lab)
    node_ptr = np.concatenate([[0], np.cumsum(np.bincount(node_gid, minlength=n))])
    edge_ptr = np.concatenate([[0], np.cumsum(np.bincount(edge_gid, minlength=n))])
    g = dict(x=x, role=role.astype(np.int8), ei=ei, ea=ea, node_ptr=node_ptr, edge_ptr=edge_ptr,
             y=lab["label"].to_numpy().astype(np.float32))
    np.savez(cache, **g)
    log(f"cache -> {cache}")
    return g


def make_batch(g, ids, device):
    nr = [np.arange(g["node_ptr"][i], g["node_ptr"][i + 1]) for i in ids]
    er = [np.arange(g["edge_ptr"][i], g["edge_ptr"][i + 1]) for i in ids]
    nsz = np.array([len(r) for r in nr])
    esz = np.array([len(r) for r in er])
    nidx, eidx = np.concatenate(nr), np.concatenate(er)
    off = np.repeat(np.concatenate([[0], np.cumsum(nsz)[:-1]]), esz)
    ei = g["ei"][:, eidx] + off
    batch = np.repeat(np.arange(len(ids)), nsz)
    T = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    return (T(g["x"][nidx], torch.float32), T(g["role"][nidx], torch.long), T(ei, torch.long),
            T(g["ea"][eidx], torch.float32), T(batch, torch.long), T(g["y"][ids], torch.float32))


# ============================================================================ model
class GINE(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(h, 2 * h), nn.GELU(), nn.Linear(2 * h, h))
        self.eps = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(h)

    def forward(self, x, ei, e):
        msg = F.gelu(x[ei[0]] + e)
        agg = torch.zeros_like(x).index_add_(0, ei[1], msg)
        return self.norm(x + self.mlp((1 + self.eps) * x + agg))


class SubgraphGNN(nn.Module):
    def __init__(self, dx, de, h=128, layers=3, dropout=0.2):
        super().__init__()
        self.xe = nn.Sequential(nn.Linear(dx, h), nn.GELU(), nn.Linear(h, h))
        self.ee = nn.Sequential(nn.Linear(de + 1, h), nn.GELU(), nn.Linear(h, h))
        self.layers = nn.ModuleList([GINE(h) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(6 * h, 2 * h), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * h, 1))
        self.h = h

    def forward(self, x, role, ei, ea, batch, B):
        h = self.xe(x)
        # edge dua arah: flag 0 = arah asli, 1 = balik
        ei2 = torch.cat([ei, ei.flip(0)], 1)
        ea2 = torch.cat([torch.cat([ea, torch.zeros_like(ea[:, :1])], 1),
                         torch.cat([ea, torch.ones_like(ea[:, :1])], 1)], 0)
        e = self.ee(ea2)
        for lyr in self.layers:
            h = self.drop(lyr(h, ei2, e))
        grp = torch.where(role == 0, 0, torch.where(role == 2, 2, 1))  # anggota / pengirim(+keduanya) / penerima
        seg = batch * 3 + grp
        cnt = torch.zeros(B * 3, device=h.device).index_add_(0, seg, torch.ones_like(seg, dtype=h.dtype))
        mean = torch.zeros(B * 3, self.h, device=h.device).index_add_(0, seg, h) / cnt.clamp(min=1)[:, None]
        mx = torch.full((B * 3, self.h), -1e4, device=h.device).scatter_reduce(
            0, seg[:, None].expand(-1, self.h), h, reduce="amax", include_self=True)
        mx = torch.where(cnt[:, None] > 0, mx, torch.zeros_like(mx))
        z = torch.cat([mean.view(B, 3 * self.h), mx.view(B, 3 * self.h)], 1)
        return self.head(z).squeeze(-1)


@torch.no_grad()
def predict(model, g, ids, bs, device):
    model.eval()
    out = []
    for i in range(0, len(ids), bs):
        b = ids[i:i + bs]
        x, r, ei, ea, bt, _ = make_batch(g, b, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(torch.sigmoid(model(x, r, ei, ea, bt, len(b)).float()).cpu().numpy())
    return np.concatenate(out)


def train_one(g, tr, va, args, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    y = g["y"]
    model = SubgraphGNN(g["x"].shape[1], g["ea"].shape[1], args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pw = torch.tensor((len(tr) - y[tr].sum()) / max(y[tr].sum(), 1), device=device)
    best, best_state, bad = -1, None, 0
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), args.bs):
            b = perm[i:i + args.bs]
            x, r, ei, ea, bt, yb = make_batch(g, b, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logit = model(x, r, ei, ea, bt, len(b)).float()
            loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        ap = average_precision_score(y[va], predict(model, g, va, args.bs * 2, device))
        if ap > best:
            best, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"    ep{ep:02d} loss={loss.item():.4f} val PR-AUC={ap:.3f}", flush=True)
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--feat", default="features")
    ap.add_argument("--out", default="results_gnn")
    ap.add_argument("--splits", default="group_src,strat")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--M", type=int, default=32, help="maks tetangga eksternal per arah")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    print("device:", device)

    g = build_graphs(a.data, a.feat, a.M, Path(a.feat) / f"gnn_graphs_M{a.M}.npz")
    y = g["y"].astype(int)
    lab = pl.read_parquet(Path(a.feat) / "labels.parquet").sort("row_order")
    groups = pl.DataFrame({"ccId": lab["ccId"]}).join(build_groups(a.data, a.feat, Ks=(10,)), on="ccId", how="left")
    print(f"graf: {len(y):,}, node rata-rata {g['x'].shape[0] / len(y):.1f}, edge rata-rata {g['ei'].shape[1] / len(y):.1f}")

    rows, preds, t0 = [], [], time.time()
    for split, fold, tr, va, te in make_splits(y, groups, [int(s) for s in a.seeds.split(",")], a.splits.split(",")):
        print(f"[{time.time() - t0:6.0f}s] {split} fold {fold}", flush=True)
        model = train_one(g, tr, va, a, device, fold)
        p = predict(model, g, te, a.bs * 2, device)
        k = max(1, int(round(0.01 * len(te))))
        top = np.argsort(-p)[:k]
        rows.append({"split": split, "fold": fold, "set": "gnn_ctx1", "model": "GINE",
                     "pr_auc": average_precision_score(y[te], p), "roc_auc": roc_auc_score(y[te], p),
                     "rec@1%": y[te][top].sum() / max(y[te].sum(), 1), "prec@1%": y[te][top].mean(),
                     "pos_test": int(y[te].sum())})
        preds.append(pd.DataFrame({"split": split, "fold": fold, "set": "gnn_ctx1", "model": "GINE",
                                   "idx": te, "y": y[te], "p": p.astype("float32")}))
        print(f"  -> test PR-AUC={rows[-1]['pr_auc']:.3f} ROC-AUC={rows[-1]['roc_auc']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)

    res = pd.DataFrame(rows)
    pd.concat(preds).to_parquet(out / "test_preds.parquet", index=False)
    gg = res.groupby("split")["pr_auc"]
    md = ["# GNN (GINE, fitur penuh + konteks 1-hop)\n",
          (gg.mean().round(3).astype(str) + " ± " + gg.std().fillna(0).round(3).astype(str)).to_frame("PR-AUC").to_markdown()]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
