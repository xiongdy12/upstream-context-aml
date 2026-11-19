"""Tuning hyperparameter yang adil untuk RevClassify-DS (ds1) dan varian 2-hop (ds2) [dan opsional ds3].

Tahap 1 (tuning): grid lr x hidden x dropout, anggaran sama (maks 200 epoch, patience 15).
  Skor = rata-rata PR-AUC VALIDASI pada split penala: strat seed 0 + group_src fold 0.
  Data test TIDAK dipakai sama sekali pada tahap ini.
Tahap 2 (evaluasi): konfigurasi terbaik tiap varian dievaluasi pada semua split
  (strat 3 seed, group_src 5 fold, group_K10 5 fold). Output format sama dengan 12 -> bisa ke 09.

  python 13_tune.py --data dataset --feat features --out results_tuned --variants ds1,ds2
"""
import argparse
import importlib.util
import itertools
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from groups import build_groups, make_splits

spec = importlib.util.spec_from_file_location("rc", Path(__file__).with_name("12_revclassify.py"))
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)
VARIANTS = {"ds1": ["S1", "R1"], "ds2": ["S1", "R1", "S2"], "ds3": ["S1", "R1", "S2", "S3"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--feat", default="features")
    ap.add_argument("--out", default="results_tuned")
    ap.add_argument("--variants", default="ds1,ds2")
    ap.add_argument("--lrs", default="0.0003,0.001,0.003")
    ap.add_argument("--hiddens", default="64,128,256")
    ap.add_argument("--dropouts", default="0.0,0.2")
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--K_hop", type=int, default=16)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--eval_splits", default="strat,group_src,group_K10")
    ap.add_argument("--group_seeds", default="0", help="pengacakan group k-fold, mis. 0,1,2 (3x5 fold)")
    ap.add_argument("--tune_from", default=None, help="folder berisi tuning.csv yang sudah ada (lewati tuning)")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    S = rc.build_sets(a.data, a.feat, a.K, a.K_hop, Path(a.feat) / f"revsets_K{a.K}_Kh{a.K_hop}.npz")
    y = S["y"].astype(int)
    lab = pl.read_parquet(Path(a.feat) / "labels.parquet").sort("row_order")
    groups = pl.DataFrame({"ccId": lab["ccId"]}).join(build_groups(a.data, a.feat, Ks=(10,)), on="ccId", how="left")
    base_splits = list(make_splits(y, groups, [0], ["strat", "group_src"]))
    tune_splits = [sp for sp in base_splits if (sp[0], sp[1]) in (("strat", 0), ("group_src", 0))]
    all_splits = list(make_splits(y, groups, [int(s) for s in a.seeds.split(",")], a.eval_splits.split(","),
                                  group_seeds=[int(s) for s in a.group_seeds.split(",")]))
    if a.tune_from:
        import shutil
        shutil.copy(Path(a.tune_from) / "tuning.csv", out / "tuning.csv")
    grid = list(itertools.product([float(x) for x in a.lrs.split(",")], [int(x) for x in a.hiddens.split(",")],
                                  [float(x) for x in a.dropouts.split(",")]))
    variants = {k: VARIANTS[k] for k in a.variants.split(",")}
    print(f"device {device}; {len(grid)} konfigurasi x {len(variants)} varian x {len(tune_splits)} split penala", flush=True)

    # ------------------------------------------------------------------ tahap 1: tuning (validasi saja)
    t0, trows = time.time(), []
    tune_csv = out / "tuning.csv"
    done = pd.read_csv(tune_csv) if tune_csv.exists() else pd.DataFrame()
    for vname, names in variants.items():
        for lr, h, dp in grid:
            for split, fold, tr, va, te in tune_splits:
                key = (vname, lr, h, dp, split, fold)
                if len(done) and ((done.variant == vname) & (done.lr == lr) & (done.hidden == h) &
                                  (done.dropout == dp) & (done.split == split) & (done.fold == fold)).any():
                    continue
                cfg = SimpleNamespace(hidden=h, dropout=dp, lr=lr, bs=a.bs, epochs=a.epochs, patience=a.patience)
                _, best_va, eps = rc.train_eval(S, names, tr, va, va, cfg, device, fold)  # test = val (tidak dipakai)
                trows.append(dict(zip(["variant", "lr", "hidden", "dropout", "split", "fold"], key),
                                  val_pr_auc=best_va, epochs=eps))
                print(f"[{time.time() - t0:6.0f}s] tuning {vname} lr={lr} h={h} dp={dp} {split} f{fold}: "
                      f"val PR-AUC={best_va:.3f} ({eps} ep)", flush=True)
                pd.concat([done, pd.DataFrame(trows)]).to_csv(tune_csv, index=False)
    tun = pd.read_csv(tune_csv)
    agg = tun.groupby(["variant", "lr", "hidden", "dropout"])["val_pr_auc"].mean().reset_index()
    best = agg.loc[agg.groupby("variant")["val_pr_auc"].idxmax()].set_index("variant")
    best.to_csv(out / "best_configs.csv")
    json.dump(best.reset_index().to_dict("records"), open(out / "best_configs.json", "w"), indent=1)
    print("konfigurasi terbaik (rata-rata val PR-AUC):\n", best, flush=True)

    # ------------------------------------------------------------------ tahap 2: evaluasi penuh
    rows, preds = [], []
    for split, fold, tr, va, te in all_splits:
        for vname, names in variants.items():
            b = best.loc[vname]
            cfg = SimpleNamespace(hidden=int(b.hidden), dropout=float(b.dropout), lr=float(b.lr), bs=a.bs,
                                  epochs=a.epochs, patience=a.patience)
            p, best_va, eps = rc.train_eval(S, names, tr, va, te, cfg, device, fold)
            k = max(1, int(round(0.01 * len(te))))
            top = np.argsort(-p)[:k]
            rows.append({"split": split, "fold": fold, "set": f"{vname}_tuned", "model": "DeepSets",
                         "pr_auc": average_precision_score(y[te], p), "roc_auc": roc_auc_score(y[te], p),
                         "rec@1%": y[te][top].sum() / max(y[te].sum(), 1), "prec@1%": y[te][top].mean(),
                         "pos_test": int(y[te].sum()), "val_pr_auc": best_va, "epochs": eps})
            preds.append(pd.DataFrame({"split": split, "fold": fold, "set": f"{vname}_tuned", "model": "DeepSets",
                                       "idx": te, "y": y[te], "p": p.astype("float32")}))
            print(f"[{time.time() - t0:6.0f}s] eval {split:<9} f{fold} {vname} PR-AUC={rows[-1]['pr_auc']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)
    res = pd.DataFrame(rows)
    pd.concat(preds).to_parquet(out / "test_preds.parquet", index=False)
    g = res.groupby(["set", "split"])["pr_auc"]
    tab = (g.mean().round(3).astype(str) + " ± " + g.std().fillna(0).round(3).astype(str)).unstack("split")
    per = res[res.split == "group_src"].pivot_table(index="fold", columns="set", values="pr_auc").round(3)
    sig = []
    if {"ds1_tuned", "ds2_tuned"} <= set(per.columns):
        from scipy.stats import wilcoxon
        for a_, b_ in (("ds2_tuned", "ds1_tuned"), ("ds3_tuned", "ds1_tuned"), ("ds3_tuned", "ds2_tuned")):
            if a_ in per.columns and b_ in per.columns:
                dlt = (per[a_] - per[b_]).dropna()
                pv = wilcoxon(dlt, alternative="greater").pvalue if len(dlt) >= 6 else float("nan")
                sig.append({"vs": f"{a_} - {b_}", "n_fold": len(dlt), "selisih_mean": round(dlt.mean(), 4),
                            "selisih_median": round(dlt.median(), 4), "menang": f"{int((dlt > 0).sum())}/{len(dlt)}",
                            "wilcoxon_p_satu_sisi": pv})
    md = ["# Setelah tuning adil (dipilih dengan PR-AUC validasi; test tidak dipakai saat tuning)\n",
          "## Konfigurasi terpilih\n", best.round(4).to_markdown(),
          "\n\n## PR-AUC test (mean ± std)\n", tab.to_markdown(),
          "\n\n## group_src per fold (fold = 100*pengacakan + k)\n", per.to_markdown(),
          "\n\n## Uji berpasangan group_src (Wilcoxon signed-rank, satu sisi)\n",
          pd.DataFrame(sig).to_markdown(index=False) if sig else "-"]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
