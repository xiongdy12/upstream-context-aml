"""Gabungkan prediksi tabular (07) & GNN (08), plus uji signifikansi.

Untuk tiap split & fold yang ada di kedua hasil:
  - model tunggal: semua kombinasi (set, model) dari 07 + GNN dari 08
  - ensemble     : rata-rata RANK (tahan beda skala skor) dari GNN + model tabular terbaik per set
Uji: selisih berpasangan per fold (menang x/y) + bootstrap 95% CI selisih PR-AUC pada gabungan test fold.

  python 09_ensemble.py --tab results_phase2 --gnn results_gnn --out results_ensemble
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


def boot_diff(y, pa, pb, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(n):  # bootstrap terstratifikasi
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(average_precision_score(y[i], pa[i]) - average_precision_score(y[i], pb[i]))
    return np.percentile(d, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", default="results_phase2")
    ap.add_argument("--gnn", default="results_gnn")
    ap.add_argument("--out", default="results_ensemble")
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rd = lambda dirs: pd.concat([pd.read_parquet(Path(x) / "test_preds.parquet") for x in dirs.split(",")],
                                ignore_index=True).drop_duplicates(["split", "fold", "set", "model", "idx"], keep="last")
    tab = rd(a.tab)  # boleh beberapa folder dipisah koma
    gnn = rd(a.gnn)
    tab["name"] = tab["set"] + "|" + tab["model"]
    gnn["name"] = "GNN|" + gnn["model"]
    allp = pd.concat([tab, gnn], ignore_index=True)

    rows, pooled = [], {}
    for (split, fold), d in allp.groupby(["split", "fold"]):
        names = d["name"].unique()
        gname = [n for n in names if n.startswith("GNN")]
        if not gname:
            continue
        piv = d.pivot_table(index="idx", columns="name", values="p")
        y = d.drop_duplicates("idx").set_index("idx").loc[piv.index, "y"].values
        piv = piv.dropna(axis=1)
        cand = {n: piv[n].values for n in piv.columns}
        g = cand[gname[0]]
        for n in list(piv.columns):
            if n.startswith("GNN"):
                continue
            cand[f"ENS[{gname[0]} + {n}]"] = (rankdata(g) + rankdata(piv[n].values)) / (2 * len(g))
        for n, p in cand.items():
            rows.append({"split": split, "fold": fold, "name": n,
                         "pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p)})
            pooled.setdefault((split, n), []).append((y, p))

    res = pd.DataFrame(rows)
    res.to_csv(out / "scores.csv", index=False)
    g = res.groupby(["name", "split"])["pr_auc"]
    tab_s = (g.mean().round(3).astype(str) + " ± " + g.std().fillna(0).round(3).astype(str)).unstack("split")
    order = res[res.split == "group_src"].groupby("name")["pr_auc"].mean().sort_values(ascending=False).index \
        if "group_src" in res.split.values else tab_s.index
    tab_s = tab_s.reindex([n for n in order if n in tab_s.index])

    # uji: model/ensemble terbaik di group_src vs pembanding kunci
    md = ["# Ensemble & signifikansi\n", "## PR-AUC (mean ± std per fold), urut menurut group_src\n",
          tab_s.to_markdown(), "\n## Uji berpasangan (group_src)\n"]
    if "group_src" in res.split.values:
        gs = res[res.split == "group_src"].pivot_table(index="fold", columns="name", values="pr_auc")
        best = gs.mean().idxmax()
        refs = [n for n in gs.columns if n != best and (n.startswith("GNN") or "XGBoost" in n)]
        lines = []
        for r in refs:
            diff = gs[best] - gs[r]
            ys = np.concatenate([y for y, _ in pooled[("group_src", best)]])
            pb = np.concatenate([p for _, p in pooled[("group_src", best)]])
            pr = np.concatenate([p for _, p in pooled[("group_src", r)]])
            lo, hi = boot_diff(ys, pb, pr, a.n_boot)
            lines.append({"terbaik": best, "vs": r, "selisih_mean": round(diff.mean(), 3),
                          "menang_fold": f"{int((diff > 0).sum())}/{len(diff)}",
                          "CI95_bootstrap": f"[{lo:+.3f}, {hi:+.3f}]"})
        md.append(pd.DataFrame(lines).to_markdown(index=False))
        md.append("\nCatatan: CI dihitung pada gabungan prediksi test semua fold (pooled); "
                  "rank-ensemble dihitung per fold.")
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
