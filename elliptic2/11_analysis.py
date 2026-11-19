"""Fase 3: ablasi blok fitur, uji shortcut top-k, dan interpretasi SHAP pada model terbaik.

Model dasar : XGBoost dengan set fitur terlengkap (internal+ctx1+ctx2[+ctx3])
Split       : group_src (5 fold, utama) + strat seed 0
Ablasi      : buang satu blok fitur sekaligus (leave-one-block-out) -> penurunan PR-AUC
Shortcut    : model hanya dengan top-1/5/20 fitur (peringkat SHAP)
SHAP        : XGBoost pred_contribs (TreeSHAP) pada test fold group_src -> kontribusi per fitur & blok

  python 11_analysis.py --data dataset --feat features --out results_analysis
"""
import argparse
import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from groups import build_groups, make_splits

warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("ev", Path(__file__).with_name("07_eval.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

BLOCK_RULES = [  # (nama blok, fungsi pencocok) — urutan penting
    ("hop1_node_pengirim", lambda c: c.startswith("bg_innb")),
    ("hop1_node_penerima", lambda c: c.startswith("bg_outnb")),
    ("hop1_edge", lambda c: c.startswith("bg_ine") or c.startswith("bg_oute")),
    ("hop1_derajat", lambda c: c.startswith("bg_")),
    ("hop2_node", lambda c: c.startswith("h2nb")),
    ("hop2_edge", lambda c: c.startswith("h2e")),
    ("hop2_derajat", lambda c: c.startswith("h2_")),
    ("hop3_node", lambda c: c.startswith("h3nb")),
    ("hop3_edge", lambda c: c.startswith("h3e")),
    ("hop3_derajat", lambda c: c.startswith("h3_")),
]


def block_of(c, internal_cols):
    if c in internal_cols:
        return "internal"
    for name, f in BLOCK_RULES:
        if f(c):
            return name
    return "lain"


def fit_xgb(X, y, tr, va, seed, gpu):
    import xgboost as xgb
    pw = (len(tr) - y[tr].sum()) / max(y[tr].sum(), 1)
    m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=8, subsample=0.8,
                          colsample_bytree=0.6, tree_method="hist", device="cuda" if gpu else "cpu",
                          scale_pos_weight=pw, eval_metric="aucpr", early_stopping_rounds=150,
                          random_state=seed)
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--feat", default="features")
    ap.add_argument("--out", default="results_analysis")
    ap.add_argument("--splits", default="group_src,strat")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gpu = not a.cpu
    t0 = time.time()

    df, blocks = ev.load(a.feat)
    y = df["label"].values.astype(int)
    fsets = ev.feature_sets(blocks)
    base_name = "internal+ctx1+ctx2+ctx3" if "internal+ctx1+ctx2+ctx3" in fsets else "internal+ctx1+ctx2"
    base = fsets[base_name]
    X = df[base].astype("float32")
    bmap = {c: block_of(c, set(blocks["internal"])) for c in base}
    blist = sorted(set(bmap.values()), key=lambda b: list(bmap.values()).index(b))
    print(f"model dasar: {base_name} ({len(base)} fitur); blok: "
          + ", ".join(f"{b}={sum(v == b for v in bmap.values())}" for b in blist), flush=True)
    groups = pl.DataFrame({"ccId": df["ccId"].values}).join(build_groups(a.data, a.feat, Ks=(10,)),
                                                           on="ccId", how="left")
    splits = list(make_splits(y, groups, (0,), a.splits.split(",")))

    # ---------------------------------------------------------------- model penuh + SHAP
    rows, shap_abs, shap_sum, n_shap = [], np.zeros(len(base)), np.zeros(len(base)), 0
    dir_rows = []
    import xgboost as xgb
    for split, fold, tr, va, te in splits:
        m = fit_xgb(X, y, tr, va, fold, gpu)
        p = m.predict_proba(X.iloc[te])[:, 1]
        rows.append({"split": split, "fold": fold, "variant": "PENUH", "pr_auc": average_precision_score(y[te], p)})
        if split == "group_src":
            contrib = m.get_booster().predict(xgb.DMatrix(X.iloc[te]), pred_contribs=True)[:, :-1]
            shap_abs += np.abs(contrib).sum(0)
            shap_sum += contrib.sum(0)
            n_shap += len(te)
            dir_rows.append((contrib, X.iloc[te].values, y[te]))
        print(f"[{time.time() - t0:6.0f}s] PENUH {split} f{fold} PR-AUC={rows[-1]['pr_auc']:.3f}", flush=True)

    imp = pd.DataFrame({"fitur": base, "blok": [bmap[c] for c in base],
                        "mean_abs_shap": shap_abs / max(n_shap, 1), "mean_shap": shap_sum / max(n_shap, 1)})
    # arah pengaruh: korelasi Spearman nilai fitur vs SHAP (gabungan fold)
    C = np.concatenate([d[0] for d in dir_rows])
    V = np.concatenate([d[1] for d in dir_rows])
    Ysh = np.concatenate([d[2] for d in dir_rows])
    top_idx = np.argsort(-imp["mean_abs_shap"].values)[:30]
    arah, mean_sus, mean_lic = {}, {}, {}
    for j in top_idx:
        v, c = V[:, j], C[:, j]
        ok = ~np.isnan(v)
        arah[j] = spearmanr(v[ok], c[ok]).correlation if ok.sum() > 10 else np.nan
        mean_sus[j] = np.nanmean(v[Ysh == 1])
        mean_lic[j] = np.nanmean(v[Ysh == 0])
    imp["arah_rho"] = [arah.get(j, np.nan) for j in range(len(base))]
    imp["rata2_nilai_suspicious"] = [mean_sus.get(j, np.nan) for j in range(len(base))]
    imp["rata2_nilai_licit"] = [mean_lic.get(j, np.nan) for j in range(len(base))]
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(out / "shap_fitur.csv", index=False)
    blk = imp.groupby("blok")["mean_abs_shap"].sum().sort_values(ascending=False)
    blk_share = (blk / blk.sum()).round(3)
    ranked = imp["fitur"].tolist()

    # ---------------------------------------------------------------- ablasi & shortcut
    variants = {f"tanpa_{b}": [c for c in base if bmap[c] != b] for b in blist}
    for k in (1, 5, 20):
        variants[f"hanya_top{k}"] = ranked[:k]
    for vname, cols in variants.items():
        if not cols:
            continue
        Xv = df[cols].astype("float32")
        for split, fold, tr, va, te in splits:
            m = fit_xgb(Xv, y, tr, va, fold, gpu)
            p = m.predict_proba(Xv.iloc[te])[:, 1]
            rows.append({"split": split, "fold": fold, "variant": vname,
                         "pr_auc": average_precision_score(y[te], p)})
        print(f"[{time.time() - t0:6.0f}s] {vname} selesai", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(out / "ablation_scores.csv", index=False)
    full = res[res.variant == "PENUH"].set_index(["split", "fold"])["pr_auc"]
    res["delta"] = res.apply(lambda r: r.pr_auc - full.get((r.split, r.fold), np.nan), axis=1)
    summ = res.groupby(["variant", "split"]).agg(pr_auc=("pr_auc", "mean"), delta=("delta", "mean"),
                                                 turun_di=("delta", lambda d: f"{int((d < 0).sum())}/{len(d)}"))
    summ = summ.unstack("split")
    summ.columns = [f"{a_}_{b_}" for a_, b_ in summ.columns]
    order = ["PENUH"] + [v for v in variants if v in summ.index]
    summ = summ.reindex(order)

    md = [f"# Fase 3: ablasi, shortcut, SHAP\n\nModel dasar: XGBoost `{base_name}` ({len(base)} fitur). "
          "delta = selisih PR-AUC terhadap model penuh pada fold yang sama (negatif = blok itu penting).\n",
          "## Ablasi & uji shortcut\n", summ.round(3).to_markdown(),
          "\n\n## Porsi kontribusi SHAP per blok (group_src)\n", blk_share.to_frame("porsi").to_markdown(),
          "\n\n## 20 fitur teratas (mean |SHAP|, group_src)\n",
          "arah_rho > 0: nilai fitur tinggi mendorong ke 'suspicious'.\n",
          imp.head(20).round(4).to_markdown(index=False),
          "\n\nCatatan: peringkat top-k diambil dari SHAP fold yang sama (sedikit bias optimistis untuk "
          "varian hanya_topk); ini justru memperketat uji shortcut."]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2, 1]})
        t = imp.head(20).iloc[::-1]
        ax[0].barh(t["fitur"], t["mean_abs_shap"], color=["#c0392b" if r > 0 else "#2980b9" for r in t["arah_rho"]])
        ax[0].set_xlabel("mean |SHAP| (merah: nilai tinggi -> suspicious)")
        ax[0].tick_params(axis="y", labelsize=7)
        ax[1].barh(blk_share.index[::-1], blk_share.values[::-1], color="#7f8c8d")
        ax[1].set_xlabel("porsi kontribusi SHAP")
        fig.tight_layout()
        fig.savefig(out / "shap_summary.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        print("plot dilewati:", e)


if __name__ == "__main__":
    main()
