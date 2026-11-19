"""Data tiruan kecil berskema Elliptic2 ASLI (uji pipeline saja, hasil tidak bermakna).
Skema: background_nodes(clId, feat#1..43), background_edges(clId1, clId2, txId, feat#1..95),
nodes(clId, ccId), edges(clId1, clId2, txId), connected_components(ccId, ccLabel)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
N, E, S = 20000, 80000, 3000
F = [f"feat#{i}" for i in range(1, 96)]
bn = pd.DataFrame(rng.integers(0, 100, (N, 43)), columns=[f"feat#{i}" for i in range(1, 44)])
bn.insert(0, "clId", np.arange(N) * 7 + 11)
cid = bn.clId.values
be_parts = [pd.DataFrame({"clId1": cid[rng.integers(0, N, E)], "clId2": cid[rng.integers(0, N, E)]})
            .assign(**{f: rng.integers(0, 100, E) for f in F})]
lab = rng.random(S) < 0.05
rows_n, rows_e, perm, k = [], [], rng.permutation(N), 0
for s in range(S):
    m = rng.integers(2, 6); ids = perm[k:k + m]; k += m
    if lab[s]: bn.loc[ids, "feat#3"] += 20
    rows_n += [(cid[i], s) for i in ids]
    for j in range(m - 1):
        f = rng.integers(0, 100, 95); f[5] += 40 if lab[s] else 0
        rows_e.append((cid[ids[j]], cid[ids[j + 1]]))
        be_parts.append(pd.DataFrame([[cid[ids[j]], cid[ids[j + 1]], *f]], columns=["clId1", "clId2"] + F))
    if lab[s]:
        extra = cid[rng.integers(0, N, 15)]
        be_parts.append(pd.DataFrame({"clId1": cid[ids[0]], "clId2": extra}).assign(**{f: rng.integers(50, 100, 15) for f in F}))
be = pd.concat(be_parts, ignore_index=True)
be.insert(2, "txId", rng.permutation(len(be)) + 1000)
key = be.set_index(["clId1", "clId2"])["txId"]
key = key[~key.index.duplicated()]
ed = pd.DataFrame(rows_e, columns=["clId1", "clId2"]); ed["txId"] = key.loc[list(zip(ed.clId1, ed.clId2))].values
bn.to_csv(out / "background_nodes.csv", index=False); be.to_csv(out / "background_edges.csv", index=False)
pd.DataFrame(rows_n, columns=["clId", "ccId"]).to_csv(out / "nodes.csv", index=False)
ed.to_csv(out / "edges.csv", index=False)
pd.DataFrame({"ccId": range(S), "ccLabel": np.where(lab, "suspicious", "licit")}).to_csv(out / "connected_components.csv", index=False)
print("toy ->", out, "suspicious:", lab.sum())
