"""Data tiruan berformat HI-Small_Trans.csv (uji pipeline saja). Ada pola layering:
sumber ilegal (format Bitcoin) -> mule1 -> mule2 -> target; seluruh rantai berlabel 1."""
import sys, numpy as np, pandas as pd
from pathlib import Path
rng = np.random.default_rng(0)
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
N_ACC = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
N_TX = int(sys.argv[3]) if len(sys.argv) > 3 else 120000
t0 = pd.Timestamp("2022-09-01")
fmts = ["ACH", "Cheque", "Credit Card", "Wire", "Cash", "Reinvestment", "Bitcoin"]
curs = ["US Dollar", "Euro", "Yuan", "Bitcoin"]
acc = [(f"{rng.integers(1, 50):03d}", f"{i:08X}") for i in range(N_ACC)]
rows = []
for _ in range(N_TX):
    a, b = rng.integers(0, N_ACC, 2)
    ts = t0 + pd.Timedelta(minutes=int(rng.integers(0, 10 * 1440)))
    amt = float(np.round(rng.lognormal(7, 1.5), 2)); c = curs[rng.integers(0, 3)]
    rows.append((ts, *acc[a], *acc[b], amt, c, amt, c, fmts[rng.integers(0, 6)], 0))
for _ in range(600):  # rantai layering
    s, m1, m2, tg = rng.integers(0, N_ACC, 4)
    ts = t0 + pd.Timedelta(minutes=int(rng.integers(0, 9 * 1440)))
    amt = float(np.round(rng.lognormal(9, 0.5), 2))
    rows.append((ts, *acc[s], *acc[m1], amt, "Bitcoin", amt, "Bitcoin", "Bitcoin", 1))
    ts2 = ts + pd.Timedelta(minutes=int(rng.integers(30, 600)))
    rows.append((ts2, *acc[m1], *acc[m2], amt * 0.97, "US Dollar", amt * 0.97, "US Dollar", "Cash", 1))
    ts3 = ts2 + pd.Timedelta(minutes=int(rng.integers(30, 600)))
    rows.append((ts3, *acc[m2], *acc[tg], amt * 0.95, "US Dollar", amt * 0.95, "US Dollar", "ACH", 1))
df = pd.DataFrame(rows).sort_values(0)
df[0] = df[0].dt.strftime("%Y/%m/%d %H:%M")
hdr = "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering"
with open(out / "HI-Small_Trans.csv", "w") as f:
    f.write(hdr + "\n"); df.to_csv(f, header=False, index=False)
print("toy ->", out / "HI-Small_Trans.csv", len(df), "baris; laundering", int(df[10].sum()))
