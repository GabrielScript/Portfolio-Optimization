import json, sys
from pathlib import Path
import numpy as np, pandas as pd

D = Path(sys.argv[1])
rot = json.loads((D / "rotulos.json").read_text(encoding="utf-8"))
r = pd.read_json(D / "respostas.jsonl", lines=True)
print("errors:", int(r["erro"].notna().sum()) if "erro" in r else 0)
r = r[r["p"].notna()].copy()
r["y"] = r["id"].map(rot["material"]); r["claro"] = ~r["id"].isin(rot["ambiguo"])

def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))

def medidas(g):
    y, p = g["y"].to_numpy(), np.clip(g["p"].to_numpy(), 1e-6, 1 - 1e-6)
    return pd.Series({"n": len(g), "acerto@0.5": np.mean((p >= 0.5) == y), "Brier": np.mean((p - y) ** 2),
                      "log loss": -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)), "AUC": auc(y, p),
                      "falsos +": int(((p >= 0.5) & (y == 0)).sum()), "falsos -": int(((p < 0.5) & (y == 1)).sum()),
                      "p distintos": g["p"].round(3).nunique()})

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print("\n== all 60 =="); print(r.groupby("juiz").apply(medidas))
print("\n== clear only =="); print(r[r["claro"]].groupby("juiz").apply(medidas))
lat = r.groupby("juiz").agg(mediana_s=("s", "median"), p90_s=("s", lambda s: s.quantile(0.9)), media_s=("s", "mean"),
                            tokens=("tokens", "mean"), custo_por_1000=("custo", lambda c: 1000 * c.mean()))
print("\n== latency and cost =="); print(lat)
# calibration by bins, all items
r["faixa"] = pd.cut(r["p"], [0, 0.2, 0.5, 0.8, 1.0], include_lowest=True)
print("\n== calibration (share material by bin) ==")
print(r.groupby(["juiz", "faixa"], observed=True).agg(n=("y", "size"), p_media=("p", "mean"), real=("y", "mean")).to_string())
print("\nmodels:", r.groupby("juiz")["modelo"].first().to_dict())
