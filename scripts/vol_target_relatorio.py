"""The volatility-target overlay measured on the live Composite curve, with the CDI from the lake.

    python -m scripts.vol_target_relatorio

It reads the saved live curves (fase 0 — the current simulation), takes the CDI the cash leg would actually earn from the
lake, and prints the plain Composite beside the overlay at three targets. This is the screening number; it assumes the
exposure can be adjusted every session, which production cannot do without selling part of every position, so the
engine run with real rebalancing and tax is still owed before anything changes in the app."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from core import cdi_lago, vol_target

CURVAS = Path("docs/dados/curvas-ao-vivo.parquet")
ALVOS = (0.16, 0.20, 0.24)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curvas", type=Path, default=CURVAS)
    p.add_argument("--janela", type=int, default=21)
    p.add_argument("--custo-bps", type=float, default=10.0)
    args = p.parse_args(argv)

    curvas = pd.read_parquet(args.curvas)
    atual = curvas[curvas["fase"] == 0].set_index("date").sort_index()
    serie = atual["Composto"].astype(float).dropna()
    retornos = serie.pct_change().dropna()

    try:
        cdi = cdi_lago.daily_cdi(17, date.today())
    except Exception as erro:                     # noqa: BLE001 - the report still runs without the cash rate
        print(f"CDI unavailable ({type(erro).__name__}: {erro}); the cash leg earns nothing in this run.", flush=True)
        cdi = pd.Series(0.0, index=retornos.index)

    base = vol_target.medir(retornos)
    print(f"{'Composite base':22s} CAGR {base['retorno_anual']:7.2%}  Sharpe {base['sharpe']:5.2f}  "
          f"MDD {base['mdd']:7.1%}", flush=True)
    for alvo in ALVOS:
        com = vol_target.aplicar(retornos, cdi, alvo=alvo, janela=args.janela, custo_bps=args.custo_bps)
        m = vol_target.medir(com)
        e = vol_target.exposicao(retornos, alvo, args.janela)
        print(f"alvo {alvo:4.0%} jan{args.janela:<3d}  CAGR {m['retorno_anual']:7.2%}  Sharpe {m['sharpe']:5.2f}  "
              f"MDD {m['mdd']:7.1%}  exposicao media {e.mean():.0%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
