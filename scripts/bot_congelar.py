"""
Aleph Bot, stage 0 (docs/superpowers/specs/2026-11-12-aleph-bot-design.md): the universe by month and the bars of
every name ever in it, downloaded from XP's MetaTrader 5 demo terminal (open and logged in) and frozen.

    python -X utf8 scripts/com_python_base.py scripts/bot_congelar.py

Bars go to ``D:/aleph_bot/bars/<M15|M5>/<symbol>.parquet`` (too large for the repo); the repo keeps the universe
and the manifest with each file's sha256 (``docs/dados/2026-11-12-aleph-bot/``). No return is computed here.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from core.bot import universo  # noqa: E402
from core.mt5_corretora import TERMINAL_PATH  # noqa: E402

LAGO = Path("D:/b3-datalake/exports")
BARRAS = Path("D:/aleph_bot/bars")
PASTA = RAIZ / "docs" / "dados" / "2026-11-12-aleph-bot"
INICIO = pd.Timestamp("2021-11-01")
MAXIMO = 99_999                     # the terminal serves at most 100,000 bars a request


def universe_table() -> pd.DataFrame:
    painel = pd.read_parquet(LAGO / "daily_liquid_panel.parquet", columns=["date", "security", "volume_brl"],
                             filters=[("date", ">=", (INICIO - pd.DateOffset(months=6)).date())])
    painel["date"] = pd.to_datetime(painel["date"])
    volume = painel.pivot(index="date", columns="security", values="volume_brl").fillna(0.0)   # no trade = no volume
    casa = pd.read_parquet(LAGO / "universe_monthly.parquet")
    meses = pd.date_range(INICIO, volume.index.max(), freq="MS")
    return pd.DataFrame([{"month": mes, "security": nome} for mes in meses
                         for nome in universo.of_month(volume, casa, mes)])


def main() -> int:
    sys.path.insert(0, str(RAIZ / ".venv-mt5"))
    sys.path.insert(0, str(RAIZ.parent / "Portfolio-Optimization" / ".venv-mt5"))
    import MetaTrader5 as mt5
    tabela = universe_table()
    PASTA.mkdir(parents=True, exist_ok=True)
    tabela.to_parquet(PASTA / "universe.parquet", index=False)
    nomes = sorted(tabela["security"].unique())
    print(f"universe: {tabela['month'].nunique()} months, {len(nomes)} names ever, "
          f"{tabela.groupby('month').size().min()} to {tabela.groupby('month').size().max()} a month", flush=True)
    if not mt5.initialize(path=TERMINAL_PATH, timeout=60_000):
        raise SystemExit(f"terminal refused: {mt5.last_error()}")
    linhas = []
    try:
        for posicao, nome in enumerate(nomes, start=1):
            mt5.symbol_select(nome, True)
            for rotulo, periodo in (("M15", mt5.TIMEFRAME_M15), ("M5", mt5.TIMEFRAME_M5)):
                barras = mt5.copy_rates_from_pos(nome, periodo, 0, MAXIMO)
                if barras is None or len(barras) == 0:
                    linhas.append({"symbol": nome, "timeframe": rotulo, "bars": 0, "first": "", "last": "", "sha256": ""})
                    continue
                quadro = pd.DataFrame(barras)
                quadro["time"] = pd.to_datetime(quadro["time"], unit="s")       # the terminal's wall clock (Brasília)
                arquivo = BARRAS / rotulo / f"{nome}.parquet"
                arquivo.parent.mkdir(parents=True, exist_ok=True)
                quadro.to_parquet(arquivo, index=False)
                linhas.append({"symbol": nome, "timeframe": rotulo, "bars": len(quadro),
                               "first": str(quadro["time"].iloc[0]), "last": str(quadro["time"].iloc[-1]),
                               "sha256": hashlib.sha256(arquivo.read_bytes()).hexdigest()})
            print(f"[{posicao}/{len(nomes)}] {nome}: {linhas[-2]['bars']} M15, {linhas[-1]['bars']} M5", flush=True)
    finally:
        mt5.shutdown()
    pd.DataFrame(linhas).to_csv(PASTA / "manifest.csv", index=False, lineterminator="\n")
    sem = sorted({linha["symbol"] for linha in linhas if linha["timeframe"] == "M15" and linha["bars"] == 0})
    print(f"names without bars on the terminal (survivorship): {len(sem)} {sem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
