"""
Aleph Bot, stage 1 (docs/superpowers/specs/2026-11-12-aleph-bot-design.md and its two amendments, committed before
any trade return): the two intraday event studies on the frozen 15-minute bars.

    python -X utf8 scripts/bot_estagio1.py estudo  > docs/aleph-bot-estagio1-estudo-2026-11-12.txt
    python -X utf8 scripts/bot_estagio1.py holdout > docs/aleph-bot-estagio1-holdout-2026-11-12.txt

``estudo`` reads bars up to 2024-12-31 only. ``holdout`` (2025-01-01 on) runs only the patterns the study passed
(``docs/dados/2026-11-12-aleph-bot/estagio1-estudo.csv``) and is read once.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from core.bot import eventos as ev  # noqa: E402

BARRAS = Path("D:/aleph_bot/bars/M15")
SAIDA = Path("D:/aleph_bot/stage1")
PASTA = RAIZ / "docs" / "dados" / "2026-11-12-aleph-bot"
LAGO = Path("D:/b3-datalake")
SETORES = RAIZ / "docs" / "dados" / "2026-10-22" / "sectors.parquet"
FIM_DO_ESTUDO = pd.Timestamp("2024-12-31 23:59")
MERCADO = "BOVA11"
MAX_DIFF = 0.005
MIN_TRADES, T_ESTUDO, T_HOLDOUT = 300, 3.0, 2.0
PADROES = {"overreaction": ev.overreaction, "opening_gap": ev.opening_gap}


def filing_days() -> dict[str, set[pd.Timestamp]]:
    """Issuer -> the days with a CVM filing (the lake's IPE table has the delivery date, no time of day)."""
    ipe = pd.read_parquet(LAGO / "sentiment" / "ipe" / "ipe.parquet", columns=["codigo_cvm", "data_entrega"])
    setores = pd.read_parquet(SETORES).dropna(subset=["code_cvm"]).drop_duplicates("code_cvm")
    emissor = setores.set_index(setores["code_cvm"].astype(int))["issuer"]
    ipe = ipe.assign(issuer=ipe["codigo_cvm"].map(emissor), day=pd.to_datetime(ipe["data_entrega"], errors="coerce"))
    ipe = ipe.dropna(subset=["issuer", "day"])
    return {nome: set(grupo) for nome, grupo in ipe.groupby("issuer")["day"]}


def lake_returns() -> pd.DataFrame:
    painel = pd.read_parquet(LAGO / "exports" / "daily_liquid_panel.parquet", columns=["date", "security", "adj_close"],
                             filters=[("date", ">=", pd.Timestamp("2021-10-01").date())])
    painel["date"] = pd.to_datetime(painel["date"])
    return painel.pivot(index="date", columns="security", values="adj_close").pct_change(fill_method=None)


def excluded_days(nome: str, fechamentos: pd.Series, do_lago: pd.Series | None, no_universo: set[pd.Period],
                  com_documento: set[pd.Timestamp], gap: bool) -> set[pd.Timestamp]:
    dias = fechamentos.index
    fora = {dia for dia in dias if dia.to_period("M") not in no_universo}
    seguinte = pd.Series(dias[1:], index=dias[:-1])
    if do_lago is not None:
        diferenca = (fechamentos.pct_change() - do_lago.reindex(dias)).abs()
        ruins = set(diferenca.index[diferenca > MAX_DIFF])
        fora |= ruins | {seguinte[dia] for dia in ruins if dia in seguinte.index}
    fora |= com_documento & set(dias)
    if gap:                                   # a filing dated the previous session also excludes the gap
        fora |= {seguinte[dia] for dia in com_documento if dia in seguinte.index}
    return fora


def summary(trades: pd.DataFrame) -> dict:
    t, media, dias = ev.clustered_t(trades)
    return {"trades": len(trades), "days": dias, "mean gross bps": trades["gross"].mean() * 1e4 if len(trades) else float("nan"),
            "mean net bps": media * 1e4, "t": t, "win rate": float((trades["net"] > 0).mean()) if len(trades) else float("nan")}


def main(periodo: str) -> int:
    inicio = time.perf_counter()
    estudo = periodo == "estudo"
    if estudo:
        padroes = dict(PADROES)
    else:
        aprovados = pd.read_csv(PASTA / "estagio1-estudo.csv").query("passed")["pattern"].tolist()
        padroes = {nome: PADROES[nome] for nome in aprovados}
        if not padroes:
            raise SystemExit("no pattern passed the study period: the holdout is not read")
    universo = pd.read_parquet(PASTA / "universe.parquet")
    meses = universo.assign(m=pd.to_datetime(universo["month"]).dt.to_period("M")).groupby("security")["m"].apply(set)
    documentos, retornos_do_lago = filing_days(), lake_returns()

    def ler(nome: str) -> pd.DataFrame:
        barras = pd.read_parquet(BARRAS / f"{nome}.parquet", columns=["time", "open", "close", "spread"])
        return barras[barras["time"] <= FIM_DO_ESTUDO] if estudo else barras

    mercado = ler(MERCADO)
    fech_mercado = ev.daily_closes(mercado)
    nomes = sorted(arquivo.stem for arquivo in BARRAS.glob("*.parquet") if arquivo.stem in meses.index)
    resultado = {nome: {1.0: [], 2.0: []} for nome in padroes}
    for nome in nomes:
        barras = ler(nome)
        if barras.empty:
            continue
        fechamentos = ev.daily_closes(barras)
        beta = ev.rolling_beta(fechamentos, fech_mercado)
        do_lago = retornos_do_lago[nome] if nome in retornos_do_lago else None
        com_documento = documentos.get(nome[:4], set())      # the issuer is the ticker's first four letters
        for padrao, funcao in padroes.items():
            fora = excluded_days(nome, fechamentos, do_lago, meses[nome], com_documento, gap=padrao == "opening_gap")
            for multiplo in (1.0, 2.0):
                trades = funcao(barras, mercado, beta, excluded=fora, cost_multiple=multiplo).assign(name=nome)
                resultado[padrao][multiplo].append(trades)
    print(f"Aleph Bot stage 1, {periodo}: {len(nomes)} names with bars, market {MERCADO}; "
          f"{'bars up to 2024-12-31' if estudo else 'trades from 2025-01-01'}")
    linhas = []
    SAIDA.mkdir(parents=True, exist_ok=True)
    for padrao in padroes:
        tabelas = {m: pd.concat(resultado[padrao][m], ignore_index=True) for m in (1.0, 2.0)}
        if not estudo:
            tabelas = {m: t[t["day"] > FIM_DO_ESTUDO] for m, t in tabelas.items()}
        base, dobro = summary(tabelas[1.0]), summary(tabelas[2.0])
        tabelas[1.0].to_parquet(SAIDA / f"{padrao}-{periodo}.parquet", index=False)
        limite = T_ESTUDO if estudo else T_HOLDOUT
        passou = bool(base["mean net bps"] > 0 and base["t"] >= limite and (base["trades"] >= MIN_TRADES or not estudo))
        print(f"\n=== {padrao} ===")
        print(pd.DataFrame({"cost x1": base, "cost x2": dobro}).T.to_string(float_format=lambda x: f"{x:.2f}"))
        por_ano = tabelas[1.0].groupby(tabelas[1.0]["day"].dt.year)["net"].agg(["size", "mean"])
        print("by year (trades, mean net bps):", {int(a): (int(l["size"]), round(l["mean"] * 1e4, 1)) for a, l in por_ano.iterrows()})
        print(f"names traded: {tabelas[1.0]['name'].nunique()}; gate (mean net > 0, t >= {limite}"
              f"{', trades >= ' + str(MIN_TRADES) if estudo else ''}): {'PASSED' if passou else 'fails'}")
        linhas.append({"pattern": padrao, "period": periodo, **base, "t at cost x2": dobro["t"], "passed": passou})
    pd.DataFrame(linhas).to_csv(PASTA / f"estagio1-{periodo}.csv", index=False, lineterminator="\n")
    print(f"\n(total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("estudo", "holdout"):
        raise SystemExit("usage: bot_estagio1.py [estudo | holdout]")
    raise SystemExit(main(sys.argv[1]))
