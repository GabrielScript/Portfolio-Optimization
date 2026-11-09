"""
qml_12_3_avaliar.py - the QML arm on the user's fixed 12 BR + 3 US BDR book (test branch only).

The B3 lake's price matrix has no BDRs, so this script merges the app's daily BDR closes (the
``ativos_eua`` list, suffix 34) into the point-in-time engine and registers them in the monthly
universe from their first quote. The BR sleeve is the Composite's 12 names (its own rule, 4 per
sector); the US sleeve is the 3 BDRs of highest 12-1 momentum, exactly the 12/3 rule of DECISOES
298-299. The engine starts in 2011 so the BDRs have a 252-session feature window.

Arms: the 12/3 equal-weight book, the QML-weighted 12/3 book (half-Kelly and max Sharpe), and the
Composite on the same dates. Descriptive, not a pre-registered reading.

    .venv-qpo/Scripts/python -X utf8 scripts/qml_12_3_avaliar.py [--fase 0]
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from ativos_eua import ATIVOS_EUA  # noqa: E402
from core.backtest import ResultadoBacktest  # noqa: E402
from core.cdi_lago import daily_cdi  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.config import DIAS_UTEIS_ANO  # noqa: E402
from core.precos_do_app import ler_fechamentos  # noqa: E402
from core.qml import QMLConfig, QMLPredictor, qml_strategy  # noqa: E402
from core.qpo import CFG_MAX_SHARPE  # noqa: E402
from core.sinais import POR_SETOR, carteira_da_nota, estrategia_conservadora  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, qubo_universo_avaliar  # noqa: E402
from scripts.avaliar_robustez import _acoes, referencias  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT, CFG_KELLY  # noqa: E402
from scripts.curvas_ao_vivo import ARQUIVOS, LAGO, congelar  # noqa: E402
from scripts.qml_avaliar import (ProbabilidadesCongeladas, _linha, _probabilidades,  # noqa: E402
                                 _rodar, _tabela)

SAIDA = RAIZ / "docs" / "qml-12-3-2026-11-09.txt"
CURVAS_CSV = RAIZ / "docs" / "dados" / "qml-12-3-comparacao.csv"
CACHE = RAIZ / "docs" / "dados" / "qml-probabilidades-12-3.parquet"
BDRS = tuple(ativo["ticker"] for ativo in ATIVOS_EUA if ativo["ticker"].endswith("34"))
SETORES_BDR = {ativo["ticker"]: ativo["setor"] for ativo in ATIVOS_EUA if ativo["ticker"].endswith("34")}
NOMES_BR, NOMES_EUA = 12, 3
CFG_12_3 = replace(CFG_BT, janela_treino=1_500)      # 252 extra sessions: the BDRs' feature window


def precos_das_bdrs(ate: date) -> pd.DataFrame:
    """Daily closes of the 30 US BDRs in the app's spliced files, renamed to the B3 ticker."""
    fechamentos = ler_fechamentos([f"{ticker}.SA" for ticker in BDRS], date(2009, 1, 1), ate)
    return fechamentos.rename(columns=lambda coluna: coluna.removesuffix(".SA"))


def universo_com_bdrs(universo: pd.DataFrame, precos: pd.DataFrame) -> pd.DataFrame:
    """The lake's monthly point-in-time universe plus every BDR from its first quote month on."""
    meses = pd.to_datetime(universo["month"])
    fim = meses.max()
    linhas = []
    for ticker in BDRS:
        if ticker not in precos.columns:
            continue
        primeira = precos[ticker].first_valid_index()
        if primeira is None:
            continue
        for mes in pd.date_range(primeira.to_period("M").to_timestamp(), fim, freq="MS"):
            linhas.append({"month": mes, "security": ticker, "issuer": ticker[:4]})
    return pd.concat([universo, pd.DataFrame(linhas)], ignore_index=True)


def seletor_12_3(nota, setores, bdrs: Sequence[str] = BDRS,
                 nomes_br: int = NOMES_BR, nomes_eua: int = NOMES_EUA, por_setor: int = POR_SETOR):
    """The Composite's 12 BR names plus the 3 BDRs of highest 12-1 momentum in the training window."""
    def _select(data: pd.Timestamp, treino: pd.DataFrame) -> list[str]:
        mes = data.to_period("M").to_timestamp()
        b3 = [papel for papel in treino.columns if papel not in bdrs]
        br = list(carteira_da_nota(nota(mes, b3).dropna(), setores, nomes_br, por_setor).index)
        janela = treino[[papel for papel in bdrs if papel in treino.columns]].iloc[-252:-21]
        momentum = (1.0 + janela).prod(min_count=231) - 1.0
        eua = list(momentum.dropna().nlargest(nomes_eua).index)
        return br + eua
    return _select


def estrategia_pesos_iguais(selector):
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        nomes = selector(treino.index[-1], treino)
        if len(nomes) < 2:
            raise ValueError(f"{treino.index[-1]:%Y-%m-%d}: fewer than 2 names")
        return pd.Series(1.0 / len(nomes), index=nomes)
    return _estrategia


def main(argv: list[str]) -> int:
    inicio = time.perf_counter()
    if "--fase" in argv and argv.index("--fase") + 1 == len(argv):
        raise ValueError("--fase needs a value, e.g. --fase 0")
    fase = int(argv[argv.index("--fase") + 1]) if "--fase" in argv else 0
    with tempfile.TemporaryDirectory(prefix="qml-12-3-") as temporaria:
        entradas = Path(temporaria)
        somas = congelar(LAGO, entradas, ARQUIVOS)
        avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = entradas
        qubo_universo_avaliar.FOLDER = qubo_universo_avaliar.PASTA_CVM = entradas
        dados = avaliar_robustez.carregar()
        mensal = avaliar_conservadora.carregar_mensal(entradas)
        nota, _ = qubo_universo_avaliar._value_score(mensal, com_crescimento=True)
        ultima = dados.precos.index.max()
        cdi = daily_cdi(ANOS_DE_CDI, ultima.date(), path=entradas / "cdi_daily.parquet")
        bdrs = precos_das_bdrs(ultima.date()).reindex(dados.precos.index)
        setores = {**dados.setores, **SETORES_BDR}
        universo = universo_com_bdrs(dados.universo, bdrs)
        precos12 = dados.precos.join(bdrs)
        retornos_qml = precos12.ffill(limit=5).pct_change().iloc[1:]
        acoes = _acoes(precos12).iloc[fase:]
        print(f"12/3: {len(BDRS)} BDRs merged; engine starts at {acoes.index[0]:%d/%m/%Y}", flush=True)
        predictor = QMLPredictor(retornos_qml, QMLConfig(), log=print)
        fonte = ProbabilidadesCongeladas(_probabilidades(predictor, acoes, CFG_12_3, CACHE))
        cfg_ms = replace(CFG_MAX_SHARPE, taxa_livre_risco=float((1 + cdi.iloc[-1]) ** DIAS_UTEIS_ANO - 1))
        seletor = seletor_12_3(nota, setores)
        curvas: dict[str, ResultadoBacktest] = {
            "Composite": _rodar(acoes, universo, estrategia_por_nota(nota, setores), "Composite", inicio, CFG_12_3),
            "Conservadora": _rodar(acoes, universo,
                                   estrategia_conservadora(mensal.retornos, mensal.proventos, setores),
                                   "Conservadora", inicio, CFG_12_3),
            "12/3 (pesos iguais)": _rodar(acoes, universo, estrategia_pesos_iguais(seletor),
                                          "12/3 (pesos iguais)", inicio, CFG_12_3),
            "QML-12/3 (Meio-Kelly)": _rodar(
                acoes, universo,
                qml_strategy(fonte, CFG_KELLY, setores, cdi, selector=seletor, tilt=1.0),
                "QML-12/3 (Meio-Kelly)", inicio, CFG_12_3),
            "QML-12/3 (Máx. Sharpe)": _rodar(
                acoes, universo,
                qml_strategy(fonte, cfg_ms, setores, cdi, selector=seletor, tilt=1.0),
                "QML-12/3 (Máx. Sharpe)", inicio, CFG_12_3),
        }
        bova = referencias(precos12, CFG_12_3)["BOVA11"]
        valores = {nome: curva.valores for nome, curva in curvas.items()}
        valores["BOVA11"] = bova.valores
        rf = cdi.reindex(next(iter(valores.values())).index).fillna(0.0)
        tabela = _tabela(curvas, bova.valores, rf, curvas["Composite"].valores)
        tabela.loc["BOVA11"] = _linha(bova.valores, bova.valores, rf)
        tabela.insert(0, "Fase", fase)
        with SAIDA.open("w", encoding="utf-8", newline="\n") as saida, contextlib.redirect_stdout(saida):
            print(f"12/3 QML arm (descriptive, test branch). Lake exports sha256: {somas}")
            print(f"BDRs: {len(BDRS)} from the app's closes; the US sleeve is the 3 of highest 12-1 momentum. "
                  f"Engine start {acoes.index[0]:%d/%m/%Y} (252 sessions of BDR history).")
            print(tabela.to_string(float_format=lambda x: f"{x:.4f}"))
        tabela.to_csv(CURVAS_CSV, encoding="utf-8", lineterminator="\n")
        print(tabela.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n{SAIDA.relative_to(RAIZ)} and {CURVAS_CSV.relative_to(RAIZ)} written "
          f"({time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
