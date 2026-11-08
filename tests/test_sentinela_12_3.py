"""The fictitious "12/3 + Sentinela" account (the user, 2026-11-08): the 12/3 record plus the agent's sales, in its own
ledger, beside the "Composto + Sentinela" account it must not touch."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from core import simulador
from core.sentinela import conta, registro, rodada

DIAS = pd.bdate_range("2026-11-09", periods=20)


def _precos() -> pd.DataFrame:
    passos = np.linspace(0, 0.1, len(DIAS))
    return pd.DataFrame({"AAAA3": 10 * (1 + passos), "BBBB3": 20 * (1 - passos), "MUTC34": 50 * (1 + 2 * passos)},
                        index=DIAS)


def _registro_12_3(*datas: str) -> pd.DataFrame:
    return pd.DataFrame([{"data": pd.Timestamp(data), "ticker": nome, "peso": peso, "base": pd.Timestamp("2026-10-27")}
                         for data in datas for nome, peso in (("AAAA3", 0.4), ("BBBB3", 0.4), ("MUTC34", 0.2))])


def _custo(ticker: str, valor: float) -> float:
    return 0.001 * abs(valor)


CDI = pd.Series(0.0, index=DIAS)


def _acao(dia, ticker: str, ident: str, acao: str = "vender") -> dict:
    decidida = datetime.combine(pd.Timestamp(dia).date(), datetime.min.time()).replace(hour=10)
    return {"id": ident, "decidida_em": decidida, "ticker": ticker, "acao": acao, "confianca": 0.9,
            "executa_em": pd.Timestamp(dia), "estado": "aceita", "motivo": None}


def test_the_ledger_can_carry_other_portfolio_names_and_the_default_ones_are_unchanged():
    ordens, _ = conta.ordens_devidas(conta.vazio(), _registro_12_3("2026-11-09"), [], _precos(), CDI, _custo,
                                     inicio=DIAS[0], base=conta.BASE_12_3, nome=conta.CONTA_12_3)
    assert set(ordens["carteira"]) == {conta.BASE_12_3, conta.CONTA_12_3}
    curvas = conta.curvas(ordens, _precos(), CDI, nomes=(conta.BASE_12_3, conta.CONTA_12_3))
    assert {conta.BASE_12_3, conta.CONTA_12_3} <= set(curvas.columns)
    padrao, _ = conta.ordens_devidas(conta.vazio(), _registro_12_3("2026-11-09"), [], _precos(), CDI, _custo,
                                     inicio=DIAS[0])
    assert set(padrao["carteira"]) == {conta.BASE, conta.CONTA}


def test_both_12_3_portfolios_pay_the_bdr_tax_like_the_12_3_ghost():
    assert {"12/3", conta.BASE_12_3, conta.CONTA_12_3} <= set(simulador.CARTEIRAS_COM_IR)


def _arquivos(tmp_path, acoes: list[dict]) -> rodada.Arquivos:
    arquivos = rodada.Arquivos(tmp_path)
    registro.acrescentar_acoes(arquivos.acoes, acoes)
    return arquivos


def test_the_12_3_account_takes_the_agents_sales_from_its_start_on_in_its_own_files(tmp_path):
    antes = _acao("2026-11-06", "BBBB3", "velha")                     # decided before the 12/3 account existed
    venda = _acao(DIAS[3], "BBBB3", "nova")
    fora = _acao(DIAS[4], "ZZZZ3", "outra")                           # a name the 12/3 does not hold
    arquivos = _arquivos(tmp_path, [antes, venda, fora])
    acoes_antes = arquivos.acoes.read_bytes()
    feitas = rodada.atualizar_conta_12_3(arquivos, _registro_12_3("2026-11-09"), _precos(), CDI, _custo, DIAS[0])
    assert sorted(feitas) == ["nova", "outra"]
    assert arquivos.acoes.read_bytes() == acoes_antes                 # the Composite account's state is untouched
    assert not arquivos.ordens.exists()
    ordens = conta.ler(arquivos.ordens_12_3)
    agente = ordens[ordens["carteira"] == conta.CONTA_12_3]
    assert agente.loc[(agente["data"] == DIAS[3]) & (agente["ticker"] == "BBBB3"), "ordem"].iloc[0] < 0
    assert (ordens.loc[ordens["carteira"] == conta.BASE_12_3, "ordem"] > 0).all()
    # a second run adds nothing and executes nothing again
    assert rodada.atualizar_conta_12_3(arquivos, _registro_12_3("2026-11-09"), _precos(), CDI, _custo, DIAS[0]) == []
    assert len(conta.ler(arquivos.ordens_12_3)) == len(ordens)


def test_an_action_executed_in_the_composite_account_still_executes_in_the_12_3_account(tmp_path):
    venda = _acao(DIAS[2], "AAAA3", "x1")
    executada = venda | {"estado": "executada"}
    arquivos = _arquivos(tmp_path, [venda, executada])
    assert rodada.atualizar_conta_12_3(arquivos, _registro_12_3("2026-11-09"), _precos(), CDI, _custo,
                                       DIAS[0]) == ["x1"]
