from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from core.sentinela import conta

DIAS = pd.bdate_range("2026-12-01", periods=30)
NOMES = ["AAAA3", "BBBB3", "CCCC3"]


def _precos() -> pd.DataFrame:
    passos = np.linspace(0, 0.1, len(DIAS))
    return pd.DataFrame({"AAAA3": 10 * (1 + passos), "BBBB3": 20 * (1 - passos), "CCCC3": np.full(len(DIAS), 30.0)},
                        index=DIAS)


def _registro(*datas: str) -> pd.DataFrame:
    return pd.DataFrame([{"data": pd.Timestamp(data), "ticker": nome, "peso": 0.3}
                         for data in datas for nome in NOMES])


CDI = pd.Series(0.0, index=DIAS)


def _custo(ticker: str, valor: float) -> float:
    return 0.001 * abs(valor)


def _acao(dia: int, ticker: str = "BBBB3", acao: str = "vender", confianca: float = 0.8, hora: int = 10,
          ident: str = "a1") -> dict:
    return {"id": ident, "decidida_em": datetime.combine(DIAS[dia].date(), datetime.min.time()).replace(hour=hora),
            "ticker": ticker, "acao": acao, "confianca": confianca}


def test_a_decision_before_17h_on_a_session_executes_at_its_close_else_at_the_next_session():
    assert conta.data_de_execucao(datetime(2026, 12, 2, 16, 59), DIAS) == pd.Timestamp("2026-12-02")
    assert conta.data_de_execucao(datetime(2026, 12, 2, 17, 0), DIAS) == pd.Timestamp("2026-12-03")
    assert conta.data_de_execucao(datetime(2026, 12, 5, 10), DIAS) == pd.Timestamp("2026-12-07")      # a Saturday
    assert conta.data_de_execucao(datetime(2027, 3, 1, 10), DIAS) is None


def test_only_actions_at_or_above_70_percent_are_accepted():
    assert conta.recusa(_acao(1, confianca=0.69), [], pd.Timestamp("2026-12-01")) is not None
    assert conta.recusa(_acao(1, confianca=0.70), [], pd.Timestamp("2026-12-01")) is None
    assert conta.recusa(_acao(1, acao="manter"), [], pd.Timestamp("2026-12-01")) is not None


def test_one_action_per_name_a_day_only_vender_after_a_cut_and_nothing_after_a_sale_until_the_next_record():
    registro = pd.Timestamp("2026-12-01")
    corte = _acao(2, acao="reduzir_metade")
    assert conta.recusa(_acao(2, hora=11, ident="a2"), [corte], registro) is not None          # same day
    assert conta.recusa(_acao(3, acao="reduzir_metade", ident="a2"), [corte], registro) is not None
    assert conta.recusa(_acao(3, ident="a2"), [corte], registro) is None
    venda = _acao(3, ident="a2")
    assert conta.recusa(_acao(5, ident="a3"), [corte, venda], registro) is not None
    assert conta.recusa(_acao(5, ident="a3"), [corte, venda], pd.Timestamp(DIAS[4])) is None   # a new record
    assert conta.recusa(_acao(2, ticker="AAAA3", ident="a4"), [corte], registro) is None       # another name


def test_base_and_agent_accounts_start_the_same_day_with_the_same_record_and_capital():
    ordens, feitas = conta.ordens_devidas(conta.vazio(), _registro("2026-12-01"), [], _precos(), CDI, _custo,
                                          inicio=DIAS[0])
    assert feitas == []
    base = ordens[ordens["carteira"] == conta.BASE].drop(columns="carteira").reset_index(drop=True)
    agente = ordens[ordens["carteira"] == conta.CONTA].drop(columns="carteira").reset_index(drop=True)
    pd.testing.assert_frame_equal(base, agente)
    curvas = conta.curvas(ordens, _precos(), CDI)
    assert curvas[conta.BASE].iloc[0] == pytest.approx(conta.CAPITAL, rel=0.01)
    pd.testing.assert_series_equal(curvas[conta.BASE], curvas[conta.CONTA], check_names=False)


def test_a_sale_executes_at_its_close_moves_the_name_to_cash_and_pays_its_cost():
    precos = _precos()
    ordens, _ = conta.ordens_devidas(conta.vazio(), _registro("2026-12-01"), [], precos.loc[:DIAS[4]], CDI, _custo,
                                     inicio=DIAS[0])
    acao = _acao(5) | {"executa_em": DIAS[5]}
    novas, feitas = conta.ordens_devidas(ordens, _registro("2026-12-01"), [acao], precos.loc[:DIAS[9]], CDI, _custo,
                                         inicio=DIAS[0])
    assert feitas == ["a1"]
    venda = novas[(novas["carteira"] == conta.CONTA) & (novas["data"] == DIAS[5])].set_index("ticker")
    tinha = ordens[(ordens["carteira"] == conta.CONTA) & (ordens["ticker"] == "BBBB3")]["ordem"].sum()
    assert venda.loc["BBBB3", "ordem"] == -tinha and venda.loc["AAAA3", "ordem"] == 0
    assert venda.loc["BBBB3", "preco"] == pytest.approx(precos.loc[DIAS[5], "BBBB3"])
    assert venda.loc["BBBB3", "custo"] == pytest.approx(0.001 * tinha * precos.loc[DIAS[5], "BBBB3"])
    curvas = conta.curvas(pd.concat([ordens, novas]), precos.loc[:DIAS[9]], CDI)
    assert curvas[conta.CONTA].iloc[-1] > curvas[conta.BASE].iloc[-1]            # BBBB3 kept falling after the sale
    assert curvas[conta.CONTA].loc[DIAS[5]] == pytest.approx(
        curvas[conta.BASE].loc[DIAS[5]] - venda.loc["BBBB3", "custo"])


def test_a_cut_sells_half_the_shares_rounded_down():
    precos = _precos()
    ordens, _ = conta.ordens_devidas(conta.vazio(), _registro("2026-12-01"), [], precos.loc[:DIAS[1]], CDI, _custo,
                                     inicio=DIAS[0])
    tinha = ordens[(ordens["carteira"] == conta.CONTA) & (ordens["ticker"] == "BBBB3")]["ordem"].sum()
    acao = _acao(2, acao="reduzir_metade") | {"executa_em": DIAS[2]}
    novas, _ = conta.ordens_devidas(ordens, _registro("2026-12-01"), [acao], precos.loc[:DIAS[2]], CDI, _custo,
                                    inicio=DIAS[0])
    assert novas.set_index("ticker").loc["BBBB3", "ordem"] == -(tinha // 2)


def test_an_action_not_yet_in_the_price_files_waits():
    precos = _precos()
    ordens, _ = conta.ordens_devidas(conta.vazio(), _registro("2026-12-01"), [], precos.loc[:DIAS[1]], CDI, _custo,
                                     inicio=DIAS[0])
    novas, feitas = conta.ordens_devidas(ordens, _registro("2026-12-01"), [_acao(3) | {"executa_em": DIAS[3]}],
                                         precos.loc[:DIAS[2]], CDI, _custo, inicio=DIAS[0])
    assert feitas == [] and novas.empty


def test_the_next_record_returns_the_account_to_the_composite_weights():
    precos = _precos()
    ordens, _ = conta.ordens_devidas(conta.vazio(), _registro("2026-12-01"), [_acao(2) | {"executa_em": DIAS[2]}],
                                     precos.loc[:DIAS[5]], CDI, _custo, inicio=DIAS[0])
    assert ordens[(ordens["carteira"] == conta.CONTA) & (ordens["ticker"] == "BBBB3")]["ordem"].sum() == 0
    novas, _ = conta.ordens_devidas(ordens, _registro("2026-12-01", "2026-12-10"), [], precos, CDI, _custo,
                                    inicio=DIAS[0])
    compra = novas[(novas["carteira"] == conta.CONTA) & (novas["data"] == pd.Timestamp("2026-12-10"))]
    assert compra.set_index("ticker").loc["BBBB3", "ordem"] > 0


def test_a_sale_on_a_record_day_is_merged_into_one_row_per_name():
    precos = _precos()
    ordens, _ = conta.ordens_devidas(conta.vazio(), _registro("2026-12-01"), [], precos.loc[:DIAS[3]], CDI, _custo,
                                     inicio=DIAS[0])
    dia = pd.Timestamp("2026-12-10")
    novas, feitas = conta.ordens_devidas(ordens, _registro("2026-12-01", "2026-12-10"),
                                         [_acao(7) | {"executa_em": dia}], precos.loc[:dia], CDI, _custo,
                                         inicio=DIAS[0])
    do_dia = novas[(novas["carteira"] == conta.CONTA) & (novas["data"] == dia)]
    assert feitas == ["a1"] and not do_dia["ticker"].duplicated().any()
    tudo = pd.concat([ordens, novas])
    assert tudo[(tudo["carteira"] == conta.CONTA) & (tudo["ticker"] == "BBBB3")]["ordem"].sum() == 0


def test_ledger_round_trip(tmp_path):
    arquivo = tmp_path / "ordens.csv"
    ordens, _ = conta.ordens_devidas(conta.vazio(), _registro("2026-12-01"), [], _precos(), CDI, _custo,
                                     inicio=DIAS[0])
    conta.acrescentar(arquivo, ordens)
    lido = conta.ler(arquivo)
    assert len(lido) == len(ordens) and lido["data"].dtype.kind == "M"
