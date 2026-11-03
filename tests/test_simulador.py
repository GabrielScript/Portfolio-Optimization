"""The fictitious account (``core.simulador``): orders due from the paper records, curve, returns and positions."""

import pandas as pd
import pytest

from core import simulador as sim

INICIO = pd.Timestamp("2026-11-03")
SEM_CDI = pd.Series(dtype=float)
TIMEOUT_S = 60
CINCO = {"AAAA3": 0.2, "BBBB3": 0.2, "CCCC3": 0.2, "DDDD3": 0.2, "EEEE3": 0.2}      # the risk rule caps a stock at 20%
OUTROS = {"BBBB3": 0.2, "CCCC3": 0.2, "DDDD3": 0.2, "EEEE3": 0.2, "FFFF3": 0.2}


def tabela_de_precos(aaaa3: list[float], **outras) -> pd.DataFrame:
    """``aaaa3`` per session from the start; each other stock at 10.0 unless given."""
    dias = pd.bdate_range(INICIO, periods=len(aaaa3))
    papeis = sorted({*CINCO, *OUTROS} - {"AAAA3"})
    return pd.DataFrame({"AAAA3": aaaa3} | {papel: outras.get(papel, [10.0] * len(aaaa3)) for papel in papeis},
                        index=dias, dtype=float)


def sem_custo(_ticker: str, _valor: float) -> float:
    return 0.0


def registro(data, pesos=None) -> pd.DataFrame:
    pesos = CINCO if pesos is None else pesos
    return pd.DataFrame({"data": pd.Timestamp(data), "ticker": list(pesos), "peso": list(pesos.values()), "preco": 1.0})


def ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=sim.COLUNAS).astype({"data": "datetime64[ns]"})


def test_nothing_is_due_before_the_first_close_is_in_the_files():
    precos = pd.DataFrame({"AAAA3": [10.0]}, index=[pd.Timestamp("2026-10-30")])
    assert sim.ordens_devidas(ledger(), {"Composto": registro("2026-10-27")}, precos, SEM_CDI, sem_custo).empty


def test_the_first_buy_takes_the_last_record_at_the_first_close_in_whole_shares():
    precos = tabela_de_precos([10.0], BBBB3=[33.0])
    registros = {"Composto": pd.concat([registro("2026-08-03", OUTROS), registro("2026-10-27")])}
    novas = sim.ordens_devidas(ledger(), registros, precos, SEM_CDI, sem_custo)
    assert set(novas["data"]) == {INICIO}
    assert set(novas["carteira"]) == {"Composto"}
    assert set(novas["ticker"]) == set(CINCO)                    # the last record, not the older one
    compra = novas.set_index("ticker")
    assert compra.loc["BBBB3", "preco"] == 33.0
    assert compra.loc["BBBB3", "ordem"] == 54                    # 20% of R$ 9,000 less the plan's 10 bps, whole shares
    assert 8_900 < (compra["ordem"] * compra["preco"]).sum() <= 9_000


def test_a_trade_already_in_the_ledger_is_not_repeated():
    precos = tabela_de_precos([10.0, 11.0])
    registros = {"Composto": registro("2026-10-27")}
    primeira = sim.ordens_devidas(ledger(), registros, precos, SEM_CDI, sem_custo)
    assert sim.ordens_devidas(primeira, registros, precos, SEM_CDI, sem_custo).empty


def test_the_deepseek_portfolio_is_the_third_one_and_trades_at_the_first_session_on_or_after_its_record():
    # DECISOES 240: the opinion is formed after a close, so its record is dated the next calendar day and trades at the
    # first session from then on (a Saturday record trades on Monday)
    assert sim.CARTEIRAS == ("Composto", "Conservadora", "DeepSeek")
    precos = tabela_de_precos([10.0, 10.0, 10.0, 10.0, 10.0])          # Tue 03/11 to Mon 09/11
    registros = {"DeepSeek": registro("2026-11-07", {"AAAA3": 0.2, "BBBB3": 0.2})}
    novas = sim.ordens_devidas(ledger(), registros, precos, SEM_CDI, sem_custo)
    assert set(novas["data"]) == {pd.Timestamp("2026-11-09")} and set(novas["ticker"]) == {"AAAA3", "BBBB3"}
    assert sim.ordens_devidas(novas, registros, precos, SEM_CDI, sem_custo).empty


def test_a_record_with_only_zero_weights_sells_everything_into_cash():
    precos = tabela_de_precos([10.0, 10.0, 10.0])
    registros = {"DeepSeek": pd.concat([registro("2026-11-04", {"AAAA3": 0.2}), registro("2026-11-05", {"AAAA3": 0.0})])}
    novas = sim.ordens_devidas(ledger(), registros, precos, SEM_CDI, sem_custo)
    compra, venda = novas.groupby("data")["ordem"].sum()
    assert compra > 0 and venda == -compra


def test_each_portfolio_has_its_own_orders():
    precos = tabela_de_precos([10.0])
    registros = {"Composto": registro("2026-10-27"), "Conservadora": registro("2026-10-27", OUTROS)}
    novas = sim.ordens_devidas(ledger(), registros, precos, SEM_CDI, sem_custo)
    assert set(novas.loc[novas["carteira"] == "Composto", "ticker"]) == set(CINCO)
    assert set(novas.loc[novas["carteira"] == "Conservadora", "ticker"]) == set(OUTROS)


def test_a_later_record_trades_at_its_own_close_even_when_seen_late():
    precos = tabela_de_precos([10.0, 10.0, 20.0, 40.0])
    dias = precos.index
    registros = {"Composto": pd.concat([registro("2026-10-27"), registro(dias[2], OUTROS)])}
    novas = sim.ordens_devidas(ledger(), registros, precos, SEM_CDI, sem_custo)
    assert list(novas["data"].drop_duplicates()) == [dias[0], dias[2]]
    compra = novas[novas["data"] == dias[0]].set_index("ticker")
    troca = novas[novas["data"] == dias[2]].set_index("ticker")
    assert troca.loc["AAAA3", "ordem"] == -compra.loc["AAAA3", "ordem"]     # the whole position
    assert troca.loc["AAAA3", "preco"] == 20.0                              # not the 40.0 of the last row in the file
    assert troca.loc["FFFF3", "ordem"] > 0


def test_a_record_whose_close_is_not_in_the_files_yet_waits():
    precos = tabela_de_precos([10.0])
    registros = {"Composto": pd.concat([registro("2026-10-27"), registro("2026-11-05")])}
    novas = sim.ordens_devidas(ledger(), registros, precos, SEM_CDI, sem_custo)
    assert set(novas["data"]) == {INICIO}


def test_a_stock_with_no_price_that_day_uses_its_last_close():
    precos = tabela_de_precos([10.0, float("nan")])
    dias = precos.index
    registros = {"Composto": pd.concat([registro("2026-10-27"), registro(dias[1], OUTROS)])}
    novas = sim.ordens_devidas(ledger(), registros, precos, SEM_CDI, sem_custo)
    assert novas[(novas["data"] == dias[1]) & (novas["ticker"] == "AAAA3")]["preco"].item() == 10.0


def test_costs_come_from_the_given_function():
    novas = sim.ordens_devidas(ledger(), {"Composto": registro("2026-10-27")}, tabela_de_precos([10.0]), SEM_CDI,
                               lambda _ticker, valor: valor * 0.001)
    assert novas["custo"].sum() == pytest.approx((novas["ordem"] * novas["preco"]).sum() * 0.001)


def linhas(*rebalanceios) -> pd.DataFrame:
    """Ledger rows from (date, portfolio, {ticker: (order, close, cost)})."""
    return pd.DataFrame([{"data": pd.Timestamp(data), "carteira": carteira, "ticker": ticker, "ordem": ordem,
                          "preco": preco, "custo": custo}
                         for data, carteira, ordens in rebalanceios
                         for ticker, (ordem, preco, custo) in ordens.items()], columns=sim.COLUNAS)


def curva_de_uma_conta(ordens, precos, cdi=SEM_CDI):
    return sim.curvas(ordens, precos, cdi)["Composto"]


def com_referencias(tabela: pd.DataFrame) -> pd.DataFrame:
    return tabela.assign(BOVA11=100.0, IVVB11=100.0)


def test_the_first_day_is_the_capital_less_costs_plus_the_options_sleeve():
    precos = com_referencias(tabela_de_precos([10.0, 10.0]))
    ordens = linhas((INICIO, "Composto", {"AAAA3": (800, 10.0, 8.0)}))
    assert curva_de_uma_conta(ordens, precos).iloc[0] == pytest.approx(10_000.0 - 8.0)


def test_a_position_moves_with_the_ratio_of_the_index_not_with_its_level():
    """The lake re-anchors its total-return index: a file 10 times higher must value the position the same."""
    ordens = linhas((INICIO, "Composto", {"AAAA3": (800, 10.0, 0.0)}))
    base = com_referencias(tabela_de_precos([10.0, 12.0]))
    reancorado = base.assign(AAAA3=base["AAAA3"] * 10)
    assert curva_de_uma_conta(ordens, base).iloc[-1] == pytest.approx(9_600 + 1_000 + 1_000)
    assert curva_de_uma_conta(ordens, reancorado).iloc[-1] == pytest.approx(9_600 + 1_000 + 1_000)


def test_cash_and_the_options_sleeve_earn_the_cdi():
    precos = com_referencias(tabela_de_precos([10.0, 10.0]))
    ordens = linhas((INICIO, "Composto", {"AAAA3": (800, 10.0, 0.0)}))
    cdi = pd.Series(0.01, index=precos.index)
    assert curva_de_uma_conta(ordens, precos, cdi).iloc[-1] == pytest.approx(8_000 + 1_000 * 1.01 + 1_000 * 1.01)


def test_the_last_session_without_cdi_repeats_the_last_known_rate():
    precos = com_referencias(tabela_de_precos([10.0, 10.0, 10.0]))
    ordens = linhas((INICIO, "Composto", {"AAAA3": (900, 10.0, 0.0)}))
    cdi = pd.Series(0.01, index=precos.index[:2])
    assert curva_de_uma_conta(ordens, precos, cdi).iloc[-1] == pytest.approx(9_000 + 1_000 * 1.01 ** 2)


def test_on_a_rebalance_day_the_dividends_in_the_index_become_cash_and_nothing_vanishes():
    """AAAA3's index gains 10% while its close stays flat: a dividend. The sale at 10.0 cannot lose that gain."""
    precos = com_referencias(tabela_de_precos([10.0, 11.0, 11.0]))
    dias = precos.index
    ordens = linhas((dias[0], "Composto", {"AAAA3": (900, 10.0, 0.0)}),
                    (dias[1], "Composto", {"AAAA3": (-900, 10.0, 0.0), "BBBB3": (900, 10.0, 0.0)}))
    curva = curva_de_uma_conta(ordens, precos)
    assert curva.loc[dias[1]] == pytest.approx(9_900 + 1_000)
    assert curva.loc[dias[2]] == pytest.approx(9_900 + 1_000)


def test_a_rebalance_day_shows_the_value_after_its_costs():
    precos = com_referencias(tabela_de_precos([10.0, 10.0]))
    dias = precos.index
    ordens = linhas((dias[0], "Composto", {"AAAA3": (900, 10.0, 0.0)}),
                    (dias[1], "Composto", {"AAAA3": (-900, 10.0, 5.0), "BBBB3": (900, 10.0, 7.0)}))
    assert curva_de_uma_conta(ordens, precos).loc[dias[1]] == pytest.approx(10_000 - 12)


def test_the_benchmarks_start_at_the_whole_capital_on_the_first_buy_with_no_costs():
    precos = tabela_de_precos([10.0, 10.0]).assign(BOVA11=[100.0, 110.0], IVVB11=[200.0, 190.0])
    ordens = linhas((INICIO, "Composto", {"AAAA3": (900, 10.0, 9.0)}))
    cdi = pd.Series(0.01, index=precos.index)
    curvas = sim.curvas(ordens, precos, cdi)
    assert list(curvas.columns) == ["Composto", "BOVA11", "IVVB11", "CDI"]
    assert curvas.iloc[0][["BOVA11", "IVVB11", "CDI"]].tolist() == pytest.approx([10_000.0] * 3)
    assert curvas.iloc[-1][["BOVA11", "IVVB11", "CDI"]].tolist() == pytest.approx([11_000.0, 9_500.0, 10_100.0])


def test_the_control_line_comes_after_the_composite():
    precos = com_referencias(tabela_de_precos([10.0]))
    ordens = linhas((INICIO, "Conservadora", {"BBBB3": (900, 10.0, 0.0)}), (INICIO, "Composto", {"AAAA3": (900, 10.0, 0.0)}))
    assert list(sim.curvas(ordens, precos, SEM_CDI).columns[:2]) == ["Composto", "Conservadora"]


def test_a_rebalance_planned_on_the_replay_costs_exactly_its_orders():
    """Plan and curve read the same account: the value only drops by the rebalance's costs that day."""
    precos = com_referencias(tabela_de_precos([10.0, 12.0, 15.0, 15.0], BBBB3=[10.0, 11.0, 9.0, 9.0]))
    dias = precos.index
    registros = {"Composto": pd.concat([registro("2026-10-27"), registro(dias[2], OUTROS)])}
    custo = lambda _ticker, valor: valor * 0.002     # noqa: E731
    ordens = sim.ordens_devidas(ledger(), registros, precos, SEM_CDI, custo)
    sem_troca = ordens[ordens["data"] == dias[0]]
    antes = curva_de_uma_conta(sem_troca, precos).loc[dias[2]]
    depois = curva_de_uma_conta(ordens, precos).loc[dias[2]]
    assert antes - depois == pytest.approx(ordens.loc[ordens["data"] == dias[2], "custo"].sum())


def test_the_recent_returns_read_the_day_the_calendar_week_and_month_and_the_whole_period():
    dias = pd.DatetimeIndex(["2026-11-27", "2026-11-30", "2026-12-01", "2026-12-04", "2026-12-07", "2026-12-08"])
    curvas = pd.DataFrame({"Composto": [100.0, 110.0, 121.0, 110.0, 99.0, 104.0]}, index=dias)
    linha = sim.retornos_recentes(curvas).loc["Composto"]
    assert linha["Dia"] == pytest.approx(104 / 99 - 1)
    assert linha["Semana"] == pytest.approx(104 / 110 - 1)       # from Friday's close, 04/12
    assert linha["Mês"] == pytest.approx(104 / 110 - 1)          # from November's last close, 30/11
    assert linha["Desde o início"] == pytest.approx(0.04)
    assert linha["Queda máx."] == pytest.approx(99 / 121 - 1)


def test_the_first_session_has_no_return_yet():
    curvas = pd.DataFrame({"Composto": [100.0]}, index=[INICIO])
    linha = sim.retornos_recentes(curvas).loc["Composto"]
    assert linha[["Dia", "Semana", "Mês", "Desde o início", "Queda máx."]].tolist() == [0.0] * 5


def test_a_period_that_began_after_the_first_buy_starts_from_the_first_value():
    curvas = pd.DataFrame({"Composto": [100.0, 110.0]}, index=pd.DatetimeIndex(["2026-11-03", "2026-11-04"]))
    linha = sim.retornos_recentes(curvas).loc["Composto"]
    assert linha["Semana"] == pytest.approx(0.10)
    assert linha["Mês"] == pytest.approx(0.10)


def test_the_drawdown_is_measured_from_each_curves_own_peak():
    curvas = pd.DataFrame({"Composto": [100.0, 120.0, 90.0]}, index=pd.bdate_range(INICIO, periods=3))
    assert sim.quedas(curvas)["Composto"].tolist() == pytest.approx([0.0, 0.0, -0.25])


def test_positions_show_the_last_rebalance_at_todays_value():
    precos = com_referencias(tabela_de_precos([10.0, 12.0, 15.0]))
    dias = precos.index
    ordens = linhas((dias[0], "Composto", {"AAAA3": (400, 10.0, 0.0), "BBBB3": (400, 10.0, 0.0)}),
                    (dias[1], "Composto", {"AAAA3": (-200, 12.0, 0.0), "BBBB3": (0, 10.0, 0.0),
                                           "CCCC3": (100, 10.0, 0.0)}),
                    (dias[1], "Conservadora", {"DDDD3": (900, 10.0, 0.0)}))
    tabela = sim.posicoes(ordens, "Composto", precos, pd.Series(0.01, index=dias))
    assert tabela.loc["AAAA3", "Qtd."] == 200
    assert tabela.loc["AAAA3", "Valor (R$)"] == pytest.approx(200 * 12.0 * 15.0 / 12.0)
    assert tabela.loc["AAAA3", "Desde a troca"] == pytest.approx(15.0 / 12.0 - 1)
    assert "DDDD3" not in tabela.index
    na_troca = 1_000 * 1.01 + 400 * 12 + 400 * 10          # the day's cash and the two positions before the orders
    caixa = na_troca - (200 * 12 + 400 * 10 + 100 * 10)
    assert tabela.loc[sim.LINHA_CAIXA, "Valor (R$)"] == pytest.approx(caixa * 1.01)
    assert tabela.loc[sim.LINHA_OPCOES, "Valor (R$)"] == pytest.approx(1_000 * 1.01 ** 2)
    assert tabela["Peso"].sum() == pytest.approx(1.0)


def test_a_position_sold_in_full_leaves_the_table():
    precos = com_referencias(tabela_de_precos([10.0, 10.0]))
    dias = precos.index
    ordens = linhas((dias[0], "Composto", {"AAAA3": (900, 10.0, 0.0)}),
                    (dias[1], "Composto", {"AAAA3": (-900, 10.0, 0.0), "BBBB3": (900, 10.0, 0.0)}))
    assert "AAAA3" not in sim.posicoes(ordens, "Composto", precos, SEM_CDI).index


def test_the_panel_holds_the_curves_and_each_portfolios_positions():
    precos = com_referencias(tabela_de_precos([10.0, 11.0]))
    ordens = linhas((INICIO, "Composto", {"AAAA3": (900, 10.0, 0.0)}), (INICIO, "Conservadora", {"BBBB3": (900, 10.0, 0.0)}))
    painel = sim.painel(ordens, precos, pd.Series(0.01, index=precos.index))
    assert list(painel.curvas.columns) == ["Composto", "Conservadora", "BOVA11", "IVVB11", "CDI"]
    assert set(painel.posicoes) == {"Composto", "Conservadora"}
    assert painel.posicoes["Conservadora"].index[0] == "BBBB3"
    assert painel.com_cdi


def test_the_panel_says_when_the_cdi_is_missing():
    precos = com_referencias(tabela_de_precos([10.0]))
    assert not sim.painel(linhas((INICIO, "Composto", {"AAAA3": (900, 10.0, 0.0)})), precos, SEM_CDI).com_cdi


def _tela_vazia():
    from ui import simulador as tela
    tela.secao(None)


def _tela_com_conta():
    import pandas as pd

    from core import simulador as sim
    from ui import simulador as tela
    dias = pd.bdate_range("2026-11-03", periods=3)
    precos = pd.DataFrame({"AAAA3": [10.0, 11.0, 12.0], "BBBB3": [10.0, 9.0, 8.0], "BOVA11": [100.0, 101.0, 99.0],
                           "IVVB11": [100.0, 100.0, 102.0]}, index=dias)
    ordens = pd.DataFrame({"data": dias[0], "carteira": ["Composto", "Conservadora"], "ticker": ["AAAA3", "BBBB3"],
                           "ordem": [800, 800], "preco": [10.0, 10.0], "custo": [8.0, 8.0]})
    tela.secao(sim.painel(ordens, precos, pd.Series(0.0005, index=dias)))


def _tela_sem_cdi():
    import pandas as pd

    from core import simulador as sim
    from ui import simulador as tela
    dias = pd.bdate_range("2026-11-03", periods=2)
    precos = pd.DataFrame({"AAAA3": [10.0, 11.0], "BOVA11": [100.0, 101.0], "IVVB11": [100.0, 100.0]}, index=dias)
    ordens = pd.DataFrame({"data": [dias[0]], "carteira": ["Composto"], "ticker": ["AAAA3"], "ordem": [800],
                           "preco": [10.0], "custo": [8.0]})
    tela.secao(sim.painel(ordens, precos, pd.Series(dtype=float)))


def test_before_the_first_buy_the_screen_says_when_it_starts():
    from streamlit.testing.v1 import AppTest
    tela = AppTest.from_function(_tela_vazia, default_timeout=TIMEOUT_S).run()
    assert not tela.exception
    assert "03/11/2026" in tela.info[0].value
    assert not tela.dataframe


def test_the_screen_shows_the_account_value_returns_and_both_position_tables():
    from streamlit.testing.v1 import AppTest
    tela = AppTest.from_function(_tela_com_conta, default_timeout=TIMEOUT_S).run()
    assert not tela.exception
    assert "fictícia" in tela.caption[0].value
    assert len(tela.metric) == 4
    assert tela.metric[0].label == "Patrimônio do Composto"
    assert len(tela.dataframe) == 3                              # returns, Composite positions, Conservadora positions
    assert list(tela.dataframe[0].value.index[:2]) == ["Composto", "Conservadora"]
    assert not tela.warning


def test_the_screen_warns_when_the_cdi_is_missing():
    from streamlit.testing.v1 import AppTest
    tela = AppTest.from_function(_tela_sem_cdi, default_timeout=TIMEOUT_S).run()
    assert not tela.exception
    assert "CDI" in tela.warning[0].value


def test_the_loader_has_nothing_before_the_first_buy(tmp_path):
    from ui.servicos import carregar_simulador
    assert carregar_simulador(pd.Timestamp("2026-11-04").date(), 0, tmp_path / "ordens.csv") is None


def test_the_loader_reads_the_stocks_of_the_ledger_and_the_benchmarks(tmp_path, monkeypatch):
    from ui import servicos
    arquivo = tmp_path / "ordens.csv"
    sim.acrescentar_ordens(arquivo, linhas((INICIO, "Composto", {"AAAA3": (900, 10.0, 9.0)})))
    pedidos = []

    def fechamentos(tickers, inicio, _fim):
        pedidos.append((tickers, inicio))
        return pd.DataFrame(10.0, index=pd.bdate_range(INICIO, periods=2), columns=tickers)
    monkeypatch.setattr(servicos.precos_do_app, "ler_fechamentos", fechamentos)
    monkeypatch.setattr(servicos.cdi_lago, "daily_cdi", lambda _anos, _dia: pd.Series(dtype=float))
    resultado = servicos.carregar_simulador(pd.Timestamp("2026-11-05").date(), 1, arquivo)
    assert pedidos == [(["AAAA3.SA", "BOVA11.SA", "IVVB11.SA"], INICIO.date())]
    assert resultado.curvas["Composto"].iloc[0] == pytest.approx(10_000 - 9)


def test_the_ledger_file_round_trips_and_a_missing_one_is_empty(tmp_path):
    arquivo = tmp_path / "simulador" / "ordens.csv"
    assert sim.ler_ordens(arquivo).empty
    assert list(sim.ler_ordens(arquivo).columns) == sim.COLUNAS
    novas = linhas((INICIO, "Composto", {"AAAA3": (900, 10.5, 1.25)}))
    sim.acrescentar_ordens(arquivo, novas)
    sim.acrescentar_ordens(arquivo, linhas(("2026-11-04", "Composto", {"AAAA3": (-900, 11.0, 1.5)})))
    lido = sim.ler_ordens(arquivo)
    assert lido["ordem"].tolist() == [900, -900]
    assert lido["data"].tolist() == [INICIO, pd.Timestamp("2026-11-04")]
    assert arquivo.read_bytes().count(b"\r") == 0
