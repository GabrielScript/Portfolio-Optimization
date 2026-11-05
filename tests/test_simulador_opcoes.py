import numpy as np
import pandas as pd
import pytest

from core import simulador_opcoes as so

DIA = pd.Timestamp("2026-11-06")


def _linha(ticker, ativo, tipo, strike, vencimento, du, close, bid, ask, trades=5, spot=30.0, data=DIA, volume=2e6):
    return {"ticker": ticker, "date": data, "underlying": ativo, "option_type": tipo, "strike": strike,
            "expiry": pd.Timestamp(vencimento), "du": du, "close": close, "best_bid": bid, "best_ask": ask,
            "trades": trades, "volume_brl": 1000.0, "spot_close": spot, "underlying_volume_21": volume}


def test_the_first_cycle_is_the_first_session_from_the_start_then_each_month_s_first_session():
    pregoes = pd.DatetimeIndex(["2026-11-05", "2026-11-06", "2026-11-30", "2026-12-01", "2026-12-02", "2027-01-04"])
    assert so.ciclos(pregoes, pd.Timestamp("2026-11-06")) == [pd.Timestamp("2026-11-06"), pd.Timestamp("2026-12-01"),
                                                              pd.Timestamp("2027-01-04")]


def _fechamentos():
    dias = pd.bdate_range(end=DIA, periods=260)
    passo = np.arange(260)
    return pd.DataFrame({
        "ALTA3": 10 * 1.004 ** passo,                                   # steady winner, at its high
        "CAIU3": np.r_[10 * 1.006 ** passo[:200], 10 * 1.006 ** 199 * 0.97 ** passo[:60]],   # ran up, now far below
        "VOLA3": 10 * (1.003 ** passo) * (1 + 0.05 * np.sin(passo)),    # volatile winner
        "CALM3": 10 * 1.0005 ** passo}, index=dias)


def test_idea_a_takes_the_biggest_252_session_winner_still_within_5_percent_of_its_high():
    assert so.ideia_a(_fechamentos(), ["ALTA3", "CAIU3", "CALM3"]) == "ALTA3"
    assert so.ideia_a(_fechamentos(), ["CAIU3"]) is None


def test_idea_b_takes_the_top_momentum_name_of_the_most_volatile_third():
    assert so.ideia_b(_fechamentos(), ["ALTA3", "VOLA3", "CALM3"]) == "VOLA3"


def test_the_contract_is_the_traded_one_nearest_21_business_days_and_the_target_strike():
    cadeia = pd.DataFrame([
        _linha("ALTAW285", "ALTA3", "put", 28.5, "2026-12-04", 20, 0.50, 0.45, 0.55),
        _linha("ALTAW300", "ALTA3", "put", 30.0, "2026-12-04", 20, 1.20, 1.10, 1.30),
        _linha("ALTAX285", "ALTA3", "put", 28.5, "2027-01-08", 42, 0.90, 0.85, 0.95),
        _linha("ALTAW270", "ALTA3", "put", 27.0, "2026-12-04", 20, 0.20, 0.0, 0.0),          # no ask: never bought
        _linha("ALTAL315", "ALTA3", "call", 31.5, "2026-12-04", 20, 0.40, 0.35, 0.45, trades=0)])
    assert so.contrato(cadeia, "ALTA3", "put", 0.95)["ticker"] == "ALTAW285"
    assert so.contrato(cadeia, "ALTA3", "call", 1.05) is None                                # did not trade


def test_a_lone_ask_with_no_real_bid_is_never_bought_and_the_next_strike_is_taken():
    # the 2026-11-05 dry run: VAMO3's call asked R$ 3.00 against a R$ 0.22 bid
    cadeia = pd.DataFrame([
        _linha("VAMOK246", "VAMO3", "call", 2.46, "2026-12-04", 20, 0.25, 0.20, 3.00, spot=2.34),
        _linha("VAMOK250", "VAMO3", "call", 2.50, "2026-12-04", 20, 0.20, 0.18, 0.22, spot=2.34),
        _linha("VAMOK240", "VAMO3", "call", 2.40, "2026-12-04", 20, 0.30, 0.0, 0.31, spot=2.34)])   # no bid at all
    assert so.contrato(cadeia, "VAMO3", "call", 1.05)["ticker"] == "VAMOK250"


def test_a_cycle_spends_half_the_budget_per_idea_in_lots_of_100_or_records_why_not():
    cadeia = pd.DataFrame([_linha("ALTAW285", "ALTA3", "put", 28.5, "2026-12-04", 20, 0.50, 0.45, 0.55),
                           _linha("VOLAL315", "VOLA3", "call", 31.5, "2026-12-04", 20, 9.0, 8.0, 9.0)])
    ordens = so.ordens_do_ciclo(DIA, cadeia, _fechamentos(), ["ALTA3", "VOLA3", "CALM3"], patrimonio=10_000.0)
    put = ordens.set_index("ideia").loc["A"]
    assert put["ticker"] == "ALTAW285" and put["quantidade"] == 900 and put["preco"] == 0.55       # R$ 500 / 55
    assert put["custo"] == pytest.approx(900 * 0.55 * so.EMOLUMENTOS)
    call = ordens.set_index("ideia").loc["B"]
    assert call["quantidade"] == 0 and "lote" in call["motivo"]                  # one lot is R$ 900 > R$ 500


def test_marks_follow_the_mid_then_the_intrinsic_value_on_expiry():
    livro = pd.DataFrame([{"data": DIA, "ideia": "A", "ticker": "ALTAW285", "ativo": "ALTA3", "tipo": "put",
                           "strike": 28.5, "vencimento": pd.Timestamp("2026-11-10"), "quantidade": 100,
                           "preco": 0.55, "custo": 0.02, "motivo": ""}])
    cadeia = pd.DataFrame([
        _linha("ALTAW285", "ALTA3", "put", 28.5, "2026-11-10", 1, 0.70, 0.60, 0.80, data=pd.Timestamp("2026-11-09")),
        _linha("ALTAW300", "ALTA3", "put", 30.0, "2026-11-10", 0, 3.10, 3.0, 3.2, spot=26.0,
               data=pd.Timestamp("2026-11-10"))])
    marcas = so.marcas_devidas(livro, cadeia, pd.DataFrame(columns=so.COLUNAS_MARCAS))
    assert marcas["valor"].tolist() == [pytest.approx(0.70), pytest.approx(2.5)]               # mid, then 28.5 - 26
    assert marcas["vencida"].tolist() == [False, True]
    assert so.marcas_devidas(livro, cadeia, marcas).empty                                     # nothing marked twice


def test_a_mark_ignores_a_lone_ask_and_takes_the_last_trade():
    linha = pd.Series({"best_bid": 0.20, "best_ask": 3.00, "close": 0.25})
    assert so._valor_de_tela(linha) == pytest.approx(0.25)


def test_the_curve_pays_the_premium_values_the_option_and_taxes_the_month_s_net_gain():
    dias = pd.DatetimeIndex(["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11"])
    livro = pd.DataFrame([{"data": dias[0], "ideia": "A", "ticker": "X", "ativo": "ALTA3", "tipo": "put",
                           "strike": 28.5, "vencimento": dias[2], "quantidade": 100, "preco": 0.5, "custo": 0.0,
                           "motivo": ""}])
    marcas = pd.DataFrame({"data": [dias[1], dias[2]], "ticker": ["X", "X"], "valor": [0.7, 2.5],
                           "vencida": [False, True]})
    selic = pd.Series(0.0, index=dias)
    curva = so.curva(livro, marcas, selic, dias, capital=1_000.0)
    ganho = 100 * 2.5 - 100 * 0.5
    depois = 1_000 + ganho * (1 - so.IR)                                     # IR on the expiry session's net gain
    assert curva.tolist() == [pytest.approx(1_000.0), pytest.approx(1_020.0), pytest.approx(depois),
                              pytest.approx(depois)]


def test_an_update_trades_each_pending_cycle_once_and_marks_what_it_bought():
    proximo = pd.Timestamp("2026-11-09")
    cadeia = pd.DataFrame([_linha("ALTAW285", "ALTA3", "put", 28.5, "2026-12-04", 20, 0.50, 0.45, 0.55),
                           _linha("ALTAW285", "ALTA3", "put", 28.5, "2026-12-04", 19, 0.60, 0.58, 0.62,
                                  data=proximo)])
    vazio = (pd.DataFrame(columns=so.COLUNAS), pd.DataFrame(columns=so.COLUNAS_MARCAS))
    selic = pd.Series(0.0, index=pd.DatetimeIndex([DIA, proximo]))
    ordens, marcas = so.atualizacao(*vazio, cadeia, _fechamentos(), ["ALTA3", "CALM3"], selic, inicio=DIA)
    assert ordens["data"].tolist() == [DIA, DIA] and ordens["quantidade"].tolist() == [900, 0]
    assert marcas["data"].tolist() == [proximo] and marcas["valor"].tolist() == [pytest.approx(0.60)]
    ordens2, marcas2 = so.atualizacao(ordens, marcas, cadeia, _fechamentos(), ["ALTA3", "CALM3"], selic, inicio=DIA)
    assert ordens2.empty and marcas2.empty                                                   # nothing twice


def test_the_panel_shows_the_curve_beside_the_selic_the_orders_and_the_open_options():
    dias = pd.DatetimeIndex(["2026-11-06", "2026-11-09"])
    livro = pd.DataFrame([{"data": dias[0], "ideia": "A", "ticker": "X", "ativo": "ALTA3", "tipo": "put",
                           "strike": 28.5, "vencimento": pd.Timestamp("2026-12-04"), "quantidade": 100, "preco": 0.5,
                           "custo": 0.0, "motivo": ""},
                          {"data": dias[0], "ideia": "B", "ticker": None, "ativo": None, "tipo": "call",
                           "strike": None, "vencimento": None, "quantidade": 0, "preco": None, "custo": 0.0,
                           "motivo": "nenhum ativo passou na regra"}])
    marcas = pd.DataFrame({"data": [dias[1]], "ticker": ["X"], "valor": [0.7], "vencida": [False]})
    painel = so.painel(livro, marcas, pd.Series(0.001, index=dias), capital=1_000.0)
    assert painel.curvas.columns.tolist() == ["Opções", "Selic"]
    assert painel.curvas["Selic"].tolist() == [pytest.approx(1_000.0), pytest.approx(1_001.0)]
    assert painel.curvas["Opções"].iloc[-1] == pytest.approx((1_000 - 50) * 1.001 + 70)
    assert painel.ordens["Motivo"].tolist() == ["", "nenhum ativo passou na regra"]
    aberta = painel.abertas.iloc[0]
    assert aberta["Opção"] == "X" and aberta["Valor (R$)"] == pytest.approx(70.0)


def test_a_loss_is_carried_and_offsets_the_next_gain_before_any_tax():
    dias = pd.DatetimeIndex(["2026-11-06", "2026-11-20", "2026-12-01", "2026-12-18"])
    base = {"ideia": "A", "ativo": "ALTA3", "tipo": "put", "strike": 28.5, "custo": 0.0, "motivo": ""}
    livro = pd.DataFrame([base | {"data": dias[0], "ticker": "X", "vencimento": dias[1], "quantidade": 100,
                                  "preco": 1.0},
                          base | {"data": dias[2], "ticker": "Y", "vencimento": dias[3], "quantidade": 100,
                                  "preco": 1.0}])
    marcas = pd.DataFrame({"data": [dias[1], dias[3]], "ticker": ["X", "Y"], "valor": [0.0, 2.5],
                           "vencida": [True, True]})
    curva = so.curva(livro, marcas, pd.Series(0.0, index=dias), dias, capital=1_000.0)
    # -100 then +150: tax only on the 50 above the carried loss
    assert curva.iloc[-1] == pytest.approx(1_000 - 100 + 150 - so.IR * 50)
