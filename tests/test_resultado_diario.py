"""The 20:00 report (the user, 2026-11-13): the Composite's simulator account and the MT5 demo book after B3's close."""

from __future__ import annotations

import pandas as pd
import pytest

from core import resultado_diario as rd

ONTEM, HOJE = pd.Timestamp("2026-11-12"), pd.Timestamp("2026-11-13")


def test_todays_row_carries_each_total_return_price_by_its_official_close_move():
    precos = pd.DataFrame({"AAAA3": [9.0, 10.0], "BBBB3": [5.0, 4.0]}, index=[pd.Timestamp("2026-11-11"), ONTEM])
    base = pd.Series({"AAAA3": 20.0, "BBBB3": 8.0})
    hoje = pd.Series({"AAAA3": 22.0})                    # BBBB3 without a close today: left to the account's ffill
    estendidos = rd.com_fechamento_de_hoje(precos, base, hoje, HOJE)
    assert estendidos.index[-1] == HOJE and estendidos.loc[HOJE, "AAAA3"] == pytest.approx(11.0)
    assert pd.isna(estendidos.loc[HOJE, "BBBB3"]) and estendidos.loc[ONTEM].tolist() == [10.0, 4.0]


def test_b3s_print_is_the_official_close_only_when_it_is_todays_closing_auction():
    """B3's quote service stamps the closing auction 18:05 (DECISOES 383): a print of today from 18:00 on is the
    official close; yesterday's, one from the continuous session, or none is not."""
    assert rd.fechamento_oficial((43.98, pd.Timestamp("2026-11-13 18:05")), HOJE) == 43.98
    assert rd.fechamento_oficial((43.98, pd.Timestamp("2026-11-13 18:00")), HOJE) == 43.98
    assert rd.fechamento_oficial((43.50, pd.Timestamp("2026-11-13 17:54")), HOJE) is None
    assert rd.fechamento_oficial((44.02, pd.Timestamp("2026-11-12 18:05")), HOJE) is None
    assert rd.fechamento_oficial(None, HOJE) is None


def test_the_report_waits_for_the_session_before_giving_up():
    """The 16/11 and 17/11 runs stopped on the first look although B3 traded: the check is repeated, sleeping between
    tries, and gives up only after the last one."""
    respostas, esperas = iter([False, False, True]), []
    assert rd.aguardar_pregao(lambda: next(respostas), esperas.append, tentativas=6, intervalo=120.0)
    assert esperas == [120.0, 120.0]
    esperas.clear()
    assert not rd.aguardar_pregao(lambda: False, esperas.append, tentativas=3, intervalo=60.0)
    assert esperas == [60.0, 60.0]


def test_todays_close_is_b3s_official_one_and_mt5_only_fills_a_gap():
    """With B3's official close, today's price is that close (the lake's last price is the last official close, so it
    is the base); MT5's daily bars move a name only when B3 has none, with a note; with neither the name keeps its
    price. On 16/11 the MT5 move put the account R$ 44.72 above the official closes (DECISOES 383)."""
    ultimos = pd.Series({"AAAA3": 10.04, "BBBB3": 32.28, "CCCC3": 5.0})
    oficiais = {"AAAA3": 9.95, "BBBB3": None, "CCCC3": None}
    mt5_base, mt5_hoje = {"AAAA3": 10.10, "BBBB3": 32.30}, {"AAAA3": 10.20, "BBBB3": 32.11}
    notas: list[str] = []
    base, hoje = rd.fechamentos_de_hoje(ultimos, oficiais, mt5_base, mt5_hoje, notas)
    precos = pd.DataFrame([ultimos], index=[ONTEM])
    linha = rd.com_fechamento_de_hoje(precos, base, hoje, HOJE).loc[HOJE]
    assert linha["AAAA3"] == pytest.approx(9.95)                       # B3's close, not MT5's +1%
    assert linha["BBBB3"] == pytest.approx(32.28 * 32.11 / 32.30)       # MT5's move where B3 had nothing
    assert pd.isna(linha["CCCC3"])
    assert any("BBBB3" in nota and "MT5" in nota for nota in notas)
    assert any("CCCC3" in nota for nota in notas) and not any("AAAA3" in nota for nota in notas)


def test_the_simulators_line_is_today_against_the_last_session_and_the_start():
    curva = pd.Series([10_000.0, 10_508.74, 10_696.17], index=[pd.Timestamp("2026-11-03"), ONTEM, HOJE])
    linha = rd.linha(curva, 10_000.0)
    assert linha.patrimonio == pytest.approx(10_696.17)
    assert linha.dia == pytest.approx(10_696.17 / 10_508.74 - 1) and linha.dia_reais == pytest.approx(187.43)
    assert linha.desde_inicio == pytest.approx(0.069617)


def test_the_books_first_day_is_measured_from_its_starting_money():
    linha = rd.linha(pd.Series([99_850.0], index=[HOJE]), 100_000.0)
    assert linha.dia == pytest.approx(-0.0015) and linha.desde_inicio == pytest.approx(-0.0015)


def test_the_books_history_keeps_one_value_per_day_and_a_rerun_replaces_it():
    historico = rd.anotar_patrimonio(rd.anotar_patrimonio(pd.DataFrame(columns=["data", "patrimonio"]), ONTEM, 100.0),
                                     HOJE, 101.0)
    historico = rd.anotar_patrimonio(historico, HOJE, 102.0)
    assert historico["patrimonio"].tolist() == [100.0, 102.0]
    assert rd.serie_do_historico(historico).index.tolist() == [ONTEM, HOJE]


def test_the_email_gives_both_values_in_brazilian_numbers_and_says_the_simulator_number_is_a_preview():
    sim = rd.Linha(patrimonio=10_696.17, dia=0.0178, dia_reais=187.43, desde_inicio=0.0696, inicio=pd.Timestamp("2026-11-03"))
    livro = rd.Linha(patrimonio=99_850.0, dia=-0.0015, dia_reais=-150.0, desde_inicio=-0.0015, inicio=HOJE)
    assunto, corpo = rd.mensagem(HOJE, sim, 0.0074, livro, ["MLAS3: último negócio do dia"])
    assert assunto == "Aleph Quant 13/11/2026: Composite R$ 10.696,17 (+1,78%), MT5 R$ 99.850,00 (-0,15%)"
    assert "Simulador (Composite, conta fictícia de R$ 10.000,00 desde 03/11/2026)" in corpo
    assert "Patrimônio: R$ 10.696,17" in corpo and "Dia: +1,78% (+R$ 187,43)" in corpo
    assert "BOVA11 no dia: +0,74%" in corpo and "22:30" in corpo
    assert "fechamentos oficiais da B3" in corpo and "oficiais no MetaTrader 5" not in corpo     # DECISOES 383
    assert "MetaTrader 5 (conta demo da XP" in corpo and "Dia: -0,15% (-R$ 150,00)" in corpo
    assert "MLAS3: último negócio do dia" in corpo


def test_without_the_book_the_email_says_why():
    sim = rd.Linha(patrimonio=10_000.0, dia=0.0, dia_reais=0.0, desde_inicio=0.0, inicio=pd.Timestamp("2026-11-03"))
    assunto, corpo = rd.mensagem(HOJE, sim, 0.0, None, [], sem_livro="terminal fechado")
    assert "MT5" not in assunto and "MetaTrader 5: sem leitura (terminal fechado)" in corpo


class _SemNegocioHoje:
    """The demo terminal on 13/11: AAAA3 held (100 shares bought at R$ 10), no trade today, no daily bar today."""
    ACCOUNT_TRADE_MODE_DEMO, POSITION_TYPE_BUY, DEAL_TYPE_BUY, DEAL_TYPE_SELL, TIMEFRAME_D1 = 0, 0, 0, 1, 16408

    def account_info(self):
        from types import SimpleNamespace as NS
        return NS(trade_mode=0, server="XPMT5-DEMO", login=1, balance=1e6)

    def symbol_select(self, symbol, enable):
        return True

    def symbol_info_tick(self, symbol):
        from types import SimpleNamespace as NS
        return NS(bid=0.0, ask=0.0, last=0.0, time=int(ONTEM.timestamp()))

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        return [{"time": int(ONTEM.timestamp()), "close": 11.0}]

    def positions_get(self):
        from types import SimpleNamespace as NS
        from core.mt5_corretora import MAGIC
        return (NS(symbol="AAAA3", volume=100.0, type=0, magic=MAGIC),)

    def history_deals_get(self, date_from, date_to):
        from types import SimpleNamespace as NS
        from core.mt5_corretora import MAGIC
        return (NS(symbol="AAAA3", type=0, volume=100.0, price=10.0, commission=0.0, time=int(ONTEM.timestamp()),
                   order=1, magic=MAGIC),)


class _TerminalSemBarraDeHoje(_SemNegocioHoje):
    """A terminal that connects but holds no daily bar for today yet (the 16/11 and 17/11 runs)."""

    def initialize(self, path=None, timeout=None):
        return True

    def shutdown(self):
        return None

    def last_error(self):
        return (1, "ok")


def _rodar_com(monkeypatch, b3_bova):
    from datetime import datetime
    from scripts import resultado_diario as script
    monkeypatch.setattr(script, "linha_do_simulador", lambda c, hoje, notas, ler_b3=None: (
        rd.Linha(10_606.92, 0.0, 0.0, 0.06, pd.Timestamp("2026-11-03")), 0.0))
    monkeypatch.setattr(script, "linha_do_livro", lambda c, hoje, agora, notas, gravar: (None, "test"))
    esperas: list[float] = []
    resultado = script.rodar(_TerminalSemBarraDeHoje(), datetime(2026, 11, 13, 21, 0), simular=True,
                             ler_b3=lambda papel: b3_bova if papel == "BOVA11" else None, esperar=esperas.append)
    return resultado, esperas


def test_a_terminal_without_bars_gives_an_empty_series_that_compares_with_dates():
    """Now that B3 can confirm the session, the simulator line may meet a symbol with no MT5 bar at all: the empty
    series must still compare with a date (it had a RangeIndex and raised TypeError)."""
    from core.mt5_corretora import Corretora

    class SemBarras:
        TIMEFRAME_D1 = 16408

        def symbol_select(self, *argumentos):
            return True

        def copy_rates_from_pos(self, *argumentos):
            return []

    vazia = Corretora(SemBarras()).fechamentos("BOVA11", 10)
    assert vazia.empty and len(vazia.index[vazia.index < HOJE]) == 0


def test_b3s_closing_print_confirms_the_session_when_mt5_has_no_bar_yet(monkeypatch):
    resultado, esperas = _rodar_com(monkeypatch, (130.5, pd.Timestamp("2026-11-13 18:05")))
    assert resultado.desfecho == "simulado" and esperas == []


def test_without_b3_or_mt5_the_report_retries_then_says_no_session(monkeypatch):
    from scripts import resultado_diario as script
    resultado, esperas = _rodar_com(monkeypatch, None)
    assert resultado.desfecho == "sem pregão"
    assert len(esperas) == script.TENTATIVAS - 1


def test_a_held_name_without_a_trade_today_keeps_its_previous_close_in_the_books_value(monkeypatch, tmp_path):
    from datetime import datetime
    from core.mt5_corretora import Corretora
    from scripts import resultado_diario as script
    monkeypatch.setattr(script, "PASTA_MT5", tmp_path)
    notas: list[str] = []
    script.linha_do_livro(Corretora(_SemNegocioHoje()), HOJE, datetime(2026, 11, 13, 20, 0), notas, gravar=True)
    patrimonio = pd.read_csv(tmp_path / "patrimonio.csv")["patrimonio"].iloc[-1]
    assert patrimonio == pytest.approx(100_000 - 1_000 + 100 * 11.0)
    assert any("mantido o fechamento anterior" in nota for nota in notas)
