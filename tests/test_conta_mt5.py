"""The MT5 demo book on the screen (spec ``2026-11-17-conta-mt5-no-app-design.md``)."""

from __future__ import annotations

import pandas as pd
import pytest

COLUNAS_FILLS = ["name", "symbol", "side", "volume", "price", "commission", "time", "order"]


def _fills(*linhas: tuple) -> pd.DataFrame:
    return pd.DataFrame(list(linhas), columns=COLUNAS_FILLS)


def test_positions_merge_lot_and_fractional_and_average_the_buy_price():
    from core.conta_mt5 import posicoes
    fills = _fills(("ALLD3", "ALLD3", "buy", 900, 7.20, 0.0, "2026-11-13 18:00:00", 1),
                   ("ALLD3", "ALLD3F", "buy", 35, 7.30, 0.0, "2026-11-13 18:00:00", 2),
                   ("FIQE3", "FIQE3", "buy", 1200, 5.54, 0.0, "2026-11-13 18:03:00", 3),
                   ("FIQE3", "FIQE3", "sell", 200, 5.80, 0.0, "2026-11-16 18:00:00", 4),
                   ("MILS3", "MILS3F", "buy", 6, 16.64, 0.0, "2026-11-13 18:00:00", 5),
                   ("MILS3", "MILS3F", "sell", 6, 17.00, 0.0, "2026-11-16 18:00:00", 6))
    tabela = posicoes(fills)
    assert tabela["Papel"].tolist() == ["ALLD3", "FIQE3"]          # a name sold to zero leaves the table
    assert tabela["Qtd."].tolist() == [935, 1000]
    assert tabela["Preço médio (R$)"].tolist() == pytest.approx([(900 * 7.20 + 35 * 7.30) / 935, 5.54])
    assert tabela["Custo (R$)"].tolist() == pytest.approx([935 * (900 * 7.20 + 35 * 7.30) / 935, 1000 * 5.54])


def _gravar(pasta, patrimonio: str | None, fills: pd.DataFrame | None) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    if patrimonio is not None:
        (pasta / "patrimonio.csv").write_text(patrimonio, encoding="utf-8")
    if fills is not None:
        fills.to_csv(pasta / "fills.csv", index=False)


def test_the_panel_reads_the_value_curve_the_day_and_the_cash_from_the_two_files(tmp_path):
    from core.conta_mt5 import INICIAL, ler
    fills = _fills(("FIQE3", "FIQE3", "buy", 1000, 5.00, 1.50, "2026-11-13 18:03:00", 3),
                   ("ALLD3", "ALLD3F", "buy", 10, 7.00, 0.0, "2026-11-13 18:00:00", 2))
    _gravar(tmp_path, "data,patrimonio\n2026-11-13,97719.38\n2026-11-16,96828.81\n", fills)
    painel = ler(tmp_path)
    assert painel.curva.index.tolist() == [pd.Timestamp("2026-11-13"), pd.Timestamp("2026-11-16")]
    assert painel.linha.patrimonio == pytest.approx(96828.81)
    assert painel.linha.dia_reais == pytest.approx(96828.81 - 97719.38)
    assert painel.linha.desde_inicio == pytest.approx(96828.81 / INICIAL - 1)
    assert painel.caixa == pytest.approx(INICIAL - 1000 * 5.00 - 1.50 - 10 * 7.00)
    assert painel.posicoes["Papel"].tolist() == ["ALLD3", "FIQE3"]


def test_no_panel_without_the_value_file_and_an_empty_book_without_fills(tmp_path):
    from core.conta_mt5 import INICIAL, ler
    assert ler(tmp_path / "nada") is None
    _gravar(tmp_path, "data,patrimonio\n", None)
    assert ler(tmp_path) is None                                    # header only: no day yet
    _gravar(tmp_path, "data,patrimonio\n2026-11-13,100000.0\n", None)
    painel = ler(tmp_path)
    assert painel.caixa == INICIAL and painel.posicoes.empty


def test_the_initial_amount_is_the_executors():
    from core.conta_mt5 import INICIAL
    from scripts.mt5_executor import INICIAL as DO_EXECUTOR
    assert INICIAL == DO_EXECUTOR


def test_side_by_side_starts_both_accounts_at_the_initial_amount_on_the_close_before_the_book():
    from core.conta_mt5 import INICIAL, lado_a_lado
    mt5 = pd.Series([97_000.0, 96_000.0], index=pd.to_datetime(["2026-11-13", "2026-11-16"]))
    composto = pd.Series([10_000.0, 10_200.0, 10_100.0, 10_302.0],
                         index=pd.to_datetime(["2026-11-11", "2026-11-12", "2026-11-13", "2026-11-16"]))
    tabela = lado_a_lado(mt5, composto)
    assert tabela.index.tolist() == pd.to_datetime(["2026-11-12", "2026-11-13", "2026-11-16"]).tolist()
    assert tabela["MetaTrader 5"].tolist() == [INICIAL, 97_000.0, 96_000.0]
    assert tabela["Composite (simulador)"].tolist() == pytest.approx([INICIAL, INICIAL * 10_100 / 10_200,
                                                                      INICIAL * 10_302 / 10_200])


def test_side_by_side_without_the_simulator_or_a_close_before_is_the_book_alone():
    from core.conta_mt5 import INICIAL, lado_a_lado
    mt5 = pd.Series([97_000.0], index=pd.to_datetime(["2026-11-13"]))
    tardio = pd.Series([10_000.0], index=pd.to_datetime(["2026-11-13"]))
    for composto in (None, tardio):
        tabela = lado_a_lado(mt5, composto)
        assert tabela.columns.tolist() == ["MetaTrader 5"]
        assert tabela["MetaTrader 5"].tolist() == [97_000.0]


TIMEOUT_S = 60


def _tela_vazia():
    from ui.conta_mt5 import secao
    secao(None, None)


def _tela_com_conta(caixa: float = -1798.33):
    import pandas as pd

    from core.conta_mt5 import INICIAL, PainelMT5, posicoes
    from core.resultado_diario import linha
    from ui.conta_mt5 import secao
    fills = pd.DataFrame([("FIQE3", "FIQE3", "buy", 1000, 5.5, 0.0, "2026-11-13 18:03:00", 3)],
                         columns=["name", "symbol", "side", "volume", "price", "commission", "time", "order"])
    curva = pd.Series([97_719.38, 96_828.81], index=pd.to_datetime(["2026-11-13", "2026-11-16"]))
    composto = pd.Series([10_000.0, 10_050.0, 9_990.0], index=pd.to_datetime(["2026-11-12", "2026-11-13", "2026-11-16"]))
    secao(PainelMT5(curva, linha(curva, INICIAL), caixa, posicoes(fills)), composto)


def test_the_screen_before_any_value_says_what_the_account_is():
    from streamlit.testing.v1 import AppTest
    tela = AppTest.from_function(_tela_vazia, default_timeout=TIMEOUT_S).run()
    assert not tela.exception
    assert "demo" in tela.caption[0].value and "fictício" in tela.caption[0].value
    assert not tela.metric and not tela.dataframe


def test_the_screen_shows_the_value_the_day_the_cash_and_the_composite_and_flags_a_negative_cash():
    from streamlit.testing.v1 import AppTest
    tela = AppTest.from_function(_tela_com_conta, default_timeout=TIMEOUT_S).run()
    assert not tela.exception
    assert [m.label for m in tela.metric] == ["Patrimônio no MetaTrader 5", "Resultado do dia", "Caixa",
                                              "Composite no simulador"]
    assert tela.metric[0].value == "R$ 96.828,81"
    assert tela.metric[2].value == "-R$ 1.798,33"
    assert tela.metric[3].value == "R$ 99.900,00"          # 9,990 / 10,000 (12/11, the close before) on R$ 100,000
    assert len(tela.warning) == 1 and "negativo" in tela.warning[0].value
    assert tela.dataframe[0].value["Papel"].tolist() == ["FIQE3"]
    assert any("16/11/2026" in nota.value for nota in tela.caption)


def test_a_positive_cash_raises_no_warning():
    from streamlit.testing.v1 import AppTest
    tela = AppTest.from_function(_tela_com_conta, args=(1_500.0,), default_timeout=TIMEOUT_S).run()
    assert not tela.exception and not tela.warning


def test_the_loader_reads_the_published_folder_and_its_key_follows_the_value_file(tmp_path):
    from ui.servicos import assinatura_mt5, carregar_conta_mt5
    assert assinatura_mt5(tmp_path) == 0 and carregar_conta_mt5(0, tmp_path) is None
    _gravar(tmp_path, "data,patrimonio\n2026-11-13,97719.38\n", None)
    chave = assinatura_mt5(tmp_path)
    assert chave > 0
    assert carregar_conta_mt5(chave, tmp_path).linha.patrimonio == pytest.approx(97719.38)
