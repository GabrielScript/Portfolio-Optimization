"""Blocos de ui/secoes.py: PSR e DSR do walk-forward e o resumo do plano de compra."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from streamlit.proto.Metric_pb2 import Metric as MetricProto
from streamlit.testing.v1 import AppTest

from core.backtest import ResultadoBacktest
from core.market_data import FONTE_FECHAMENTO
from core.metrics import momentos_sharpe, sharpe_probabilistico, sharpe_referencia_deflacionado
from ui.secoes import _psr_e_dsr, _rotulo_cotacao

DATAS = pd.bdate_range("2022-01-03", periods=300)
RF_ZERO = pd.Series(0.0, index=DATAS)


def _backtest(nome: str, valores: np.ndarray) -> ResultadoBacktest:
    return ResultadoBacktest(nome, "walk-forward OOS", pd.Series(valores, index=DATAS),
                             pd.DataFrame(), pd.Series(dtype=float), 0.0)


def _curva(semente: int, media: float) -> np.ndarray:
    retornos = np.random.default_rng(semente).normal(media, 0.01, len(DATAS) - 1)
    return 100 * np.concatenate([[1.0], np.cumprod(1 + retornos)])


def test_estrategia_com_sharpe_nan_nao_anula_o_dsr_das_outras():
    """Curva constante tem Sharpe NaN: o DSR das demais deflaciona só pelas 2 com Sharpe finito."""
    a, b = _backtest("A", _curva(1, 0.001)), _backtest("B", _curva(2, 0.0005))
    resultados = {"A": a, "B": b, "Parada": _backtest("Parada", np.full(len(DATAS), 100.0))}

    saida = _psr_e_dsr(resultados, {}, RF_ZERO)

    momentos = {n: momentos_sharpe(bt.valores.pct_change().dropna()) for n, bt in [("A", a), ("B", b)]}
    variancia = float(np.var([m.sharpe for m in momentos.values()], ddof=1))
    referencia = sharpe_referencia_deflacionado(variancia, 2)
    assert saida["A"][1] == pytest.approx(sharpe_probabilistico(momentos["A"], referencia))
    assert saida["B"][1] == pytest.approx(sharpe_probabilistico(momentos["B"], referencia))
    assert np.isnan(saida["Parada"][0]) and np.isnan(saida["Parada"][1])


def test_referencia_de_mercado_tem_psr_mas_nao_tem_dsr():
    a, b = _backtest("A", _curva(1, 0.001)), _backtest("B", _curva(2, 0.0005))
    saida = _psr_e_dsr({"A": a}, {"B": b}, RF_ZERO)
    assert np.isfinite(saida["B"][0]) and np.isnan(saida["B"][1])


def test_nenhuma_das_referencias_de_mercado_entra_no_n_do_dsr():
    # BOVA11, SMAL11 e DIVO11 não foram escolhas entre tentativas: o DSR das estratégias não muda com elas
    a, b = _backtest("A", _curva(1, 0.001)), _backtest("B", _curva(2, 0.0005))
    referencias = {"SMAL11": _backtest("SMAL11", _curva(3, 0.002)), "DIVO11": _backtest("DIVO11", _curva(4, 0.0))}
    com, sem = _psr_e_dsr({"A": a, "B": b}, referencias, RF_ZERO), _psr_e_dsr({"A": a, "B": b}, {}, RF_ZERO)
    assert (com["A"][1], com["B"][1]) == pytest.approx((sem["A"][1], sem["B"][1]))


def test_aviso_de_fora_da_o_motivo_e_o_conselho_do_teto():
    def script():
        import pandas as pd

        from core.ordens import RegrasCompra, plano_de_compra
        from ui.secoes import resumo_plano

        # Teto 30% = R$ 30; 2 ações de BBBB3 = R$ 70: cabem no caixa que sobra, mas não no teto.
        regras = RegrasCompra(orcamento=100.0, custo_bps=0.0, peso_maximo=0.3, quantidade_minima=2)
        plano = plano_de_compra(pd.Series({"AAAA3.SA": 0.3, "BBBB3.SA": 0.3}),
                                pd.Series({"AAAA3.SA": 1.0, "BBBB3.SA": 35.0}), regras)
        resumo_plano(plano, regras)

    at = AppTest.from_function(script, default_timeout=60).run()

    assert not at.exception
    avisos = [w.value for w in at.warning]
    assert len(avisos) == 1
    assert "BBBB3" in avisos[0] and "teto de 30%" in avisos[0]
    assert "peso máximo" in avisos[0]


def test_aviso_de_fora_da_o_motivo_e_o_conselho_do_teto_por_setor():
    def script():
        import pandas as pd

        from core.config import LimiteGrupo
        from core.ordens import RegrasCompra, plano_de_compra
        from ui.secoes import resumo_plano

        # Setor {AAAA3, BBBB3} com teto de 30% = R$ 30: AAAA3 enche o setor; sobram R$ 10, que pagam 2 de BBBB3.
        setor = LimiteGrupo("Utilidade Pública", frozenset({"AAAA3.SA", "BBBB3.SA"}), 0.3)
        regras = RegrasCompra(orcamento=100.0, custo_bps=0.0, peso_maximo=1.0, quantidade_minima=2,
                              grupos=(setor,))
        plano = plano_de_compra(pd.Series({"AAAA3.SA": 0.28, "BBBB3.SA": 0.02, "CCCC3.SA": 0.70}),
                                pd.Series({"AAAA3.SA": 1.0, "BBBB3.SA": 5.0, "CCCC3.SA": 30.0}), regras)
        resumo_plano(plano, regras)

    at = AppTest.from_function(script, default_timeout=60).run()

    assert not at.exception
    avisos = [w.value for w in at.warning]
    assert len(avisos) == 1
    assert "BBBB3" in avisos[0] and "teto de 30% do setor Utilidade Pública" in avisos[0]
    assert "teto por setor" in avisos[0]


def test_rotulo_do_fechamento_nao_mostra_a_data():
    # O preço diário passa por ffill: a data da última linha pode ser de um dia em que a ação não negociou.
    assert _rotulo_cotacao(pd.Timestamp("2026-10-09"), FONTE_FECHAMENTO, date(2026, 10, 14)) == "último fechamento"


def test_investido_e_custo_estimado_nao_mostram_seta():
    # O delta deles é informativo ("100.0% do valor", "10 bps"): uma seta para cima sugere alta.
    def script():
        import pandas as pd

        from core.ordens import RegrasCompra, plano_de_compra
        from ui.secoes import resumo_plano

        regras = RegrasCompra(orcamento=1_000.0, custo_bps=10.0, peso_maximo=1.0, quantidade_minima=2)
        plano = plano_de_compra(pd.Series({"AAAA3.SA": 1.0}), pd.Series({"AAAA3.SA": 10.0}), regras)
        resumo_plano(plano, regras)

    at = AppTest.from_function(script, default_timeout=60).run()

    assert not at.exception
    direcoes = {m.label: m.proto.direction for m in at.metric}
    assert direcoes["Investido"] == MetricProto.MetricDirection.NONE
    assert direcoes["Custo estimado"] == MetricProto.MetricDirection.NONE


def test_sem_acao_descartada_nao_ha_aviso():
    from ui.secoes import aviso_descartados
    assert aviso_descartados([], []) is None


def _tabela_dinheiro_real(total: float, preco_nucleo: float, precos_satelite: list[float]) -> None:
    """O satélite registrado com pesos iguais nos papéis P0, P1, ... (cotações ``precos_satelite``) e o núcleo NUC."""
    from datetime import date

    import pandas as pd

    from core.market_data import FONTE_FECHAMENTO
    from core.ordens import RegrasDinheiroReal, plano_dinheiro_real
    from ui.secoes import tabela_dinheiro_real

    pesos = pd.Series(1 / len(precos_satelite), index=[f"P{i}.SA" for i in range(len(precos_satelite))])
    precos = pd.concat([pd.Series(precos_satelite, index=pesos.index), pd.Series({"NUC.SA": preco_nucleo})])
    regras = RegrasDinheiroReal(total=total, custo_bps=0.0, nucleo="NUC.SA")
    plano = plano_dinheiro_real(pesos, precos, {t: f"Setor {i}" for i, t in enumerate(pesos.index)}, regras)
    cotacoes = pd.DataFrame({"preco": precos, "horario": pd.Timestamp("2026-10-16"), "fonte": FONTE_FECHAMENTO})
    tabela_dinheiro_real(plano, regras, cotacoes, {}, date(2026, 10, 16))


def test_peso_alvo_do_dinheiro_real_e_a_regra_70_30_e_soma_100_por_cento():
    # núcleo a R$ 300 com R$ 1.000: 2 cotas (60%) e R$ 400 (40%) no satélite; o alvo segue 70% + 30% × peso registrado
    at = AppTest.from_function(_tabela_dinheiro_real, kwargs={"total": 1_000.0, "preco_nucleo": 300.0,
                                                                "precos_satelite": [1.0] * 5},
                               default_timeout=60).run()
    assert not at.exception
    alvo = at.dataframe[0].value.set_index("Ticker")["Peso alvo"]
    assert alvo["NUC"] == pytest.approx(0.70) and alvo.drop("NUC").tolist() == pytest.approx([0.06] * 5)


def test_dinheiro_real_avisa_o_papel_do_satelite_que_ficou_sem_acoes():
    # 6 papéis com 1/6 cada e R$ 300 no satélite: 2 ações de P5 (R$ 2.000) não cabem no teto de R$ 60; os outros 5
    # bastam para 100% (teto de 20%) e o plano sai, sem a P5 na tabela
    at = AppTest.from_function(_tabela_dinheiro_real, kwargs={"total": 1_000.0, "preco_nucleo": 100.0,
                                                                "precos_satelite": [1.0] * 5 + [1_000.0]},
                               default_timeout=60).run()
    assert not at.exception
    assert "P5" not in at.dataframe[0].value["Ticker"].tolist()
    avisos = [w.value for w in at.warning if "P5" in w.value and "teto de 20%" in w.value]
    assert len(avisos) == 1
    assert "peso máximo" not in avisos[0]       # no dinheiro real os tetos são fixos: só o valor a investir muda


def test_dinheiro_real_sem_uma_cota_do_nucleo_diz_que_esta_tudo_em_caixa():
    at = AppTest.from_function(_tabela_dinheiro_real, kwargs={"total": 100.0, "preco_nucleo": 300.0,
                                                                "precos_satelite": [1.0] * 5},
                               default_timeout=60).run()
    assert not at.exception
    assert [i.value.split(":")[0] for i in at.info] == ["Tudo em caixa"]


def test_dinheiro_real_depois_do_stop_diz_que_o_satelite_esta_no_nucleo():
    def script():
        from datetime import date

        import pandas as pd

        from core.market_data import FONTE_FECHAMENTO
        from core.ordens import RegrasDinheiroReal, plano_dinheiro_real
        from ui.secoes import tabela_dinheiro_real

        regras = RegrasDinheiroReal(total=1_000.0, custo_bps=0.0, nucleo="NUC.SA")
        precos = pd.Series({"NUC.SA": 100.0})
        plano = plano_dinheiro_real(pd.Series({"NUC.SA": 1.0}), precos, {}, regras)
        cotacoes = pd.DataFrame({"preco": precos, "horario": pd.Timestamp("2026-10-16"), "fonte": FONTE_FECHAMENTO})
        tabela_dinheiro_real(plano, regras, cotacoes, {}, date(2026, 10, 16))

    at = AppTest.from_function(script, default_timeout=60).run()
    assert not at.exception
    assert [i.value.split(":")[0] for i in at.info] == ["Satélite no núcleo"]
    assert at.dataframe[0].value.set_index("Ticker").at["NUC", "Peso alvo"] == pytest.approx(1.0)
