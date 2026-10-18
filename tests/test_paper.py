"""Paper trading (S5): curva do satélite registrado, rebalanceio devido e decisão do dia, sem rede."""

import numpy as np
import pandas as pd
import pytest

from core.paper import (Acao, curva_do_registro, curva_liquida, decidir_do_dia, item_3_cumprido, ler_o_paper,
                        rebalanceio_devido)

DATAS = pd.bdate_range("2026-10-12", periods=6)


def _registro(*linhas: tuple[str, str, float]) -> pd.DataFrame:
    """(data, ticker, peso) no formato do ``paper/carteiras.csv`` lido (preço não entra na curva)."""
    return pd.DataFrame([(pd.Timestamp(d), t, p, 1.0) for d, t, p in linhas], columns=["data", "ticker", "peso", "preco"])


def test_curva_segura_entre_registros_e_troca_os_pesos_no_fechamento_do_dia_registrado():
    # 13/10: 50% A + 50% B; A dobra até 15/10 (B parado) -> 1,5; no fechamento de 15/10 vira 100% B, que sobe 10%
    precos = pd.DataFrame({"A": [10.0, 10.0, 15.0, 20.0, 40.0, 40.0], "B": [5.0, 5.0, 5.0, 5.0, 5.5, 5.5]},
                          index=DATAS)
    registro = _registro(("2026-10-13", "A", 0.5), ("2026-10-13", "B", 0.5), ("2026-10-15", "B", 1.0))
    curva = curva_do_registro(registro, precos)
    assert curva.index[0] == DATAS[1]
    assert curva.tolist() == pytest.approx([1.0, 1.25, 1.5, 1.65, 1.65])


def test_papel_registrado_que_sumiu_do_arquivo_do_lago_e_erro_claro():
    # o export do app só traz quem negociou nos últimos 15 dias: uma deslistada some e a curva não pode inventar
    precos = pd.DataFrame({"A": [10.0] * 6}, index=DATAS)
    with pytest.raises(ValueError, match="XPTO3"):
        curva_do_registro(_registro(("2026-10-13", "A", 0.5), ("2026-10-13", "XPTO3", 0.5)), precos)


def test_rebalanceio_devido_conta_pregoes_do_lago_depois_do_ultimo_registro():
    # registro no 1º pregão; o próximo vem 63 pregões depois (como no motor), não 63 dias corridos
    calendario = pd.bdate_range("2026-01-02", periods=100)
    assert rebalanceio_devido(None, calendario[:5])
    assert not rebalanceio_devido(calendario[0], calendario[:63])        # 62 pregões depois do registro
    assert rebalanceio_devido(calendario[0], calendario[:64])            # 63


def _curva(valores: list[float]) -> pd.Series:
    return pd.Series(valores, index=pd.bdate_range("2026-07-01", periods=len(valores)))


def test_perda_de_5_por_cento_no_dia_congela_mesmo_com_rebalanceio_e_stop_devidos():
    # satélite 100 -> 70 (razão −30% contra o BOVA11 parado), o último pregão −6%: nenhuma ordem amanhã, nem o stop
    satelite, bova = _curva([100.0, 80.0, 74.5, 70.0]), _curva([100.0] * 4)
    assert decidir_do_dia(satelite, bova, devido=True, parado=False) is Acao.CONGELADO


def test_razao_25_por_cento_abaixo_do_pico_no_dia_de_rebalanceio_dispara_o_stop():
    # queda de 26% contra o BOVA11 parado, nenhum pregão perto de −5%: o satélite inteiro vai para o BOVA11
    satelite, bova = _curva([100.0, 97.0, 94.0, 91.0, 88.0, 85.0, 82.0, 79.0, 76.0, 74.0]), _curva([100.0] * 10)
    assert decidir_do_dia(satelite, bova, devido=True, parado=False) is Acao.STOP


def test_stop_ja_disparado_fica_parado_mesmo_com_rebalanceio_devido():
    # depois do stop o satélite é 100% BOVA11 e só volta com pré-registro novo: nem rebalanceio, nem stop de novo
    satelite, bova = _curva([100.0, 97.0, 94.0, 91.0, 88.0, 85.0, 82.0, 79.0, 76.0, 74.0]), _curva([100.0] * 10)
    assert decidir_do_dia(satelite, bova, devido=True, parado=True) is Acao.PARADO


def test_rebalanceio_devido_sem_stop_nem_congelamento_registra_a_carteira_nova():
    satelite, bova = _curva([100.0, 101.0, 99.0, 102.0]), _curva([100.0, 100.5, 99.5, 101.0])
    assert decidir_do_dia(satelite, bova, devido=True, parado=False) is Acao.REBALANCEAR


def test_stop_so_e_conferido_em_dia_de_rebalanceio():
    # a razão já caiu 26%, mas faltam pregões para o rebalanceio: segura a carteira (o stop espera o dia devido)
    satelite, bova = _curva([100.0, 97.0, 94.0, 91.0, 88.0, 85.0, 82.0, 79.0, 76.0, 74.0]), _curva([100.0] * 10)
    assert decidir_do_dia(satelite, bova, devido=False, parado=False) is Acao.AGUARDAR


def test_sem_registro_o_primeiro_dia_e_rebalanceio():
    # antes da 1ª carteira as curvas são vazias: nada congela nem para, e o rebalanceio é devido
    vazia = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    assert decidir_do_dia(vazia, vazia, devido=True, parado=False) is Acao.REBALANCEAR


def test_curva_liquida_paga_10_bps_por_lado_sobre_o_giro_de_cada_registro():
    # 13/10: monta 100% A (giro 1); A dobra até 15/10; no fechamento de 15/10 vende A e compra B (giro 2)
    precos = pd.DataFrame({"A": [10.0, 10.0, 15.0, 20.0, 20.0, 20.0], "B": [5.0, 5.0, 5.0, 5.0, 5.5, 5.5]},
                          index=DATAS)
    registro = _registro(("2026-10-13", "A", 1.0), ("2026-10-15", "B", 1.0))
    liquida = curva_liquida(registro, precos, custo_bps=10.0)
    assert liquida.tolist() == pytest.approx([0.999, 1.5 * 0.999, 2.0 * 0.999 * 0.998, 2.2 * 0.999 * 0.998,
                                              2.2 * 0.999 * 0.998])


def test_giro_do_rebalanceio_conta_os_pesos_ja_derivados_pelos_precos():
    # 50% A + 50% B; A dobra (vira 2/3 A, 1/3 B); de volta a 50/50: giro 1/6 + 1/6 = 1/3, não 0
    precos = pd.DataFrame({"A": [10.0, 10.0, 20.0, 20.0, 20.0, 20.0], "B": [5.0] * 6}, index=DATAS)
    registro = _registro(("2026-10-13", "A", 0.5), ("2026-10-13", "B", 0.5),
                         ("2026-10-14", "A", 0.5), ("2026-10-14", "B", 0.5))
    liquida = curva_liquida(registro, precos, custo_bps=10.0)
    assert liquida.loc["2026-10-14"] == pytest.approx(1.5 * 0.999 * (1 - 0.001 / 3))


def _lago(satelite: list[float], bova: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"A": satelite, "BOVA11": bova}, index=pd.bdate_range("2026-10-16", periods=len(satelite)))


def test_leitura_mede_o_satelite_liquido_contra_o_bova11_e_o_cdi_nos_mesmos_pregoes():
    # 1 registro (100% A) no 1º pregão; A sobe 10% e o BOVA11 5% em 4 pregões; CDI de 0,05% ao dia
    precos = _lago([10.0, 10.2, 10.5, 10.8, 11.0], [100.0, 101.0, 102.0, 104.0, 105.0])
    registro = _registro((str(precos.index[0].date()), "A", 1.0))
    leitura = ler_o_paper(registro, precos, pd.Series(0.0005, index=precos.index))
    assert (leitura.pregoes, leitura.satelite, leitura.bova, leitura.cdi, leitura.comparacao) == (
        4, pytest.approx(1.1 * 0.999 - 1), pytest.approx(0.05), pytest.approx(1.0005 ** 4 - 1), None)


def test_cdi_do_ultimo_pregao_ainda_nao_publicado_repete_o_do_dia_anterior():
    # às 8h30 o SGS 12 ainda não tem o CDI do último fechamento: o CDI do período não pode sumir por isso
    precos = _lago([10.0, 10.2, 10.5, 10.8, 11.0], [100.0, 101.0, 102.0, 104.0, 105.0])
    registro = _registro((str(precos.index[0].date()), "A", 1.0))
    leitura = ler_o_paper(registro, precos, pd.Series(0.0005, index=precos.index[:-1]))
    assert leitura.cdi == pytest.approx(1.0005 ** 4 - 1)


def test_item_3_do_criterio_sai_so_com_252_pregoes_e_na_janela_fixa():
    # o satélite rende 0,02%/dia e o BOVA11 0,01% até o 252º pregão; depois o satélite cai 20%: vale a janela fixa
    n = 300
    satelite = 10.0 * 1.0002 ** np.arange(n + 1)
    satelite[260:] *= 0.8
    precos = _lago(list(satelite), list(100.0 * 1.0001 ** np.arange(n + 1)))
    registro = _registro((str(precos.index[0].date()), "A", 1.0))
    seis_meses = ler_o_paper(registro, precos, None, ate=126)
    um_ano = ler_o_paper(registro, precos, None, ate=252)
    assert (seis_meses.pregoes, item_3_cumprido(seis_meses), um_ano.pregoes, item_3_cumprido(um_ano)) == (
        126, None, 252, True)


def test_leitura_ate_hoje_depois_dos_252_pregoes_nao_da_veredito():
    # no 300º pregão o satélite já caiu 20% (depois do 260º): a leitura até hoje não decide, só a janela fixa de 252
    n = 300
    satelite = 10.0 * 1.0002 ** np.arange(n + 1)
    satelite[260:] *= 0.8
    precos = _lago(list(satelite), list(100.0 * 1.0001 ** np.arange(n + 1)))
    ate_hoje = ler_o_paper(_registro((str(precos.index[0].date()), "A", 1.0)), precos, None)
    assert (ate_hoje.pregoes, item_3_cumprido(ate_hoje)) == (300, None)


def test_stop_disparado_reprova_o_item_3_mesmo_com_o_satelite_a_frente():
    # o satélite subiu 0,1%/dia até o stop do 63º pregão (registro de 100% BOVA11) e ficou à frente em 252 pregões
    ruido = 1.0 + np.random.default_rng(3).normal(0.0, 0.01, 253)
    precos = _lago(list(10.0 * 1.001 ** np.arange(253)), list(100.0 * np.cumprod(ruido) / ruido[0]))
    registro = _registro((str(precos.index[0].date()), "A", 1.0), (str(precos.index[63].date()), "BOVA11", 1.0))
    leitura = ler_o_paper(registro, precos, None, ate=252)
    assert (leitura.stop, leitura.satelite > leitura.bova, item_3_cumprido(leitura)) == (precos.index[63], True, False)


def test_stop_com_ruido_de_ponto_flutuante_no_peso_ainda_e_o_stop():
    # 100% BOVA11 gravado como 0,9999999999: a rotina já o trata como stop (fica parada), a leitura também tem de tratar
    precos = _lago([10.0, 10.0, 10.0, 10.0], [100.0, 100.0, 100.0, 100.0])
    registro = _registro((str(precos.index[0].date()), "A", 1.0), (str(precos.index[2].date()), "BOVA11", 1 - 1e-10))
    assert ler_o_paper(registro, precos, None).stop == precos.index[2]


def test_leitura_conta_os_registros_e_os_pregoes_de_perda_de_5_por_cento_do_satelite():
    # 2 registros; o satélite perde 6% no 2º pregão e 4,9% no 4º: um pregão congelado
    precos = _lago([10.0, 10.0, 9.4, 9.4, 8.9394, 9.0], [100.0, 101.0, 100.0, 99.0, 98.0, 99.0])
    registro = _registro((str(precos.index[0].date()), "A", 1.0), (str(precos.index[3].date()), "A", 1.0))
    leitura = ler_o_paper(registro, precos, None)
    assert (leitura.registros, leitura.congelados) == (2, 1)


def test_p_de_superar_o_bova11_e_o_psr_da_diferenca_diaria():
    # o satélite ganha do BOVA11 0,2% por pregão em média com ruído pequeno: P perto de 1; o contrário, perto de 0
    gerador = np.random.default_rng(5)
    bova = 100.0 * np.cumprod(1.0 + gerador.normal(0.0, 0.01, 120))
    satelite = bova * np.cumprod(1.0 + gerador.normal(0.002, 0.002, 120))
    ganha = ler_o_paper(_registro(("2026-10-16", "A", 1.0)), _lago(list(satelite), list(bova)), None)
    perde = ler_o_paper(_registro(("2026-10-16", "A", 1.0)), _lago(list(bova), list(satelite)), None)
    assert (ganha.p_supera > 0.99, perde.p_supera < 0.01) == (True, True)
