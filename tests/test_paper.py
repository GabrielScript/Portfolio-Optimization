"""Paper trading (S5): curva do satélite registrado, rebalanceio devido e decisão do dia, sem rede."""

import pandas as pd
import pytest

from core.paper import Acao, curva_do_registro, decidir_do_dia, rebalanceio_devido

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
