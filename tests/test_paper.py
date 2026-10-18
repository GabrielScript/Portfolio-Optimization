"""Paper trading (S5): curva do satélite registrado, rebalanceio devido e decisão do dia, sem rede."""

import pandas as pd
import pytest

from core.paper import curva_do_registro, rebalanceio_devido

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
