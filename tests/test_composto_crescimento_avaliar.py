"""The saved curves of the composite reading (scripts/composto_crescimento_avaliar.py), the bench's input."""

import pandas as pd


def test_the_curves_frame_has_one_row_per_phase_and_session_with_the_arm_the_conservadora_and_bova11():
    from scripts.composto_crescimento_avaliar import ARM, curves_frame
    from scripts.qpo_avaliar import CONSERVADORA
    datas = pd.to_datetime(["2010-02-09", "2010-02-10"])
    fase0 = {ARM: pd.Series([1.0, 1.1], index=datas), CONSERVADORA: pd.Series([1.0, 1.2], index=datas),
             "BOVA11": pd.Series([1.0, 0.9], index=datas), "IVVB11": pd.Series([1.0, 1.0], index=datas)}
    quadro = curves_frame([(0, fase0), (21, fase0)])
    assert list(quadro.columns) == ["fase", "date", "Composto", "Conservadora", "BOVA11"]
    assert quadro["fase"].tolist() == [0, 0, 21, 21]
    assert quadro["Conservadora"].tolist() == [1.0, 1.2, 1.0, 1.2]


def test_the_phase_header_reads_refusals_and_annual_turnover_of_real_backtest_results():
    from core.backtest import ResultadoBacktest
    from scripts.composto_crescimento_avaliar import phase_header
    datas = pd.bdate_range("2020-01-01", periods=253)

    def resultado(giro: float, recusas: dict) -> ResultadoBacktest:
        return ResultadoBacktest(nome="x", metodologia="m", valores=pd.Series(1.0, index=datas),
                                 pesos_alvo=pd.DataFrame(), turnover=pd.Series([giro], index=datas[:1]),
                                 custos_totais=0.0, recusas=recusas)

    texto = phase_header(21, datas, resultado(1.5, {"2020-01-02": ()}), resultado(0.5, {}))
    assert texto == (f"=== Phase 21: 01/01/2020 to {datas[-1]:%d/%m/%Y}; refusals 1 (Conservadora 0); annual turnover 1.50 "
                     "(Conservadora 0.50) ===")
