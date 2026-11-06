"""The Composite top-5, a ghost strategy followed in the simulator (the user, 2026-11-06; DECISOES 290)."""

import pandas as pd
import pytest

from core import top5
from core.composto import estrategia_por_nota

SETORES = {"A.SA": "Bancos", "B.SA": "Bancos", "C.SA": "Energia", "D.SA": "Saúde", "E.SA": "Varejo", "F.SA": "Varejo",
           "G.SA": "Mineração", "H.SA": "Telecom", "I.SA": "Saneamento"}
NOTAS = pd.Series({"A.SA": 9.0, "B.SA": 8.0, "C.SA": 7.0, "D.SA": 6.0, "E.SA": 5.0, "F.SA": 4.0, "G.SA": 3.0,
                   "H.SA": 2.0, "I.SA": 1.0})


def _treino() -> pd.DataFrame:
    datas = pd.bdate_range("2024-01-01", periods=300)
    gerador = pd.Series(range(300), index=datas)
    return pd.DataFrame({ticker: 0.001 * ((gerador + posicao) % 7 - 3) for posicao, ticker in enumerate(NOTAS.index)},
                        index=datas)


def test_the_rule_takes_the_five_highest_scores_one_per_sector_at_20_percent_each():
    # 20% a name and the fixed 30% sector cap allow a single name per sector (DECISOES 290)
    assert (top5.NOMES, top5.POR_SETOR) == (5, 1)
    pesos = top5.estrategia(lambda mes, candidatos: NOTAS.reindex(candidatos), SETORES)(_treino())
    pesos = pesos[pesos > 0]
    assert sorted(pesos.index) == ["A.SA", "C.SA", "D.SA", "E.SA", "G.SA"]       # B and F share a sector with A and E
    assert pesos.to_numpy() == pytest.approx([0.2] * 5)


def test_the_rule_is_the_composites_with_five_names_and_one_per_sector():
    nota = lambda mes, candidatos: NOTAS.reindex(candidatos)       # noqa: E731
    esperado = estrategia_por_nota(nota, SETORES, nomes=5, por_setor=1)(_treino())
    pd.testing.assert_series_equal(top5.estrategia(nota, SETORES)(_treino()), esperado)


def _composto(*datas: str) -> pd.DataFrame:
    return pd.DataFrame({"data": pd.to_datetime([d for d in datas for _ in range(2)]),
                         "ticker": ["X", "Y"] * len(datas), "peso": 0.5})


def _pesos(data: pd.Timestamp) -> pd.Series:
    return pd.Series(0.2, index=[f"{data:%m%d}{n}.SA" for n in range(5)])


def test_the_first_record_carries_the_composites_record_in_force_and_trades_at_the_launch_close():
    # the 27/10 decision, recorded on 06/11 (the registration day): nothing before the launch is traded
    composto = _composto("2026-07-27", "2026-10-27")
    novas = top5.registros_devidos(composto, top5.vazio(), _pesos)
    assert list(novas.columns) == top5.COLUNAS
    assert set(novas["data"]) == {pd.Timestamp("2026-11-06")} == {top5.INICIO}
    assert set(novas["base"]) == {pd.Timestamp("2026-10-27")}
    assert sorted(novas["ticker"]) == [f"1027{n}" for n in range(5)]        # no ".SA", like the other records
    assert novas["peso"].tolist() == pytest.approx([0.2] * 5)


def test_each_later_composite_record_gives_one_top5_record_of_its_own_date_once():
    composto = _composto("2026-10-27", "2027-01-26")
    feito = top5.registros_devidos(_composto("2026-10-27"), top5.vazio(), _pesos)
    novas = top5.registros_devidos(composto, feito, _pesos)
    assert set(novas["data"]) == set(novas["base"]) == {pd.Timestamp("2027-01-26")}
    todas = pd.concat([feito, novas], ignore_index=True)
    assert top5.registros_devidos(composto, todas, _pesos).empty                 # nothing twice


def test_without_a_composite_record_there_is_nothing_to_record():
    assert top5.registros_devidos(_composto()[:0], top5.vazio(), _pesos).empty


def test_the_record_file_round_trips(tmp_path):
    arquivo = tmp_path / "top5.csv"
    assert top5.ler(arquivo).empty and list(top5.ler(arquivo).columns) == top5.COLUNAS
    novas = top5.registros_devidos(_composto("2026-10-27"), top5.vazio(), _pesos)
    top5.acrescentar(arquivo, novas)
    pd.testing.assert_frame_equal(top5.ler(arquivo), novas.reset_index(drop=True))
