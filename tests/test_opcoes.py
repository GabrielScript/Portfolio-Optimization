import numpy as np
import pandas as pd
import pytest

from core import opcoes
from core.poder import ic_por_mes

MES = pd.Timestamp("2026-10-01")


def test_each_issuer_takes_its_most_traded_chain_for_levels_and_every_chain_for_the_put_share():
    cadeias = pd.DataFrame({
        "month": [MES] * 3, "underlying_isin": ["ITUB3", "ITUB4", "PETR4"], "underlying_issuer": ["ITUB", "ITUB", "PETR"],
        "iv_atm": [0.50, 0.30, 0.40], "skew": [0.10, 0.05, 0.02], "pc_ratio": [0.5, 0.25, 0.4],
        "contracts": [100, 300, 50], "n_days": [5, 5, 5]})
    sinais = opcoes.sinais_por_emissor(cadeias).set_index(["month", "issuer"])
    itub = sinais.loc[(MES, "ITUB")]
    assert itub["iv_atm"] == 0.30 and itub["skew"] == 0.05                   # ITUB4, the most traded chain
    assert itub["pc_ratio"] == pytest.approx((50 + 75) / 400)                 # puts over contracts of both chains
    assert sinais.loc[(MES, "PETR"), "pc_ratio"] == pytest.approx(0.4)


def test_the_signals_rank_so_that_higher_is_expected_better():
    linha = pd.DataFrame({"month": [MES], "issuer": ["PETR"], "iv_atm": [0.4], "skew": [0.1], "pc_ratio": [0.3]})
    pontos = opcoes.pontuar(linha).iloc[0]
    assert (pontos["S1"], pontos["S2"], pontos["S3"]) == (-0.4, -0.1, -0.3)


def test_the_partial_ic_removes_what_the_conservadora_score_already_knows():
    gerador = np.random.default_rng(3)
    meses = pd.date_range("2020-01-01", periods=40, freq="MS")
    papeis = [f"P{i}" for i in range(60)]
    nota = pd.DataFrame(gerador.normal(size=(40, 60)), index=meses, columns=papeis)
    retornos = nota + gerador.normal(scale=1.0, size=(40, 60))
    copia = nota + gerador.normal(scale=0.05, size=(40, 60))                   # a signal that only repeats the score
    assert ic_por_mes(copia, retornos).mean() > 0.4
    assert abs(opcoes.ic_parcial_por_mes(copia, retornos, nota).mean()) < 0.05
    novo = pd.DataFrame(gerador.normal(size=(40, 60)), index=meses, columns=papeis)
    retornos2 = nota + novo
    assert opcoes.ic_parcial_por_mes(novo, retornos2, nota).mean() > 0.4


def test_the_veto_drops_the_bottom_fifth_of_issuers_with_the_signal_only_with_20_or_more():
    pontos = pd.Series(np.arange(25, dtype=float), index=[f"E{i:02d}" for i in range(25)])
    candidatos = pd.Series(1.0, index=["E00X3", "E00X4", "E04X3", "E05X3", "SEMX3"])
    emissor = lambda papel: papel[:3]                                        # noqa: E731
    mantidos = opcoes.vetar(candidatos, pontos, emissor)
    assert list(mantidos.index) == ["E05X3", "SEMX3"]           # 5 bottom issuers out (both classes); no signal: kept
    poucos = pontos.iloc[:19]
    assert list(opcoes.vetar(candidatos, poucos, emissor).index) == list(candidatos.index)


def test_the_frozen_panel_keeps_only_chains_of_issuers_in_that_month_s_universe():
    from scripts.opcoes_painel import painel_no_universo
    cadeias = pd.DataFrame({"month": [MES, MES, MES], "underlying_isin": ["A", "B", "C"],
                            "underlying_issuer": ["PETR", "BOVA", "VALE"], "iv_atm": [0.3, 0.2, 0.4],
                            "skew": [0.1, 0.0, 0.2], "pc_ratio": [0.4, 0.5, 0.3], "contracts": [10, 10, 10],
                            "n_days": [5, 5, 5]})
    universo = pd.DataFrame({"month": [MES, MES, pd.Timestamp("2026-09-01")],
                             "security": ["PETR4", "PETR3", "VALE3"]})       # VALE only in another month
    assert opcoes_issuers(painel_no_universo(cadeias, universo)) == ["PETR"]


def opcoes_issuers(painel: pd.DataFrame) -> list[str]:
    return sorted(painel["issuer"])
