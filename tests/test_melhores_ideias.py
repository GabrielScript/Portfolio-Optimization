"""Fund managers' best ideas from the CVM CDA (docs/superpowers/specs/2026-10-29-cda-best-ideas-design.md)."""

import numpy as np
import pandas as pd
import pytest

from core.melhores_ideias import carteira_usada, ideias_por_carteira, painel_de_ideias, publico_a_partir, referencia

T = pd.Timestamp


def _posicoes(linhas: list[tuple]) -> pd.DataFrame:
    """(cnpj, month-end, ticker, value[, confid_until]) rows."""
    linhas = [(*linha, None) if len(linha) == 4 else linha for linha in linhas]
    quadro = pd.DataFrame(linhas, columns=["cnpj", "month", "ticker", "value", "confid_until"])
    return quadro.assign(month=pd.to_datetime(quadro["month"]), confid_until=pd.to_datetime(quadro["confid_until"]))


def test_public_position_is_known_two_months_after_and_a_confidential_one_after_its_release_plus_10_days():
    posicoes = _posicoes([("A", "2015-01-31", "PETR4", 1.0), ("A", "2015-01-31", "VALE3", 1.0, "2015-04-30"),
                          ("B", "2015-01-31", "PETR4", 1.0), ("C", "2015-01-31", "PETR4", 1.0, "2015-04-20")])
    publico = publico_a_partir(posicoes)
    assert publico[("A", T("2015-01-31"))] == T("2015-06-01")     # 30/04 + 10 days = 10/05 -> June
    assert publico[("B", T("2015-01-31"))] == T("2015-03-01")
    assert publico[("C", T("2015-01-31"))] == T("2015-05-01")     # 20/04 + 10 days = 30/04 -> May


def test_each_fund_uses_its_latest_fully_public_portfolio_within_12_months_and_before_the_cut():
    publico = pd.Series({("A", T("2015-01-31")): T("2015-03-01"), ("A", T("2015-02-28")): T("2015-07-01"),
                         ("B", T("2014-01-31")): T("2014-03-01"), ("C", T("2026-05-31")): T("2026-07-01"),
                         ("C", T("2026-04-30")): T("2026-06-01")})
    assert carteira_usada(publico, T("2015-05-01"), corte=T("2026-04-30")).to_dict() == {"A": T("2015-01-31")}
    assert carteira_usada(publico, T("2015-08-01"), corte=T("2026-04-30"))["A"] == T("2015-02-28")
    assert carteira_usada(publico, T("2026-09-01"), corte=T("2026-04-30")).to_dict() == {"C": T("2026-04-30")}


def _mercado(emissores: list[str], mes_da_carteira: str = "2015-01-31") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal caps for ``emissores`` at the row after the portfolio month, and the universe of that month."""
    linha = T(mes_da_carteira) + pd.offsets.MonthBegin(1)
    caps = pd.DataFrame({"issuer": emissores, "month": linha, "market_cap": 100.0})
    universo = pd.DataFrame({"month": linha, "security": [f"{e}3" for e in emissores]})
    return caps, universo


EMISSORES = [f"E{i:03d}" for i in range(20)]


def _fundo(cnpj: str, pesos: dict[str, float], mes: str = "2015-01-31") -> list[tuple]:
    return [(cnpj, mes, f"{emissor}3", valor) for emissor, valor in pesos.items()]


def test_best_ideas_are_the_three_largest_active_weights_and_classes_of_an_issuer_are_summed():
    pesos = {e: 1e6 for e in EMISSORES[:10]} | {"E000": 5e6, "E001": 4e6, "E002": 3e6}
    linhas = _fundo("A", pesos) + [("A", "2015-01-31", "E003" + "4", 2.5e6)]   # E003: 1 + 2.5 million (two classes)
    navs = pd.DataFrame({"cnpj": ["A"], "month": [T("2015-01-31")], "nav": [30e6]})
    caps, universo = _mercado(EMISSORES)
    ideias = ideias_por_carteira(_posicoes(linhas), navs, caps, universo)
    linha = ideias.loc[("A", T("2015-01-31"))]
    assert bool(linha["qualifica"]) and list(linha["ideias"]) == ["E000", "E001", "E003"]


ATIVO = {e: 0.5e6 for e in EMISSORES[:12]} | {"E000": 10e6}             # 15.5 million in 12 issuers, active share 0.6
CASOS = {"qualifies": (ATIVO, 20e6),
         "nav below 10 million": ({e: v / 2 for e, v in ATIVO.items()}, 9e6),
         "stocks below 67% of nav": (ATIVO, 15e6 / 0.6),
         "fewer than 10 issuers": ({e: 2e6 for e in EMISSORES[:9]}, 20e6),
         "closet indexer": ({e: 1e6 for e in EMISSORES}, 20e6)}            # the benchmark itself: active share 0


@pytest.mark.parametrize("caso", list(CASOS))
def test_each_rule_alone_stops_a_fund_from_qualifying(caso):
    pesos, nav = CASOS[caso]
    caps, universo = _mercado(EMISSORES)
    navs = pd.DataFrame({"cnpj": ["A"], "month": [T("2015-01-31")], "nav": [nav]})
    ideias = ideias_por_carteira(_posicoes(_fundo("A", pesos)), navs, caps, universo)
    assert bool(ideias.loc[("A", T("2015-01-31")), "qualifica"]) == (caso == "qualifies")


def test_benchmark_of_a_portfolio_month_is_the_cap_weight_of_the_next_months_universe_issuers():
    linha = T("2015-02-01")
    caps = pd.DataFrame({"issuer": ["AAAA", "BBBB", "CCCC"], "month": linha, "market_cap": [300.0, 100.0, 600.0]})
    universo = pd.DataFrame({"month": linha, "security": ["AAAA3", "AAAA4", "BBBB3"]})   # CCCC out of the universe
    bench = referencia(caps, universo).set_index(["month", "issuer"])["w_b"]
    assert bench.to_dict() == {(T("2015-01-31"), "AAAA"): pytest.approx(0.75), (T("2015-01-31"), "BBBB"): pytest.approx(0.25)}


def test_tickers_without_a_market_cap_are_dropped():
    caps, universo = _mercado(EMISSORES)
    pesos = {e: 1.0 for e in EMISSORES[:10]} | {"E000": 3.0, "E001": 2.0, "E002": 1.5}
    linhas = [(c, m, t, v * 1e6) for c, m, t, v in _fundo("A", pesos)] + [("A", "2015-01-31", "AAPL34", 50e6)]
    navs = pd.DataFrame({"cnpj": ["A"], "month": [T("2015-01-31")], "nav": [100e6]})
    ideias = ideias_por_carteira(_posicoes(linhas), navs, caps, universo)
    assert list(ideias.loc[("A", T("2015-01-31")), "ideias"]) == ["E000", "E001", "E002"]


def test_score_is_the_share_of_qualifying_funds_naming_the_issuer_among_their_best_ideas():
    ideias = pd.DataFrame({"qualifica": [True, True, False],
                           "ideias": [("E000", "E001", "E002"), ("E000", "E005", "E006"), ("E009", "E008", "E007")],
                           "pesos": [{"E000": 0.30, "E001": 0.05}, {"E000": 0.10}, {"E009": 0.9}]},
                          index=pd.MultiIndex.from_tuples([("A", T("2015-01-31")), ("B", T("2015-01-31")),
                                                           ("C", T("2015-01-31"))]))
    publico = pd.Series(T("2015-03-01"), index=ideias.index)
    universo = pd.DataFrame({"month": T("2015-06-01"), "security": [f"{e}3" for e in EMISSORES]})
    bench = pd.DataFrame({"month": T("2015-01-31"), "issuer": ["E000", "E001"], "w_b": [0.05, 0.10]})
    painel = painel_de_ideias(ideias, publico, bench, universo, [T("2015-06-01")], corte=T("2026-04-30"))
    linha = painel.set_index("issuer")
    assert linha.loc["E000", "score"] == pytest.approx(1.0)
    assert linha.loc["E005", "score"] == pytest.approx(0.5)
    assert linha.loc["E009", "score"] == pytest.approx(0.0)              # named only by a non-qualifying fund
    assert set(painel["qualifying"]) == {2} and len(painel) == len(EMISSORES)
    assert np.isfinite(painel["score"]).all()
    assert linha.loc["E000", "consensus"] == pytest.approx(0.20 - 0.05)  # mean w_f - mean w_b, 2 qualifying funds
    assert linha.loc["E001", "consensus"] == pytest.approx(0.025 - 0.10)
    assert linha.loc["E009", "consensus"] == pytest.approx(0.0)
