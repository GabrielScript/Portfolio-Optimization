"""Buyback announcements, signal 2 of docs/superpowers/specs/2026-11-08-pead-recompra-design.md (section 4)."""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(("categoria", "tipo", "especie", "assunto", "esperado"), [
    ("Fato Relevante", "", "", "Programa de Recompra de Ações", True),
    ("Reunião da Administração", "Conselho de Administração", "Ata", "Aquisição de ações de própria emissão", True),
    ("Comunicado ao Mercado", "Outros", "", "Aquisição de ações de sua própria emissão", True),
    ("Assembleia", "AGE", "Ata", "Ações de emissão própria para tesouraria", True),
    ("Reunião da Administração", "Conselho de Administração", "Ata", "Ações de própria emissão", False),  # no pattern
    ("Fato Relevante", "", "", "Encerramento do Programa de Recompra de Ações", False),
    ("Reunião da Administração", "", "", "Cancelamento de ações em tesouraria||Encerramento do programa de recompra"
     "||Novo programa de recompra", True),
    ("Reunião da Administração", "", "", "Alienação de ações em tesouraria||recompra", False),
    ("Aviso aos Acionistas", "", "", "Programa de Recompra de Ações", False),          # category not in the set
    ("Fato Relevante", "", "", "Aumento de capital", False),
])
def test_a_buyback_row_is_a_program_text_in_the_four_categories_and_not_the_end_of_one(categoria, tipo, especie,
                                                                                          assunto, esperado):
    from core.recompra import e_recompra
    assert e_recompra(categoria, tipo, especie, assunto) is esperado


def _linha(emissor: str, referencia: str, entrega: str) -> dict:
    return {"issuer": emissor, "Data_Referencia": pd.Timestamp(referencia), "Data_Entrega": pd.Timestamp(entrega)}


def test_events_take_the_later_of_reference_and_first_delivery_enter_next_session_and_one_per_63_sessions():
    from core.recompra import eventos
    sessoes = pd.bdate_range("2020-01-01", "2020-12-31")
    linhas = pd.DataFrame([
        _linha("AAAA", "2020-02-03", "2020-02-05"),          # a = 05/02 (later than the reference): entry 06/02
        _linha("AAAA", "2020-02-03", "2020-02-04"),          # same document, earlier delivery: a = 04/02, entry 05/02
        _linha("AAAA", "2020-03-02", "2020-03-02"),          # 18 sessions after the kept entry: dropped
        _linha("AAAA", "2020-06-01", "2020-05-29"),          # delivery before the reference: a = the reference
        _linha("BBBB", "2020-01-06", "2020-02-20"),          # delivered 45 days late: dropped
        _linha("BBBB", "2020-12-31", "2020-12-31"),          # no session after it: dropped
    ])
    resultado = eventos(linhas, sessoes)
    assert list(resultado.columns) == ["issuer", "informacao", "entrada"]
    assert resultado.to_dict("list") == {
        "issuer": ["AAAA", "AAAA"],
        "informacao": [pd.Timestamp("2020-02-04"), pd.Timestamp("2020-06-01")],
        "entrada": [pd.Timestamp("2020-02-05"), pd.Timestamp("2020-06-02")]}


def test_an_event_63_sessions_after_the_kept_one_is_kept():
    from core.recompra import eventos
    sessoes = pd.bdate_range("2020-01-01", "2020-12-31")
    primeira, depois_62, depois_63 = sessoes[10], sessoes[10 + 62], sessoes[10 + 63]
    linhas = pd.DataFrame([_linha("AAAA", str(dia - pd.offsets.BDay(1)), str(dia - pd.offsets.BDay(1)))
                           for dia in (primeira, depois_62, depois_63)])
    assert list(eventos(linhas, sessoes)["entrada"]) == [primeira, depois_63]


def test_the_abnormal_return_is_the_issuers_classes_minus_the_months_universe_equal_weighted_with_prices_carried():
    from core.recompra import retornos_anormais
    sessoes = pd.bdate_range("2020-01-01", periods=12)
    precos = pd.DataFrame({"AAAA3": np.linspace(10.0, 12.0, 12), "AAAA4": np.linspace(10.0, 11.0, 12),
                           "BBBB3": np.full(12, 5.0), "CCCC3": np.r_[np.linspace(8.0, 6.0, 6), [np.nan] * 6],
                           "DDDD3": np.full(12, 3.0)}, index=sessoes)
    universo = pd.DataFrame({"month": pd.Timestamp("2020-01-01"),
                             "security": ["AAAA3", "AAAA4", "BBBB3", "CCCC3"]})          # DDDD3 not in the universe
    eventos = pd.DataFrame({"issuer": ["AAAA", "DDDD", "AAAA"],
                            "informacao": [sessoes[0], sessoes[0], sessoes[6]],
                            "entrada": [sessoes[1], sessoes[1], sessoes[7]]})
    resultado = retornos_anormais(eventos, precos, universo, janela=5)
    retorno = lambda papel, de, ate: precos[papel].ffill().iloc[ate] / precos[papel].iloc[de] - 1.0   # noqa: E731
    do_emissor = np.mean([retorno("AAAA3", 1, 6), retorno("AAAA4", 1, 6)])
    do_universo = np.mean([retorno(papel, 1, 6) for papel in ("AAAA3", "AAAA4", "BBBB3", "CCCC3")])
    # DDDD has no class in the universe; the second AAAA event ends after the last session
    assert resultado["issuer"].tolist() == ["AAAA"] and resultado["entrada"].tolist() == [sessoes[1]]
    assert resultado["anormal"].iloc[0] == pytest.approx(do_emissor - do_universo)
    assert retorno("CCCC3", 1, 6) == pytest.approx(6.0 / precos["CCCC3"].iloc[1] - 1.0)    # carried from its last price


def test_monthly_portfolio_is_the_mean_abnormal_return_of_the_events_entering_each_month():
    from core.recompra import carteira_mensal
    anormais = pd.DataFrame({"issuer": ["A", "B", "C"], "anormal": [0.02, 0.04, -0.01],
                             "entrada": pd.to_datetime(["2020-01-03", "2020-01-30", "2020-03-02"])})
    assert carteira_mensal(anormais).to_dict() == {pd.Timestamp("2020-01-01"): pytest.approx(0.03),
                                                   pd.Timestamp("2020-03-01"): pytest.approx(-0.01)}


def test_the_arm_adds_a_quarter_to_the_composite_percentile_of_issuers_with_a_buyback_in_the_63_sessions_before():
    import core.recompra as modulo
    sessoes = pd.bdate_range("2020-01-01", periods=200)
    treino = pd.DataFrame(0.0, index=sessoes, columns=["AAAA3", "BBBB3", "CCCC3", "DDDD3", "XXXX3"])
    decisao = sessoes[-1]
    eventos = pd.DataFrame({"issuer": ["AAAA", "CCCC", "DDDD", "XXXX"],
                            "informacao": [sessoes[-63], sessoes[-64], decisao, sessoes[-2]],
                            "entrada": [sessoes[-62], sessoes[-63], decisao, sessoes[-1]]})
    composto = pd.Series({"AAAA3": 1.0, "BBBB3": 2.0, "CCCC3": 3.0, "DDDD3": 4.0, "XXXX3": np.nan})
    vistos = {}

    def falsa(nota, setores):
        def _estrategia(janela):
            vistos["nota"] = nota(pd.Timestamp("2020-10-01"), list(janela.columns))
            return pd.Series(dtype=float)
        return _estrategia

    original = modulo.estrategia_por_nota
    modulo.estrategia_por_nota = falsa
    try:
        modulo.estrategia_recompra(lambda mes, candidatos: composto.reindex(candidatos), eventos, {})(treino)
    finally:
        modulo.estrategia_por_nota = original
    # AAAA: information at the window's first session (in); CCCC one before it (out); DDDD on the decision (out);
    # XXXX has no Composite score and stays out
    esperado = pd.Series({"AAAA3": 0.25 + 0.25, "BBBB3": 0.5, "CCCC3": 0.75, "DDDD3": 1.0, "XXXX3": np.nan})
    pd.testing.assert_series_equal(vistos["nota"], esperado, check_names=False)
