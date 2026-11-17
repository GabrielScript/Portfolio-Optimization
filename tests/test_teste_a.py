"""Tests for Test A (amendment of 2026-11-17): CVM material filings as a veto on the Composite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

JEVSCORE = Path(__file__).resolve().parents[3] / "b3-datalake" / "src" / "b3lake" / "jevscore.py"


@pytest.mark.skipif(not JEVSCORE.exists(), reason="the lake's source is not beside this repository")
def test_the_questions_are_read_verbatim_from_the_lake() -> None:
    """The amendment fixes `filings-v1` as the lake defines it: read from its source, never retyped."""
    from core.teste_a import perguntas_do_lago

    perguntas, versao = perguntas_do_lago(JEVSCORE)
    assert versao == "filings-v1"
    assert list(perguntas) == ["filing_tone", "filing_kind", "filing_intensity"]
    assert list(perguntas["filing_tone"]["criteria"]) == ["bad", "neutral", "good", "unclear"]
    assert len(perguntas["filing_intensity"]["criteria"]) == 4
    assert perguntas["filing_intensity"]["criteria"][2].startswith("Major")


def test_the_state_is_the_lake_filing_request_without_name_or_date() -> None:
    """The state carries category, subject, type and species only; a missing field is an empty string; identical
    states share one key, so each distinct subject is asked once."""
    from core.teste_a import chave_do_estado, estado_do_documento

    linha = {"categoria": "Fato Relevante", "assunto": "Aquisição da Empresa X", "tipo": None, "especie": "",
             "nome_companhia": "SECRETA S.A.", "data_entrega": "2015-03-02"}
    estado = estado_do_documento(linha)
    assert estado == {"filing": {"category": "Fato Relevante", "subject": "Aquisição da Empresa X", "type": "",
                                 "species": ""}}
    outra = dict(linha, nome_companhia="OUTRA S.A.", data_entrega="2020-01-01")
    assert chave_do_estado(estado_do_documento(outra)) == chave_do_estado(estado)
    assert chave_do_estado(estado_do_documento(dict(linha, assunto="Venda"))) != chave_do_estado(estado)


def test_only_material_filings_of_candidate_issuers_inside_the_window_are_kept() -> None:
    """Material categories only, delivered inside [inicio, fim], of an issuer that is a candidate in some month; a
    CVM code shared by two candidate prefixes (a renamed company) counts for both."""
    from core.teste_a import documentos_do_universo

    ipe = pd.DataFrame({
        "codigo_cvm": [1, 1, 1, 2, 3],
        "categoria": ["Fato Relevante", "Dados Econômico-Financeiros", "Comunicado ao Mercado", "Fato Relevante",
                      "Fato Relevante"],
        "assunto": ["Aquisição", "ITR", "Antigo", "Fora do universo", "Renomeada"],
        "tipo": [None] * 5, "especie": [None] * 5,
        "data_entrega": ["2015-03-02", "2015-03-02", "2009-01-05", "2015-03-02", "2016-07-01"],
    })
    setores = pd.DataFrame({"issuer": ["AAAA", "BBBB", "ELET", "AXIA"], "code_cvm": [1, 2, 3, 3]})
    universo = pd.DataFrame({"month": pd.to_datetime(["2015-03-01", "2016-07-01", "2026-01-01"]),
                             "security": ["AAAA3", "ELET3", "AXIA3"], "issuer": ["AAAA", "ELET", "AXIA"]})
    docs = documentos_do_universo(ipe, setores, universo, pd.Timestamp("2009-12-01"), pd.Timestamp("2026-10-16"))
    assert sorted(zip(docs["emissor"], docs["assunto"])) == [("AAAA", "Aquisição"), ("AXIA", "Renomeada"),
                                                             ("ELET", "Renomeada")]
    assert docs["data"].dtype.kind == "M"
    assert docs["chave"].str.len().eq(40).all()


def test_the_veto_window_is_the_21_sessions_before_the_decision() -> None:
    """Bad (P >= 0.5) and intensity >= 2.0, delivered from the 21st session before t up to, not including, t: a
    filing dated t may come after the close the engine trades at; a weekend filing before t counts."""
    from core.teste_a import emissores_vetados

    pregoes = pd.bdate_range("2020-01-01", periods=60)
    t = pregoes[43]
    assert t.day_name() == "Monday"
    docs = pd.DataFrame([
        ("DIAT", t, 0.9, 2.5),                                  # dated t: excluded
        ("D21A", pregoes[43 - 21], 0.9, 2.5),                   # first session of the window: included
        ("D22A", pregoes[43 - 22], 0.9, 2.5),                   # one session too early: excluded
        ("FDSM", t - pd.Timedelta(days=2), 0.9, 2.5),           # the Saturday before the Monday t: included
        ("PBAI", pregoes[38], 0.49, 3.0),                       # not bad enough
        ("IBAI", pregoes[38], 0.9, 1.99),                       # not strong enough
        ("LIMI", pregoes[38], 0.5, 2.0),                        # exactly on both thresholds: included
    ], columns=["emissor", "data", "p_ruim", "intensidade"])
    assert emissores_vetados(docs, pregoes, t) == {"D21A", "FDSM", "LIMI"}


def test_an_answer_becomes_bad_good_and_intensity_columns() -> None:
    """P(bad) and P(good) come from the tone's probabilities, the intensity from the score's expected level."""
    from core.teste_a import colunas_da_resposta

    respostas = {"filing_tone": {"type": "choice", "choice": "bad",
                                 "probabilities": {"bad": 0.7, "neutral": 0.2, "good": 0.05, "unclear": 0.05}},
                 "filing_kind": {"type": "choice", "choice": "deals", "probabilities": {"deals": 1.0}},
                 "filing_intensity": {"type": "score", "score": 2.3, "probabilities": {"0": 0, "1": 0.1}}}
    assert colunas_da_resposta(respostas) == {"p_ruim": 0.7, "p_bom": 0.05, "intensidade": 2.3, "tipo_do_fato": "deals"}
