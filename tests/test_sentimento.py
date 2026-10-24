"""The screen's news, sentiment and filings panel (``core.sentimento``): information only, built from the lake's
forward log (Jev scores ``items-v1``) and the CVM IPE filings. Nothing here reaches the optimizer (DECISOES 124)."""

import pandas as pd
import pytest

from core.sentimento import notas_por_item


def _nota(chave: str, relevante: float, tom: dict, topico: dict, eleicao: float, intensidade: dict,
          empresa: dict | None = None) -> dict:
    empresa = empresa or {"bad": 0, "neutral": 0, "good": 0, "not_one_company": 1}
    return {"key": chave, "target": "item", "questions_version": "items-v1", "answers": {
        "relevant": {"type": "noul", "noul": relevante},
        "market_tone": {"type": "choice", "probabilities": tom},
        "topic": {"type": "choice", "probabilities": topico},
        "company_tone": {"type": "choice", "probabilities": empresa},
        "intensity": {"type": "score", "probabilities": intensidade},
        "election": {"type": "noul", "noul": eleicao}}}


TOPICO = {"election_politics": 0.8, "monetary_policy": 0.1, "fiscal_policy": 0.0, "company_specific": 0.0,
          "commodities_global": 0.0, "currency": 0.1, "personal_finance": 0.0, "other": 0.0}


def test_each_scored_item_becomes_one_row_of_probabilities():
    linhas = [_nota("a", 0.9, {"negative": 0.6, "neutral": 0.3, "positive": 0.1, "not_applicable": 0.0}, TOPICO, 0.95,
                    {"0": 0.2, "1": 0.3, "2": 0.4, "3": 0.1})]
    notas = notas_por_item(linhas)
    assert notas.loc["a", "relevante"] == pytest.approx(0.9)
    assert notas.loc["a", "tom"] == pytest.approx(0.1 - 0.6)
    assert notas.loc["a", "eleicao"] == pytest.approx(0.95)
    assert notas.loc["a", "forte"] == pytest.approx(0.5)            # P(Major) + P(Extraordinary)
    assert notas.loc["a", "election_politics"] == pytest.approx(0.8)


def test_the_daily_tone_is_the_mean_over_the_relevant_items_by_brasilia_date():
    from core.sentimento import tom_diario
    tom = {"negative": 0.0, "neutral": 0.0, "positive": 0.0, "not_applicable": 0.0}
    notas = notas_por_item([
        _nota("a", 0.9, {**tom, "positive": 0.8}, TOPICO, 0.9, {"0": 1, "1": 0, "2": 0, "3": 0}),
        _nota("b", 0.8, {**tom, "negative": 0.6}, TOPICO, 0.1, {"0": 0, "1": 0, "2": 1, "3": 0}),
        _nota("c", 0.1, {**tom, "negative": 1.0}, TOPICO, 0.0, {"0": 1, "1": 0, "2": 0, "3": 0})])
    itens = pd.DataFrame({"key": ["a", "b", "c", "d"],
                          "fetched_at": ["2026-10-24T01:00:00+00:00", "2026-10-23T15:00:00+00:00",
                                         "2026-10-23T16:00:00+00:00", "2026-10-24T12:00:00+00:00"]})
    dias = tom_diario(itens, notas)
    dia = dias.loc[pd.Timestamp("2026-10-23")]                # 01:00 UTC on 24/10 is 22:00 on 23/10 in Brasília
    assert (dia["itens"], dia["relevantes"]) == (3, 2)
    assert dia["tom"] == pytest.approx((0.8 - 0.6) / 2)
    assert dia["eleicao"] == pytest.approx(0.5)                 # share of the relevant items about the election
    assert dia["forte"] == pytest.approx(0.5)
    assert dias.loc[pd.Timestamp("2026-10-24"), "itens"] == 1    # "d" has no score yet: counted, no tone
    assert pd.isna(dias.loc[pd.Timestamp("2026-10-24"), "tom"])


def test_material_filings_of_the_apps_stocks_are_listed_by_ticker_newest_first():
    from datetime import date
    from core.sentimento import fatos_recentes
    ipe = pd.DataFrame({
        "codigo_cvm": [9512, 9512, 4170, 1],
        "nome_companhia": ["PETROBRAS", "PETROBRAS", "VALE", "OUTRA"],
        "categoria": ["Fato Relevante", "Assembleia", "Comunicado ao Mercado", "Fato Relevante"],
        "assunto": ["Produção 3T26", "AGO", "Esclarecimento", "x"],
        "data_entrega": [date(2026, 10, 10), date(2026, 10, 11), date(2026, 10, 15), date(2026, 10, 15)],
        "link_download": ["l1", "l2", "l3", "l4"]})
    setores = pd.DataFrame({"issuer": ["PETR", "VALE"], "code_cvm": [9512, 4170]})
    fatos = fatos_recentes(ipe, setores, ["PETR4.SA", "VALE3.SA", "ITUB4.SA"], desde=date(2026, 10, 1))
    assert fatos["ticker"].tolist() == ["VALE3", "PETR4"]                 # material only, app stocks only
    assert fatos.iloc[1][["categoria", "assunto", "link"]].tolist() == ["Fato Relevante", "Produção 3T26", "l1"]


def test_headlines_are_the_relevant_scored_items_newest_first_with_their_main_topic():
    from core.sentimento import manchetes
    tom = {"negative": 0.0, "neutral": 0.0, "positive": 0.7, "not_applicable": 0.0}
    notas = notas_por_item([_nota("a", 0.9, tom, TOPICO, 0.9, {"0": 1, "1": 0, "2": 0, "3": 0}),
                            _nota("b", 0.2, tom, TOPICO, 0.0, {"0": 1, "1": 0, "2": 0, "3": 0}),
                            _nota("c", 0.7, tom, {**TOPICO, "election_politics": 0.0, "currency": 0.9}, 0.0,
                                  {"0": 1, "1": 0, "2": 0, "3": 0})])
    itens = pd.DataFrame({"key": ["a", "b", "c"], "source": ["exame", "reddit", "infomoney"],
                          "title": ["Pesquisa", "Dúvida", "Dólar cai"], "url": ["u1", "u2", "u3"],
                          "fetched_at": ["2026-10-23T15:00:00+00:00", "2026-10-23T16:00:00+00:00",
                                         "2026-10-23T17:00:00+00:00"]})
    lista = manchetes(itens, notas, quantas=10)
    assert lista["titulo"].tolist() == ["Dólar cai", "Pesquisa"]
    assert lista["tema"].tolist() == ["Câmbio", "Eleição e política"]
    assert lista["tom"].tolist() == pytest.approx([0.7, 0.7])


def test_the_log_is_read_from_the_lakes_jsonl_files_skipping_a_partial_line(tmp_path):
    import json
    from datetime import date
    from core.sentimento import ler_log
    (tmp_path / "raw" / "items").mkdir(parents=True)
    (tmp_path / "scores").mkdir()
    item = {"key": "a", "source": "exame", "kind": "feed", "fetched_at": "2026-10-23T15:00:00+00:00",
            "title": "Pesquisa", "summary": "texto", "url": "u1"}
    (tmp_path / "raw" / "items" / "2026-10-22.jsonl").write_text(json.dumps({**item, "key": "velho"}) + "\n",
                                                                   encoding="utf-8")
    (tmp_path / "raw" / "items" / "2026-10-23.jsonl").write_text(json.dumps(item) + "\n{\"key\": \"cort",
                                                                   encoding="utf-8")
    tom = {"negative": 0.0, "neutral": 1.0, "positive": 0.0, "not_applicable": 0.0}
    (tmp_path / "scores" / "2026-10-24.jsonl").write_text(
        json.dumps(_nota("a", 0.9, tom, TOPICO, 0.0, {"0": 1, "1": 0, "2": 0, "3": 0})) + "\n", encoding="utf-8")
    itens, notas = ler_log(tmp_path, desde=date(2026, 10, 23))
    assert itens["key"].tolist() == ["a"]
    assert "summary" not in itens.columns
    assert notas.index.tolist() == ["a"]


def test_with_no_score_yet_the_days_count_the_items_and_have_no_tone():
    from core.sentimento import tom_diario
    itens = pd.DataFrame({"key": ["a"], "fetched_at": ["2026-10-23T15:00:00+00:00"]})
    dias = tom_diario(itens, notas_por_item([]))
    assert dias.loc[pd.Timestamp("2026-10-23"), "itens"] == 1
    assert pd.isna(dias.loc[pd.Timestamp("2026-10-23"), "tom"])


def test_a_cvm_code_with_two_issuer_prefixes_maps_to_the_one_the_app_lists():
    # ELET3 became AXIA3 (assets.py, 2026-10): the lake keeps both prefixes under one CVM code
    from datetime import date
    from core.sentimento import fatos_recentes
    ipe = pd.DataFrame({"codigo_cvm": [2437], "nome_companhia": ["AXIA"], "categoria": ["Fato Relevante"],
                        "assunto": ["x"], "data_entrega": [date(2026, 10, 10)], "link_download": ["l1"]})
    setores = pd.DataFrame({"issuer": ["ELET", "AXIA"], "code_cvm": [2437, 2437]})
    assert fatos_recentes(ipe, setores, ["AXIA3.SA"], desde=date(2026, 10, 1))["ticker"].tolist() == ["AXIA3"]


def test_the_election_timing_counts_the_days_to_the_runoff_and_names_the_studied_window():
    from datetime import date
    from core.sentimento import janela_eleitoral
    antes = janela_eleitoral(date(2026, 10, 23))
    assert (antes.turno, antes.dias_ate) == (date(2026, 10, 25), 2)
    assert (antes.janela_inicio, antes.janela_fim) == (date(2026, 10, 26), date(2026, 10, 30))
    assert not antes.dentro
    assert janela_eleitoral(date(2026, 10, 28)).dentro
    assert janela_eleitoral(date(2026, 11, 3)) is None                    # after the window: nothing to show
