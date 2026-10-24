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
        "link_download": [f"https://www.rad.cvm.gov.br/{n}" for n in ("l1", "l2", "l3", "l4")]})
    setores = pd.DataFrame({"issuer": ["PETR", "VALE"], "code_cvm": [9512, 4170]})
    fatos = fatos_recentes(ipe, setores, ["PETR4.SA", "VALE3.SA", "ITUB4.SA"], desde=date(2026, 10, 1))
    assert fatos["ticker"].tolist() == ["VALE3", "PETR4"]                 # material only, app stocks only
    assert fatos.iloc[1][["categoria", "assunto", "link"]].tolist() == ["Fato Relevante", "Produção 3T26",
                                                                       "https://www.rad.cvm.gov.br/l1"]


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


def test_a_headline_link_that_is_not_http_or_https_is_dropped():
    # the links come from third-party feeds: a javascript: or data: URL must never become a clickable link
    from core.sentimento import manchetes
    tom = {"negative": 0.0, "neutral": 1.0, "positive": 0.0, "not_applicable": 0.0}
    notas = notas_por_item([_nota(chave, 0.9, tom, TOPICO, 0.0, {"0": 1, "1": 0, "2": 0, "3": 0})
                            for chave in ("a", "b", "c")])
    itens = pd.DataFrame({"key": ["a", "b", "c"], "source": ["x"] * 3, "title": ["A", "B", "C"],
                          "url": ["https://exame.com/a", "javascript:alert(1)", "data:text/html,oi"],
                          "fetched_at": ["2026-10-23T17:00:00+00:00", "2026-10-23T16:00:00+00:00",
                                         "2026-10-23T15:00:00+00:00"]})
    links = manchetes(itens, notas)["link"].tolist()
    assert links[0] == "https://exame.com/a"
    assert pd.isna(links[1]) and pd.isna(links[2])


def test_the_log_reader_skips_stray_file_names_and_score_files_older_than_the_window(tmp_path):
    # review finding 7: a stray file must not break the panel, and the 10-minute refresh must not parse the whole
    # score history (an item's score is written on or after its fetch day, so older score files hold none)
    import json
    from datetime import date
    from core.sentimento import ler_log
    (tmp_path / "raw" / "items").mkdir(parents=True)
    (tmp_path / "scores").mkdir()
    item = {"key": "a", "source": "exame", "kind": "feed", "fetched_at": "2026-10-23T15:00:00+00:00",
            "title": "Pesquisa", "summary": "texto", "url": "https://exame.com/a"}
    (tmp_path / "raw" / "items" / "2026-10-23.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")
    (tmp_path / "raw" / "items" / "backup.jsonl").write_text("{}\n", encoding="utf-8")
    tom = {"negative": 0.0, "neutral": 1.0, "positive": 0.0, "not_applicable": 0.0}
    velha = _nota("velho", 0.9, tom, TOPICO, 0.0, {"0": 1, "1": 0, "2": 0, "3": 0})
    (tmp_path / "scores" / "2026-10-01.jsonl").write_text(json.dumps(velha) + "\n", encoding="utf-8")
    (tmp_path / "scores" / "2026-10-23.part.jsonl").write_text("{}\n", encoding="utf-8")
    itens, notas = ler_log(tmp_path, desde=date(2026, 10, 23))
    assert itens["key"].tolist() == ["a"]
    assert "velho" not in notas.index


def test_the_last_filing_of_each_stock_keeps_its_own_date_and_subject_and_an_empty_subject_reads_blank():
    # review finding 4: groupby().last() skipped the NaN subject and paired the newest date with an older subject
    from datetime import date
    from core.sentimento import ultimo_fato
    fatos = pd.DataFrame({"ticker": ["PETR4", "PETR4", "VALE3"],
                          "data": [date(2026, 10, 16), date(2026, 10, 10), date(2026, 10, 15)],
                          "assunto": [None, "Produção 3T26", "Esclarecimento"]})
    assert ultimo_fato(fatos).to_dict() == {"PETR4": "16/10", "VALE3": "15/10 Esclarecimento"}


def _nota_de_fato(link: str, tom: dict, intensidade: dict, versao: str = "filings-v1") -> dict:
    return {"key": link, "target": "filing", "questions_version": versao, "answers": {
        "filing_tone": {"type": "choice", "probabilities": tom},
        "filing_kind": {"type": "choice", "probabilities": {"other": 1.0}},
        "filing_intensity": {"type": "score", "probabilities": intensidade}}}


def test_each_scored_filing_becomes_one_row_with_its_top_tone_and_strength():
    from core.sentimento import notas_por_fato
    ruim = {"bad": 0.40, "neutral": 0.45, "good": 0.05, "unclear": 0.10}
    forte = {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}
    linhas = [_nota_de_fato("https://cvm/a", ruim, forte),
              _nota_de_fato("https://cvm/b", ruim, forte, versao="filings-v0"),
              _nota("item1", 0.9, {"positive": 1, "negative": 0, "neutral": 0, "not_applicable": 0}, TOPICO, 0.0,
                    {"0": 1, "1": 0, "2": 0, "3": 0})]
    notas = notas_por_fato(linhas)
    assert list(notas.index) == ["https://cvm/a"]
    assert notas.loc["https://cvm/a", "tom_do_fato"] == "neutral"
    assert notas.loc["https://cvm/a", "p_ruim"] == pytest.approx(0.40)
    assert notas.loc["https://cvm/a", "p_bom"] == pytest.approx(0.05)
    assert notas.loc["https://cvm/a", "p_forte"] == pytest.approx(0.7)


def test_the_filings_get_their_jev_scores_by_link_and_an_unscored_filing_stays_without_them():
    from core.sentimento import com_notas_dos_fatos, notas_por_fato
    fatos = pd.DataFrame({"ticker": ["PETR4", "VALE3"], "link": ["https://cvm/a", "https://cvm/b"]})
    notas = notas_por_fato([_nota_de_fato("https://cvm/a", {"bad": 0.8, "neutral": 0.1, "good": 0.0, "unclear": 0.1},
                                          {"0": 0.0, "1": 0.2, "2": 0.5, "3": 0.3})])
    juntos = com_notas_dos_fatos(fatos, notas)
    assert list(juntos["ticker"]) == ["PETR4", "VALE3"]
    assert juntos.loc[0, "tom_do_fato"] == "bad" and juntos.loc[0, "p_forte"] == pytest.approx(0.8)
    assert juntos.loc[1, ["tom_do_fato", "p_ruim", "p_bom", "p_forte"]].isna().all()


def _fatos_pontuados(*linhas) -> pd.DataFrame:
    from datetime import date
    colunas = ["ticker", "assunto", "data", "link", "tom_do_fato", "p_ruim", "p_bom", "p_forte"]
    quadro = pd.DataFrame([dict(zip(colunas, linha)) for linha in linhas], columns=colunas)
    quadro["data"] = [date.fromisoformat(dia) for dia in quadro["data"]]
    return quadro


def test_a_strong_bad_filing_in_the_window_turns_the_stock_red_with_its_date_and_subject():
    from datetime import date
    from core.sentimento import semaforo
    fatos = _fatos_pontuados(("PETR4", "Venda de ativo", "2026-10-20", "https://cvm/a", "bad", 0.8, 0.1, 0.6))
    luzes = semaforo(fatos, ["PETR4.SA"], desde=date(2026, 9, 24))
    assert luzes.loc["PETR4.SA", "cor"] == "vermelho"
    assert luzes.loc["PETR4.SA", "motivo"] == "20/10 Venda de ativo"
    assert luzes.loc["PETR4.SA", "data"] == date(2026, 10, 20)
    assert luzes.loc["PETR4.SA", "link"] == "https://cvm/a"


def test_a_strong_good_filing_turns_the_stock_green():
    from datetime import date
    from core.sentimento import semaforo
    fatos = _fatos_pontuados(("VALE3", "Dividendos", "2026-10-21", "https://cvm/b", "good", 0.0, 0.9, 0.7))
    luzes = semaforo(fatos, ["VALE3.SA"], desde=date(2026, 9, 24))
    assert luzes.loc["VALE3.SA", ["cor", "motivo", "link"]].tolist() == ["verde", "21/10 Dividendos", "https://cvm/b"]


def test_without_a_strong_scored_filing_in_the_window_the_light_is_yellow_and_says_why():
    from datetime import date
    from core.sentimento import semaforo
    fatos = _fatos_pontuados(
        ("PETR4", "Venda de ativo", "2026-10-20", "https://cvm/a", "neutral", 0.40, 0.05, 0.9),
        ("PETR4", "Multa", "2026-09-01", "https://cvm/c", "bad", 0.9, 0.0, 0.9),
        ("VALE3", "Assembleia", "2026-10-21", "https://cvm/b", None, None, None, None))
    luzes = semaforo(fatos, ["PETR4.SA", "VALE3.SA", "ITSA4.SA"], desde=date(2026, 9, 24))
    assert luzes["cor"].tolist() == ["amarelo"] * 3
    assert luzes["motivo"].tolist() == ["neutro (1 fato pontuado)", "sem fato pontuado", "sem fato pontuado"]
    assert luzes[["data", "link"]].isna().all().all()


def test_a_filings_file_from_before_the_scores_were_joined_gives_yellow_lights_instead_of_failing():
    from datetime import date
    from core.sentimento import semaforo
    fatos = _fatos_pontuados(("PETR4", "Venda de ativo", "2026-10-20", "https://cvm/a", "bad", 0.8, 0.1, 0.6))
    antigo = fatos.drop(columns=["tom_do_fato", "p_ruim", "p_bom", "p_forte"])
    luzes = semaforo(antigo, ["PETR4.SA"], desde=date(2026, 9, 24))
    assert luzes.loc["PETR4.SA", ["cor", "motivo"]].tolist() == ["amarelo", "sem fato pontuado"]


def test_a_red_filing_beats_a_newer_green_one():
    from datetime import date
    from core.sentimento import semaforo
    fatos = _fatos_pontuados(("PETR4", "Multa", "2026-10-02", "https://cvm/a", "bad", 0.8, 0.1, 0.6),
                             ("PETR4", "Dividendos", "2026-10-20", "https://cvm/b", "good", 0.0, 0.9, 0.9))
    assert semaforo(fatos, ["PETR4.SA"], desde=date(2026, 9, 24)).loc["PETR4.SA", "motivo"] == "02/10 Multa"


def test_the_next_rebalance_is_63_lake_sessions_after_the_registered_one_counting_weekdays_past_the_last_session():
    from datetime import date
    from core.sentimento import proxima_troca
    calendario = pd.bdate_range("2026-10-01", "2026-10-23")      # 4 sessions after the registered 19/10
    assert proxima_troca(calendario, date(2026, 10, 19)) == (pd.Timestamp("2026-10-23") + pd.offsets.BDay(59)).date()


def test_only_a_stock_that_turned_red_after_the_registered_portfolio_is_flagged_with_the_next_rebalance():
    from datetime import date
    from core.sentimento import alertas_antes_da_troca
    luzes = pd.DataFrame({"cor": ["vermelho", "vermelho", "verde"],
                          "data": [date(2026, 10, 20), date(2026, 10, 16), date(2026, 10, 21)]},
                         index=["PETR4.SA", "VALE3.SA", "ITSA4.SA"])
    assert alertas_antes_da_troca(luzes, date(2026, 10, 19), date(2027, 1, 19)) == [
        "PETR4 ficou vermelho em 20/10, depois da carteira de 19/10; próxima troca ~19/01/2027"]


def test_the_filing_scores_are_read_from_the_lakes_score_files(tmp_path):
    import json
    from datetime import date
    from core.sentimento import ler_notas_dos_fatos
    (tmp_path / "scores").mkdir()
    nota = _nota_de_fato("https://cvm/a", {"bad": 0.1, "neutral": 0.2, "good": 0.7, "unclear": 0.0},
                         {"0": 0.0, "1": 0.5, "2": 0.5, "3": 0.0})
    (tmp_path / "scores" / "2026-10-26.jsonl").write_text(json.dumps(nota) + "\n{\"key\": \"cort", encoding="utf-8")
    notas = ler_notas_dos_fatos(tmp_path, desde=date(2026, 10, 23))
    assert notas.index.tolist() == ["https://cvm/a"] and notas.loc["https://cvm/a", "tom_do_fato"] == "good"


def test_with_no_filing_scored_yet_the_probability_columns_are_still_numbers():
    from core.sentimento import notas_por_fato
    notas = notas_por_fato([])
    assert notas.empty and all(notas[coluna].dtype == float for coluna in ("p_ruim", "p_bom", "p_forte"))
