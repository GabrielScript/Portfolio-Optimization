"""The DeepSeek adviser (``core.assessor``, DECISOES 136): an evidence pack with ids, tools that add evidence, and a
check that keeps only the claims whose quote is in the evidence they cite. Display only: nothing here reaches the
weights, the orders or the paper routines."""

import pytest

from core.assessor import Evidencia


def _evidencias() -> dict[str, Evidencia]:
    lista = [Evidencia("E1", "bcb", "2026-10-24", "Selic meta: 15,00% a.a."),
             Evidencia("E2", "bcb", "2026-09-30", "IPCA 12 meses: 5,17%"),
             Evidencia("E3", "bcb", "2026-10-17", "Focus Selic fim de 2027: 12,25%"),
             Evidencia("E4", "mercado", "2026-10-24", "PETR4: P/L 4,8; dividend yield 12m 11,2%; vol anual 27,5%"),
             Evidencia("E5", "noticia", "2026-10-24", "Petrobras anuncia novo plano de investimentos")]
    return {evidencia.id: evidencia for evidencia in lista}


def _afirmacao(texto: str, evidencia: str, citacao: str) -> dict:
    return {"afirmacao": texto, "evidencia": evidencia, "citacao": citacao}


def test_a_claim_survives_only_when_its_quote_is_in_the_cited_evidence_and_its_numbers_are_in_the_quote():
    from core.assessor import validar_afirmacao
    evidencias = _evidencias()
    assert validar_afirmacao(_afirmacao("P/L baixo, de 4,8", "E4", "p/l 4,8"), evidencias) is None
    assert validar_afirmacao(_afirmacao("A Selic está em 15%", "E1", "Selic meta:  15,00%"), evidencias) is None
    assert "não existe" in validar_afirmacao(_afirmacao("P/L de 4,8", "E9", "P/L 4,8"), evidencias)
    assert "citação" in validar_afirmacao(_afirmacao("P/L de 4,8", "E4", "P/L 3,9"), evidencias)
    assert "9,5" in validar_afirmacao(_afirmacao("Yield de 9,5%", "E4", "dividend yield 12m 11,2%"), evidencias)
    assert validar_afirmacao({"afirmacao": "sem fonte"}, evidencias) is not None
    # digits inside a word (a ticker, a quarter, "12m") are names, not numbers to check
    assert validar_afirmacao(_afirmacao("PETR4 tem P/L de 4,8 desde o 2T26", "E4", "P/L 4,8"), evidencias) is None
    assert "não está na evidência E4: «P/L 3,9»" in validar_afirmacao(_afirmacao("P/L", "E4", "P/L 3,9"), evidencias)
    # accents do not matter in the quote (the model often writes "liq" for "líq"); letters and numbers do
    assert validar_afirmacao(_afirmacao("Selic de 15%", "E1", "SELIC META: 15,00% A.A."), evidencias) is None
    assert validar_afirmacao(_afirmacao("IPCA de 5,17%", "E2", "ipca 12 meses: 5,17%"), evidencias) is None
    assert validar_afirmacao(_afirmacao("Plano", "E5", "Petrobras anúncia novo plano"), evidencias) is None


def _acao(ticker: str, contra: list[dict], postura: str = "favoravel") -> dict:
    return {"ticker": ticker, "tese": [_afirmacao("P/L de 4,8", "E4", "P/L 4,8")], "contra_tese_macro": contra,
            "riscos": [_afirmacao("Novo plano de investimentos", "E5", "novo plano de investimentos")],
            "postura": postura, "confianca": 0.6}


def test_the_opinion_keeps_plan_tickers_once_and_needs_three_bcb_counter_reasons_for_a_stance():
    from core.assessor import validar_parecer
    contra = [_afirmacao("Selic de 15%", "E1", "15,00%"), _afirmacao("IPCA de 5,17%", "E2", "5,17%"),
              _afirmacao("Focus vê Selic de 12,25%", "E3", "12,25%")]
    bruto = {"resumo": "Carteira defensiva com bom carrego.", "cenario_macro": [contra[0]],
             "acoes": [_acao("PETR4", contra), _acao("VALE3.SA", contra[:2] + [_afirmacao("Plano", "E5", "plano")]),
                       _acao("PETR4", contra), _acao("XPTO3", contra)],
             "ranking": ["VALE3", "XPTO3", "PETR4", "PETR4"], "limitacoes": ["Sem dados da ANBIMA."]}
    resultado = validar_parecer(bruto, _evidencias(), ["PETR4", "VALE3"])
    assert resultado.estrutura == []
    acoes = {acao["ticker"]: acao for acao in resultado.parecer["acoes"]}
    assert list(acoes) == ["PETR4", "VALE3"]
    assert acoes["PETR4"]["postura"] == "favoravel" and len(acoes["PETR4"]["contra_tese_macro"]) == 3
    assert acoes["VALE3"]["postura"] == "evidencia_insuficiente" and len(acoes["VALE3"]["contra_tese_macro"]) == 2
    assert resultado.parecer["ranking"] == ["VALE3", "PETR4"]
    assert resultado.parecer["resumo"] == "Carteira defensiva com bom carrego."
    assert any("XPTO3" in aviso for aviso in resultado.descartes)
    assert any("VALE3" in aviso and "bcb" in aviso for aviso in resultado.descartes)


def test_free_text_with_numbers_is_dropped_and_a_non_object_answer_is_a_structural_error():
    from core.assessor import validar_parecer
    bruto = {"resumo": "Retorno esperado de 30% ao ano.", "cenario_macro": [], "acoes": [], "ranking": [],
             "limitacoes": ["Faltam 3 fontes.", "Sem ANBIMA."]}
    resultado = validar_parecer(bruto, _evidencias(), ["PETR4"])
    assert resultado.parecer["resumo"] == "" and resultado.parecer["limitacoes"] == ["Sem ANBIMA."]
    assert len(resultado.descartes) == 2
    assert validar_parecer([], _evidencias(), ["PETR4"]).estrutura
    assert validar_parecer({"resumo": "ok"}, _evidencias(), ["PETR4"]).estrutura


def _tabela_das_acoes():
    import pandas as pd
    return pd.DataFrame({"Empresa": ["Petrobras", "Vale"], "Setor": ["Petróleo", "Mineração"],
                         "Preço (R$)": [30.126, 61.0], "Fechamento de": pd.to_datetime(["2026-10-24", "2026-10-24"]),
                         "Retorno 12m": [0.051, -0.1234], "Vol anual": [0.275, 0.31], "Beta": [0.85, float("nan")],
                         "P/L": [4.8, 6.25], "DY 12m": [0.112, 0.08], "Balanço de": pd.to_datetime(["2026-06-30"] * 2)},
                        index=["PETR4", "VALE3"])


def test_the_pack_numbers_each_fact_in_a_fixed_order_with_brazilian_numbers_and_says_what_is_missing():
    from datetime import date

    import pandas as pd
    from core.assessor import montar_pacote
    argumentos = dict(
        plano=pd.Series({"VALE3.SA": 0.5, "PETR4.SA": 0.5}), registrada=date(2026, 10, 23),
        acoes=_tabela_das_acoes(), percentuais=("Retorno 12m", "Vol anual", "DY 12m"),
        macro=[("2026-10-24", "Selic meta: 15,00% a.a. (BCB SGS 432)")],
        fatos=pd.DataFrame({"ticker": ["PETR4", "VALE3"], "categoria": ["Fato Relevante"] * 2,
                            "assunto": ["Plano de negócios", float("nan")], "data": [date(2026, 10, 20)] * 2,
                            "tom_do_fato": ["good", None], "p_forte": [0.7, None]}),
        luzes=pd.DataFrame({"cor": ["amarelo", "verde"], "motivo": ["sem fato pontuado", "20/10 Plano de negócios"]},
                           index=["VALE3.SA", "PETR4.SA"]),
        manchetes=None)
    pacote = montar_pacote(**argumentos)
    assert pacote == montar_pacote(**argumentos)
    assert [e.id for e in pacote] == [f"E{i}" for i in range(1, len(pacote) + 1)]
    assert [e.classe for e in pacote] == ["bcb", "plano", "plano", "mercado", "mercado", "cvm", "cvm", "cvm", "cvm",
                                          "aviso"]
    textos = [e.texto for e in pacote]
    assert textos[1] == "PETR4 está na carteira do Alepha registrada em 23/10/2026, peso alvo 50,0%"
    assert textos[3] == ("PETR4 (Petrobras, Petróleo): Preço (R$) 30,13; Retorno 12m 5,1%; Vol anual 27,5%; "
                         "Beta 0,85; P/L 4,80; DY 12m 11,2%; Fechamento de 24/10/2026; Balanço de 30/06/2026")
    assert "Beta" not in textos[4] and "Retorno 12m -12,3%" in textos[4]
    assert textos[5] == "PETR4 Fato Relevante em 20/10/2026: Plano de negócios; tom Jev: good"
    assert textos[6] == "VALE3 Fato Relevante em 20/10/2026: sem assunto; tom Jev: sem nota"
    assert textos[7] == "PETR4 semáforo: verde (20/10 Plano de negócios)"
    assert textos[8] == "VALE3 semáforo: amarelo (sem fato pontuado)"
    assert "manchetes" in textos[9].lower()


def _ferramentas(**chaves):
    from datetime import date

    import pandas as pd
    from core.assessor import Fontes, Ferramentas
    fatos = pd.DataFrame({"ticker": ["PETR4"], "categoria": ["Fato Relevante"], "assunto": ["Plano de negócios"],
                          "data": [date(2026, 10, 20)], "tom_do_fato": ["good"], "p_forte": [0.7]})
    fontes = Fontes(hoje=date(2026, 10, 25), acoes=_tabela_das_acoes(), percentuais=("Retorno 12m", "DY 12m"),
                    fatos=fatos, serie_bcb=None, **chaves)
    return Ferramentas(fontes, _evidencias())


def test_the_local_tools_add_numbered_evidence_and_a_tool_without_its_key_says_it_is_unavailable():
    import json
    ferramentas = _ferramentas()
    resposta = json.loads(ferramentas.executar("metricas_da_acao", '{"ticker": "vale3.SA"}'))
    assert resposta["evidencias"][0]["id"] == "E1001" and resposta["evidencias"][0]["classe"] == "mercado"
    assert resposta["evidencias"][0]["texto"].startswith("VALE3 (Vale, Mineração): ")
    resposta = json.loads(ferramentas.executar("fatos_relevantes", '{"ticker": "PETR4"}'))
    assert resposta["evidencias"][0]["id"] == "E1002" and "Plano de negócios" in resposta["evidencias"][0]["texto"]
    assert ferramentas.evidencias["E1002"].classe == "cvm" and "E1" in ferramentas.evidencias
    assert "erro" in json.loads(ferramentas.executar("metricas_da_acao", '{"ticker": "XPTO3"}'))
    assert "erro" in json.loads(ferramentas.executar("metricas_da_acao", "não é json"))
    assert "erro" in json.loads(ferramentas.executar("apagar_tudo", "{}"))
    for nome, argumentos in (("buscar_web", '{"consulta": "Selic"}'), ("buscar_noticias", '{"consulta": "Selic"}'),
                             ("buscar_posts", '{"consulta": "Selic"}'), ("ler_pagina", '{"url": "https://a.com"}'),
                             ("serie_bcb", '{"serie": "selic_meta"}')):
        assert json.loads(ferramentas.executar(nome, argumentos)) == {"erro": "indisponível nesta versão do app"}


class _Resposta:
    def __init__(self, status: int, dados=None) -> None:
        self.status_code, self._dados = status, dados

    def json(self):
        return self._dados


def test_the_remote_tools_send_the_key_only_in_the_header_and_turn_results_into_evidence():
    import json

    import pandas as pd
    pedidos = []

    def post(url, json, timeout, headers):
        pedidos.append((url, json, headers))
        if url.endswith("/v1/tools/search"):
            return _Resposta(200, {"results": [
                {"title": "Copom mantém Selic", "url": "https://bcb.gov.br/a", "summary": "O Copom manteve a taxa.",
                 "site_name": "BCB", "date": "2026-10-20T10:00:00+00:00", "authority": 4, "rerank_score": 2.41},
                {"title": "Golpe", "url": "javascript:alert(1)", "summary": "x"}]})
        if url.endswith("/v1/tools/fetch"):
            return _Resposta(200, {"title": "Ata", "content": "# Ata\n\n" + "a" * 5000})
        return _Resposta(200, {"data": {"news": [{"title": "Vale sobe", "snippet": "Minério alta",
                                                  "url": "https://valor.com.br/v", "date": "2 hours ago"}]}})

    def get(url, params, timeout, headers):
        pedidos.append((url, params, headers))
        return _Resposta(200, {"posts": [{"caption": {"text": "Vale  vai bombar"}, "taken_at": 1761400000,
                                          "like_count": 7, "user": {"username": "investidor"}}]})
    ferramentas = _ferramentas(chave_deepseek="ds", chave_firecrawl="fc", chave_scrapecreators="sc")
    ferramentas.fontes.post, ferramentas.fontes.get = post, get
    ferramentas.fontes.serie_bcb = lambda codigo, inicio, hoje: pd.Series(
        [14.75, 15.0, 15.0], index=pd.to_datetime(["2026-08-01", "2026-09-30", "2026-10-24"]))
    web = json.loads(ferramentas.executar("buscar_web", '{"consulta": "Selic Copom"}'))["evidencias"]
    assert len(web) == 1 and web[0]["classe"] == "web" and web[0]["data"] == "2026-10-20"
    assert web[0]["texto"] == "BCB: Copom mantém Selic. O Copom manteve a taxa. [autoridade 4, relevância 2,41; " \
                              "https://bcb.gov.br/a]"
    url, corpo, cabecalhos = pedidos[0]
    assert url == "https://api.deepseek.com/v1/tools/search" and cabecalhos == {"Authorization": "Bearer ds"}
    assert corpo == {"query": "Selic Copom lang:pt after:2026-09-25", "count": 5}
    pagina = json.loads(ferramentas.executar("ler_pagina", '{"url": "https://bcb.gov.br/a"}'))["evidencias"][0]
    assert pagina["texto"].startswith("Ata (https://bcb.gov.br/a): # Ata a") and len(pagina["texto"]) < 3100
    noticia = json.loads(ferramentas.executar("buscar_noticias", '{"consulta": "Vale"}'))["evidencias"][0]
    assert noticia["classe"] == "noticia" and noticia["texto"] == "Vale sobe. Minério alta [https://valor.com.br/v]"
    assert pedidos[2][2] == {"Authorization": "Bearer fc"} and pedidos[2][1]["sources"] == ["news"]
    post_ = json.loads(ferramentas.executar("buscar_posts", '{"consulta": "Vale"}'))["evidencias"][0]
    assert post_["classe"] == "social" and post_["texto"] == "@investidor: Vale vai bombar [curtidas 7]"
    assert pedidos[3][2] == {"x-api-key": "sc"} and pedidos[3][1] == {"query": "Vale"}
    selic = json.loads(ferramentas.executar("serie_bcb", '{"serie": "selic_meta"}'))["evidencias"]
    assert [e["texto"] for e in selic][-1] == "Selic meta (% a.a.), SGS 432, em 24/10/2026: 15,00"
    assert len(selic) == 3 and all(e["classe"] == "bcb" for e in selic)
    assert ferramentas.chamadas_pagas == {"buscar_web": 1, "ler_pagina": 1, "buscar_noticias": 1, "buscar_posts": 1}
    ferramentas.fontes.post = lambda url, json, timeout, headers: _Resposta(429, {"error": {"message": "QPS"}})
    erro = json.loads(ferramentas.executar("buscar_web", '{"consulta": "Selic"}'))["erro"]
    assert "429" in erro and "ds" not in erro


def _uso(entrada: int, cache: int, saida: int) -> dict:
    return {"prompt_tokens": entrada, "completion_tokens": saida, "prompt_cache_hit_tokens": cache,
            "prompt_cache_miss_tokens": entrada - cache, "completion_tokens_details": {"reasoning_tokens": saida // 2}}


def _resposta_do_modelo(mensagem: dict, uso: dict) -> _Resposta:
    return _Resposta(200, {"choices": [{"message": {"role": "assistant", **mensagem}, "finish_reason": "stop"}],
                           "usage": uso})


def _parecer_com(evidencia: str, citacao: str) -> str:
    import json
    contra = [_afirmacao("Selic de 15%", "E1", "15,00%"), _afirmacao("IPCA de 5,17%", "E2", "5,17%"),
              _afirmacao("Focus vê Selic de 12,25%", "E3", "12,25%")]
    return json.dumps({"cenario_macro": [], "acoes": [
        {"ticker": "VALE3", "tese": [_afirmacao("Retorno de -12,3% em 12 meses", evidencia, citacao)],
         "contra_tese_macro": contra, "riscos": [], "postura": "neutra", "confianca": 0.4}],
        "ranking": ["VALE3"], "resumo": "Sem convicção.", "limitacoes": []}, ensure_ascii=False)


def test_a_tool_round_sends_the_reasoning_back_keeps_the_cached_prefix_and_checks_claims_from_tool_evidence():
    import copy
    from datetime import datetime, timezone

    from core.assessor import FERRAMENTAS, pedir_parecer
    corpos = []
    respostas = iter([
        _resposta_do_modelo({"content": "", "reasoning_content": "preciso da Vale", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "metricas_da_acao", "arguments": '{"ticker": "VALE3"}'}}]},
            _uso(1000, 0, 100)),
        _resposta_do_modelo({"content": _parecer_com("E1001", "Retorno 12m -12,3%"), "reasoning_content": "ok"},
                            _uso(1200, 1000, 400))])

    def post(url, json, timeout, headers):
        assert url == "https://api.deepseek.com/chat/completions" and headers["Authorization"] == "Bearer ds"
        corpos.append(copy.deepcopy(json))
        return next(respostas)
    ferramentas = _ferramentas()
    resultado = pedir_parecer(list(_evidencias().values()), ["VALE3"], ferramentas, "ds",
                              datetime(2026, 10, 25, 15, 0, tzinfo=timezone.utc), post=post)
    assert resultado.erro is None and resultado.validacao.descartes == []
    assert resultado.validacao.parecer["acoes"][0]["tese"][0]["evidencia"] == "E1001"
    primeiro, segundo = corpos
    assert primeiro["model"] == "deepseek-flash" and primeiro["thinking"] == {"type": "enabled"}
    assert primeiro["reasoning_effort"] == "high" and primeiro["response_format"] == {"type": "json_object"}
    assert "temperature" not in primeiro and primeiro["tools"] == FERRAMENTAS == segundo["tools"]
    assert segundo["messages"][:2] == primeiro["messages"]
    assert "E4 | mercado | 2026-10-24 | PETR4: P/L 4,8" in primeiro["messages"][0]["content"]
    assistente, ferramenta = segundo["messages"][2:]
    assert assistente["reasoning_content"] == "preciso da Vale" and assistente["tool_calls"][0]["id"] == "c1"
    assert ferramenta["role"] == "tool" and ferramenta["tool_call_id"] == "c1" and "E1001" in ferramenta["content"]
    uso = resultado.uso
    assert (uso.entrada, uso.cache, uso.saida, uso.raciocinio, uso.requisicoes) == (2200, 1000, 500, 250, 2)
    assert uso.custo_usd == pytest.approx((1200 * 0.15 + 1000 * 0.003 + 500 * 0.6) / 1e6)
    assert resultado.chamadas == ["metricas_da_acao"]


def test_a_broken_answer_gets_one_repair_an_empty_one_one_retry_and_api_errors_come_back_without_the_key():
    from datetime import datetime, timezone

    from core.assessor import pedir_parecer
    corpos = []

    def roteiro(*respostas):
        fila = iter(respostas)

        def post(url, json, timeout, headers):
            corpos.append(json)
            return next(fila)
        return post
    agora = datetime(2026, 10, 26, 7, 0, tzinfo=timezone.utc)          # Monday 07:00 UTC: peak, double price
    valido = _parecer_com("E4", "dividend yield 12m 11,2%").replace("Retorno de -12,3% em 12 meses", "Yield alto")
    post = roteiro(_resposta_do_modelo({"content": ""}, _uso(100, 0, 10)),
                   _resposta_do_modelo({"content": "não é JSON", "reasoning_content": "r1"}, _uso(100, 0, 10)),
                   _resposta_do_modelo({"content": valido}, _uso(100, 100, 10)))
    resultado = pedir_parecer(list(_evidencias().values()), ["VALE3"], _ferramentas(), "ds", agora, post=post)
    assert resultado.erro is None and resultado.validacao.parecer["acoes"][0]["postura"] == "neutra"
    reparo = corpos[2]["messages"]
    assert reparo[2] == {"role": "assistant", "content": "não é JSON", "reasoning_content": "r1"}
    assert reparo[3]["role"] == "user" and "checagem" in reparo[3]["content"]
    assert resultado.uso.custo_usd == pytest.approx(2 * (200 * 0.15 + 100 * 0.003 + 30 * 0.6) / 1e6)

    post = roteiro(*[_resposta_do_modelo({"content": "{}"}, _uso(100, 0, 10))] * 2)
    resultado = pedir_parecer(list(_evidencias().values()), ["VALE3"], _ferramentas(), "ds", agora, post=post)
    assert resultado.validacao is None and resultado.erro.startswith("parecer recusado")

    post = roteiro(_Resposta(401, {"error": {"message": "Authentication Fails, key ds"}}))
    resultado = pedir_parecer(list(_evidencias().values()), ["VALE3"], _ferramentas(), "ds", agora, post=post)
    assert resultado.erro == "chave do DeepSeek recusada (HTTP 401)"


def test_search_and_fetch_cost_starts_on_november_first_in_beijing():
    from datetime import datetime, timezone

    from core.assessor import custo_das_ferramentas
    chamadas = {"buscar_web": 3, "ler_pagina": 1, "buscar_noticias": 2}
    assert custo_das_ferramentas(chamadas, datetime(2026, 10, 31, 15, 59, tzinfo=timezone.utc)) == 0
    assert custo_das_ferramentas(chamadas, datetime(2026, 10, 31, 16, 0, tzinfo=timezone.utc)) == pytest.approx(0.003)


def test_the_macro_evidence_comes_from_the_selic_the_bcb_series_and_the_focus_medians_with_their_dates():
    from datetime import date

    import pandas as pd
    from core.assessor import macro_do_painel
    from core.market_data import TaxaSelic
    resumo = pd.DataFrame({"2026": [15.0, float("nan")], "2027": [12.25, 2.1],
                           "Mudança em 4 semanas (2026)": [0.0, -0.1],
                           "Pesquisa": pd.to_datetime(["2026-10-17"] * 2)}, index=["Selic", "PIB Total"])
    macro = macro_do_painel(date(2026, 10, 25), TaxaSelic(0.15, "BCB SGS 432"),
                            {"IPCA 12 meses (%)": (pd.Timestamp("2026-09-30"), 5.17)}, resumo)
    assert macro == [("2026-10-25", "Selic meta: 15,00% a.a. (BCB SGS 432)"),
                     ("2026-09-30", "IPCA 12 meses (%) em 30/09/2026: 5,17 (BCB SGS)"),
                     ("2026-10-17", "Focus (mediana) Selic: 2026 15,00; 2027 12,25; Mudança em 4 semanas (2026) 0,00"),
                     ("2026-10-17", "Focus (mediana) PIB Total: 2027 2,10; Mudança em 4 semanas (2026) -0,10")]
    assert macro_do_painel(date(2026, 10, 25), TaxaSelic(0.15, "fallback 2026-03"), None, None) == [
        ("2026-10-25", "Selic meta: 15,00% a.a. (fallback 2026-03)")]


def test_an_answer_cut_at_the_token_limit_is_repaired_with_a_request_to_be_shorter():
    from datetime import datetime, timezone

    from core.assessor import MAX_TOKENS, pedir_parecer
    corpos = []
    cortada = _Resposta(200, {"choices": [{"message": {"role": "assistant", "content": '{"acoes": [{"ticker": "VA'},
                                           "finish_reason": "length"}], "usage": _uso(100, 0, 10)})
    respostas = iter([cortada, _resposta_do_modelo({"content": _parecer_com("E1", "15,00%")}, _uso(100, 0, 10))])

    def post(url, json, timeout, headers):
        corpos.append(json)
        return next(respostas)
    resultado = pedir_parecer(list(_evidencias().values()), ["VALE3"], _ferramentas(), "ds",
                              datetime(2026, 10, 25, 15, 0, tzinfo=timezone.utc), post=post)
    assert resultado.erro is None and MAX_TOKENS >= 64_000
    assert "cortada" in corpos[1]["messages"][-1]["content"]


def test_claims_and_free_text_with_links_are_dropped_and_shown_text_is_markdown_escaped():
    from core.assessor import validar_afirmacao, validar_parecer
    from ui.assessor import sem_markdown
    evidencias = _evidencias()
    assert "link" in validar_afirmacao(_afirmacao("Veja https://golpe.com", "E5", "novo plano"), evidencias)
    assert "link" in validar_afirmacao(_afirmacao("![x](www.golpe.com/a)", "E5", "novo plano"), evidencias)
    resultado = validar_parecer({"acoes": [], "resumo": "Clique em http://golpe.com", "limitacoes": []}, evidencias,
                                ["PETR4"])
    assert resultado.parecer["resumo"] == ""
    assert sem_markdown("**a** [b](c) `d` <e> #f !g _h_") == r"\*\*a\*\* \[b\]\(c\) \`d\` \<e\> \#f \!g \_h\_"


def test_the_pro_model_is_sent_by_name_and_priced_with_its_own_table():
    from datetime import datetime, timezone

    from core.assessor import MODELO_PRO, pedir_parecer
    corpos = []

    def post(url, json, timeout, headers):
        corpos.append(json)
        return _resposta_do_modelo({"content": _parecer_com("E1", "15,00%")}, _uso(1000, 400, 100))
    resultado = pedir_parecer(list(_evidencias().values()), ["VALE3"], _ferramentas(), "ds",
                              datetime(2026, 10, 25, 15, 0, tzinfo=timezone.utc), post=post, modelo=MODELO_PRO)
    assert MODELO_PRO == "deepseek-v4-pro" and corpos[0]["model"] == "deepseek-v4-pro"
    assert resultado.uso.custo_usd == pytest.approx((600 * 0.66 + 400 * 0.022 + 100 * 1.98) / 1e6)


def test_the_prompt_judges_for_an_extremely_aggressive_investor_and_keeps_the_evidence_rules():
    # the user's choice of 2026-10-25 (DECISOES 138): "extremamente arrojado", no longer "volatility not too high"
    from core.assessor import INSTRUCOES, PEDIDO
    assert "extremely aggressive" in INSTRUCOES and "extremamente arrojado" in INSTRUCOES
    assert "avoids extreme volatility" not in INSTRUCOES
    assert "never change the plan's weights" in INSTRUCOES
    assert "an invented fact is the worst possible failure" in INSTRUCOES
    assert "extremamente arrojado" in PEDIDO
