import io

import pandas as pd
import pytest

from core.sentinela import fontes
from core.sentinela.itens import Serie, itens_macro

SIDRA = [
    {"V": "Valor", "D3C": "Trimestre (Código)", "D3N": "Trimestre"},
    {"V": "0.2", "D3C": "202504", "D3N": "4º trimestre 2025"},
    {"V": "1.1", "D3C": "202601", "D3N": "1º trimestre 2026"},
    {"V": "...", "D3C": "202602", "D3N": "2º trimestre 2026"},
]


def test_sidra_quarters_and_months_become_dated_values_with_the_official_period_names():
    valores, nomes = fontes.serie_sidra(SIDRA)
    assert list(valores.index) == [pd.Timestamp("2025-10-01"), pd.Timestamp("2026-01-01")]
    assert list(valores) == [0.2, 1.1] and nomes[pd.Timestamp("2026-01-01")] == "1º trimestre 2026"
    mensal = [{"V": "Valor", "D3C": "Mês (Código)", "D3N": "Mês"}, {"V": "-0.5", "D3C": "202607", "D3N": "julho 2026"}]
    valores, nomes = fontes.serie_sidra(mensal)
    assert valores.index[0] == pd.Timestamp("2026-07-01") and nomes[valores.index[0]] == "julho 2026"


def test_macro_items_use_the_official_period_name_when_given():
    valores, nomes = fontes.serie_sidra(SIDRA)
    (item,) = itens_macro([Serie("pib", "ibge", "PIB, variação trimestral (%), IBGE SIDRA 5932", valores, nomes=nomes)])
    assert item.texto == ("PIB, variação trimestral (%), IBGE SIDRA 5932: 1,1 em 1º trimestre 2026 (anterior: 0,2 em "
                          "4º trimestre 2025)")


def _planilha() -> bytes:
    linhas = [["Voltar"], ["Tabela 1.1. Resultado Primário do Governo Central"], ["R$ Milhões"], [],
              ["Discriminação", pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-01")],
              ["1. RECEITA TOTAL 1/", 1.0, 2.0],
              ["3. RECEITA LÍQUIDA  (1-2)", 187959.89, 204947.49],
              ["4. DESPESA TOTAL", 201517.57, 200108.41],
              ["5. RESULTADO PRIMÁRIO GOVERNO CENTRAL - ACIMA DA LINHA (3 - 4)", -13557.668, 4839.084]]
    saida = io.BytesIO()
    with pd.ExcelWriter(saida) as escritor:
        pd.DataFrame([["Índice"]]).to_excel(escritor, sheet_name="Índice", header=False, index=False)
        pd.DataFrame(linhas).to_excel(escritor, sheet_name="1.1", header=False, index=False)
    return saida.getvalue()


def test_rtn_table_1_1_gives_net_revenue_spending_and_primary_result_in_r_millions_one_decimal():
    series = fontes.ler_rtn(_planilha())
    assert set(series) == {"rtn_receita_liquida", "rtn_despesa_total", "rtn_resultado_primario"}
    primario = series["rtn_resultado_primario"]
    assert list(primario.index) == [pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-01")]
    assert list(primario) == [-13557.7, 4839.1]


def test_rtn_link_is_found_in_the_page_or_in_its_rendered_markdown():
    html = '<a href="https://thot-arquivos.tesouro.gov.br/publicacao-anexo/29566">serie_historica_set26.xlsx</a>'
    assert fontes.link_do_rtn(html) == "https://thot-arquivos.tesouro.gov.br/publicacao-anexo/29566"
    markdown = "- [serie_historica_set26.xlsx](https://thot-arquivos.tesouro.gov.br/publicacao-anexo/29566)"
    assert fontes.link_do_rtn(markdown) == "https://thot-arquivos.tesouro.gov.br/publicacao-anexo/29566"
    assert fontes.link_do_rtn("<p>Falha na integração THOT/3PRO.</p>") is None


def test_rtn_pages_are_tried_newest_month_first():
    assert fontes.paginas_do_rtn(pd.Timestamp("2026-11-05").date())[:2] == [
        "https://www.tesourotransparente.gov.br/publicacoes/boletim-resultado-do-tesouro-nacional-rtn/2026/10",
        "https://www.tesourotransparente.gov.br/publicacoes/boletim-resultado-do-tesouro-nacional-rtn/2026/9"]


class _Http:
    def __init__(self, respostas: dict):
        self.respostas, self.pedidos = respostas, []

    def __call__(self, url, **kwargs):
        self.pedidos.append(url)
        conteudo = self.respostas.get(url)

        class R:
            ok = conteudo is not None
            status_code = 200 if ok else 404
            content = conteudo if isinstance(conteudo, bytes) else (conteudo.encode() if isinstance(conteudo, str)
                                                                     else b"")
            text = conteudo if isinstance(conteudo, str) else ""

            def json(self):
                return conteudo
        return R()


def test_rtn_download_falls_back_from_the_page_to_firecrawl_to_ckan():
    planilha = _planilha()
    anexo = "https://thot-arquivos.tesouro.gov.br/publicacao-anexo/1"
    paginas = fontes.paginas_do_rtn(pd.Timestamp("2026-11-05").date())
    http = _Http({paginas[0]: "<p>Falha na integração THOT/3PRO.</p>", anexo: planilha})
    renderizada = {paginas[0]: f"[serie_historica_out26.xlsx]({anexo})"}
    conteudo, origem = fontes.baixar_rtn(pd.Timestamp("2026-11-05").date(), http, renderizada.get)
    assert conteudo == planilha and origem == f"firecrawl {paginas[0]}"
    ckan = {"result": {"resources": [{"format": "XLSX", "name": "Série Histórica - Mensal",
                                      "url": "https://ckan/serie.xlsx"}]}}
    http = _Http({fontes.URL_CKAN_RTN: ckan, "https://ckan/serie.xlsx": planilha})
    conteudo, origem = fontes.baixar_rtn(pd.Timestamp("2026-11-05").date(), http, lambda url: None)
    assert conteudo == planilha and origem == "ckan"
    with pytest.raises(fontes.FonteIndisponivel):
        fontes.baixar_rtn(pd.Timestamp("2026-11-05").date(), _Http({}), lambda url: None)


def test_selic_target_keeps_only_the_days_it_changed():
    diaria = pd.Series([15.0, 15.0, 14.75, 14.75], index=pd.bdate_range("2026-11-02", periods=4))
    mudancas = fontes.so_mudancas(diaria)
    assert list(mudancas.index) == [pd.Timestamp("2026-11-02"), pd.Timestamp("2026-11-04")]


def test_weekly_keeps_the_last_survey_of_each_week():
    focus = pd.Series([4.5, 4.6, 4.7], index=pd.to_datetime(["2026-10-30", "2026-11-02", "2026-11-06"]))
    assert list(fontes.semanal(focus)) == [4.5, 4.7]
