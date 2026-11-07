"""The whole-page finish review of 2026-11-06: below the simulator, tables kept US decimals and "None", the news table
showed URLs for headlines without a title, prose sat in blue info boxes and the Sentinela had odd placeholders."""

from pathlib import Path

import pandas as pd

from ui import painel, sentinela

UI = Path(__file__).resolve().parents[1] / "ui"


def test_a_headline_without_a_title_reads_as_its_source_not_as_a_url():
    manchetes = pd.DataFrame({"titulo": ["Copom mantém a Selic", "", None], "fonte": ["Valor", "InfoMoney", "Exame"],
                              "link": ["https://a.com/1", "https://b.com/2", "https://c.com/3"]})
    titulos = painel.manchetes_com_link(manchetes)["titulo"].tolist()
    assert titulos[0].endswith("#manchete=Copom mantém a Selic")
    assert titulos[1].endswith("#manchete=(sem título) · InfoMoney")
    assert titulos[2].endswith("#manchete=(sem título) · Exame")


def test_the_market_tone_reads_with_a_comma_and_a_true_minus():
    assert painel.tom_em_texto(-0.14) == "\u22120,14" and painel.tom_em_texto(0.45) == "+0,45"
    assert painel.tom_em_texto(float("nan")) == "\u2014"


def test_the_sentinela_placeholders_read_like_money_and_a_dash():
    assert sentinela.dolares(0.0) == "US$ 0,00" and sentinela.dolares(1.234) == "US$ 1,23"
    assert sentinela.descartadas_em_texto(None) == "\u2014" and sentinela.descartadas_em_texto(0.034) == "3,4%"


def test_prose_is_never_in_a_coloured_info_box_on_the_account_screens():
    for nome in ("simulador.py", "simulador_opcoes.py", "sentinela.py"):
        assert "st.info(" not in (UI / nome).read_text(encoding="utf-8"), nome


def test_no_table_uses_a_format_that_writes_a_decimal_point():
    # printf and "percent" formats ignore the pt-BR locale: every table goes through ui.formato.br
    for nome in ("formulas.py", "simulador_opcoes.py", "sentinela.py", "alepha.py", "comparacao.py", "painel.py",
                 "cotacoes.py"):
        codigo = (UI / nome).read_text(encoding="utf-8")
        for formato in ('format="percent"', 'format="%.', 'format="%+', 'format="R$'):
            assert formato not in codigo, (nome, formato)
        assert "st.line_chart" not in codigo, nome


def test_both_per_year_tables_use_the_wide_year_column():
    for nome in ("alepha.py", "comparacao.py"):
        assert "COLUNA_DO_ANO" in (UI / nome).read_text(encoding="utf-8"), nome


def test_a_headline_stored_as_its_url_reads_as_words_from_the_url():
    # the final live review: rows showed "https://www.infomoney.com.br/economia/ipca-..." as the headline
    manchetes = pd.DataFrame({"titulo": ["https://www.infomoney.com.br/economia/ipca-sobe-0-5-em-outubro/"],
                              "fonte": ["InfoMoney"], "link": ["https://www.infomoney.com.br/economia/ipca-sobe-0-5-em-outubro/"]})
    titulo = painel.manchetes_com_link(manchetes)["titulo"].iloc[0]
    assert titulo.endswith("#manchete=Ipca sobe 0 5 em outubro · InfoMoney")


def test_a_percent_sign_in_a_headline_does_not_break_its_link():
    # live: "XP reduz projeção do PIB a 0,3% em 2027" showed the raw URL (a bare % is malformed in a link)
    manchetes = pd.DataFrame({"titulo": ["XP reduz o PIB a 0,3% em 2027", "Selic a 13,75%"], "fonte": ["a", "b"],
                              "link": ["https://a.com/1", "https://a.com/2"]})
    titulos = painel.manchetes_com_link(manchetes)["titulo"].tolist()
    assert all("%" not in t for t in titulos)
    assert titulos[0].endswith("#manchete=XP reduz o PIB a 0,3 por cento em 2027")
    assert titulos[1].endswith("#manchete=Selic a 13,75 por cento")
