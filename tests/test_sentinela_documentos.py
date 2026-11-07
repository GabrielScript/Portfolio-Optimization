"""A new Fato Relevante's document text for the Sentinela's scan (core/sentinela/documentos.py, spec amendment 7)."""

import pandas as pd
import requests

from core.sentinela.documentos import com_corpos, texto_do_pdf
from core.sentinela.itens import itens_cvm


def _pdf(texto: str) -> bytes:
    """A one-page PDF whose page shows ``texto`` (offsets computed, so any reader opens it)."""
    conteudo = f"BT /F1 12 Tf 72 712 Td ({texto}) Tj ET".encode("latin-1")
    objetos = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
               b"/Resources << /Font << /F1 5 0 R >> >> >>",
               b"<< /Length %d >>\nstream\n" % len(conteudo) + conteudo + b"\nendstream",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    saida, posicoes = bytearray(b"%PDF-1.4\n"), []
    for numero, objeto in enumerate(objetos, start=1):
        posicoes.append(len(saida))
        saida += b"%d 0 obj\n" % numero + objeto + b"\nendobj\n"
    inicio = len(saida)
    saida += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objetos) + 1)
    saida += b"".join(b"%010d 00000 n \n" % posicao for posicao in posicoes)
    saida += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objetos) + 1, inicio)
    return bytes(saida)


class _Resposta:
    def __init__(self, conteudo: bytes, status: int = 200):
        self.content, self.status_code, self.ok = conteudo, status, status < 400


def _fatos() -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["BBSE3", "BBSE3", "PSSA3"], "empresa": ["BB SEGURIDADE", "BB SEGURIDADE", "PORTO"],
                         "categoria": ["Fato Relevante", "Comunicado ao Mercado", "Fato Relevante"],
                         "assunto": ["Modificação de Projeções Divulgadas", "Apresentação 3T26", "Modificação de Projeções"],
                         "data": ["2026-11-03", "2026-11-04", "2026-11-05"],
                         "link": ["https://rad/1", "https://rad/2", "https://rad/3"]})


def test_the_text_of_a_pdf_has_its_whitespace_collapsed():
    assert texto_do_pdf(_pdf("FATO   RELEVANTE: guidance de -3% a +2% para -15% a -11%")) == \
        "FATO RELEVANTE: guidance de -3% a +2% para -15% a -11%"


def test_only_new_fatos_relevantes_are_downloaded():
    pedidos = []

    def get(url, **_):
        pedidos.append(url)
        return _Resposta(_pdf("Brasilseg: de -3% a +2% para -15% a -11%"))

    visto = itens_cvm(_fatos(), ["PSSA3"])[0].id                     # PSSA3's filing went to the model already
    fatos, falhas = com_corpos(_fatos(), {visto}, get)
    assert pedidos == ["https://rad/1"] and falhas == []             # the Comunicado stays subject-only
    assert fatos["corpo"].tolist()[0] == "Brasilseg: de -3% a +2% para -15% a -11%"
    assert fatos["corpo"].isna().tolist()[1:] == [True, True]
    assert "-15% a -11%" in itens_cvm(fatos, ["BBSE3"])[0].texto


def test_a_failed_download_or_a_page_that_is_no_pdf_keeps_the_subject_and_is_noted():
    def fora_do_ar(url, **_):
        raise requests.ConnectionError("down")

    fatos, falhas = com_corpos(_fatos(), set(), fora_do_ar)
    assert fatos["corpo"].isna().all() and len(falhas) == 2 and "BBSE3" in falhas[0]
    fatos, falhas = com_corpos(_fatos(), set(), lambda url, **_: _Resposta(b"<html>sessao expirada</html>"))
    assert fatos["corpo"].isna().all() and len(falhas) == 2
    assert itens_cvm(fatos, ["BBSE3"])[0].texto == itens_cvm(_fatos(), ["BBSE3"])[0].texto


def test_an_empty_table_comes_back_unchanged():
    vazia = _fatos().iloc[:0]
    fatos, falhas = com_corpos(vazia, set(), lambda url, **_: None)
    assert fatos.empty and falhas == []


def test_the_run_downloads_each_new_fato_relevante_of_the_names_in_the_week_once(tmp_path):
    from datetime import date

    from core.sentinela.vistos import marcar
    from scripts.sentinela import _com_documentos
    semanal = _fatos().assign(data=["2026-10-20", "2026-11-04", "2026-11-05"])      # BBSE3's is three weeks old
    diaria = pd.concat([_fatos().iloc[[2]], _fatos().iloc[[0]].assign(ticker="VALE3", link="https://rad/9",
                                                                       data="2026-11-05")])
    pedidos = []

    def get(url, **_):
        pedidos.append(url)
        return _Resposta(_pdf("texto do documento"))

    vistos = tmp_path / "vistos.csv"
    falhas = []
    tabelas = _com_documentos([semanal, diaria], ["BBSE3", "PSSA3"], date(2026, 10, 30), vistos, falhas, get)
    assert pedidos == ["https://rad/3"] and falhas == []      # PSSA3's, once though both tables list it
    assert len(tabelas) == 1 and len(tabelas[0]) == 4                                  # the duplicate is gone
    marcar(vistos, [("PSSA3", itens_cvm(_fatos(), ["PSSA3"])[0].id)], pd.Timestamp("2026-11-07 09:20"))
    pedidos.clear()
    _com_documentos([semanal, diaria], ["BBSE3", "PSSA3"], date(2026, 10, 30), vistos, falhas, get)
    assert pedidos == []                                                               # seen: no second download


def test_the_rad_page_that_comes_before_the_pdf_gets_another_try():
    # the RAD sometimes answers an HTML page first and the PDF on the next request (seen on 07/11)
    respostas = iter([_Resposta(b"\r\n<!DOCTYPE html><html></html>"), _Resposta(_pdf("o documento"))])
    fatos, falhas = com_corpos(_fatos().iloc[[0]], set(), lambda url, **_: next(respostas))
    assert falhas == [] and fatos["corpo"].tolist() == ["o documento"]
