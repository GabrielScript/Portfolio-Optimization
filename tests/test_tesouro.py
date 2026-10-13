"""
Testes de core/tesouro.py (sem rede).

O CSV falso segue o formato real (sondado em 2026-10-12/13): ``;``, vírgula
decimal, latin-1, datas dd/mm/aaaa e as 8 colunas do arquivo publicado.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import requests

from core import tesouro

HOJE = date(2026, 10, 13)
CABECALHO = ("Tipo Titulo;Data Vencimento;Data Base;Taxa Compra Manha;Taxa Venda Manha;"
             "PU Compra Manha;PU Venda Manha;PU Base Manha")


def _csv(linhas: list[tuple[str, str, str, str]], cabecalho: str = CABECALHO) -> bytes:
    """linhas: (tipo, vencimento, data_base, taxa_compra com vírgula)."""
    corpo = [f"{tipo};{venc};{base};{taxa};{taxa};900,00;900,00;900,00" for tipo, venc, base, taxa in linhas]
    return "\n".join([cabecalho, *corpo]).encode("latin-1")


LTN, NTNF, SELIC = tesouro.TIPO_LTN, tesouro.TIPO_NTNF, "Tesouro Selic"


def test_le_so_prefixados_converte_tipos_e_calcula_prazo():
    tabela = tesouro.ler_csv_tesouro(_csv([
        (LTN, "01/01/2028", "09/10/2026", "12,49"),
        (SELIC, "01/03/2027", "09/10/2026", "0,01"),
        (NTNF, "01/01/2031", "09/10/2026", "12,48"),
    ]), HOJE)
    assert sorted(tabela["tipo"]) == [LTN, NTNF]
    linha = tabela[tabela["tipo"] == LTN].iloc[0]
    assert linha["taxa"] == pytest.approx(12.49)
    assert linha["prazo_anos"] == pytest.approx((pd.Timestamp("2028-01-01") - pd.Timestamp("2026-10-09")).days / 365.25)


def test_taxa_zero_de_titulo_fora_de_venda_e_descartada():
    tabela = tesouro.ler_csv_tesouro(_csv([
        (NTNF, "01/01/2021", "04/03/2010", "0,00"),
        (LTN, "01/01/2012", "04/03/2010", "11,00"),
    ]), HOJE)
    assert tabela["vencimento"].tolist() == [pd.Timestamp("2012-01-01")]


def test_mesmo_vencimento_fica_a_ltn():
    tabela = tesouro.ler_csv_tesouro(_csv([
        (NTNF, "01/01/2027", "09/10/2026", "13,20"),
        (LTN, "01/01/2027", "09/10/2026", "13,28"),
    ]), HOJE)
    assert len(tabela) == 1 and tabela["tipo"].iloc[0] == LTN
    assert tabela["taxa"].iloc[0] == pytest.approx(13.28)


def test_datas_base_depois_de_hoje_ficam_de_fora():
    tabela = tesouro.ler_csv_tesouro(_csv([
        (LTN, "01/01/2028", "09/10/2026", "12,49"),
        (LTN, "01/01/2028", "14/10/2026", "99,00"),
    ]), HOJE)
    assert tabela["data_base"].max() == pd.Timestamp("2026-10-09")


def test_coluna_renomeada_levanta_erro_com_o_nome():
    cabecalho = CABECALHO.replace("Taxa Compra Manha", "Taxa Compra")
    with pytest.raises(tesouro.ErroTesouro, match="Taxa Compra Manha"):
        tesouro.ler_csv_tesouro(_csv([(LTN, "01/01/2028", "09/10/2026", "12,49")], cabecalho), HOJE)


def test_csv_sem_prefixados_levanta_erro():
    with pytest.raises(tesouro.ErroTesouro):
        tesouro.ler_csv_tesouro(_csv([(SELIC, "01/03/2027", "09/10/2026", "0,01")]), HOJE)


def test_interpolacao_linear_entre_dois_prazos():
    assert tesouro.interpolar_taxa(np.array([1.5, 0.5]), np.array([12.0, 10.0]), 1.0) == pytest.approx(11.0)


@pytest.mark.parametrize("prazos, alvo", [([0.2, 4.43], 5.0), ([1.2, 3.0], 1.0), ([], 1.0)])
def test_fora_do_intervalo_de_prazos_e_nan(prazos, alvo):
    taxas = np.full(len(prazos), 12.0)
    assert np.isnan(tesouro.interpolar_taxa(np.array(prazos), taxas, alvo))


def test_curva_fim_de_mes_usa_o_ultimo_dia_com_cotacao():
    titulos = tesouro.ler_csv_tesouro(_csv([
        (LTN, "01/07/2006", "01/09/2005", "19,00"),        # dia que não é o último do mês
        (LTN, "01/07/2006", "30/09/2005", "18,00"),        # prazo ~0,75a
        (NTNF, "01/01/2008", "30/09/2005", "17,00"),       # prazo ~2,25a
        (LTN, "01/07/2006", "31/10/2005", "17,50"),
        (NTNF, "01/01/2012", "31/10/2005", "16,00"),       # prazo ~6,17a
    ]), HOJE)
    curva = tesouro.curva_fim_de_mes(titulos)
    assert list(curva.index) == [pd.Period("2005-09", freq="M"), pd.Period("2005-10", freq="M")]
    p_curto = (pd.Timestamp("2006-07-01") - pd.Timestamp("2005-09-30")).days / 365.25
    p_longo = (pd.Timestamp("2008-01-01") - pd.Timestamp("2005-09-30")).days / 365.25
    esperado = 18.0 + (17.0 - 18.0) * (1.0 - p_curto) / (p_longo - p_curto)
    assert curva.loc[pd.Period("2005-09", freq="M"), "juros_1a"] == pytest.approx(esperado)
    assert np.isnan(curva.loc[pd.Period("2005-09", freq="M"), "juros_5a"])
    assert not np.isnan(curva.loc[pd.Period("2005-10", freq="M"), "juros_5a"])


def test_curva_mais_recente():
    titulos = tesouro.ler_csv_tesouro(_csv([
        (LTN, "01/01/2027", "09/10/2026", "13,28"),
        (LTN, "01/01/2028", "09/10/2026", "12,49"),
        (NTNF, "01/01/2033", "09/10/2026", "12,59"),
        (LTN, "01/01/2028", "08/10/2026", "99,00"),
    ]), HOJE)
    curva = tesouro.curva_mais_recente(titulos)
    assert curva.data_base == pd.Timestamp("2026-10-09")
    assert 12.49 < curva.juros_1a < 13.28
    assert 12.49 < curva.juros_5a < 12.59


class _Resposta:
    def __init__(self, status_code: int, conteudo: bytes = b""):
        self.status_code, self.content = status_code, conteudo

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error", response=self)


def test_baixar_prefixados_le_o_conteudo_baixado(monkeypatch):
    conteudo = _csv([(LTN, "01/01/2028", "09/10/2026", "12,49")])
    monkeypatch.setattr(tesouro.requests, "get", lambda url, timeout=None: _Resposta(200, conteudo))
    assert len(tesouro.baixar_prefixados(HOJE)) == 1


@pytest.mark.parametrize("falha", [requests.ConnectionError("sem rede"), _Resposta(503)])
def test_baixar_prefixados_traduz_falha_de_rede(monkeypatch, falha):
    def get(url, timeout=None):
        if isinstance(falha, Exception):
            raise falha
        return falha
    monkeypatch.setattr(tesouro.requests, "get", get)
    with pytest.raises(tesouro.ErroTesouro):
        tesouro.baixar_prefixados(HOJE)
