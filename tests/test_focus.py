"""
Testes de core/focus.py (sem rede).

``FocusFalso`` imita o OData do Olinda (sondado ao vivo em 2026-10-12/13):
HTTP 400 se os parâmetros vierem por ``params`` ou se o ``$filter`` não estiver
codificado; resposta ``{"value": [...]}`` com só os campos do ``$select``.
"""

import re
from datetime import date
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd
import pytest
import requests

from core import focus

HOJE = date(2026, 10, 13)


class RespostaFalsa:
    def __init__(self, status_code: int, corpo=None):
        self.status_code = status_code
        self._corpo = corpo

    def json(self):
        if not isinstance(self._corpo, (dict, list)):
            raise requests.JSONDecodeError("corpo não é JSON", "", 0)
        return self._corpo

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error", response=self)


def _linha_anual(indicador: str, data: str, ano: int, mediana: float, desvio: float | None) -> dict:
    return {"Indicador": indicador, "IndicadorDetalhe": None, "Data": data, "DataReferencia": str(ano),
            "Media": mediana, "Mediana": mediana, "DesvioPadrao": desvio, "Minimo": None,
            "Maximo": None, "numeroRespondentes": None, "baseCalculo": 0}


def _linha_ipca(data: str, mediana: float, desvio: float, suavizada: str = "S") -> dict:
    return {"Indicador": "IPCA", "Data": data, "Suavizada": suavizada, "Media": mediana,
            "Mediana": mediana, "DesvioPadrao": desvio, "Minimo": None, "Maximo": None,
            "numeroRespondentes": None, "baseCalculo": 0}


class FocusFalso:
    """Servidor OData em memória; ``linhas[endpoint]`` são os registros completos."""

    def __init__(self, linhas: dict[str, list[dict]]):
        self.linhas = linhas
        self.urls: list[str] = []

    def __call__(self, url, params=None, timeout=None):
        self.urls.append(url)
        if params is not None:
            return RespostaFalsa(400, {"error": "params não suportado"})
        partes = urlsplit(url)
        consulta = dict(parte.split("=", 1) for parte in partes.query.split("&"))
        if re.search(r"[ ']", consulta.get("$filter", "")):
            return RespostaFalsa(400, {"error": "filtro sem codificação"})
        filtro = unquote(consulta["$filter"])
        if "baseCalculo eq 0" not in filtro:
            return RespostaFalsa(200, {"value": []})
        indicador = re.search(r"Indicador eq '([^']+)'", filtro).group(1)
        suavizada = re.search(r"Suavizada eq '([^']+)'", filtro)
        campos = consulta["$select"].split(",")
        endpoint = partes.path.rsplit("/", 1)[-1]
        valor = [{c: r[c] for c in campos if c in r}
                 for r in self.linhas.get(endpoint, [])
                 if r["Indicador"] == indicador
                 and (suavizada is None or r.get("Suavizada") == suavizada.group(1))]
        return RespostaFalsa(200, {"value": valor})


@pytest.fixture
def servidor(monkeypatch):
    falso = FocusFalso({
        focus.ENDPOINT_ANUAIS: [
            _linha_anual("Câmbio", "2026-10-09", 2026, 5.1535, 0.1612),
            _linha_anual("Câmbio", "2026-10-09", 2027, 5.26, 0.204),
            _linha_anual("Câmbio", "2026-10-14", 2026, 9.99, 0.1),     # depois de HOJE
            _linha_anual("Selic", "2026-10-09", 2027, 12.00, 0.7543),
            _linha_anual("Selic", "2026-10-09", 2026, 13.25, 0.2236),
            _linha_anual("PIB Total", "1999-07-01", 1999, -1.0, None),
        ],
        focus.ENDPOINT_IPCA_12M: [
            _linha_ipca("2026-10-08", 4.5221, 0.4855),
            _linha_ipca("2026-10-09", 4.4772, 0.495),
            _linha_ipca("2026-10-09", 9.99, 9.99, suavizada="N"),
        ],
    })
    monkeypatch.setattr(focus.requests, "get", falso)
    return falso


def test_filtro_vai_codificado_na_url_e_nao_por_params(servidor):
    """O FocusFalso devolve 400 para ``params`` ou filtro cru, como o Olinda real."""
    cambio = focus.baixar_focus_anual(focus.INDICADOR_CAMBIO, HOJE)
    assert len(cambio) == 2
    assert "%C3%A2" in servidor.urls[0]          # "â" de Câmbio, em UTF-8
    assert " " not in servidor.urls[0]


def test_anual_converte_tipos_ordena_e_corta_pesquisas_futuras(servidor):
    tabela = focus.baixar_focus_anual(focus.INDICADOR_SELIC, HOJE)
    assert list(tabela.columns) == ["data", "ano_referencia", "mediana", "desvio"]
    assert list(tabela["ano_referencia"]) == [2026, 2027]
    assert tabela["ano_referencia"].dtype.kind == "i"
    assert tabela["mediana"].tolist() == [13.25, 12.00]

    cambio = focus.baixar_focus_anual(focus.INDICADOR_CAMBIO, HOJE)
    assert cambio["data"].max() == pd.Timestamp("2026-10-09")


def test_desvio_nulo_das_pesquisas_antigas_vira_nan(servidor):
    pib = focus.baixar_focus_anual(focus.INDICADOR_PIB, HOJE)
    assert len(pib) == 1 and np.isnan(pib["desvio"].iloc[0])


def test_ipca_12m_usa_so_a_serie_suavizada(servidor):
    ipca = focus.baixar_focus_ipca_12m(HOJE)
    assert ipca["mediana"].tolist() == [4.5221, 4.4772]


def test_resposta_que_nao_e_json_levanta_erro(monkeypatch):
    monkeypatch.setattr(focus.requests, "get", lambda *a, **k: RespostaFalsa(200, "<html>manutenção</html>"))
    with pytest.raises(focus.ErroFocus):
        focus.baixar_focus_ipca_12m(HOJE)


def test_http_500_levanta_erro(monkeypatch):
    monkeypatch.setattr(focus.requests, "get", lambda *a, **k: RespostaFalsa(500, {}))
    with pytest.raises(focus.ErroFocus):
        focus.baixar_focus_anual(focus.INDICADOR_PIB, HOJE)


def test_campo_faltando_levanta_erro_com_o_nome_do_campo(monkeypatch):
    linha = _linha_ipca("2026-10-09", 4.4772, 0.495)
    del linha["DesvioPadrao"]
    monkeypatch.setattr(focus.requests, "get", FocusFalso({focus.ENDPOINT_IPCA_12M: [linha]}))
    with pytest.raises(focus.ErroFocus, match="DesvioPadrao"):
        focus.baixar_focus_ipca_12m(HOJE)


def test_sem_pesquisas_ate_hoje_levanta_erro(servidor):
    with pytest.raises(focus.ErroFocus):
        focus.baixar_focus_ipca_12m(date(2001, 1, 1))


@pytest.mark.parametrize("data, peso", [("2026-01-31", 1.0), ("2026-10-09", 0.25), ("2026-12-01", 1 / 12)])
def test_peso_do_ano_corrente(data, peso):
    assert focus.peso_ano_corrente(pd.Timestamp(data)) == pytest.approx(peso)
    assert focus.peso_ano_corrente(pd.Period(data, freq="M")) == pytest.approx(peso)


def _anual(linhas: list[tuple[str, int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame({"data": pd.to_datetime([l[0] for l in linhas]),
                         "ano_referencia": [l[1] for l in linhas],
                         "mediana": [l[2] for l in linhas], "desvio": [l[3] for l in linhas]})


def test_expectativa_12m_pondera_ano_corrente_e_seguinte():
    """Outubro: 3/12 do ano corrente + 9/12 do seguinte (Selic Focus de 2026-10-09)."""
    anual = _anual([("2026-10-09", 2026, 13.25, 0.2236), ("2026-10-09", 2027, 12.00, 0.7543)])
    exp = focus.expectativa_12m(anual, pd.Timestamp("2026-10-13"))
    assert exp.mediana == pytest.approx(0.25 * 13.25 + 0.75 * 12.00)
    assert exp.desvio == pytest.approx(0.25 * 0.2236 + 0.75 * 0.7543)
    assert exp.data == pd.Timestamp("2026-10-09")


def test_expectativa_12m_usa_a_ultima_pesquisa_ate_a_data():
    anual = _anual([("2026-09-30", 2026, 14.0, 0.1), ("2026-09-30", 2027, 13.0, 0.1),
                    ("2026-10-09", 2026, 13.25, 0.2), ("2026-10-09", 2027, 12.0, 0.2)])
    exp = focus.expectativa_12m(anual, pd.Timestamp("2026-10-01"))
    assert exp.data == pd.Timestamp("2026-09-30")
    assert exp.mediana == pytest.approx((4 / 12) * 14.0 + (8 / 12) * 13.0)


def test_expectativa_12m_em_janeiro_dispensa_o_ano_seguinte():
    anual = _anual([("2027-01-08", 2027, 11.0, 0.5)])
    exp = focus.expectativa_12m(anual, pd.Timestamp("2027-01-10"))
    assert exp.mediana == pytest.approx(11.0)


def test_expectativa_12m_sem_o_ano_seguinte_levanta_erro():
    anual = _anual([("2026-10-09", 2026, 13.25, 0.2)])
    with pytest.raises(focus.ErroFocus, match="2027"):
        focus.expectativa_12m(anual, pd.Timestamp("2026-10-13"))


def test_mediana_mensal_por_ano_pega_a_ultima_pesquisa_do_mes():
    anual = _anual([("2026-09-01", 2026, 1.0, 0.1), ("2026-09-30", 2026, 2.0, 0.1),
                    ("2026-09-30", 2027, 3.0, 0.1), ("2026-10-09", 2026, 4.0, 0.1)])
    mensal = focus.mediana_mensal_por_ano(anual)
    assert isinstance(mensal.index, pd.PeriodIndex)
    assert mensal.loc[pd.Period("2026-09", freq="M"), 2026] == 2.0
    assert mensal.loc[pd.Period("2026-09", freq="M"), 2027] == 3.0
    assert mensal.loc[pd.Period("2026-10", freq="M"), 2026] == 4.0
    assert np.isnan(mensal.loc[pd.Period("2026-10", freq="M"), 2027])


def test_mediana_mensal_da_serie_12m():
    serie = pd.DataFrame({"data": pd.to_datetime(["2026-09-02", "2026-09-29", "2026-10-09"]),
                          "mediana": [4.8, 4.6, 4.5], "desvio": [0.5, 0.5, 0.5]})
    mensal = focus.mediana_mensal(serie)
    assert mensal.to_dict() == {pd.Period("2026-09", freq="M"): 4.6, pd.Period("2026-10", freq="M"): 4.5}
