"""
Testes de core/bcb.py e da integração BCB em core/market_data.py (sem rede).

``SgsFalso`` imita o comportamento observado da API SGS (sondada ao vivo em
2026-09-23 e 2026-10-11): HTTP 406 para janela > 10 anos, HTTP 404 para
janela sem observações, datas dd/mm/aaaa e valores em string decimal.
"""

import re
from datetime import date, datetime

import pandas as pd
import pytest
import requests

from core import bcb, market_data

HOJE = date(2026, 10, 12)          # segunda-feira
SEIS_HORAS = 6 * 3600


class RespostaFalsa:
    def __init__(self, status_code: int, registros: list[dict] | None = None):
        self.status_code = status_code
        self._registros = registros
        self.text = "Value(s) not found" if status_code == 404 else ""

    def json(self):
        if self._registros is None:
            raise ValueError("corpo não é JSON")
        return self._registros

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error", response=self)


class SgsFalso:
    """Servidor SGS em memória; registra toda chamada como (série, início, fim)."""

    def __init__(self, series: dict[int, dict[date, str]]):
        self.series = series
        self.falhas: dict[tuple[int, int], int] = {}     # (série, ano inicial da janela) -> HTTP
        self.chamadas: list[tuple[int, date, date]] = []

    def __call__(self, url, params=None, timeout=None):
        codigo = int(re.search(r"bcdata\.sgs\.(\d+)/dados$", url).group(1))
        inicio = datetime.strptime(params["dataInicial"], "%d/%m/%Y").date()
        fim = datetime.strptime(params["dataFinal"], "%d/%m/%Y").date()
        self.chamadas.append((codigo, inicio, fim))
        if (codigo, inicio.year) in self.falhas:
            return RespostaFalsa(self.falhas[(codigo, inicio.year)])
        if fim > inicio.replace(year=inicio.year + 10):
            return RespostaFalsa(406)
        linhas = [{"data": d.strftime("%d/%m/%Y"), "valor": v}
                  for d, v in sorted(self.series.get(codigo, {}).items()) if inicio <= d <= fim]
        return RespostaFalsa(200, linhas) if linhas else RespostaFalsa(404)


def _cdi_por_dia_util(primeiro: str, ultimo: str) -> dict[date, str]:
    """CDI em % ao dia; o valor muda por ano para denunciar desalinhamento."""
    return {d.date(): f"{0.030 + 0.001 * (d.year - 2010):.6f}"
            for d in pd.bdate_range(primeiro, ultimo)}


@pytest.fixture(autouse=True)
def cache_isolado(monkeypatch):
    monkeypatch.setattr(bcb, "_cache", {})


@pytest.fixture
def relogio(monkeypatch):
    agora = [1_000_000.0]
    monkeypatch.setattr(bcb, "_agora", lambda: agora[0])
    return agora


@pytest.fixture
def sgs(monkeypatch):
    servidor = SgsFalso({12: _cdi_por_dia_util("2012-01-02", "2026-10-09")})
    monkeypatch.setattr(bcb.requests, "get", servidor)
    return servidor


@pytest.mark.parametrize("anos", [10, 12])
def test_cdi_com_10_anos_ou_mais_cruza_decadas_sem_buraco(sgs, anos):
    """O SGS recusa janela > 10 anos (406): o histórico máximo do app (10a) falhava."""
    cdi = bcb.baixar_cdi_diario(anos, HOJE)

    for _, inicio, fim in sgs.chamadas:
        assert fim <= inicio.replace(year=inicio.year + 10)
    inicio_esperado = date(2016, 9, 12) if anos == 10 else date(2014, 9, 12)
    esperado = [pd.Timestamp(d) for d in sorted(sgs.series[12]) if d >= inicio_esperado]
    assert list(cdi.index) == esperado
    assert cdi[pd.Timestamp("2019-12-31")] == pytest.approx(0.039 / 100)
    assert cdi[pd.Timestamp("2020-01-01")] == pytest.approx(0.040 / 100)


def test_janela_sem_observacoes_e_vazio_e_nao_erro(monkeypatch):
    """404 = 'Value(s) not found' (década antes do início da série): não é falha."""
    servidor = SgsFalso({12: _cdi_por_dia_util("2021-03-01", "2026-10-09")})
    monkeypatch.setattr(bcb.requests, "get", servidor)

    cdi = bcb.baixar_cdi_diario(12, HOJE)

    assert cdi.index[0] == pd.Timestamp("2021-03-01")
    assert cdi.index[-1] == pd.Timestamp("2026-10-09")


def test_falha_em_uma_decada_levanta_erro_e_nao_envenena_o_cache(sgs):
    sgs.falhas[(12, 2020)] = 500
    with pytest.raises(bcb.ErroBcb):
        bcb.baixar_cdi_diario(12, HOJE)

    del sgs.falhas[(12, 2020)]
    cdi = bcb.baixar_cdi_diario(12, HOJE)
    assert cdi.index[-1] == pd.Timestamp("2026-10-09")
    assert len(cdi) == len([d for d in sgs.series[12] if d >= date(2014, 9, 12)])


def test_resposta_que_nao_e_json_levanta_erro(monkeypatch):
    monkeypatch.setattr(bcb.requests, "get", lambda *a, **k: RespostaFalsa(200, None))
    with pytest.raises(bcb.ErroBcb):
        bcb.baixar_cdi_diario(5, HOJE)


def test_decada_fechada_fica_em_cache_e_a_corrente_expira_em_6h(sgs, relogio):
    bcb.baixar_cdi_diario(12, HOJE)
    assert {inicio.year for _, inicio, _ in sgs.chamadas} == {2010, 2020}

    relogio[0] += SEIS_HORAS - 1
    bcb.baixar_cdi_diario(12, HOJE)
    assert len(sgs.chamadas) == 2

    relogio[0] += 2
    bcb.baixar_cdi_diario(12, HOJE)
    assert len(sgs.chamadas) == 3
    assert sgs.chamadas[-1][1] == date(2020, 1, 1)


def test_cache_separa_series(sgs):
    sgs.series[1] = {date(2026, 10, 9): "4.9892"}
    bcb.baixar_cdi_diario(5, HOJE)
    ptax = bcb.baixar_ptax(date(2026, 10, 1), HOJE)
    assert list(ptax) == [pytest.approx(4.9892)]


def test_selic_ignora_linhas_pre_preenchidas_com_data_futura(monkeypatch):
    """A SGS 432 traz a meta repetida até a próxima reunião do Copom (data > hoje)."""
    linhas = {d.date(): "14.25" for d in pd.date_range("2026-07-01", "2026-10-12")}
    linhas |= {d.date(): "13.75" for d in pd.date_range("2026-10-13", "2026-11-04")}
    monkeypatch.setattr(bcb.requests, "get", SgsFalso({432: linhas}))

    assert bcb.obter_selic_meta(HOJE) == pytest.approx(0.1425)
    assert bcb.obter_selic_meta(date(2026, 11, 3)) == pytest.approx(0.1375)


def test_market_data_traduz_erro_bcb(sgs):
    sgs.falhas[(12, 2020)] = 503
    with pytest.raises(market_data.ErroDadosMercado):
        market_data.baixar_cdi_diario(5, hoje=HOJE)


def test_selic_cai_no_fallback_se_o_bcb_falhar(monkeypatch):
    monkeypatch.setattr(bcb.requests, "get", lambda *a, **k: RespostaFalsa(500))
    taxa = market_data.obter_selic_meta(hoje=HOJE)
    assert taxa.valor == market_data.SELIC_FALLBACK
    assert "fallback" in taxa.fonte


# ---------------------------------------------------------------------------
# Câmbio: PTAX (BCB) como fonte principal, Yahoo USDBRL=X como reserva
# ---------------------------------------------------------------------------

def _yahoo_falso(pedidos_registrados: list[list[str]]):
    """Yahoo: ação B3 em dias úteis; BTC 24/7 a US$ 100; USDBRL=X a 4,0."""
    def baixar(tickers, inicio, fim):
        pedidos_registrados.append(list(tickers))
        dias = pd.date_range(inicio, fim)
        precos = pd.DataFrame(index=dias)
        for t in tickers:
            if t == "BTC-USD":
                precos[t] = 100.0
            elif t == "USDBRL=X":
                precos[t] = pd.Series(4.0, index=dias[dias.dayofweek < 5])
            else:
                precos[t] = pd.Series(10.0, index=dias[dias.dayofweek < 5])
        return precos
    return baixar


def test_ptax_converte_cripto_e_dispensa_o_yahoo(monkeypatch):
    pedidos: list[list[str]] = []
    monkeypatch.setattr(market_data, "baixar_fechamentos", _yahoo_falso(pedidos))
    ptax = {d.date(): "5.0000" for d in pd.bdate_range("2024-08-01", "2026-10-09")}
    monkeypatch.setattr(bcb.requests, "get", SgsFalso({1: ptax}))

    dados = market_data.carregar_precos(["AAA.SA", "BTC-USD"], 2, hoje=HOJE)

    assert "USDBRL=X" not in pedidos[0]
    assert dados.precos["BTC-USD"].notna().all()
    assert (dados.precos["BTC-USD"] == 500.0).all()
    assert "PTAX" in dados.fonte_cambio


def test_cambio_cai_no_yahoo_se_o_bcb_falhar(monkeypatch):
    pedidos: list[list[str]] = []
    monkeypatch.setattr(market_data, "baixar_fechamentos", _yahoo_falso(pedidos))
    monkeypatch.setattr(bcb.requests, "get", lambda *a, **k: RespostaFalsa(500))

    dados = market_data.carregar_precos(["AAA.SA", "BTC-USD"], 2, hoje=HOJE)

    assert "USDBRL=X" in pedidos[0]
    assert (dados.precos["BTC-USD"].dropna() == 400.0).all()
    assert "Yahoo" in dados.fonte_cambio


def test_cambio_preenche_dias_sem_ptax_com_a_ultima_cotacao():
    """O preço do BTC no sábado usa a PTAX de sexta; datas da PTAX fora da grade de preços contam."""
    sabado, domingo = pd.Timestamp("2026-10-10"), pd.Timestamp("2026-10-11")
    precos = pd.DataFrame({"BTC-USD": [100.0, 100.0]}, index=[sabado, domingo])
    ptax = pd.Series([5.0], index=[pd.Timestamp("2026-10-09")])

    convertido = market_data.converter_usd_para_brl(precos, ptax)

    assert list(convertido["BTC-USD"]) == [500.0, 500.0]


def test_carteira_so_de_acoes_nao_consulta_cambio(monkeypatch):
    pedidos: list[list[str]] = []
    monkeypatch.setattr(market_data, "baixar_fechamentos", _yahoo_falso(pedidos))

    def proibido(*a, **k):
        raise AssertionError("não deveria consultar o BCB")
    monkeypatch.setattr(bcb.requests, "get", proibido)

    dados = market_data.carregar_precos(["AAA.SA", "BBB.SA"], 2, hoje=HOJE)

    assert pedidos == [["AAA.SA", "BBB.SA"]]
    assert dados.fonte_cambio is None
