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
        self.transitorias: dict[tuple[int, int], list[int]] = {}   # HTTPs devolvidos antes de responder
        self.chamadas: list[tuple[int, date, date]] = []

    def __call__(self, url, params=None, timeout=None):
        codigo = int(re.search(r"bcdata\.sgs\.(\d+)/dados$", url).group(1))
        inicio = datetime.strptime(params["dataInicial"], "%d/%m/%Y").date()
        fim = datetime.strptime(params["dataFinal"], "%d/%m/%Y").date()
        self.chamadas.append((codigo, inicio, fim))
        pendentes = self.transitorias.get((codigo, inicio.year))
        if pendentes:
            return RespostaFalsa(pendentes.pop(0))
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


@pytest.fixture(autouse=True)
def esperas(monkeypatch):
    """Sem sleep real entre tentativas; registra as esperas pedidas."""
    registradas: list[float] = []
    monkeypatch.setattr(bcb, "_dormir", registradas.append)
    return registradas


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


def _chamadas_da_janela(sgs: SgsFalso, ano: int) -> int:
    return sum(1 for _, inicio, _ in sgs.chamadas if inicio.year == ano)


def test_erro_5xx_transitorio_tenta_de_novo_e_recupera(sgs, esperas):
    """O SGS devolve 502 intermitente (sonda de 2026-10-13: 20,7 s, 502, 25,8 s na mesma janela)."""
    sgs.transitorias[(12, 2020)] = [502]
    cdi = bcb.baixar_cdi_diario(5, HOJE)
    assert cdi.index[-1] == pd.Timestamp("2026-10-09")
    assert _chamadas_da_janela(sgs, 2020) == 2
    assert esperas == [bcb._ESPERA_ENTRE_TENTATIVAS_S]


def test_timeout_transitorio_tenta_de_novo_e_recupera(sgs, monkeypatch):
    estourou: list[bool] = []

    def get(url, params=None, timeout=None):
        if not estourou:
            estourou.append(True)
            raise requests.ReadTimeout("Read timed out.")
        return sgs(url, params=params, timeout=timeout)
    monkeypatch.setattr(bcb.requests, "get", get)

    assert bcb.baixar_cdi_diario(5, HOJE).index[-1] == pd.Timestamp("2026-10-09")


def test_falha_5xx_persistente_desiste_apos_as_tentativas(sgs, esperas):
    sgs.falhas[(12, 2020)] = 503
    with pytest.raises(bcb.ErroBcb, match="503"):
        bcb.baixar_cdi_diario(5, HOJE)
    assert _chamadas_da_janela(sgs, 2020) == bcb._TENTATIVAS > 1
    assert len(esperas) == bcb._TENTATIVAS - 1


@pytest.mark.parametrize("status", [400, 406])
def test_erro_do_cliente_nao_tenta_de_novo(sgs, esperas, status):
    sgs.falhas[(12, 2020)] = status
    with pytest.raises(bcb.ErroBcb):
        bcb.baixar_cdi_diario(5, HOJE)
    assert _chamadas_da_janela(sgs, 2020) == 1 and esperas == []


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

