"""
bcb.py - Séries do Banco Central (API SGS), sem dependências do pacote pai.

Portado de ``src/tools/api_bcb.py`` do repositório pai em vez de importado:
aquele módulo depende de ``src.data.models`` (pydantic), e este pacote precisa
rodar sozinho no Cloud Run.

Peculiaridades da API (sondadas ao vivo em 2026-09-23 e 2026-10-11):
- Séries diárias recusam janelas > 10 anos com HTTP 406. Toda consulta é
  quebrada em décadas fixas (01/01/2020 a 31/12/2029, ...).
- Janela sem observações devolve HTTP 404 ("Value(s) not found"): é vazio, não falha.
- A série 432 (Selic meta) vem pré-preenchida até a próxima reunião do Copom,
  com linhas de data futura. Toda leitura corta em ``data <= hoje``.
- Datas em dd/mm/aaaa; valores em string decimal.

Cache por processo, por (série, década). Década fechada não muda mais e fica
para sempre. A década corrente expira em 6 h: num container de vida longa, um
cache eterno congelaria o CDI. Falha em qualquer janela levanta ``ErroBcb``,
porque uma série truncada em silêncio entraria no Sharpe como se fosse completa
(o ``api_bcb`` original devolve [] e trunca).

``hoje`` é sempre explícito. Nada aqui chama ``date.today()``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd
import requests

SERIE_PTAX_VENDA = 1        # R$ por US$, PTAX de venda
SERIE_CDI = 12              # CDI, % ao dia (o backtest usa decimal/dia x 252)
SERIE_SELIC_META = 432      # Selic meta, % a.a.

_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados"
_TIMEOUT_HTTP = 30
_TTL_DECADA_ABERTA = 6 * 3600
_JANELA_SELIC_DIAS = 120    # cobre feriados longos; a 432 tem linha todo dia


class ErroBcb(RuntimeError):
    """Falha ao obter uma série do BCB."""


_agora: Callable[[], float] = time.monotonic
_cache: dict[tuple[int, int], tuple[float, pd.Series]] = {}


def _serie_vazia() -> pd.Series:
    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def _baixar_decada(serie: int, decada: int) -> pd.Series:
    params = {"formato": "json", "dataInicial": f"01/01/{decada}", "dataFinal": f"31/12/{decada + 9}"}
    try:
        resposta = requests.get(_URL.format(serie=serie), params=params, timeout=_TIMEOUT_HTTP)
        if resposta.status_code == 404:
            return _serie_vazia()
        resposta.raise_for_status()
        registros = resposta.json()
        return pd.Series(
            [float(str(r["valor"]).replace(",", ".")) for r in registros],
            index=pd.to_datetime([r["data"] for r in registros], format="%d/%m/%Y"),
            dtype=float,
        )
    except (requests.RequestException, ValueError, KeyError, TypeError) as erro:
        raise ErroBcb(f"BCB SGS {serie}, janela {decada}-{decada + 9}: {erro}") from erro


def _decada(serie: int, decada: int, hoje: date) -> pd.Series:
    agora = _agora()
    em_cache = _cache.get((serie, decada))
    fechada = decada + 9 < hoje.year
    if em_cache is not None and (fechada or agora - em_cache[0] < _TTL_DECADA_ABERTA):
        return em_cache[1]
    dados = _baixar_decada(serie, decada)
    _cache[(serie, decada)] = (agora, dados)
    return dados


def baixar_serie(serie: int, inicio: date, hoje: date) -> pd.Series:
    """Observações com ``inicio <= data <= hoje``, em ordem, nos valores crus da SGS."""
    if inicio > hoje:
        raise ValueError(f"inicio ({inicio}) depois de hoje ({hoje}).")
    partes = [_decada(serie, decada, hoje)
              for decada in range(inicio.year // 10 * 10, hoje.year // 10 * 10 + 1, 10)]
    partes = [p for p in partes if not p.empty]
    if not partes:
        return _serie_vazia()
    completa = pd.concat(partes).sort_index()
    return completa.loc[pd.Timestamp(inicio):pd.Timestamp(hoje)]


def _nao_vazia(serie: pd.Series, codigo: int, inicio: date, hoje: date) -> pd.Series:
    if serie.empty:
        raise ErroBcb(f"BCB SGS {codigo} sem observações entre {inicio} e {hoje}.")
    return serie


def baixar_cdi_diario(anos: int, hoje: date) -> pd.Series:
    """CDI diário (SGS 12) em decimal ao dia, cobrindo ``anos`` até ``hoje``."""
    inicio = hoje - timedelta(days=int(anos * 365.25) + 30)
    cdi = _nao_vazia(baixar_serie(SERIE_CDI, inicio, hoje), SERIE_CDI, inicio, hoje)
    return (cdi / 100.0).rename("cdi_diario")


def baixar_ptax(inicio: date, hoje: date) -> pd.Series:
    """PTAX de venda (SGS 1), R$ por US$. Não há linhas em fins de semana e feriados."""
    ptax = _nao_vazia(baixar_serie(SERIE_PTAX_VENDA, inicio, hoje), SERIE_PTAX_VENDA, inicio, hoje)
    return ptax.rename("ptax")


def obter_selic_meta(hoje: date) -> float:
    """Selic meta vigente em ``hoje``, em decimal a.a. As linhas com data futura ficam de fora."""
    inicio = hoje - timedelta(days=_JANELA_SELIC_DIAS)
    selic = _nao_vazia(baixar_serie(SERIE_SELIC_META, inicio, hoje), SERIE_SELIC_META, inicio, hoje)
    return float(selic.iloc[-1]) / 100.0
