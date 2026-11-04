"""
parecer.py - The DeepSeek adviser's inputs for any registered portfolio: the evidence pack and the tools' sources,
built the same way for the screen's plan (app.py) and for the simulator's DeepSeek portfolio
(``scripts/carteira_deepseek.py``, DECISOES 240).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from assets import BENCHMARK_ACOES, get_all_tickers, get_ticker_info
from ativos_eua import get_ticker_info_eua
from ativos_tela import info_curada, info_da_tela
from core import assessor, bcb, precos_do_app
from core.indicadores import tabela_por_acao
from core.macro import resumo_focus
from core.sentimento import semaforo
from ui import alepha, painel, servicos


def pacote_e_fontes(alvo: pd.Series, registrada: date, chave: str,
                    hoje: date) -> tuple[list[assessor.Evidencia], assessor.Fontes]:
    """The pack on ``alvo`` (ticker.SA -> target weight, registered on ``registrada``) and the tools' sources."""
    _, fatos, fundamentos = servicos.carregar_painel()
    # the plan's own names first: the Composite holds stocks of the whole lake, outside the app's 181 (MLAS3 and SAUD3
    # reached the first opinion with no market numbers, 03/11/2026)
    tickers = tuple(dict.fromkeys((*alvo.index, *get_all_tickers(), BENCHMARK_ACOES)))
    precos = servicos.carregar_precos_da_tabela(tickers, hoje, precos_do_app.versao_dos_dados())
    curados = info_curada(get_ticker_info(), get_ticker_info_eua())
    info = {**curados, **info_da_tela(servicos.carregar_ativos_da_tela(), curados)}
    acoes = tabela_por_acao(precos, info, fundamentos, BENCHMARK_ACOES)
    focus_anual = servicos.carregar_focus(hoje)
    macro = assessor.macro_do_painel(hoje, servicos.carregar_selic(hoje), servicos.carregar_indicadores_bcb(hoje),
                                     None if focus_anual is None else resumo_focus(focus_anual, hoje.year))
    desde = precos.index[-min(alepha.SESSOES_DO_SEMAFORO, len(precos.index))].date()
    luzes = None if fatos is None else semaforo(fatos, alvo.index, desde)
    pacote = assessor.montar_pacote(plano=alvo, registrada=registrada, acoes=acoes, percentuais=painel.PERCENTUAIS,
                                    macro=macro, fatos=fatos, luzes=luzes, manchetes=servicos.carregar_manchetes()[0])
    fontes = assessor.Fontes(hoje=hoje, acoes=acoes, percentuais=painel.PERCENTUAIS, fatos=fatos,
                             serie_bcb=bcb.baixar_serie, chave_deepseek=chave,
                             chave_firecrawl=servicos.chave_da_api("firecrawl", "FIRECRAWL_API_KEY"),
                             chave_scrapecreators=servicos.chave_da_api("scrapecreators", "SCRAPECREATORS_API_KEY"))
    return pacote, fontes
