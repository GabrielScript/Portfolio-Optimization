"""
Avaliação PRÉ-REGISTRADA da janela de estimação (rolling 2, 3 e 5 anos e Sigma EWMA), COM REDE (~5 min).

Uso: .venv/Scripts/python -X utf8 scripts/avaliar_janelas.py [AAAA-MM-DD]

Pré-registro (2026-10-15, escrito antes de rodar; mesmo texto do §B do checkpoint da sessão 8 no session_state.md):
- Dados: lista líquida (`assets.get_tickers_liquidos()`, sem o BOVA11), 13 anos de preços do Yahoo
  (`market_data.carregar_precos`, cobertura ≥ 95%), amostra comum das ações e do BOVA11. Sonda de 2026-10-15 (sem rodar
  estratégia): 52 ações, amostra 24/01/2014 a 15/10/2026 (3.166 pregões), fora da amostra desde 19/02/2019 (~7,6 anos,
  31 rebalanceamentos), CDI de 13 anos disponível. Os saltos suspeitos do Yahoo (PCAR3, UGPA3, PRIO3, AXIA3) ficam como
  estão neste tiro (decisão do usuário).
- Walk-forward: treino de 1.260 pregões, rebalanceamento a cada 63, 10 bps por lado, capital 10 mil, taxa livre de
  risco = CDI na data (Selic constante se o BCB falhar). Janelas menores recortam o fim do treino (`com_janela`): todas
  decidem nas mesmas datas, com as mesmas ações.
- Otimização: Kelly λ = 2, peso máximo 20%, até 10 ativos, teto de vol 40%, teto de 30% por setor, Jorion ligado,
  sem reamostragem, sem Growth.
- Candidatas (N = 4): R5 (controle: μ e Σ Ledoit-Wolf em 1.260 pregões, padrão do app), R3 (756), R2 (504), EWMA
  (μ em 1.260 pregões; Σ EWMA com meia-vida de 126 pregões e encolhimento Ledoit-Wolf; o Jorion usa esse Σ).
- Regra: uma desafiante substitui a R5 só se passar em TODOS: C1 P[desafiante > R5] ≥ 1 − 0,05/3 = 0,9833 (PSR da
  série diferença diária contra 0; Bonferroni para 3); C2 drawdown máximo no máximo 5 p.p. pior que o da R5; C3 giro
  anual ≤ 2 × o da R5. Se mais de uma passar, fica a de maior Sharpe. Falhas do otimizador são reportadas e não
  excluem ninguém (o motor mantém a carteira). O CDI se cancela na diferença: a falta dele não muda a decisão.
- Dois tiros: este (Yahoo) e outro depois do S1-P, com o mesmo script e os mesmos limites. A janela do app só muda se
  a MESMA desafiante passar nos dois.
- Só reportados: retorno total, CAGR, vol, Sharpe, Sortino, MDD, CVaR 95% diário, giro, PSR, DSR (N = 4), P[> R5],
  falhas; 1/N e BOVA11 como referência.
- Expectativa registrada: erro do Sharpe ≈ ±0,4 em 7,6 anos → o mais provável é "inconclusivo, fica R5".
Saída: código 0 se uma desafiante passar, 2 se ficar a R5 (resultado, não erro).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assets import BENCHMARK_ACOES, get_ticker_info, get_tickers_liquidos  # noqa: E402
from core.backtest import (Estrategia, ResultadoBacktest, com_janela, estrategia_buy_and_hold_ativo,  # noqa: E402
                           estrategia_equiponderada, estrategia_otimizada, walk_forward)
from core import market_data  # noqa: E402
from core.config import (DIAS_UTEIS_ANO, ConfigBacktest, ConfigOtimizacao, Objetivo,  # noqa: E402
                         limites_por_setor, taxa_anual_para_aritmetica)
from core.estimation import retornos_simples  # noqa: E402
from core.metrics import (momentos_sharpe, resumo_desempenho, retornos_de_valores,  # noqa: E402
                          sharpe_probabilistico, sharpe_referencia_deflacionado)

ANOS_DE_DADOS = 13
CONTROLE = "R5"
MEIA_VIDA_EWMA = 126                        # pregões (6 meses)
PREGOES_DE_TREINO = {"R5": 5 * DIAS_UTEIS_ANO, "R3": 3 * DIAS_UTEIS_ANO, "R2": 2 * DIAS_UTEIS_ANO,
                     "EWMA": 5 * DIAS_UTEIS_ANO}
# Um cfg só para candidatas e referências: treino da maior janela, mesmas datas, mesmos custos.
CFG_BT = ConfigBacktest(janela_treino=max(PREGOES_DE_TREINO.values()), janela_teste=63,
                        capital_inicial=10_000.0, custo_bps=10.0, shrink_mu=True)
CFG_KELLY = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.20, n_ativos_max=10, vol_maxima=0.40,
                             aversao_risco=2.0)
TETO_SETOR = 0.30
N_DESAFIANTES = 3
LIMIAR_P = 1.0 - 0.05 / N_DESAFIANTES       # Bonferroni: 0,95 dividido entre os 3 desafiantes
PIORA_MAXIMA_DRAWDOWN = 0.05                # em pontos do drawdown máximo
GIRO_MAXIMO_RELATIVO = 2.0                  # giro anual da desafiante <= 2x o do controle


@dataclass(frozen=True)
class Dados:
    retornos: pd.DataFrame       # amostra comum das ações e do BOVA11
    acoes: list[str]
    setores: dict[str, str]      # ticker -> setor
    cdi: pd.Series               # vazia se o BCB falhar
    selic: float                 # Selic meta de hoje (fallback da taxa livre de risco)


def montar_dados(precos: pd.DataFrame, cdi: pd.Series, selic: float) -> Dados:
    """``Dados`` a partir dos preços já limpos (``carregar_precos``): ações = colunas menos o BOVA11."""
    if BENCHMARK_ACOES not in precos.columns:
        raise market_data.ErroDadosMercado(f"{BENCHMARK_ACOES} sem preços: o teste não roda sem a referência.")
    acoes = [ticker for ticker in precos.columns if ticker != BENCHMARK_ACOES]
    info = get_ticker_info()
    return Dados(retornos=retornos_simples(precos[acoes + [BENCHMARK_ACOES]]), acoes=acoes,
                 setores={ticker: info[ticker]["setor"] for ticker in acoes}, cdi=cdi, selic=selic)


def cdi_ou_vazio(hoje: date) -> pd.Series:
    """CDI diário; se o BCB falhar, série vazia: a taxa livre de risco vira a Selic constante e a decisão não muda."""
    try:
        return market_data.baixar_cdi_diario(ANOS_DE_DADOS, hoje=hoje)
    except market_data.ErroDadosMercado as erro:
        print(f"AVISO: CDI indisponível ({erro}); taxa livre de risco = Selic constante.")
        return pd.Series(dtype=float)


def carregar(hoje: date) -> Dados:
    """Rede (Yahoo e BCB). Imprime o universo e a amostra para conferir com a sonda do pré-registro."""
    pedidas = [ticker for ticker in get_tickers_liquidos() if ticker != BENCHMARK_ACOES]
    precos = market_data.carregar_precos(pedidas + [BENCHMARK_ACOES], ANOS_DE_DADOS, hoje=hoje)
    selic = market_data.obter_selic_meta(hoje)
    dados = montar_dados(precos.precos, cdi_ou_vazio(hoje), selic.valor)
    datas = dados.retornos.index
    print(f"Universo: {len(dados.acoes)} de {len(pedidas)} ações; amostra comum {datas[0]:%d/%m/%Y} a "
          f"{datas[-1]:%d/%m/%Y} ({len(datas)} pregões); Selic {selic.valor:.2%} ({selic.fonte}); "
          f"CDI {'indisponível' if dados.cdi.empty else 'ok'}")
    print(f"Descartados sem dados: {precos.descartados_sem_dados}")
    print(f"Descartados por cobertura: {precos.descartados_cobertura}")
    return dados


def estrategias(cfg: ConfigOtimizacao, cdi_diario: pd.Series | None) -> dict[str, Estrategia]:
    """As 4 candidatas, cada uma no próprio recorte do treino (o walk-forward entrega o treino da maior)."""
    return {
        nome: com_janela(estrategia_otimizada(cfg, True, cdi_diario,
                                              meia_vida_cov=MEIA_VIDA_EWMA if nome == "EWMA" else None),
                         pregoes)
        for nome, pregoes in PREGOES_DE_TREINO.items()
    }


def rodar(retornos: pd.DataFrame, acoes: list[str], cfg: ConfigOtimizacao,
          cdi_diario: pd.Series | None) -> dict[str, ResultadoBacktest]:
    """Candidatas, 1/N e BOVA11 no mesmo walk-forward. ``retornos``: amostra comum das ações e do BOVA11."""
    resultados = {nome: walk_forward(retornos[acoes], estrategia, CFG_BT, nome)
                  for nome, estrategia in estrategias(cfg, cdi_diario).items()}
    resultados["1/N"] = walk_forward(retornos[acoes], estrategia_equiponderada(), CFG_BT, "1/N")
    resultados[BENCHMARK_ACOES] = walk_forward(retornos, estrategia_buy_and_hold_ativo(BENCHMARK_ACOES),
                                               CFG_BT, BENCHMARK_ACOES)
    return resultados


@dataclass(frozen=True)
class Medidas:
    """O que a regra de decisão usa de cada estratégia no walk-forward."""

    sharpe: float
    max_drawdown: float          # <= 0
    giro_anual: float
    p_supera_controle: float     # PSR da diferença diária contra o controle (NaN no próprio controle)


def medir(resultados: Mapping[str, ResultadoBacktest], rf_diaria: pd.Series,
          controle: str = CONTROLE) -> dict[str, Medidas]:
    """
    Medidas de cada estratégia. O P vem da série diferença diária (estratégia - controle):
    a taxa livre de risco se cancela nela, então a falta do CDI não muda a decisão.
    """
    retornos_controle = retornos_de_valores(resultados[controle].valores)
    medidas = {}
    for nome, resultado in resultados.items():
        resumo = resumo_desempenho(resultado.valores, rf_diaria)
        diferenca = retornos_de_valores(resultado.valores) - retornos_controle
        p = float("nan") if nome == controle else sharpe_probabilistico(momentos_sharpe(diferenca))
        medidas[nome] = Medidas(resumo.sharpe, resumo.max_drawdown, resultado.turnover_anual_medio, p)
    return medidas


@dataclass(frozen=True)
class Veredito:
    escolhida: str
    criterios: dict[str, dict[str, bool]]     # desafiante -> {critério: passou}


def criterios(desafiante: Medidas, controle: Medidas) -> dict[str, bool]:
    """Os três critérios pré-registrados. NaN reprova: toda comparação com NaN é falsa."""
    return {
        f"C1 P[supera {CONTROLE}] >= {LIMIAR_P:.4f}": desafiante.p_supera_controle >= LIMIAR_P,
        f"C2 drawdown no máximo {PIORA_MAXIMA_DRAWDOWN:.0%} pior":
            desafiante.max_drawdown >= controle.max_drawdown - PIORA_MAXIMA_DRAWDOWN,
        f"C3 giro <= {GIRO_MAXIMO_RELATIVO:g}x": desafiante.giro_anual <= GIRO_MAXIMO_RELATIVO * controle.giro_anual,
    }


def decidir(medidas: Mapping[str, Medidas], controle: str = CONTROLE) -> Veredito:
    """A desafiante aprovada de maior Sharpe; sem aprovada, o controle."""
    avaliacoes = {nome: criterios(m, medidas[controle]) for nome, m in medidas.items() if nome != controle}
    aprovadas = [nome for nome, resultado in avaliacoes.items() if all(resultado.values())]
    return Veredito(max(aprovadas, key=lambda nome: medidas[nome].sharpe, default=controle), avaliacoes)


def montar_tabela(resultados: Mapping[str, ResultadoBacktest], medidas: Mapping[str, Medidas],
                  rf_diaria: pd.Series) -> pd.DataFrame:
    """Uma linha por estratégia. DSR só nas candidatas (N = 4): as referências não foram escolhidas."""
    linhas, momentos = {}, {}
    for nome, resultado in resultados.items():
        resumo = resumo_desempenho(resultado.valores, rf_diaria)
        retornos = retornos_de_valores(resultado.valores)
        momentos[nome] = momentos_sharpe(retornos - rf_diaria.reindex(retornos.index).fillna(0.0))
        linhas[nome] = {"Retorno total": resumo.retorno_total, "CAGR": resumo.cagr, "Vol": resumo.volatilidade,
                        "Sharpe": resumo.sharpe, "Sortino": resumo.sortino, "MDD": resumo.max_drawdown,
                        "CVaR95 diário": resumo.cvar_95_diario, "Giro anual": resultado.turnover_anual_medio,
                        "PSR": sharpe_probabilistico(momentos[nome]), "DSR": float("nan"),
                        f"P[> {CONTROLE}]": medidas[nome].p_supera_controle, "Falhas": len(resultado.falhas)}
    sharpes = [momentos[nome].sharpe for nome in PREGOES_DE_TREINO]
    referencia = sharpe_referencia_deflacionado(float(np.var(sharpes, ddof=1)), len(sharpes))
    for nome in PREGOES_DE_TREINO:
        linhas[nome]["DSR"] = sharpe_probabilistico(momentos[nome], referencia)
    return pd.DataFrame(linhas).T


@dataclass(frozen=True)
class Avaliacao:
    resultados: dict[str, ResultadoBacktest]
    medidas: dict[str, Medidas]
    veredito: Veredito
    tabela: pd.DataFrame


def avaliar(dados: Dados) -> Avaliacao:
    """O teste inteiro sobre dados já carregados (sem rede)."""
    cfg = replace(CFG_KELLY, taxa_livre_risco=taxa_anual_para_aritmetica(dados.selic),
                  grupos=limites_por_setor(dados.setores, TETO_SETOR))
    resultados = rodar(dados.retornos, dados.acoes, cfg, dados.cdi if not dados.cdi.empty else None)
    rf = market_data.serie_rf_diaria(resultados[CONTROLE].valores.index, dados.cdi, dados.selic)
    medidas = medir(resultados, rf)
    veredito = decidir({nome: medidas[nome] for nome in PREGOES_DE_TREINO})
    return Avaliacao(resultados, medidas, veredito, montar_tabela(resultados, medidas, rf))


def _imprimir(avaliacao: Avaliacao) -> None:
    """Tabela, falhas do otimizador e cada critério de cada desafiante (só ASCII e Latin-1 na saída)."""
    tabela = avaliacao.tabela.astype(float)
    formatada = tabela.copy().astype(object)
    for coluna in ["Retorno total", "CAGR", "Vol", "MDD", "CVaR95 diário", "Giro anual"]:
        formatada[coluna] = (tabela[coluna] * 100).map("{:.2f}%".format)
    for coluna in ["Sharpe", "Sortino", "PSR", "DSR", f"P[> {CONTROLE}]"]:
        formatada[coluna] = tabela[coluna].map("{:.3f}".format)
    formatada["Falhas"] = tabela["Falhas"].astype(int)
    datas = avaliacao.resultados[CONTROLE].valores.index
    print(f"\nWalk-forward fora da amostra, {datas[0]:%d/%m/%Y} a {datas[-1]:%d/%m/%Y} "
          f"({len(datas) - 1} pregões), {CFG_BT.custo_bps:g} bps por lado")
    print(formatada.to_string())
    for nome, resultado in avaliacao.resultados.items():
        for falha in resultado.falhas:
            print(f"  falha {nome} {falha}")
    print(f"\nVeredito pré-registrado (controle {CONTROLE})")
    for nome, resultado_criterios in avaliacao.veredito.criterios.items():
        for rotulo, passou in resultado_criterios.items():
            print(f"  {nome}: {'PASSOU    ' if passou else 'NÃO PASSOU'}  {rotulo}")


def main() -> int:
    hoje = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    inicio = time.perf_counter()
    avaliacao = avaliar(carregar(hoje))
    _imprimir(avaliacao)
    escolhida = avaliacao.veredito.escolhida
    if escolhida == CONTROLE:
        print(f"\nNenhuma desafiante passou: fica a {CONTROLE}.", end=" ")
    else:
        print(f"\n{escolhida} passou neste tiro; a janela do app só muda se passar também no tiro 2.", end=" ")
    print(f"(tempo total {time.perf_counter() - inicio:.0f} s)")
    return 2 if escolhida == CONTROLE else 0


if __name__ == "__main__":
    raise SystemExit(main())
