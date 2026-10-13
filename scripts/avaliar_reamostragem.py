"""
Avaliação PRÉ-REGISTRADA da carteira reamostrada (bagging), COM REDE (~10 min).

Uso: .venv/Scripts/python scripts/avaliar_reamostragem.py [AAAA-MM-DD]

Pré-registro (2026-10-13, escrito antes de rodar e copiado no session_state.md):
- Universo: assets.get_all_tickers(); horizontes de 5 e 10 anos; walk-forward 504/63,
  custo de 10 bps por lado, taxa livre de risco = CDI na data.
- Kelly lambda=2, peso máximo 20%, até 10 ativos, teto de vol 40%. Reamostragem com
  ConfigReamostragem() (B=100, blocos de 21 pregões, semente 0).
- A reamostrada vira PADRÃO no app só se passar em TODOS:
  C1 Sharpe(reamostrada) >= Sharpe(Kelly) - 0,05, em 5 e em 10 anos;
  C2 MDD(reamostrada) estritamente melhor (menos negativo) que o do Kelly, em 5 e em 10 anos;
  C3 CVaR 95% diário(reamostrada) estritamente menor que o do Kelly, em 5 e em 10 anos;
  C4 em 10 anos, Sharpe e Sortino da reamostrada maiores que os da réplica do otimizador antigo.
- Só reportados (não decidem): PSR, DSR (N = estratégias que não são benchmark) e o PSR das
  séries diferença (estratégia A menos estratégia B, dia a dia).
Saída: código 0 se passar, 2 se não passar (resultado, não erro).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assets import BENCHMARK_ACOES, get_all_tickers  # noqa: E402
from core import market_data  # noqa: E402
from core.backtest import (estrategia_buy_and_hold_ativo, estrategia_equiponderada,  # noqa: E402
                           estrategia_otimizada, estrategia_reamostrada, walk_forward)
from core.config import ConfigBacktest, ConfigOtimizacao, Objetivo, taxa_anual_para_aritmetica  # noqa: E402
from core.estimation import retornos_simples  # noqa: E402
from core.metrics import (momentos_sharpe, resumo_desempenho, retornos_de_valores,  # noqa: E402
                          sharpe_probabilistico, sharpe_referencia_deflacionado)

KELLY, REAMOSTRADA, REPLICA, BENCHMARK = "Kelly λ=2", "Kelly reamostrado", "Réplica do antigo", "BOVA11"
TOLERANCIA_SHARPE = 0.05
CFG_KELLY = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.20, n_ativos_max=10,
                             vol_maxima=0.40, aversao_risco=2.0)
CFG_BT = ConfigBacktest(capital_inicial=10_000.0, custo_bps=10.0, shrink_mu=True)


def estrategia_replica_antiga(treino: pd.DataFrame) -> pd.Series:
    """Perfil Agressivo antigo: as 5 maiores médias de LOG-retornos, 20% cada (sem shrinkage)."""
    maiores = np.log1p(treino).mean().nlargest(5).index
    return pd.Series(0.2, index=maiores).reindex(treino.columns).fillna(0.0)


@dataclass(frozen=True)
class Horizonte:
    anos: int
    periodo: str
    tabela: pd.DataFrame
    diferencas: dict[str, float]


def _curvas(anos: int, hoje: date) -> tuple[dict[str, pd.Series], pd.Series, float]:
    acoes = [t for t in dict.fromkeys(get_all_tickers()) if t != BENCHMARK_ACOES]
    dados = market_data.carregar_precos(acoes + [BENCHMARK_ACOES], anos, hoje=hoje)
    cdi = market_data.baixar_cdi_diario(anos, hoje=hoje)
    selic = market_data.obter_selic_meta(hoje).valor
    universo = [t for t in acoes if t in dados.precos.columns]
    retornos = retornos_simples(dados.precos[universo + [BENCHMARK_ACOES]])   # amostra comum a todos
    cfg = replace(CFG_KELLY, taxa_livre_risco=taxa_anual_para_aritmetica(selic))
    estrategias = {
        KELLY: estrategia_otimizada(cfg, True, cdi),
        REAMOSTRADA: estrategia_reamostrada(cfg, True, cdi),
        "Max Retorno + Jorion": estrategia_otimizada(replace(cfg, objetivo=Objetivo.MAX_RETORNO), True, cdi),
        REPLICA: estrategia_replica_antiga,
        "1/N": estrategia_equiponderada(),
    }
    print(f"{anos} anos: universo {len(universo)}, amostra {retornos.index[0]:%Y-%m-%d} a "
          f"{retornos.index[-1]:%Y-%m-%d}, descartados {len(dados.descartados_sem_dados + dados.descartados_cobertura)}")
    curvas = {}
    for nome, estrategia in estrategias.items():
        inicio = time.perf_counter()
        resultado = walk_forward(retornos[universo], estrategia, CFG_BT, nome)
        curvas[nome] = resultado.valores
        print(f"  {nome}: {time.perf_counter() - inicio:.0f} s, falhas {len(resultado.falhas)}")
    curvas[BENCHMARK] = walk_forward(retornos, estrategia_buy_and_hold_ativo(BENCHMARK_ACOES),
                                     CFG_BT, BENCHMARK).valores
    return curvas, cdi, selic


def _excesso(valores: pd.Series, cdi: pd.Series, selic: float) -> pd.Series:
    retornos = retornos_de_valores(valores)
    return retornos - market_data.serie_rf_diaria(retornos.index, cdi, selic)


def avaliar(anos: int, hoje: date) -> Horizonte:
    curvas, cdi, selic = _curvas(anos, hoje)
    linhas, momentos = {}, {}
    for nome, valores in curvas.items():
        s = resumo_desempenho(valores, market_data.serie_rf_diaria(valores.index, cdi, selic))
        momentos[nome] = momentos_sharpe(_excesso(valores, cdi, selic))
        linhas[nome] = {"Retorno total": s.retorno_total, "CAGR": s.cagr, "Vol": s.volatilidade,
                        "Sharpe": s.sharpe, "Sortino": s.sortino, "MDD": s.max_drawdown,
                        "CVaR95 diário": s.cvar_95_diario, "PSR": sharpe_probabilistico(momentos[nome])}
    tentativas = [m.sharpe for nome, m in momentos.items() if nome != BENCHMARK]
    referencia = sharpe_referencia_deflacionado(float(np.var(tentativas, ddof=1)), len(tentativas))
    for nome in linhas:
        linhas[nome]["DSR"] = (float("nan") if nome == BENCHMARK
                               else sharpe_probabilistico(momentos[nome], referencia))
    diferencas = {}
    for a, b in [(REAMOSTRADA, REPLICA), (KELLY, REPLICA), (REAMOSTRADA, KELLY)]:
        serie = retornos_de_valores(curvas[a]) - retornos_de_valores(curvas[b])
        diferencas[f"P[{a} > {b}]"] = sharpe_probabilistico(momentos_sharpe(serie))
    valores = curvas[KELLY]
    periodo = f"{valores.index[0]:%Y-%m-%d} a {valores.index[-1]:%Y-%m-%d}"
    return Horizonte(anos, periodo, pd.DataFrame(linhas).T, diferencas)


def criterios(h5: Horizonte, h10: Horizonte) -> dict[str, bool]:
    ambos = (h5, h10)

    def m(h: Horizonte, nome: str, coluna: str) -> float:
        return float(h.tabela.loc[nome, coluna])

    return {
        "C1 Sharpe >= Kelly - 0,05 (5 e 10 anos)": all(
            m(h, REAMOSTRADA, "Sharpe") >= m(h, KELLY, "Sharpe") - TOLERANCIA_SHARPE for h in ambos),
        "C2 MDD melhor que o Kelly (5 e 10 anos)": all(
            m(h, REAMOSTRADA, "MDD") > m(h, KELLY, "MDD") for h in ambos),
        "C3 CVaR menor que o Kelly (5 e 10 anos)": all(
            m(h, REAMOSTRADA, "CVaR95 diário") < m(h, KELLY, "CVaR95 diário") for h in ambos),
        "C4 Sharpe e Sortino > réplica (10 anos)": (
            m(h10, REAMOSTRADA, "Sharpe") > m(h10, REPLICA, "Sharpe")
            and m(h10, REAMOSTRADA, "Sortino") > m(h10, REPLICA, "Sortino")),
    }


def _imprimir(h: Horizonte) -> None:
    tabela = h.tabela.copy()
    for coluna in ["Retorno total", "CAGR", "Vol", "MDD", "CVaR95 diário"]:
        tabela[coluna] = (tabela[coluna] * 100).map("{:.2f}%".format)
    for coluna in ["Sharpe", "Sortino", "PSR", "DSR"]:
        tabela[coluna] = tabela[coluna].map("{:.3f}".format)
    print(f"\nWalk-forward fora da amostra, {h.anos} anos ({h.periodo}), 10 bps, CDI na data")
    print(tabela.to_string())
    for rotulo, valor in h.diferencas.items():
        print(f"  PSR da diferença {rotulo}: {valor:.3f}")


def main() -> int:
    hoje = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    inicio = time.perf_counter()
    h5, h10 = avaliar(5, hoje), avaliar(10, hoje)
    _imprimir(h5)
    _imprimir(h10)
    resultado = criterios(h5, h10)
    print("\nVeredito pré-registrado")
    for rotulo, passou in resultado.items():
        print(f"  {'PASSOU' if passou else 'NÃO PASSOU'}  {rotulo}")
    aprovado = all(resultado.values())
    print(f"\n{'APROVADA: vira padrão' if aprovado else 'NÃO APROVADA: fica opcional'} "
          f"(tempo total {time.perf_counter() - inicio:.0f} s)")
    return 0 if aprovado else 2


if __name__ == "__main__":
    raise SystemExit(main())
