"""
Avaliação PRÉ-REGISTRADA com o universo ponto no tempo do lago (S1-P.b), com rede só para o CDI e a Selic.

Uso: .venv/Scripts/python -X utf8 scripts/avaliar_universo_dinamico.py

Pré-registro (2026-10-17, escrito e commitado antes de rodar):
- Dados: exports do lago `b3-datalake` `d47f2fa` em `data/b3lake/`: `universe_prices.parquet` (índice de retorno total
  diário dos 417 papéis que passaram pelo universo desde 2005, 186 deles hoje deslistados, mais BOVA11, SMAL11 e
  DIVO11; vazio onde não houve negócio) e `universe_monthly.parquet` (regra D1: no mês m, ação ou unit com negócio em
  ≥ 90% dos pregões de m−12 a m−1 e volume diário mediano ≥ R$ 1 mi de 09/2026 corrigido pelo IPCA conhecido na data;
  um papel por emissor). A saída imprime o sha256 dos dois arquivos e as datas do manifest do lago.
- Walk-forward `core.backtest.walk_forward_dinamico`: em cada rebalanceio, só os elegíveis do mês com a janela de
  treino inteira (sem buraco de mais de 5 pregões); treino de 1.260 pregões (R5, a janela do app, mantida no tiro 2),
  rebalanceio a cada 63, 10 bps por lado, capital 10 mil, Jorion ligado, taxa livre de risco = CDI na data (Selic
  constante se o BCB falhar). A deslistada fica no último fechamento (rendendo 0) até o rebalanceio seguinte, que a
  vende.
- Otimização: peso máximo 20%, até 10 ativos, teto de vol 40%, sem teto de setor (o lago não tem o setor das
  deslistadas; o teste da sessão 13 também rodou sem), sem reamostragem, sem Growth.
- Horizontes: os últimos 5 e os últimos 10 anos fora da amostra, terminando no último pregão do arquivo.
- Estratégias (lista fechada, N = 11 no DSR): Kelly λ = 1, λ = 2 (controle, padrão do app) e λ = 4, Máx. Sharpe,
  Mín. Vol, HRP, 1/N (sobre o mesmo conjunto treinado) e 4 misturas núcleo + satélite com satélite Kelly λ = 2: núcleo
  Mín. Vol 70/30 e 50/50, núcleo BOVA11 70/30 e 50/50, levadas de volta aos pesos-alvo nas mesmas datas de
  rebalanceio, com 10 bps sobre o que muda entre núcleo e satélite (a montagem inicial é cobrada de novo, ~0,1% uma
  vez, contra as misturas). Referências fora do N: BOVA11, SMAL11 e DIVO11; sem cotação na 1ª decisão = "sem
  histórico" naquele horizonte.
- Regra A (Kelly × objetivos, a régua da sessão 13), alternativos = Máx. Sharpe, Mín. Vol, HRP e 1/N:
  4 (remover o Kelly) se algum alternativo vence o Kelly λ = 2 em Sharpe E em MDD nos dois horizontes;
  1 (principal) se o Kelly λ = 2 tem Sharpe maior que todos os alternativos nos dois horizontes, com P[Kelly > X] (PSR
    da diferença diária) ≥ 0,90 contra cada um, e MDD no máximo 5 p.p. pior que o melhor MDD entre eles;
  senão 3 se o Kelly perde Sharpe para algum alternativo em algum horizonte, 2 se vence sem significância.
- Regra B (núcleo + satélite, §D.1 da sessão 13): uma mistura vira o padrão recomendado só se, nos dois horizontes,
  tiver Sharpe ≥ o do Kelly λ = 2 − 0,05 e MDD pelo menos 10 p.p. melhor que o dele. Mais de uma: a de maior Sharpe
  médio nos dois horizontes. Nenhuma: vale a regra A.
- Só reportados (não decidem): retorno total, CAGR, vol, Sortino, CVaR 95% diário, giro, nº médio de ativos, HHI,
  falhas, PSR, DSR, P[> BOVA11] e nº médio de papéis treinados por rebalanceio. Critério de pronto (ROTEIRO §2) sem
  mudança.
- Expectativa registrada: os números caem frente aos da sessão 13 (o viés de sobrevivência some); o erro do Sharpe é
  ≈ ±0,45 em 5 anos e ±0,3 em 10, então o mais provável é a regra A repetir a opção 3 e a regra B não aprovar mistura
  (em 5 anos a Mín. Vol rendeu quase o CDI na sessão 13).
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtest import ResultadoBacktest, backtest_pesos_fixos  # noqa: E402
from core.config import ConfigBacktest  # noqa: E402
from core.metrics import retornos_de_valores  # noqa: E402

CONTROLE = "Kelly λ=2"
ALTERNATIVOS = ("Máx. Sharpe", "Mín. Vol", "HRP", "1/N")
LIMIAR_P_OBJETIVO = 0.90             # P[Kelly > X] mínimo contra cada alternativo para a opção 1
PIORA_MAXIMA_MDD = 0.05              # MDD do Kelly no máximo 5 p.p. pior que o melhor dos alternativos


@dataclass(frozen=True)
class Medidas:
    """O que as regras usam de cada estratégia num horizonte."""

    sharpe: float
    max_drawdown: float          # <= 0
    p_controle_supera: float     # P[Kelly λ=2 > estratégia]: PSR da diferença diária (NaN no próprio controle)


Horizontes = Mapping[int, Mapping[str, Medidas]]       # anos fora da amostra -> estratégia -> medidas


def recortar_horizonte(precos: pd.DataFrame, anos: int, janela_treino: int) -> pd.DataFrame:
    """
    Preços para um walk-forward cujo trecho fora da amostra são os últimos ``anos``: começa no 1º pregão depois de
    (último pregão − ``anos``), com ``janela_treino`` retornos antes dele (e o pregão que dá o 1º retorno).
    """
    primeiro_fora = precos.index.searchsorted(precos.index[-1] - pd.DateOffset(years=anos), side="right")
    inicio = primeiro_fora - janela_treino - 1
    if inicio < 0:
        raise ValueError(f"Histórico insuficiente para {anos} anos fora da amostra com treino de {janela_treino}.")
    return precos.iloc[inicio:]


def misturar(nucleo: ResultadoBacktest, satelite: ResultadoBacktest, peso_nucleo: float, cfg: ConfigBacktest,
             nome: str) -> ResultadoBacktest:
    """
    Núcleo + satélite de volta aos pesos-alvo a cada ``cfg.janela_teste`` pregões (as mesmas datas de rebalanceio das
    duas curvas, que vêm do mesmo walk-forward), com ``cfg.custo_bps`` sobre o que muda entre elas.
    """
    retornos = pd.DataFrame({"núcleo": retornos_de_valores(nucleo.valores),
                             "satélite": retornos_de_valores(satelite.valores)})
    alvo = pd.Series({"núcleo": peso_nucleo, "satélite": 1.0 - peso_nucleo})
    mistura = backtest_pesos_fixos(retornos, alvo, cfg, nome)
    return replace(mistura, metodologia="walk-forward OOS, núcleo + satélite",
                   valores=mistura.valores.set_axis(nucleo.valores.index))


def decidir_objetivo(h: Horizontes) -> str:
    """Regra A do pré-registro (a da sessão 13): opção 1, 2, 3 ou 4 para o Kelly λ=2."""
    def vence_em_tudo(alternativo: str) -> bool:
        return all(m[alternativo].sharpe > m[CONTROLE].sharpe
                   and m[alternativo].max_drawdown > m[CONTROLE].max_drawdown for m in h.values())

    if any(vence_em_tudo(alternativo) for alternativo in ALTERNATIVOS):
        return "4"
    alternativos = [(m[CONTROLE], m[x]) for m in h.values() for x in ALTERNATIVOS]
    vence = all(kelly.sharpe > outro.sharpe for kelly, outro in alternativos)
    significativo = all(outro.p_controle_supera >= LIMIAR_P_OBJETIVO for _, outro in alternativos)
    mdd_aceitavel = all(m[CONTROLE].max_drawdown >= max(m[x].max_drawdown for x in ALTERNATIVOS) - PIORA_MAXIMA_MDD
                        for m in h.values())
    if vence and significativo and mdd_aceitavel:
        return "1"
    return "3"
