"""
Núcleo BOVA11 + IVVB11 (PLAN §11, alavanca 1, DECISOES 64): trocar o núcleo só de BOVA11 por metade BOVA11 e metade
IVVB11 (S&P 500 em reais, sem hedge) ganha do BOVA11? Não muda o paper trading (paper-v1) nem o critério de pronto.
Rede só para o CDI e a Selic meta (o Sharpe).

Uso: .venv/Scripts/python -X utf8 scripts/avaliar_nucleo.py > docs/nucleo-2026-10-20.txt

Pré-registro (2026-10-20, escrito e commitado antes de rodar; depois desse commit o texto não muda):
- Dados: `docs/dados/2026-10-19/universe_prices.parquet` (índice diário de retorno total do lago; sha256 conferido
  pelo `SHA256SUMS`, o script para se não bater), só as colunas BOVA11 e IVVB11. Trecho: do 1º pregão em que os dois
  têm cotação (29/04/2014, a 1ª do IVVB11) a 16/10/2026, ~12,5 anos; antes disso o motor daria retorno 0 à perna que
  não existia. Dia sem negócio no meio rende 0 e o movimento cai no pregão seguinte (`retornos_da_carteira`).
- Configuração única: 50% BOVA11 + 50% IVVB11, pesos fixos, de volta ao alvo a cada 63 pregões, 10 bps por lado
  (`backtest_pesos_fixos` e o `CFG_BT` dos outros testes). Os 50% são o 1/N das duas pernas, escolhido sem estimar
  nada no trecho (a mínima variância com as vols e a correlação do trecho daria ~60% de IVVB11, estimada nos mesmos
  dados que testa: fica de fora). Referências: BOVA11 comprado e mantido e IVVB11 sozinho, no mesmo motor e nas
  mesmas datas. Uma fração fixa desse núcleo dentro do dinheiro real (70% núcleo + 30% satélite) tem o mesmo P contra
  o BOVA11 que o núcleo sozinho, por isso só o núcleo é testado.
- Medidas, em 3 fases (0, 21 e 42 pregões cortados do começo do trecho): CAGR, vol anual, MDD, giro; Sharpe sobre o
  CDI; IR contra o BOVA11; P[> BOVA11] aritmético e em log (`psr_da_diferenca`); DSR do IR (N = linhas de
  `docs/tentativas.csv`, já com esta: 23; V = variância dos IR diários do núcleo misto e do IVVB11 sozinho, as duas
  pontas da alavanca); MinTRL. Excesso de CAGR sobre o BOVA11 nos subperíodos fixados agora: 2014–2018 (do começo
  do trecho), 2019–2022 e 2023–2026 (até 16/10/2026). Janelas móveis de 36 e 60 meses (`janelas_moveis`): fração
  acima do BOVA11, excesso mediano e p10, pior MDD. Correlação diária e mensal BOVA11 × IVVB11 no trecho.
- Decisão fixa agora: núcleo candidato à prova final e ao paper-v2 (nunca ao paper-v1) se P[> BOVA11] em log >= 0,95
  nas 3 fases (condição 1; o log porque o P aritmético favorece a curva mais volátil, e aqui o BOVA11 é a mais
  volátil das duas) e, em cada fase, excesso de CAGR positivo em pelo menos 2 dos 3 subperíodos (condição 2). Fora
  disso, só descritivo: o núcleo segue BOVA11. Passar não diz nada do otimizador: é uma alavanca de diversificação
  medida contra o benchmark do usuário.
- Limitações escritas aqui: um regime de câmbio só (o real saiu de ~2,2 para ~5,4 por dólar no trecho) e nenhum dado
  antes de 2014; o teste não é cego, porque o IVVB11 entrou na lista depois de já se ver que ganhou do BOVA11 desde
  2014 (DECISOES 61 e 72), então o P desta rodada é otimista; venda de cota de ETF paga 15% sobre o ganho sem a
  isenção de R$ 20 mil por mês, e cada volta ao alvo realiza ganho (fica para o S4); taxa de administração e imposto
  retido nos dividendos americanos já estão dentro do preço do IVVB11; cotas inteiras com pouco dinheiro ficam para o
  plano de compra.
- Expectativa registrada: CAGR do misto entre o do BOVA11 e o do IVVB11, vol e MDD bem menores que os do BOVA11; P
  em log entre 0,85 e 0,97 (a conta de cabeça, com excesso em log de ~6% a.a. e erro de rastreio de ~15%, dá ~0,93),
  abaixo de 0,95 em pelo menos uma fase; excesso positivo em 2014–2018 e 2019–2022 e negativo em 2023–2026; DSR
  abaixo de 0,95; não candidato.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import ResultadoBacktest, backtest_pesos_fixos, retornos_da_carteira  # noqa: E402
from core.config import DIAS_UTEIS_ANO  # noqa: E402
from core.metrics import cagr, drawdown, retornos_de_valores  # noqa: E402
from core.robustez import janelas_moveis, resumir_janelas  # noqa: E402
from scripts.avaliar_robustez import (FASES, PORCENTO_JANELAS, _tabela, conferir_hashes,  # noqa: E402
                                      contar_tentativas, tabela_do_periodo, trecho)
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT, _porcento  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
PASTA_PRECOS = RAIZ / "docs" / "dados" / "2026-10-19"
PAPEIS = ("BOVA11", "IVVB11")
MISTO = "Núcleo BOVA11 50% + IVVB11 50%"
PESOS = {MISTO: {"BOVA11": 0.5, "IVVB11": 0.5}, "IVVB11": {"IVVB11": 1.0}, "BOVA11": {"BOVA11": 1.0}}
SUBPERIODOS = {"2014–2018": ("2014-01-01", "2018-12-31"), "2019–2022": ("2019-01-01", "2022-12-31"),
               "2023–2026": ("2023-01-01", "2026-10-16")}
MESES_DAS_JANELAS = (36, 60)
P_MINIMO = 0.95
SUBPERIODOS_MINIMOS = 2          # dos 3, com o excesso de CAGR sobre o BOVA11 positivo
_TOLERANCIA = 1e-9


def comeco_comum(precos: pd.DataFrame) -> pd.DataFrame:
    """Do 1º pregão em que todos os papéis têm cotação em diante. Antes dele, o ``retornos_da_carteira`` daria 0 ao
    papel que ainda não existia (uma perna em caixa sem aviso); depois, a lacuna de um dia sem negócio fica."""
    return precos.loc[max(precos[papel].first_valid_index() for papel in precos.columns):]


@dataclass(frozen=True)
class Decisao:
    condicao_1: bool        # P[> BOVA11] em log >= 0,95 nas 3 fases
    condicao_2: bool        # excesso de CAGR positivo em >= 2 dos 3 subperíodos, em cada fase
    candidato: bool


def decidir(p_log_por_fase: Sequence[float], excessos_por_fase: Sequence[pd.Series]) -> Decisao:
    """As regras fixas do pré-registro: P[> BOVA11] em log de cada fase e o excesso de CAGR de cada subperíodo."""
    condicao_1 = all(p >= P_MINIMO - _TOLERANCIA for p in p_log_por_fase)
    condicao_2 = all(int((excessos > 0).sum()) >= SUBPERIODOS_MINIMOS for excessos in excessos_por_fase)
    return Decisao(condicao_1, condicao_2, condicao_1 and condicao_2)


def excessos_por_subperiodo(valores: pd.Series, bova: pd.Series,
                            subperiodos: Mapping[str, tuple[str, str]]) -> pd.Series:
    """CAGR da curva menos o do BOVA11 em cada subperíodo, do fechamento anterior ao 1º dia (``trecho``) ao último
    pregão dele."""
    excessos = {}
    for rotulo, (de, ate) in subperiodos.items():
        inicio, fim = trecho(valores.index, de, ate)
        excessos[rotulo] = cagr(valores.loc[inicio:fim]) - cagr(bova.loc[inicio:fim])
    return pd.Series(excessos)


# ---------------------------------------------------------------- rodada (com o CDI da rede)

def carregar(pasta: Path = PASTA_PRECOS) -> tuple[pd.DataFrame, dict[str, str]]:
    """BOVA11 e IVVB11 do arquivo do pré-registro, conferido pelo hash, do 1º pregão com os dois em diante."""
    sha256 = conferir_hashes(pasta, ["universe_prices.parquet"])
    tabela = pd.read_parquet(pasta / "universe_prices.parquet", columns=["date", *PAPEIS])
    precos = tabela.set_index(pd.DatetimeIndex(pd.to_datetime(tabela.pop("date")), name=None)).astype("float64")
    return comeco_comum(precos), sha256


def curvas_da_fase(precos: pd.DataFrame, fase: int) -> dict[str, ResultadoBacktest]:
    """O núcleo misto e as duas referências, de volta ao alvo a cada 63 pregões, nas mesmas datas."""
    retornos = retornos_da_carteira(precos.iloc[fase:])
    return {nome: backtest_pesos_fixos(retornos, pd.Series(pesos).reindex(retornos.columns).fillna(0.0), CFG_BT, nome)
            for nome, pesos in PESOS.items()}


def medidas_da_fase(curvas: Mapping[str, ResultadoBacktest], rf: pd.Series, n_tentativas: int) -> pd.DataFrame:
    """CAGR, vol, MDD e giro das três curvas e as medidas contra o BOVA11 do misto e do IVVB11."""
    tabela = tabela_do_periodo({nome: curvas[nome].valores for nome in (MISTO, "IVVB11")}, curvas["BOVA11"].valores,
                               rf, n_tentativas)
    linhas = {nome: {"CAGR": cagr(curva.valores),
                     "Vol": float(retornos_de_valores(curva.valores).std() * np.sqrt(DIAS_UTEIS_ANO)),
                     "MDD": drawdown(curva.valores)[1], "Giro anual": curva.turnover_anual_medio}
              for nome, curva in curvas.items()}
    return pd.DataFrame(linhas).T.join(tabela)


def imprimir_fase(fase: int, curvas: Mapping[str, ResultadoBacktest], tabela: pd.DataFrame,
                  por_subperiodo: pd.DataFrame) -> None:
    misto = curvas[MISTO]
    print(f"\n=== Fase {fase}: {misto.valores.index[1]:%d/%m/%Y} a {misto.valores.index[-1]:%d/%m/%Y}, "
          f"{len(misto.pesos_alvo)} rebalanceios ===")
    print(_tabela(tabela, ["CAGR", "Vol", "MDD", "Giro anual"]))
    print("Excesso de CAGR sobre o BOVA11 por subperíodo:")
    print(_tabela(por_subperiodo, list(SUBPERIODOS)))
    valores = pd.DataFrame({nome: curva.valores for nome, curva in curvas.items()})
    for meses in MESES_DAS_JANELAS:
        print(f"Janelas móveis de {meses} meses:")
        print(_tabela(resumir_janelas(janelas_moveis(valores, "BOVA11", meses)), PORCENTO_JANELAS))


def main() -> int:
    inicio_rodada = time.perf_counter()
    hoje = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    precos, sha256 = carregar()
    selic = market_data.obter_selic_meta(hoje)
    try:
        cdi = market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=hoje)
    except market_data.ErroDadosMercado as erro:
        print(f"CDI: {erro} (Selic constante)")
        cdi = pd.Series(dtype=float)
    cdi_vigente = cdi if not cdi.empty else None
    n_tentativas = contar_tentativas()
    print(f"Arquivo (sha256): {sha256}; trecho {precos.index[0]:%d/%m/%Y} a {precos.index[-1]:%d/%m/%Y} "
          f"({len(precos)} pregões); Selic meta {_porcento(selic.valor)} ({selic.fonte}); "
          f"CDI {'do BCB' if cdi_vigente is not None else 'indisponível'}; N do DSR = {n_tentativas}")
    retornos = retornos_da_carteira(precos)
    mensais = (1.0 + retornos).groupby(retornos.index.to_period("M")).prod() - 1.0
    print(f"Correlação BOVA11 × IVVB11 no trecho: diária {retornos.corr().iloc[0, 1]:.2f}, "
          f"mensal {mensais.corr().iloc[0, 1]:.2f}")

    p_log, excessos = [], []
    for fase in FASES:
        curvas = curvas_da_fase(precos, fase)
        rf = market_data.serie_rf_diaria(curvas[MISTO].valores.index, cdi_vigente, selic.valor)
        tabela = medidas_da_fase(curvas, rf, n_tentativas)
        por_subperiodo = pd.DataFrame({nome: excessos_por_subperiodo(curvas[nome].valores, curvas["BOVA11"].valores,
                                                                     SUBPERIODOS)
                                       for nome in (MISTO, "IVVB11")}).T
        p_log.append(float(tabela.loc[MISTO, "P log"]))
        excessos.append(por_subperiodo.loc[MISTO])
        imprimir_fase(fase, curvas, tabela, por_subperiodo)
        print(f"(fase {fase} pronta em {time.perf_counter() - inicio_rodada:.0f} s)", flush=True)

    decisao = decidir(p_log, excessos)
    print(f"\nCondição 1 (P[> BOVA11] em log >= 0,95 nas 3 fases): "
          f"{'cumprida' if decisao.condicao_1 else 'não cumprida'}")
    print(f"Condição 2 (excesso positivo em >= 2 dos 3 subperíodos, em cada fase): "
          f"{'cumprida' if decisao.condicao_2 else 'não cumprida'}")
    print(f"Núcleo BOVA11 + IVVB11: {'CANDIDATO à prova final e ao paper-v2' if decisao.candidato else 'não candidato'}")
    print(f"(tempo total {time.perf_counter() - inicio_rodada:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
