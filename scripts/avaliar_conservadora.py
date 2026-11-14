"""
S2.1: a Conservadora de Blitz e van Vliet (2018) no lago da B3 (PLAN §5, S2, estratégia 1 da lista fechada). IC do
sinal mês a mês e walk-forward da carteira contra o BOVA11 e o Kelly. Não muda o paper trading (paper-v1) nem o
critério de pronto. Rede só para o CDI e a Selic meta.

Uso: .venv/Scripts/python -X utf8 scripts/avaliar_conservadora.py > docs/conservadora-2026-10-20.txt

Pré-registro (2026-10-20, escrito e commitado antes de rodar; depois desse commit o texto não muda):
- Dados: de `docs/dados/2026-10-20/`, `monthly_total_returns.parquet` (retorno total de cada mês desde 07/1994,
  deslistadas incluídas) e `dividends.parquet` (proventos em dinheiro na data ex, com o yield sobre o fechamento cru
  do dia ex); de `docs/dados/2026-10-19/`, `universe_prices.parquet` (índice diário de retorno total); de
  `docs/dados/2026-10-18/`, `universe_monthly.parquet` (universo ponto no tempo do lago, regra D1) e
  `sectors.parquet`. O script confere os cinco pelo `SHA256SUMS` de cada pasta e para se algum não bater.
- Sinal (`core.sinais.conservadora`) do mês m, só com meses antes de m e proventos com data ex antes de m: vol =
  desvio-padrão dos 36 retornos mensais de m-36 a m-1 (os 36 presentes); momentum 12-1 = retorno acumulado de m-12 a
  m-2 (os 11 presentes; mês sem retorno não vira 0); dividend yield de 12 meses = soma dos yields das datas ex de
  m-12 a m-1 (0 sem provento; os quatro tipos do arquivo, todos em dinheiro: dividendo, JCP, rendimento e
  restituição de capital). Nota = 1 para os n//2 de menor vol entre os papéis do universo do mês com as duas
  medidas (0 para os outros) + a média dos postos percentuais do dividend yield e do momentum dentro da metade (empate
  com o posto médio). Sem recompras: o payout líquido do artigo precisa da contagem de ações (CVM FRE), que o lago
  ainda não tem, e fica para um pré-registro próprio, contado no N. Postos: os yields extremos do arquivo (há eventos
  acima de 100% do preço) só valem o 1º posto.
- Portão (o escolhido pela análise de poder, `docs/poder-dos-portoes-2026-10-18.txt`): IC de Spearman de cada mês
  entre a nota e o retorno total do próprio mês, nos papéis do universo do mês com os dois. Condição 1: IC médio
  >= 0,02 e t de Student da média >= 2 de 2008-01 a 2026-09 (o último mês fechado; o período da análise de poder).
  Condição 2: IC médio > 0 de 1998-01 a 2007-12 (uma década que nenhum teste do projeto olhou; universo menor, ~45
  papéis com nota em 1998). Aprovada = as duas. Subperíodos 2008–2014, 2015–2019 e 2020–2026 só descritivos.
- Carteira (`core.sinais.estrategia_conservadora`): as 15 maiores notas com pesos iguais, um papel por emissor (o de
  nota maior), até 4 por setor do lago (26,7% < 30%), papel sem setor fora, conferida por `checar_carteira` com a cov
  de Ledoit-Wolf do treino (violação = recusa: o motor mantém a carteira atual e conta). Na decisão de um pregão do mês
  m, o sinal é o do mês m (meses até m-1), o mesmo do IC daquele mês. Limitações escritas aqui: o artigo rebalanceia
  todo mês e escolhe 100 de 500; aqui o rebalanceio é o do motor (63 pregões, as datas do Kelly) e 15 de ~70.
- Walk-forward: o motor do `scripts/avaliar_robustez.py`, sem mudança, nas 3 fases (0, 21 e 42 pregões cortados do
  começo do arquivo de preços): universo ponto no tempo, treino de 1.260 pregões (só fica elegível quem tem a janela
  inteira, como no Kelly), rebalanceio a cada 63, 10 bps por lado, perda diária de 5% da carteira congelando o pregão
  seguinte. Curvas nas mesmas datas: a Conservadora, o "Kelly λ=2 com regras" (o satélite do estudo, sem stop) e as
  referências BOVA11 e IVVB11. Por fase, no período inteiro (~2010 a 16/10/2026): Sharpe sobre o CDI, IR contra o
  BOVA11, P[> BOVA11] aritmético e em log, DSR do IR (N = linhas de `docs/tentativas.csv`, já com esta: 22; V =
  variância dos IR diários da Conservadora e do Kelly) e MinTRL; P[Conservadora > Kelly]; CAGR, MDD, giro anual e
  recusas.
- Decisão fixa agora: aprovada = sinal candidato a views do S3 e a entrar nas provas seguintes (EGARCH, prova final);
  candidata à prova final = aprovada e P[> BOVA11] aritmético >= 0,95 nas 3 fases. Reprovada = sai da lista; não
  volta com outro parâmetro sem pré-registro novo, contado no N.
- Expectativa registrada: IC médio de 2008 a 2026 entre 0,01 e 0,04, com t entre 1 e 3 (a vol baixa e o dividend
  yield funcionaram na B3; o momentum é fraco aqui); IC de 1998–2007 positivo; no walk-forward, CAGR perto do BOVA11
  com vol e MDD menores, P[> BOVA11] entre 0,5 e 0,9 e abaixo de 0,95 em pelo menos uma fase, P[> Kelly] perto de
  0,5, DSR abaixo de 0,95, giro anual de 100% a 200% e nenhuma recusa.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, ResultadoBacktest, estrategia_otimizada_nas_regras, walk_forward_dinamico  # noqa: E402
from core.config import taxa_anual_para_aritmetica  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.poder import Portao, ic_por_mes  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import estrategia_conservadora, painel_de_notas  # noqa: E402
from scripts.avaliar_robustez import (FASES, Dados, _acoes, _tabela, carregar, conferir_hashes,  # noqa: E402
                                      contar_tentativas, referencias, tabela_do_periodo)
from scripts.avaliar_regras_no_motor import SATELITE  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT, CFG_KELLY, _porcento  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
PASTA_MENSAL = RAIZ / "docs" / "dados" / "2026-10-20"
CONSERVADORA = "Conservadora (15 nomes, pesos iguais, com regras)"
PORTAO = Portao("IC ≥ 0,02 e t ≥ 2", ic_minimo=0.02, t_minimo=2.0)
PRINCIPAL = (pd.Timestamp("2008-01-01"), pd.Timestamp("2026-09-01"))      # condição 1
EXTENSAO = (pd.Timestamp("1998-01-01"), pd.Timestamp("2007-12-01"))       # condição 2: IC médio > 0
SUBPERIODOS_DO_IC = {"2008–2014": ("2008-01-01", "2014-12-01"), "2015–2019": ("2015-01-01", "2019-12-01"),
                     "2020–2026": ("2020-01-01", "2026-09-01")}
P_MINIMO_DA_PROVA = 0.95
_TOLERANCIA = 1e-9


@dataclass(frozen=True)
class Decisao:
    condicao_1: bool        # portão do IC de 2008-01 a 2026-09
    condicao_2: bool        # IC médio > 0 de 1998-01 a 2007-12
    aprovada: bool
    candidata: bool         # à prova final: aprovada e P[> BOVA11] >= 0,95 nas 3 fases


def _recorte(ics: pd.Series, periodo: tuple[pd.Timestamp, pd.Timestamp]) -> pd.Series:
    return ics.loc[periodo[0]:periodo[1]].dropna()


def decidir(ics: pd.Series, p_bova_por_fase: Sequence[float]) -> Decisao:
    """As regras fixas do pré-registro sobre o IC de cada mês e o P[> BOVA11] aritmético de cada fase."""
    condicao_1 = PORTAO.aprova(_recorte(ics, PRINCIPAL))
    condicao_2 = bool(_recorte(ics, EXTENSAO).mean() > 0.0)
    aprovada = condicao_1 and condicao_2
    candidata = aprovada and all(p >= P_MINIMO_DA_PROVA - _TOLERANCIA for p in p_bova_por_fase)
    return Decisao(condicao_1, condicao_2, aprovada, candidata)


def resumo_do_ic(ics: pd.Series) -> pd.Series:
    """Meses, IC médio, desvio e t de Student da média."""
    ics = ics.dropna()
    t = float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))) if len(ics) > 1 else float("nan")
    return pd.Series({"meses": float(len(ics)), "IC médio": float(ics.mean()), "desvio": float(ics.std(ddof=1)),
                      "t": t, "meses com IC > 0": float((ics > 0).mean())})


# ---------------------------------------------------------------- rodada (com o CDI da rede)

@dataclass(frozen=True)
class Mensal:
    retornos: pd.DataFrame      # mês (1º dia) x papel
    proventos: pd.DataFrame
    sha256: dict[str, str]


def carregar_mensal(pasta: Path = PASTA_MENSAL) -> Mensal:
    sha256 = conferir_hashes(pasta, ["monthly_total_returns.parquet", "dividends.parquet"])
    retornos = pd.read_parquet(pasta / "monthly_total_returns.parquet")
    retornos = retornos.set_index(pd.DatetimeIndex(pd.to_datetime(retornos.pop("month")), name=None)).astype("float64")
    proventos = pd.read_parquet(pasta / "dividends.parquet", columns=["security", "ex_date", "yield"])
    return Mensal(retornos, proventos, sha256)


def retornos_no_universo(retornos: pd.DataFrame, universo: pd.DataFrame, meses: pd.DatetimeIndex) -> pd.DataFrame:
    """O retorno de cada mês só dos papéis do universo daquele mês (``universo``: ``month``, ``security``)."""
    dentro = universo.assign(dentro=True, month=pd.to_datetime(universo["month"])).pivot(
        index="month", columns="security", values="dentro")
    return retornos.reindex(meses).where(dentro.reindex(index=meses, columns=retornos.columns).notna())


def ics_da_conservadora(mensal: Mensal, universo: pd.DataFrame) -> pd.Series:
    """IC de cada mês de 1998-01 a 2026-09: a nota do mês contra o retorno do próprio mês, no universo dele."""
    meses = pd.date_range(EXTENSAO[0], PRINCIPAL[1], freq="MS")
    notas = painel_de_notas(mensal.retornos, mensal.proventos, universo, meses)
    return ic_por_mes(notas, retornos_no_universo(mensal.retornos, universo, meses))


def curvas_da_fase(dados: Dados, mensal: Mensal, fase: int, cdi: pd.Series | None,
                   cfg_kelly) -> dict[str, ResultadoBacktest]:
    """A Conservadora, o Kelly com regras e as referências nas mesmas datas da fase."""
    precos = dados.precos.iloc[fase:]
    conservadora = estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores)
    kelly = estrategia_otimizada_nas_regras(replace(cfg_kelly, aversao_risco=2.0), True, dados.setores, cdi)
    curvas = {CONSERVADORA: walk_forward_dinamico(_acoes(precos), dados.universo, conservadora, CFG_BT, CONSERVADORA,
                                                  Congelamento()),
              SATELITE: walk_forward_dinamico(_acoes(precos), dados.universo, kelly, CFG_BT, SATELITE, Congelamento()),
              **referencias(precos, CFG_BT)}
    if any(not curva.valores.index.equals(curvas[SATELITE].valores.index) for curva in curvas.values()):
        raise SystemExit(f"Fase {fase}: curvas com datas diferentes")
    return curvas


def medidas_da_fase(curvas: Mapping[str, ResultadoBacktest], rf: pd.Series, n_tentativas: int) -> pd.DataFrame:
    """CAGR, MDD, giro e recusas das quatro curvas (a referência que começa depois, do 1º pregão dela) e as medidas
    contra o BOVA11 da Conservadora e do Kelly."""
    tabela = tabela_do_periodo({nome: curvas[nome].valores for nome in (CONSERVADORA, SATELITE)},
                               curvas["BOVA11"].valores, rf, n_tentativas)
    tabela.loc[CONSERVADORA, "P[> Kelly]"] = psr_da_diferenca(curvas[CONSERVADORA].valores, curvas[SATELITE].valores)
    linhas = {}
    for nome in (CONSERVADORA, SATELITE, "IVVB11", "BOVA11"):
        valores = curvas[nome].valores.dropna()
        linhas[nome] = {"CAGR": cagr(valores), "MDD": drawdown(valores)[1],
                        "Giro anual": curvas[nome].turnover_anual_medio, "Recusas": len(curvas[nome].recusas)}
    return pd.DataFrame(linhas).T.astype({"Recusas": int}).join(tabela)


def main() -> int:
    inicio_rodada = time.perf_counter()
    hoje = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    dados, mensal = carregar(), carregar_mensal()
    selic = market_data.obter_selic_meta(hoje)
    try:
        cdi = market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=hoje)
    except market_data.ErroDadosMercado as erro:
        print(f"CDI: {erro} (Selic constante)")
        cdi = pd.Series(dtype=float)
    cdi_vigente = cdi if not cdi.empty else None
    cfg = replace(CFG_KELLY, taxa_livre_risco=taxa_anual_para_aritmetica(selic.valor))
    n_tentativas = contar_tentativas()
    print(f"Arquivos (sha256): {dados.sha256 | mensal.sha256}")
    print(f"Manifest: {dados.manifest}; Selic meta {_porcento(selic.valor)} ({selic.fonte}); "
          f"CDI {'do BCB' if cdi_vigente is not None else 'indisponível'}; N do DSR = {n_tentativas}")

    ics = ics_da_conservadora(mensal, dados.universo)
    periodos = {"1998–2007 (condição 2)": EXTENSAO, "2008–2026 (condição 1)": PRINCIPAL,
                **{rotulo: tuple(map(pd.Timestamp, datas)) for rotulo, datas in SUBPERIODOS_DO_IC.items()}}
    print("\nIC de Spearman mensal (nota do mês x retorno total do mês, universo do mês):")
    print(pd.DataFrame({rotulo: resumo_do_ic(_recorte(ics, periodo)) for rotulo, periodo in periodos.items()}).T
          .to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"(fase de IC pronta em {time.perf_counter() - inicio_rodada:.0f} s)", flush=True)

    p_bova = []
    for fase in FASES:
        curvas = curvas_da_fase(dados, mensal, fase, cdi_vigente, cfg)
        rf = market_data.serie_rf_diaria(curvas[SATELITE].valores.index, cdi_vigente, selic.valor)
        tabela = medidas_da_fase(curvas, rf, n_tentativas)
        p_bova.append(float(tabela.loc[CONSERVADORA, "P aritmético"]))
        conservadora = curvas[CONSERVADORA]
        print(f"\n=== Fase {fase}: 1ª decisão {conservadora.pesos_alvo.index[0]:%d/%m/%Y}, fim "
              f"{conservadora.valores.index[-1]:%d/%m/%Y}, {len(conservadora.pesos_alvo)} rebalanceios; falhas "
              f"{len(conservadora.falhas)}; adiados {len(conservadora.adiamentos)} ===")
        print(_tabela(tabela, ["CAGR", "MDD", "Giro anual"]))
        print(f"(fase {fase} pronta em {time.perf_counter() - inicio_rodada:.0f} s)", flush=True)

    decisao = decidir(ics, p_bova)
    print(f"\nCondição 1 ({PORTAO.nome}, 2008-01 a 2026-09): {'cumprida' if decisao.condicao_1 else 'não cumprida'}")
    print(f"Condição 2 (IC médio > 0, 1998-01 a 2007-12): {'cumprida' if decisao.condicao_2 else 'não cumprida'}")
    print(f"Conservadora: {'APROVADA' if decisao.aprovada else 'REPROVADA'}; candidata à prova final (P[> BOVA11] "
          f">= 0,95 nas 3 fases): {'sim' if decisao.candidata else 'não'}")
    print(f"(tempo total {time.perf_counter() - inicio_rodada:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
