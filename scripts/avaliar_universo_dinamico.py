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
  senão 3 se o Kelly perde Sharpe para algum alternativo em algum horizonte, 2 se vence em Sharpe sem cumprir o
    resto da 1 (sem significância ou com MDD mais de 5 p.p. pior).
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

import hashlib
import json
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtest import (Estrategia, ResultadoBacktest, backtest_pesos_fixos,  # noqa: E402
                           estrategia_buy_and_hold_ativo, estrategia_equiponderada, estrategia_hrp,
                           estrategia_otimizada, retornos_da_carteira, walk_forward, walk_forward_dinamico)
from core import market_data  # noqa: E402
from core.config import ConfigBacktest, ConfigOtimizacao, Objetivo, taxa_anual_para_aritmetica  # noqa: E402
from core.metrics import (momentos_sharpe, resumo_desempenho, retornos_de_valores,  # noqa: E402
                          sharpe_probabilistico, sharpe_referencia_deflacionado)

PASTA_LAGO = Path(__file__).resolve().parent.parent / "data" / "b3lake"
ARQUIVO_PRECOS = "universe_prices.parquet"
ARQUIVO_UNIVERSO = "universe_monthly.parquet"
CAMPOS_DO_MANIFEST = ("generated_at", "data_from", "data_through")

REFERENCIAS = ("BOVA11", "SMAL11", "DIVO11")   # fora do N do DSR
HORIZONTES = (5, 10)                            # anos fora da amostra, terminando no último pregão do arquivo
CFG_BT = ConfigBacktest(janela_treino=1_260, janela_teste=63, capital_inicial=10_000.0, custo_bps=10.0, shrink_mu=True)
CFG_KELLY = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.20, n_ativos_max=10, vol_maxima=0.40,
                             aversao_risco=2.0)
ANOS_DE_CDI = 22                                # o arquivo começa em 2005
CONTROLE = "Kelly λ=2"
ALTERNATIVOS = ("Máx. Sharpe", "Mín. Vol", "HRP", "1/N")
LIMIAR_P_OBJETIVO = 0.90             # P[Kelly > X] mínimo contra cada alternativo para a opção 1
PIORA_MAXIMA_MDD = 0.05              # MDD do Kelly no máximo 5 p.p. pior que o melhor dos alternativos
ESTRATEGIAS = ("Kelly λ=1", CONTROLE, "Kelly λ=4", *ALTERNATIVOS)
# mistura -> (núcleo, peso do núcleo); o satélite é sempre o Kelly λ=2
COMPOSICAO_MISTURAS = {"Núcleo Mín. Vol 70/30": ("Mín. Vol", 0.70), "Núcleo Mín. Vol 50/50": ("Mín. Vol", 0.50),
                       "Núcleo BOVA11 70/30": ("BOVA11", 0.70), "Núcleo BOVA11 50/50": ("BOVA11", 0.50)}
MISTURAS = tuple(COMPOSICAO_MISTURAS)
FOLGA_SHARPE_MISTURA = 0.05          # regra B: Sharpe da mistura >= Sharpe do Kelly λ=2 - 0,05
MELHORA_MINIMA_MDD = 0.10            # regra B: MDD da mistura pelo menos 10 p.p. melhor que o do Kelly λ=2
_TOLERANCIA = 1e-9                   # "pelo menos" inclui a fronteira, que o float erra por 1e-17


@dataclass(frozen=True)
class Medidas:
    """O que as regras usam de cada estratégia num horizonte."""

    sharpe: float
    max_drawdown: float          # <= 0
    p_controle_supera: float     # P[Kelly λ=2 > estratégia]: PSR da diferença diária (NaN no próprio controle)


Horizontes = Mapping[int, Mapping[str, Medidas]]       # anos fora da amostra -> estratégia -> medidas


def medir(resultados: Mapping[str, ResultadoBacktest], rf_diaria: pd.Series) -> dict[str, Medidas]:
    """
    Medidas das curvas de um horizonte. P[Kelly λ=2 > X] = PSR da diferença diária (Kelly − X): a taxa livre de
    risco se cancela nela, então a falta do CDI não muda as regras.
    """
    retornos_controle = retornos_de_valores(resultados[CONTROLE].valores)
    medidas = {}
    for nome, resultado in resultados.items():
        resumo = resumo_desempenho(resultado.valores, rf_diaria)
        diferenca = retornos_controle - retornos_de_valores(resultado.valores)
        p = float("nan") if nome == CONTROLE else sharpe_probabilistico(momentos_sharpe(diferenca))
        medidas[nome] = Medidas(sharpe=resumo.sharpe, max_drawdown=resumo.max_drawdown, p_controle_supera=p)
    return medidas


@dataclass(frozen=True)
class DadosLago:
    precos: pd.DataFrame         # data x papel, float64, vazio onde não houve negócio
    universo: pd.DataFrame       # month (Timestamp, 1º dia do mês), security
    sha256: dict[str, str]       # arquivo -> sha256, impresso na saída
    manifest: dict[str, str]     # datas do manifest do lago


def carregar_lago(pasta: Path = PASTA_LAGO) -> DadosLago:
    """Os dois exports do lago no formato do motor (o lago grava ``date``/``month`` como ``datetime.date``)."""
    tabela = pd.read_parquet(pasta / ARQUIVO_PRECOS)
    precos = tabela.set_index(pd.DatetimeIndex(pd.to_datetime(tabela.pop("date")), name=None)).astype("float64")
    universo = pd.read_parquet(pasta / ARQUIVO_UNIVERSO, columns=["month", "security"])
    universo["month"] = pd.to_datetime(universo["month"])
    sha256 = {nome: hashlib.sha256((pasta / nome).read_bytes()).hexdigest() for nome in (ARQUIVO_PRECOS, ARQUIVO_UNIVERSO)}
    # o lago grava o manifest em cp1252 (não é UTF-8 válido); só as datas, que são ASCII, interessam
    manifest = json.loads((pasta / "manifest.json").read_bytes().decode("utf-8", errors="replace"))
    return DadosLago(precos=precos, universo=universo, sha256=sha256,
                     manifest={campo: manifest[campo] for campo in CAMPOS_DO_MANIFEST})


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


def rodar_referencias(precos: pd.DataFrame, cfg: ConfigBacktest) -> dict[str, ResultadoBacktest | None]:
    """
    BOVA11, SMAL11 e DIVO11 pelo ``walk_forward`` comum sobre ``retornos_da_carteira`` (as mesmas datas e custos do
    motor dinâmico). None = "sem histórico": a referência não tinha cotação no fechamento da 1ª decisão.
    """
    retornos = retornos_da_carteira(precos[list(REFERENCIAS)])
    primeira_decisao = retornos.index[cfg.janela_treino - 1]      # fechamento de t-1 do 1º rebalanceio
    return {ref: walk_forward(retornos, estrategia_buy_and_hold_ativo(ref), cfg, ref)
            if precos.loc[:primeira_decisao, ref].notna().any() else None
            for ref in REFERENCIAS}


def estrategias(cfg: ConfigOtimizacao, cdi: pd.Series | None) -> dict[str, Estrategia]:
    """As 7 estratégias da lista fechada (as misturas saem delas), montadas como no teste da sessão 13."""
    return {
        "Kelly λ=1": estrategia_otimizada(replace(cfg, aversao_risco=1.0), True, cdi),
        CONTROLE: estrategia_otimizada(replace(cfg, aversao_risco=2.0), True, cdi),
        "Kelly λ=4": estrategia_otimizada(replace(cfg, aversao_risco=4.0), True, cdi),
        "Máx. Sharpe": estrategia_otimizada(replace(cfg, objetivo=Objetivo.MAX_SHARPE), True, cdi),
        "Mín. Vol": estrategia_otimizada(replace(cfg, objetivo=Objetivo.MIN_VOL), True, cdi),
        "HRP": estrategia_hrp(),
        "1/N": estrategia_equiponderada(),
    }


def rodar_horizonte(precos: pd.DataFrame, universo: pd.DataFrame, cfg: ConfigOtimizacao, cfg_bt: ConfigBacktest,
                    cdi: pd.Series | None) -> dict[str, ResultadoBacktest | None]:
    """
    Um horizonte já recortado: as 7 estratégias no motor dinâmico, as 3 referências e as 4 misturas, todas nas
    mesmas datas. Referência sem histórico = None; o BOVA11 é obrigatório (é núcleo de duas misturas).
    """
    acoes = precos.drop(columns=list(REFERENCIAS))
    resultados: dict[str, ResultadoBacktest | None] = {
        nome: walk_forward_dinamico(acoes, universo, estrategia, cfg_bt, nome)
        for nome, estrategia in estrategias(cfg, cdi).items()}
    resultados.update(rodar_referencias(precos, cfg_bt))
    if resultados["BOVA11"] is None:
        raise ValueError("BOVA11 sem histórico na 1ª decisão: as misturas com núcleo BOVA11 não rodam.")
    for nome, (nucleo, peso_nucleo) in COMPOSICAO_MISTURAS.items():
        resultados[nome] = misturar(resultados[nucleo], resultados[CONTROLE], peso_nucleo, cfg_bt, nome)
    return resultados


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


def _psr_da_diferenca(valores: pd.Series, valores_referencia: pd.Series) -> float:
    return sharpe_probabilistico(momentos_sharpe(retornos_de_valores(valores) - retornos_de_valores(valores_referencia)))


def montar_tabela(resultados: Mapping[str, ResultadoBacktest], medidas: Mapping[str, Medidas],
                  rf_diaria: pd.Series) -> pd.DataFrame:
    """
    Uma linha por curva. DSR só nas 11 da lista fechada (as referências não foram escolhidas); nº de ativos e HHI
    médios dos pesos-alvo só nas 7 estratégias (os pesos das misturas são núcleo e satélite).
    """
    bova = resultados.get("BOVA11")
    linhas, momentos = {}, {}
    for nome, resultado in resultados.items():
        resumo = resumo_desempenho(resultado.valores, rf_diaria)
        retornos = retornos_de_valores(resultado.valores)
        momentos[nome] = momentos_sharpe(retornos - rf_diaria.reindex(retornos.index).fillna(0.0))
        pesos = resultado.pesos_alvo if nome in ESTRATEGIAS else pd.DataFrame()
        linhas[nome] = {
            "Retorno total": resumo.retorno_total, "CAGR": resumo.cagr, "Vol": resumo.volatilidade,
            "Sharpe": resumo.sharpe, "Sortino": resumo.sortino, "MDD": resumo.max_drawdown,
            "CVaR95 diário": resumo.cvar_95_diario, "Giro anual": resultado.turnover_anual_medio,
            "Nº ativos": float((pesos > 1e-6).sum(axis=1).mean()) if not pesos.empty else float("nan"),
            "HHI": float((pesos**2).sum(axis=1).mean()) if not pesos.empty else float("nan"),
            "PSR": sharpe_probabilistico(momentos[nome]), "DSR": float("nan"),
            f"P[{CONTROLE} > X]": medidas[nome].p_controle_supera,
            "P[> BOVA11]": (_psr_da_diferenca(resultado.valores, bova.valores)
                            if bova is not None and nome != "BOVA11" else float("nan")),
            "Falhas": len(resultado.falhas)}
    lista_fechada = [nome for nome in (*ESTRATEGIAS, *MISTURAS) if nome in resultados]
    sharpes = [momentos[nome].sharpe for nome in lista_fechada]
    referencia = sharpe_referencia_deflacionado(float(np.var(sharpes, ddof=1)), len(lista_fechada))
    for nome in lista_fechada:
        linhas[nome]["DSR"] = sharpe_probabilistico(momentos[nome], referencia)
    return pd.DataFrame(linhas).T


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
    return "2" if vence else "3"


def decidir_mistura(h: Horizontes) -> str | None:
    """Regra B do pré-registro: a mistura aprovada nos dois horizontes com maior Sharpe médio, ou None."""
    def aprovada(mistura: str) -> bool:
        return all(m[mistura].sharpe >= m[CONTROLE].sharpe - FOLGA_SHARPE_MISTURA - _TOLERANCIA
                   and m[mistura].max_drawdown - m[CONTROLE].max_drawdown >= MELHORA_MINIMA_MDD - _TOLERANCIA
                   for m in h.values())

    aprovadas = [mistura for mistura in MISTURAS if aprovada(mistura)]
    if not aprovadas:
        return None
    return max(aprovadas, key=lambda mistura: sum(m[mistura].sharpe for m in h.values()) / len(h))


@dataclass(frozen=True)
class Avaliacao:
    dados: DadosLago
    selic: float                                          # Selic meta de hoje (taxa constante sem o CDI)
    cdi_disponivel: bool
    resultados: dict[int, dict[str, ResultadoBacktest | None]]
    rf_diaria: dict[int, pd.Series]
    tabelas: dict[int, pd.DataFrame]
    opcao_objetivo: str                                   # regra A
    mistura: str | None                                   # regra B


def avaliar(dados: DadosLago, cdi: pd.Series, selic: float, cfg_bt: ConfigBacktest = CFG_BT,
            horizontes: tuple[int, ...] = HORIZONTES) -> Avaliacao:
    """O teste inteiro sobre dados já carregados (sem rede). CDI vazio = Selic constante."""
    cfg = replace(CFG_KELLY, taxa_livre_risco=taxa_anual_para_aritmetica(selic))
    cdi_vigente = cdi if not cdi.empty else None
    resultados, rf, tabelas, medidas = {}, {}, {}, {}
    for anos in horizontes:
        recorte = recortar_horizonte(dados.precos, anos, cfg_bt.janela_treino)
        resultados[anos] = rodar_horizonte(recorte, dados.universo, cfg, cfg_bt, cdi_vigente)
        com_historico = {nome: r for nome, r in resultados[anos].items() if r is not None}
        rf[anos] = market_data.serie_rf_diaria(com_historico[CONTROLE].valores.index, cdi_vigente, selic)
        medidas[anos] = medir(com_historico, rf[anos])
        tabelas[anos] = montar_tabela(com_historico, medidas[anos], rf[anos])
    return Avaliacao(dados=dados, selic=selic, cdi_disponivel=cdi_vigente is not None, resultados=resultados,
                     rf_diaria=rf, tabelas=tabelas, opcao_objetivo=decidir_objetivo(medidas),
                     mistura=decidir_mistura(medidas))


def _porcento(valor: float) -> str:
    return f"{valor:.2%}".replace(".", ",")


_COLUNAS_EM_PORCENTO = ("Retorno total", "CAGR", "Vol", "MDD", "CVaR95 diário", "Giro anual")
_COLUNAS_EM_DECIMAL = ("Sharpe", "Sortino", "PSR", "DSR", f"P[{CONTROLE} > X]", "P[> BOVA11]", "HHI", "Nº ativos")


def _formatar(tabela: pd.DataFrame) -> pd.DataFrame:
    formatada = tabela.astype(float).copy().astype(object)
    for coluna in _COLUNAS_EM_PORCENTO:
        formatada[coluna] = tabela[coluna].astype(float).map(_porcento)
    for coluna in _COLUNAS_EM_DECIMAL:
        formatada[coluna] = tabela[coluna].astype(float).map("{:.3f}".format)
    formatada["Falhas"] = tabela["Falhas"].astype(int)
    return formatada


def imprimir(avaliacao: Avaliacao) -> None:
    """Dados (sha256 e manifest), uma tabela por horizonte, falhas e as duas regras."""
    dados = avaliacao.dados
    print("Dados do lago: " + ", ".join(f"{campo} {valor}" for campo, valor in dados.manifest.items()))
    for nome, digest in dados.sha256.items():
        print(f"  sha256 {nome}: {digest}")
    if avaliacao.cdi_disponivel:
        print("Taxa livre de risco: CDI na data (BCB SGS 12).")
    else:
        print(f"AVISO: CDI indisponível: taxa livre de risco = Selic constante de {_porcento(avaliacao.selic)} "
              "(as diferenças entre estratégias não mudam; os Sharpes absolutos, sim).")
    for anos, resultados in avaliacao.resultados.items():
        datas = resultados[CONTROLE].valores.index
        print(f"\n=== {anos} anos fora da amostra: {datas[0]:%d/%m/%Y} a {datas[-1]:%d/%m/%Y} "
              f"({len(datas) - 1} pregões), {CFG_BT.custo_bps:g} bps por lado ===")
        print(_formatar(avaliacao.tabelas[anos]).to_string())
        print(f"Papéis treinados por rebalanceio (média, = nº de ativos do 1/N): "
              f"{avaliacao.tabelas[anos].loc['1/N', 'Nº ativos']:.1f}")
        for nome, resultado in resultados.items():
            if resultado is None:
                print(f"  {nome}: sem histórico na 1ª decisão")
                continue
            for falha in resultado.falhas:
                print(f"  falha {nome} {falha}")
    print(f"\nRegra A (Kelly × objetivos): opção {avaliacao.opcao_objetivo}")
    print(f"Regra B (núcleo + satélite): {avaliacao.mistura or 'nenhuma mistura aprovada; vale a regra A'}")


def main() -> int:
    hoje = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    inicio = time.perf_counter()
    dados = carregar_lago()
    selic = market_data.obter_selic_meta(hoje)
    try:
        cdi = market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=hoje)
    except market_data.ErroDadosMercado as erro:
        print(f"CDI: {erro}")
        cdi = pd.Series(dtype=float)
    print(f"Selic meta {_porcento(selic.valor)} ({selic.fonte})")
    imprimir(avaliar(dados, cdi, selic.valor))
    print(f"(tempo total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
