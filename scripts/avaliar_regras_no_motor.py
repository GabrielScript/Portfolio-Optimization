"""
Regras de risco fixas no motor dinâmico (item g do PLAN), com contagem de disparos. Rede só para o CDI e a Selic.

Uso: .venv/Scripts/python -X utf8 scripts/avaliar_regras_no_motor.py

Pré-registro (2026-10-18, escrito e commitado antes de rodar; depois desse commit o texto não muda):
- Dados: exports do lago `b3-datalake` `023f28e` em `data/b3lake/`: `universe_prices.parquet` e
  `universe_monthly.parquet` (os do S1-P.b, refeitos depois da correção do custo atribuído de bonificação, lago
  `89803d5`: AXIA3, AXIA6, CYRE3 e RENT3 mudam um dia cada) e `sectors.parquet` (setor por emissor: `SETOR_ATIV` do
  cadastro da CVM, com as deslistadas, mapeado nos 10 setores do app). A saída imprime o sha256 dos três e as datas
  do manifest.
- Protocolo do S1-P.b sem mudança (`scripts/avaliar_universo_dinamico.py`): horizontes de 5 e 10 anos fora da
  amostra, treino de 1.260 pregões, rebalanceio a cada 63, 10 bps por lado, Jorion, taxa livre de risco = CDI na data
  (Selic constante se o BCB falhar, com aviso), peso máximo 20%, até 10 ativos, teto de vol 40%.
- Linha de base, regras desligadas: as 11 curvas da lista fechada do S1-P.b e as 3 referências, pelo `avaliar` dele.
  É a checagem de robustez da DECISOES 38: mesmo protocolo, dados corrigidos; imprime a diferença de Sharpe e MDD
  para `docs/universo-dinamico-2026-10-18.txt` e as regras A e B de novo. As duas saídas ficam; nenhuma substitui a
  outra.
- Regras ligadas só no que vira ordem, o padrão da tela ("Núcleo Mín. Vol 70/30 com regras"):
  1. núcleo Mín. Vol e satélite Kelly λ = 2 por `estrategia_otimizada_nas_regras`: papel de emissor sem setor no lago
     não é elegível; teto de 30% por setor no otimizador (o setor do lago para vivas e mortas: uma taxonomia só); a
     carteira passa por `checar_carteira` (ação 20%, emissor 20%, setor 30%, vol ex-ante 40%, alavancagem 1,0), e a
     que viola é recusada: o motor mantém a atual. Recusa no 1º rebalanceio de um horizonte (não há carteira para
     manter) é erro do script, não resultado: vai ao DECISOES e o pré-registro é refeito.
  2. perda diária: satélite com perda de 5% ou mais no pregão da decisão = nenhuma ordem no seguinte; o rebalanceio
     do satélite, o do núcleo e a volta aos 70/30 vão para o 1º pregão livre (`Congelamento`, a regra do
     `pode_negociar`).
  3. stop relativo: em cada rebalanceio do satélite, se a razão satélite/BOVA11 sabida na véspera caiu 25% ou mais
     desde o pico, o satélite inteiro vai para o BOVA11 nesse rebalanceio (10 bps na venda e 10 na compra) e fica lá
     até o fim do horizonte.
  Também impressas: "Kelly λ=2 com regras" (o satélite com 1 e 2, sem o stop) e "Mín. Vol com regras".
- Regras só contadas no resto (as 7 estratégias sem regras, que não são bloqueadas): rebalanceios cuja carteira-alvo
  violaria cada regra, sem a vol ex-ante (pede a cov do treino), com o setor do lago.
- Saída por horizonte: CAGR, vol, Sharpe, MDD, giro, nº médio de ativos, PSR e DSR com e sem regras, DSR com N = 12
  (as 11 + a mistura com regras); P[com regras > sem] (PSR da diferença diária); recusas por regra; rebalanceios
  adiados; pregões em que o satélite perdeu 5% ou mais; data do stop ou "não disparou"; papel-meses do universo sem
  setor no lago.
- Não há troca a decidir: as regras são fixas e valem para o que vira ordem. Registro, antes de olhar: se a mistura
  com regras tiver Sharpe mais de 0,10 abaixo da sem regras em algum horizonte, ou se o satélite for recusado em mais
  de 1 a cada 4 rebalanceios, o achado vai para o DECISOES e as regras ficam como estão (mudar um valor = pré-registro
  novo).
- Expectativa registrada: recusas perto de 0 nas otimizadas (o teto de setor vem no solver; ação e vol já vinham); o
  teto de setor custa pouco Sharpe ao Kelly e menos à Mín. Vol; stop: com TE de 16–19% sem alfa, 14–25% de chance no
  1º ano e 60–75% em 3 anos, então o mais provável é disparar nos 10 anos e talvez nos 5; perda de 5% num pregão no
  satélite: poucas dezenas de pregões em 10 anos, 0 a 2 rebalanceios adiados.
"""

from __future__ import annotations

import hashlib
import sys
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import (Congelamento, ResultadoBacktest, backtest_pesos_fixos,  # noqa: E402
                           estrategia_otimizada_nas_regras, walk_forward_dinamico)
from core.config import ConfigBacktest, ConfigOtimizacao, Objetivo, taxa_anual_para_aritmetica  # noqa: E402
from core.lago import ler_setores  # noqa: E402
from core.metrics import (momentos_sharpe, resumo_desempenho, retornos_de_valores,  # noqa: E402
                          sharpe_probabilistico, sharpe_referencia_deflacionado)
from core.risco import LIMITES, TOLERANCIA, Limites, checar_carteira, stop_relativo  # noqa: E402
from scripts.avaliar_universo_dinamico import (ANOS_DE_CDI, ARQUIVO_UNIVERSO, CFG_BT, CFG_KELLY,  # noqa: E402
                                               CONTROLE, ESTRATEGIAS, HORIZONTES, MISTURAS, PASTA_LAGO, REFERENCIAS,
                                               Avaliacao, DadosLago, _porcento, _psr_da_diferenca, avaliar,
                                               carregar_lago, imprimir, recortar_horizonte)

ARQUIVO_SETORES = "sectors.parquet"
SATELITE, NUCLEO, MISTURA = "Kelly λ=2 com regras", "Mín. Vol com regras", "Núcleo Mín. Vol 70/30 com regras"
PESO_NUCLEO = 0.70
PARES = {SATELITE: CONTROLE, NUCLEO: "Mín. Vol", MISTURA: "Núcleo Mín. Vol 70/30"}     # com regras -> sem regras
LISTA_DO_DSR = (*ESTRATEGIAS, *MISTURAS, MISTURA)                                       # N = 12
# Sharpe e MDD da saída commitada do S1-P.b (docs/universo-dinamico-2026-10-18.txt), para a robustez
S1PB_COMMITADO: dict[int, dict[str, tuple[float, float]]] = {
    5: {"Kelly λ=1": (-0.357, -0.4224), "Kelly λ=2": (-0.310, -0.3876), "Kelly λ=4": (-0.306, -0.3127),
        "Máx. Sharpe": (-0.314, -0.3364), "Mín. Vol": (0.036, -0.1367), "HRP": (-0.174, -0.1836),
        "1/N": (-0.222, -0.2704), "BOVA11": (0.056, -0.2103), "SMAL11": (-0.387, -0.3618),
        "DIVO11": (0.229, -0.1612), "Núcleo Mín. Vol 70/30": (-0.128, -0.1637),
        "Núcleo Mín. Vol 50/50": (-0.209, -0.2313), "Núcleo BOVA11 70/30": (-0.076, -0.2299),
        "Núcleo BOVA11 50/50": (-0.159, -0.2757)},
    10: {"Kelly λ=1": (0.371, -0.6421), "Kelly λ=2": (0.363, -0.6305), "Kelly λ=4": (0.302, -0.6167),
         "Máx. Sharpe": (0.294, -0.5632), "Mín. Vol": (0.280, -0.3940), "HRP": (0.131, -0.4598),
         "1/N": (0.082, -0.5075), "BOVA11": (0.093, -0.4693), "SMAL11": (-0.032, -0.5106),
         "DIVO11": (0.235, -0.4010), "Núcleo Mín. Vol 70/30": (0.346, -0.4625),
         "Núcleo Mín. Vol 50/50": (0.362, -0.5094), "Núcleo BOVA11 70/30": (0.198, -0.5160),
         "Núcleo BOVA11 50/50": (0.259, -0.5447)},
}
S1PB_REGRAS = ("3", "Núcleo Mín. Vol 70/30")       # regra A e regra B da saída commitada


def data_do_stop(satelite: ResultadoBacktest, referencia: ResultadoBacktest,
                 limites: Limites = LIMITES) -> pd.Timestamp | None:
    """
    1º rebalanceio do satélite em que a razão satélite/BOVA11, sabida até o fechamento da véspera (a decisão), caiu
    ``queda_relativa_stop`` ou mais desde o pico; None se não disparou.
    """
    for data in satelite.pesos_alvo.index:
        sabido = satelite.valores.index < data
        if stop_relativo(satelite.valores[sabido], referencia.valores[referencia.valores.index < data], limites):
            return data
    return None


def satelite_com_stop(satelite: ResultadoBacktest, referencia: ResultadoBacktest, data: pd.Timestamp,
                      custo_bps: float) -> ResultadoBacktest:
    """
    O satélite inteiro vai para o BOVA11 no rebalanceio ``data`` e fica lá até o fim (volta só com pré-registro novo):
    até a véspera, a curva do satélite; daí em diante, o valor da véspera menos o custo de vender tudo e comprar o
    BOVA11 (2 x ``custo_bps``), andando com a referência.
    """
    antes = satelite.valores[satelite.valores.index < data]
    depois = referencia.valores[referencia.valores.index >= data]
    base = referencia.valores[referencia.valores.index < data].iloc[-1]
    trocado = antes.iloc[-1] * (1.0 - 2.0 * custo_bps / 1e4) * depois / base
    return replace(satelite, nome=f"{satelite.nome} com stop", valores=pd.concat([antes, trocado]))


def setores_dos_papeis(universo: pd.DataFrame, setores: pd.Series) -> dict[str, str]:
    """Papel -> setor do lago pelo emissor do universo (``issuer``); papel de emissor sem setor fica de fora."""
    emissores = universo.drop_duplicates("security").set_index("security")["issuer"]
    return {papel: setores[emissor] for papel, emissor in emissores.items() if emissor in setores.index}


def violacoes_por_regra(resultado: ResultadoBacktest, setores: Mapping[str, str],
                        limites: Limites = LIMITES) -> dict[str, int]:
    """Rebalanceios cuja carteira-alvo violaria cada regra fixa (sem a vol ex-ante, que pede a cov do treino)."""
    contagem: Counter[str] = Counter()
    for _, alvo in resultado.pesos_alvo.iterrows():
        contagem.update({violacao.regra.value for violacao in checar_carteira(alvo[alvo > 0], setores, None, limites)})
    return dict(contagem)


def recusas_por_regra(resultado: ResultadoBacktest) -> dict[str, int]:
    """Rebalanceios recusados pelo motor, por regra violada (um rebalanceio conta uma vez em cada regra)."""
    contagem: Counter[str] = Counter()
    for violacoes in resultado.recusas.values():
        contagem.update({violacao.regra.value for violacao in violacoes})
    return dict(contagem)


def misturar_congelando(nucleo: ResultadoBacktest, satelite: ResultadoBacktest, peso_nucleo: float,
                        cfg: ConfigBacktest, nome: str, congelamento: Congelamento) -> ResultadoBacktest:
    """O ``misturar`` do S1-P.b com a volta aos pesos-alvo congelada pela perda diária do satélite."""
    retornos = pd.DataFrame({"núcleo": retornos_de_valores(nucleo.valores),
                             "satélite": retornos_de_valores(satelite.valores)})
    alvo = pd.Series({"núcleo": peso_nucleo, "satélite": 1.0 - peso_nucleo})
    mistura = backtest_pesos_fixos(retornos, alvo, cfg, nome, congelamento)
    return replace(mistura, metodologia="walk-forward OOS, núcleo + satélite com regras",
                   valores=mistura.valores.set_axis(nucleo.valores.index))


@dataclass(frozen=True)
class Disparos:
    stop: pd.Timestamp | None                  # rebalanceio em que o satélite foi para o BOVA11
    pregoes_de_perda: int                      # pregões em que o satélite perdeu 5% ou mais
    adiamentos: dict[str, dict[str, str]]      # curva com regras -> data marcada -> data feita
    recusas: dict[str, dict[str, int]]         # curva com regras -> regra -> rebalanceios recusados
    rebalanceios: dict[str, int]               # curva com regras -> nº de rebalanceios
    papel_meses_sem_setor: int                 # no trecho fora da amostra
    contadas: dict[str, dict[str, int]]        # estratégia sem regras -> regra -> rebalanceios que violariam


def rodar_com_regras(recorte: pd.DataFrame, universo: pd.DataFrame, setores: Mapping[str, str],
                     cfg: ConfigOtimizacao, cfg_bt: ConfigBacktest, cdi: pd.Series | None,
                     bova: ResultadoBacktest) -> tuple[dict[str, ResultadoBacktest], Disparos]:
    """Satélite, núcleo e mistura com as três regras, nas datas da linha de base (``bova`` é a do mesmo recorte)."""
    acoes = recorte.drop(columns=list(REFERENCIAS))
    kelly = estrategia_otimizada_nas_regras(replace(cfg, aversao_risco=2.0), True, setores, cdi)
    satelite = walk_forward_dinamico(acoes, universo, kelly, cfg_bt, SATELITE, Congelamento())
    stop = data_do_stop(satelite, bova)
    efetivo = satelite if stop is None else satelite_com_stop(satelite, bova, stop, cfg_bt.custo_bps)
    retornos_do_satelite = retornos_de_valores(efetivo.valores)
    congelamento = Congelamento(retornos=retornos_do_satelite)
    minima = estrategia_otimizada_nas_regras(replace(cfg, objetivo=Objetivo.MIN_VOL), True, setores, cdi)
    nucleo = walk_forward_dinamico(acoes, universo, minima, cfg_bt, NUCLEO, congelamento)
    mistura = misturar_congelando(nucleo, efetivo, PESO_NUCLEO, cfg_bt, MISTURA, congelamento)
    regras = {SATELITE: satelite, NUCLEO: nucleo, MISTURA: mistura}
    meses = universo["month"].between(satelite.valores.index[1].to_period("M").to_timestamp(),
                                      satelite.valores.index[-1].to_period("M").to_timestamp())
    fora = universo[meses]
    disparos = Disparos(
        stop=stop,
        pregoes_de_perda=int((retornos_do_satelite <= -LIMITES.perda_diaria_maxima + TOLERANCIA).sum()),
        adiamentos={nome: resultado.adiamentos for nome, resultado in regras.items()},
        recusas={nome: recusas_por_regra(resultado) for nome, resultado in regras.items() if nome != MISTURA},
        rebalanceios={nome: len(resultado.pesos_alvo) for nome, resultado in regras.items()},
        papel_meses_sem_setor=int((~fora["security"].isin(list(setores))).sum()),
        contadas={})
    return regras, disparos


def tabela_com_regras(base: Mapping[str, ResultadoBacktest], regras: Mapping[str, ResultadoBacktest],
                      rf_diaria: pd.Series) -> pd.DataFrame:
    """Cada curva com regras ao lado da sem regras, e o BOVA11; DSR com N = 12 (as 11 + a mistura com regras)."""
    curvas = {**base, **regras}
    momentos = {}
    for nome, resultado in curvas.items():
        retornos = retornos_de_valores(resultado.valores)
        momentos[nome] = momentos_sharpe(retornos - rf_diaria.reindex(retornos.index).fillna(0.0))
    sharpes = [momentos[nome].sharpe for nome in LISTA_DO_DSR]
    referencia = sharpe_referencia_deflacionado(float(np.var(sharpes, ddof=1)), len(LISTA_DO_DSR))
    linhas = {}
    for nome in (*[curva for com, sem in PARES.items() for curva in (sem, com)], "BOVA11"):
        resultado = curvas[nome]
        resumo = resumo_desempenho(resultado.valores, rf_diaria)
        pesos = resultado.pesos_alvo if nome not in (MISTURA, *MISTURAS, "BOVA11") else pd.DataFrame()
        linhas[nome] = {
            "CAGR": resumo.cagr, "Vol": resumo.volatilidade, "Sharpe": resumo.sharpe, "MDD": resumo.max_drawdown,
            "Giro anual": resultado.turnover_anual_medio,
            "Nº ativos": float((pesos > 1e-6).sum(axis=1).mean()) if not pesos.empty else float("nan"),
            "PSR": sharpe_probabilistico(momentos[nome]),
            "DSR": sharpe_probabilistico(momentos[nome], referencia) if nome in LISTA_DO_DSR else float("nan"),
            "P[com > sem]": (_psr_da_diferenca(resultado.valores, curvas[PARES[nome]].valores)
                             if nome in PARES else float("nan")),
            "Recusas": len(resultado.recusas), "Adiados": len(resultado.adiamentos)}
    return pd.DataFrame(linhas).T


def diferencas_do_s1pb(avaliacao: Avaliacao) -> pd.DataFrame:
    """Sharpe e MDD de agora (dados corrigidos) contra os da saída commitada do S1-P.b, por horizonte e curva."""
    linhas = []
    for anos, commitado in S1PB_COMMITADO.items():
        if anos not in avaliacao.tabelas:
            continue
        tabela = avaliacao.tabelas[anos]
        for nome, (sharpe, mdd) in commitado.items():
            agora_sharpe, agora_mdd = float(tabela.loc[nome, "Sharpe"]), float(tabela.loc[nome, "MDD"])
            linhas.append({"Horizonte": f"{anos} anos", "Curva": nome, "Sharpe": agora_sharpe,
                           "Sharpe commitado": sharpe, "Δ Sharpe": agora_sharpe - sharpe, "MDD": agora_mdd,
                           "MDD commitado": mdd, "Δ MDD (p.p.)": 100 * (agora_mdd - mdd)})
    colunas = ["Horizonte", "Curva", "Sharpe", "Sharpe commitado", "Δ Sharpe", "MDD", "MDD commitado", "Δ MDD (p.p.)"]
    return pd.DataFrame(linhas, columns=colunas).set_index(["Horizonte", "Curva"])


@dataclass(frozen=True)
class DadosRegras:
    lago: DadosLago
    emissores: pd.DataFrame        # month, security, issuer (o universe_monthly inteiro)
    setores: dict[str, str]        # papel -> setor do lago
    sha256_setores: str


def carregar(pasta: Path = PASTA_LAGO) -> DadosRegras:
    """Os exports do S1-P.b (pelo ``carregar_lago`` dele) e o setor de cada papel pelo emissor."""
    lago = carregar_lago(pasta)
    emissores = pd.read_parquet(pasta / ARQUIVO_UNIVERSO, columns=["month", "security", "issuer"])
    emissores["month"] = pd.to_datetime(emissores["month"])
    setores = setores_dos_papeis(emissores, ler_setores(pasta / ARQUIVO_SETORES))
    return DadosRegras(lago=lago, emissores=emissores, setores=setores,
                       sha256_setores=hashlib.sha256((pasta / ARQUIVO_SETORES).read_bytes()).hexdigest())


@dataclass(frozen=True)
class AvaliacaoRegras:
    base: Avaliacao
    dados: DadosRegras
    regras: dict[int, dict[str, ResultadoBacktest]]
    disparos: dict[int, Disparos]
    tabelas: dict[int, pd.DataFrame]


def avaliar_regras(dados: DadosRegras, cdi: pd.Series, selic: float, cfg_bt: ConfigBacktest = CFG_BT,
                   horizontes: tuple[int, ...] = HORIZONTES) -> AvaliacaoRegras:
    """Linha de base pelo ``avaliar`` do S1-P.b e, nos mesmos recortes, as curvas com regras."""
    base = avaliar(dados.lago, cdi, selic, cfg_bt, horizontes)
    cfg = replace(CFG_KELLY, taxa_livre_risco=taxa_anual_para_aritmetica(selic))
    cdi_vigente = cdi if not cdi.empty else None
    regras, disparos, tabelas = {}, {}, {}
    for anos in horizontes:
        recorte = recortar_horizonte(dados.lago.precos, anos, cfg_bt.janela_treino)
        resultados = base.resultados[anos]
        regras[anos], disparo = rodar_com_regras(recorte, dados.lago.universo, dados.setores, cfg, cfg_bt,
                                                 cdi_vigente, resultados["BOVA11"])
        disparos[anos] = replace(disparo, contadas={nome: violacoes_por_regra(resultados[nome], dados.setores)
                                                    for nome in ESTRATEGIAS})
        com_historico = {nome: resultado for nome, resultado in resultados.items() if resultado is not None}
        tabelas[anos] = tabela_com_regras(com_historico, regras[anos], base.rf_diaria[anos])
    return AvaliacaoRegras(base=base, dados=dados, regras=regras, disparos=disparos, tabelas=tabelas)


def _formatar(tabela: pd.DataFrame) -> pd.DataFrame:
    formatada = tabela.astype(object).copy()
    for coluna in ("CAGR", "Vol", "MDD", "Giro anual"):
        formatada[coluna] = tabela[coluna].astype(float).map(_porcento)
    for coluna in ("Sharpe", "PSR", "DSR", "P[com > sem]", "Nº ativos"):
        formatada[coluna] = tabela[coluna].astype(float).map("{:.3f}".format)
    for coluna in ("Recusas", "Adiados"):
        formatada[coluna] = tabela[coluna].astype(int)
    return formatada


def imprimir_regras(avaliacao: AvaliacaoRegras) -> None:
    """Parte 1: linha de base (robustez do S1-P.b) e diferenças; parte 2: as regras no motor e os disparos."""
    print(f"  sha256 {ARQUIVO_SETORES}: {avaliacao.dados.sha256_setores}")
    print("\n##### PARTE 1: linha de base, regras desligadas (robustez do S1-P.b nos dados corrigidos; a saída de "
          "2026-10-18 continua valendo) #####")
    imprimir(avaliacao.base)
    print(f"(saída commitada: regra A opção {S1PB_REGRAS[0]}, regra B {S1PB_REGRAS[1]})")
    print("\nDiferença para docs/universo-dinamico-2026-10-18.txt:")
    print(diferencas_do_s1pb(avaliacao.base).to_string(float_format="{:.3f}".format))
    print("\n##### PARTE 2: regras ligadas no que vira ordem #####")
    for anos, tabela in avaliacao.tabelas.items():
        disparos = avaliacao.disparos[anos]
        stop = f"{disparos.stop:%d/%m/%Y}" if disparos.stop is not None else "não disparou"
        print(f"\n=== {anos} anos fora da amostra ===")
        print(_formatar(tabela).to_string())
        print(f"Stop relativo do satélite: {stop}")
        print(f"Pregões com perda de 5% ou mais no satélite: {disparos.pregoes_de_perda}")
        for nome, adiados in disparos.adiamentos.items():
            datas = ", ".join(f"{marcada} -> {feita}" for marcada, feita in adiados.items()) or "nenhum"
            print(f"Rebalanceios adiados, {nome}: {datas}")
        for nome, recusas in disparos.recusas.items():
            print(f"Recusas, {nome}: {len(avaliacao.regras[anos][nome].recusas)} de {disparos.rebalanceios[nome]} "
                  f"rebalanceios; por regra {recusas or 'nenhuma'}")
        print(f"Papel-meses do universo sem setor no lago (fora da amostra): {disparos.papel_meses_sem_setor}")
        print("Violações só contadas (rebalanceios cuja carteira-alvo violaria, sem a vol ex-ante):")
        for nome, contagem in disparos.contadas.items():
            print(f"  {nome}: {contagem or 'nenhuma'}")


def main() -> int:
    hoje = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    inicio = time.perf_counter()
    dados = carregar()
    selic = market_data.obter_selic_meta(hoje)
    try:
        cdi = market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=hoje)
    except market_data.ErroDadosMercado as erro:
        print(f"CDI: {erro}")
        cdi = pd.Series(dtype=float)
    print(f"Selic meta {_porcento(selic.valor)} ({selic.fonte})")
    imprimir_regras(avaliar_regras(dados, cdi, selic.valor))
    print(f"(tempo total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
