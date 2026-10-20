"""
Alvo de volatilidade com EGARCH (DECISOES 55, PLAN §11: depois do S2, pré-registro próprio): cada estratégia do
S2.2/S2.3 corta a exposição quando a vol prevista passa da vol recente e deixa o resto no CDI. Não muda o paper trading
(paper-v1) nem o critério de pronto. Rede só para o CDI e a Selic meta.

Uso: .venv/Scripts/python -X utf8 scripts/avaliar_vol_alvo.py > docs/vol-alvo-2026-10-20.txt

Pré-registro (2026-10-20, escrito e commitado antes de rodar; depois desse commit o texto não muda):
- Dados e curvas originais: os do `scripts/avaliar_valor.py` (os mesmos 7 arquivos, conferidos pelo hash), com o motor
  e as funções dele sem mudança: Fórmula Mágica, Trending Value, Conservadora e "Kelly λ=2 com regras", nas 3 fases
  com a 1ª decisão em 02/01, 01/02 e 05/03/2012, e o BOVA11 nas mesmas datas.
- Alvo de vol (`core.vol_alvo.curva_com_alvo` com `core.vol_alvo.vol_egarch`), sobre a curva diária de cada uma: no
  fechamento de cada decisão, um EGARCH(1,1) de Nelson com o termo de assimetria, erros t de Student e média constante,
  por máxima verossimilhança nos 504 retornos diários da curva até ali (x 100); vol prevista = raiz(252 x a média das
  variâncias dos 21 pregões seguintes), de 1.000 caminhos simulados com a semente 20.261.020 na t; alvo = desvio-padrão
  dos mesmos 504 retornos x raiz(252) (a vol realizada de 2 anos: nenhum parâmetro livre); exposição = min(1, alvo /
  prevista), sem alavancagem; o resto rende o CDI do BCB (Selic meta constante onde faltar). Reajuste a cada 21
  pregões, a 1ª decisão no fechamento de 15/01, 13/02 e 18/03/2014 nas 3 fases; entre os reajustes as duas partes
  derivam; 10 bps sobre |exposição nova - a que derivou|; antes do 1º reajuste a curva está toda exposta. Falha do
  ajuste (o otimizador não converge, a previsão é 0, infinita ou fica a mais de 10x do alvo para cima ou para baixo) =
  mantém a exposição que derivou, sem negócio, e conta. A regra dos 10x veio de um diagnóstico nas curvas da fase 0
  desta rodada, feito antes deste texto e só sobre a previsão (nenhum retorno com alvo calculado): 5 a 11% das
  decisões sem convergência e previsões de 1e-33 a 6e15 vezes a vol da janela.
- Aproximação escrita aqui: a curva original já é líquida dos custos dela com 100% exposta; com exposição e, o custo
  e o giro dos rebalanceios dela entram x e.
- Comparação, por fase, no trecho do alvo (do fechamento da 1ª decisão dele a 16/10/2026; ~12,7 anos): P[com alvo >
  original] aritmético, MDD das duas, giro anual das duas (`comparar`), exposição média e falhas; e a tabela do
  trecho contra o BOVA11 (Sharpe sobre o CDI, IR, P[> BOVA11] aritmético e em log, DSR com N = linhas de
  `docs/tentativas.csv`, já com estas quatro: 29, V = variância dos IR diários das quatro com alvo; MinTRL; CAGR e MDD)
  para as quatro com alvo e as quatro originais no mesmo trecho.
- Decisão fixa agora, para cada estratégia (o molde da R10 do passo 3): aprovado = nas 3 fases, P[com alvo >
  original] >= 0,95 (C1), MDD com alvo no máximo 5 p.p. pior que o da original (C2) e giro com alvo <= 2x o da
  original (C3). Aprovado = a versão com alvo substitui a original nas provas seguintes (na prova final, a candidata é
  a Conservadora). Reprovado = fica a original; o alvo não volta com outra janela, passo, alvo ou modelo sem
  pré-registro novo, contado no N.
- Expectativa registrada: o alvo corta o MDD (Conservadora de −41% para perto de −30%; Kelly de −62% para perto de
  −45%) e a vol, com exposição média de 0,8 a 0,9; P[com alvo > original] entre 0,2 e 0,8 (sair nas altas de vol
  perde parte das recuperações, o CDI alto compensa uma parte); nenhuma aprovada; falhas em 5 a 15% das decisões.
"""

from __future__ import annotations

import sys
import time
import warnings
from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import ResultadoBacktest  # noqa: E402
from core.config import DIAS_UTEIS_ANO, taxa_anual_para_aritmetica  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.vol_alvo import CurvaComAlvo, curva_com_alvo, vol_egarch  # noqa: E402
from scripts import avaliar_valor as valor  # noqa: E402
from scripts.avaliar_robustez import FASES, _tabela, carregar, contar_tentativas, tabela_do_periodo  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_KELLY, _porcento  # noqa: E402

P_MINIMO = 0.95             # C1: P[com alvo > original], aritmético
PIORA_MAXIMA_MDD = 0.05     # C2: MDD com alvo no máximo 5 p.p. pior
GIRO_MAXIMO = 2.0           # C3: giro anual com alvo <= 2x o da original
_TOLERANCIA = 1e-9
COM_ALVO = {nome: f"{nome.split(' (')[0]} com alvo de vol" for nome in valor.ESTRATEGIAS}


def alvo_passa(comparacoes: pd.DataFrame) -> bool:
    """C1, C2 e C3 em todas as linhas (uma por fase), como a R10 do passo 3."""
    c1 = comparacoes["P[alvo > original]"] >= P_MINIMO - _TOLERANCIA
    c2 = comparacoes["MDD com alvo"] >= comparacoes["MDD original"] - PIORA_MAXIMA_MDD - _TOLERANCIA
    c3 = comparacoes["Giro com alvo"] <= GIRO_MAXIMO * comparacoes["Giro original"] + _TOLERANCIA
    return bool((c1 & c2 & c3).all())


def comparar(original: ResultadoBacktest, alvo: CurvaComAlvo) -> dict[str, float]:
    """
    A curva com alvo contra a original no trecho do alvo (do fechamento da 1ª decisão dele ao fim): P[com alvo >
    original] aritmético, MDD das duas, giro anual da original (os rebalanceios depois do começo do trecho) e com alvo
    (cada rebalanceio da original vezes a exposição em vigor nele, a decidida antes, mais o |Δe| de cada reajuste),
    exposição média e falhas do ajuste.
    """
    inicio = alvo.valores.index[0]
    no_trecho = original.valores.loc[inicio:]
    anos = (len(alvo.valores) - 1) / DIAS_UTEIS_ANO
    giro = original.turnover[original.turnover.index > inicio]
    em_vigor = alvo.exposicoes.to_numpy()[alvo.exposicoes.index.searchsorted(giro.index, side="left") - 1]
    return {"P[alvo > original]": psr_da_diferenca(alvo.valores, no_trecho),
            "MDD com alvo": drawdown(alvo.valores)[1], "MDD original": drawdown(no_trecho)[1],
            "Giro com alvo": float((giro.to_numpy() * em_vigor).sum() + alvo.giro.sum()) / anos,
            "Giro original": float(giro.sum()) / anos, "Exposição média": float(alvo.exposicoes.mean()),
            "Falhas do ajuste": float(len(alvo.falhas))}


# ---------------------------------------------------------------- rodada (com o CDI da rede)

def curvas_com_alvo(curvas: Mapping[str, ResultadoBacktest], rf: pd.Series) -> dict[str, CurvaComAlvo]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")         # avisos do otimizador do arch a cada ajuste; as falhas contam
        alvos = {nome: curva_com_alvo(curvas[nome].valores, rf, vol_egarch) for nome in valor.ESTRATEGIAS}
    if len({alvo.valores.index[0] for alvo in alvos.values()}) != 1:
        raise SystemExit("Curvas com alvo começando em datas diferentes")
    return alvos


def tabela_da_fase(curvas: Mapping[str, ResultadoBacktest], alvos: Mapping[str, CurvaComAlvo], rf: pd.Series,
                   n_tentativas: int) -> pd.DataFrame:
    """As quatro com alvo e as quatro originais no trecho do alvo contra o BOVA11, com CAGR e MDD (e os do BOVA11)."""
    inicio = alvos[valor.SATELITE].valores.index[0]
    bova = curvas["BOVA11"].valores.loc[inicio:]
    com_alvo = {COM_ALVO[nome]: alvos[nome].valores for nome in valor.ESTRATEGIAS}
    originais = {nome: curvas[nome].valores.loc[inicio:] for nome in valor.ESTRATEGIAS}
    tabela = pd.concat([tabela_do_periodo(com_alvo, bova, rf, n_tentativas),
                        tabela_do_periodo(originais, bova, rf, n_tentativas)])
    medidas = {nome: {"CAGR": cagr(serie), "MDD": drawdown(serie)[1]}
               for nome, serie in {**com_alvo, **originais, "BOVA11": bova}.items()}
    return pd.DataFrame(medidas).T.join(tabela)


def main() -> int:
    inicio_rodada = time.perf_counter()
    hoje = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    dados, mensal = carregar(), valor.carregar_mensal()
    balancos, sha256_cvm = valor.carregar_balancos()
    selic = market_data.obter_selic_meta(hoje)
    try:
        cdi = market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=hoje)
    except market_data.ErroDadosMercado as erro:
        print(f"CDI: {erro} (Selic constante)")
        cdi = pd.Series(dtype=float)
    cdi_vigente = cdi if not cdi.empty else None
    cfg = replace(CFG_KELLY, taxa_livre_risco=taxa_anual_para_aritmetica(selic.valor))
    n_tentativas = contar_tentativas()
    corte = valor.corte_dos_precos(dados.precos.index, valor.PRIMEIRA_DECISAO, valor.CFG_BT.janela_treino)
    print(f"Arquivos (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm}")
    print(f"Manifest: {dados.manifest}; Selic meta {_porcento(selic.valor)} ({selic.fonte}); "
          f"CDI {'do BCB' if cdi_vigente is not None else 'indisponível'}; N do DSR = {n_tentativas}; "
          f"preços desde {dados.precos.index[corte]:%d/%m/%Y}")

    notas_do_mes = valor.notas(balancos, mensal)
    por_estrategia: dict[str, list[dict[str, float]]] = {nome: [] for nome in valor.ESTRATEGIAS}
    for fase in FASES:
        curvas = valor.curvas_da_fase(dados, mensal, notas_do_mes, corte, fase, cdi_vigente, cfg)
        rf = market_data.serie_rf_diaria(curvas[valor.SATELITE].valores.index, cdi_vigente, selic.valor)
        alvos = curvas_com_alvo(curvas, rf)
        comparacoes = pd.DataFrame({nome: comparar(curvas[nome], alvos[nome]) for nome in valor.ESTRATEGIAS}).T
        for nome in valor.ESTRATEGIAS:
            por_estrategia[nome].append(comparacoes.loc[nome].to_dict())
        primeiro = alvos[valor.SATELITE]
        print(f"\n=== Fase {fase}: 1ª decisão do alvo {primeiro.exposicoes.index[0]:%d/%m/%Y}, fim "
              f"{primeiro.valores.index[-1]:%d/%m/%Y}, {len(primeiro.exposicoes)} reajustes ===")
        print(_tabela(comparacoes.astype({"Falhas do ajuste": int}),
                      ["MDD com alvo", "MDD original", "Giro com alvo", "Giro original", "Exposição média"]))
        print(_tabela(tabela_da_fase(curvas, alvos, rf, n_tentativas), ["CAGR", "MDD"]))
        print(f"(fase {fase} pronta em {time.perf_counter() - inicio_rodada:.0f} s)", flush=True)

    print()
    for nome in valor.ESTRATEGIAS:
        aprovado = alvo_passa(pd.DataFrame(por_estrategia[nome]))
        print(f"{COM_ALVO[nome]}: {'APROVADO (substitui a original)' if aprovado else 'REPROVADO (fica a original)'}")
    print(f"(tempo total {time.perf_counter() - inicio_rodada:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
