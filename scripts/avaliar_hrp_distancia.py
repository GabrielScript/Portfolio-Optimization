"""
HRP com a distância do paper (DECISOES 87): o HRP e o HRP-Árvore do app clusterizam d_ij = sqrt((1 - rho_ij) / 2);
López de Prado (2016) clusteriza d~_ij = ||D_i - D_j|| (Estágio 1, 2º passo). Testa se a d~ melhora as duas nos
ativos da B3. Não muda o paper trading (paper-v1, paper-v2) nem o critério de pronto. Rede só para o CDI e a Selic.

Uso: .venv/Scripts/python -X utf8 scripts/avaliar_hrp_distancia.py > docs/hrp-distancia-2026-10-20.txt

Pré-registro (2026-10-20, escrito e commitado antes de rodar; depois desse commit o texto não muda):
- Dados: os do `scripts/avaliar_robustez.py` (`carregar`, conferidos pelo hash): `universe_prices.parquet` de
  `docs/dados/2026-10-19/` e `universe_monthly.parquet` de `docs/dados/2026-10-18/`.
- Motor: `walk_forward_dinamico` (universo ponto no tempo, deslistadas incluídas), treino de 1.260 pregões,
  rebalanceio a cada 63, 10 bps por lado (`CFG_BT` do `avaliar_universo_dinamico`), as 3 fases do
  `avaliar_robustez` (0, 21 e 42 pregões cortados do começo). Sem as regras de risco (tetos, setor, vol, stop): é o
  HRP das comparações do app (`estrategia_hrp`), com a covariância Ledoit-Wolf da janela.
- Curvas (lista fechada; as duas com d~ entram no N do DSR, as com d já são do app e não contam de novo):
  HRP (d), HRP (d~), HRP-Árvore (d), HRP-Árvore (d~); linkage single no HRP e ward no HRP-Árvore, como no app.
  Referência fora do N: BOVA11.
- Comparação, por família (HRP; HRP-Árvore) e por fase, no período inteiro (1ª decisão a 16/10/2026): P[d~ > d]
  aritmético (`psr_da_diferenca` da diferença diária), MDD das duas, giro anual das duas e a tabela contra o BOVA11
  (Sharpe sobre o CDI, IR, P aritmético e em log, DSR do IR com N = linhas de `docs/tentativas.csv`, já com estas
  duas, e V = variância dos IR diários das 4 curvas; MinTRL; CAGR e MDD).
- Decisão fixa agora, por família: a d~ vira o padrão (no app e no `core.hierarchical`) se, nas 3 fases, P[d~ > d]
  >= 0,50 (C1), MDD com d~ no máximo 2 p.p. pior (C2) e giro com d~ <= 1,25 x o com d (C3). O limiar de C1 é mais
  fraco que o 0,95 da R10 de propósito: a d~ é o método do paper (o ônus da prova está na variante d) e o HRP é só
  comparador, sem dinheiro. Reprovada = fica a d; a d~ não volta com outro linkage, janela ou estimador sem
  pré-registro novo, contado no N.
- Expectativa registrada: diferença pequena (o mesmo IVP nos dois; só a ordem muda), P[d~ > d] entre 0,3 e 0,7 nas
  duas famílias, MDD a menos de 2 p.p., giro parecido; nenhuma das quatro com P[> BOVA11] >= 0,95.
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Estrategia, ResultadoBacktest, walk_forward_dinamico  # noqa: E402
from core.estimation import covariancia_ledoit_wolf  # noqa: E402
from core.hrp_paper import otimizar_hrp_arvore_paper, otimizar_hrp_paper  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from scripts.avaliar_robustez import (FASES, _acoes, _tabela, carregar, contar_tentativas,  # noqa: E402
                                      referencias, tabela_do_periodo)
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT, _porcento  # noqa: E402

P_MINIMO = 0.50             # C1: P[d~ > d], aritmético
PIORA_MAXIMA_MDD = 0.02     # C2: MDD com d~ no máximo 2 p.p. pior
GIRO_MAXIMO = 1.25          # C3: giro anual com d~ <= 1,25x o com d
_TOLERANCIA = 1e-9
# o core.hierarchical está no caminho vigiado do paper-v1: a opção da distância vive no core.hrp_paper, que com
# distancia="d" dá os mesmos pesos do core.hierarchical (tests/test_hrp_paper.py)
FAMILIAS = {"HRP": otimizar_hrp_paper, "HRP-Árvore": otimizar_hrp_arvore_paper}
DISTANCIAS = {"d": "d", "d~": "d_til"}


def _nome(familia: str, rotulo: str) -> str:
    return f"{familia} ({rotulo})"


def estrategia(familia: str, distancia: str) -> Estrategia:
    alocador = FAMILIAS[familia]
    return lambda treino: alocador(covariancia_ledoit_wolf(treino), distancia=distancia)


def d_til_passa(comparacoes: pd.DataFrame) -> bool:
    """C1, C2 e C3 em todas as linhas (uma por fase)."""
    c1 = comparacoes["P[d~ > d]"] >= P_MINIMO - _TOLERANCIA
    c2 = comparacoes["MDD d~"] >= comparacoes["MDD d"] - PIORA_MAXIMA_MDD - _TOLERANCIA
    c3 = comparacoes["Giro d~"] <= GIRO_MAXIMO * comparacoes["Giro d"] + _TOLERANCIA
    return bool((c1 & c2 & c3).all())


def comparar(d: ResultadoBacktest, d_til: ResultadoBacktest) -> dict[str, float]:
    return {"P[d~ > d]": psr_da_diferenca(d_til.valores, d.valores),
            "MDD d~": drawdown(d_til.valores)[1], "MDD d": drawdown(d.valores)[1],
            "Giro d~": d_til.turnover_anual_medio, "Giro d": d.turnover_anual_medio}


def main() -> int:
    inicio_rodada = time.perf_counter()
    hoje = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    dados = carregar()
    selic = market_data.obter_selic_meta(hoje)
    try:
        cdi = market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=hoje)
    except market_data.ErroDadosMercado as erro:
        print(f"CDI: {erro} (Selic constante)")
        cdi = pd.Series(dtype=float)
    cdi_vigente = cdi if not cdi.empty else None
    n_tentativas = contar_tentativas()
    print(f"Arquivos (sha256): {dados.sha256}")
    print(f"Manifest: {dados.manifest}; Selic meta {_porcento(selic.valor)} ({selic.fonte}); "
          f"CDI {'do BCB' if cdi_vigente is not None else 'indisponível'}; N do DSR = {n_tentativas}")

    por_familia: dict[str, list[dict[str, float]]] = {familia: [] for familia in FAMILIAS}
    for fase in FASES:
        precos = dados.precos.iloc[fase:]
        curvas = {_nome(familia, rotulo): walk_forward_dinamico(_acoes(precos), dados.universo,
                                                                estrategia(familia, distancia), CFG_BT,
                                                                _nome(familia, rotulo))
                  for familia in FAMILIAS for rotulo, distancia in DISTANCIAS.items()}
        bova = referencias(precos, CFG_BT)["BOVA11"].valores
        indice = next(iter(curvas.values())).valores.index
        if any(not curva.valores.index.equals(indice) for curva in curvas.values()) or not bova.index.equals(indice):
            raise SystemExit(f"Fase {fase}: curvas com datas diferentes")
        rf = market_data.serie_rf_diaria(indice, cdi_vigente, selic.valor)
        comparacoes = pd.DataFrame({familia: comparar(curvas[_nome(familia, "d")], curvas[_nome(familia, "d~")])
                                    for familia in FAMILIAS}).T
        for familia in FAMILIAS:
            por_familia[familia].append(comparacoes.loc[familia].to_dict())
        tabela = tabela_do_periodo({nome: curva.valores for nome, curva in curvas.items()}, bova, rf, n_tentativas)
        medidas = {nome: {"CAGR": cagr(serie), "MDD": drawdown(serie)[1]}
                   for nome, serie in {**{n: c.valores for n, c in curvas.items()}, "BOVA11": bova}.items()}
        print(f"\n=== Fase {fase}: 1ª decisão {indice[0]:%d/%m/%Y}, fim {indice[-1]:%d/%m/%Y} ===")
        print(_tabela(comparacoes, ["MDD d~", "MDD d", "Giro d~", "Giro d"]))
        print(_tabela(pd.DataFrame(medidas).T.join(tabela), ["CAGR", "MDD"]))
        print(f"(fase {fase} pronta em {time.perf_counter() - inicio_rodada:.0f} s)", flush=True)

    print()
    for familia in FAMILIAS:
        aprovada = d_til_passa(pd.DataFrame(por_familia[familia]))
        print(f"{familia} com d~: {'APROVADA (vira o padrão)' if aprovada else 'REPROVADA (fica a d)'}")
    print(f"(tempo total {time.perf_counter() - inicio_rodada:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
