"""
No-trade zone (S4, PLAN "Path to v1.0" step 2b): the Conservadora of the final proof refeita with and without the zone
of `core.execucao.ZonaNoMotor`, against BOVA11 after IR. It changes which orders the real-money plan sends, so it is a
new configuration: pre-registered here, one line in `docs/tentativas.csv`. Network only for the CDI and the Selic.

Usage: .venv/Scripts/python -X utf8 scripts/avaliar_zona.py > docs/zona-2026-10-23.txt

Pre-registration (2026-10-23, written and committed before running; after that commit this text does not change):
- Engine curves and data: exactly the final proof's (`scripts/avaliar_execucao.py`, DECISOES 82): the files and
  hashes of `docs/dados/2026-10-19/` and `2026-10-20/`, the liquidity of `docs/dados/2026-10-20-liquidez/`, phases 0,
  21 and 42 from the cut that puts the 1st decision on 02/01/2012, the same costs (emolumentos 0.030%, half spread
  and the square-root impact with Y = 1), the same IR and dividends in the base. The run date is the final proof's
  (2026-10-20, the same Selic target). The anchor of the final proof is checked again. The Conservadora without the
  zone must reproduce `docs/execucao-2026-10-20.txt` to the digit (checked by hand after the run, written in
  DECISOES).
- The zone (`ZonaNoMotor`, the same rule as the real-money plan's `ZonaSemNegociacao`): at each rebalance, a ticker
  held and kept in the target skips its order when the order's cost plus the sale's marginal IR in the month is above
  its expected gain, (lambda / 2) x vol_diaria^2 x 63 x (order / patrimony)^2 x patrimony, lambda = 2 (the study's),
  vol_diaria = the liquidity file's daily vol of the month before the order; unless skipping leaves it above 20% of
  the patrimony. Entries and exits always trade. Skipped positions keep their value; the others go to their own
  target value; the rest stays in cash at 0% until the next rebalance (conservative: in the real plan it would earn
  the CDI); when the skipped ones grew, the others shrink in proportion. Simplification written here: the sector
  ceiling is not checked in the engine (the real plan forces an order that would break it).
- Measures, per phase and capital (R$ 10 thousand decides; R$ 100 thousand and R$ 1 million are descriptive): the
  final proof's `medidas` (P[> BOVA11] arithmetic and log of the exit values, CAGR and MDD of the exit, final exit,
  cost per year, IR paid and IR at exit) for the Conservadora without and with the zone, and P[with > without] of
  the exit values (descriptive).
- Decision fixed now: the zone ships in the real-money plan (orders skipped by default) if, with R$ 10 thousand, the
  exit value of the Conservadora with the zone has P[> BOVA11] arithmetic >= 0.95 in the 3 phases and an exit CAGR
  not lower than without the zone in each of the 3 phases. Otherwise it ships as information only: the plan shows
  each order's cost and expected gain and sends every order. Either way paper-v2 is unchanged (it measures the
  registered full rebalance).
- Expectation registered: with costs of ~0.1-0.25% per order and daily vols of 1.5-2.5%, the band is about 5-8 p.p.
  of weight, so almost every resize of a name kept from one quarter to the next is skipped and only entries and
  exits trade. Cost per year of the Conservadora with R$ 10 thousand from 0.59% to 0.35-0.50%; exit CAGR within
  -0.3 to +0.3 p.p. of the run without the zone (less cost against the cash at 0% and the drift toward the
  winners); P[> BOVA11] still 0.97-0.995; P[with > without] between 0.3 and 0.8. Decision: ships in 2 of 3 phases at
  least, so a coin flip on "every phase".
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import retornos_da_carteira  # noqa: E402
from core.config import taxa_anual_para_aritmetica  # noqa: E402
from core.execucao import ZonaNoMotor, refazer  # noqa: E402
from core.impostos import ISENCAO_DE_VENDAS  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from scripts.avaliar_conservadora import CONSERVADORA, PASTA_MENSAL, carregar_mensal  # noqa: E402
from scripts.avaliar_execucao import (CAPITAIS, CAPITAL_DA_PROVA, CUSTOS_DA_B3, PASTA_LIQUIDEZ, _formatar,  # noqa: E402
                                      conferir_ancora, liquidez_da_tabela, medidas, proventos_diarios)
from scripts.avaliar_robustez import FASES, carregar, conferir_hashes, contar_tentativas  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT, CFG_KELLY, _porcento  # noqa: E402
from scripts.avaliar_valor import PRIMEIRA_DECISAO, corte_dos_precos, curvas_da_fase  # noqa: E402

DATA_DA_PROVA = "2026-10-20"
SEM_ZONA, COM_ZONA = "Conservadora sem a zona", "Conservadora com a zona"
ZONA = ZonaNoMotor()
P_MINIMO = 0.95
_TOLERANCIA = 1e-9


def decidir_zona(p_com_zona: Sequence[float], cagr_com_zona: Sequence[float], cagr_sem_zona: Sequence[float]) -> bool:
    """The fixed rule: P[> BOVA11] >= 0.95 with the zone and an exit CAGR not lower than without it, in every phase."""
    return (all(p >= P_MINIMO - _TOLERANCIA for p in p_com_zona)
            and all(com >= sem - _TOLERANCIA for com, sem in zip(cagr_com_zona, cagr_sem_zona, strict=True)))


def main() -> int:
    inicio_rodada = time.perf_counter()
    hoje = date.fromisoformat(DATA_DA_PROVA)
    dados, mensal = carregar(), carregar_mensal()
    sha256 = dados.sha256 | mensal.sha256 | conferir_hashes(PASTA_LIQUIDEZ, ["liquidez.parquet"])
    liquidez = liquidez_da_tabela(pd.read_parquet(PASTA_LIQUIDEZ / "liquidez.parquet"))
    dividendos = pd.read_parquet(PASTA_MENSAL / "dividends.parquet",
                                 columns=["security", "ex_date", "corporate_action", "yield"])
    selic = market_data.obter_selic_meta(hoje)
    try:
        cdi = market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=hoje)
    except market_data.ErroDadosMercado as erro:
        print(f"CDI: {erro} (Selic constante)")
        cdi = pd.Series(dtype=float)
    cdi_vigente = cdi if not cdi.empty else None
    cfg = replace(CFG_KELLY, taxa_livre_risco=taxa_anual_para_aritmetica(selic.valor))
    corte = corte_dos_precos(dados.precos.index, PRIMEIRA_DECISAO, CFG_BT.janela_treino)
    print(f"Arquivos (sha256): {sha256}")
    print(f"Manifest: {dados.manifest}; Selic meta {_porcento(selic.valor)} ({selic.fonte}); N = {contar_tentativas()} "
          f"(sem DSR); preços desde {dados.precos.index[corte]:%d/%m/%Y}; {CUSTOS_DA_B3}; {ZONA}")

    p_com, cagr_com, cagr_sem = [], [], []
    for fase in FASES:
        curvas = curvas_da_fase(dados, mensal, {}, corte, fase, cdi_vigente, cfg)
        retornos = retornos_da_carteira(dados.precos.iloc[corte + fase:])
        for nome in (CONSERVADORA, "BOVA11"):
            conferir_ancora(curvas[nome], retornos, CFG_BT.custo_bps)
        proventos = proventos_diarios(dividendos, retornos.index, retornos.columns)
        print(f"\n=== Fase {fase}: 1ª decisão {curvas[CONSERVADORA].pesos_alvo.index[0]:%d/%m/%Y}, fim "
              f"{curvas[CONSERVADORA].valores.index[-1]:%d/%m/%Y}; âncora conferida ===")
        for capital in CAPITAIS:
            refeitas = {nome: refazer(curvas[CONSERVADORA], retornos, capital, CUSTOS_DA_B3, liquidez,
                                      isencao=ISENCAO_DE_VENDAS, proventos=proventos, zona=zona)
                        for nome, zona in ((SEM_ZONA, None), (COM_ZONA, ZONA))}
            refeitas["BOVA11"] = refazer(curvas["BOVA11"], retornos, capital, CUSTOS_DA_B3, liquidez, isencao=0.0)
            tabela = medidas(refeitas)
            tabela.loc[COM_ZONA, "P[com > sem]"] = psr_da_diferenca(refeitas[COM_ZONA].saida, refeitas[SEM_ZONA].saida)
            if capital == CAPITAL_DA_PROVA:
                p_com.append(float(tabela.loc[COM_ZONA, "P aritmético"]))
                cagr_com.append(float(tabela.loc[COM_ZONA, "CAGR"]))
                cagr_sem.append(float(tabela.loc[SEM_ZONA, "CAGR"]))
            print(f"\n--- R$ {capital:,.0f} ---".replace(",", "."))
            print(_formatar(tabela))
        print(f"(fase {fase} pronta em {time.perf_counter() - inicio_rodada:.0f} s)", flush=True)

    vai = decidir_zona(p_com, cagr_com, cagr_sem)
    print("\nCom R$ 10.000, com a zona: P[> BOVA11] aritmético da saída " + ", ".join(f"{p:.3f}" for p in p_com)
          + "; CAGR da saída com / sem a zona "
          + ", ".join(f"{_porcento(c)} / {_porcento(s)}" for c, s in zip(cagr_com, cagr_sem, strict=True))
          + f"; zona {'NO PLANO (ordens puladas por padrão)' if vai else 'SÓ INFORMATIVA (o plano manda todas as ordens)'}")
    print(f"(tempo total {time.perf_counter() - inicio_rodada:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
