"""
The Composite with a volatility target, replayed with real orders, costs and income tax.

    ../Portfolio-Optimization/.venv/Scripts/python -X utf8 scripts/vol_target_motor_avaliar.py > docs/vol-target-motor-2026-11-11.txt

Pre-registered in ``docs/superpowers/specs/2026-11-11-vol-target-motor-design.md`` (committed before any number):
attempt 51's Composite on the frozen exports of ``docs/dados/2026-10-22`` (to 16/10/2026), phases 0, 21 and 42; the
share in stocks min(1, target / 21-session realized volatility) read at each close; ``core.execucao.refazer`` with the
``ExposureOverlay`` (band 0.10, house freeze rule, cash at the CDI net of 22.5%), B3 fees, liquidity cost and personal
income tax, at R$ 10 thousand, 100 thousand and 1 million. The gate is ``aprovado``. Three anchors stop the script.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import vol_target  # noqa: E402
from core.backtest import Congelamento, ResultadoBacktest, retornos_da_carteira, walk_forward_dinamico  # noqa: E402
from core.cdi_lago import daily_cdi  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.config import DIAS_UTEIS_ANO  # noqa: E402
from core.execucao import ExposureOverlay, Refeita, refazer  # noqa: E402
from core.impostos import ISENCAO_DE_VENDAS  # noqa: E402
from core.metrics import cagr, drawdown, sharpe  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_execucao import (CUSTOS_DA_B3, conferir_ancora, liquidez_da_tabela,  # noqa: E402
                                      proventos_diarios)
from scripts.avaliar_robustez import FASES, _acoes, conferir_hashes, contar_tentativas  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import _value_score  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
MACRO = FOLDER.parent / "2026-10-21-macro"
LIQUIDEZ = FOLDER.parent / "2026-11-11-liquidez"
CURVAS_51 = FOLDER.parent / "composto-curvas-2026-10-28.parquet"
CAPITAIS = (10_000.0, 100_000.0, 1_000_000.0)
CAPITAIS_DO_PORTAO = (10_000.0, 100_000.0)
JANELA = 21
IR_DO_CAIXA = 0.225                 # shortest fixed-income bracket
MDD_MINIMO = 0.10                   # drawdown at least 10 points shallower
CAGR_MAXIMO = 0.01                  # CAGR at most 1 point lower
TOLERANCIA = 1e-9


@dataclass(frozen=True)
class Braco:
    nome: str
    alvo: float
    banda: float
    caixa_rende: bool


PRINCIPAL = Braco("Composite + 20% vol target", 0.20, 0.10, True)
SENSIBILIDADE = (Braco("target 16%", 0.16, 0.10, True), Braco("target 24%", 0.24, 0.10, True),
                 Braco("band 0.05", 0.20, 0.05, True), Braco("band 0.20", 0.20, 0.20, True),
                 Braco("cash at 0%", 0.20, 0.10, False))


@dataclass(frozen=True)
class Leitura:
    cagr_arm: float
    mdd_arm: float
    cagr_plain: float
    mdd_plain: float


def parte_no_fechamento(valores: pd.Series, alvo: float, janela: int = JANELA) -> pd.Series:
    """Share in stocks decided at each close: min(1, ``alvo`` / realized volatility of the ``janela`` sessions ending at
    that close); 1 before a full window. ``vol_target.exposicao`` gives the same number one session later (its shift
    makes it the share *held* in a session); here the order is sent at the close."""
    seguinte = vol_target.exposicao(valores.pct_change().dropna(), alvo, janela).shift(-1)
    return seguinte.reindex(valores.index).fillna(1.0)


def aprovado(leituras: Mapping[tuple[float, int], Leitura]) -> bool:
    """The spec's gate on the exit values: at R$ 10 and 100 thousand, in every phase, the drawdown at least
    ``MDD_MINIMO`` shallower and the CAGR at most ``CAGR_MAXIMO`` lower than the plain Composite's."""
    celulas = [(c, f) for c in CAPITAIS_DO_PORTAO for f in FASES]
    faltam = [c for c in celulas if c not in leituras]
    if faltam:
        raise ValueError(f"gate cells missing: {faltam}")
    return all(leituras[c].mdd_arm >= leituras[c].mdd_plain + MDD_MINIMO - TOLERANCIA
               and leituras[c].cagr_arm >= leituras[c].cagr_plain - CAGR_MAXIMO - TOLERANCIA for c in celulas)


def _linha(nome: str, refeita: Refeita, plain: Refeita, cdi: pd.Series, parte: pd.Series | None,
           rebalanceios: pd.DatetimeIndex) -> dict[str, float | str]:
    saida = refeita.saida
    anos = (len(saida) - 1) / DIAS_UTEIS_ANO
    excesso = saida.pct_change().dropna() - cdi.reindex(saida.index[1:]).fillna(0.0)
    ordens = refeita.custos.index.difference(rebalanceios)
    return {"arm": nome, "CAGR": cagr(saida), "MDD": drawdown(saida)[1], "Sharpe/CDI": sharpe(excesso),
            "exit R$": float(saida.iloc[-1]), "cost/yr": float(refeita.custos.sum() / refeita.valores.mean() / anos),
            "IR paid R$": float(refeita.impostos.sum()), "IR at exit R$": float(refeita.valores.iloc[-1] - saida.iloc[-1]),
            "mean target share": float(parte.iloc[:-1].mean()) if parte is not None else 1.0,
            "exposure orders/yr": len(ordens) / anos,
            "P[> plain]": psr_da_diferenca(saida, plain.saida) if parte is not None else float("nan"),
            "P log": psr_da_diferenca(saida, plain.saida, log=True) if parte is not None else float("nan")}


def _texto(linhas: list[dict]) -> str:
    quadro = pd.DataFrame(linhas).set_index("arm")
    fmt = {"CAGR": "{:.2%}", "MDD": "{:.1%}", "Sharpe/CDI": "{:.2f}", "exit R$": "{:,.0f}", "cost/yr": "{:.2%}",
           "IR paid R$": "{:,.0f}", "IR at exit R$": "{:,.0f}", "mean target share": "{:.0%}",
           "exposure orders/yr": "{:.1f}", "P[> plain]": "{:.3f}", "P log": "{:.3f}"}
    return quadro.apply(lambda col: col.map(fmt[col.name].format)).to_string()


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, sha_cvm = _value_score(mensal, com_crescimento=True)
    sha = dados.sha256 | mensal.sha256 | sha_cvm | conferir_hashes(MACRO, ["cdi_daily.parquet"]) \
        | conferir_hashes(LIQUIDEZ, ["liquidez.parquet"])
    cdi = daily_cdi(ANOS_DE_CDI, LAST_SESSION.date(), path=MACRO / "cdi_daily.parquet")
    liquidez = liquidez_da_tabela(pd.read_parquet(LIQUIDEZ / "liquidez.parquet"))
    dividendos = pd.read_parquet(FOLDER / "dividends.parquet",
                                 columns=["security", "ex_date", "corporate_action", "yield"])
    publicadas = pd.read_parquet(CURVAS_51)
    print(f"Files (sha256): {sha}\nDSR N before this test: {contar_tentativas()}; costs {CUSTOS_DA_B3}; "
          f"cash at the CDI net of {IR_DO_CAIXA:.1%}; window {JANELA}", flush=True)

    leituras: dict[tuple[float, int], Leitura] = {}
    for fase in FASES:
        acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])
        composite: ResultadoBacktest = walk_forward_dinamico(acoes, dados.universo, estrategia_por_nota(
            nota, dados.setores), CFG_BT, "Composite", Congelamento())
        valores = composite.valores.dropna()
        referencia = publicadas.loc[publicadas["fase"] == fase].set_index("date")["Composto"]
        desvio = float((valores.reindex(referencia.index) / referencia - 1.0).abs().max())
        if not desvio <= TOLERANCIA or len(referencia) != len(valores):
            raise SystemExit(f"phase {fase}: anchor 1 failed, Composite differs from attempt 51 by {desvio:.2e}")
        retornos = retornos_da_carteira(acoes)
        conferir_ancora(composite, retornos, CFG_BT.custo_bps)                         # anchor 2
        proventos = proventos_diarios(dividendos, retornos.index, retornos.columns)
        datas = composite.valores.index
        juros = cdi.reindex(datas).ffill().fillna(0.0) * (1.0 - IR_DO_CAIXA)
        um = ExposureOverlay(pd.Series(1.0, index=datas), juros)
        ancora = refazer(composite, retornos, 10_000.0, CUSTOS_DA_B3, liquidez, proventos=proventos)
        com_um = refazer(composite, retornos, 10_000.0, CUSTOS_DA_B3, liquidez, proventos=proventos, overlay=um)
        desvio = float((com_um.saida / ancora.saida - 1.0).abs().max())
        if not desvio <= TOLERANCIA:
            raise SystemExit(f"phase {fase}: anchor 3 failed, share 1 differs from the plain replay by {desvio:.2e}")
        print(f"\n=== Phase {fase}: {datas[0]:%d/%m/%Y} to {datas[-1]:%d/%m/%Y}; three anchors checked "
              f"({time.perf_counter() - inicio:.0f} s) ===", flush=True)

        for capital in CAPITAIS:
            plain = refazer(composite, retornos, capital, CUSTOS_DA_B3, liquidez, isencao=ISENCAO_DE_VENDAS,
                            proventos=proventos)
            linhas = [_linha("Composite (plain)", plain, plain, cdi, None, composite.pesos_alvo.index)]
            for braco in (PRINCIPAL, *SENSIBILIDADE):
                parte = parte_no_fechamento(composite.valores, braco.alvo)
                overlay = ExposureOverlay(parte, juros if braco.caixa_rende else juros * 0.0, braco.banda)
                feita = refazer(composite, retornos, capital, CUSTOS_DA_B3, liquidez, isencao=ISENCAO_DE_VENDAS,
                                proventos=proventos, overlay=overlay)
                linhas.append(_linha(braco.nome, feita, plain, cdi, parte, composite.pesos_alvo.index))
                if braco is PRINCIPAL:
                    leituras[(capital, fase)] = Leitura(cagr(feita.saida), drawdown(feita.saida)[1],
                                                        cagr(plain.saida), drawdown(plain.saida)[1])
            print(f"\n--- R$ {capital:,.0f} (exit values; sensitivity arms decide nothing) ---".replace(",", "."))
            print(_texto(linhas), flush=True)
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    passa = aprovado(leituras)
    print("\nGate (R$ 10 and 100 thousand, every phase: MDD >= plain + 10 points and CAGR >= plain - 1 point):")
    for (capital, fase), l in sorted(leituras.items()):
        print(f"  R$ {capital:>9,.0f} phase {fase:2d}: MDD {l.mdd_arm:.1%} vs {l.mdd_plain:.1%}, "
              f"CAGR {l.cagr_arm:.2%} vs {l.cagr_plain:.2%}")
    print("APPROVED: candidate for a fictitious account; advisor review before any real order" if passa else
          "REJECTED: the volatility-target line is closed")
    print(f"(total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
