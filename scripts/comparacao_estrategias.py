"""
comparacao_estrategias.py - The strategies comparison of the Aleph Quant page (the user's item 15, 2026-11-05): every
portfolio method of the project in one descriptive run on the lake's live exports, same engine, dates, universe and
costs, so the screen compares like with like. Not a reading and not a trial: nothing here is pre-registered, no gate,
N does not change (DECISOES 257).

    .venv/Scripts/python -X utf8 scripts/comparacao_estrategias.py [LAKE_EXPORTS_FOLDER]     (default: data/b3lake)
    .venv/Scripts/python -X utf8 scripts/comparacao_estrategias.py --so-top5 [LAKE_EXPORTS_FOLDER]

``--so-top5`` runs only the Composite and the Composite top-5 (DECISOES 290) and adds the top-5 to the saved files, on
their dates, once the recomputed Composite matches the saved one (otherwise it stops: run the full comparison).

Engine: ``core.backtest.walk_forward_dinamico`` with attempt 51's ``CFG_BT`` (5-year training, a rebalance every 63
sessions, 10 bps a side, R$ 10 thousand), the point-in-time universe of the lake (delisted names included), phase 0.
- Composite and Conservadora: the registered rules (``scripts/curvas_ao_vivo.py``'s, the same code); the Composite
  top-5 (``core.top5``): the Composite's score, 5 names, 1 per sector (DECISOES 290).
- Mean-variance family on the whole eligible universe with the app's limits (``CFG_KELLY``: 20% per name, up to 10
  names, 40% vol cap): Markowitz (maximum Sharpe on sample means), Bayes-Stein (maximum Sharpe on Jorion-shrunk means),
  minimum variance, Kelly lambda = 2 and Michaud's resampled maximum Sharpe (B = 100 blocks of 21, seed 0).
- Risk-based on the whole eligible universe: risk parity (ERC, ``core.paridade_risco``), HRP and the tree HRP
  (``core.hierarchical.otimizar_hrp_arvore``), and 1/N.
- Quantum: the QUBO-universe arm of attempt 46 (simulated annealing picks 15, then maximum Sharpe; classical hardware).
- References: BOVA11 (Ibovespa), SMAL11 (small caps) and DIVO11 (dividends index) bought and held in the same engine,
  and the Tesouro IPCA+ index (``core.titulos_publicos``, Tesouro Transparente's open prices). The Selic comes from the
  app's ``docs/dados/juros-diarios.parquet``.

Writes docs/dados/comparacao-estrategias.parquet (date, one column per curve, CDI), docs/dados/comparacao-giro.csv
(annual turnover, mean names, refusals and failures per strategy) and docs/comparacao-estrategias.txt.
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core import titulos_publicos  # noqa: E402
from core.backtest import (Congelamento, ResultadoBacktest, estrategia_buy_and_hold_ativo, estrategia_equiponderada,  # noqa: E402
                           estrategia_hrp, estrategia_otimizada, estrategia_reamostrada, retornos_da_carteira,
                           walk_forward, walk_forward_dinamico)
from core.config import Objetivo  # noqa: E402
from core.estimation import covariancia_ledoit_wolf  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.paridade_risco import otimizar_paridade_de_risco  # noqa: E402
from scripts.curvas_ao_vivo import ARQUIVOS, LAGO, congelar  # noqa: E402

CURVAS = RAIZ / "docs" / "dados" / "comparacao-estrategias.parquet"
GIRO = RAIZ / "docs" / "dados" / "comparacao-giro.csv"
SAIDA = RAIZ / "docs" / "comparacao-estrategias.txt"
COMPOSITE, CONSERVADORA = "Composite", "Conservadora"
ETFS = {"BOVA11": "BOVA11 (Ibovespa)", "SMAL11": "SMAL11 (small caps)", "DIVO11": "DIVO11 (dividendos)"}
FASE = 0                              # the screen's phase (DECISOES 252)
PESO_MINIMO = 1e-6                    # a weight below it is not a holding
TOLERANCIA = 1e-6                     # relative gap that still counts as the same Composite curve


def juntar_curva(curvas: pd.DataFrame, giro: pd.DataFrame, nome: str, composite: pd.Series, nova: pd.Series,
                 giro_novo: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The saved ``curvas`` (date x curve) and ``giro`` (strategy x turnover) with ``nome`` = ``nova`` on their dates,
    right after the Composite (replacing an older ``nome``). ``composite`` is the Composite recomputed in the same run
    as ``nova``: it must match the saved one on those dates (``ValueError`` if not), so both curves stay like with
    like."""
    salva = curvas[COMPOSITE].dropna()
    refeita = composite.reindex(salva.index)
    gap = float(((refeita - salva).abs() / salva.abs()).max())
    if not gap <= TOLERANCIA:
        raise ValueError(f"the recomputed Composite differs from the saved one (max relative gap {gap:.2e}): the "
                         "inputs moved since the saved run; run the full comparison instead")
    juntas = curvas.drop(columns=[nome], errors="ignore")
    juntas.insert(juntas.columns.get_loc(COMPOSITE) + 1, nome, nova.reindex(juntas.index))
    linhas = giro.drop(index=[nome], errors="ignore")
    posicao = list(linhas.index).index(COMPOSITE) + 1
    giro_junto = pd.concat([linhas.iloc[:posicao], giro_novo.to_frame(nome).T.reindex(columns=linhas.columns),
                            linhas.iloc[posicao:]]).rename_axis(linhas.index.name)
    return juntas, giro_junto


def estrategia_paridade_de_risco() -> Callable[[pd.DataFrame], pd.Series]:
    return lambda treino: otimizar_paridade_de_risco(covariancia_ledoit_wolf(treino))


def otimizadores(cdi: pd.Series) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    """The universe-wide methods, in the screen's order (``CFG_KELLY``'s limits for the mean-variance family)."""
    from scripts.avaliar_universo_dinamico import CFG_KELLY
    sharpe = replace(CFG_KELLY, objetivo=Objetivo.MAX_SHARPE)
    return {
        "Markowitz (média-variância)": estrategia_otimizada(sharpe, False, cdi),
        "Bayes-Stein (Jorion)": estrategia_otimizada(sharpe, True, cdi),
        "Mínima variância": estrategia_otimizada(replace(CFG_KELLY, objetivo=Objetivo.MIN_VOL), True, cdi),
        "Kelly (λ = 2)": estrategia_otimizada(CFG_KELLY, True, cdi),
        "Reamostrada (Michaud)": estrategia_reamostrada(sharpe, True, cdi),
        "Paridade de risco (ERC)": estrategia_paridade_de_risco(),
        "HRP": estrategia_hrp(),
        "HRP em árvore (grafo)": estrategia_hrp(arvore=True),
        "1/N": estrategia_equiponderada(),
    }


def etfs_comprados_e_mantidos(precos: pd.DataFrame, cfg_bt) -> dict[str, pd.Series]:
    """Each ETF of ``ETFS`` bought and held in the engine; empty before its first quote (``avaliar_robustez``'s way)."""
    retornos = retornos_da_carteira(precos[list(ETFS)])
    curvas = {}
    for papel, nome in ETFS.items():
        valores = walk_forward(retornos, estrategia_buy_and_hold_ativo(papel), cfg_bt, nome).valores
        curvas[nome] = valores.where(valores.index >= precos[papel].first_valid_index())
    return curvas


def resumo_de_giro(resultados: Mapping[str, ResultadoBacktest]) -> pd.DataFrame:
    """Annual turnover, mean names held at the rebalances, refusals and failures of each strategy."""
    return pd.DataFrame({nome: {"giro_anual": r.turnover_anual_medio,
                                "nomes_medios": float((r.pesos_alvo > PESO_MINIMO).sum(axis=1).mean()),
                                "recusas": len(r.recusas), "falhas": len(r.falhas)}
                         for nome, r in resultados.items()}).T.rename_axis("estrategia")


def tesouro_nas_datas(datas: pd.DatetimeIndex) -> pd.Series:
    """The Tesouro IPCA+ index on the engine's sessions (its last published day carried, never past its end)."""
    indice = titulos_publicos.indice_tesouro_ipca(titulos_publicos.ler_precos_e_taxas(titulos_publicos.baixar()))
    return indice.reindex(datas, method="ffill").where(datas <= indice.index[-1])


def main(argv: list[str]) -> int:
    from core import top5
    from core.cdi_lago import daily_cdi
    from core.composto import estrategia_por_nota
    from core.qpo import CFG_MAX_SHARPE, K, qubo_selector, selection_then_max_sharpe
    from core.qubo_universo import FULL_UNIVERSE, SECTOR_CAP, anneal_solver
    from core.sinais import estrategia_conservadora
    from scripts import avaliar_conservadora, avaliar_robustez, qubo_universo_avaliar
    from scripts.avaliar_robustez import _acoes
    from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT

    inicio = time.perf_counter()
    so_top5 = "--so-top5" in argv
    argv = [argumento for argumento in argv if argumento != "--so-top5"]
    lago = Path(argv[0]) if argv else LAGO
    with tempfile.TemporaryDirectory(prefix="comparacao-") as temporaria:
        entradas = Path(temporaria)
        somas = congelar(lago, entradas, ARQUIVOS)
        avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = entradas
        qubo_universo_avaliar.FOLDER = qubo_universo_avaliar.PASTA_CVM = entradas
        dados = avaliar_robustez.carregar()
        mensal = avaliar_conservadora.carregar_mensal(entradas)
        nota, _ = qubo_universo_avaliar._value_score(mensal, com_crescimento=True)       # attempt 51's score
        ultima = dados.precos.index.max()
        cdi = daily_cdi(ANOS_DE_CDI, ultima.date(), path=entradas / "cdi_daily.parquet")
        precos = dados.precos.loc[:ultima].iloc[FASE:]
        acoes = _acoes(precos)

        def rodar(estrategia, nome: str) -> ResultadoBacktest:
            resultado = walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())
            print(f"{nome}: CAGR {cagr(resultado.valores.dropna()):.1%}, max drawdown "
                  f"{drawdown(resultado.valores.dropna())[1]:.1%}, turnover {resultado.turnover_anual_medio:.2f}/yr "
                  f"({time.perf_counter() - inicio:.0f} s)", flush=True)
            return resultado

        if so_top5:
            return _so_top5(rodar(estrategia_por_nota(nota, dados.setores), COMPOSITE),
                            rodar(top5.estrategia(nota, dados.setores), top5.NOME), top5.NOME, lago, somas,
                            ultima, inicio)
        quantica = selection_then_max_sharpe(
            qubo_selector(dados.setores, FULL_UNIVERSE, K, anneal_solver(dados.setores, SECTOR_CAP, {}), {}),
            mensal.retornos, mensal.proventos, dados.setores,
            replace(CFG_MAX_SHARPE, taxa_livre_risco=float((1 + cdi.iloc[-1]) ** 252 - 1)), cdi, None)
        with SAIDA.open("w", encoding="utf-8", newline="\n") as saida, contextlib.redirect_stdout(saida):
            print(f"DESCRIPTIVE strategies comparison to {ultima:%d/%m/%Y} (not a reading, not a trial). Lake exports "
                  f"from {lago.name} (sha256): {somas}")
            resultados = {
                COMPOSITE: rodar(estrategia_por_nota(nota, dados.setores), COMPOSITE),
                top5.NOME: rodar(top5.estrategia(nota, dados.setores), top5.NOME),
                CONSERVADORA: rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores),
                                    CONSERVADORA),
                **{nome: rodar(estrategia, nome) for nome, estrategia in otimizadores(cdi).items()},
                "Quântica (QUBO)": rodar(quantica, "Quântica (QUBO)"),
            }
            datas = resultados[COMPOSITE].valores.index
            curvas = pd.DataFrame({nome: r.valores.reindex(datas) for nome, r in resultados.items()})
            for nome, valores in etfs_comprados_e_mantidos(precos, CFG_BT).items():
                curvas[nome] = valores.reindex(datas)
            curvas[titulos_publicos.NOME] = tesouro_nas_datas(datas)
            curvas["CDI"] = cdi.reindex(datas)
            giro = resumo_de_giro(resultados)
            print(giro.to_string(float_format=lambda x: f"{x:.2f}"))
    curvas.rename_axis("date").reset_index().to_parquet(CURVAS, index=False)
    giro.to_csv(GIRO, encoding="utf-8", lineterminator="\n")
    print(f"{CURVAS.relative_to(RAIZ)} and {GIRO.relative_to(RAIZ)} up to {ultima:%d/%m/%Y} in "
          f"{time.perf_counter() - inicio:.0f} s; log in {SAIDA.relative_to(RAIZ)}")
    return 0


def _so_top5(composite: ResultadoBacktest, nova: ResultadoBacktest, nome: str, lago: Path, somas: dict,
             ultima: pd.Timestamp, inicio: float) -> int:
    """``--so-top5``: adds ``nova`` to the saved files (``juntar_curva``) and appends its lines to the log."""
    curvas = pd.read_parquet(CURVAS).set_index("date")
    giro = pd.read_csv(GIRO, index_col="estrategia")
    juntas, giro_junto = juntar_curva(curvas, giro, nome, composite.valores, nova.valores,
                                      resumo_de_giro({nome: nova}).loc[nome])
    juntas.rename_axis("date").reset_index().to_parquet(CURVAS, index=False)
    giro_junto.to_csv(GIRO, encoding="utf-8", lineterminator="\n")
    valores = juntas[nome].dropna()
    with SAIDA.open("a", encoding="utf-8", newline="\n") as saida:
        print(f"\n{nome} added by --so-top5 (DECISOES 290) on the saved dates to {valores.index[-1]:%d/%m/%Y}; the "
              f"Composite recomputed to {ultima:%d/%m/%Y} from {lago.name} (sha256 {somas}) matched the saved one. "
              f"{nome}: CAGR {cagr(valores):.1%}, max drawdown {drawdown(valores)[1]:.1%}, turnover "
              f"{nova.turnover_anual_medio:.2f}/yr.", file=saida)
    print(f"{nome} added to {CURVAS.relative_to(RAIZ)} and {GIRO.relative_to(RAIZ)} in "
          f"{time.perf_counter() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
