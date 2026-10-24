"""
Classic value strategies against the Conservadora (user's request of 2026-10-24: "compare our Alepha strategy, the
Conservative Formula, with three consolidated strategies for the aggressive profile, and BOVA11, on our lake"):
Piotroski's F-score, Carlisle's Acquirer's Multiple and Greenblatt's Magic Formula. Research only: the Conservadora,
paper-v2 and the screen do not change whatever this shows.

Usage: .venv/Scripts/python -X utf8 scripts/avaliar_classicas.py > docs/classicas-2026-10-24.txt

Pre-registration (2026-10-24, written and committed before the run; the text does not change after that commit):
- Data: prices, point-in-time universe (rule D1), sectors, monthly total returns and dividends from
  `docs/dados/2026-10-22/` (the Conservadora's rerun, DECISOES 115); CVM statements and market values from
  `docs/dados/2026-10-24-cvm/` (the same lake code plus the DRE gross result, lake `0b0bfc6`). Each file is checked
  against its folder's SHA256SUMS. Fundamentals of month m = documents received before the 1st day of m
  (`core.valor`), flows over 12 months. Financeiro and issuers without a sector are out of the three classics.
- Piotroski (`core.classicas.piotroski_no_mes`): the F-score of Walkshäusl (2020, the paper in the user's
  `piotroski_strategy` folder, which applies it to all non-financial firms, as Hyde (2016) there does, not only the
  cheapest fifth as Piotroski (2000)): nine points, net income > 0, operating cash flow > 0, ROA up (net income over
  the year-earlier total assets, the year before over the total assets two years earlier), cash flow > net income,
  long-term debt / total assets down, current ratio up, no new shares (split-adjusted count of the market-value file
  at m against m-12, a rise up to 0.1% is rounding), gross margin up (DRE 3.03 / 3.01), asset turnover up (revenue
  over the year-earlier total assets). Years compare the issuer's latest period end with the same quarter one and
  two years earlier, each received before m; an issuer without any input is out (unknown, not weak); equity <= 0 is
  out. Ties in F are broken by book-to-market (equity / market value, percentile x 0.5). Consolidated net income
  (3.11) and total assets.
- Acquirer's Multiple (`core.classicas.multiplo_do_adquirente`): EV / EBIT, lowest first, with the Magic Formula's EV
  (market value + short- and long-term loans + minority interest - cash - short-term investments); EV <= 0 out, an
  operating loss ranks last. Carlisle's book uses operating earnings top-down (revenue - COGS - SG&A - D&A); the CVM's
  EBIT (3.05) is that line in the standard DRE. It is the Magic Formula without the return on capital: this tests
  Carlisle's claim that the quality half adds nothing.
- Magic Formula: `core.valor.formula_magica`, unchanged since S2.2 (DECISOES 79); rerun here only so all curves share
  the same data and dates. It counts no new trial (N already has it).
- Conservadora: `core.sinais.estrategia_conservadora`, unchanged (the registered strategy, DECISOES 82-83, 115).
- Same portfolio rule for all four (as S2.2): the 15 highest scores, equal weights, one stock per issuer, at most 4
  per sector, checked by `checar_carteira` (violation = refusal: the engine keeps the current portfolio); rebalance
  every 63 sessions; training window 1,260 sessions; 10 bps per side; a 5% daily loss freezes the next session.
  Deviations from the books, written here: Greenblatt buys 20-30 and holds a year, Carlisle 30 for a year, Piotroski
  a year from June; here 15 names re-chosen every quarter, the Conservadora's harness, so the comparison is about
  the selection rule. Before tax (the R$ 10 thousand after-tax proof was the Conservadora's, DECISOES 82).
- Dates: Piotroski needs two years of statements and the CVM's open data starts in 2010 (DFP) and 2011 (ITR): its
  first scores are in 06/2013 (coverage probe of 24/10, counts only: ~110 of ~150 universe stocks, 10-18 with
  F >= 8). So every curve starts with a first decision on the 1st session of 2014 (prices cut so the engine's 1st
  decision falls there, `corte_dos_precos`), phases 0, 21 and 42, to the last session of the price file.
- Gate (as S2.2): monthly Spearman IC of the score against the same month's total return, stocks of the month's
  universe with both. Approved = mean IC >= 0.02 and t >= 2 over 2014-01 to 2026-09; 2014-2017, 2018-2021 and
  2022-2026 descriptive. Candidate for a final proof = approved and P[> BOVA11] arithmetic >= 0.95 in all 3 phases.
  A candidate would only earn its own pre-registered proof; it would not replace the Conservadora.
- Per phase: CAGR, MDD, annual volatility, annual turnover, refusals, failures, Sharpe over the CDI, IR against
  BOVA11, P[> BOVA11] (arithmetic, log), DSR of the IR (N = rows of `docs/tentativas.csv`, now with Piotroski and the
  Acquirer's Multiple), MinTRL, and P[> Conservadora] of the three classics. Phase 0 also gets each calendar year's
  return of the four strategies, BOVA11 and IVVB11 (descriptive).
- Expected (written now): IC 2014-2026 of the Acquirer's Multiple about the Magic Formula's 0.06 (0.04-0.07),
  Piotroski lower (0.01-0.04, the gate may fail); CAGR of the three classics between BOVA11's and the Conservadora's,
  MDD worse than the Conservadora's; P[> Conservadora] below 0.5 for all three; no candidate.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, ResultadoBacktest, walk_forward_dinamico  # noqa: E402
from core.classicas import multiplo_do_adquirente_no_mes, piotroski_no_mes  # noqa: E402
from core.lago import ler_setores  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.periodos import retornos_por_ano  # noqa: E402
from core.poder import Portao, ic_por_mes  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from core.valor import Balancos, NotaDoMes, estrategia_da_nota, formula_magica_no_mes, painel_da_nota  # noqa: E402
from core.volatilidade import volatilidades  # noqa: E402
from scripts import avaliar_robustez  # noqa: E402
from scripts.avaliar_conservadora import (CONSERVADORA, Mensal, carregar_mensal, resumo_do_ic,  # noqa: E402
                                          retornos_no_universo)
from scripts.avaliar_robustez import (FASES, Dados, _acoes, _tabela, conferir_hashes, contar_tentativas,  # noqa: E402
                                      referencias, tabela_do_periodo)
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT, _porcento  # noqa: E402
from scripts.avaliar_valor import FORMULA, corte_dos_precos  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = RAIZ / "docs" / "dados" / "2026-10-22"
PASTA_CVM = RAIZ / "docs" / "dados" / "2026-10-24-cvm"
PIOTROSKI = "Piotroski F-score (15 nomes, pesos iguais, com regras)"
ADQUIRENTE = "Acquirer's Multiple (15 nomes, pesos iguais, com regras)"
NOVAS = (PIOTROSKI, ADQUIRENTE)
CLASSICAS = (PIOTROSKI, ADQUIRENTE, FORMULA)
ESTRATEGIAS = (*CLASSICAS, CONSERVADORA)
PORTAO = Portao("IC ≥ 0,02 e t ≥ 2", ic_minimo=0.02, t_minimo=2.0)
PERIODO_DO_IC = (pd.Timestamp("2014-01-01"), pd.Timestamp("2026-09-01"))
SUBPERIODOS_DO_IC = {"2014–2017": ("2014-01-01", "2017-12-01"), "2018–2021": ("2018-01-01", "2021-12-01"),
                     "2022–2026": ("2022-01-01", "2026-09-01")}
PRIMEIRA_DECISAO = pd.Timestamp("2014-01-01")
P_MINIMO_DA_PROVA = 0.95
_TOLERANCIA = 1e-9


@dataclass(frozen=True)
class Decisao:
    aprovada: bool          # gate of the IC over 2014-01 to 2026-09
    candidata: bool         # for a final proof: approved and P[> BOVA11] >= 0.95 in the 3 phases


def decidir(ics: pd.Series, p_bova_por_fase: Sequence[float]) -> Decisao:
    """The pre-registered rules on each month's IC and each phase's arithmetic P[> BOVA11]."""
    aprovada = PORTAO.aprova(ics.loc[PERIODO_DO_IC[0]:PERIODO_DO_IC[1]].dropna())
    candidata = aprovada and all(p >= P_MINIMO_DA_PROVA - _TOLERANCIA for p in p_bova_por_fase)
    return Decisao(aprovada, candidata)


# ---------------------------------------------------------------- run (CDI from the network)

def carregar() -> tuple[Dados, Mensal, Balancos, dict[str, str]]:
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = PASTA_DADOS
    dados, mensal = avaliar_robustez.carregar(), carregar_mensal(PASTA_DADOS)
    sha256 = conferir_hashes(PASTA_CVM, ["fundamentals.parquet", "market_caps.parquet"])
    fundamentos = pd.read_parquet(PASTA_CVM / "fundamentals.parquet")
    for coluna in ("dt_refer", "available_from"):
        fundamentos[coluna] = pd.to_datetime(fundamentos[coluna])
    valor_mercado = pd.read_parquet(PASTA_CVM / "market_caps.parquet",
                                    columns=["issuer", "month", "market_cap", "shares_today_units"])
    valor_mercado["month"] = pd.to_datetime(valor_mercado["month"])
    balancos = Balancos(fundamentos, valor_mercado, ler_setores(PASTA_DADOS / "sectors.parquet").to_dict())
    return dados, mensal, balancos, sha256


def notas(balancos: Balancos) -> dict[str, NotaDoMes]:
    return {PIOTROSKI: lambda mes, candidatos: piotroski_no_mes(balancos, mes, candidatos),
            ADQUIRENTE: lambda mes, candidatos: multiplo_do_adquirente_no_mes(balancos, mes, candidatos),
            FORMULA: lambda mes, candidatos: formula_magica_no_mes(balancos, mes, candidatos)}


def ics_e_cobertura(nota_do_mes: NotaDoMes, mensal: Mensal, universo: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    meses = pd.date_range(PERIODO_DO_IC[0], PERIODO_DO_IC[1], freq="MS")
    painel = painel_da_nota(nota_do_mes, universo, meses, mensal.retornos.columns)
    return ic_por_mes(painel, retornos_no_universo(mensal.retornos, universo, meses)), painel.notna().sum(axis=1)


def curvas_da_fase(dados: Dados, mensal: Mensal, notas_do_mes: Mapping[str, NotaDoMes], corte: int,
                   fase: int) -> dict[str, ResultadoBacktest]:
    """The three classics, the Conservadora and the references on the same dates."""
    precos = dados.precos.iloc[corte + fase:]
    estrategias = {nome: estrategia_da_nota(nota, dados.setores) for nome, nota in notas_do_mes.items()}
    estrategias[CONSERVADORA] = estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores)
    curvas = {nome: walk_forward_dinamico(_acoes(precos), dados.universo, estrategia, CFG_BT, nome, Congelamento())
              for nome, estrategia in estrategias.items()}
    curvas.update(referencias(precos, CFG_BT))
    if any(not curva.valores.index.equals(curvas[CONSERVADORA].valores.index) for curva in curvas.values()):
        raise SystemExit(f"Fase {fase}: curvas com datas diferentes")
    return curvas


def medidas_da_fase(curvas: Mapping[str, ResultadoBacktest], rf: pd.Series, n_tentativas: int) -> pd.DataFrame:
    tabela = tabela_do_periodo({nome: curvas[nome].valores for nome in ESTRATEGIAS}, curvas["BOVA11"].valores, rf,
                               n_tentativas)
    for nome in CLASSICAS:
        tabela.loc[nome, "P[> Conservadora]"] = psr_da_diferenca(curvas[nome].valores, curvas[CONSERVADORA].valores)
    linhas = {}
    for nome in (*ESTRATEGIAS, "IVVB11", "BOVA11"):
        valores = curvas[nome].valores.dropna()
        linhas[nome] = {"CAGR": cagr(valores), "MDD": drawdown(valores)[1], "Vol anual": volatilidades(valores).anual,
                        "Giro anual": curvas[nome].turnover_anual_medio, "Recusas": len(curvas[nome].recusas),
                        "Falhas": len(curvas[nome].falhas)}
    return pd.DataFrame(linhas).T.astype({"Recusas": int, "Falhas": int}).join(tabela)


def main() -> int:
    inicio_rodada = time.perf_counter()
    hoje = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    dados, mensal, balancos, sha256_cvm = carregar()
    selic = market_data.obter_selic_meta(hoje)
    try:
        cdi = market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=hoje)
    except market_data.ErroDadosMercado as erro:
        print(f"CDI: {erro} (Selic constante)")
        cdi = pd.Series(dtype=float)
    cdi_vigente = cdi if not cdi.empty else None
    n_tentativas = contar_tentativas()
    corte = corte_dos_precos(dados.precos.index, PRIMEIRA_DECISAO, CFG_BT.janela_treino)
    print(f"Arquivos (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm}")
    print(f"Manifest: {dados.manifest}; Selic meta {_porcento(selic.valor)} ({selic.fonte}); "
          f"CDI {'do BCB' if cdi_vigente is not None else 'indisponível'}; N do DSR = {n_tentativas}; "
          f"preços desde {dados.precos.index[corte]:%d/%m/%Y}")

    notas_do_mes = notas(balancos)
    ics = {}
    periodos = {"2014–2026 (portão)": PERIODO_DO_IC,
                **{rotulo: tuple(map(pd.Timestamp, datas)) for rotulo, datas in SUBPERIODOS_DO_IC.items()}}
    for nome, nota_do_mes in notas_do_mes.items():
        ics[nome], cobertura = ics_e_cobertura(nota_do_mes, mensal, dados.universo)
        print(f"\n{nome}: IC de Spearman mensal (nota do mês x retorno total do mês, universo do mês); papéis com "
              f"nota por mês: mínimo {cobertura.min()}, mediana {cobertura.median():.0f}, máximo {cobertura.max()}")
        print(pd.DataFrame({rotulo: resumo_do_ic(ics[nome].loc[inicio:fim]) for rotulo, (inicio, fim)
                            in periodos.items()}).T.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"(fase de IC pronta em {time.perf_counter() - inicio_rodada:.0f} s)", flush=True)

    p_bova = {nome: [] for nome in CLASSICAS}
    for fase in FASES:
        curvas = curvas_da_fase(dados, mensal, notas_do_mes, corte, fase)
        rf = market_data.serie_rf_diaria(curvas[CONSERVADORA].valores.index, cdi_vigente, selic.valor)
        tabela = medidas_da_fase(curvas, rf, n_tentativas)
        for nome in CLASSICAS:
            p_bova[nome].append(float(tabela.loc[nome, "P aritmético"]))
        primeira = curvas[CONSERVADORA]
        print(f"\n=== Fase {fase}: 1ª decisão {primeira.pesos_alvo.index[0]:%d/%m/%Y}, fim "
              f"{primeira.valores.index[-1]:%d/%m/%Y}, {len(primeira.pesos_alvo)} rebalanceios ===")
        print(_tabela(tabela, ["CAGR", "MDD", "Vol anual", "Giro anual"]))
        if fase == FASES[0]:
            anos = retornos_por_ano(pd.DataFrame({nome: curvas[nome].valores
                                                  for nome in (*ESTRATEGIAS, "BOVA11", "IVVB11")}))
            anos.columns = [nome.split(" (")[0] for nome in anos.columns]
            print("\nRetorno de cada ano (fase 0; o 1º ano começa na 1ª decisão, o último vai até o fim do arquivo)")
            print(anos.map(lambda v: "—" if pd.isna(v) else _porcento(float(v))).to_string())
        print(f"(fase {fase} pronta em {time.perf_counter() - inicio_rodada:.0f} s)", flush=True)

    for nome in CLASSICAS:
        decisao = decidir(ics[nome], p_bova[nome])
        print(f"\n{nome}: portão ({PORTAO.nome}, 2014-01 a 2026-09) "
              f"{'cumprido' if decisao.aprovada else 'não cumprido'}; "
              f"{'APROVADA' if decisao.aprovada else 'REPROVADA'}; candidata a uma prova final (P[> BOVA11] >= 0,95 "
              f"nas 3 fases): {'sim' if decisao.candidata else 'não'}")
    print(f"(tempo total {time.perf_counter() - inicio_rodada:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
