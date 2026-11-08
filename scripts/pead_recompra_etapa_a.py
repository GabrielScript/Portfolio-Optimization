"""
pead_recompra_etapa_a.py - Stage A of the SUE and buyback test, read once (docs/superpowers/specs/
2026-11-08-pead-recompra-design.md, sections 4 and 5; pre-registered in `e6e5f44`, signals and frozen rows in `05e60fb`).
No network: prices, universe, statements and IPE rows frozen.

    python -X utf8 scripts/pead_recompra_etapa_a.py > docs/pead-recompra-etapa-a-2026-11-08.txt

SUE: A1 = monthly Spearman IC of the SUE score, A2 = the same of its residual on the Composite score, 2012-01 to
2026-09, gate IC >= 0.02 and t >= 2 each (`avaliar_valor.PORTAO`). Buybacks: 63-session abnormal return of each event
against the month's universe, monthly portfolio, gate mean >= +1.0% and Newey-West t >= 2.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.crescimento_lucro import nota_de_crescimento  # noqa: E402
from core.pead import nota_sue  # noqa: E402
from core.poder import ic_por_mes  # noqa: E402
from core.qml_ic import t_newey_west  # noqa: E402
from core.recompra import carteira_mensal, eventos, retornos_anormais  # noqa: E402
from core.valor import NotaDoMes, painel_da_nota  # noqa: E402
from core.views import escore_normal  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor  # noqa: E402
from scripts.avaliar_conservadora import resumo_do_ic  # noqa: E402
from scripts.avaliar_robustez import conferir_hashes  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import _value_score  # noqa: E402

PASTA_CVM = FOLDER.parent / "2026-10-24-cvm"
PASTA_IPE = FOLDER.parent / "2026-11-08-ipe"
PRIMEIRA_ENTRADA = pd.Timestamp("2011-01-03")
MEDIA_MINIMA = 0.01
T_MINIMO = 2.0
SUBPERIODOS = {"2011-2016": ("2011-01-01", "2016-12-31"), "2017-2021": ("2017-01-01", "2021-12-31"),
               "2022-2026": ("2022-01-01", "2026-12-31")}


def residuo_no_composto(sinal: NotaDoMes, composto: NotaDoMes) -> NotaDoMes:
    """Each month, the signal's normal score minus its cross-sectional OLS fit on the Composite's normal score."""
    def _nota(mes: pd.Timestamp, candidatos) -> pd.Series:
        base = composto(mes, candidatos).dropna()
        s = escore_normal(sinal(mes, list(base.index)).dropna())
        if len(s) < 3:
            return pd.Series(dtype=float)
        c = escore_normal(base)[s.index]
        inclinacao, intercepto = np.polyfit(c.to_numpy(), s.to_numpy(), 1)
        return s - (intercepto + inclinacao * c)
    return _nota


def tabela_do_ic(ics: pd.Series) -> pd.DataFrame:
    periodos = {"2012-2026 (gate)": avaliar_valor.PERIODO_DO_IC,
                **{rotulo: tuple(map(pd.Timestamp, datas)) for rotulo, datas in avaliar_valor.SUBPERIODOS_DO_IC.items()}}
    return pd.DataFrame({rotulo: resumo_do_ic(ics.loc[de:ate]) for rotulo, (de, ate) in periodos.items()}).T


def etapa_sue(dados, mensal, fundamentos, composto: NotaDoMes) -> tuple[bool, bool]:
    sue = nota_sue(fundamentos)
    aprovacoes = []
    for rotulo, nota in (("A1: SUE alone", sue), ("A2: SUE beyond the Composite (residual)",
                                                   residuo_no_composto(sue, composto))):
        ics, cobertura = avaliar_valor.ics_e_cobertura(nota, mensal, dados.universo)
        aprovada = avaliar_valor.PORTAO.aprova(ics.loc[avaliar_valor.PERIODO_DO_IC[0]:avaliar_valor.PERIODO_DO_IC[1]])
        aprovacoes.append(aprovada)
        print(f"\n{rotulo}: monthly Spearman IC; names scored per month min {cobertura.min()}, median "
              f"{cobertura.median():.0f}, max {cobertura.max()}")
        print(tabela_do_ic(ics).to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"Gate ({avaliar_valor.PORTAO.nome}): {'PASSED' if aprovada else 'NOT passed'}")
    meses = pd.date_range(*avaliar_valor.PERIODO_DO_IC, freq="MS")
    colunas = mensal.retornos.columns
    painel = painel_da_nota(sue, dados.universo, meses, colunas)
    for nome, outra in (("Composite", composto), ("profit growth", nota_de_crescimento(fundamentos))):
        correlacoes = ic_por_mes(painel, painel_da_nota(outra, dados.universo, meses, colunas)).dropna()
        print(f"Rank correlation of SUE with the {nome} score: mean {correlacoes.mean():.3f} over "
              f"{len(correlacoes)} months")
    return aprovacoes[0], aprovacoes[1]


def etapa_recompra(dados) -> bool:
    sha = conferir_hashes(PASTA_IPE, ["recompras.parquet"])
    linhas = pd.read_parquet(PASTA_IPE / "recompras.parquet")
    print(f"\nIPE buyback rows (sha256 {sha['recompras.parquet'][:12]}): {len(linhas):,} row x issuer pairs, "
          f"{int(linhas['issuer'].isna().sum())} without an issuer (out)")
    linhas = linhas.dropna(subset=["issuer"])
    precos = dados.precos.loc[:LAST_SESSION]
    sessoes = precos.index
    documentos = linhas.groupby(["issuer", "Data_Referencia"])["Data_Entrega"].min()
    atrasados = int((documentos > documentos.index.get_level_values(1) + pd.Timedelta(days=30)).sum())
    todos = eventos(linhas, sessoes)
    todos = todos[todos["entrada"] >= PRIMEIRA_ENTRADA]
    anormais = retornos_anormais(todos, precos, dados.universo)
    curtos = retornos_anormais(todos, precos, dados.universo, janela=21)
    print(f"Documents (issuer, reference date) {len(documentos):,}; delivered > 30 days late {atrasados}; events "
          f"kept from {PRIMEIRA_ENTRADA:%d/%m/%Y} {len(todos):,}; with a class in the month's universe and a full "
          f"63-session window {len(anormais):,}")
    mensal_63 = carteira_mensal(anormais)
    media, t = float(mensal_63.mean()), t_newey_west(mensal_63)
    aprovada = media >= MEDIA_MINIMA and t >= T_MINIMO
    print(f"63 sessions, monthly portfolio: months {len(mensal_63)}, mean {media:+.2%}, Newey-West t {t:+.2f}; "
          f"events: mean {anormais['anormal'].mean():+.2%}, median {anormais['anormal'].median():+.2%}, "
          f"positive {(anormais['anormal'] > 0).mean():.0%}")
    mensal_21 = carteira_mensal(curtos)
    print(f"21 sessions (reported): months {len(mensal_21)}, mean {mensal_21.mean():+.2%}, Newey-West t "
          f"{t_newey_west(mensal_21):+.2f}")
    for rotulo, (de, ate) in SUBPERIODOS.items():
        parte = mensal_63.loc[de:ate]
        print(f"  {rotulo}: months {len(parte)}, mean {parte.mean():+.2%}, t {t_newey_west(parte):+.2f}")
    print(f"Gate (mean >= +1.0% and t >= 2): {'PASSED' if aprovada else 'NOT passed'}")
    return aprovada


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = avaliar_valor.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    composto, sha256_cvm = _value_score(mensal, com_crescimento=True)
    balancos, _ = avaliar_valor.carregar_balancos(PASTA_CVM)
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm}")
    a1, a2 = etapa_sue(dados, mensal, balancos.fundamentos, composto)
    recompra = etapa_recompra(dados)
    print(f"\nStage A: SUE A1 {'passed' if a1 else 'failed'}, A2 {'passed' if a2 else 'failed'} -> Composto+SUE-15 "
          f"{'RUNS' if a1 and a2 else 'is not run (line closed)'}; buybacks {'passed' if recompra else 'failed'} -> "
          f"Composto+Recompra-15 {'RUNS' if recompra else 'is not run (line closed)'}")
    print(f"(total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    raise SystemExit(main())
