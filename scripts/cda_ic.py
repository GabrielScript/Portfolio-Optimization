"""
Stage A of the CDA best-ideas pre-registration (`docs/superpowers/specs/2026-10-29-cda-best-ideas-design.md`, `132a4c1`).
Run once, after `scripts/cda_painel.py` froze the panel:

    python -X utf8 scripts/cda_ic.py > docs/cda-ic-2026-10-29.txt

Monthly Spearman IC of the best-ideas score (share of qualifying equity funds naming the issuer among their three
largest active weights) against the month's total return in the month's universe, 2011-01 to 2026-09, months with at
least 30 qualifying funds. Gate: IC >= 0.02 and t >= 2. No network.
"""
from __future__ import annotations

import sys
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.poder import ic_por_mes  # noqa: E402
from core.risco import emissor  # noqa: E402
from core.sinais import conservadora  # noqa: E402
from core.valor import NotaDoMes, painel_da_nota  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_conservadora import resumo_do_ic, retornos_no_universo  # noqa: E402
from scripts.avaliar_robustez import conferir_hashes  # noqa: E402
from scripts.avaliar_valor import PORTAO  # noqa: E402
from scripts.cda_painel import PAINEL, SAIDA  # noqa: E402
from scripts.qpo_montar import FOLDER  # noqa: E402

PERIODO = (pd.Timestamp("2011-01-01"), pd.Timestamp("2026-09-01"))
FUNDOS_MINIMOS = 30
SUBPERIODOS = {"2011-2015": ("2011-01-01", "2015-12-01"), "2016-2020": ("2016-01-01", "2020-12-01"),
               "2021-2026": ("2021-01-01", "2026-09-01")}


def meses_lidos(painel: pd.DataFrame) -> list[pd.Timestamp]:
    """The months of the period with at least ``FUNDOS_MINIMOS`` qualifying funds."""
    fundos = painel.groupby("month")["qualifying"].first()
    fundos = fundos.loc[PERIODO[0]:PERIODO[1]]
    return list(fundos.index[fundos >= FUNDOS_MINIMOS])


def nota_do_painel(painel: pd.DataFrame, coluna: str) -> NotaDoMes:
    """The panel's ``coluna`` of month m on each candidate security, through its issuer."""
    por_mes = {mes: grupo.set_index("issuer")[coluna] for mes, grupo in painel.groupby("month")}

    def _nota(mes: pd.Timestamp, candidatos: Iterable[str]) -> pd.Series:
        valores = por_mes.get(mes, pd.Series(dtype=float))
        papeis = [papel for papel in candidatos if emissor(papel) in valores.index]
        return pd.Series([valores[emissor(papel)] for papel in papeis], index=papeis, dtype=float)
    return _nota


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    sha256_painel = conferir_hashes(SAIDA, [PAINEL.name])
    painel = pd.read_parquet(PAINEL)
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_painel}")

    meses = pd.DatetimeIndex(meses_lidos(painel))
    todos = pd.date_range(*PERIODO, freq="MS")
    print(f"Months read: {len(meses)} of {len(todos)} (dropped with < {FUNDOS_MINIMOS} qualifying funds: "
          f"{len(todos) - len(meses)})")
    colunas = mensal.retornos.columns
    retornos = retornos_no_universo(mensal.retornos, dados.universo, meses)
    notas = {"Best ideas (score)": nota_do_painel(painel, "score"),
             "Consensus active weight (reported)": nota_do_painel(painel, "consensus")}
    paineis = {nome: painel_da_nota(nota, dados.universo, meses, colunas) for nome, nota in notas.items()}
    periodos = {"2011-01 to 2026-09 (gate)": PERIODO,
                **{rotulo: tuple(map(pd.Timestamp, datas)) for rotulo, datas in SUBPERIODOS.items()}}
    ics = {}
    for nome, quadro in paineis.items():
        ics[nome] = ic_por_mes(quadro, retornos)
        print(f"\n{nome}: monthly Spearman IC, score of the month x total return of the month, universe of the month")
        print(pd.DataFrame({rotulo: resumo_do_ic(ics[nome].loc[de:ate]) for rotulo, (de, ate) in periodos.items()}).T
              .to_string(float_format=lambda x: f"{x:.3f}"))

    lidos = painel[painel["month"].isin(meses)]
    por_mes = lidos.groupby("month").agg(fundos=("qualifying", "first"), nomeados=("score", lambda s: (s > 0).sum()))
    print(f"\nQualifying funds per month: median {por_mes['fundos'].median():.0f} (min {por_mes['fundos'].min()}, "
          f"max {por_mes['fundos'].max()}); issuers with score > 0: median {por_mes['nomeados'].median():.0f}")
    conservadora_painel = painel_da_nota(
        lambda mes, c: conservadora(mensal.retornos, mensal.proventos, mes, c), dados.universo, meses, colunas)
    correlacao = ic_por_mes(paineis["Best ideas (score)"], conservadora_painel).dropna()
    print(f"Rank correlation with the Conservadora score: mean {correlacao.mean():.3f} over {len(correlacao)} months")

    aprovada = PORTAO.aprova(ics["Best ideas (score)"].loc[PERIODO[0]:PERIODO[1]].dropna())
    print(f"\nGate ({PORTAO.nome}, 2011-01 to 2026-09): {'PASSED' if aprovada else 'NOT passed'}. "
          + ("Next: stage B (attempt 52) after its own commit adding the trial row." if aprovada else
             "Stop: no attempt row; the best-ideas score is display only at most."))
    print(f"(total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
