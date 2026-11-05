"""
Stage A of the options-veto pre-registration (`docs/superpowers/specs/2026-10-29-opcoes-veto-design.md`, section 4).
Run once, after `scripts/opcoes_painel.py` froze the panel:

    python -X utf8 scripts/opcoes_ic.py > docs/opcoes-ic-2026-11-05.txt

For each signal (S1 -iv_atm, S2 -skew, S3 -pc_ratio): the monthly Spearman IC against the month's total return in
the month's universe, 2008-01 to 2026-09, months with at least 20 issuers having the signal, and the partial IC given
the Conservadora score. Gate per signal, both: mean IC >= 0.02 with t >= 2.4, and mean partial IC >= 0.02 with
t >= 2.4. No network.
"""
from __future__ import annotations

import sys
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.opcoes import EMISSORES_MINIMOS, PORTAO, SINAIS, ic_parcial_por_mes, pontuar  # noqa: E402
from core.poder import ic_por_mes  # noqa: E402
from core.risco import emissor  # noqa: E402
from core.sinais import NOMES, POR_SETOR, carteira_da_nota, conservadora  # noqa: E402
from core.valor import NotaDoMes, painel_da_nota  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_conservadora import resumo_do_ic, retornos_no_universo  # noqa: E402
from scripts.avaliar_robustez import conferir_hashes  # noqa: E402
from scripts.opcoes_painel import PAINEL, PERIODO, SAIDA  # noqa: E402
from scripts.qpo_montar import FOLDER  # noqa: E402

SUBPERIODOS = {"2008-2015": ("2008-01-01", "2015-12-01"), "2016-2020": ("2016-01-01", "2020-12-01"),
               "2021-2026": ("2021-01-01", "2026-09-01")}


def nota_do_painel(pontos: pd.DataFrame, coluna: str) -> NotaDoMes:
    """The signal of month m on each candidate security, through its issuer (PETR3 and PETR4 take PETR's)."""
    por_mes = {mes: grupo.set_index("issuer")[coluna].dropna() for mes, grupo in pontos.groupby("month")}

    def _nota(mes: pd.Timestamp, candidatos: Iterable[str]) -> pd.Series:
        valores = por_mes.get(mes, pd.Series(dtype=float))
        papeis = [papel for papel in candidatos if emissor(papel) in valores.index]
        return pd.Series([valores[emissor(papel)] for papel in papeis], index=papeis, dtype=float)
    return _nota


def meses_lidos(pontos: pd.DataFrame, coluna: str) -> pd.DatetimeIndex:
    por_mes = pontos.dropna(subset=[coluna]).groupby("month").size().loc[PERIODO[0]:PERIODO[1]]
    return pd.DatetimeIndex(por_mes.index[por_mes >= EMISSORES_MINIMOS])


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | conferir_hashes(SAIDA, [PAINEL.name])}")
    pontos = pontuar(pd.read_parquet(PAINEL))
    todos = pd.date_range(*PERIODO, freq="MS")
    colunas = mensal.retornos.columns
    conservadora_quadro = painel_da_nota(lambda mes, c: conservadora(mensal.retornos, mensal.proventos, mes, c),
                                         dados.universo, todos, colunas)
    periodos = {"2008-01 to 2026-09 (gate)": PERIODO,
                **{rotulo: tuple(map(pd.Timestamp, datas)) for rotulo, datas in SUBPERIODOS.items()}}
    aprovados = {}
    for nome, coluna in SINAIS.items():
        meses = meses_lidos(pontos, nome)
        print(f"\n{nome} (-{coluna}): months read {len(meses)} of {len(todos)} (dropped with < {EMISSORES_MINIMOS} "
              f"issuers: {len(todos) - len(meses)})")
        if len(meses) < 2:
            aprovados[nome] = (False, float("nan"))
            continue
        quadro = painel_da_nota(nota_do_painel(pontos, nome), dados.universo, meses, colunas)
        retornos = retornos_no_universo(mensal.retornos, dados.universo, meses)
        nota = conservadora_quadro.reindex(meses)
        ic, parcial = ic_por_mes(quadro, retornos), ic_parcial_por_mes(quadro, retornos, nota)
        for rotulo_ic, serie in (("IC", ic), ("Partial IC given the Conservadora score", parcial)):
            print(f"  {rotulo_ic}:")
            print(pd.DataFrame({rotulo: resumo_do_ic(serie.loc[de:ate]) for rotulo, (de, ate) in periodos.items()}).T
                  .to_string(float_format=lambda x: f"{x:.3f}"))
        correlacao = ic_por_mes(quadro, nota).dropna()
        cobertos = []
        for mes in meses:
            escolhas = carteira_da_nota(nota.loc[mes].dropna(), dados.setores, NOMES, POR_SETOR)
            cobertos.append(quadro.loc[mes].reindex(escolhas.index).notna().mean() if len(escolhas) else float("nan"))
        print(f"  Rank correlation with the Conservadora score: mean {correlacao.mean():.3f}; issuers per month median "
              f"{quadro.notna().sum(axis=1).median():.0f}; share of the Conservadora's {NOMES} picks with the signal: "
              f"mean {pd.Series(cobertos).mean():.2f}")
        passou = PORTAO.aprova(ic.dropna()) and PORTAO.aprova(parcial.dropna())
        resumo = resumo_do_ic(parcial)
        aprovados[nome] = (passou, float(resumo["t"]))

    print(f"\nGate ({PORTAO.nome}, on both the IC and the partial IC):")
    for nome, (passou, t) in aprovados.items():
        print(f"  {nome}: {'PASSED' if passou else 'NOT passed'}")
    passaram = {nome: t for nome, (passou, t) in aprovados.items() if passou}
    if passaram:
        escolhido = max(passaram, key=passaram.get)
        print(f"Next: stage B (attempt 54) only for {escolhido} (largest partial-IC t), after its own commit adding "
              "the trial row.")
    else:
        print("Stop: no attempt row; option signals stay display-only at most. No other window, threshold or rerun.")
    print(f"(total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
