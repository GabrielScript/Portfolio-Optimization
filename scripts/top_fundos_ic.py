"""
Stage A of the top-funds pre-registration (`docs/superpowers/specs/2026-10-29-top-fundos-copia-design.md`, `feb033f`).
Run once, after `scripts/top_fundos_painel.py` froze the panel:

    python -X utf8 scripts/top_fundos_ic.py > docs/top-fundos-ic-2026-10-29.txt

Monthly Spearman IC of the top-funds score (the mean CDA issuer weight of the 15 qualifying funds with the best
12-month quota return) against the month's total return in the month's universe, 2011-01 to 2026-09, months with at
least 30 qualifying and 15 ranked funds, exactly as attempt 52's `scripts/cda_ic.py`. Gate, all three: (a) IC >= 0.02
and t >= 2; (b) IC >= 0.032 (attempt 52); (c) the paired monthly IC(score) - IC(score over all ranked funds) has a
positive mean with t >= 2. No network.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.poder import ic_por_mes  # noqa: E402
from core.sinais import conservadora  # noqa: E402
from core.top_fundos import meses_cheios  # noqa: E402
from core.valor import painel_da_nota  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_conservadora import resumo_do_ic, retornos_no_universo  # noqa: E402
from scripts.avaliar_robustez import conferir_hashes  # noqa: E402
from scripts.avaliar_valor import PORTAO  # noqa: E402
from scripts.cda_ic import PERIODO, SUBPERIODOS, nota_do_painel  # noqa: E402
from scripts.cda_painel import PASTA_CAPS  # noqa: E402
from scripts.qpo_montar import FOLDER  # noqa: E402
from scripts.top_fundos_painel import PAINEL, SAIDA  # noqa: E402

IC_DA_TENTATIVA_52 = 0.032
T_MINIMO = 2.0


@dataclass(frozen=True)
class Veredito:
    s2: bool                   # (a) IC >= 0.02 and t >= 2
    acima_da_52: bool          # (b) IC >= attempt 52's 0.032
    top_acrescenta: bool       # (c) paired IC(top) - IC(all ranked): mean > 0 and t >= 2

    @property
    def aprovado(self) -> bool:
        return self.s2 and self.acima_da_52 and self.top_acrescenta


def veredito(ic_top: pd.Series, ic_todos: pd.Series) -> Veredito:
    """The three-part gate of the spec (§4) on the monthly ICs of the top-funds score and of the all-fund baseline;
    the difference is paired month by month, only where both ICs exist."""
    diferenca = resumo_do_ic((ic_top - ic_todos).dropna())
    return Veredito(s2=PORTAO.aprova(ic_top), acima_da_52=float(ic_top.dropna().mean()) >= IC_DA_TENTATIVA_52,
                    top_acrescenta=diferenca["IC médio"] > 0 and diferenca["t"] >= T_MINIMO)


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    sha256 = dados.sha256 | mensal.sha256 | conferir_hashes(SAIDA, [PAINEL.name]) \
        | conferir_hashes(PASTA_CAPS, ["market_caps.parquet"])
    painel = pd.read_parquet(PAINEL)
    print(f"Files (sha256): {sha256}")

    todos = pd.date_range(*PERIODO, freq="MS")
    meses = pd.DatetimeIndex([mes for mes in meses_cheios(painel) if PERIODO[0] <= mes <= PERIODO[1]])
    print(f"Months read: {len(meses)} of {len(todos)} (dropped with < 30 qualifying or < 15 ranked funds: "
          f"{len(todos) - len(meses)})")
    colunas = mensal.retornos.columns
    retornos = retornos_no_universo(mensal.retornos, dados.universo, meses)
    notas = {"Top-15 funds (score, gate)": "score", "All ranked funds (score_all, baseline)": "score_all",
             "Top-15 active weight (reported)": "active"}
    paineis = {nome: painel_da_nota(nota_do_painel(painel, coluna), dados.universo, meses, colunas)
               for nome, coluna in notas.items()}
    periodos = {"2011-01 to 2026-09 (gate)": PERIODO,
                **{rotulo: tuple(map(pd.Timestamp, datas)) for rotulo, datas in SUBPERIODOS.items()}}
    ics = {nome: ic_por_mes(quadro, retornos).loc[PERIODO[0]:PERIODO[1]] for nome, quadro in paineis.items()}
    ics["Paired difference top-15 - all ranked (gate c)"] = (ics["Top-15 funds (score, gate)"]
                                                            - ics["All ranked funds (score_all, baseline)"])
    for nome, serie in ics.items():
        print(f"\n{nome}: monthly Spearman IC, score of the month x total return of the month, universe of the month")
        print(pd.DataFrame({rotulo: resumo_do_ic(serie.loc[de:ate]) for rotulo, (de, ate) in periodos.items()}).T
              .to_string(float_format=lambda x: f"{x:.3f}"))

    lidos = painel[painel["month"].isin(meses)].groupby("month")[["qualifying", "with_quotas", "ranked"]].first()
    print(f"\nPer month read (median): qualifying {lidos['qualifying'].median():.0f}, with a return in the window "
          f"{lidos['with_quotas'].median():.0f}, ranked {lidos['ranked'].median():.0f}")
    principal = paineis["Top-15 funds (score, gate)"]
    conservadora_painel = painel_da_nota(
        lambda mes, c: conservadora(mensal.retornos, mensal.proventos, mes, c), dados.universo, meses, colunas)
    caps = pd.read_parquet(PASTA_CAPS / "market_caps.parquet", columns=["issuer", "month", "market_cap"])
    caps = caps.assign(month=pd.to_datetime(caps["month"]))
    caps_painel = painel_da_nota(nota_do_painel(caps, "market_cap"), dados.universo, meses, colunas)
    for rotulo, outro in {"the Conservadora score": conservadora_painel, "market cap": caps_painel}.items():
        correlacao = ic_por_mes(principal, outro).dropna()
        print(f"Rank correlation of the top-15 score with {rotulo}: mean {correlacao.mean():.3f} over "
              f"{len(correlacao)} months")

    resultado = veredito(ics["Top-15 funds (score, gate)"], ics["All ranked funds (score_all, baseline)"])
    print(f"\nGate (2011-01 to 2026-09): (a) {PORTAO.nome}: {'yes' if resultado.s2 else 'no'}; "
          f"(b) IC >= {IC_DA_TENTATIVA_52}: {'yes' if resultado.acima_da_52 else 'no'}; "
          f"(c) paired difference > 0 with t >= {T_MINIMO:.0f}: {'yes' if resultado.top_acrescenta else 'no'}.")
    print("PASSED. Next: stage B (attempt 54) after its own commit adding the trial row (N 53 -> 54)."
          if resultado.aprovado else "NOT passed. Stop: no attempt row; the copycat line is closed.")
    print(f"(total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
