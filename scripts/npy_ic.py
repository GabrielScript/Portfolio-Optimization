"""
Stage A of the NPY pre-registration (`docs/superpowers/specs/2026-10-29-npy-conservadora-design.md`, committed in
`d1b3740` before any code or reading). Run once:

    python -X utf8 scripts/npy_ic.py > docs/npy-ic-2026-10-29.txt

Score (`core.npy.conservadora_npy`): the Conservadora with the net payout yield (12-month dividend yield minus the
issuer's net issuance, latest count over its 24-month average) in the place of the dividend yield. Monthly Spearman IC
against the month's total return in the month's universe, 2012-06 to 2026-09. Gate, both required: NPY-Conservadora IC
>= 0.02 with t >= 2, and the mean monthly paired difference NPY-Conservadora IC minus DY-Conservadora IC >= 0. No
network: prices, universe and dividends from docs/dados/2026-10-22/, counts from docs/dados/2026-10-29-cvm/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.npy import contagens_por_mes, conservadora_npy, emissao_liquida, net_payout_yield  # noqa: E402
from core.poder import ic_por_mes  # noqa: E402
from core.risco import emissor  # noqa: E402
from core.sinais import conservadora, dividend_yield_12m  # noqa: E402
from core.valor import painel_da_nota  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_conservadora import resumo_do_ic, retornos_no_universo  # noqa: E402
from scripts.avaliar_robustez import conferir_hashes  # noqa: E402
from scripts.avaliar_valor import PORTAO  # noqa: E402
from scripts.qpo_montar import FOLDER  # noqa: E402

PASTA_CONTAGENS = FOLDER.parent / "2026-10-29-cvm"
PERIODO = (pd.Timestamp("2012-06-01"), pd.Timestamp("2026-09-01"))
SUBPERIODOS = {"2012-2016": ("2012-06-01", "2016-12-01"), "2017-2021": ("2017-01-01", "2021-12-01"),
               "2022-2026": ("2022-01-01", "2026-09-01")}


def gate(ics_npy: pd.Series, ics_dy: pd.Series) -> tuple[bool, bool]:
    """(S2 gate on the NPY-Conservadora IC, mean paired difference NPY minus DY >= 0), on the pre-registered months."""
    npy = ics_npy.loc[PERIODO[0]:PERIODO[1]].dropna()
    diferenca = (ics_npy - ics_dy).loc[PERIODO[0]:PERIODO[1]].dropna()
    return bool(PORTAO.aprova(npy)), bool(diferenca.mean() >= 0.0)


def carregar_contagens() -> tuple[pd.DataFrame, dict[str, str]]:
    sha256 = conferir_hashes(PASTA_CONTAGENS, ["market_caps.parquet"])
    arquivo = pd.read_parquet(PASTA_CONTAGENS / "market_caps.parquet", columns=["issuer", "month", "shares_today_units"])
    return contagens_por_mes(arquivo), {f"{nome} (2026-10-29)": valor for nome, valor in sha256.items()}


def cobertura_da_emissao(contagens: pd.DataFrame, universo: pd.DataFrame) -> tuple[float, float]:
    """Share of the universe's name-months (in the period) with a 24-month issuance, and its mean absolute value."""
    meses = pd.date_range(*PERIODO, freq="MS")
    no_periodo = universo.assign(month=pd.to_datetime(universo["month"]))
    no_periodo = no_periodo[no_periodo["month"].isin(meses)]
    emissoes = {mes: emissao_liquida(contagens, mes) for mes in meses}
    valores = [emissoes[linha.month].get(emissor(linha.security)) for linha in no_periodo.itertuples()]
    presentes = pd.Series([v for v in valores if v is not None], dtype=float)
    return len(presentes) / len(valores), float(presentes.abs().mean())


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    contagens, sha256_contagens = carregar_contagens()
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_contagens}")

    meses = pd.date_range(*PERIODO, freq="MS")
    colunas = mensal.retornos.columns
    retornos = retornos_no_universo(mensal.retornos, dados.universo, meses)
    notas = {
        "NPY-Conservadora": lambda mes, c: conservadora_npy(mensal.retornos, mensal.proventos, contagens, mes, c),
        "DY-Conservadora": lambda mes, c: conservadora(mensal.retornos, mensal.proventos, mes, c),
        "NPY alone": lambda mes, c: net_payout_yield(mensal.proventos, contagens, mes, c),
        "DY alone": lambda mes, c: dividend_yield_12m(mensal.proventos, mes, c),
    }
    paineis = {nome: painel_da_nota(nota, dados.universo, meses, colunas) for nome, nota in notas.items()}
    ics = {nome: ic_por_mes(painel, retornos) for nome, painel in paineis.items()}
    ics["NPY minus DY (Conservadora)"] = ics["NPY-Conservadora"] - ics["DY-Conservadora"]

    periodos = {"2012-06 to 2026-09 (gate)": PERIODO,
                **{rotulo: tuple(map(pd.Timestamp, datas)) for rotulo, datas in SUBPERIODOS.items()}}
    for nome, serie in ics.items():
        print(f"\n{nome}: monthly Spearman IC, score of the month x total return of the month, universe of the month")
        print(pd.DataFrame({rotulo: resumo_do_ic(serie.loc[de:ate]) for rotulo, (de, ate) in periodos.items()}).T
              .to_string(float_format=lambda x: f"{x:.3f}"))

    share, media_abs = cobertura_da_emissao(contagens, dados.universo)
    correlacao = ic_por_mes(paineis["NPY-Conservadora"], paineis["DY-Conservadora"]).dropna()
    print(f"\nUniverse name-months with a 24-month issuance: {share:.1%}; mean |issuance| {media_abs:.3f}")
    print(f"Rank correlation NPY-Conservadora vs DY-Conservadora: mean {correlacao.mean():.3f} over {len(correlacao)} "
          f"months")

    forte, nao_pior = gate(ics["NPY-Conservadora"], ics["DY-Conservadora"])
    print(f"\nGate: (1) {PORTAO.nome} on NPY-Conservadora: {'passed' if forte else 'NOT passed'}; (2) mean paired "
          f"difference >= 0: {'passed' if nao_pior else 'NOT passed'}. "
          + ("PASSED: stage B (attempt 52) after its own commit adding the trial row." if forte and nao_pior else
             "NOT passed: stop; no attempt row; the Conservadora keeps the dividend yield."))
    print(f"(total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
