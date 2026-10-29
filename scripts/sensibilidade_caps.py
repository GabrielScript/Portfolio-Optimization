"""
sensibilidade_caps.py - the value score's and attempt 51's composite's ICs, and attempt 51's walk-forward, on market
values with the 1000x share-count slips fixed (lake `7f365a3`, DECISOES 170). A data correction, not a new trial: same
code, same frozen prices, universe, statements and CDI; only `market_caps.parquet` comes from
`docs/dados/2026-10-29-cvm/` (statements stay `docs/dados/2026-10-24-cvm/`). The DSR N stays 51.

    python -X utf8 scripts/sensibilidade_caps.py > docs/sensibilidade-caps-2026-10-29.txt

Attempt 51's curves go to a temporary folder: `docs/dados/composto-curvas-2026-10-28.parquet` stays the bench's input.
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.poder import ic_por_mes  # noqa: E402
from core.valor import Balancos, painel_da_nota  # noqa: E402
from core.views import nota_media_dos_escores  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor, composto_crescimento_avaliar  # noqa: E402
from scripts.avaliar_conservadora import resumo_do_ic, retornos_no_universo  # noqa: E402
from scripts.avaliar_robustez import conferir_hashes  # noqa: E402
from scripts.qpo_montar import FOLDER  # noqa: E402
from scripts.qubo_universo_avaliar import PASTA_CVM, _value_score  # noqa: E402

PASTA_CORRIGIDA = FOLDER.parent / "2026-10-29-cvm"
carregar_balancos_de_2026_10_24 = avaliar_valor.carregar_balancos


def com_caps_corrigidos(pasta: Path = PASTA_CVM) -> tuple[Balancos, dict[str, str]]:
    """The statements of ``pasta`` with the market values of ``PASTA_CORRIGIDA``, and the hashes of both."""
    balancos, sha256 = carregar_balancos_de_2026_10_24(pasta)
    hash_caps = conferir_hashes(PASTA_CORRIGIDA, ["market_caps.parquet"])["market_caps.parquet"]
    valor_mercado = pd.read_parquet(PASTA_CORRIGIDA / "market_caps.parquet", columns=["issuer", "month", "market_cap"])
    valor_mercado["month"] = pd.to_datetime(valor_mercado["month"])
    return (dataclasses.replace(balancos, valor_mercado=valor_mercado),
            sha256 | {"market_caps.parquet (2026-10-29)": hash_caps})


def tabela_de_ics() -> pd.DataFrame:
    """Monthly Spearman IC of each score on the old and the fixed market values, as `scripts/skfolio_ic_check.py`."""
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = avaliar_valor.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    meses = pd.date_range(*avaliar_valor.PERIODO_DO_IC, freq="MS")
    colunas = mensal.retornos.columns
    retornos = retornos_no_universo(mensal.retornos, dados.universo, meses).reindex(columns=colunas)
    linhas = []
    for caps, carregar in (("2026-10-24", carregar_balancos_de_2026_10_24), ("fixed", com_caps_corrigidos)):
        avaliar_valor.carregar_balancos = carregar                       # what _value_score reads
        balancos, _ = carregar(PASTA_CVM)
        notas = {"value (MF + TV)": nota_media_dos_escores(*avaliar_valor.notas(balancos, mensal).values()),
                 "composite (attempt 51)": _value_score(mensal, com_crescimento=True)[0]}
        for nome, nota in notas.items():
            resumo = resumo_do_ic(ic_por_mes(painel_da_nota(nota, dados.universo, meses, colunas), retornos))
            linhas.append({"signal": nome, "caps": caps, "months": int(resumo["meses"]),
                           "IC": resumo["IC médio"], "t (iid)": resumo["t"]})
    return pd.DataFrame(linhas).set_index(["signal", "caps"])


def main() -> int:
    print(f"Monthly Spearman IC, {avaliar_valor.PERIODO_DO_IC[0]:%Y-%m} to {avaliar_valor.PERIODO_DO_IC[1]:%Y-%m}, "
          "score of the month x total return of the month, point-in-time universe")
    print(tabela_de_ics().to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nAttempt 51 (scripts/composto_crescimento_avaliar.py) on the fixed market values:", flush=True)
    avaliar_valor.carregar_balancos = com_caps_corrigidos
    with tempfile.TemporaryDirectory() as pasta:
        composto_crescimento_avaliar.CURVES_FILE = Path(pasta) / "composto-curvas.parquet"
        return composto_crescimento_avaliar.main()


if __name__ == "__main__":
    raise SystemExit(main())
