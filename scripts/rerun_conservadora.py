"""
Descriptive rerun of the Conservadora (S2.1, `scripts/avaliar_conservadora.py`) on the corrected lake: a data
correction (DECISOES 106 and 115), not a new trial; N and the registered verdict (DECISOES 82-83) do not change.

Usage: .venv/Scripts/python -X utf8 scripts/rerun_conservadora.py > docs/conservadora-vol-2026-10-23.txt
(the 22/10 run, `docs/conservadora-2026-10-22.txt`, is the same output without the two volatility columns)

The pre-registered script runs unchanged except for its inputs:
- the five files come from `docs/dados/2026-10-22/` (the lake's exports of 21/10/2026 21:48, with the CODBDI 58 and
  RPMG3 fixes of DECISOES 103-104), each checked against that folder's SHA256SUMS;
- the daily prices are cut at 16/10/2026, the first run's last session, so the period and the phases are the same;
- the run date is the first run's (2026-10-20), so the Selic target is the same; the CDI comes from the BCB as then;
- the DSR's N is the first run's (22), not today's `docs/tentativas.csv` count, so the DSR is comparable.

Added on 23/10/2026 (user's volatility goal, DECISOES 116), descriptive only: each phase table also shows the
annualized volatility and downside volatility of the four curves (`core.volatilidade`, on the engine's daily values).
Every other number must stay the 22/10 run's to the digit.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.volatilidade import volatilidades  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import Dados  # noqa: E402

FOLDER = Path(__file__).resolve().parent.parent / "docs" / "dados" / "2026-10-22"
FIRST_RUN_LAST_SESSION = pd.Timestamp("2026-10-16")
FIRST_RUN_DATE = "2026-10-20"
FIRST_RUN_TRIALS = 22
VOLATILITY_COLUMNS = ("Vol anual", "Vol de baixa")


def cut_to_first_run(dados: Dados) -> Dados:
    """The prices up to the first run's last session; everything else as loaded."""
    return replace(dados, precos=dados.precos.loc[:FIRST_RUN_LAST_SESSION])


def with_volatility(tabela: pd.DataFrame, curvas: Mapping[str, pd.Series]) -> pd.DataFrame:
    """``tabela`` with both volatilities of each row's curve (``curvas``: name -> daily values) as the last columns."""
    medidas = {nome: volatilidades(curvas[nome]) for nome in tabela.index}
    return tabela.assign(**{VOLATILITY_COLUMNS[0]: [medidas[nome].anual for nome in tabela.index],
                            VOLATILITY_COLUMNS[1]: [medidas[nome].de_baixa for nome in tabela.index]})


def main() -> int:
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    load_prices, load_monthly = avaliar_robustez.carregar, avaliar_conservadora.carregar_mensal
    avaliar_conservadora.carregar = lambda: cut_to_first_run(load_prices())
    avaliar_conservadora.carregar_mensal = lambda: load_monthly(FOLDER)
    avaliar_conservadora.contar_tentativas = lambda: FIRST_RUN_TRIALS
    phase_measures, table_text = avaliar_conservadora.medidas_da_fase, avaliar_conservadora._tabela
    avaliar_conservadora.medidas_da_fase = lambda curvas, rf, n: with_volatility(
        phase_measures(curvas, rf, n), {nome: curva.valores for nome, curva in curvas.items()})
    avaliar_conservadora._tabela = lambda tabela, porcento: table_text(tabela, [*porcento, *VOLATILITY_COLUMNS])
    sys.argv[1:] = [FIRST_RUN_DATE]
    print(f"Data correction of docs/conservadora-2026-10-20.txt: inputs from {FOLDER.name}, prices cut at "
          f"{FIRST_RUN_LAST_SESSION:%d/%m/%Y}, run date {FIRST_RUN_DATE}, DSR N {FIRST_RUN_TRIALS}; plus "
          f"{' and '.join(VOLATILITY_COLUMNS)} (std of the daily returns x sqrt(252); downside: target 0, all days)")
    return avaliar_conservadora.main()


if __name__ == "__main__":
    raise SystemExit(main())
