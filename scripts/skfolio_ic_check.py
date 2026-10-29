"""
Our monthly Spearman ICs against skfolio's AlphaForecastEvaluation on the same frozen panels (DECISOES 169).

Run with the test-only venv (`../.venv-skfolio`, requirements-skfolio.txt); reads `docs/dados/2026-10-22` and
`docs/dados/2026-10-24-cvm`, writes nothing. skfolio's `signal_lag=0` pairs the month-m score, built only from data
before m, with the return of m: our IC's definition. Its t uses a HAC standard error (quadratic spectral, fixed-b,
wild bootstrap), stricter than our iid t when the monthly ICs are autocorrelated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from skfolio.alpha import alpha_forecast_evaluation  # noqa: E402
from skfolio.containers import AssetPanel  # noqa: E402
from sklearn.base import BaseEstimator  # noqa: E402

from core.crescimento_lucro import nota_de_crescimento  # noqa: E402
from core.poder import ic_por_mes  # noqa: E402
from core.sinais import conservadora  # noqa: E402
from core.valor import painel_da_nota  # noqa: E402
from core.views import nota_media_dos_escores  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor  # noqa: E402
from scripts.avaliar_conservadora import resumo_do_ic, retornos_no_universo  # noqa: E402
from scripts.paper_v3 import nota_do_composto  # noqa: E402
from scripts.qpo_montar import FOLDER  # noqa: E402


class SinalPronto(BaseEstimator):
    """The 'sinal' field as the forecast: the score is already built point in time outside skfolio."""

    def fit(self, X, y=None):  # noqa: N803 (sklearn's name)
        return self

    def fit_transform(self, X, y=None):  # noqa: N803
        return np.asarray(X["sinal"], dtype=float)


def main() -> int:
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = avaliar_valor.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    balancos, _ = avaliar_valor.carregar_balancos(FOLDER.parent / "2026-10-24-cvm")
    meses = pd.date_range(*avaliar_valor.PERIODO_DO_IC, freq="MS")
    colunas = mensal.retornos.columns
    retornos = retornos_no_universo(mensal.retornos, dados.universo, meses).reindex(columns=colunas)
    notas = {
        "Conservadora": lambda mes, c: conservadora(mensal.retornos, mensal.proventos, mes, c),
        "value (MF + TV)": nota_media_dos_escores(*avaliar_valor.notas(balancos, mensal).values()),
        "profit growth": nota_de_crescimento(balancos.fundamentos),
        "composite (attempt 51 / paper-v3)": nota_do_composto(mensal.retornos, mensal.proventos, balancos),
    }
    linhas = []
    for nome, nota in notas.items():
        painel = painel_da_nota(nota, dados.universo, meses, colunas)
        nossos = resumo_do_ic(ic_por_mes(painel, retornos))
        validos = painel.notna() & retornos.notna()
        X = AssetPanel(fields={"sinal": painel.where(validos).to_numpy(), "retorno": retornos.where(validos).to_numpy()},
                       observations=np.arange(len(meses)), asset_names=list(colunas))
        avaliacao = alpha_forecast_evaluation(SinalPronto(), X, target="retorno", holding_period=1, signal_lag=0,
                                              evaluation_step=1, factor_exposures=None, annualization_factor=12.0)
        deles = avaliacao.ic_summary().loc["spearman_ic"]
        linhas.append({"signal": nome, "months": int(nossos["meses"]), "our IC": nossos["IC médio"],
                       "skfolio IC": deles["mean"], "our t (iid)": nossos["t"], "skfolio t (HAC)": deles["t_stat"],
                       "skfolio p": deles["p_value"],
                       "95% CI": f"[{deles['mean_ci_lower']:.3f}, {deles['mean_ci_upper']:.3f}]",
                       "hit rate": deles["hit_rate"]})
    print(f"Monthly Spearman IC, {meses[0]:%Y-%m} to {meses[-1]:%Y-%m}, score of the month x total return of the month, "
          "point-in-time universe")
    print(pd.DataFrame(linhas).set_index("signal").to_string(float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
