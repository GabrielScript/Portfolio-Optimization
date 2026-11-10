"""
qml_g4_evaluate.py - G4 of the QML line (test branch only): Fama-MacBeth filter, PCA factor covariance, GRS.

Spec: ``docs/superpowers/specs/2026-11-10-qml-g4-design.md`` (committed before this script existed).

    .venv-qpo/Scripts/python -X utf8 scripts/qml_g4_evaluate.py --lago <lake> --fama-macbeth
    .venv-qpo/Scripts/python -X utf8 scripts/qml_g4_evaluate.py --lago <lake> --fase 0

P(up) comes from the stored caches (QML v3 and the logistic baseline); nothing is retrained.
"""

from __future__ import annotations

import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core.backtest import ResultadoBacktest  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.metrics import comparar_benchmark, retornos_de_valores  # noqa: E402
from core.multivariate import FamaMacBethResult, fama_macbeth, grs_test, pca_factor_model  # noqa: E402
from core.qml import forward_returns, qml_strategy  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from scripts.avaliar_robustez import _acoes, referencias  # noqa: E402
from scripts.avaliar_universo_dinamico import CFG_BT, CFG_KELLY  # noqa: E402
from scripts.curvas_ao_vivo import ARQUIVOS, LAGO, congelar  # noqa: E402
from scripts.qml_avaliar import CACHE, ProbabilidadesCongeladas, _linha, _rodar, _tabela, seletor_do_composite  # noqa: E402
from scripts.qml_g2_avaliar import _carregar  # noqa: E402
from scripts.qml_g3_avaliar import CACHE_LOGISTICA, ULTIMA_SESSAO, SemSinal  # noqa: E402

HORIZON = 63
FM_LAGS = 3                    # 63-session labels sampled every 21 sessions overlap three times
TOP_SHARE = 0.20               # the top quintile of the Composite score
MIN_NAMES = 10
COMPOSITE = "Composite"
ARMS = ("A1 QML-Composite (Meio-Kelly), Ledoit-Wolf", "A2 QML-Composite (Meio-Kelly), PCA",
        "A3 No signal, Ledoit-Wolf", "A4 No signal, PCA", "A5 Logistic (Meio-Kelly), Ledoit-Wolf")


# --- Fama-MacBeth ------------------------------------------------------------------------------------


def decision_dates(cached: pd.DatetimeIndex, sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The rebalance dates of the cache: the first session of each consecutive pair (the second is the
    session after, kept only for a freeze-day shift)."""
    posicoes = sessions.get_indexer(cached.sort_values())
    if (posicoes < 0).any():
        raise ValueError("a cached date is not a session of the price frame")
    manter = np.r_[True, np.diff(posicoes) > 1]
    return pd.DatetimeIndex(cached.sort_values()[manter])


def top_share_mask(score: pd.DataFrame, share: float = TOP_SHARE) -> pd.DataFrame:
    """True where a name is in the top ``share`` of that date's score (ties at the cut are kept)."""
    corte = score.quantile(1.0 - share, axis=1)
    return score.ge(corte, axis=0) & score.notna()


def fama_macbeth_panel(retornos: pd.DataFrame, universo: pd.DataFrame, nota: Callable, sinais: Mapping[str, pd.DataFrame],
                       datas: Sequence[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Target (forward 63-session return), Composite score and signals on the decision dates, over the
    engine's eligible names with a full training window."""
    futuro = forward_returns(retornos, HORIZON)
    meses = pd.to_datetime(universo["month"])
    elegiveis = universo["security"].groupby(meses).apply(list).to_dict()
    alvos, notas = {}, {}
    for data in datas:
        posicao = int(retornos.index.get_loc(data))
        if posicao < CFG_BT.janela_treino or posicao + HORIZON >= len(retornos):
            continue
        mes = data.to_period("M").to_timestamp()
        pit = [papel for papel in elegiveis.get(mes, []) if papel in retornos.columns]
        janela = retornos.iloc[posicao - CFG_BT.janela_treino:posicao][pit].dropna(axis=1)
        nomes = list(janela.columns)
        notas[data] = nota(mes, nomes).reindex(nomes)
        alvos[data] = futuro.loc[data].reindex(nomes)
    alvo = pd.DataFrame(alvos).T.sort_index()
    score = pd.DataFrame(notas).T.reindex(index=alvo.index, columns=alvo.columns)
    return alvo, score, {nome: quadro.reindex(index=alvo.index, columns=alvo.columns) for nome, quadro in sinais.items()}


def _linha_fm(resultado: FamaMacBethResult, sinal: str) -> dict[str, float]:
    return {"coef signal": resultado.mean[sinal], "t signal": resultado.t_stat[sinal],
            "coef Composite": resultado.mean["composite"], "t Composite": resultado.t_stat["composite"],
            "dates": float(len(resultado.coefficients)), "mean names": float(resultado.n_names.mean())}


def run_fama_macbeth(lago: Path) -> int:
    inicio = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="qml-g4-") as temporaria:
        somas = congelar(lago, Path(temporaria), ARQUIVOS)
        dados, _mensal, nota, _ultima, _cdi = _carregar(Path(temporaria))
    retornos = dados.precos.loc[:ULTIMA_SESSAO].ffill(limit=5).pct_change().iloc[1:]
    sinais = {"QML": pd.read_parquet(CACHE), "Logistic": pd.read_parquet(CACHE_LOGISTICA)}
    datas = decision_dates(sinais["QML"].index, retornos.index)
    alvo, score, paineis = fama_macbeth_panel(retornos, dados.universo, nota, sinais, datas)
    topo = top_share_mask(score)
    linhas = {}
    for nome, painel in paineis.items():
        for amostra, mascara in (("all names", None), ("top quintile of the Composite", topo)):
            y = alvo if mascara is None else alvo.where(mascara)
            resultado = fama_macbeth(y, {"composite": score, nome: painel}, lags=FM_LAGS, min_names=MIN_NAMES)
            linhas[(nome, amostra)] = _linha_fm(resultado, nome)
    tabela = pd.DataFrame(linhas).T
    veredito = {}
    for nome in paineis:
        cheio, alto = tabela.loc[(nome, "all names")], tabela.loc[(nome, "top quintile of the Composite")]
        veredito[nome] = bool(cheio["t signal"] >= 2.0 and alto["coef signal"] > 0 and alto["t signal"] >= 1.0)
    saida = RAIZ / "docs" / "qml-g4-fama-macbeth-2026-11-10.txt"
    texto = (f"G4 item 1, Fama-MacBeth filter (spec docs/superpowers/specs/2026-11-10-qml-g4-design.md). "
             f"Lake exports sha256: {somas}\n"
             f"{len(datas)} decision dates in the cache, {len(alvo)} with a realized {HORIZON}-session return; "
             f"Newey-West {FM_LAGS} lags; regressors z-scored per date and clipped to +-3; coefficients = forward "
             f"{HORIZON}-session return per one cross-sectional standard deviation.\n\n"
             f"{tabela.to_string(float_format=lambda x: f'{x:.4f}')}\n\n"
             f"Filter (t >= 2 on all names AND positive with t >= 1 in the top quintile): "
             + ", ".join(f"{nome} {'PASSES' if passa else 'fails'}" for nome, passa in veredito.items()) + "\n")
    saida.write_text(texto, encoding="utf-8", newline="\n")
    tabela.to_csv(RAIZ / "docs" / "dados" / "qml-g4-fama-macbeth.csv", encoding="utf-8", lineterminator="\n")
    print(texto)
    print(f"{saida.relative_to(RAIZ)} written ({time.perf_counter() - inicio:.0f} s)")
    return 0


# --- covariance arms and GRS -------------------------------------------------------------------------


class FactorCountLog:
    """The PCA covariance estimator, recording the number of factors Marchenko-Pastur kept per decision."""

    def __init__(self) -> None:
        self.counts: dict[pd.Timestamp, int] = {}

    def __call__(self, janela: pd.DataFrame) -> pd.DataFrame:
        modelo = pca_factor_model(janela)
        self.counts[janela.index[-1]] = modelo.n_factors
        return modelo.cov


def _pareado(curvas: Mapping[str, ResultadoBacktest], braco: str, referencia: str, rf: pd.Series) -> str:
    comparacao = comparar_benchmark(curvas[braco].valores, curvas[referencia].valores, rf)
    return (f"  {braco} against {referencia}: alpha {comparacao.alfa_anual:+.4f} a year, t {comparacao.t_stat_alfa:.2f}, "
            f"P[>] {psr_da_diferenca(curvas[braco].valores, curvas[referencia].valores):.3f}")


def run_phase(lago: Path, fase: int) -> int:
    inicio = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="qml-g4-") as temporaria:
        somas = congelar(lago, Path(temporaria), ARQUIVOS)
        dados, _mensal, nota, _ultima, cdi = _carregar(Path(temporaria))
    precos = dados.precos.loc[:ULTIMA_SESSAO].iloc[fase:]
    acoes = _acoes(precos)
    cdi = cdi.loc[:ULTIMA_SESSAO]
    setores = dict(dados.setores)
    seletor = seletor_do_composite(nota, setores)
    qml = ProbabilidadesCongeladas(pd.read_parquet(CACHE))
    logistica = ProbabilidadesCongeladas(pd.read_parquet(CACHE_LOGISTICA))
    pca_a2, pca_a4 = FactorCountLog(), FactorCountLog()

    def braco(fonte, covariance=None):
        return qml_strategy(fonte, CFG_KELLY, setores, cdi, selector=seletor, tilt=1.0, covariance=covariance)

    estrategias = dict(zip(ARMS, (braco(qml), braco(qml, pca_a2), braco(SemSinal(qml)), braco(SemSinal(qml), pca_a4),
                                  braco(logistica))))
    curvas: dict[str, ResultadoBacktest] = {
        COMPOSITE: _rodar(acoes, dados.universo, estrategia_por_nota(nota, setores), COMPOSITE, inicio)}
    for nome, estrategia in estrategias.items():
        curvas[nome] = _rodar(acoes, dados.universo, estrategia, nome, inicio)
    bova = referencias(precos, CFG_BT)["BOVA11"]
    rf = cdi.reindex(curvas[COMPOSITE].valores.index).fillna(0.0)
    tabela = _tabela(curvas, bova.valores, rf, curvas[COMPOSITE].valores)
    tabela.loc["BOVA11"] = _linha(bova.valores, bova.valores, rf)
    tabela.insert(0, "Fase", fase)

    excesso = pd.DataFrame({nome: retornos_de_valores(curvas[nome].valores) for nome in ARMS})
    excesso = excesso.sub(rf.reindex(excesso.index).fillna(0.0), axis=0)
    fator = retornos_de_valores(curvas[COMPOSITE].valores).sub(rf, fill_value=0.0).reindex(excesso.index).to_frame(COMPOSITE)
    grs = grs_test(excesso.dropna(), fator.dropna())
    contagens = pd.Series(pca_a2.counts)
    recusas = {nome: len(c.recusas) + len(c.falhas) for nome, c in curvas.items()}
    linhas = [
        f"G4 items 2-3, phase {fase} (spec docs/superpowers/specs/2026-11-10-qml-g4-design.md). Lake exports sha256: {somas}",
        f"First decision {acoes.index[CFG_BT.janela_treino]:%d/%m/%Y}; last session {acoes.index[-1]:%d/%m/%Y}; "
        f"refusals + failures {recusas}.",
        f"PCA factors kept by Marchenko-Pastur per decision (A2): mean {contagens.mean():.2f}, min {contagens.min()}, "
        f"max {contagens.max()}, over {len(contagens)} decisions.",
        tabela.to_string(float_format=lambda x: f"{x:.4f}"),
        "",
        "Covariance's own contribution (alpha regression of one arm on the other, Newey-West):",
        _pareado(curvas, ARMS[1], ARMS[0], rf),
        _pareado(curvas, ARMS[3], ARMS[2], rf),
        "",
        f"GRS, the five arms against the Composite (daily excess over CDI, T={grs.n_obs}): F = {grs.f_stat:.3f}, "
        f"p = {grs.p_value:.4f}; daily alphas annualized: "
        + ", ".join(f"{nome.split()[0]} {alfa * 252:+.4f}" for nome, alfa in grs.alphas.items()),
        "Caveat: GRS assumes iid normal errors; daily returns have fat tails and autocorrelation.",
    ]
    texto = "\n".join(linhas) + "\n"
    (RAIZ / "docs" / f"qml-g4-2026-11-10-f{fase}.txt").write_text(texto, encoding="utf-8", newline="\n")
    tabela.to_csv(RAIZ / "docs" / "dados" / f"qml-g4-f{fase}.csv", encoding="utf-8", lineterminator="\n")
    print(texto)
    print(f"phase {fase} done ({time.perf_counter() - inicio:.0f} s)")
    return 0


def main(argv: list[str]) -> int:
    lago = Path(argv[argv.index("--lago") + 1]) if "--lago" in argv else LAGO
    if "--fama-macbeth" in argv:
        return run_fama_macbeth(lago)
    return run_phase(lago, int(argv[argv.index("--fase") + 1]) if "--fase" in argv else 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
