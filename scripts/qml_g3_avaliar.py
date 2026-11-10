"""
qml_g3_avaliar.py - G3 of the QML arm (test branch only): the signal in the selection, not in the weights.

Spec: ``docs/superpowers/specs/2026-11-10-qml-g3-design.md`` (committed before the first G3 run).

Default run (arms A-D of the spec), all with the Composite's book rule and equal weights:
  * Composite                       - the benchmark, unchanged.
  * Composite + QML score           - z(Composite score) + 0.5 z(QML conviction): the gate arm.
  * QML top 15                      - the book rule ordered by P(up) alone (descriptive).
  * Composite + logistic score      - the gate arm with a classical logistic regression (control).
  * Composite + shuffled score 0-4  - the gate arm with each date's P(up) dealt at random (placebo).

``--atribuicao`` runs the controls of the frozen QML-Composite (Meio-Kelly) arm instead: no signal,
anti-signal, shuffled P(up) and the logistic baseline inside the Composite's 15 names.

    .venv-qpo/Scripts/python -X utf8 scripts/qml_g3_avaliar.py --lago <lake> --fase 0 [--atribuicao]

P(up) comes from the stored v3 cache; nothing is retrained. Writes docs/qml-g3-<mode>2026-11-10-f<fase>.txt
and docs/dados/qml-g3-<mode>f<fase>.csv.
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core.backtest import CarteiraRecusada, Estrategia, ResultadoBacktest  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.estimation import estimar_momentos  # noqa: E402
from core.qml import (ProbabilitySource, QMLConfig, UniverseSelector, angles, build_panel,  # noqa: E402
                      cross_sectional_z, features_at, qml_strategy)
from core.risco import LIMITES, Limites, checar_carteira  # noqa: E402
from core.sinais import NOMES, POR_SETOR, carteira_da_nota  # noqa: E402
from scripts.avaliar_robustez import _acoes, referencias  # noqa: E402
from scripts.avaliar_universo_dinamico import CFG_BT, CFG_KELLY  # noqa: E402
from scripts.curvas_ao_vivo import ARQUIVOS, LAGO, congelar  # noqa: E402
from scripts.qml_avaliar import (CACHE, ProbabilidadesCongeladas, _linha, _rodar, _tabela,  # noqa: E402
                                 seletor_combinado, seletor_do_composite)
from scripts.qml_g2_avaliar import _carregar  # noqa: E402

ULTIMA_SESSAO = pd.Timestamp("2026-11-06")       # the window of DECISOES 318/319: later sessions stay out
PESO_QML = 0.5                                   # fixed by the spec, never scanned
PLACEBOS = 5
CACHE_LOGISTICA = RAIZ / "docs" / "dados" / "qml-logistica-v3.parquet"
COMPOSITE = "Composite"


# --- probability sources used as controls ------------------------------------------------------------


class SemSinal:
    """p = 0.5 for every name: the conviction is zero and the half-Kelly book is minimum variance."""

    def __init__(self, base: ProbabilitySource) -> None:
        self._base = base

    def predict(self, date: pd.Timestamp) -> pd.Series:
        return self._base.predict(date) * 0.0 + 0.5


class AntiSinal:
    """1 - p: a signal with real direction must lose here."""

    def __init__(self, base: ProbabilitySource) -> None:
        self._base = base

    def predict(self, date: pd.Timestamp) -> pd.Series:
        return 1.0 - self._base.predict(date)


class Embaralhada:
    """Placebo: the date's own P(up) values dealt to the names at random (same seed and date, same deal)."""

    def __init__(self, base: ProbabilitySource, seed: int) -> None:
        self._base, self._seed = base, seed

    def predict(self, date: pd.Timestamp) -> pd.Series:
        linha = self._base.predict(date)
        rng = np.random.default_rng([self._seed, int(date.value // 10**9)])
        return pd.Series(rng.permutation(linha.to_numpy()), index=linha.index)


def probabilidades_logisticas(retornos: pd.DataFrame, datas: Sequence[pd.Timestamp],
                              config: QMLConfig) -> pd.DataFrame:
    """
    The classical baseline: a logistic regression on the circuit's own panel (same six angles, same relative
    label), one model per two-year block trained on labels realized before 1 January of the block.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    painel = build_panel(retornos, config)
    modelos: dict[int, LogisticRegression] = {}
    linhas: dict[pd.Timestamp, pd.Series] = {}
    for data in datas:
        bloco = data.year - data.year % 2
        if bloco not in modelos:
            treino = np.asarray(painel.label_dates < pd.Timestamp(year=bloco, month=1, day=1))
            modelos[bloco] = LogisticRegression(C=1.0, max_iter=500).fit(painel.x[treino], painel.y[treino])
        z = cross_sectional_z(features_at(retornos.loc[:data], config.min_history))
        linhas[data] = pd.Series(modelos[bloco].predict_proba(angles(z))[:, 1], index=z.index)
    return pd.DataFrame(linhas).T.sort_index()


# --- the book rule with equal weights ----------------------------------------------------------------


def estrategia_de_pesos_iguais(selector: UniverseSelector, setores: Mapping[str, str],
                               limites: Limites = LIMITES) -> Estrategia:
    """Equal weights on the selector's names, re-checked by the fixed rules as the Composite's book is."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        data = treino.index[-1]
        nomes = [papel for papel in selector(data, treino) if papel in treino.columns]
        if not nomes:
            raise ValueError(f"{data:%Y-%m-%d}: the selector returned no eligible name")
        pesos = pd.Series(1.0 / len(nomes), index=nomes)
        violacoes = checar_carteira(pesos, setores, estimar_momentos(treino[nomes], shrink_mu=False).cov, limites)
        if violacoes:
            raise CarteiraRecusada(violacoes)
        return pesos
    return _estrategia


def seletor_por_probabilidade(fonte: ProbabilitySource, setores: Mapping[str, str], nomes: int = NOMES,
                              por_setor: int = POR_SETOR) -> UniverseSelector:
    """The Composite's book rule ordered by P(up) alone, over the engine's eligible names."""
    def _select(data: pd.Timestamp, treino: pd.DataFrame) -> list[str]:
        probabilidades = fonte.predict(data).reindex(treino.columns).dropna()
        return list(carteira_da_nota(probabilidades, setores, nomes, por_setor).index)
    return _select


# --- the two arm sets ---------------------------------------------------------------------------------


def bracos_g3(nota, setores: Mapping[str, str], qml: ProbabilitySource,
              logistica: ProbabilitySource) -> dict[str, Estrategia]:
    """Arms A-D of the spec, in the order they are reported."""
    def combinado(fonte: ProbabilitySource) -> Estrategia:
        return estrategia_de_pesos_iguais(seletor_combinado(nota, fonte, setores, PESO_QML), setores)

    bracos = {"Composite + QML score": combinado(qml),
              "QML top 15": estrategia_de_pesos_iguais(seletor_por_probabilidade(qml, setores), setores),
              "Composite + logistic score": combinado(logistica)}
    for seed in range(PLACEBOS):
        bracos[f"Composite + shuffled score {seed}"] = combinado(Embaralhada(qml, seed))
    return bracos


def bracos_de_atribuicao(nota, setores: Mapping[str, str], qml: ProbabilitySource, logistica: ProbabilitySource,
                         cdi: pd.Series, placebos: range = range(3)) -> dict[str, Estrategia]:
    """The frozen Meio-Kelly arm inside the Composite's names and its controls."""
    seletor = seletor_do_composite(nota, setores)
    fontes: dict[str, ProbabilitySource] = {"QML-Composite (Meio-Kelly)": qml, "No signal (p=0.5)": SemSinal(qml),
                                            "Anti-signal (1-p)": AntiSinal(qml),
                                            "Logistic (Meio-Kelly)": logistica}
    for seed in placebos:
        fontes[f"Shuffled P(up) {seed}"] = Embaralhada(qml, seed)
    return {nome: qml_strategy(fonte, CFG_KELLY, setores, cdi, selector=seletor, tilt=1.0)
            for nome, fonte in fontes.items()}


def _opcao(argv: Sequence[str], nome: str, padrao: str) -> str:
    return argv[argv.index(nome) + 1] if nome in argv else padrao


def main(argv: list[str]) -> int:
    inicio = time.perf_counter()
    atribuicao = "--atribuicao" in argv
    fase = int(_opcao(argv, "--fase", "0"))
    lago = Path(_opcao(argv, "--lago", str(LAGO)))
    placebos = range(*(int(v) for v in _opcao(argv, "--placebos", "0,3").split(",")))
    modo = "atribuicao-" if atribuicao else ""
    saida_txt = RAIZ / "docs" / f"qml-g3-{modo}2026-11-10-f{fase}.txt"
    saida_csv = RAIZ / "docs" / "dados" / f"qml-g3-{modo}f{fase}.csv"
    with tempfile.TemporaryDirectory(prefix="qml-g3-") as temporaria:
        entradas = Path(temporaria)
        somas = congelar(lago, entradas, ARQUIVOS)
        dados, _mensal, nota, _ultima, cdi = _carregar(entradas)
    precos_todos = dados.precos.loc[:ULTIMA_SESSAO]
    precos = precos_todos.iloc[fase:]
    acoes = _acoes(precos)
    cdi = cdi.loc[:ULTIMA_SESSAO]
    quadro = pd.read_parquet(CACHE)
    qml = ProbabilidadesCongeladas(quadro)
    if CACHE_LOGISTICA.exists():
        logit = pd.read_parquet(CACHE_LOGISTICA)
    else:
        logit = probabilidades_logisticas(precos_todos.ffill(limit=5).pct_change().iloc[1:], list(quadro.index),
                                          QMLConfig())
        logit.to_parquet(CACHE_LOGISTICA)
    logistica = ProbabilidadesCongeladas(logit)
    setores = dict(dados.setores)
    bracos = (bracos_de_atribuicao(nota, setores, qml, logistica, cdi, placebos) if atribuicao
              else bracos_g3(nota, setores, qml, logistica))
    curvas: dict[str, ResultadoBacktest] = {
        COMPOSITE: _rodar(acoes, dados.universo, estrategia_por_nota(nota, setores), COMPOSITE, inicio)}
    for nome, estrategia in bracos.items():
        curvas[nome] = _rodar(acoes, dados.universo, estrategia, nome, inicio)
    bova = referencias(precos, CFG_BT)["BOVA11"]
    rf = cdi.reindex(curvas[COMPOSITE].valores.index).fillna(0.0)
    tabela = _tabela(curvas, bova.valores, rf, curvas[COMPOSITE].valores)
    tabela.loc["BOVA11"] = _linha(bova.valores, bova.valores, rf)
    tabela.insert(0, "Fase", fase)
    recusas = {nome: len(curva.recusas) + len(curva.falhas) for nome, curva in curvas.items()}
    with saida_txt.open("w", encoding="utf-8", newline="\n") as saida, contextlib.redirect_stdout(saida):
        print(f"G3 {'attribution controls' if atribuicao else 'arms A-D'} "
              f"(spec docs/superpowers/specs/2026-11-10-qml-g3-design.md). Lake exports sha256: {somas}")
        print(f"First decision {acoes.index[CFG_BT.janela_treino]:%d/%m/%Y}; last session "
              f"{acoes.index[-1]:%d/%m/%Y}; P(up) cache rows {len(quadro)}; refusals + failures {recusas}.")
        print(tabela.to_string(float_format=lambda x: f"{x:.4f}"))
    tabela.to_csv(saida_csv, encoding="utf-8", lineterminator="\n")
    print(tabela.drop(columns=["Sortino", "MinTRL vs Comp. (anos)"]).to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n{saida_txt.relative_to(RAIZ)} and {saida_csv.relative_to(RAIZ)} written "
          f"({time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
