"""
qml_avaliar.py - the QML arm of the QPO, on the test branch only (never production).

The quantum circuit is the brain: a PennyLane variational classifier reads six
public, cross-sectionally standardized features of each asset and returns P(up)
over the next 63 sessions. The classical CVXPY engine is the muscle: the
probabilities become an expected-return tilt and ``core.optimization.otimizar``
allocates the money under the fixed rules (20% per name, 30% per sector, 40%
ex-ante volatility, no leverage).

The run is descriptive (as ``scripts/comparacao_estrategias.py``): the same lake
exports, engine, dates, universe, costs and rebalances for every arm. It is not a
pre-registered reading and does not change N.

    .venv-qpo/Scripts/python -X utf8 scripts/qml_avaliar.py [LAKE_EXPORTS_FOLDER]
    .venv-qpo/Scripts/python -X utf8 scripts/qml_avaliar.py --sem-qubo
    .venv-qpo/Scripts/python -X utf8 scripts/qml_avaliar.py --fases 0 21 42

Writes docs/qml-2026-11-09.txt and docs/dados/qml-comparacao.csv.
"""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core.backtest import Congelamento, ResultadoBacktest, walk_forward_dinamico  # noqa: E402
from core.cdi_lago import daily_cdi  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.config import ConfigOtimizacao, DIAS_UTEIS_ANO, limites_por_setor  # noqa: E402
from core.estimation import estimar_momentos  # noqa: E402
from core.metrics import comparar_benchmark, momentos_sharpe, resumo_desempenho, retornos_de_valores  # noqa: E402
from core.optimization import otimizar  # noqa: E402
from core.qml import (MODEL_VERSION, QMLConfig, QMLPredictor, estrategia_media_variancia_no_universo,  # noqa: E402
                      qml_strategy)
from core.qpo import CFG_MAX_SHARPE, K, qubo_selector, selection_then_max_sharpe  # noqa: E402
from core.qubo_universo import FULL_UNIVERSE, SECTOR_CAP, anneal_solver  # noqa: E402
from core.robustez import anos_minimos_para_psr, psr_da_diferenca  # noqa: E402
from core.sinais import NOMES, POR_SETOR, carteira_da_nota, estrategia_conservadora  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, qubo_universo_avaliar  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, referencias  # noqa: E402,F401
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT, CFG_KELLY  # noqa: E402
from scripts.curvas_ao_vivo import ARQUIVOS, LAGO, congelar  # noqa: E402
from scripts.qpo_avaliar import CONSERVADORA, check_aligned  # noqa: E402

SAIDA = RAIZ / "docs" / "qml-2026-11-09.txt"
CURVAS_CSV = RAIZ / "docs" / "dados" / "qml-comparacao.csv"
CURVAS_PARQUET = RAIZ / "docs" / "dados" / "qml-curvas.parquet"
CACHE = RAIZ / "docs" / "dados" / f"qml-probabilidades-v{MODEL_VERSION}.parquet"
FASE_PADRAO = 0
TILT_PADRAO = 1.0
COMPOSITE = "Composite"
QUBO = "Quântica (QUBO)"
QML_COMPOSITE = "QML-Composite (Máx. Sharpe)"
QML_COMPOSITE_KELLY = "QML-Composite (Meio-Kelly)"
QML_KELLY = "QML-Meio-Kelly (Máx. Sharpe)"
QML_ALL = "QML-Universo (Máx. Sharpe)"
MV_COMPOSITE = "Média-variância no Composite (Meio-Kelly)"
COLUNAS_METRICAS = ("Retorno", "CAGR", "Sharpe", "Sortino", "Alfa a.a.", "t alfa", "MDD", "Vol", "Nomes")


def seletor_do_composite(
    nota: Callable[[pd.Timestamp, Sequence[str]], pd.Series],
    setores: Mapping[str, str],
    nomes: int = NOMES,
    por_setor: int = POR_SETOR,
) -> Callable[[pd.Timestamp, pd.DataFrame], list[str]]:
    """The Composite's 15 names of the decision month (the 12 stocks + 3 BDRs of the screen's book)."""
    def _select(data: pd.Timestamp, treino: pd.DataFrame) -> list[str]:
        mes = data.to_period("M").to_timestamp()
        return list(carteira_da_nota(nota(mes, list(treino.columns)).dropna(), setores, nomes, por_setor).index)
    return _select


def seletor_kelly(
    setores: Mapping[str, str],
    cfg: ConfigOtimizacao,
    rf_diaria: pd.Series,
) -> Callable[[pd.Timestamp, pd.DataFrame], list[str]]:
    """The names the half-Kelly (lambda = 2) mean-variance book holds in that month."""
    def _select(data: pd.Timestamp, treino: pd.DataFrame) -> list[str]:
        nomes = [papel for papel in treino.columns if setores.get(papel)]
        t = treino[nomes]
        momentos = estimar_momentos(t, shrink_mu=True)
        vigente = rf_diaria.asof(data)
        rf = cfg.taxa_livre_risco if pd.isna(vigente) else float(vigente) * DIAS_UTEIS_ANO
        configuracao = replace(cfg, taxa_livre_risco=rf,
                               grupos=limites_por_setor({papel: setores[papel] for papel in nomes}, 0.30))
        pesos = otimizar(momentos.mu, momentos.cov, configuracao).pesos
        return list(pesos[pesos > 1e-8].index)
    return _select


def seletor_combinado(
    nota: Callable[[pd.Timestamp, Sequence[str]], pd.Series],
    fonte,
    setores: Mapping[str, str],
    peso_qml: float,
    nomes: int = NOMES,
    por_setor: int = POR_SETOR,
) -> Callable[[pd.Timestamp, pd.DataFrame], list[str]]:
    """The Composite score plus a QML conviction tilt, then the Composite's book rule (sector and issuer caps)."""
    def _select(data: pd.Timestamp, treino: pd.DataFrame) -> list[str]:
        mes = data.to_period("M").to_timestamp()
        base = nota(mes, list(treino.columns)).dropna()
        if base.empty:
            return []
        probabilidades = fonte.predict(data).reindex(base.index).fillna(0.5)
        z_composite = (base - base.mean()) / (base.std(ddof=0) or 1.0)
        z_qml = 2.0 * probabilidades - 1.0
        desvio = z_qml.std(ddof=0)
        z_qml = (z_qml - z_qml.mean()) / desvio if desvio > 0 else z_qml * 0.0
        combinado = z_composite + peso_qml * z_qml
        return list(carteira_da_nota(combinado, setores, nomes, por_setor).index)
    return _select


def estrategia_combinada(estrategias: Sequence, pesos: Sequence[float]) -> Callable[[pd.DataFrame], pd.Series]:
    """A fixed convex combination of target weights: the fixed caps are linear, so feasibility survives."""
    if len(estrategias) != len(pesos) or abs(sum(pesos) - 1.0) > 1e-9:
        raise ValueError("one weight per strategy, summing to 1")
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        colunas = treino.columns
        total = sum(peso * estrategia(treino).reindex(colunas).fillna(0.0)
                    for estrategia, peso in zip(estrategias, pesos))
        return total[total > 1e-8]
    return _estrategia


def estrategia_selecao_qml(fonte, setores: Mapping[str, str], selector: Callable,
                           nomes: int = 12, por_setor: int = POR_SETOR) -> Callable[[pd.DataFrame], pd.Series]:
    """The Composite's pool, the same 12-name size, the names with the highest QML P(up), equal weights."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        data = treino.index[-1]
        candidatos = [papel for papel in selector(data, treino) if papel in treino.columns]
        probabilidades = fonte.predict(data).reindex(candidatos).dropna()
        pesos = carteira_da_nota(probabilidades, setores, nomes, por_setor)
        if pesos.empty:
            raise ValueError(f"{data:%Y-%m-%d}: no name selected")
        return pesos
    return _estrategia


def seletor_qml_12(fonte, setores: Mapping[str, str], selector: Callable,
                   nomes: int = 12, por_setor: int = POR_SETOR) -> Callable:
    """The same QML selection, exposed as a universe selector for the classical optimizer."""
    def _select(data: pd.Timestamp, treino: pd.DataFrame) -> list[str]:
        candidatos = [papel for papel in selector(data, treino) if papel in treino.columns]
        probabilidades = fonte.predict(data).reindex(candidatos).dropna()
        return list(carteira_da_nota(probabilidades, setores, nomes, por_setor).index)
    return _select


def _linha(valores: pd.Series, bova: pd.Series, rf: pd.Series,
           composite: pd.Series | None = None) -> dict[str, float]:
    """One table row: the user's metrics plus the volatility and the number of names held."""
    resumo = resumo_desempenho(valores, rf)
    capm = comparar_benchmark(valores, bova, rf)
    linha = {
        "Retorno": resumo.retorno_total,
        "CAGR": resumo.cagr,
        "Sharpe": resumo.sharpe,
        "Sortino": resumo.sortino,
        "Alfa a.a.": capm.alfa_anual,
        "t alfa": capm.t_stat_alfa,
        "MDD": resumo.max_drawdown,
        "Vol": resumo.volatilidade,
        "Nomes": float("nan"),
    }
    if composite is not None:
        contra = comparar_benchmark(valores, composite, rf)
        diferenca = retornos_de_valores(valores) - retornos_de_valores(composite)
        linha.update({"Alfa vs Comp.": contra.alfa_anual, "t alfa vs Comp.": contra.t_stat_alfa,
                      "P[> Comp.]": psr_da_diferenca(valores, composite),
                      "MinTRL vs Comp. (anos)": anos_minimos_para_psr(momentos_sharpe(diferenca))})
    return linha


def _tabela(curvas: Mapping[str, ResultadoBacktest], bova: pd.Series, rf: pd.Series,
            composite: pd.Series) -> pd.DataFrame:
    linhas = {}
    for nome, resultado in curvas.items():
        linhas[nome] = _linha(resultado.valores, bova, rf, None if nome == COMPOSITE else composite)
        linhas[nome]["Nomes"] = float((resultado.pesos_alvo > 1e-8).sum(axis=1).mean())
    return pd.DataFrame(linhas).T.reindex(columns=list(COLUNAS_METRICAS)
                                          + ["Alfa vs Comp.", "t alfa vs Comp.", "P[> Comp.]",
                                             "MinTRL vs Comp. (anos)"])


def _rodar(acoes: pd.DataFrame, universo: pd.DataFrame, estrategia, nome: str, inicio: float,
           cfg_bt=CFG_BT) -> ResultadoBacktest:
    resultado = walk_forward_dinamico(acoes, universo, estrategia, cfg_bt, nome, Congelamento())
    print(f"  {nome}: {len(resultado.recusas)} refusals, {len(resultado.falhas)} failures, "
          f"{time.perf_counter() - inicio:.0f} s", flush=True)
    return resultado


class ProbabilidadesCongeladas:
    """P(up) of the cached matrix as of the decision session (a freeze-day shift uses the prior session)."""

    def __init__(self, quadro: pd.DataFrame) -> None:
        self.quadro = quadro[~quadro.index.duplicated(keep="last")].sort_index()

    def predict(self, date: pd.Timestamp) -> pd.Series:
        if self.quadro.empty:
            return pd.Series(dtype=float)
        posicao = int(self.quadro.index.searchsorted(date, side="right")) - 1
        if posicao < 0:
            return pd.Series(dtype=float)
        return self.quadro.iloc[posicao].dropna()


def _impressao_da_config(config: QMLConfig) -> dict[str, object]:
    """Everything that changes P(up): the model version plus every field of the config.

    A field left at ``None`` is an option that is switched off; it stays out, so the caches written
    before the option existed remain valid.
    """
    campos = {chave: valor for chave, valor in asdict(config).items() if valor is not None}
    return {"model_version": MODEL_VERSION, "config": campos}


def _ler_cache(cache: Path, config: QMLConfig) -> pd.DataFrame:
    """
    The stored P(up), or an empty frame when it cannot be trusted: a corrupt file, a missing sidecar,
    or a sidecar written by a different model version or configuration.
    """
    if not cache.exists():
        return pd.DataFrame()
    lado = cache.with_suffix(".json")
    try:
        guardado = json.loads(lado.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"  QML: cache {cache.name} has no readable sidecar; rebuilding it", flush=True)
        return pd.DataFrame()
    if guardado != _impressao_da_config(config):
        print(f"  QML: cache {cache.name} came from another configuration; rebuilding it", flush=True)
        return pd.DataFrame()
    try:
        return pd.read_parquet(cache)
    except (OSError, ValueError) as erro:                       # a truncated write from an interrupted run
        print(f"  QML: cache {cache.name} is unreadable ({type(erro).__name__}); rebuilding it", flush=True)
        return pd.DataFrame()


def _gravar_cache(quadro: pd.DataFrame, cache: Path, config: QMLConfig) -> None:
    """Parquet first, sidecar last: an interrupted write leaves no valid sidecar next to a half file."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    limpo = quadro[~quadro.index.duplicated(keep="last")].sort_index()
    limpo.to_parquet(cache)
    cache.with_suffix(".json").write_text(json.dumps(_impressao_da_config(config), sort_keys=True),
                                          encoding="utf-8")


def memoizar_previsoes(predictor: QMLPredictor, datas: Sequence[pd.Timestamp], cache: Path, *,
                       gravar_a_cada: int, rotulo: str, unidade: str) -> pd.DataFrame:
    """
    P(up) on ``datas``, read from ``cache`` where it is already there and predicted (causally, by the
    predictor) where it is not. The file is a memo over dates: it is rewritten every ``gravar_a_cada`` new
    dates, so an interrupted run resumes instead of starting over.
    """
    quadro = _ler_cache(cache, predictor.config)
    faltam = [data for data in datas if data not in quadro.index]
    if faltam:
        print(f"  {rotulo}: {len(faltam)} {unidade} to predict (cache has {len(quadro.index)})", flush=True)
    for indice, data in enumerate(faltam, start=1):
        linha = predictor.predict(data)
        if not linha.empty:
            nova = linha.to_frame().T
            nova.index = pd.DatetimeIndex([data])
            quadro = pd.concat([quadro[quadro.index != data], nova])
        if indice % gravar_a_cada == 0 or indice == len(faltam):
            _gravar_cache(quadro.sort_index(), cache, predictor.config)
            print(f"  {rotulo}: {indice}/{len(faltam)} {unidade} predicted", flush=True)
    quadro = quadro.sort_index()
    _gravar_cache(quadro, cache, predictor.config)
    return quadro


def _probabilidades(predictor: QMLPredictor, acoes: pd.DataFrame, cfg_bt=CFG_BT,
                    cache: Path | None = None) -> pd.DataFrame:
    """P(up) on every marked rebalance (and the session after it, where a freeze can move the rebalance), for the
    exact dates of the phase being run: a shifted phase gets its own predictions, never a neighbour's. The file is a
    memo over dates, so a rerun only fills the dates it lacks. ``cache`` resolves at call time, so a test can
    redirect it without touching the real cache.
    """
    cache = CACHE if cache is None else cache
    datas: list[pd.Timestamp] = []
    for posicao in range(cfg_bt.janela_treino, len(acoes), cfg_bt.janela_teste):
        datas.append(acoes.index[posicao])
        if posicao + 1 < len(acoes):
            datas.append(acoes.index[posicao + 1])
    return memoizar_previsoes(predictor, datas, cache, gravar_a_cada=10, rotulo="QML", unidade="decision dates")


def _armar(args: Sequence[str]) -> tuple[Path, bool, tuple[int, ...], bool, bool]:
    """(lake folder, skip QUBO, phases, probabilities only, quick arms)."""
    lago = LAGO
    sem_qubo = "--sem-qubo" in args
    so_probabilidades = "--so-probabilidades" in args
    rapido = "--rapido" in args
    fases: tuple[int, ...] = (FASE_PADRAO,)
    restantes = [a for a in args if a not in ("--sem-qubo", "--so-probabilidades", "--rapido")]
    if "--fases" in restantes:
        posicao = restantes.index("--fases")
        valores = restantes[posicao + 1:]
        if not valores:
            raise ValueError("--fases needs at least one phase, e.g. --fases 0 21 42")
        fases = tuple(int(valor) for valor in valores)
        if any(fase < 0 for fase in fases):
            raise ValueError(f"phases must be >= 0: {fases}")
        restantes = restantes[:posicao]
    if restantes:
        lago = Path(restantes[0])
    return lago, sem_qubo, fases, so_probabilidades, rapido


def main(argv: list[str]) -> int:
    inicio = time.perf_counter()
    lago, sem_qubo, fases, so_probabilidades, rapido = _armar(argv)
    with tempfile.TemporaryDirectory(prefix="qml-") as temporaria:
        entradas = Path(temporaria)
        somas = congelar(lago, entradas, ARQUIVOS)
        avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = entradas
        qubo_universo_avaliar.FOLDER = qubo_universo_avaliar.PASTA_CVM = entradas
        dados = avaliar_robustez.carregar()
        mensal = avaliar_conservadora.carregar_mensal(entradas)
        nota, _ = qubo_universo_avaliar._value_score(mensal, com_crescimento=True)
        ultima = dados.precos.index.max()
        cdi = daily_cdi(ANOS_DE_CDI, ultima.date(), path=entradas / "cdi_daily.parquet")

        seletores = {"composite": seletor_do_composite(nota, dados.setores),
                     "kelly": seletor_kelly(dados.setores, CFG_KELLY, cdi)}
        predictor = QMLPredictor(dados.precos.ffill(limit=5).pct_change().iloc[1:], QMLConfig(),
                                 log=print)

        with SAIDA.open("w", encoding="utf-8", newline="\n") as saida, contextlib.redirect_stdout(saida):
            print(f"QML arm of the QPO (descriptive, test branch; not a pre-registered reading). "
                  f"Lake exports from {lago.name} (sha256): {somas}")
            print(f"Quantum model: {QMLConfig().n_qubits} qubits, {QMLConfig().n_layers} layers, "
                  f"{QMLConfig().epochs} epochs max, horizon {QMLConfig().horizon} sessions; "
                  f"trained once per two-year block on labels realized before 1 January of the block.")
            tabelas = []
            quadros = []
            for fase in fases:
                precos = dados.precos.loc[:ultima].iloc[fase:]
                acoes = _acoes(precos)
                inicio_das_decisoes = acoes.index[min(CFG_BT.janela_treino, len(acoes) - 1)]
                print(f"\n=== Phase {fase}: first decision {inicio_das_decisoes:%d/%m/%Y}, "
                      f"last session {acoes.index[-1]:%d/%m/%Y} ===", flush=True)
                fonte = ProbabilidadesCongeladas(_probabilidades(predictor, acoes))
                if so_probabilidades:
                    print(f"  QML: {len(fonte.quadro)} dates cached; stopping before the backtests", flush=True)
                    return 0
                cfg_ms = replace(CFG_MAX_SHARPE, taxa_livre_risco=float((1 + cdi.iloc[-1]) ** DIAS_UTEIS_ANO - 1))
                curvas: dict[str, ResultadoBacktest] = {
                    COMPOSITE: _rodar(acoes, dados.universo, estrategia_por_nota(nota, dados.setores),
                                      COMPOSITE, inicio),
                    CONSERVADORA: _rodar(acoes, dados.universo,
                                         estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores),
                                         CONSERVADORA, inicio),
                }
                if not sem_qubo:
                    qubo = selection_then_max_sharpe(
                        qubo_selector(dados.setores, FULL_UNIVERSE, K,
                                      anneal_solver(dados.setores, SECTOR_CAP, {}), {}),
                        mensal.retornos, mensal.proventos, dados.setores, cfg_ms, cdi, None)
                    curvas[QUBO] = _rodar(acoes, dados.universo, qubo, QUBO, inicio)
                curvas[QML_COMPOSITE] = _rodar(
                    acoes, dados.universo,
                    qml_strategy(fonte, cfg_ms, dados.setores, cdi,
                                 selector=seletores["composite"], tilt=TILT_PADRAO),
                    QML_COMPOSITE, inicio)
                qml_kelly = qml_strategy(fonte, CFG_KELLY, dados.setores, cdi,
                                         selector=seletores["composite"], tilt=TILT_PADRAO)
                curvas[QML_COMPOSITE_KELLY] = _rodar(acoes, dados.universo, qml_kelly,
                                                     QML_COMPOSITE_KELLY, inicio)
                curvas["QML-Jorion-Composite (Meio-Kelly)"] = _rodar(
                    acoes, dados.universo,
                    qml_strategy(fonte, CFG_KELLY, dados.setores, cdi,
                                 selector=seletores["composite"], tilt=0.5, base="jorion"),
                    "QML-Jorion-Composite (Meio-Kelly)", inicio)
                if not rapido:
                    curvas[QML_KELLY] = _rodar(
                        acoes, dados.universo,
                        qml_strategy(fonte, cfg_ms, dados.setores, cdi,
                                     selector=seletores["kelly"], tilt=TILT_PADRAO),
                        QML_KELLY, inicio)
                    curvas[QML_ALL] = _rodar(
                        acoes, dados.universo,
                        qml_strategy(fonte, cfg_ms, dados.setores, cdi, selector=None, tilt=TILT_PADRAO),
                        QML_ALL, inicio)
                for peso in ((0.5,) if rapido else (0.25, 0.5, 1.0)):
                    nome = f"QML+Composite (λ={peso:g}, Máx. Sharpe)"
                    curvas[nome] = _rodar(
                        acoes, dados.universo,
                        qml_strategy(fonte, cfg_ms, dados.setores, cdi,
                                     selector=seletor_combinado(nota, fonte, dados.setores, peso),
                                     tilt=TILT_PADRAO),
                        nome, inicio)
                qml_jorion = qml_strategy(fonte, cfg_ms, dados.setores, cdi, selector=seletores["composite"],
                                          tilt=0.5, base="jorion")
                curvas["QML-Jorion-Composite (Máx. Sharpe)"] = _rodar(
                    acoes, dados.universo, qml_jorion, "QML-Jorion-Composite (Máx. Sharpe)", inicio)
                curvas["QML-Ensemble (Meio-Kelly + Jorion)"] = _rodar(
                    acoes, dados.universo, estrategia_combinada([qml_kelly, qml_jorion], [0.5, 0.5]),
                    "QML-Ensemble (Meio-Kelly + Jorion)", inicio)
                curvas["QML-Ensemble (60/40 Jorion)"] = _rodar(
                    acoes, dados.universo, estrategia_combinada([qml_kelly, qml_jorion], [0.4, 0.6]),
                    "QML-Ensemble (60/40 Jorion)", inicio)
                curvas["QML-Seleção-12 (Composite, pesos iguais)"] = _rodar(
                    acoes, dados.universo,
                    estrategia_selecao_qml(fonte, dados.setores, seletores["composite"]),
                    "QML-Seleção-12 (Composite, pesos iguais)", inicio)
                curvas["QML-Seleção-12 (Máx. Sharpe)"] = _rodar(
                    acoes, dados.universo,
                    qml_strategy(fonte, cfg_ms, dados.setores, cdi,
                                 selector=seletor_qml_12(fonte, dados.setores, seletores["composite"])),
                    "QML-Seleção-12 (Máx. Sharpe)", inicio)
                if not rapido:
                    curvas["QML-Jorion+Composite (λ=0.5)"] = _rodar(
                        acoes, dados.universo,
                        qml_strategy(fonte, cfg_ms, dados.setores, cdi,
                                     selector=seletor_combinado(nota, fonte, dados.setores, 0.5),
                                     tilt=0.5, base="jorion"),
                        "QML-Jorion+Composite (λ=0.5)", inicio)
                curvas[MV_COMPOSITE] = _rodar(
                    acoes, dados.universo,
                    estrategia_media_variancia_no_universo(CFG_KELLY, dados.setores, cdi,
                                                           selector=seletores["composite"]),
                    MV_COMPOSITE, inicio)
                if not rapido:
                    for tilt in (0.5, 2.0):
                        nome = f"QML-Composite (Máx. Sharpe, tilt {tilt:g})"
                        curvas[nome] = _rodar(
                            acoes, dados.universo,
                            qml_strategy(fonte, cfg_ms, dados.setores, cdi,
                                         selector=seletores["composite"], tilt=tilt), nome, inicio)
                bova = referencias(precos, CFG_BT)["BOVA11"]
                valores = {nome: curva.valores for nome, curva in curvas.items()}
                valores["BOVA11"] = bova.valores
                check_aligned(valores)
                quadros.append(pd.DataFrame({"fase": fase, "date": next(iter(valores.values())).index,
                                             **{nome: serie.to_numpy() for nome, serie in valores.items()}}))
                rf = cdi.reindex(next(iter(valores.values())).index).fillna(0.0)
                tabela = _tabela(curvas, bova.valores, rf, curvas[COMPOSITE].valores)
                tabela.loc["BOVA11"] = _linha(bova.valores, bova.valores, rf)
                tabela.loc["BOVA11", "Nomes"] = float("nan")
                tabela.insert(0, "Fase", fase)
                tabelas.append(tabela)
                print(tabela.to_string(float_format=lambda x: f"{x:.4f}"))
                print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)
            saida.write("\n")
            print("Legend: Retorno = total; Alfa a.a./t alfa = CAPM against BOVA11 (Newey-West); "
                  "Alfa vs Comp./t = same regression against the Composite; Nomes = mean names held.")
            final = pd.concat(tabelas)
    final.to_csv(CURVAS_CSV, encoding="utf-8", lineterminator="\n")
    pd.concat(quadros, ignore_index=True).to_parquet(CURVAS_PARQUET, index=False)
    print(f"\n{SAIDA.relative_to(RAIZ)}, {CURVAS_CSV.relative_to(RAIZ)} and {CURVAS_PARQUET.relative_to(RAIZ)} "
          f"written ({time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
