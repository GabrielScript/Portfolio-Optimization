"""
qml_holdout.py - the forward holdouts of the QML-Composite (Meio-Kelly) arm: two R$ 10,000 fictitious
accounts run by the same harness.

    ledoit-wolf  the account frozen on 2026-11-09 (spec ``…/2026-11-09-qml-g2-design.md``): Ledoit-Wolf
                 covariance, legacy model seed. Ledger under ``simulador/qml_holdout/``.
    pca          the second account pre-registered on 2026-11-10 (spec
                 ``…/2026-11-10-qml-holdout-pca-design.md``): PCA factor covariance, model seed pinned to
                 20261110, starting on 2026-11-10. Ledger under ``simulador/qml_holdout_pca/``.

It does NOT forecast portfolio returns. The model returns P(up) per asset; the classical engine turns that
into weights under the fixed rules. This script records what actually happens.

The harness is strategy-agnostic: ``Contexto.pesos`` is the only piece tied to this arm.

    .venv-qpo/Scripts/python -X utf8 scripts/qml_holdout.py --lago <lake> [--conta ledoit-wolf|pca] iniciar
    .venv-qpo/Scripts/python -X utf8 scripts/qml_holdout.py --lago <lake> [--conta ledoit-wolf|pca] rodar
    .venv-qpo/Scripts/python -X utf8 scripts/qml_holdout.py [--conta ledoit-wolf|pca] relatorio

``--conta`` defaults to ``ledoit-wolf``. Each account keeps ``estado.json``, ``ordens.csv`` and
``marcas.csv`` under its own ledger folder.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class SessionNotStarted(ValueError):
    """The lake has no session on or after the account's pre-registered start date yet."""

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

INICIO = pd.Timestamp("2026-11-09")      # the first session on or after the pre-registration
CAPITAL = 10_000.0
CUSTO_BPS = 10.0
PASSO = 63                                # the arm's rebalance: 63 sessions
PASTA = RAIZ / "simulador" / "qml_holdout"


@dataclass(frozen=True)
class Conta:
    """A fictitious account: its ledger folder, its name and its two model choices."""

    pasta: Path
    estrategia: str
    covariance: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    init_seed: int | None = None
    inicio: pd.Timestamp = INICIO

    def __post_init__(self) -> None:
        object.__setattr__(self, "pasta", Path(self.pasta))
        object.__setattr__(self, "inicio", pd.Timestamp(self.inicio))

    @property
    def estado(self) -> Path:
        return self.pasta / "estado.json"

    @property
    def ordens(self) -> Path:
        return self.pasta / "ordens.csv"

    @property
    def marcas(self) -> Path:
        return self.pasta / "marcas.csv"


def _pca(returns: pd.DataFrame) -> pd.DataFrame:
    """The PCA factor covariance, imported lazily so this module stays light."""
    from core.multivariate import pca_covariance  # noqa: PLC0415
    return pca_covariance(returns)


CONTA_FIXA = Conta(PASTA, "QML-Composite (Meio-Kelly)")
CONTA_PCA = Conta(RAIZ / "simulador" / "qml_holdout_pca",
                  "QML-Composite (Meio-Kelly), PCA covariance",
                  covariance=_pca, init_seed=20261110, inicio=pd.Timestamp("2026-11-10"))
CONTAS = {"ledoit-wolf": CONTA_FIXA, "pca": CONTA_PCA}


# --- pure helpers (tested without the lake) ------------------------------------------------------------


def primeira_sessao(pregoes: pd.DatetimeIndex, inicio: pd.Timestamp = INICIO) -> pd.Timestamp:
    """The first session on or after ``inicio``; raises if there is none yet."""
    sessoes = pregoes[pregoes >= inicio]
    if sessoes.empty:
        raise SessionNotStarted(f"no session on or after {inicio:%d/%m/%Y}")
    return pd.Timestamp(sessoes[0])


def rebalanceio_devido(pregoes: pd.DatetimeIndex, dia: pd.Timestamp, ultimo: pd.Timestamp, passo: int = PASSO) -> bool:
    """True when ``dia`` is at least ``passo`` sessions after ``ultimo``."""
    return int(pregoes.get_loc(dia)) - int(pregoes.get_loc(ultimo)) >= passo


def comprar(pesos: pd.Series, precos: pd.Series, capital: float, custo_bps: float = CUSTO_BPS) -> pd.Series:
    """Shares bought with ``capital`` at ``pesos`` and ``precos``, after ``custo_bps`` per side."""
    disponivel = capital * (1.0 - custo_bps / 10_000.0)
    precos = precos.reindex(pesos.index)
    invalidos = list(precos[(precos <= 0) | precos.isna()].index)
    if invalidos:
        raise ValueError(f"no usable price for {invalidos}")
    return (pesos * disponivel) / precos


def valorizar(cotas: pd.Series, precos: pd.Series) -> float:
    """Mark-to-market value of a holding; a held name without a price is an error, never a silent zero."""
    alinhados = precos.reindex(cotas.index)
    sem_preco = list(alinhados[alinhados.isna() & (cotas != 0)].index)
    if sem_preco:
        raise ValueError(f"no price to mark {sem_preco}")
    return float((cotas * alinhados.fillna(0.0)).sum())


def cotas_em_carteira(ordens: pd.DataFrame, marcas: pd.DataFrame, precos: pd.DataFrame) -> pd.Series:
    """
    The shares held since the last rebalance: its target weights, the account value marked on that
    session (already net of the trade's cost) and that session's prices. Between rebalances the book
    drifts with the prices, so the shares must not be rebuilt from a later mark: that would bring the
    weights back to the target for free every time the ledger runs.
    """
    dia = ordens["data"].max()
    alvo = ordens[ordens["data"] == ordens["data"].max()].set_index("ticker")["peso"]
    valor = float(marcas.loc[marcas["data"] == dia, "valor"].iloc[-1])
    return comprar(alvo, precos.loc[dia], valor, 0.0)


def pesos_do_rebalanceio(cotas: pd.Series, precos: pd.Series) -> pd.Series:
    """Current weights, so the turnover can be computed."""
    valor = valorizar(cotas, precos)
    if valor <= 0:
        return pd.Series(dtype=float)
    return (cotas * precos.reindex(cotas.index)) / valor


def rebalancear(cotas: pd.Series, alvo: pd.Series, precos: pd.Series,
                custo_bps: float = CUSTO_BPS) -> tuple[pd.Series, float]:
    """Trade to ``alvo``, charging ``custo_bps`` per side on the turnover only. Returns (shares, turnover)."""
    valor = valorizar(cotas, precos)
    atual = pesos_do_rebalanceio(cotas, precos).reindex(alvo.index).fillna(0.0)
    giro = float((alvo - atual).abs().sum())
    liquido = valor * (1.0 - custo_bps / 10_000.0 * giro)
    return comprar(alvo, precos, liquido, 0.0), giro


# --- the arm's weights (the only strategy-specific piece) ----------------------------------------------


class Contexto:
    """Frozen lake, the return matrix the strategy expects, and the arm's weight function."""

    def __init__(self, precos: pd.DataFrame, retornos: pd.DataFrame, elegiveis: dict, pesos_fn,
                 cdi: pd.Series, janela: int) -> None:
        self.precos = precos
        self.retornos = retornos
        self.elegiveis = elegiveis
        self.pesos = pesos_fn
        self.cdi = cdi
        self.janela = janela


def montar_contexto(lago: Path, conta: Conta = CONTA_FIXA) -> Contexto:
    """Load the lake exports, train the frozen model and return the weight function.

    The strategy receives exactly what ``walk_forward_dinamico`` gives it: the return matrix of the
    training window, restricted to the month's point-in-time universe, with columns that have a hole
    dropped. Passing prices here is what made the estimator see NaN.
    """
    import tempfile  # noqa: PLC0415

    from core.qml import QMLConfig, QMLPredictor, qml_strategy  # noqa: PLC0415
    from scripts.avaliar_universo_dinamico import CFG_BT, CFG_KELLY  # noqa: PLC0415
    from scripts.curvas_ao_vivo import ARQUIVOS, congelar  # noqa: PLC0415
    from scripts.qml_avaliar import seletor_do_composite  # noqa: PLC0415
    from scripts.qml_g2_avaliar import _carregar  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="qml-holdout-") as temporaria:
        entradas = Path(temporaria)
        congelar(lago, entradas, ARQUIVOS)
        dados, _mensal, nota, _ultima, cdi = _carregar(entradas)
    setores = dict(dados.setores)
    retornos = dados.precos.ffill(limit=5).pct_change().iloc[1:]
    meses = pd.to_datetime(dados.universo["month"])
    elegiveis = dados.universo["security"].groupby(meses).apply(list).to_dict()
    predictor = QMLPredictor(retornos, QMLConfig(init_seed=conta.init_seed), log=print)
    seletor = seletor_do_composite(nota, setores)
    janela = CFG_BT.janela_treino

    def pesos(data: pd.Timestamp) -> pd.Series:
        posicao = int(retornos.index.get_loc(data))
        if posicao < janela:
            raise ValueError(f"{data:%d/%m/%Y}: only {posicao} sessions of history, need {janela}")
        mes = data.to_period("M").to_timestamp()
        nomes = [papel for papel in elegiveis.get(mes, []) if papel in retornos.columns]
        treino = retornos.iloc[posicao - janela:posicao][nomes].dropna(axis=1)
        if treino.empty:
            raise ValueError(f"{data:%d/%m/%Y}: no eligible name with a full training window")
        estrategia = qml_strategy(_FonteDoPredictor(predictor), CFG_KELLY, setores, cdi,
                                  selector=seletor, tilt=1.0, covariance=conta.covariance)
        return estrategia(treino)

    return Contexto(dados.precos, retornos, elegiveis, pesos, cdi, janela)


class _FonteDoPredictor:
    """Adapter: the strategy asks ``predict(date)``; the predictor trains causally and answers."""

    def __init__(self, predictor) -> None:
        self._predictor = predictor

    def predict(self, data: pd.Timestamp) -> pd.Series:
        return self._predictor.predict(data)


# --- the ledger -----------------------------------------------------------------------------------------


def _append(caminho: Path, quadro: pd.DataFrame) -> None:
    quadro.to_csv(caminho, mode="a" if caminho.exists() else "w", header=not caminho.exists(),
                  index=False, encoding="utf-8", lineterminator="\n")


def iniciar(lago: Path, capital: float = CAPITAL, conta: Conta = CONTA_FIXA) -> dict:
    """Open the account on the first session on or after its start date and buy the target book."""
    if conta.estado.exists():
        raise FileExistsError(f"{conta.estado} exists: the holdout was already started")
    conta.pasta.mkdir(parents=True, exist_ok=True)
    ctx = montar_contexto(lago, conta)
    dia = primeira_sessao(ctx.precos.index, conta.inicio)
    alvo = ctx.pesos(dia)
    cotas = comprar(alvo, ctx.precos.loc[dia], capital)
    estado = {"inicio": str(dia.date()), "capital": capital, "ultimo_rebalanceio": str(dia.date()),
              "custo_bps": CUSTO_BPS, "passo": PASSO, "estrategia": conta.estrategia}
    conta.estado.write_text(json.dumps(estado, indent=2), encoding="utf-8")
    _append(conta.ordens, pd.DataFrame({"data": dia, "ticker": alvo.index, "peso": alvo.to_numpy()}))
    _append(conta.marcas, pd.DataFrame({"data": [dia], "valor": [valorizar(cotas, ctx.precos.loc[dia])]}))
    return estado


def rodar(lago: Path, conta: Conta = CONTA_FIXA) -> dict:
    """Mark to market at every new session and rebalance when the 63-session step is due."""
    if not conta.estado.exists():
        raise FileNotFoundError("the holdout was not started: run `iniciar` first")
    estado = json.loads(conta.estado.read_text(encoding="utf-8"))
    ctx = montar_contexto(lago, conta)
    marcas = pd.read_csv(conta.marcas, parse_dates=["data"])
    ordens = pd.read_csv(conta.ordens, parse_dates=["data"])
    ultima_marca = marcas["data"].max()
    novas = ctx.precos.index[ctx.precos.index > ultima_marca]
    if len(novas) == 0:
        return estado
    precos = ctx.precos.ffill()               # a session without a trade keeps the last close
    cotas = cotas_em_carteira(ordens, marcas, precos)
    ultimo_reb = pd.Timestamp(estado["ultimo_rebalanceio"])
    for dia in novas:
        preco = precos.loc[dia]
        if rebalanceio_devido(ctx.precos.index, dia, ultimo_reb, int(estado["passo"])):
            alvo = ctx.pesos(dia)
            cotas, giro = rebalancear(cotas, alvo, preco)
            _append(conta.ordens, pd.DataFrame({"data": dia, "ticker": alvo.index, "peso": alvo.to_numpy()}))
            estado["ultimo_rebalanceio"] = str(dia.date())
            estado["ultimo_giro"] = round(giro, 4)
            ultimo_reb = dia
        _append(conta.marcas, pd.DataFrame({"data": [dia], "valor": [valorizar(cotas, preco)]}))
    conta.estado.write_text(json.dumps(estado, indent=2), encoding="utf-8")
    return estado


def relatorio(lago: Path, conta: Conta = CONTA_FIXA) -> str:
    """The holdout's own numbers: CAGR, volatility, drawdown and the comparison with the Composite and CDI."""
    marcas = pd.read_csv(conta.marcas, parse_dates=["data"]).set_index("data")["valor"]
    if len(marcas) < 2:
        return f"holdout with {len(marcas)} mark(s): nothing to report yet"
    ctx = montar_contexto(lago, conta)
    cdi = ctx.cdi.reindex(marcas.index).ffill().fillna(0.0)
    retornos = marcas.pct_change().dropna()
    anos = len(retornos) / 252
    cagr = (marcas.iloc[-1] / marcas.iloc[0]) ** (1 / anos) - 1 if anos > 0 else float("nan")
    vol = retornos.std() * (252 ** 0.5)
    mdd = float((marcas / marcas.cummax() - 1).min())
    return (f"{conta.estrategia} holdout - start {marcas.index.min():%d/%m/%Y}, "
            f"{len(marcas)} marks\n"
            f"  value R$ {marcas.iloc[-1]:,.2f} (from R$ {marcas.iloc[0]:,.2f})\n"
            f"  CAGR {cagr:.2%} | vol {vol:.2%} | MDD {mdd:.2%} | CDI no periodo {(1 + cdi).prod() - 1:.2%}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    lago = None
    if "--lago" in argv:
        indice = argv.index("--lago")
        lago = Path(argv[indice + 1])
        del argv[indice:indice + 2]
    conta = CONTA_FIXA
    if "--conta" in argv:
        indice = argv.index("--conta")
        nome = argv[indice + 1]
        if nome not in CONTAS:
            raise SystemExit(f"unknown account {nome!r}: choose from {', '.join(CONTAS)}")
        conta = CONTAS[nome]
        del argv[indice:indice + 2]
    comando = argv[0] if argv else "relatorio"
    if comando == "iniciar":
        print(iniciar(lago, conta=conta))
    elif comando == "rodar":
        print(rodar(lago, conta=conta))
    else:
        print(relatorio(lago, conta=conta))
