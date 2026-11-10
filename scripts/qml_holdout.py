"""
qml_holdout.py - the forward holdout of the QML-Composite (Meio-Kelly) arm: a R$ 10,000 fictitious account
with the rules frozen on 2026-11-09 (spec ``docs/superpowers/specs/2026-11-09-qml-g2-design.md``, Stage 2).

It does NOT forecast portfolio returns. The model returns P(up) per asset; the classical engine turns that
into weights under the fixed rules. This script records what actually happens.

The harness is strategy-agnostic: ``Contexto.pesos`` is the only piece tied to this arm.

    .venv-qpo/Scripts/python -X utf8 scripts/qml_holdout.py --lago <lake> iniciar
    .venv-qpo/Scripts/python -X utf8 scripts/qml_holdout.py --lago <lake> rodar
    .venv-qpo/Scripts/python -X utf8 scripts/qml_holdout.py relatorio

Files (under ``simulador/qml_holdout/``): ``estado.json``, ``ordens.csv``, ``marcas.csv``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

INICIO = pd.Timestamp("2026-11-09")      # the first session on or after the pre-registration
CAPITAL = 10_000.0
CUSTO_BPS = 10.0
PASSO = 63                                # the arm's rebalance: 63 sessions
PASTA = RAIZ / "simulador" / "qml_holdout"
ESTADO = PASTA / "estado.json"
ORDENS = PASTA / "ordens.csv"
MARCAS = PASTA / "marcas.csv"


# --- pure helpers (tested without the lake) ------------------------------------------------------------


def primeira_sessao(pregoes: pd.DatetimeIndex, inicio: pd.Timestamp = INICIO) -> pd.Timestamp:
    """The first session on or after ``inicio``; raises if there is none yet."""
    sessoes = pregoes[pregoes >= inicio]
    if sessoes.empty:
        raise ValueError(f"no session on or after {inicio:%d/%m/%Y}")
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
    """Mark-to-market value of a holding."""
    return float((cotas * precos.reindex(cotas.index)).sum())


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


def montar_contexto(lago: Path) -> Contexto:
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
    predictor = QMLPredictor(retornos, QMLConfig(), log=print)
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
                                  selector=seletor, tilt=1.0)
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


def iniciar(lago: Path, capital: float = CAPITAL) -> dict:
    """Open the account on the first session on or after INICIO and buy the target book."""
    if ESTADO.exists():
        raise FileExistsError(f"{ESTADO} exists: the holdout was already started")
    PASTA.mkdir(parents=True, exist_ok=True)
    ctx = montar_contexto(lago)
    dia = primeira_sessao(ctx.precos.index)
    alvo = ctx.pesos(dia)
    cotas = comprar(alvo, ctx.precos.loc[dia], capital)
    estado = {"inicio": str(dia.date()), "capital": capital, "ultimo_rebalanceio": str(dia.date()),
              "custo_bps": CUSTO_BPS, "passo": PASSO, "estrategia": "QML-Composite (Meio-Kelly)"}
    ESTADO.write_text(json.dumps(estado, indent=2), encoding="utf-8")
    _append(ORDENS, pd.DataFrame({"data": dia, "ticker": alvo.index, "peso": alvo.to_numpy()}))
    _append(MARCAS, pd.DataFrame({"data": [dia], "valor": [valorizar(cotas, ctx.precos.loc[dia])]}))
    return estado


def rodar(lago: Path) -> dict:
    """Mark to market at every new session and rebalance when the 63-session step is due."""
    if not ESTADO.exists():
        raise FileNotFoundError("the holdout was not started: run `iniciar` first")
    estado = json.loads(ESTADO.read_text(encoding="utf-8"))
    ctx = montar_contexto(lago)
    marcas = pd.read_csv(MARCAS, parse_dates=["data"])
    ordens = pd.read_csv(ORDENS, parse_dates=["data"])
    ultima_marca = marcas["data"].max()
    novas = ctx.precos.index[ctx.precos.index > ultima_marca]
    if len(novas) == 0:
        return estado
    alvo_anterior = ordens[ordens["data"] == ordens["data"].max()].set_index("ticker")["peso"]
    cotas = comprar(alvo_anterior, ctx.precos.loc[ultima_marca], float(marcas["valor"].iloc[-1]), 0.0)
    ultimo_reb = pd.Timestamp(estado["ultimo_rebalanceio"])
    for dia in novas:
        preco = ctx.precos.loc[dia]
        if rebalanceio_devido(ctx.precos.index, dia, ultimo_reb, int(estado["passo"])):
            alvo = ctx.pesos(dia)
            cotas, giro = rebalancear(cotas, alvo, preco)
            _append(ORDENS, pd.DataFrame({"data": dia, "ticker": alvo.index, "peso": alvo.to_numpy()}))
            estado["ultimo_rebalanceio"] = str(dia.date())
            estado["ultimo_giro"] = round(giro, 4)
            ultimo_reb = dia
        _append(MARCAS, pd.DataFrame({"data": [dia], "valor": [valorizar(cotas, preco)]}))
    ESTADO.write_text(json.dumps(estado, indent=2), encoding="utf-8")
    return estado


def relatorio(lago: Path) -> str:
    """The holdout's own numbers: CAGR, volatility, drawdown and the comparison with the Composite and CDI."""
    marcas = pd.read_csv(MARCAS, parse_dates=["data"]).set_index("data")["valor"]
    if len(marcas) < 2:
        return f"holdout with {len(marcas)} mark(s): nothing to report yet"
    ctx = montar_contexto(lago)
    cdi = ctx.cdi.reindex(marcas.index).ffill().fillna(0.0)
    retornos = marcas.pct_change().dropna()
    anos = len(retornos) / 252
    cagr = (marcas.iloc[-1] / marcas.iloc[0]) ** (1 / anos) - 1 if anos > 0 else float("nan")
    vol = retornos.std() * (252 ** 0.5)
    mdd = float((marcas / marcas.cummax() - 1).min())
    return (f"QML-Composite (Meio-Kelly) holdout - start {marcas.index.min():%d/%m/%Y}, "
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
    comando = argv[0] if argv else "relatorio"
    if comando == "iniciar":
        print(iniciar(lago))
    elif comando == "rodar":
        print(rodar(lago))
    else:
        print(relatorio(lago))
