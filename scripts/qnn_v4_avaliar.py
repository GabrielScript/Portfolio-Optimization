"""
qnn_v4_avaliar.py - QNN v4 (``core.qnn``) screened into every classical method (test branch only).

Spec: ``docs/superpowers/specs/2026-11-10-qnn-v4-design.md`` (committed before this script ran).

    .venv-qpo/Scripts/python -X utf8 scripts/qnn_v4_avaliar.py --lago <lake> --probabilidades
    .venv-qpo/Scripts/python -X utf8 scripts/qnn_v4_avaliar.py --lago <lake> --fase 0 [--lote principal|michaud]
    .venv-qpo/Scripts/python -X utf8 scripts/qnn_v4_avaliar.py --relatorio

``--probabilidades`` trains the walk-forward models and stores P(up) on the decision dates of the three phases.
``--fase`` runs one phase's arms from the stored P(up). ``--relatorio`` joins the phases and applies the gates.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core.backtest import Estrategia, ResultadoBacktest  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.hrp_paper import estrategia_hrp_paper  # noqa: E402
from core.metrics import comparar_benchmark, resumo_desempenho  # noqa: E402
from core.multivariate import newey_west_mean_se  # noqa: E402
from core.qml import ProbabilitySource, qml_strategy  # noqa: E402
from core.qnn import (MODEL_VERSION, QNNConfig, Trainer, WalkForwardPredictor, eligible_by_month,  # noqa: E402
                      holder_returns, logistic_trainer, quantum_trainer, screened)
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes  # noqa: E402
from scripts.avaliar_universo_dinamico import CFG_BT, CFG_KELLY  # noqa: E402
from scripts.comparacao_estrategias import otimizadores  # noqa: E402
from scripts.curvas_ao_vivo import ARQUIVOS, LAGO, congelar  # noqa: E402
from scripts.qml_avaliar import CACHE as CACHE_V3  # noqa: E402
from scripts.qml_avaliar import ProbabilidadesCongeladas, _rodar, seletor_do_composite  # noqa: E402
from scripts.qml_g2_avaliar import _carregar  # noqa: E402
from scripts.qml_g3_avaliar import AntiSinal, Embaralhada, SemSinal  # noqa: E402

ULTIMA_SESSAO = pd.Timestamp("2026-11-06")       # the window of DECISOES 318
SEEDS = (0, 1, 2)                                # fixed by the spec
KEEP_SHARE = 0.5
T_SIGNAL = 2.87                                  # 5% two-sided, Bonferroni over 12 methods
T_HOUSE = 2.0
IC_LAGS = 3                                      # dates 21 sessions apart, 63-session returns
DADOS = RAIZ / "docs" / "dados"
RELATORIO = RAIZ / "docs" / "qnn-v4-2026-11-10.txt"
COMPOSITE = "Composite"
MICHAUD = "Reamostrada (Michaud)"
HRP_PAPER = "HRP (distância do paper)"
SUFIXOS = {"qnn": " + QNN", "logistic": " + logistic", "shuffled": " + shuffled", "anti": " + anti"}
TILT = {"v4": "Tilt QNN v4 (Composite 15, Meio-Kelly)", "v3": "Tilt QML v3 stored (Composite 15, Meio-Kelly)",
        "none": "Tilt no signal (Composite 15, Meio-Kelly)", "logistic": "Tilt logistic (Composite 15, Meio-Kelly)",
        "shuffled": "Tilt shuffled (Composite 15, Meio-Kelly)"}


def _cache(nome: str) -> Path:
    return DADOS / f"qnn-v{MODEL_VERSION}-{nome}.parquet"


# --- probabilities ----------------------------------------------------------------------------------------------------


def decision_dates(precos: pd.DataFrame, fases: Sequence[int] = FASES) -> list[pd.Timestamp]:
    """
    For every phase, the price session before each marked rebalance (the last row of the engine's training window)
    and the session after it (read when a frozen session moves the rebalance by one day).
    """
    datas: set[pd.Timestamp] = set()
    for fase in fases:
        indice = _acoes(precos).iloc[fase:].index
        for posicao in range(CFG_BT.janela_treino, len(indice), CFG_BT.janela_teste):
            datas.add(indice[posicao])
            if posicao + 1 < len(indice):
                datas.add(indice[posicao + 1])
    return sorted(datas)


def predict_frame(predictor: WalkForwardPredictor, datas: Sequence[pd.Timestamp]) -> pd.DataFrame:
    """P(up), dates x names; a date without a cross-section gives no row."""
    linhas = {data: linha for data in datas if not (linha := predictor.predict(data)).empty}
    return pd.DataFrame(linhas).T.sort_index()


def rank_ic(probabilidades: pd.DataFrame, futuro: pd.DataFrame) -> pd.Series:
    """Spearman correlation per date between P(up) and the realized forward return (dates without one are skipped)."""
    valores = {}
    for data, linha in probabilidades.iterrows():
        if data not in futuro.index:
            continue
        par = pd.concat([linha, futuro.loc[data]], axis=1, keys=["p", "r"]).dropna()
        if len(par) >= 10 and par["p"].nunique() > 1:
            valores[data] = par["p"].rank().corr(par["r"].rank())
    return pd.Series(valores, dtype=float).sort_index()


def _ic_line(nome: str, ic: pd.Series) -> str:
    erro = newey_west_mean_se(ic, IC_LAGS)
    return (f"  {nome:<14} mean rank IC {ic.mean():+.4f}, Newey-West t {ic.mean() / erro:+.2f}, "
            f"positive on {(ic > 0).mean():.0%} of {len(ic)} dates")


def run_probabilities(lago: Path) -> int:
    import torch  # noqa: PLC0415
    torch.set_num_threads(4)                                             # four physical cores
    inicio = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="qnn-v4-") as temporaria:
        somas = congelar(lago, Path(temporaria), ARQUIVOS)
        dados, _mensal, _nota, _ultima, _cdi = _carregar(Path(temporaria))
    precos = _acoes(dados.precos.loc[:ULTIMA_SESSAO])
    retornos = precos.ffill(limit=5).pct_change().iloc[1:]
    carregador = holder_returns(precos)
    elegiveis = eligible_by_month(dados.universo)
    datas = decision_dates(dados.precos.loc[:ULTIMA_SESSAO])

    def prever(config: QNNConfig, trainer: Trainer) -> pd.DataFrame:
        return predict_frame(WalkForwardPredictor(retornos, carregador, elegiveis, config, trainer, print), datas)

    quadros: dict[str, pd.DataFrame] = {}
    for seed in SEEDS:
        config = QNNConfig(seed=seed)
        print(f"QNN seed {seed} ({time.perf_counter() - inicio:.0f} s)", flush=True)
        quadros[f"seed{seed}"] = prever(config, quantum_trainer(config, print))
    quadros["qnn"] = sum(quadros[f"seed{seed}"] for seed in SEEDS) / len(SEEDS)
    quadros["logistic"] = prever(QNNConfig(), logistic_trainer())
    DADOS.mkdir(parents=True, exist_ok=True)
    for nome, quadro in quadros.items():
        quadro.to_parquet(_cache(nome))
    crescimento = (1.0 + carregador).cumprod()
    futuro = crescimento.shift(-QNNConfig().horizon) / crescimento - 1.0
    linhas = [f"QNN v{MODEL_VERSION} probabilities (spec docs/superpowers/specs/2026-11-10-qnn-v4-design.md). "
              f"Lake exports sha256: {somas}",
              f"Config {json.dumps(asdict(QNNConfig()), sort_keys=True)}; seeds {SEEDS}; {len(datas)} decision dates "
              f"to {ULTIMA_SESSAO:%d/%m/%Y}; {len(quadros['qnn'])} with a cross-section.",
              f"Rank IC of P(up) against the {QNNConfig().horizon}-session holder return, eligible names, "
              f"out of sample (Newey-West {IC_LAGS} lags):"]
    linhas += [_ic_line(nome, rank_ic(quadro, futuro)) for nome, quadro in quadros.items()]
    correlacao = pd.concat({f"seed{seed}": quadros[f"seed{seed}"].stack() for seed in SEEDS}, axis=1).corr()
    linhas.append(f"Correlation of P(up) between seeds: {correlacao.round(3).to_dict()}")
    linhas.append(f"Correlation of the QNN mean with the logistic: "
                  f"{quadros['qnn'].stack().corr(quadros['logistic'].stack()):.3f}")
    texto = "\n".join(linhas) + "\n"
    (RAIZ / "docs" / "qnn-v4-probabilidades-2026-11-10.txt").write_text(texto, encoding="utf-8", newline="\n")
    print(texto)
    print(f"probabilities written ({time.perf_counter() - inicio:.0f} s)")
    return 0


# --- arms -------------------------------------------------------------------------------------------------------------


def base_methods(nota, mensal, setores: Mapping[str, str], cdi: pd.Series) -> dict[str, Estrategia]:
    """The twelve methods of the spec, built by the comparison screen's own factories."""
    return {COMPOSITE: estrategia_por_nota(nota, setores),
            "Conservadora": estrategia_conservadora(mensal.retornos, mensal.proventos, setores),
            **otimizadores(cdi),
            HRP_PAPER: estrategia_hrp_paper()}


def screened_arms(bases: Mapping[str, Estrategia], fontes: Mapping[str, ProbabilitySource],
                  so_qnn: frozenset[str] = frozenset({MICHAUD})) -> dict[str, Estrategia]:
    """Each base alone, then the base behind each source's screen (only the QNN screen for the slow methods)."""
    bracos: dict[str, Estrategia] = {}
    for nome, base in bases.items():
        bracos[nome] = base
        for chave, fonte in fontes.items():
            if chave == "qnn" or nome not in so_qnn:
                bracos[nome + SUFIXOS[chave]] = screened(base, fonte, KEEP_SHARE)
    return bracos


def tilt_arms(nota, setores: Mapping[str, str], cdi: pd.Series,
              fontes: Mapping[str, ProbabilitySource]) -> dict[str, Estrategia]:
    """The old headline arm (signal as an expected-return tilt inside the Composite's 15 names) and its controls."""
    seletor = seletor_do_composite(nota, setores)
    return {TILT[chave]: qml_strategy(fonte, CFG_KELLY, setores, cdi, selector=seletor, tilt=1.0)
            for chave, fonte in fontes.items()}


@dataclass(frozen=True)
class Linha:
    """One table row."""

    cagr: float
    sharpe: float
    sortino: float
    mdd: float
    vol: float
    nomes: float
    giro: float
    falhas: int
    t_base: float = float("nan")
    p_base: float = float("nan")
    t_composite: float = float("nan")


def medir(curva: ResultadoBacktest, rf: pd.Series, base: pd.Series | None, composite: pd.Series | None) -> Linha:
    resumo = resumo_desempenho(curva.valores, rf)

    def t_contra(referencia: pd.Series | None) -> float:
        return float("nan") if referencia is None else comparar_benchmark(curva.valores, referencia, rf).t_stat_alfa

    return Linha(resumo.cagr, resumo.sharpe, resumo.sortino, resumo.max_drawdown, resumo.volatilidade,
                 float((curva.pesos_alvo > 1e-8).sum(axis=1).mean()), curva.turnover_anual_medio,
                 len(curva.recusas) + len(curva.falhas), t_contra(base),
                 float("nan") if base is None else psr_da_diferenca(curva.valores, base), t_contra(composite))


def base_of(nome: str) -> str | None:
    """The base method of a screened arm's name; None for a base or a tilt arm."""
    for sufixo in SUFIXOS.values():
        if nome.endswith(sufixo):
            return nome[: -len(sufixo)]
    return None


def run_phase(lago: Path, fase: int, lote: str) -> int:
    inicio = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="qnn-v4-") as temporaria:
        somas = congelar(lago, Path(temporaria), ARQUIVOS)
        dados, mensal, nota, _ultima, cdi = _carregar(Path(temporaria))
    acoes = _acoes(dados.precos.loc[:ULTIMA_SESSAO].iloc[fase:])
    cdi = cdi.loc[:ULTIMA_SESSAO]
    setores = dict(dados.setores)
    qnn = ProbabilidadesCongeladas(pd.read_parquet(_cache("qnn")))
    logistica = ProbabilidadesCongeladas(pd.read_parquet(_cache("logistic")))
    fontes = {"qnn": qnn, "logistic": logistica, "shuffled": Embaralhada(qnn, 0), "anti": AntiSinal(qnn)}
    bases = base_methods(nota, mensal, setores, cdi)
    if lote == "michaud":
        bracos = screened_arms({COMPOSITE: bases[COMPOSITE], MICHAUD: bases[MICHAUD]}, {"qnn": qnn})
        bracos = {nome: braco for nome, braco in bracos.items() if nome != COMPOSITE + SUFIXOS["qnn"]}
    else:
        bracos = screened_arms({nome: base for nome, base in bases.items() if nome != MICHAUD}, fontes)
        bracos |= tilt_arms(nota, setores, cdi, {"v4": qnn, "v3": ProbabilidadesCongeladas(pd.read_parquet(CACHE_V3)),
                                                 "none": SemSinal(qnn), "logistic": logistica,
                                                 "shuffled": Embaralhada(qnn, 0)})
    curvas = {nome: _rodar(acoes, dados.universo, braco, nome, inicio) for nome, braco in bracos.items()}
    rf = cdi.reindex(curvas[COMPOSITE].valores.index).fillna(0.0)
    composite = curvas[COMPOSITE].valores
    linhas = {}
    for nome, curva in curvas.items():
        base = base_of(nome)
        linhas[nome] = asdict(medir(curva, rf, None if base is None else curvas[base].valores,
                                    None if nome == COMPOSITE else composite))
    tabela = pd.DataFrame(linhas).T
    tabela.insert(0, "fase", fase)
    tabela.rename_axis("arm").to_csv(DADOS / f"qnn-v4-f{fase}-{lote}.csv", encoding="utf-8", lineterminator="\n")
    pd.DataFrame({nome: curva.valores for nome, curva in curvas.items()}).rename_axis("date").reset_index() \
        .to_parquet(DADOS / f"qnn-v4-curvas-f{fase}-{lote}.parquet", index=False)
    (DADOS / f"qnn-v4-f{fase}-{lote}.json").write_text(
        json.dumps({"sha256": somas, "first_decision": f"{acoes.index[CFG_BT.janela_treino]:%Y-%m-%d}",
                    "last_session": f"{acoes.index[-1]:%Y-%m-%d}"}, sort_keys=True), encoding="utf-8")
    print(tabela.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"phase {fase}, batch {lote} done ({time.perf_counter() - inicio:.0f} s)")
    return 0


# --- report and gates -------------------------------------------------------------------------------------------------


def signal_gate(tabelas: Mapping[int, pd.DataFrame], base: str) -> tuple[bool, bool]:
    """(signal gate, quantum gate) of ``base + QNN`` over the phases given, as written in the spec."""
    sinal = quantico = True
    for tabela in tabelas.values():
        x, braco = tabela.loc[base], tabela.loc[base + SUFIXOS["qnn"]]
        sinal &= bool(braco["cagr"] > x["cagr"] and braco["sharpe"] > x["sharpe"] and braco["t_base"] >= T_SIGNAL)
        for controle in ("shuffled", "anti"):
            nome = base + SUFIXOS[controle]
            if nome in tabela.index:
                sinal &= bool(braco["cagr"] > tabela.loc[nome, "cagr"])
        nome = base + SUFIXOS["logistic"]
        quantico &= bool(nome in tabela.index and braco["cagr"] > tabela.loc[nome, "cagr"])
    return sinal, sinal and quantico


def _fases(valores: Sequence[float], formato: str = "{:6.2%}") -> str:
    return " / ".join(formato.format(valor) for valor in valores)


def run_report() -> int:
    tabelas: dict[int, pd.DataFrame] = {}
    for fase in FASES:
        partes = [pd.read_csv(arquivo, index_col="arm") for arquivo in sorted(DADOS.glob(f"qnn-v4-f{fase}-*.csv"))]
        if partes:
            junta = pd.concat(partes)
            tabelas[fase] = junta[~junta.index.duplicated(keep="first")]
    if not tabelas:
        raise SystemExit("no phase table found: run --fase first")
    fases = sorted(tabelas)
    bases = [nome for nome in tabelas[fases[0]].index
             if base_of(nome) is None and not nome.startswith("Tilt") and nome + SUFIXOS["qnn"] in tabelas[fases[0]].index]

    def coluna(nome: str, campo: str) -> list[float]:
        return [tabelas[fase].loc[nome, campo] if nome in tabelas[fase].index else float("nan") for fase in fases]

    linhas = [f"QNN v4 screened into every classical method - phases {fases} "
              f"(spec docs/superpowers/specs/2026-11-10-qnn-v4-design.md)", "",
              f"CAGR by phase ({' / '.join(str(fase) for fase in fases)}). t = alpha of X + QNN regressed on X "
              f"(Newey-West); signal gate needs t >= {T_SIGNAL} in every phase.", ""]
    for base in bases:
        sinal, quantico = signal_gate(tabelas, base)
        linhas.append(f"{base}")
        linhas.append(f"  alone        {_fases(coluna(base, 'cagr'))}   Sharpe {_fases(coluna(base, 'sharpe'), '{:5.2f}')}"
                      f"   MDD {_fases(coluna(base, 'mdd'))}")
        for chave, sufixo in SUFIXOS.items():
            nome = base + sufixo
            if nome not in tabelas[fases[0]].index:
                continue
            extra = (f"   t vs X {_fases(coluna(nome, 't_base'), '{:+5.2f}')}"
                     f"   t vs Composite {_fases(coluna(nome, 't_composite'), '{:+5.2f}')}" if chave == "qnn" else "")
            linhas.append(f"  {sufixo.strip():<12} {_fases(coluna(nome, 'cagr'))}   Sharpe "
                          f"{_fases(coluna(nome, 'sharpe'), '{:5.2f}')}   MDD {_fases(coluna(nome, 'mdd'))}{extra}")
        linhas.append(f"  signal gate: {'PASSES' if sinal else 'fails'}; quantum gate: "
                      f"{'PASSES' if quantico else 'fails'}")
        linhas.append("")
    linhas.append("Tilt re-test (signal as an expected-return tilt inside the Composite's 15 names, Meio-Kelly):")
    for nome in TILT.values():
        if nome in tabelas[fases[0]].index:
            linhas.append(f"  {nome:<48} {_fases(coluna(nome, 'cagr'))}   Sharpe "
                          f"{_fases(coluna(nome, 'sharpe'), '{:5.2f}')}   t vs Composite "
                          f"{_fases(coluna(nome, 't_composite'), '{:+5.2f}')}")
    falhas = {nome: int(sum(coluna(nome, "falhas"))) for nome in tabelas[fases[0]].index if sum(coluna(nome, "falhas"))}
    linhas += ["", f"Arms with refused or failed rebalances (the engine kept the book), summed over phases: {falhas}"]
    texto = "\n".join(linhas) + "\n"
    RELATORIO.write_text(texto, encoding="utf-8", newline="\n")
    pd.concat(tabelas, names=["fase_lida"]).to_csv(DADOS / "qnn-v4-tabela.csv", encoding="utf-8", lineterminator="\n")
    print(texto)
    return 0


def main(argv: list[str]) -> int:
    def opcao(nome: str, padrao: str) -> str:
        return argv[argv.index(nome) + 1] if nome in argv else padrao

    if "--relatorio" in argv:
        return run_report()
    lago = Path(opcao("--lago", str(LAGO)))
    if "--probabilidades" in argv:
        return run_probabilities(lago)
    lote = opcao("--lote", "principal")
    if lote not in ("principal", "michaud"):
        raise SystemExit(f"--lote must be principal or michaud: {lote!r}")
    return run_phase(lago, int(opcao("--fase", "0")), lote)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
