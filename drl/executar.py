"""One cell of a run: one strategy on one walk-forward window, written to its own JSON file.

    .venv-drl/Scripts/python -m drl.executar --lago <lake exports> --run <name> --strategy ppo/dsr --year 2016
    .venv-drl/Scripts/python -m drl.executar --lago <lake exports> --run <name> --strategy min_variance --all-years

A strategy is a rule of ``drl.baselines.RULES``, ``algorithm/reward`` (a Stable-Baselines3 agent) or
``direct/sharpe`` / ``direct_score/sharpe`` (the shared network trained through the simulator, ``drl.direto``). An
SB3 agent is trained on the window's five training years once per seed, the seed with the highest reward on the
validation year is kept, and only that agent sees the test year. A direct arm trains every seed, keeps each seed's
best validation check, and trades the test year with the mean of the seeds' books. A rule has nothing to fit.

The agents' environments are built on the window's training universe (``training_names``); their validation and
test stretches trade only the names known on their own first session (``stretches``).

One file per cell, never an appended CSV: the first version appended rows whose columns changed between runs and left
two of its result files unreadable. A cell whose file exists is skipped, so a run that dies is restarted by the same
command. A cell that fails raises: nothing is swallowed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from drl import agentes, baselines, protocolo
from drl.ambiente import OBSERVATIONS, EnvConfig, PortfolioEnv, simulate
from drl.dados import Panel, load_panel
from drl.relatorio import metrics

RAIZ = Path(__file__).resolve().parent.parent
SAIDAS = RAIZ / "drl" / "saidas"
LAGO = RAIZ / "data" / "b3lake"
FIRST_YEAR, LAST_YEAR = 2012, 2026           # the first test year with five training years and one of validation
DIRECT = {"direct": "cross", "direct_score": "cross_score"}      # family -> the observation it reads
DIRECT_OBJECTIVE = "sharpe"


@dataclass(frozen=True)
class RunSpec:
    """Everything that defines a run besides the strategy and the year; stored in every cell file."""

    env: EnvConfig = field(default_factory=EnvConfig)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)     # P1 trains five agents per window
    timesteps: int = 150_000                     # decisions, not sessions, when ``env.hold`` > 1
    n_envs: int = 4                              # four physical cores
    last_session: str | None = None
    episode: int | None = 12                     # decisions per training episode; None replays the years as one path
    direct_steps: int = 300                      # gradient steps of the direct arms


def spec_from_dict(campos: dict) -> RunSpec:
    """A ``RunSpec`` back from ``asdict``: what crosses to a Modal container."""
    return RunSpec(**{**campos, "env": EnvConfig(**campos["env"]), "seeds": tuple(campos["seeds"])})


def corrected_env(cost_bps: float = 10.0, hold: int = 21, observation: str = "cross") -> EnvConfig:
    """The v3 environment (docs/superpowers/specs/2026-11-12-drl-v3-design.md): fully invested, the reward measured
    over the CDI from a warm start, inputs standardized across the names."""
    return EnvConfig(reward="dsr_cdi", cost_bps=cost_bps, hold=hold, observation=observation, invest_all=True,
                     dsr_warmup=252)


@dataclass(frozen=True)
class Cell:
    """One strategy on one test year: its curve (one point per session, starting at 1) and how it was obtained."""

    strategy: str
    year: int
    names: int
    sessions: list[str]
    values: list[float]
    turnover: list[float]
    chosen_seed: int | None
    validation_rewards: dict[str, float]


def window_names(panel: Panel, janela: protocolo.Janela) -> list[str]:
    """The names an environment of this window may hold: those eligible at the test year's first session. Known when
    the test year starts; a name listed later in the year is not traded until the next window."""
    return panel.eligible_names(janela.teste.start)


def training_names(panel: Panel, janela: protocolo.Janela) -> list[str]:
    """The names an agent's environments are built on: every name eligible on some session of the training or
    validation years, or on the first test session. With ``window_names`` alone the five training years held only
    the names still listed six years later, and the agent never saw a name die."""
    vistos = panel.eligible[janela.treino.start:janela.validacao.stop].any(axis=0) | panel.eligible[janela.teste.start]
    return [nome for nome, sim in zip(panel.names, vistos) if sim]


def _environment(panel: Panel, columns: np.ndarray, trecho: slice, config: EnvConfig, **extra) -> PortfolioEnv:
    return PortfolioEnv(panel, columns, trecho.start, min(trecho.stop, len(panel.sessions) - 1), config, **extra)


@dataclass(frozen=True)
class Stretches:
    """An agent's three environments of one window. ``train`` builds a fresh copy (one per vectorized worker)."""

    train: Callable[[], PortfolioEnv]
    validation: PortfolioEnv
    test: PortfolioEnv


def stretches(panel: Panel, janela: protocolo.Janela, config: EnvConfig, episode: int | None) -> Stretches:
    """Training trades whatever is eligible each session among the training universe, in episodes of ``episode``
    decisions; validation and test trade only the names eligible on their first session, so the test book is drawn
    from the same names as the rules' (``window_names``)."""
    colunas = panel.columns(training_names(panel, janela))

    def conhecidos(trecho: slice) -> np.ndarray:
        return panel.eligible[trecho.start, colunas]

    def treino() -> PortfolioEnv:
        return _environment(panel, colunas, janela.treino, config, episode=episode)

    return Stretches(treino,
                     _environment(panel, colunas, janela.validacao, config, allowed=conhecidos(janela.validacao)),
                     _environment(panel, colunas, janela.teste, config, allowed=conhecidos(janela.teste)))


def _rule(panel: Panel, janela: protocolo.Janela, strategy: str, spec: RunSpec) -> PortfolioEnv:
    teste = _environment(panel, panel.columns(window_names(panel, janela)), janela.teste, spec.env)
    return simulate(teste, baselines.RULES[strategy])


def _direct(panel: Panel, janela: protocolo.Janela, strategy: str, spec: RunSpec
            ) -> tuple[PortfolioEnv, dict[int, float]]:
    """Every seed's scorer, then the test year with the mean of their books. Returns each seed's validation Sharpe."""
    from drl import direto  # Torch stays out of the rules' path

    familia, _, objetivo = strategy.partition("/")
    if objetivo != DIRECT_OBJECTIVE or spec.episode is None:
        raise ValueError(f"a direct arm maximizes {DIRECT_OBJECTIVE!r} over episodes of a fixed length: {strategy!r}, "
                         f"episode {spec.episode}")
    trechos = stretches(panel, janela, replace(spec.env, observation=DIRECT[familia]), episode=None)
    treino = trechos.train()
    parametros = direto.DirectParams(steps=spec.direct_steps, episode=spec.episode)
    ajustes = {seed: direto.fit(treino, trechos.validation, seed, parametros) for seed in spec.seeds}
    simulate(trechos.test, direto.policy_of([ajuste.scorer for ajuste in ajustes.values()]))
    return trechos.test, {seed: ajuste.validation_sharpe for seed, ajuste in ajustes.items()}


def _agent(panel: Panel, janela: protocolo.Janela, strategy: str, spec: RunSpec
           ) -> tuple[PortfolioEnv, agentes.Selection]:
    algoritmo, _, recompensa = strategy.partition("/")
    trechos = stretches(panel, janela, replace(spec.env, reward=recompensa), spec.episode)
    escolha = agentes.select(spec.seeds,
                             lambda seed: agentes.train(algoritmo, trechos.train, spec.timesteps, seed, spec.n_envs),
                             lambda modelo: agentes.evaluate(modelo, trechos.validation)[1])
    agentes.evaluate(escolha.model, trechos.test)
    return trechos.test, escolha


def run_cell(panel: Panel, janela: protocolo.Janela, strategy: str, spec: RunSpec) -> Cell:
    nomes = window_names(panel, janela)
    if len(nomes) < 2:
        raise ValueError(f"{janela.ano_teste}: only {len(nomes)} eligible names")
    semente, validacao = None, {}
    if strategy in baselines.RULES:
        teste = _rule(panel, janela, strategy, spec)
    elif strategy.partition("/")[0] in DIRECT:
        teste, validacao = _direct(panel, janela, strategy, spec)
    else:
        teste, escolha = _agent(panel, janela, strategy, spec)
        semente, validacao = escolha.seed, escolha.validation_rewards
    sessoes = panel.sessions[teste.start:teste.start + len(teste.values)]
    return Cell(strategy, janela.ano_teste, len(nomes), [f"{data:%Y-%m-%d}" for data in sessoes],
                [float(valor) for valor in teste.values], [float(giro) for giro in teste.turnover], semente,
                {str(seed): float(valor) for seed, valor in validacao.items()})


def cell_path(run: str, strategy: str, year: int, saidas: Path = SAIDAS) -> Path:
    return saidas / run / f"{strategy.replace('/', '-')}-{year}.json"


def write_cell(cell: Cell, spec: RunSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporario = path.with_suffix(".tmp")
    temporario.write_text(json.dumps({"spec": asdict(spec), **asdict(cell)}, sort_keys=True), encoding="utf-8")
    temporario.replace(path)                     # a killed run leaves no half file under the final name


def read_cell(path: Path) -> Cell:
    conteudo = json.loads(path.read_text(encoding="utf-8"))
    return Cell(**{chave: conteudo[chave] for chave in Cell.__dataclass_fields__})


def cell_metrics(cell: Cell, panel: Panel) -> dict[str, float]:
    """The cell's own year, measured over the CDI of its sessions."""
    inicio = int(panel.sessions.get_loc(cell.sessions[0]))
    return metrics(cell.values, panel.cdi[inicio:inicio + len(cell.values) - 1], cell.turnover)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lago", type=Path, default=LAGO)
    p.add_argument("--run", required=True, help="folder under drl/saidas that holds this run's cells")
    p.add_argument("--strategy", required=True)
    p.add_argument("--year", type=int)
    p.add_argument("--all-years", action="store_true")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--hold", type=int, default=21)
    p.add_argument("--observation", default="cross", choices=OBSERVATIONS)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--timesteps", type=int, default=150_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--last-session", default=None)
    p.add_argument("--direct-steps", type=int, default=300)
    p.add_argument("--scores", type=Path, default=None, help="parquet of monthly scores (direct_score)")
    args = p.parse_args(argv)
    if not args.all_years and args.year is None:
        p.error("give --year or --all-years")
    familia = args.strategy.partition("/")[0]
    if args.strategy not in baselines.RULES and familia not in (*agentes.ALGORITHMS, *DIRECT):
        p.error(f"unknown strategy {args.strategy!r}")

    spec = RunSpec(corrected_env(args.cost_bps, args.hold, args.observation),
                   tuple(int(s) for s in args.seeds.split(",")), args.timesteps, args.n_envs, args.last_session,
                   direct_steps=args.direct_steps)
    panel = load_panel(args.lago, args.last_session, scores=args.scores)
    janelas = {j.ano_teste: j for j in protocolo.janelas(panel.sessions, FIRST_YEAR, LAST_YEAR)}
    anos = sorted(janelas) if args.all_years else [args.year]
    for ano in anos:
        if ano not in janelas:
            raise SystemExit(f"no complete window for {ano}")
        caminho = cell_path(args.run, args.strategy, ano)
        if caminho.exists():
            print(f"{args.strategy} {ano}: already there", flush=True)
            continue
        cell = run_cell(panel, janelas[ano], args.strategy, spec)
        write_cell(cell, spec, caminho)
        linha = cell_metrics(cell, panel)
        print(f"{args.strategy} {ano}: {cell.names} names, return {linha['total']:+.1%}, Sharpe {linha['sharpe']:.2f}, "
              f"MDD {linha['mdd']:.1%}, turnover {linha['turnover']:.1f}, seed {cell.chosen_seed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
