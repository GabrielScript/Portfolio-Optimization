"""One cell of a run: one strategy on one walk-forward window, written to its own JSON file.

    .venv-drl/Scripts/python -m drl.executar --lago <lake exports> --run <name> --strategy ppo/dsr --year 2016
    .venv-drl/Scripts/python -m drl.executar --lago <lake exports> --run <name> --strategy min_variance --all-years

A strategy is ``algorithm/reward`` (an agent) or a rule of ``drl.baselines.RULES``. An agent is trained on the
window's five training years once per seed, the seed with the highest reward on the validation year is kept, and
only that agent sees the test year. A rule has nothing to fit and runs the test year directly.

One file per cell, never an appended CSV: the first version appended rows whose columns changed between runs and left
two of its result files unreadable. A cell whose file exists is skipped, so a run that dies is restarted by the same
command. A cell that fails raises: nothing is swallowed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from drl import agentes, baselines, protocolo
from drl.ambiente import EnvConfig, PortfolioEnv, simulate
from drl.dados import Panel, load_panel
from drl.relatorio import metrics

RAIZ = Path(__file__).resolve().parent.parent
SAIDAS = RAIZ / "drl" / "saidas"
LAGO = RAIZ / "data" / "b3lake"
FIRST_YEAR, LAST_YEAR = 2012, 2026           # the first test year with five training years and one of validation


@dataclass(frozen=True)
class RunSpec:
    """Everything that defines a run besides the strategy and the year; stored in every cell file."""

    env: EnvConfig = field(default_factory=EnvConfig)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)     # P1 trains five agents per window
    timesteps: int = 150_000                     # decisions, not sessions, when ``env.hold`` > 1
    n_envs: int = 4                              # four physical cores
    last_session: str | None = None


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


def _environment(panel: Panel, columns: np.ndarray, trecho: slice, config: EnvConfig) -> PortfolioEnv:
    return PortfolioEnv(panel, columns, trecho.start, min(trecho.stop, len(panel.sessions) - 1), config)


def run_cell(panel: Panel, janela: protocolo.Janela, strategy: str, spec: RunSpec) -> Cell:
    nomes = window_names(panel, janela)
    if len(nomes) < 2:
        raise ValueError(f"{janela.ano_teste}: only {len(nomes)} eligible names")
    colunas = panel.columns(nomes)
    teste = _environment(panel, colunas, janela.teste, spec.env)
    semente, recompensas = None, {}
    if strategy in baselines.RULES:
        simulate(teste, baselines.RULES[strategy])
    else:
        algoritmo, _, recompensa = strategy.partition("/")
        config = EnvConfig(**{**asdict(spec.env), "reward": recompensa})
        teste = _environment(panel, colunas, janela.teste, config)
        validacao = _environment(panel, colunas, janela.validacao, config)

        def treino() -> PortfolioEnv:
            return _environment(panel, colunas, janela.treino, config)

        escolha = agentes.select(spec.seeds,
                                 lambda seed: agentes.train(algoritmo, treino, spec.timesteps, seed, spec.n_envs),
                                 lambda modelo: agentes.evaluate(modelo, validacao)[1])
        agentes.evaluate(escolha.model, teste)
        semente, recompensas = escolha.seed, {str(seed): valor for seed, valor in escolha.validation_rewards.items()}
    sessoes = panel.sessions[teste.start:teste.start + len(teste.values)]
    return Cell(strategy, janela.ano_teste, len(nomes), [f"{data:%Y-%m-%d}" for data in sessoes],
                [float(valor) for valor in teste.values], [float(giro) for giro in teste.turnover], semente, recompensas)


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
    p.add_argument("--observation", default="compact", choices=("matrix", "compact"))
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--timesteps", type=int, default=150_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--last-session", default=None)
    args = p.parse_args(argv)
    if not args.all_years and args.year is None:
        p.error("give --year or --all-years")
    algoritmo, _, recompensa = args.strategy.partition("/")
    if args.strategy not in baselines.RULES and algoritmo not in agentes.ALGORITHMS:
        p.error(f"unknown strategy {args.strategy!r}")

    spec = RunSpec(EnvConfig(reward=recompensa or "dsr", cost_bps=args.cost_bps, hold=args.hold,
                             observation=args.observation),
                   tuple(int(s) for s in args.seeds.split(",")), args.timesteps, args.n_envs, args.last_session)
    panel = load_panel(args.lago, args.last_session)
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
