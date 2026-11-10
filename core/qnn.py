"""
qnn.py - the quantum neural network signal, rewritten after the 2026-11-10 review (test branch, never production).

``core.qml`` stays as it is because the frozen forward holdout runs on it. This module is the corrected signal that
new work should use. What changed, and why:

* Point-in-time population. A sample exists only for a name the lake lists as eligible in the month the book would
  trade. ``core.qml`` trained on every column of the price file: a quarter of its samples were names that were not
  eligible that month, plus the ETFs, and each date's z-scores were taken over that wider population.
* No survivorship in the labels. The label is the return of someone who holds the name (a delisted name stays at its
  last close, as in the engine), so a name that dies inside the horizon keeps its sample.
* Purged validation. The early-stopping sample is the last share of decision dates, and every training sample whose
  label is realized inside it is dropped; a 63-session label otherwise leaks from training into validation.
* Reproducible blocks. Every random draw comes from a generator seeded by the block, so a block trains the same
  whatever was trained before it in the process. The global Torch generator is left as it was found.
* One readout. The old ``scale * (W x + b) + bias`` had two redundant parameters. The L2 penalty now applies to the
  readout only: pulling periodic rotation angles towards zero has no meaning.
* Any trainer. ``WalkForwardPredictor`` takes the trainer as an argument, so the logistic baseline goes through the
  same panel, blocks and boundary as the circuit.

The circuit itself is unchanged: six features on six wires (``AngleEmbedding``, Y rotations), ``n_layers`` strongly
entangling layers, one Z expectation per wire, a linear readout to one logit. Simulated exactly on
``default.qubit`` (CPU, float64); there is no shot noise to model.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
import pennylane as qml
import torch
from torch import nn

from core.backtest import Estrategia, retornos_da_carteira
from core.qml import MIN_HISTORY, N_FEATURES, WINDOW, Panel, ProbabilitySource, angles, cross_sectional_z, features_at

N_QUBITS = N_FEATURES
BLOCK_ANCHOR = 2000              # blocks start on years congruent to this one
MODEL_VERSION = 4                # bump when the panel, the circuit or the training changes: the caches depend on it


@dataclass(frozen=True)
class QNNConfig:
    """Everything that changes the panel, the circuit or its training; recorded next to every cache."""

    n_layers: int = 2
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 0.05
    readout_weight_decay: float = 1e-4
    validation_share: float = 0.15
    patience: int = 6
    horizon: int = 63              # label horizon, sessions
    sample_every: int = 21         # one panel date every month
    min_history: int = MIN_HISTORY
    block_years: int = 1           # a new model every year, trained on labels realized before 1 January
    min_names: int = 10            # a date with a smaller cross-section gives no sample and no prediction
    min_samples: int = 500         # a block with fewer realized samples is refused
    seed: int = 0

    def __post_init__(self) -> None:
        if min(self.n_layers, self.epochs, self.batch_size, self.patience, self.horizon, self.sample_every,
               self.block_years, self.min_samples) < 1:
            raise ValueError("layers, epochs, batch, patience, horizon, sampling step, block and samples must be >= 1")
        if not 0.0 < self.validation_share < 0.5:
            raise ValueError(f"validation_share must be in (0, 0.5): {self.validation_share}")
        if self.min_history < WINDOW:
            raise ValueError(f"min_history must cover the feature window ({WINDOW}): {self.min_history}")
        if self.min_names < 3:
            raise ValueError(f"min_names must be >= 3 for a median split: {self.min_names}")


# --- point-in-time panel ----------------------------------------------------------------------------------------------


def holder_returns(precos: pd.DataFrame) -> pd.DataFrame:
    """Daily returns of someone who holds each name: the engine's own definition (no trade or delisted = 0)."""
    return retornos_da_carteira(precos)


def eligible_by_month(universo: pd.DataFrame) -> dict[pd.Timestamp, frozenset[str]]:
    """The lake's eligible names per month (columns ``month`` and ``security``)."""
    meses = pd.to_datetime(universo["month"])
    return {pd.Timestamp(mes): frozenset(nomes) for mes, nomes in universo["security"].groupby(meses)}


def trading_month(sessions: pd.DatetimeIndex, position: int) -> pd.Timestamp:
    """
    The month of the session after ``position``: a decision taken with the close of ``position`` trades on the next
    session, and the engine reads that session's month to list the eligible names.
    """
    seguinte = sessions[position + 1] if position + 1 < len(sessions) else sessions[position] + pd.offsets.BDay(1)
    return pd.Timestamp(seguinte).to_period("M").to_timestamp()


def cross_section(returns: pd.DataFrame, position: int, eligible: Mapping[pd.Timestamp, frozenset[str]],
                  min_history: int) -> pd.DataFrame:
    """Z-scored features of the names eligible for the trade decided at ``position``, from rows up to it."""
    permitidos = eligible.get(trading_month(returns.index, position), frozenset())
    nomes = [papel for papel in returns.columns if papel in permitidos]
    if not nomes or position + 1 < min_history:
        return pd.DataFrame(columns=returns.columns[:0], dtype=float)
    janela = returns.iloc[position + 1 - min_history:position + 1][nomes]
    return cross_sectional_z(features_at(janela, min_history))


def build_pit_panel(returns: pd.DataFrame, holder: pd.DataFrame, eligible: Mapping[pd.Timestamp, frozenset[str]],
                    config: QNNConfig) -> Panel:
    """
    One cross-section every ``sample_every`` sessions. ``returns`` (gaps as NaN) feeds the features; ``holder``
    (``holder_returns``, no NaN) feeds the labels: 1 when the name's ``horizon``-session return is above the median
    of the same cross-section. The last ``horizon`` sessions have no realized label and stay out.
    """
    if not returns.index.equals(holder.index):
        raise ValueError("returns and holder must share the same sessions")
    sessoes = returns.index
    crescimento = (1.0 + holder).cumprod()
    futuro = crescimento.shift(-config.horizon) / crescimento - 1.0
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    decisoes: list[pd.Timestamp] = []
    rotulos: list[pd.Timestamp] = []
    for posicao in range(config.min_history - 1, len(sessoes) - config.horizon, config.sample_every):
        z = cross_section(returns, posicao, eligible, config.min_history)
        if len(z) < config.min_names:
            continue
        alvo = futuro.iloc[posicao].reindex(z.index)
        if alvo.isna().any():
            raise ValueError(f"{sessoes[posicao]:%Y-%m-%d}: holder return missing for {list(alvo[alvo.isna()].index)}")
        xs.append(angles(z))
        ys.append((alvo > float(alvo.median())).to_numpy(dtype=np.float64))
        decisoes.extend([sessoes[posicao]] * len(z))
        rotulos.extend([sessoes[posicao + config.horizon]] * len(z))
    if not xs:
        raise ValueError("the panel is empty: no date has an eligible cross-section with a realized label")
    return Panel(np.vstack(xs), np.concatenate(ys), pd.DatetimeIndex(rotulos), pd.DatetimeIndex(decisoes))


def purged_split(decision_dates: pd.DatetimeIndex, label_dates: pd.DatetimeIndex,
                 validation_share: float) -> tuple[np.ndarray, np.ndarray]:
    """
    (training rows, validation rows). Validation is the last ``validation_share`` of the decision dates; training
    keeps only the samples whose label was realized before the first validation decision.
    """
    datas = decision_dates.unique().sort_values()
    if len(datas) < 2:
        raise ValueError("a purged split needs at least two decision dates")
    corte = datas[min(max(int(len(datas) * (1.0 - validation_share)), 1), len(datas) - 1)]
    treino = np.flatnonzero(np.asarray(label_dates < corte))
    validacao = np.flatnonzero(np.asarray(decision_dates >= corte))
    if not len(treino):
        raise ValueError(f"no training label realized before the validation start {corte:%Y-%m-%d}")
    return treino, validacao


# --- models -----------------------------------------------------------------------------------------------------------


class ProbabilityModel(Protocol):
    """A fitted model: angles ``(n, N_FEATURES)`` in, P(up) ``(n,)`` out."""

    def probability_up(self, x: np.ndarray) -> np.ndarray: ...


Trainer = Callable[[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex, int], ProbabilityModel]
"""(x, y, decision dates, label dates, seed) -> fitted model. Every sample given is already causal for the block."""


@contextlib.contextmanager
def _global_generator_untouched() -> Iterator[None]:
    """Layer constructors draw from the global Torch generator; put it back so callers see no side effect."""
    estado = torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(estado)


class QuantumClassifier(nn.Module):
    """Six angles -> ``n_layers`` strongly entangling layers -> <Z> per wire -> one logit. Float64, exact simulation."""

    def __init__(self, n_layers: int, seed: int) -> None:
        super().__init__()
        gerador = torch.Generator().manual_seed(seed)
        dev = qml.device("default.qubit", wires=N_QUBITS)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
            qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
            return [qml.expval(qml.Z(wire)) for wire in range(N_QUBITS)]

        forma = (n_layers, N_QUBITS, 3)
        with _global_generator_untouched():
            self.quantum = qml.qnn.TorchLayer(circuit, {"weights": forma})
            self.readout = nn.Linear(N_QUBITS, 1)
        self.double()
        limite = 1.0 / np.sqrt(N_QUBITS)
        with torch.no_grad():
            self.quantum.weights.copy_(torch.rand(forma, generator=gerador, dtype=torch.float64) * 2.0 * np.pi)
            self.readout.weight.copy_((torch.rand((1, N_QUBITS), generator=gerador, dtype=torch.float64) * 2.0 - 1.0)
                                      * limite)
            self.readout.bias.zero_()

    def expectations(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (batch, N_QUBITS) angles -> (batch, N_QUBITS) expectations of Z in [-1, 1]."""
        return self.quantum(x).reshape(-1, N_QUBITS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits, shape (batch,)."""
        return self.readout(self.expectations(x)).reshape(-1)

    @torch.no_grad()
    def probability_up(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        return torch.sigmoid(self.forward(torch.as_tensor(x, dtype=torch.float64))).numpy()


def quantum_trainer(config: QNNConfig, log: Callable[[str], None] | None = None) -> Trainer:
    """Adam on the binary cross-entropy, purged time split, early stopping on the validation loss."""

    def train(x: np.ndarray, y: np.ndarray, decision_dates: pd.DatetimeIndex, label_dates: pd.DatetimeIndex,
              seed: int) -> QuantumClassifier:
        linhas_treino, linhas_validacao = purged_split(decision_dates, label_dates, config.validation_share)
        treino_x, treino_y = torch.as_tensor(x[linhas_treino]), torch.as_tensor(y[linhas_treino])
        validacao_x, validacao_y = torch.as_tensor(x[linhas_validacao]), torch.as_tensor(y[linhas_validacao])
        modelo = QuantumClassifier(config.n_layers, seed)
        gerador = torch.Generator().manual_seed(seed)
        otimizador = torch.optim.Adam(
            [{"params": modelo.quantum.parameters(), "weight_decay": 0.0},
             {"params": modelo.readout.parameters(), "weight_decay": config.readout_weight_decay}],
            lr=config.learning_rate)
        perda = nn.BCEWithLogitsLoss()
        melhor, melhor_estado, sem_melhora, epocas = float("inf"), None, 0, 0
        for epocas in range(1, config.epochs + 1):
            modelo.train()
            ordem = torch.randperm(len(treino_y), generator=gerador)
            for inicio in range(0, len(ordem), config.batch_size):
                lote = ordem[inicio:inicio + config.batch_size]
                otimizador.zero_grad(set_to_none=True)
                custo = perda(modelo(treino_x[lote]), treino_y[lote])
                custo.backward()
                nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
                otimizador.step()
            modelo.eval()
            with torch.no_grad():
                validacao = float(perda(modelo(validacao_x), validacao_y))
            if validacao < melhor - 1e-6:
                melhor, sem_melhora = validacao, 0
                melhor_estado = {chave: valor.detach().clone() for chave, valor in modelo.state_dict().items()}
            else:
                sem_melhora += 1
                if sem_melhora >= config.patience:
                    break
        if melhor_estado is not None:
            modelo.load_state_dict(melhor_estado)
        if log is not None:
            log(f"    quantum: {len(linhas_treino)} training and {len(linhas_validacao)} validation samples, "
                f"{epocas} epochs, best validation loss {melhor:.4f} (coin flip 0.6931)")
        return modelo

    return train


class _Logistic:
    def __init__(self, modelo) -> None:
        self._modelo = modelo

    def probability_up(self, x: np.ndarray) -> np.ndarray:
        return self._modelo.predict_proba(x)[:, 1]


def logistic_trainer(c: float = 1.0) -> Trainer:
    """The classical baseline on the same samples: a logistic regression on the six angles."""

    def train(x: np.ndarray, y: np.ndarray, decision_dates: pd.DatetimeIndex, label_dates: pd.DatetimeIndex,
              seed: int) -> _Logistic:
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        return _Logistic(LogisticRegression(C=c, max_iter=500).fit(x, y))

    return train


# --- walk-forward -----------------------------------------------------------------------------------------------------


class WalkForwardPredictor:
    """
    P(up) per eligible name as of a session. One model per block of ``block_years`` calendar years, trained on the
    samples whose label was realized before 1 January of the block's first year; every date of the block reuses it.
    """

    def __init__(self, returns: pd.DataFrame, holder: pd.DataFrame, eligible: Mapping[pd.Timestamp, frozenset[str]],
                 config: QNNConfig, trainer: Trainer, log: Callable[[str], None] | None = None) -> None:
        self.config = config
        self._returns = returns
        self._eligible = eligible
        self._trainer = trainer
        self._log = log or (lambda _: None)
        self._models: dict[int, ProbabilityModel] = {}
        self.panel = build_pit_panel(returns, holder, eligible, config)

    def block_of(self, year: int) -> int:
        return year - (year - BLOCK_ANCHOR) % self.config.block_years

    def model_for(self, year: int) -> ProbabilityModel:
        bloco = self.block_of(year)
        if bloco not in self._models:
            linhas = np.flatnonzero(np.asarray(self.panel.label_dates < pd.Timestamp(year=bloco, month=1, day=1)))
            if len(linhas) < self.config.min_samples:
                raise ValueError(f"block {bloco}: {len(linhas)} realized samples before its first day "
                                 f"(need {self.config.min_samples})")
            self._log(f"  block {bloco}: {len(linhas)} samples with labels realized before 01/01/{bloco}")
            self._models[bloco] = self._trainer(self.panel.x[linhas], self.panel.y[linhas],
                                                self.panel.decision_dates[linhas], self.panel.label_dates[linhas],
                                                self.config.seed * 100_000 + bloco)
        return self._models[bloco]

    def predict(self, date: pd.Timestamp) -> pd.Series:
        """P(up) from the rows up to and including ``date`` (the last session on or before it)."""
        posicao = int(self._returns.index.searchsorted(date, side="right")) - 1
        if posicao < 0:
            return pd.Series(dtype=float, name="p_up")
        z = cross_section(self._returns, posicao, self._eligible, self.config.min_history)
        if len(z) < self.config.min_names:
            return pd.Series(dtype=float, name="p_up")
        modelo = self.model_for(self._returns.index[posicao].year)
        return pd.Series(modelo.probability_up(angles(z)), index=z.index, name="p_up")


# --- the signal in a portfolio ----------------------------------------------------------------------------------------


def screened(strategy: Estrategia, source: ProbabilitySource, keep_share: float = 0.5,
             min_names: int = 20) -> Estrategia:
    """
    ``strategy`` on the training window cut to the top ``keep_share`` of its names by P(up). The signal only decides
    which names the allocator may see; the allocator is untouched, so the same screen fits every method. A window
    whose screened cross-section has fewer than ``min_names`` names raises ``ValueError`` (the engine keeps the book).
    """
    if not 0.0 < keep_share <= 1.0:
        raise ValueError(f"keep_share must be in (0, 1]: {keep_share}")

    def _strategy(treino: pd.DataFrame) -> pd.Series:
        data = treino.index[-1]
        probabilidades = source.predict(data).reindex(treino.columns).dropna()
        manter = int(np.ceil(len(probabilidades) * keep_share))
        if manter < min_names:
            raise ValueError(f"{data:%Y-%m-%d}: only {manter} names pass the screen (need {min_names})")
        escolhidos = set(probabilidades.sort_values(ascending=False, kind="stable").index[:manter])
        return strategy(treino[[papel for papel in treino.columns if papel in escolhidos]])

    return _strategy


__all__: Sequence[str] = (
    "MODEL_VERSION", "QNNConfig", "QuantumClassifier", "WalkForwardPredictor", "build_pit_panel", "cross_section",
    "eligible_by_month", "holder_returns", "logistic_trainer", "purged_split", "quantum_trainer", "screened",
    "trading_month",
)
