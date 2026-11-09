"""
qml.py - Quantum machine learning arm of the QPO (test branch, never production).

The split is deliberate: the quantum circuit is the *brain* and only reads public,
cross-sectionally standardized features of one asset at a time, returning the
probability that the asset beats the cross-section over the next ``horizon``
sessions. The classical CVXPY engine is the *muscle*: the probabilities become an
expected-return tilt and ``core.optimization.otimizar`` allocates the money under
the fixed rules (20% per name, 30% per sector, 40% ex-ante vol, no leverage).

Leakage controls
----------------
* ``features_at`` reads only rows up to the decision date.
* A panel sample carries the date its label is realized; a model for a two-year
  block is trained only on samples whose label was realized before 1 January of
  the block's first year.
* Cross-sectional standardization happens inside each date, never with full-sample
  statistics.
* The labels are relative (sign of the forward return minus the cross-sectional
  median), so the net never learns the market level.

The module is only imported by ``scripts/qml_avaliar.py`` and its tests. The
Streamlit app and the paper guards never import PennyLane or Torch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pennylane as qml
import torch
from torch import nn

from core.backtest import CarteiraRecusada, Estrategia
from core.config import DIAS_UTEIS_ANO, ConfigOtimizacao, limites_por_setor
from core.estimation import covariancia_ledoit_wolf, estimar_momentos
from core.optimization import otimizar
from core.risco import LIMITES, Limites, checar_carteira

FEATURE_NAMES = ("ret_21", "ret_63", "momentum_12_1", "vol_21", "vol_63", "drawdown_252")
N_FEATURES = len(FEATURE_NAMES)
WINDOW = 252                 # feature window in sessions: 12 months
MIN_HISTORY = WINDOW         # a name without a full year of features stays out
ANGLE_SCALE = float(np.pi) / 2.0
MODEL_VERSION = 3            # bump when the circuit or the readout changes: it invalidates the cached P(up)


@dataclass(frozen=True)
class QMLConfig:
    """Everything that changes the quantum model or the panel; recorded in every run."""

    n_qubits: int = N_FEATURES
    n_layers: int = 2
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 0.05
    weight_decay: float = 1e-4
    validation_share: float = 0.15
    patience: int = 6
    horizon: int = 63              # label horizon, sessions
    sample_every: int = 21         # panel rows: one decision date every month
    min_history: int = MIN_HISTORY
    seed: int = 0
    device: str = "default.qubit"

    def __post_init__(self) -> None:
        if self.n_qubits != N_FEATURES:
            raise ValueError(f"AngleEmbedding needs one wire per feature: {self.n_qubits} != {N_FEATURES}")
        if self.n_layers < 1 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("n_layers, epochs and batch_size must be >= 1")
        if not 0.0 < self.validation_share < 0.5:
            raise ValueError("validation_share must be in (0, 0.5)")
        if self.horizon < 1 or self.sample_every < 1 or self.min_history < WINDOW:
            raise ValueError("horizon and sample_every must be >= 1 and min_history >= the feature window")


# --- features and labels ----------------------------------------------------------------------------------------------


def features_at(retornos: pd.DataFrame, min_history: int = MIN_HISTORY) -> pd.DataFrame:
    """
    One row per ticker at the last row of ``retornos``.

    The columns are 21- and 63-session total returns, the 12-1 momentum (the last
    21 sessions left out, as in the Conservadora), 21- and 63-session realized
    volatility, and the drawdown from the 252-session high. A ticker with a missing
    value in the window is dropped; the caller intersects with its eligible names.
    """
    if len(retornos) < min_history:
        raise ValueError(f"{len(retornos)} rows < min_history {min_history}")
    janela = retornos.iloc[-min_history:]
    if janela.isna().any().any():
        janela = janela.dropna(axis=1)
    if janela.empty:
        return pd.DataFrame(columns=FEATURE_NAMES, dtype=float)
    preco = (1.0 + janela).cumprod()
    retorno_21 = (1.0 + retornos.iloc[-21:]).prod(min_count=21) - 1.0
    retorno_63 = (1.0 + retornos.iloc[-63:]).prod(min_count=63) - 1.0
    momentum = (1.0 + retornos.iloc[-WINDOW:-21]).prod(min_count=WINDOW - 21) - 1.0
    tabela = pd.DataFrame({
        "ret_21": retorno_21,
        "ret_63": retorno_63,
        "momentum_12_1": momentum,
        "vol_21": retornos.iloc[-21:].std(ddof=1),
        "vol_63": retornos.iloc[-63:].std(ddof=1),
        "drawdown_252": preco.iloc[-1] / preco.cummax().iloc[-1] - 1.0,
    })
    return tabela.reindex(columns=list(FEATURE_NAMES)).replace([np.inf, -np.inf], np.nan).dropna()


def cross_sectional_z(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column z-score inside one date (a zero-dispersion column becomes zero, not NaN)."""
    desvio = frame.std(ddof=0).replace(0.0, np.nan)
    return (frame - frame.mean()).div(desvio).fillna(0.0).clip(-3.0, 3.0)


def angles(frame: pd.DataFrame) -> np.ndarray:
    """Bounded angles in [-pi/2, pi/2]: a hyperbolic squash keeps the embedding away from period aliasing."""
    return np.tanh(frame.to_numpy(dtype=float) / 2.0) * ANGLE_SCALE


@dataclass(frozen=True)
class Panel:
    """Training samples: standardized features, 0/1 relative labels and the date each label is realized."""

    x: np.ndarray                # (n_samples, N_FEATURES)
    y: np.ndarray                # (n_samples,) in {0, 1}
    label_dates: pd.DatetimeIndex
    decision_dates: pd.DatetimeIndex

    def __post_init__(self) -> None:
        """A malformed panel would mis-index the training mask instead of failing: check the shapes here."""
        if self.x.ndim != 2 or self.x.shape[1] != N_FEATURES:
            raise ValueError(f"x must be (n_samples, {N_FEATURES}): {self.x.shape}")
        amostras = self.x.shape[0]
        if not (len(self.y) == len(self.label_dates) == len(self.decision_dates) == amostras):
            raise ValueError(f"panel lengths disagree: x {amostras}, y {len(self.y)}, "
                             f"labels {len(self.label_dates)}, decisions {len(self.decision_dates)}")
        if not np.isin(self.y, (0.0, 1.0)).all():
            raise ValueError("y must be 0/1")
        if not (self.label_dates > self.decision_dates).all():
            raise ValueError("every label must be realized after its decision date")

    def __len__(self) -> int:
        return len(self.y)


def forward_returns(retornos: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Total return from each row to ``horizon`` sessions later; the tail is NaN."""
    composto = (1.0 + retornos).cumprod()
    return composto.shift(-horizon).div(composto) - 1.0


def build_panel(retornos: pd.DataFrame, config: QMLConfig) -> Panel:
    """
    One block per decision date (every ``sample_every`` sessions), labels relative
    to that date's cross-sectional median. The last ``horizon`` sessions have no
    label and stay out.
    """
    datas = retornos.index
    futuro = forward_returns(retornos, config.horizon)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    rotulos: list[pd.Timestamp] = []
    decisoes: list[pd.Timestamp] = []
    primeiro = config.min_history - 1
    for i in range(primeiro, len(datas) - config.horizon, config.sample_every):
        data = datas[i]
        feats = features_at(retornos.iloc[: i + 1], config.min_history)
        if feats.empty:
            continue
        alvo = futuro.iloc[i].reindex(feats.index).dropna()
        if len(alvo) < 3:
            continue
        feats = feats.loc[alvo.index]
        z = cross_sectional_z(feats)
        labels = (alvo > float(alvo.median())).astype(float)
        xs.append(angles(z))
        ys.append(labels.to_numpy())
        rotulos.extend([datas[i + config.horizon]] * len(labels))
        decisoes.extend([data] * len(labels))
    if not xs:
        raise ValueError("the panel is empty: not enough history for one sample")
    return Panel(np.vstack(xs), np.concatenate(ys).astype(np.float64),
                 pd.DatetimeIndex(rotulos), pd.DatetimeIndex(decisoes))


# --- the quantum brain ------------------------------------------------------------------------------------------------


class QuantumDirectionalModel(nn.Module):
    """
    AngleEmbedding of six standardized features on six wires, ``n_layers``
    strongly-entangling layers, and one expectation per wire. A classical linear
    readout (six numbers -> one logit) and a learnable scale/bias map the quantum
    state to P(up); the sigmoid of the logit is the directional call.
    """

    def __init__(self, config: QMLConfig) -> None:
        super().__init__()
        self.config = config
        dev = qml.device(config.device, wires=config.n_qubits, seed=config.seed)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
            qml.AngleEmbedding(inputs, wires=range(config.n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(config.n_qubits))
            return [qml.expval(qml.Z(i)) for i in range(config.n_qubits)]

        self.quantum = qml.qnn.TorchLayer(circuit, {"weights": (config.n_layers, config.n_qubits, 3)})
        self.readout = nn.Linear(config.n_qubits, 1, dtype=torch.float64)
        self.scale = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits, shape ``(batch,)``; ``x`` is ``(batch, N_FEATURES)`` in float64."""
        expectativas = self.quantum(x)                      # (batch, n_qubits)
        return self.scale * self.readout(expectativas).reshape(-1) + self.bias

    @torch.no_grad()
    def probability_up(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        return torch.sigmoid(self.forward(torch.as_tensor(x, dtype=torch.float64))).numpy()


def _fit(model: QuantumDirectionalModel, x: np.ndarray, y: np.ndarray, config: QMLConfig,
         log: Callable[[str], None] | None = None) -> QuantumDirectionalModel:
    """Time-ordered train/validation split, BCE on the logits, early stopping on the validation loss."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))     # four cores, no oversubscription
    corte = max(1, int(len(y) * (1.0 - config.validation_share)))
    treino_x, treino_y = torch.as_tensor(x[:corte]), torch.as_tensor(y[:corte])
    val_x, val_y = torch.as_tensor(x[corte:]), torch.as_tensor(y[corte:])
    otimizador = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    perda = nn.BCEWithLogitsLoss()
    melhor, melhor_estado, sem_melhora = float("inf"), None, 0
    for epoca in range(config.epochs):
        model.train()
        ordem = torch.randperm(len(treino_y))
        soma = 0.0
        for inicio in range(0, len(ordem), config.batch_size):
            lote = ordem[inicio:inicio + config.batch_size]
            otimizador.zero_grad(set_to_none=True)
            saida = model(treino_x[lote])
            custo = perda(saida, treino_y[lote])
            custo.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            otimizador.step()
            soma += float(custo.detach()) * len(lote)
        model.eval()
        with torch.no_grad():
            validacao = float(perda(model(val_x), val_y))
        if validacao < melhor - 1e-6:
            melhor, sem_melhora = validacao, 0
            melhor_estado = {chave: valor.detach().clone() for chave, valor in model.state_dict().items()}
        else:
            sem_melhora += 1
            if sem_melhora >= config.patience:
                break
        if log is not None and (epoca + 1) % 10 == 0:
            log(f"    epoch {epoca + 1:3d}: train {soma / len(treino_y):.4f}, validation {validacao:.4f}")
    if melhor_estado is not None:
        model.load_state_dict(melhor_estado)
    return model


class QMLPredictor:
    """
    Walk-forward probability source. A model is trained once per calendar year on
    the samples whose labels were already realized before 1 January of that year;
    every decision inside the year reuses it. The full return matrix is kept only
    to build features and labels; the training boundary is enforced here.
    """

    def __init__(self, retornos: pd.DataFrame, config: QMLConfig | None = None,
                 log: Callable[[str], None] | None = None) -> None:
        self.config = config if config is not None else QMLConfig()
        self.retornos = retornos
        self.panel = build_panel(retornos, self.config)
        self._log = log or (lambda _: None)
        self._models: dict[int, QuantumDirectionalModel] = {}

    def _training_rows(self, ano: int) -> np.ndarray:
        corte = pd.Timestamp(year=ano, month=1, day=1)
        mascara = np.asarray(self.panel.label_dates < corte)
        if mascara.sum() < 50:
            raise ValueError(f"{ano}: only {int(mascara.sum())} samples with realized labels before the year")
        return np.flatnonzero(mascara)

    def model_for(self, ano: int) -> QuantumDirectionalModel:
        """Train (once) the model that serves ``ano`` and the next year; labels realized before the block's 1 January."""
        chave = ano - ano % 2                     # one circuit per two-year block: the labels stay causal for both years
        if chave not in self._models:
            linhas = self._training_rows(chave)
            self._log(f"  QML: training block {chave}-{chave + 1} on {len(linhas)} samples "
                      f"(labels realized before 01/01/{chave})")
            modelo = QuantumDirectionalModel(self.config)
            self._models[chave] = _fit(modelo, self.panel.x[linhas], self.panel.y[linhas], self.config, self._log)
        return self._models[chave]

    def predict(self, date: pd.Timestamp) -> pd.Series:
        """P(up) per ticker at ``date``, from data up to and including that session."""
        historico = self.retornos.loc[:date]
        feats = features_at(historico, self.config.min_history)
        if feats.empty:
            return pd.Series(dtype=float)
        z = cross_sectional_z(feats)
        modelo = self.model_for(date.year)
        return pd.Series(modelo.probability_up(angles(z)), index=z.index, name="p_up")


# --- the classical muscle ---------------------------------------------------------------------------------------------


UniverseSelector = Callable[[pd.Timestamp, pd.DataFrame], Sequence[str]]


def _rf_na_data(cfg: ConfigOtimizacao, rf_diaria: pd.Series | None, data: pd.Timestamp) -> float:
    if rf_diaria is None or rf_diaria.empty:
        return cfg.taxa_livre_risco
    vigente = rf_diaria.asof(data)
    return cfg.taxa_livre_risco if pd.isna(vigente) else float(vigente) * DIAS_UTEIS_ANO


def _universo_do_seletor(selector: UniverseSelector | None, treino: pd.DataFrame,
                         setores: Mapping[str, str]) -> list[str]:
    """Eligible names of the training window, optionally restricted to a selector's picks."""
    elegiveis = [papel for papel in treino.columns if setores.get(papel)]
    if selector is None:
        return elegiveis
    escolhidos = [papel for papel in selector(treino.index[-1], treino) if papel in elegiveis]
    return escolhidos


def qml_strategy(predictor: QMLPredictor, cfg: ConfigOtimizacao, setores: Mapping[str, str],
                 rf_diaria: pd.Series | None = None, selector: UniverseSelector | None = None,
                 tilt: float = 1.0, base: str = "rf", limites: Limites = LIMITES) -> Estrategia:
    """
    The hybrid arm: QML P(up) -> expected-return tilt, then ``otimizar`` with the fixed caps (the
    sector groups come from ``limites``). ``selector`` narrows the candidates (the Composite's book,
    the Kelly book, ...). ``base`` is the prior the tilt is added to: ``rf`` (the quantum signal is
    the only expected return) or ``jorion`` (Jorion-shrunk sample means plus the quantum tilt, a
    Black-Litterman-lite view that keeps the classical information).
    """
    if tilt <= 0:
        raise ValueError(f"tilt must be > 0: {tilt}")
    if base not in ("rf", "jorion"):
        raise ValueError(f"base must be 'rf' or 'jorion': {base!r}")

    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        nomes = _universo_do_seletor(selector, treino, setores)
        if len(nomes) < 2:
            raise ValueError(f"{treino.index[-1]:%Y-%m-%d}: fewer than 2 eligible names ({len(nomes)})")
        treino = treino[nomes]
        data = treino.index[-1]
        if base == "jorion":
            momentos = estimar_momentos(treino, shrink_mu=True)
            cov, mu_prior = momentos.cov, momentos.mu
        else:
            cov, mu_prior = covariancia_ledoit_wolf(treino) * DIAS_UTEIS_ANO, None
        probabilidades = predictor.predict(data).reindex(nomes).dropna()
        if len(probabilidades) < 2:
            raise ValueError(f"{data:%Y-%m-%d}: QML produced fewer than 2 probabilities")
        cov = cov.loc[probabilidades.index, probabilidades.index]
        sigma = np.sqrt(np.diag(cov.to_numpy(dtype=float)))
        rf = _rf_na_data(cfg, rf_diaria, data)
        if mu_prior is None:
            mu = pd.Series(rf + tilt * (2.0 * probabilidades.to_numpy() - 1.0) * sigma,
                           index=probabilidades.index)
        else:
            mu = mu_prior.reindex(probabilidades.index) + tilt * (2.0 * probabilidades.to_numpy() - 1.0) * sigma
        configuracao = replace(cfg, taxa_livre_risco=rf,
                               grupos=limites_por_setor({papel: setores[papel] for papel in mu.index},
                                                         limites.teto_setor))
        pesos = otimizar(mu, cov, configuracao).pesos
        violacoes = checar_carteira(pesos, setores, cov, limites)
        if violacoes:
            raise CarteiraRecusada(violacoes)
        return pesos[pesos > 1e-8]

    return _estrategia


def estrategia_media_variancia_no_universo(cfg: ConfigOtimizacao, setores: Mapping[str, str],
                                           rf_diaria: pd.Series | None = None,
                                           selector: UniverseSelector | None = None,
                                           limites: Limites = LIMITES) -> Estrategia:
    """
    The classical control of the same restricted universe: Jorion-shrunk sample
    means and Ledoit-Wolf covariance, no quantum signal. Isolates how much of a
    result comes from the names rather than from the QML arm.
    """
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        nomes = _universo_do_seletor(selector, treino, setores)
        if len(nomes) < 2:
            raise ValueError(f"{treino.index[-1]:%Y-%m-%d}: fewer than 2 eligible names ({len(nomes)})")
        treino = treino[nomes]
        momentos = estimar_momentos(treino, shrink_mu=True)
        rf = _rf_na_data(cfg, rf_diaria, treino.index[-1])
        configuracao = replace(cfg, taxa_livre_risco=rf,
                               grupos=limites_por_setor({papel: setores[papel] for papel in treino.columns},
                                                         limites.teto_setor))
        pesos = otimizar(momentos.mu, momentos.cov, configuracao).pesos
        violacoes = checar_carteira(pesos, setores, momentos.cov, limites)
        if violacoes:
            raise CarteiraRecusada(violacoes)
        return pesos[pesos > 1e-8]

    return _estrategia
