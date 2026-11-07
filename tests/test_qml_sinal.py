"""The forecast models of the hybrid QPO (qpo_kaggle/qml_sinal.py), section 2-3 of the design."""

import numpy as np
import pytest

qml = pytest.importorskip("pennylane")
torch = pytest.importorskip("torch")

from qpo_kaggle.qml_sinal import (N_ATRIBUTOS, Mlp, Pacote, Treino, Vqc, atributos_de_fourier,  # noqa: E402
                                 prever_fourier, prever_torch, ridge_gcv)


def _angulos(linhas: int, semente: int = 0) -> np.ndarray:
    return np.random.default_rng(semente).uniform(-np.pi / 2, np.pi / 2, (linhas, N_ATRIBUTOS))


def test_the_fourier_features_are_the_729_products_of_one_cos_and_sin_per_wire():
    theta = _angulos(5)
    x = atributos_de_fourier(theta)
    assert x.shape == (5, 3 ** N_ATRIBUTOS)
    assert np.allclose(x[:, 0], 1.0)                                    # all wires contribute 1
    produto = np.cos(theta[:, 0]) * np.sin(theta[:, 3])
    assert any(np.allclose(x[:, j], produto) for j in range(x.shape[1]))
    assert np.linalg.matrix_rank(atributos_de_fourier(_angulos(1_000))) == 3 ** N_ATRIBUTOS


def test_the_circuit_has_61_parameters_and_the_classical_twin_has_about_as_many():
    assert sum(p.numel() for p in Vqc(semente=0).parameters()) == 61
    assert sum(p.numel() for p in Mlp(semente=0).parameters()) == 57


def test_the_circuit_lies_exactly_in_the_span_of_the_729_features():
    # Prop. 1: this is what makes the Fourier ridge the circuit's dequantized twin
    modelo = Vqc(semente=3)
    theta = _angulos(1_500, 1)
    with torch.no_grad():
        saida = modelo(torch.tensor(theta, dtype=torch.float64)).numpy()
    x = atributos_de_fourier(theta)
    coeficientes, *_ = np.linalg.lstsq(x, saida, rcond=None)
    assert np.abs(x @ coeficientes - saida).max() < 1e-10


def test_the_circuit_gradient_matches_finite_differences():
    modelo = Vqc(semente=5)
    theta = torch.tensor(_angulos(4), dtype=torch.float64)
    alvo = torch.tensor([0.5, -0.5, 0.2, 0.1], dtype=torch.float64)

    def perda() -> torch.Tensor:
        return ((modelo(theta) - alvo) ** 2).mean()

    modelo.zero_grad(set_to_none=True)
    perda().backward()
    parametro = next(modelo.parameters())
    analitico = parametro.grad.flatten()[0].item()
    passo = 1e-6
    with torch.no_grad():
        parametro.flatten()[0] += passo
        cima = perda().item()
        parametro.flatten()[0] -= 2 * passo
        baixo = perda().item()
        parametro.flatten()[0] += passo
    assert analitico == pytest.approx((cima - baixo) / (2 * passo), abs=1e-6)


def test_a_model_is_reproducible_by_seed_and_differs_across_seeds():
    theta = torch.tensor(_angulos(3), dtype=torch.float64)
    with torch.no_grad():
        for classe in (Vqc, Mlp):
            assert torch.equal(classe(semente=1)(theta), classe(semente=1)(theta))
            assert not torch.equal(classe(semente=1)(theta), classe(semente=2)(theta))


# --- ridge and the walk-forward --------------------------------------------------------------------------------------


def _pacote(meses: int = 12, nomes: int = 60, semente: int = 0, ruido: float = 0.3) -> Pacote:
    """Months 0..meses-1; y depends on theta[0]; each month from 3 on is forecast from months up to t - 3."""
    gerador = np.random.default_rng(semente)
    theta = gerador.uniform(-np.pi / 2, np.pi / 2, (meses * nomes, N_ATRIBUTOS))
    mes = np.repeat(np.arange(meses), nomes)
    y = np.sin(theta[:, 0]) + ruido * gerador.normal(size=len(theta))
    previstos = np.arange(3, meses)
    treinos = [np.flatnonzero(mes <= t - 3) for t in previstos]
    return Pacote(theta, y, mes, previstos, treinos)


CURTA = Treino(passos_iniciais=30, passos_seguintes=5, lote=64, taxa=0.02)


def test_the_ridge_recovers_a_clean_linear_rule_and_gcv_shrinks_pure_noise():
    gerador = np.random.default_rng(0)
    x = gerador.normal(size=(500, 4))
    beta = np.array([1.0, -2.0, 0.5, 0.0])
    ajuste = ridge_gcv(x, x @ beta + 3.0)
    assert ajuste.prever(x) == pytest.approx(x @ beta + 3.0, abs=1e-2)
    ruido = ridge_gcv(x, gerador.normal(size=500))
    assert np.abs(ruido.prever(x)).max() < 0.3


def test_the_fourier_twin_forecasts_a_planted_signal_and_only_reads_its_training_rows():
    pacote = _pacote()
    previsao = prever_fourier(pacote)
    linhas = pacote.mes == 8
    assert np.corrcoef(previsao[linhas], pacote.y[linhas])[0, 1] > 0.8
    assert np.isnan(previsao[pacote.mes < 3]).all()
    vazado = pacote.y.copy()
    vazado[pacote.mes >= 6] = 99.0                                    # the labels month 8 may not read (8 - 3 = 5)
    outra = prever_fourier(Pacote(pacote.theta, vazado, pacote.mes, pacote.previstos, pacote.treinos))
    assert previsao[linhas] == pytest.approx(outra[linhas])
    assert not np.allclose(previsao[pacote.mes == 11], outra[pacote.mes == 11])       # month 11 does read month 8


@pytest.mark.parametrize("classe", [Mlp, Vqc])
def test_a_torch_model_learns_a_planted_signal_and_does_not_read_future_labels(classe, tmp_path):
    pacote = _pacote(meses=6, nomes=40)
    previsao = prever_torch(classe, pacote, 0, CURTA, tmp_path / "a")
    assert np.isfinite(previsao[pacote.mes == 5]).all()
    vazado = pacote.y.copy()
    vazado[pacote.mes >= 2] = 99.0                                    # month 4 may read only months 0 and 1
    outra = prever_torch(classe, Pacote(pacote.theta, vazado, pacote.mes, pacote.previstos, pacote.treinos), 0, CURTA,
                         tmp_path / "b")
    assert previsao[pacote.mes == 4] == pytest.approx(outra[pacote.mes == 4])


def test_a_planted_signal_gives_a_positive_correlation_after_enough_steps(tmp_path):
    pacote = _pacote(meses=8, nomes=80)
    previsao = prever_torch(Vqc, pacote, 0, Treino(passos_iniciais=150, passos_seguintes=20, lote=128, taxa=0.05),
                            tmp_path / "c")
    linhas = pacote.mes == 7
    assert np.corrcoef(previsao[linhas], pacote.y[linhas])[0, 1] > 0.5


def test_a_stopped_run_resumes_to_the_same_forecast(tmp_path):
    pacote = _pacote(meses=7, nomes=30)
    inteira = prever_torch(Mlp, pacote, 1, CURTA, tmp_path / "inteira")
    parcial = Pacote(pacote.theta, pacote.y, pacote.mes, pacote.previstos[:2], pacote.treinos[:2])
    prever_torch(Mlp, parcial, 1, CURTA, tmp_path / "retomada")
    retomada = prever_torch(Mlp, pacote, 1, CURTA, tmp_path / "retomada")
    assert retomada[pacote.mes >= 3] == pytest.approx(inteira[pacote.mes >= 3])


def test_the_state_file_survives_a_windows_lock_on_the_swap(tmp_path, monkeypatch):
    # 2026-11-07: Windows denied the .tmp -> .pt swap after 34 months (a scanner holding the new file); the run died
    from pathlib import Path

    from qpo_kaggle import qml_sinal

    real = Path.replace
    falhas = {"restam": 2}

    def trava(self, alvo):
        if falhas["restam"] > 0:
            falhas["restam"] -= 1
            raise PermissionError(5, "Acesso negado")
        return real(self, alvo)

    monkeypatch.setattr(Path, "replace", trava)
    monkeypatch.setattr(qml_sinal.time, "sleep", lambda _: None)
    arquivo = tmp_path / "estado_0.pt"
    qml_sinal._gravar_estado(arquivo, {"feitos": 3})
    assert torch.load(arquivo, weights_only=True)["feitos"] == 3 and falhas["restam"] == 0
