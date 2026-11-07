"""The KAN forecast models (qpo_kaggle/kan.py), sections 2-3 of docs/superpowers/specs/2026-11-07-kan-design.md."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qpo_kaggle.kan import Kan1, Kan2, KanCamada, curvas  # noqa: E402
from qpo_kaggle.qml_pacote import Pacote  # noqa: E402
from qpo_kaggle.qml_sinal import N_ATRIBUTOS, Treino, prever_torch  # noqa: E402

FAIXA = (-np.pi / 2, np.pi / 2)


def _angulos(linhas: int, semente: int = 0) -> np.ndarray:
    return np.random.default_rng(semente).uniform(*FAIXA, (linhas, N_ATRIBUTOS))


def test_the_spline_basis_reproduces_any_line_on_the_whole_grid_range_endpoints_included():
    # Section 2 (a), Marsden's identity: the shallow KAN nests the ridge on the same angles
    camada = KanCamada(1, 1, semente=0, faixa=FAIXA)
    u = torch.linspace(*FAIXA, 401, dtype=torch.float64).unsqueeze(-1)
    base = camada.base_de_splines(u)[:, 0, :].numpy()              # (401, 8)
    assert base.shape == (401, 8)
    assert np.abs(base.sum(axis=1) - 1.0).max() < 1e-12            # partition of unity, the constant
    reta = 0.7 * u[:, 0].numpy() - 0.2
    coeficientes, *_ = np.linalg.lstsq(base, reta, rcond=None)
    assert np.abs(base @ coeficientes - reta).max() < 1e-12


def test_the_spline_part_vanishes_outside_the_extended_grid():
    # faixa (-1, 1), grid 5, cubic: knots run from -2.2 to 2.2; inside that span but outside the range it does not vanish
    camada = KanCamada(1, 1, semente=0, faixa=(-1.0, 1.0))
    fora = torch.tensor([[-3.0], [2.5]], dtype=torch.float64)
    assert camada.base_de_splines(torch.tensor([[-2.0]], dtype=torch.float64)).sum() > 0.0
    assert torch.all(camada.base_de_splines(fora) == 0.0)


def test_parameter_counts_are_the_pre_registered_ones():
    assert sum(p.numel() for p in Kan1(semente=0).parameters()) == 60
    assert sum(p.numel() for p in Kan2(semente=0).parameters()) == 140


def test_kan1_can_be_set_to_the_equal_weight_average_of_the_angles():
    # Section 2 (b): the class holds an equal-weight average, so the test can always fall back to it
    modelo = Kan1(semente=1)
    camada = modelo.camadas[0]
    grade = torch.linspace(*FAIXA, 400, dtype=torch.float64).unsqueeze(-1)
    base = camada.base_de_splines(grade)[:, 0, :]
    linha = torch.linalg.lstsq(base, grade[:, 0] / N_ATRIBUTOS).solution
    with torch.no_grad():
        camada.peso_base.zero_()
        camada.escala.fill_(1.0)
        camada.coeficientes.copy_(linha.expand_as(camada.coeficientes))
        theta = torch.tensor(_angulos(50, 2), dtype=torch.float64)
        assert torch.allclose(modelo(theta), theta.mean(dim=1), atol=1e-12)


def test_same_seed_same_net_and_float64():
    theta = torch.tensor(_angulos(20, 3), dtype=torch.float64)
    with torch.no_grad():
        a, b, c = Kan2(semente=4)(theta), Kan2(semente=4)(theta), Kan2(semente=5)(theta)
    assert a.dtype == torch.float64 and a.shape == (20,)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_the_gradient_matches_finite_differences():
    modelo = Kan2(semente=6)
    theta = torch.tensor(_angulos(4, 4), dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x: modelo(x), (theta.requires_grad_(),), eps=1e-6, atol=1e-6)


def test_kan1_trains_through_the_mlp_loop_and_learns_a_curved_signal(tmp_path):
    # the walk-forward of 213, unchanged: forecasts only in the forecast months, a resumable state on disk
    rng = np.random.default_rng(7)
    meses = np.repeat(np.arange(6), 400)
    theta = rng.uniform(*FAIXA, (len(meses), N_ATRIBUTOS))
    y = np.sin(2.0 * theta[:, 0]) + 0.5 * theta[:, 1] ** 2 + 0.05 * rng.standard_normal(len(meses))
    previstos = np.array([4, 5])
    treinos = [np.flatnonzero(meses <= 1), np.flatnonzero(meses <= 2)]
    pacote = Pacote(theta, y, meses, previstos, treinos)
    previsao = prever_torch(Kan1, pacote, 0, Treino(passos_iniciais=400, passos_seguintes=50), tmp_path / "estado_KAN-1")
    assert np.isnan(previsao[meses < 4]).all() and np.isfinite(previsao[meses >= 4]).all()
    alvo = meses >= 4
    assert np.corrcoef(previsao[alvo], y[alvo])[0, 1] > 0.9
    assert (tmp_path / "estado_KAN-1" / "estado_0.pt").exists()


def test_curves_give_each_input_its_learned_function_and_its_non_linearity():
    modelo = Kan1(semente=8)
    tabela = curvas(modelo, pontos=21)
    assert list(tabela.columns) == ["atributo", "u", "phi", "nao_linearidade"]
    assert len(tabela) == N_ATRIBUTOS * 21
    assert tabela["u"].min() == pytest.approx(-np.pi / 2) and tabela["u"].max() == pytest.approx(np.pi / 2)
    assert tabela["nao_linearidade"].between(0.0, 1.0).all()
    # a curve set to a line has no non-linearity
    camada = modelo.camadas[0]
    with torch.no_grad():
        camada.peso_base.zero_()
        grade = torch.linspace(*FAIXA, 400, dtype=torch.float64).unsqueeze(-1)
        base = camada.base_de_splines(grade)[:, 0, :]
        camada.coeficientes[0, 0] = torch.linalg.lstsq(base, 3.0 * grade[:, 0]).solution
    assert curvas(modelo, pontos=21).query("atributo == 0")["nao_linearidade"].iloc[0] < 1e-12
