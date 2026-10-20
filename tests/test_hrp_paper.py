import numpy as np
import pandas as pd
import pytest

from core.hierarchical import otimizar_hrp, otimizar_hrp_arvore
from core.hrp_paper import otimizar_hrp_arvore_paper, otimizar_hrp_paper


def _hrp_do_paper(cov: pd.DataFrame) -> pd.Series:
    """
    Apêndice A.3 de López de Prado (2016), portado para Python 3 sem mudar a lógica: ``linkage`` recebe a matriz
    quadrada de distâncias, que o scipy trata como observações (daí a distância euclidiana entre colunas, d~).
    """
    from scipy.cluster.hierarchy import linkage

    def variancia_cluster(itens):
        sub = cov.loc[itens, itens].values
        w = 1.0 / np.diag(sub)
        w /= w.sum()
        return float(w @ sub @ w)

    sigma = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(sigma, sigma)
    link = linkage(np.clip((1.0 - corr) / 2.0, 0.0, None) ** 0.5, "single").astype(int)  # -1e-16 na diagonal
    n = link[-1, 3]
    ordem = [link[-1, 0], link[-1, 1]]
    while max(ordem) >= n:
        ordem = [x for i in ordem for x in ((link[i - n, 0], link[i - n, 1]) if i >= n else (i,))]
    ordem = [cov.index[i] for i in ordem]
    pesos = pd.Series(1.0, index=ordem)
    grupos = [ordem]
    while grupos:
        grupos = [g[j:k] for g in grupos for j, k in ((0, len(g) // 2), (len(g) // 2, len(g))) if len(g) > 1]
        for i in range(0, len(grupos), 2):
            v0, v1 = variancia_cluster(grupos[i]), variancia_cluster(grupos[i + 1])
            alpha = 1.0 - v0 / (v0 + v1)
            pesos[grupos[i]] *= alpha
            pesos[grupos[i + 1]] *= 1.0 - alpha
    return pesos.reindex(cov.index)


@pytest.fixture
def retornos_em_blocos() -> pd.DataFrame:
    """O ``generateData`` do paper: 5 séries independentes + 5 cópias ruidosas delas (sigma 0,25), 1.000 pregões."""
    gerador = np.random.default_rng(3)
    x = gerador.normal(0.0, 0.01, (1_000, 5))
    copias = x[:, [0, 2, 2, 4, 1]] + gerador.normal(0.0, 0.0025, (1_000, 5))
    return pd.DataFrame(np.hstack([x, copias]), columns=[f"P{i}" for i in range(10)])


def test_hrp_paper_reproduz_o_codigo_do_apendice(retornos_em_blocos):
    cov = retornos_em_blocos.cov()
    pd.testing.assert_series_equal(otimizar_hrp_paper(cov), _hrp_do_paper(cov), check_names=False,
                                   rtol=1e-12, atol=1e-14)


def test_hrp_paper_difere_do_hrp_com_d_nos_blocos(retornos_em_blocos):
    """Garante que o teste acima discrimina: a d muda a ordem e os pesos nestes dados."""
    cov = retornos_em_blocos.cov()
    assert (otimizar_hrp_paper(cov) - otimizar_hrp(cov)).abs().max() > 0.01


@pytest.mark.parametrize(("do_paper", "do_core"), [(otimizar_hrp_paper, otimizar_hrp),
                                                   (otimizar_hrp_arvore_paper, otimizar_hrp_arvore)])
def test_com_distancia_d_iguala_o_core_hierarchical(retornos_em_blocos, do_paper, do_core):
    cov = retornos_em_blocos.cov()
    pd.testing.assert_series_equal(do_paper(cov, distancia="d"), do_core(cov), rtol=0, atol=0)


@pytest.mark.parametrize("alocador", [otimizar_hrp_paper, otimizar_hrp_arvore_paper])
def test_pesos_positivos_somam_um(retornos_em_blocos, alocador):
    pesos = alocador(retornos_em_blocos.cov())
    assert (pesos > 0).all()
    assert pesos.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("alocador", [otimizar_hrp_paper, otimizar_hrp_arvore_paper])
def test_rejeita_distancia_desconhecida(retornos_em_blocos, alocador):
    with pytest.raises(ValueError, match="distancia"):
        alocador(retornos_em_blocos.cov(), distancia="euclidiana")
