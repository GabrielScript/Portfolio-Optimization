"""PSR (Bailey e López de Prado, 2012) e DSR (Bailey e López de Prado, 2014)."""

import numpy as np
import pandas as pd
import pytest

from core.metrics import (MomentosSharpe, momentos_sharpe, sharpe_probabilistico,
                          sharpe_referencia_deflacionado)


def test_momentos_sharpe_de_serie_calculada_a_mao():
    """Média 0,005; desvio (ddof=1) sqrt(0,0005/3); simétrica; curtose bruta m4/m2² = 1,64."""
    m = momentos_sharpe(pd.Series([0.01, -0.01, 0.02, 0.0]))
    assert m.sharpe == pytest.approx(0.387298, abs=1e-6)
    assert m.n_obs == 4
    assert m.assimetria == pytest.approx(0.0, abs=1e-12)
    assert m.curtose == pytest.approx(1.64, abs=1e-9)


def test_psr_vale_meio_quando_a_referencia_e_o_proprio_sharpe():
    m = MomentosSharpe(sharpe=0.1, n_obs=500, assimetria=-1.0, curtose=6.0)
    assert sharpe_probabilistico(m, sharpe_referencia=0.1) == pytest.approx(0.5, abs=1e-12)


def test_dsr_reproduz_o_exemplo_do_artigo():
    """Bailey e López de Prado (2014): SR anual 2,5; T=1250; assimetria −3; curtose 10;
    N=100 tentativas; V[SR] anual 0,5 (250 pregões) → SR0 = 0,1132 e DSR = 0,9004."""
    referencia = sharpe_referencia_deflacionado(variancia_sharpes=0.5 / 250, n_tentativas=100)
    assert referencia == pytest.approx(0.1132, abs=5e-4)
    m = MomentosSharpe(sharpe=2.5 / np.sqrt(250), n_obs=1250, assimetria=-3.0, curtose=10.0)
    assert sharpe_probabilistico(m, sharpe_referencia=referencia) == pytest.approx(0.9004, abs=1e-3)


def test_uma_tentativa_so_nao_deflaciona():
    assert sharpe_referencia_deflacionado(variancia_sharpes=0.3, n_tentativas=1) == 0.0


def test_referencia_deflacionada_cresce_com_o_numero_de_tentativas():
    assert (sharpe_referencia_deflacionado(0.002, 10)
            < sharpe_referencia_deflacionado(0.002, 100))


def test_serie_constante_nao_tem_psr():
    m = momentos_sharpe(pd.Series([0.001] * 30))
    assert np.isnan(m.sharpe)
    assert np.isnan(sharpe_probabilistico(m))
