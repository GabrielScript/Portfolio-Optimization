"""Parâmetros da sidebar → configuração do core (só ações)."""

from core.config import Objetivo
from ui.servicos import ParametrosOtimizacao


def test_parametros_so_de_acoes_viram_config_sem_limites_de_grupo():
    params = ParametrosOtimizacao(objetivo=Objetivo.KELLY.value, peso_maximo=0.2, n_ativos_max=10,
                                  vol_maxima=0.4, aversao_risco=2.0)
    cfg = params.para_config(0.12)
    assert cfg.grupos == ()
    assert (cfg.objetivo, cfg.peso_maximo, cfg.n_ativos_max, cfg.vol_maxima, cfg.aversao_risco,
            cfg.taxa_livre_risco) == (Objetivo.KELLY, 0.2, 10, 0.4, 2.0, 0.12)
