"""Parâmetros da sidebar → configuração do core (só ações)."""

import numpy as np
import pandas as pd

from core.config import ConfigBacktest, Objetivo, limites_por_setor
from ui.servicos import ParametrosOtimizacao, carregar_satelite_registrado, rodar_walk_forward


def test_parametros_so_de_acoes_viram_config_sem_limites_de_grupo():
    params = ParametrosOtimizacao(objetivo=Objetivo.KELLY.value, peso_maximo=0.2, n_ativos_max=10,
                                  vol_maxima=0.4, aversao_risco=2.0)
    cfg = params.para_config(0.12)
    assert cfg.grupos == ()
    assert (cfg.objetivo, cfg.peso_maximo, cfg.n_ativos_max, cfg.vol_maxima, cfg.aversao_risco,
            cfg.taxa_livre_risco) == (Objetivo.KELLY, 0.2, 10, 0.4, 2.0, 0.12)


def test_walk_forward_rebalanceia_pela_primeira_vez_depois_da_janela_de_treino_dada():
    datas = pd.bdate_range("2020-01-01", periods=400)
    retornos = pd.DataFrame(np.random.default_rng(1).normal(0.0005, 0.01, (400, 4)), index=datas,
                            columns=["A.SA", "B.SA", "C.SA", "D.SA"])
    params = ParametrosOtimizacao(objetivo=Objetivo.MIN_VOL.value, peso_maximo=0.5, n_ativos_max=4,
                                  vol_maxima=None, aversao_risco=2.0)
    cfg_bt = ConfigBacktest(janela_treino=300, janela_teste=50, capital_inicial=1_000.0, custo_bps=0.0)

    resultados = rodar_walk_forward(retornos, params, cfg_bt, pd.Series(dtype=float), 0.1, False)

    assert {bt.pesos_alvo.index[0] for bt in resultados.values()} == {datas[300]}


def test_teto_por_setor_vira_um_limite_de_grupo_por_setor():
    params = ParametrosOtimizacao(objetivo=Objetivo.KELLY.value, peso_maximo=0.2, n_ativos_max=10,
                                  vol_maxima=0.4, aversao_risco=2.0, teto_setor=0.3,
                                  setores=(("A.SA", "Energia"), ("B.SA", "Bancos"), ("C.SA", "Energia")))
    cfg = params.para_config(0.12)
    assert cfg.grupos == limites_por_setor({"A.SA": "Energia", "B.SA": "Bancos", "C.SA": "Energia"}, 0.3)


def test_sem_registro_do_paper_nao_ha_satelite(tmp_path):
    # deploy sem o paper/carteiras.csv, ou com só o cabeçalho: a tela avisa em vez de quebrar
    so_cabecalho = tmp_path / "vazio.csv"
    so_cabecalho.write_text("data,ticker,peso,preco\n", encoding="utf-8")
    assert carregar_satelite_registrado(tmp_path / "nao_existe.csv") is None
    assert carregar_satelite_registrado(so_cabecalho) is None


def test_cdi_of_the_screen_comes_from_the_lake_copy_without_asking_the_bcb(tmp_path, monkeypatch):
    from datetime import date

    from core import cdi_lago, market_data
    from ui import servicos

    path = tmp_path / "cdi_daily.parquet"
    days = pd.to_datetime(["2025-01-02", "2026-10-16", "2026-10-19"])
    pd.DataFrame({"date": days.date, "cdi_pct": 0.05}).to_parquet(path)
    monkeypatch.setattr(cdi_lago, "LAKE_CDI_FILE", path)

    def no_network(*args, **kwargs):
        raise AssertionError("the BCB must not be asked")

    monkeypatch.setattr(market_data, "baixar_cdi_diario", no_network)
    servicos.carregar_cdi.clear()
    cdi = servicos.carregar_cdi(1, date(2026, 10, 20))
    servicos.carregar_cdi.clear()
    assert cdi.index.tolist() == days[1:].tolist()
