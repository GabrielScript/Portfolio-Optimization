"""Poder dos portões do S2 por simulação (IC plantado no painel mensal), sem rede."""

import numpy as np
import pandas as pd
import pytest

from core.poder import Portao, ic_por_mes, sinal_plantado, taxa_de_aprovacao

MESES = pd.date_range("2010-01-01", periods=3, freq="MS")


def test_ic_de_spearman_por_mes_so_nos_papeis_com_sinal_e_retorno():
    # jan: o sinal ordena igual ao retorno (IC 1); fev: ao contrário (IC -1); mar: C sem retorno fica de fora
    retornos = pd.DataFrame({"A": [0.01, 0.03, 0.02], "B": [0.02, 0.02, 0.01], "C": [0.03, 0.01, np.nan]},
                            index=MESES)
    sinal = pd.DataFrame({"A": [1.0, 1.0, 2.0], "B": [2.0, 2.0, 1.0], "C": [3.0, 3.0, 9.0]}, index=MESES)
    assert ic_por_mes(sinal, retornos).tolist() == pytest.approx([1.0, -1.0, 1.0])


def test_sinal_plantado_tem_em_media_o_ic_de_spearman_pedido():
    # 300 meses x 150 papéis, retornos de cauda gorda: erro padrão do IC médio ~ 1/sqrt(150 x 300) = 0,005
    gerador = np.random.default_rng(1)
    retornos = pd.DataFrame(gerador.standard_t(3, (300, 150)) * 0.08)
    retornos.iloc[:, :20] = np.where(gerador.random((300, 20)) < 0.3, np.nan, retornos.iloc[:, :20])
    sinal = sinal_plantado(retornos, ic=0.05, gerador=gerador)
    assert sinal.isna().equals(retornos.isna())
    assert ic_por_mes(sinal, retornos).mean() == pytest.approx(0.05, abs=0.012)


def test_portao_pede_ic_medio_e_t_juntos():
    portao = Portao("IC >= 0,03 e t >= 2", ic_minimo=0.03, t_minimo=2.0)
    alto_e_firme = pd.Series([0.04, 0.05, 0.03, 0.04] * 25)              # média 0,04, t ~ 50
    firme_mas_baixo = pd.Series([0.02, 0.025, 0.015, 0.02] * 25)         # média 0,02 < 0,03
    alto_mas_ruidoso = pd.Series([0.5, -0.4, 0.3, -0.2] * 25)            # média 0,05, t ~ 1,3
    assert [portao.aprova(ics) for ics in (alto_e_firme, firme_mas_baixo, alto_mas_ruidoso)] == [True, False, False]


def test_falso_positivo_do_t_maior_que_2_fica_perto_de_2_por_cento():
    # IC verdadeiro 0 e portão só no t (unilateral): P[t_119 >= 2] = 2,4%
    gerador = np.random.default_rng(2)
    retornos = pd.DataFrame(gerador.standard_t(3, (120, 100)) * 0.08)
    taxa = taxa_de_aprovacao(retornos, 0.0, [Portao("t >= 2", -1.0, 2.0)], simulacoes=400, gerador=gerador)
    assert taxa["t >= 2"] == pytest.approx(0.024, abs=0.025)


def test_ic_que_varia_de_mes_a_mes_tira_poder():
    # 60 meses x 100 papéis, IC 0,03: o desvio de 0,10 no IC do mês soma ao ruído de corte (~0,10) e o t cai
    gerador = np.random.default_rng(4)
    retornos = pd.DataFrame(gerador.standard_t(3, (60, 100)) * 0.08)
    portao = [Portao("t >= 2", -1.0, 2.0)]
    fixo = taxa_de_aprovacao(retornos, 0.03, portao, simulacoes=300, gerador=gerador)["t >= 2"]
    variando = taxa_de_aprovacao(retornos, 0.03, portao, simulacoes=300, gerador=gerador, desvio_do_ic=0.10)["t >= 2"]
    assert variando < fixo - 0.10
