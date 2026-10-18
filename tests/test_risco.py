"""Regras de risco fixas (core/risco.py): aprovam ou recusam, nunca mudam pesos."""

import numpy as np
import pandas as pd
import pytest

from core.config import ConfigOtimizacao, limites_por_setor
from core.risco import (LIMITES, Limites, Regra, Violacao, checar_carteira, pode_negociar, stop_relativo,
                        validar_config)


def test_limites_fixos_tem_os_valores_do_plano():
    # mudar um valor aqui é pré-registro novo (PLAN.md, "Regras de risco fixas")
    assert LIMITES == Limites(alavancagem_maxima=1.0, peso_maximo_acao=0.20, peso_maximo_emissor=0.20,
                              teto_setor=0.30, vol_maxima=0.40, fracao_maxima_satelite=0.50,
                              fracao_padrao_satelite=0.30, queda_relativa_stop=0.25, perda_diaria_maxima=0.05,
                              pregoes_congelados=1)


SETORES = {"PETR4.SA": "Petróleo", "VALE3.SA": "Materiais Básicos", "ITUB4.SA": "Financeiro",
           "ITUB3.SA": "Financeiro", "WEGE3.SA": "Bens Industriais", "ABEV3.SA": "Consumo não Cíclico",
           "TAEE11.SA": "Utilidade Pública", "EGIE3.SA": "Utilidade Pública", "CPLE6.SA": "Utilidade Pública",
           "RENT3.SA": "Consumo Cíclico", "RADL3.SA": "Saúde"}


def _regras(violacoes) -> set:
    return {v.regra for v in violacoes}


def test_peso_negativo_viola():
    pesos = pd.Series({"PETR4.SA": 0.20, "VALE3.SA": 0.20, "WEGE3.SA": 0.20, "ABEV3.SA": 0.20,
                       "RENT3.SA": 0.20, "RADL3.SA": -0.05})
    assert checar_carteira(pesos, SETORES) == [Violacao(Regra.PESO_NEGATIVO, "RADL3.SA", -0.05, 0.0)]


def _unica(violacoes: list[Violacao]) -> Violacao:
    assert len(violacoes) == 1, violacoes
    return violacoes[0]


def test_soma_acima_de_100_por_cento_viola_alavancagem():
    pesos = pd.Series({"PETR4.SA": 0.20, "VALE3.SA": 0.20, "WEGE3.SA": 0.20, "ABEV3.SA": 0.20,
                       "RENT3.SA": 0.11, "RADL3.SA": 0.10})
    violacao = _unica(checar_carteira(pesos, SETORES))
    assert (violacao.regra, violacao.alvo, violacao.limite) == (Regra.ALAVANCAGEM, "carteira", 1.0)
    assert violacao.valor == pytest.approx(1.01)


def test_setor_acima_de_30_por_cento_viola():
    pesos = pd.Series({"TAEE11.SA": 0.12, "EGIE3.SA": 0.12, "CPLE6.SA": 0.12, "PETR4.SA": 0.20, "VALE3.SA": 0.20,
                       "WEGE3.SA": 0.20, "ABEV3.SA": 0.04})
    violacao = _unica(checar_carteira(pesos, SETORES))
    assert (violacao.regra, violacao.alvo, violacao.limite) == (Regra.SETOR, "Utilidade Pública", 0.30)
    assert violacao.valor == pytest.approx(0.36)


def test_duas_classes_do_mesmo_emissor_somam():
    pesos = pd.Series({"ITUB3.SA": 0.12, "ITUB4.SA": 0.12, "PETR4.SA": 0.20, "VALE3.SA": 0.20, "WEGE3.SA": 0.20,
                       "ABEV3.SA": 0.16})
    violacao = _unica(checar_carteira(pesos, SETORES))
    assert (violacao.regra, violacao.alvo, violacao.limite) == (Regra.EMISSOR, "ITUB", 0.20)
    assert violacao.valor == pytest.approx(0.24)


def test_acao_acima_de_20_por_cento_viola():
    pesos = pd.Series({"PETR4.SA": 0.25, "VALE3.SA": 0.20, "WEGE3.SA": 0.20, "ABEV3.SA": 0.20, "RENT3.SA": 0.15})
    assert checar_carteira(pesos, SETORES) == [Violacao(Regra.ACAO, "PETR4.SA", 0.25, 0.20)]


def test_papel_sem_setor_viola():
    # deslistada fora do assets.py: não passa calada
    pesos = pd.Series({"PETR4.SA": 0.20, "VALE3.SA": 0.20, "WEGE3.SA": 0.20, "ABEV3.SA": 0.20, "OIBR3.SA": 0.20})
    assert checar_carteira(pesos, SETORES) == [Violacao(Regra.SEM_SETOR, "OIBR3.SA", 0.20, 0.0)]


def test_vol_ex_ante_acima_de_40_por_cento_viola():
    # 5 papéis independentes com vol de 100% a.a., 20% cada: vol da carteira = raiz(5 x 0,2² x 1) = 44,7%
    tickers = ["PETR4.SA", "VALE3.SA", "WEGE3.SA", "ABEV3.SA", "RENT3.SA"]
    pesos = pd.Series(0.20, index=tickers)
    cov = pd.DataFrame(np.eye(5), index=tickers, columns=tickers)
    violacao = _unica(checar_carteira(pesos, SETORES, cov=cov))
    assert (violacao.regra, violacao.alvo, violacao.limite) == (Regra.VOL, "carteira", 0.40)
    assert violacao.valor == pytest.approx(np.sqrt(0.2))


def test_config_que_afrouxa_e_recusada():
    setores = limites_por_setor(SETORES, 0.30)
    afrouxadas = {
        Regra.ACAO: ConfigOtimizacao(peso_maximo=0.25, grupos=setores),
        Regra.VOL: ConfigOtimizacao(vol_maxima=None, grupos=setores),
        Regra.SETOR: ConfigOtimizacao(grupos=limites_por_setor(SETORES, 1.0)),
    }
    for regra, cfg in afrouxadas.items():
        assert {v.regra for v in validar_config(cfg)} == {regra}, cfg
    assert [v.regra for v in validar_config(ConfigOtimizacao(grupos=()))] == [Regra.SETOR]
    assert validar_config(ConfigOtimizacao(peso_maximo=0.15, n_ativos_max=10,
                                           grupos=limites_por_setor(SETORES, 0.20))) == []


def _serie(*valores: float) -> pd.Series:
    return pd.Series(valores, index=pd.bdate_range("2026-01-01", periods=len(valores)))


def test_stop_relativo_dispara_com_queda_de_25_por_cento_da_razao():
    assert stop_relativo(_serie(100.0, 90.0, 75.0), _serie(100.0, 100.0, 100.0))


def test_stop_relativo_ignora_queda_igual_ao_mercado():
    assert not stop_relativo(_serie(100.0, 80.0, 60.0), _serie(100.0, 80.0, 60.0))


def test_stop_relativo_nao_dispara_com_queda_de_20_por_cento_da_razao():
    assert not stop_relativo(_serie(100.0, 120.0, 96.0), _serie(100.0, 100.0, 100.0))


def test_pode_negociar_falso_no_pregao_seguinte_a_queda_de_5_por_cento():
    assert not pode_negociar(_serie(0.01, -0.05))
    assert pode_negociar(_serie(0.01, -0.049))
    assert pode_negociar(_serie(-0.051, 0.01))       # congela um pregão só
