"""
Testes de core/fatores_macro.py (sem rede). Os dados são sintéticos, com
resposta calculável à mão: o painel vai de 2005-12 a 2006-03 (hoje = 2006-04-15).
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import fatores_macro, tesouro
from core.fatores_macro import DadosMacro

HOJE = date(2006, 4, 15)


def _mes(texto: str) -> pd.Period:
    return pd.Period(texto, freq="M")


def _titulos(curvas: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Dois títulos por dia, com prazo exato de 1 e 5 anos: a curva interpolada devolve as taxas dadas."""
    linhas = []
    for dia, (taxa_1a, taxa_5a) in curvas.items():
        base = pd.Timestamp(dia)
        linhas += [{"data_base": base, "vencimento": base + pd.Timedelta(days=365), "tipo": tesouro.TIPO_LTN,
                    "taxa": taxa_1a, "prazo_anos": 1.0},
                   {"data_base": base, "vencimento": base + pd.Timedelta(days=1826), "tipo": tesouro.TIPO_NTNF,
                    "taxa": taxa_5a, "prazo_anos": 5.0}]
    return pd.DataFrame(linhas)


def _focus_anual(linhas: list[tuple[str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame({"data": pd.to_datetime([l[0] for l in linhas]),
                         "ano_referencia": [l[1] for l in linhas],
                         "mediana": [l[2] for l in linhas], "desvio": 0.5})


@pytest.fixture
def dados() -> DadosMacro:
    titulos = _titulos({
        "2005-11-16": (99.0, 99.0),                   # não é o último dia do mês: ignorado
        "2005-11-30": (18.0, 17.0), "2005-12-30": (17.5, 16.0),
        "2006-01-31": (17.0, 16.5),                   # fevereiro sem cotação
        "2006-03-31": (16.0, 15.0), "2006-04-10": (1.0, 1.0),   # abril: mês incompleto
    })
    ipca = pd.DataFrame({"data": pd.to_datetime(["2005-11-28", "2005-12-26", "2006-01-30",
                                                 "2006-02-20", "2006-02-24", "2006-03-27"]),
                         "mediana": [5.0, 4.8, 4.6, 9.9, 4.5, 4.7], "desvio": 0.4})
    pib = _focus_anual([("2005-11-28", 2005, 2.0), ("2005-11-28", 2006, 3.0),
                        ("2005-12-26", 2005, 2.2), ("2005-12-26", 2006, 3.6),
                        ("2006-01-30", 2006, 3.4), ("2006-01-30", 2007, 3.5),
                        ("2006-02-24", 2006, 3.0), ("2006-02-24", 2007, 3.5),
                        ("2006-03-27", 2006, 3.3), ("2006-03-27", 2007, 3.1)])
    ptax = pd.Series([2.30, 2.20, 2.35, 2.10, 2.15, 2.12],
                     index=pd.to_datetime(["2005-11-30", "2005-12-01", "2005-12-30",
                                           "2006-01-31", "2006-02-24", "2006-03-31"]))
    bvsp = pd.Series([30000.0, 33000.0, 36300.0, 36300.0, 34485.0],
                     index=pd.PeriodIndex(["2005-11", "2005-12", "2006-01", "2006-02", "2006-03"], freq="M"))
    dias = pd.bdate_range("2005-11-01", "2006-04-14")
    cdi = pd.Series(0.0005, index=dias)
    return DadosMacro(titulos=titulos, focus_ipca_12m=ipca, focus_pib=pib, ptax=ptax,
                      bvsp_mensal=bvsp, cdi_diario=cdi)


def test_painel_vai_de_2005_12_ao_ultimo_mes_completo(dados):
    painel = fatores_macro.montar_painel(dados, HOJE)
    assert list(painel.fatores.index) == [_mes("2005-12"), _mes("2006-01"), _mes("2006-02"), _mes("2006-03")]
    assert list(painel.fatores.columns) == list(fatores_macro.FATORES)
    assert painel.fatores_indisponiveis == {} and painel.avisos == ()


def test_juros_e_premio_sao_diferencas_de_fim_de_mes_e_mes_sem_cotacao_vira_nan(dados):
    f = fatores_macro.montar_painel(dados, HOJE).fatores
    assert f.at[_mes("2005-12"), "juros"] == pytest.approx(17.5 - 18.0)
    assert f.at[_mes("2006-01"), "juros"] == pytest.approx(17.0 - 17.5)
    assert f.at[_mes("2005-12"), "premio"] == pytest.approx((16.0 - 17.5) - (17.0 - 18.0))
    assert np.isnan(f.at[_mes("2006-02"), "juros"])        # fevereiro sem cotação
    assert np.isnan(f.at[_mes("2006-03"), "juros"])        # não vira diferença de dois meses


def test_inflacao_usa_a_ultima_pesquisa_do_mes(dados):
    f = fatores_macro.montar_painel(dados, HOJE).fatores
    assert f.at[_mes("2006-02"), "inflacao"] == pytest.approx(4.5 - 4.6)   # 24/02, não 20/02
    assert f.at[_mes("2006-03"), "inflacao"] == pytest.approx(4.7 - 4.5)


def test_cambio_e_variacao_do_log_da_ptax_de_fim_de_mes(dados):
    f = fatores_macro.montar_painel(dados, HOJE).fatores
    assert f.at[_mes("2005-12"), "cambio"] == pytest.approx(np.log(2.35 / 2.30))
    assert f.at[_mes("2006-02"), "cambio"] == pytest.approx(np.log(2.15 / 2.10))


def test_atividade_e_revisao_com_ano_de_referencia_fixo(dados):
    f = fatores_macro.montar_painel(dados, HOJE).fatores
    # dezembro/2005: w = 1/12 no ano 2005 e 11/12 em 2006
    assert f.at[_mes("2005-12"), "atividade"] == pytest.approx((1 / 12) * (2.2 - 2.0) + (11 / 12) * (3.6 - 3.0))
    # janeiro/2006: w = 1, só o ano 2006, comparado com a pesquisa de dezembro para 2006
    assert f.at[_mes("2006-01"), "atividade"] == pytest.approx(3.4 - 3.6)
    # março/2006: w = 10/12
    assert f.at[_mes("2006-03"), "atividade"] == pytest.approx((10 / 12) * (3.3 - 3.0) + (2 / 12) * (3.1 - 3.5))


def test_mercado_e_retorno_do_ibovespa_menos_cdi_composto(dados):
    painel = fatores_macro.montar_painel(dados, HOJE)
    dias_dez = len(pd.bdate_range("2005-12-01", "2005-12-31"))
    cdi_dez = 1.0005 ** dias_dez - 1.0
    assert painel.cdi_mensal[_mes("2005-12")] == pytest.approx(cdi_dez)
    assert painel.fatores.at[_mes("2005-12"), "mercado"] == pytest.approx(0.10 - cdi_dez)
    assert painel.cdi_mensal.index.max() == _mes("2006-03")   # abril, incompleto, fica de fora


def test_fonte_que_falha_tira_so_os_fatores_dela(dados):
    sem_tesouro = DadosMacro(**{**dados.__dict__, "titulos": None, "erros": {"tesouro": "timeout"}})
    painel = fatores_macro.montar_painel(sem_tesouro, HOJE)
    assert "juros" not in painel.fatores.columns and "premio" not in painel.fatores.columns
    assert painel.fatores_indisponiveis == {"juros": "tesouro: timeout", "premio": "tesouro: timeout"}
    assert len(painel.avisos) == 2
    assert painel.fatores["inflacao"].notna().any()


def test_sem_cdi_nao_ha_fator_mercado(dados):
    sem_cdi = DadosMacro(**{**dados.__dict__, "cdi_diario": None, "erros": {"cdi": "HTTP 503"}})
    painel = fatores_macro.montar_painel(sem_cdi, HOJE)
    assert painel.fatores_indisponiveis == {"mercado": "cdi: HTTP 503"}
    assert painel.cdi_mensal.empty


def test_revisao_pib_nan_quando_falta_o_ano_seguinte():
    medianas = pd.DataFrame({2026: [3.0, 3.1]}, index=pd.PeriodIndex(["2026-09", "2026-10"], freq="M"))
    assert np.isnan(fatores_macro.revisao_pib_12m(medianas)[_mes("2026-10")])
