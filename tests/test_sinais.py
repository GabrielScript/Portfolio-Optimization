"""Sinais do S2 (PLAN §5): só dado público até a data, sobre o painel mensal do lago."""

import numpy as np
import pandas as pd
import pytest

from core.backtest import CarteiraRecusada
from core.sinais import (carteira_da_nota, conservadora, dividend_yield_12m, estrategia_conservadora, momentum_12_1,
                         volatilidade_36m)

MESES = pd.date_range("2020-01-01", periods=40, freq="MS")     # o índice do monthly_total_returns: 1º dia do mês
MES = MESES[-1]                                                  # sinal do mês 2023-04: só meses até 2023-03


def test_momentum_12_1_acumula_do_mes_m_menos_12_ao_m_menos_2():
    # o mês m-1 (o último fechado) fica de fora: só ele muda e o momentum não muda
    retornos = pd.DataFrame({"A": 0.01}, index=MESES)
    outro = retornos.copy()
    outro.loc[MES - pd.DateOffset(months=1), "A"] = 0.50
    esperado = 1.01 ** 11 - 1.0
    assert (momentum_12_1(retornos, MES)["A"], momentum_12_1(outro, MES)["A"]) == (pytest.approx(esperado),
                                                                                  pytest.approx(esperado))


def test_momentum_com_mes_sem_retorno_fica_sem_nota_em_vez_de_render_zero():
    # o lago anula o retorno do mês em que o papel volta depois de > 180 dias sem negociar: não é retorno 0
    retornos = pd.DataFrame({"A": 0.01, "B": 0.01}, index=MESES)
    retornos.loc[MES - pd.DateOffset(months=5), "B"] = np.nan
    momento = momentum_12_1(retornos, MES)
    assert (momento["A"] == pytest.approx(1.01 ** 11 - 1.0), np.isnan(momento["B"])) == (True, True)


def test_vol_36m_e_o_desvio_dos_36_meses_antes_do_mes_e_exige_os_36():
    gerador = np.random.default_rng(3)
    retornos = pd.DataFrame(gerador.normal(0.0, 0.05, (len(MESES), 2)), index=MESES, columns=["A", "B"])
    retornos.loc[MES, "A"] = 9.0                                  # o próprio mês do sinal não entra
    retornos.loc[MES - pd.DateOffset(months=36), "A"] = 9.0       # nem o 37º mês antes
    retornos.loc[MES - pd.DateOffset(months=20), "B"] = np.nan
    vol = volatilidade_36m(retornos, MES)
    esperado = retornos["A"].iloc[-37:-1].std(ddof=1)
    assert (vol["A"] == pytest.approx(esperado), np.isnan(vol["B"])) == (True, True)


def test_dividend_yield_soma_os_eventos_com_data_ex_nos_12_meses_antes_do_mes():
    # cada yield é relativo ao fechamento do próprio dia ex (um desdobramento no meio não mistura escalas)
    proventos = pd.DataFrame({
        "security": ["A", "A", "A", "A", "B"],
        "ex_date": [pd.Timestamp("2022-03-31").date(), pd.Timestamp("2022-04-01").date(),
                    pd.Timestamp("2023-03-31").date(), pd.Timestamp("2023-04-03").date(),
                    pd.Timestamp("2022-10-10").date()],
        "yield": [0.50, 0.02, 0.03, 0.40, 0.01],
    })
    rendimento = dividend_yield_12m(proventos, MES, ["A", "B", "C"])
    assert rendimento.to_dict() == pytest.approx({"A": 0.05, "B": 0.01, "C": 0.0})


def _painel_de_8_papeis() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A-D com vol baixa (±1%) e E-H com vol alta (±8%); momentum crescente de A a D e de E a H (deriva 0 a 3% ao
    mês); dividend yield de A a D = 8%, 2%, 6%, 0% e nenhum provento de E a H."""
    sinal_do_mes = np.where(np.arange(len(MESES)) % 2 == 0, 1.0, -1.0)
    colunas = {}
    for papeis, amplitude in (("ABCD", 0.01), ("EFGH", 0.08)):
        for deriva, papel in zip((0.0, 0.01, 0.02, 0.03), papeis):
            colunas[papel] = deriva + amplitude * sinal_do_mes
    proventos = pd.DataFrame({"security": ["A", "B", "C"], "ex_date": [pd.Timestamp("2022-12-01").date()] * 3,
                              "yield": [0.08, 0.02, 0.06]})
    return pd.DataFrame(colunas, index=MESES), proventos


def test_nota_e_a_metade_de_menor_vol_mais_a_media_dos_postos_de_dividendos_e_momentum():
    # postos percentuais dentro de cada metade; empate (E-H sem provento) fica com o posto médio
    retornos, proventos = _painel_de_8_papeis()
    nota = conservadora(retornos, proventos, MES, list("ABCDEFGH"))
    assert nota.to_dict() == pytest.approx({"A": 1.625, "B": 1.5, "C": 1.75, "D": 1.625,
                                            "E": 0.4375, "F": 0.5625, "G": 0.6875, "H": 0.8125})


def test_sinal_nao_muda_com_dados_depois_da_data():
    # o painel inteiro do lago vai até hoje: o sinal de MES tem de ser o mesmo do painel cortado antes dele
    retornos, proventos = _painel_de_8_papeis()
    gerador = np.random.default_rng(11)
    futuro = pd.DataFrame(gerador.normal(0.0, 0.2, (6, 8)), columns=retornos.columns,
                          index=pd.date_range(MES, periods=6, freq="MS"))
    com_futuro = pd.concat([retornos.drop(MES), futuro])
    proventos_futuros = pd.concat([proventos, pd.DataFrame({"security": ["D", "H"], "yield": [0.30, 0.30],
                                                            "ex_date": [MES.date(), (MES + pd.DateOffset(days=40)).date()]})])
    antes = conservadora(retornos.loc[:MES - pd.DateOffset(months=1)], proventos, MES, list("ABCDEFGH"))
    depois = conservadora(com_futuro, proventos_futuros, MES, list("ABCDEFGH"))
    pd.testing.assert_series_equal(antes, depois)


def test_carteira_pega_as_maiores_notas_um_papel_por_emissor_e_ate_4_por_setor():
    # 20 papéis com nota decrescente: ITUB4 perde para ITUB3 (mesmo emissor), o 5º do setor "Bancos" fica de fora e o
    # papel sem setor não entra; os 15 primeiros que sobram levam 1/15 cada
    outros = [f"O{chr(65 + i)}TR3" for i in range(13)]          # emissores distintos (as 4 primeiras letras)
    papeis = ["ITUB3", "ITUB4", "BBAS3", "BBDC4", "SANB11", "BPAC11", "SEMS3", *outros]
    nota = pd.Series(np.linspace(2.0, 1.0, len(papeis)), index=papeis)
    setores = {papel: "Bancos" for papel in papeis[:6]} | {papel: f"Setor {i % 4}" for i, papel in enumerate(outros)}
    pesos = carteira_da_nota(nota, setores, nomes=15, por_setor=4)
    esperados = ["ITUB3", "BBAS3", "BBDC4", "SANB11", *outros[:11]]
    assert (list(pesos.index), pesos.to_numpy().tolist()) == (esperados, [1.0 / 15] * 15)


SETORES_DOS_8 = {papel: f"Setor {i}" for i, papel in enumerate("ABCDEFGH")}


def _treino(desvio: float) -> pd.DataFrame:
    """Retornos diários do motor até o meio do mês do sinal (a decisão cai num pregão qualquer do mês)."""
    datas = pd.bdate_range(end=MES + pd.Timedelta(days=13), periods=300)
    gerador = np.random.default_rng(5)
    return pd.DataFrame(gerador.normal(0.0, desvio, (len(datas), 8)), index=datas, columns=list("ABCDEFGH"))


def test_estrategia_no_motor_e_a_carteira_da_nota_do_mes_da_decisao():
    # decisão em 13/04: o sinal é o de abril (meses até março), igual ao do IC daquele mês
    retornos, proventos = _painel_de_8_papeis()
    estrategia = estrategia_conservadora(retornos, proventos, SETORES_DOS_8, nomes=5, por_setor=4)
    esperado = carteira_da_nota(conservadora(retornos, proventos, MES, list("ABCDEFGH")), SETORES_DOS_8, 5, 4)
    pd.testing.assert_series_equal(estrategia(_treino(0.01)), esperado)


def test_carteira_com_vol_ex_ante_acima_do_teto_e_recusada_pelas_regras():
    # 8% ao dia em cada papel: 5 papéis com pesos iguais passam dos 40% ao ano da regra fixa; o motor mantém a atual
    retornos, proventos = _painel_de_8_papeis()
    estrategia = estrategia_conservadora(retornos, proventos, SETORES_DOS_8, nomes=5, por_setor=4)
    with pytest.raises(CarteiraRecusada, match="vol"):
        estrategia(_treino(0.08))
