"""Sinais de valor do S2 (PLAN §5, estratégias 2 e 3): fundamentos da CVM só depois da data de entrega."""

import numpy as np
import pandas as pd
import pytest

from core.backtest import CarteiraRecusada

from core.valor import (Balancos, composto_de_valor, formula_magica, formula_magica_no_mes, fundamentos_no_mes,
                        estrategia_da_nota, momentum_6m, painel_da_nota, trending_value, valor_da_firma)


def _fundamentos(*linhas):
    return pd.DataFrame(linhas, columns=["issuer", "account", "dt_refer", "available_from", "value"]).astype(
        {"dt_refer": "datetime64[ns]", "available_from": "datetime64[ns]"})


def test_fundamento_so_entra_depois_da_data_de_entrega():
    # o ITR do 1º trimestre chega em 10/05: na decisão de maio vale ainda o ano de 2024; na de junho, o trimestre
    fundamentos = _fundamentos(("ACME", "ebit", "2024-12-31", "2025-03-20", 10.0),
                               ("ACME", "ebit", "2025-03-31", "2025-05-10", 12.0),
                               ("VELH", "ebit", "2023-09-30", "2023-11-10", 5.0))      # parou de entregar
    maio, junho = (fundamentos_no_mes(fundamentos, pd.Timestamp(mes)) for mes in ("2025-05-01", "2025-06-01"))
    assert (maio.loc["ACME", "ebit"], junho.loc["ACME", "ebit"], "VELH" in junho.index) == (10.0, 12.0, False)


def _balanco(**contas_por_emissor):
    """Emissor x conta com zero nas contas do capital empregado e do EV que o exemplo não dá."""
    contas = ["ebit", "current_assets", "cash", "short_term_investments", "fixed_assets", "current_liabilities",
              "short_term_debt", "long_term_debt"]
    balanco = pd.DataFrame(contas_por_emissor).T
    return balanco.reindex(columns=list(dict.fromkeys(contas + balanco.columns.tolist()))).fillna(0.0)


def test_formula_magica_ordena_pela_soma_dos_ranks():
    # valor de mercado 100 e sem dívida nem caixa: EBIT/EV = EBIT/100 e o capital empregado = ativo circulante
    #       EBIT/EV   ROC (EBIT/capital)   postos      nota
    #   A    0,20    20/250 = 0,08        1,00 0,25   0,625
    #   B    0,06     6/15  = 0,40        0,25 1,00   0,625
    #   C    0,15    15/50  = 0,30        0,75 0,75   0,750
    #   D    0,10    10/50  = 0,20        0,50 0,50   0,500
    fundamentos = _balanco(A={"ebit": 20.0, "current_assets": 250.0}, B={"ebit": 6.0, "current_assets": 15.0},
                           C={"ebit": 15.0, "current_assets": 50.0}, D={"ebit": 10.0, "current_assets": 50.0})
    valor = pd.Series(100.0, index=list("ABCD"))
    nota = formula_magica(fundamentos, valor, {emissor: "Consumo Cíclico" for emissor in "ABCD"})
    assert nota.to_dict() == pytest.approx({"A": 0.625, "B": 0.625, "C": 0.75, "D": 0.5})


def _longo(balanco: pd.DataFrame, dt_refer: str = "2024-12-31", available_from: str = "2025-03-20") -> pd.DataFrame:
    """O balanço emissor x conta no formato do lago, todo de um mesmo documento."""
    linhas = balanco.rename_axis("issuer").reset_index().melt(id_vars="issuer", var_name="account", value_name="value")
    return linhas.assign(dt_refer=pd.Timestamp(dt_refer), available_from=pd.Timestamp(available_from))


def _retornos_mensais(mes: str, por_papel: dict[str, float], meses: int = 12) -> pd.DataFrame:
    """Retorno mensal constante por papel nos ``meses`` meses antes de ``mes``."""
    indice = pd.date_range(end=pd.Timestamp(mes) - pd.DateOffset(months=1), periods=meses, freq="MS")
    return pd.DataFrame({papel: [retorno] * meses for papel, retorno in por_papel.items()}, index=indice)


SEM_PROVENTOS = pd.DataFrame({"security": pd.Series(dtype=str), "ex_date": pd.Series(dtype="datetime64[ns]"),
                              "yield": pd.Series(dtype=float)})


def test_trending_value_escolhe_por_momentum_dentro_das_baratas():
    # 10 emissores iguais menos o lucro (A = 10, ..., J = 1), na ordem J..A: os 30% mais baratos são A, B e C; dentro
    # deles manda o momentum de 6 meses (C > A > B), e J, o mais caro, tem o maior momentum de todos e fica atrás
    emissores = [letra * 4 for letra in "JIHGFEDCBA"]
    balanco = _balanco(**{e: {"net_income": 1.0 + i, "equity": 50.0, "revenue": 80.0, "ebit": 12.0,
                              "operating_cash_flow": 9.0, "current_assets": 40.0, "current_liabilities": 20.0}
                          for i, e in enumerate(emissores)})
    valor_mercado = pd.DataFrame({"issuer": emissores, "month": pd.Timestamp("2025-06-01"), "market_cap": 100.0})
    balancos = Balancos(_longo(balanco), valor_mercado, dict.fromkeys(emissores, "Bens Industriais"))
    papeis = [e + "3" for e in emissores]
    momento = dict(zip(papeis, [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.01, 0.02]))
    nota = trending_value(balancos, _retornos_mensais("2025-06-01", momento), SEM_PROVENTOS,
                          pd.Timestamp("2025-06-01"), papeis)
    assert nota.nlargest(3).index.tolist() == ["CCCC3", "AAAA3", "BBBB3"]
    assert nota["JJJJ3"] < nota["BBBB3"]


def test_formula_magica_e_trending_value_excluem_financeiras():
    # bancos e seguradoras têm outro plano de contas (sem EBIT nem capital de giro): o banco e o emissor sem setor, com
    # os melhores números, ficam fora das duas
    contas = {"net_income": 30.0, "equity": 50.0, "revenue": 80.0, "ebit": 40.0, "operating_cash_flow": 9.0,
              "current_assets": 40.0, "current_liabilities": 20.0}
    balanco = _balanco(AAAA={**contas, "ebit": 10.0, "net_income": 5.0}, BANC=contas, SEMS=contas)
    setores = {"AAAA": "Bens Industriais", "BANC": "Financeiro"}
    valor = pd.Series(100.0, index=balanco.index)
    balancos = Balancos(_longo(balanco), pd.DataFrame({"issuer": balanco.index, "month": pd.Timestamp("2025-06-01"),
                                                       "market_cap": 100.0}), setores)
    papeis = ["AAAA3", "BANC4", "SEMS3"]
    nota = trending_value(balancos, _retornos_mensais("2025-06-01", dict.fromkeys(papeis, 0.01)), SEM_PROVENTOS,
                          pd.Timestamp("2025-06-01"), papeis)
    assert (formula_magica(balanco, valor, setores).index.tolist(), nota.index.tolist()) == (["AAAA"], ["AAAA3"])


def _painel_com_futuro() -> tuple[Balancos, pd.DataFrame, pd.DataFrame, list[str]]:
    """Seis emissores com balanço, valor de mercado, retornos e proventos antes e depois de 06/2025; os de depois
    invertem a ordem de tudo (quem olhar o futuro muda de nota)."""
    gerador = np.random.default_rng(7)
    emissores = [letra * 4 for letra in "ABCDEF"]
    papeis = [e + "3" for e in emissores]
    antes = _balanco(**{e: {"ebit": 5.0 + i, "net_income": 3.0 + i, "equity": 40.0, "revenue": 90.0 - i,
                            "operating_cash_flow": 4.0, "current_assets": 30.0 + 7 * i, "current_liabilities": 10.0}
                        for i, e in enumerate(emissores)})
    depois = antes.iloc[::-1].set_axis(antes.index)
    fundamentos = pd.concat([_longo(antes), _longo(depois, "2025-03-31", "2025-06-10")], ignore_index=True)
    valor_mercado = pd.DataFrame({"issuer": emissores * 2, "month": [pd.Timestamp("2025-06-01")] * 6
                                  + [pd.Timestamp("2025-07-01")] * 6, "market_cap": [100.0 + 10 * i for i in range(6)]
                                  + [160.0 - 10 * i for i in range(6)]})
    indice = pd.date_range("2024-01-01", "2025-09-01", freq="MS")
    retornos = pd.DataFrame(gerador.normal(0.01, 0.05, (len(indice), 6)), index=indice, columns=papeis)
    proventos = pd.DataFrame({"security": papeis * 2,
                              "ex_date": [pd.Timestamp("2025-02-10")] * 6 + [pd.Timestamp("2025-06-15")] * 6,
                              "yield": [*np.linspace(0.01, 0.06, 6), *np.linspace(0.12, 0.02, 6)]})
    balancos = Balancos(fundamentos, valor_mercado, dict.fromkeys(emissores, "Bens Industriais"))
    return balancos, retornos, proventos, papeis


def test_sinal_nao_muda_com_dados_depois_da_data():
    # painel cortado no 1º dia de 06/2025 (o que se sabia na decisão) = painel inteiro, nas duas notas
    balancos, retornos, proventos, papeis = _painel_com_futuro()
    mes = pd.Timestamp("2025-06-01")
    cortado = Balancos(balancos.fundamentos[balancos.fundamentos["available_from"] < mes],
                       balancos.valor_mercado[balancos.valor_mercado["month"] <= mes], balancos.setores)
    retornos_cortados, proventos_cortados = retornos[retornos.index < mes], proventos[proventos["ex_date"] < mes]
    for inteiro, so_o_passado in ((formula_magica_no_mes(balancos, mes, papeis),
                                   formula_magica_no_mes(cortado, mes, papeis)),
                                  (composto_de_valor(balancos, proventos, mes, papeis),
                                   composto_de_valor(cortado, proventos_cortados, mes, papeis)),
                                  (trending_value(balancos, retornos, proventos, mes, papeis),
                                   trending_value(cortado, retornos_cortados, proventos_cortados, mes, papeis))):
        assert inteiro.notna().sum() == 6
        pd.testing.assert_series_equal(inteiro, so_o_passado)


PAPEIS_DO_MOTOR = [letra * 4 + "3" for letra in "ABCDEFGH"]
SETORES_DO_MOTOR = {papel: f"Setor {i % 6}" for i, papel in enumerate(PAPEIS_DO_MOTOR)} | {"GGGG3": "Setor 1"}


def _treino_ate_14_de_maio(desvio: float) -> pd.DataFrame:
    """Retornos diários do motor até 14/05/2025 (a decisão cai num pregão qualquer do mês)."""
    datas = pd.bdate_range(end="2025-05-14", periods=300)
    gerador = np.random.default_rng(5)
    return pd.DataFrame(gerador.normal(0.0, desvio, (len(datas), 8)), index=datas, columns=PAPEIS_DO_MOTOR)


def test_estrategia_de_valor_no_motor_e_a_carteira_da_nota_do_mes_da_decisao():
    # decisão em 14/05: a nota é a de maio (o que se sabia em 1º/05), nos papéis do treino; nota crescente de A a H,
    # 5 nomes e 1 por setor (G é do setor de H): H, F, E, D e C com 20% cada
    chamadas = []

    def nota_do_mes(mes: pd.Timestamp, candidatos) -> pd.Series:
        chamadas.append((mes, list(candidatos)))
        return pd.Series(np.linspace(1.0, 2.0, len(chamadas[-1][1])), index=chamadas[-1][1])

    pesos = estrategia_da_nota(nota_do_mes, SETORES_DO_MOTOR, nomes=5, por_setor=1)(_treino_ate_14_de_maio(0.01))
    assert chamadas == [(pd.Timestamp("2025-05-01"), PAPEIS_DO_MOTOR)]
    assert (list(pesos.index), pesos.tolist()) == (["HHHH3", "FFFF3", "EEEE3", "DDDD3", "CCCC3"], [0.2] * 5)


def test_carteira_de_valor_com_vol_ex_ante_acima_do_teto_e_recusada_pelas_regras():
    # 8% ao dia em cada papel: 5 papéis com pesos iguais passam dos 40% ao ano da regra fixa; o motor mantém a atual
    def nota_do_mes(mes: pd.Timestamp, candidatos) -> pd.Series:
        return pd.Series(np.linspace(1.0, 2.0, len(candidatos)), index=list(candidatos))

    with pytest.raises(CarteiraRecusada, match="vol"):
        estrategia_da_nota(nota_do_mes, SETORES_DO_MOTOR, nomes=5, por_setor=1)(_treino_ate_14_de_maio(0.08))


def test_painel_da_nota_tem_a_nota_de_cada_mes_nos_papeis_do_universo_daquele_mes():
    # o universo ponto no tempo do lago: em cada mês só os papéis elegíveis nele entram na nota
    meses = pd.DatetimeIndex(["2025-05-01", "2025-06-01"])
    universo = pd.DataFrame({"month": [meses[0].date()] * 3 + [meses[1].date()] * 2,
                             "security": ["AAAA3", "BBBB3", "CCCC3", "AAAA3", "CCCC3"]})

    def nota_do_mes(mes: pd.Timestamp, candidatos) -> pd.Series:
        return pd.Series(float(mes.month), index=list(candidatos))

    painel = painel_da_nota(nota_do_mes, universo, meses, ["AAAA3", "BBBB3", "CCCC3", "DDDD3"])
    esperado = pd.DataFrame({"AAAA3": [5.0, 6.0], "BBBB3": [5.0, np.nan], "CCCC3": [5.0, 6.0], "DDDD3": np.nan},
                            index=meses)
    pd.testing.assert_frame_equal(painel, esperado, check_freq=False)


def test_momentum_6m_acumula_do_mes_m_menos_6_ao_m_menos_1():
    # O'Shaughnessy não pula o último mês: m-1 entra; m-7 e o próprio m (ainda aberto na decisão) ficam fora
    indice = pd.date_range("2024-11-01", "2025-06-01", freq="MS")         # m-7 ... m
    retornos = pd.DataFrame({"AAAA3": [0.50, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, -0.90]}, index=indice)
    esperado = np.prod([1.01, 1.02, 1.03, 1.04, 1.05, 1.06]) - 1.0
    assert momentum_6m(retornos, pd.Timestamp("2025-06-01"))["AAAA3"] == pytest.approx(esperado)


def test_momentum_6m_com_mes_sem_retorno_fica_sem_nota_em_vez_de_render_zero():
    # o lago deixa vazio o mês sem negócio (ou com retorno anulado): sem os 6 meses, sem momentum
    indice = pd.date_range("2024-12-01", "2025-05-01", freq="MS")
    retornos = pd.DataFrame({"AAAA3": [0.01, 0.02, np.nan, 0.03, 0.04, 0.05], "BBBB3": 0.01}, index=indice)
    momento = momentum_6m(retornos, pd.Timestamp("2025-06-01"))
    assert (np.isnan(momento["AAAA3"]), momento["BBBB3"]) == (True, pytest.approx(1.01**6 - 1.0))


def test_valor_da_firma_soma_divida_e_minoritarios_e_tira_o_caixa():
    # o EBIT consolidado inclui a parte dos minoritários: o EV também (valor contábil deles), como a dívida líquida
    balanco = _balanco(AAAA={"short_term_debt": 30.0, "long_term_debt": 50.0, "cash": 15.0,
                             "short_term_investments": 5.0, "minority_interest": 12.0})
    assert valor_da_firma(balanco, pd.Series({"AAAA": 100.0}))["AAAA"] == pytest.approx(100.0 + 80.0 - 20.0 + 12.0)


def test_composto_de_valor_deixa_fora_quem_nao_tem_lucro_no_balanco():
    # só balanço patrimonial, sem demonstração de resultado: com o neutro nos 4 que faltam, ficaria no meio da fila
    contas = {"net_income": 5.0, "equity": 50.0, "revenue": 80.0, "ebit": 8.0, "operating_cash_flow": 6.0}
    balanco = pd.DataFrame({"AAAA": contas, "BBBB": {**contas, "net_income": 2.0},
                            "SOBA": {"equity": 500.0}}).T.dropna(axis=1, how="all")
    valor_mercado = pd.DataFrame({"issuer": balanco.index, "month": pd.Timestamp("2025-06-01"), "market_cap": 100.0})
    balancos = Balancos(_longo(balanco).dropna(subset=["value"]), valor_mercado,
                        dict.fromkeys(balanco.index, "Bens Industriais"))
    composto = composto_de_valor(balancos, SEM_PROVENTOS, pd.Timestamp("2025-06-01"), ["AAAA3", "BBBB3", "SOBA3"])
    assert composto.index.tolist() == ["AAAA3", "BBBB3"]
