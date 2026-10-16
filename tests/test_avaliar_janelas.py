from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import market_data
from core.backtest import ResultadoBacktest
from core.config import ConfigOtimizacao, Objetivo
from core.estimation import estimar_momentos
from core.optimization import otimizar
from scripts.avaliar_janelas import (Dados, Medidas, _imprimir, avaliar, cdi_ou_vazio, decidir, estrategias,
                                     medir, montar_dados, rodar)

CONTROLE = "R5"
CFG = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.5, n_ativos_max=None, vol_maxima=None)


@pytest.fixture
def treino_de_1260() -> pd.DataFrame:
    gerador = np.random.default_rng(3)
    medias = np.array([0.0002, 0.0005, 0.0008, 0.0003])
    vols = np.r_[np.full((900, 4), 0.01), np.full((360, 4), 0.02)] * np.array([1.0, 1.3, 1.8, 0.9])
    dados = medias + gerador.standard_normal((1260, 4)) * vols
    return pd.DataFrame(dados, index=pd.bdate_range("2014-01-02", periods=1260), columns=list("ABCD"))


@pytest.mark.parametrize("nome, pregoes, meia_vida", [
    ("R5", 1260, None), ("R3", 756, None), ("R2", 504, None), ("EWMA", 1260, 126),
])
def test_cada_candidata_estima_na_janela_pre_registrada(treino_de_1260, nome, pregoes, meia_vida):
    pesos = estrategias(CFG, cdi_diario=None)[nome](treino_de_1260)
    momentos = estimar_momentos(treino_de_1260.iloc[-pregoes:], shrink_mu=True, meia_vida_cov=meia_vida)
    pd.testing.assert_series_equal(pesos, otimizar(momentos.mu, momentos.cov, CFG).pesos)


def test_rodar_poe_candidatas_1_n_e_bova11_nas_mesmas_datas():
    gerador = np.random.default_rng(11)
    colunas = ["A", "B", "C", "D", "BOVA11.SA"]
    retornos = pd.DataFrame(gerador.normal(0.0004, 0.012, (1400, 5)), columns=colunas,
                            index=pd.bdate_range("2014-01-02", periods=1400))
    resultados = rodar(retornos, ["A", "B", "C", "D"], CFG, cdi_diario=None)
    assert set(resultados) == {"R5", "R3", "R2", "EWMA", "1/N", "BOVA11.SA"}
    datas = resultados["R5"].valores.index
    assert datas[0] == retornos.index[1259]           # véspera do 1º pregão fora da amostra (treino de 1260)
    for nome, resultado in resultados.items():
        assert resultado.valores.index.equals(datas), nome


@pytest.fixture
def dados_com_setor_dominante() -> Dados:
    """A, B e C (setor S1) com o maior retorno: sem teto de setor, o Kelly põe 20% em cada (60% no S1)."""
    gerador = np.random.default_rng(13)
    acoes = list("ABCDEFGH")
    medias = np.r_[np.full(3, 0.0015), np.full(5, 0.0002)]
    retornos = pd.DataFrame(medias + gerador.normal(0.0, 0.01, (1400, 8)), columns=acoes,
                            index=pd.bdate_range("2014-01-02", periods=1400))
    retornos["BOVA11.SA"] = gerador.normal(0.0004, 0.012, 1400)
    setores = {"A": "S1", "B": "S1", "C": "S1", "D": "S2", "E": "S2", "F": "S3", "G": "S3", "H": "S4"}
    return Dados(retornos=retornos, acoes=acoes, setores=setores, cdi=pd.Series(dtype=float), selic=0.10)


def test_avaliar_aplica_o_teto_de_30_por_cento_por_setor(dados_com_setor_dominante):
    resultado = avaliar(dados_com_setor_dominante).resultados["R5"]
    assert not resultado.falhas                   # pesos do otimizador, não carteira mantida por falha
    assert (resultado.pesos_alvo[["A", "B", "C"]].sum(axis=1) <= 0.30 + 1e-6).all()


def test_tabela_deflaciona_o_sharpe_so_das_4_candidatas(dados_com_setor_dominante):
    tabela = avaliar(dados_com_setor_dominante).tabela
    candidatas = ["R5", "R3", "R2", "EWMA"]
    assert tabela.loc[["1/N", "BOVA11.SA"], "DSR"].isna().all()
    assert (tabela.loc[candidatas, "DSR"] < tabela.loc[candidatas, "PSR"]).all()


def _resultado(nome: str, retornos: np.ndarray, giros: list[float]) -> ResultadoBacktest:
    datas = pd.bdate_range("2019-01-02", periods=len(retornos) + 1)
    valores = pd.Series(100.0 * np.r_[1.0, np.cumprod(1.0 + retornos)], index=datas)
    return ResultadoBacktest(nome=nome, metodologia="walk-forward OOS", valores=valores,
                             pesos_alvo=pd.DataFrame(), turnover=pd.Series(giros), custos_totais=0.0)


def test_medir_da_p_alto_quando_a_desafiante_rende_mais_todo_dia():
    gerador = np.random.default_rng(5)
    r_controle = gerador.normal(0.0004, 0.015, 1000)
    r_desafiante = r_controle + 0.0005 + gerador.normal(0.0, 0.001, 1000)   # +0,05 p.p. ao dia, quase sem ruído
    resultados = {CONTROLE: _resultado(CONTROLE, r_controle, [1.0]),
                  "R3": _resultado("R3", r_desafiante, [1.0])}
    rf = pd.Series(0.0004, index=resultados[CONTROLE].valores.index)
    assert medir(resultados, rf, controle=CONTROLE)["R3"].p_supera_controle > 0.99


def test_medir_usa_o_drawdown_e_o_giro_da_propria_estrategia():
    resultados = {CONTROLE: _resultado(CONTROLE, np.array([0.01, 0.01, -0.01]), [1.0]),
                  "R3": _resultado("R3", np.array([0.10, -0.20, 0.30]), [1.0, 0.5])}
    rf = pd.Series(0.0, index=resultados[CONTROLE].valores.index)
    medidas = medir(resultados, rf, controle=CONTROLE)["R3"]
    assert medidas.max_drawdown == pytest.approx(-0.20)      # 100 -> 110 -> 88: 88/110 - 1
    assert medidas.giro_anual == pytest.approx(126.0)        # (1,0 + 0,5) / (3 pregões / 252)


def _medidas(sharpe=0.40, max_drawdown=-0.20, giro_anual=1.50, p_supera_controle=float("nan")) -> Medidas:
    return Medidas(sharpe=sharpe, max_drawdown=max_drawdown, giro_anual=giro_anual,
                   p_supera_controle=p_supera_controle)


def test_decidir_adota_desafiante_que_passa_nos_tres_criterios():
    # 0,985 >= 1 - 0,05/3 = 0,98333; MDD 2 p.p. pior (limite 5); giro 1,2x (limite 2x)
    medidas = {CONTROLE: _medidas(),
               "R3": _medidas(sharpe=0.50, max_drawdown=-0.22, giro_anual=1.80, p_supera_controle=0.985)}
    assert decidir(medidas, controle=CONTROLE).escolhida == "R3"


def test_decidir_mantem_o_controle_com_p_abaixo_do_limiar_de_bonferroni():
    # 0,98 passaria num limiar de 0,95 sem correção; com 3 desafiantes o limiar é 0,98333
    medidas = {CONTROLE: _medidas(),
               "R3": _medidas(sharpe=0.50, max_drawdown=-0.20, giro_anual=1.50, p_supera_controle=0.98)}
    assert decidir(medidas, controle=CONTROLE).escolhida == CONTROLE


def test_decidir_reprova_desafiante_com_drawdown_mais_de_5_pp_pior():
    medidas = {CONTROLE: _medidas(max_drawdown=-0.20),
               "R2": _medidas(sharpe=0.60, max_drawdown=-0.26, giro_anual=1.50, p_supera_controle=0.99)}
    assert decidir(medidas, controle=CONTROLE).escolhida == CONTROLE


def test_decidir_reprova_desafiante_com_giro_acima_de_2x_o_do_controle():
    medidas = {CONTROLE: _medidas(giro_anual=1.50),
               "R2": _medidas(sharpe=0.60, max_drawdown=-0.20, giro_anual=3.10, p_supera_controle=0.99)}
    assert decidir(medidas, controle=CONTROLE).escolhida == CONTROLE


def test_decidir_entre_duas_aprovadas_escolhe_a_de_maior_sharpe():
    medidas = {CONTROLE: _medidas(),
               "R3": _medidas(sharpe=0.50, p_supera_controle=0.99),
               "EWMA": _medidas(sharpe=0.55, p_supera_controle=0.99)}
    assert decidir(medidas, controle=CONTROLE).escolhida == "EWMA"


def _precos(colunas: list[str], dias: int = 6) -> pd.DataFrame:
    gerador = np.random.default_rng(17)
    valores = 100.0 * np.cumprod(1.0 + gerador.normal(0.0, 0.01, (dias, len(colunas))), axis=0)
    return pd.DataFrame(valores, columns=colunas, index=pd.bdate_range("2026-10-01", periods=dias))


def test_montar_dados_recusa_precos_sem_o_bova11():
    with pytest.raises(market_data.ErroDadosMercado, match="BOVA11"):
        montar_dados(_precos(["PETR4.SA", "VALE3.SA"]), cdi=pd.Series(dtype=float), selic=0.15)


def test_montar_dados_separa_as_acoes_do_bova11_na_amostra_comum():
    precos = _precos(["PETR4.SA", "BOVA11.SA", "VALE3.SA"])
    precos.iloc[:2, 2] = np.nan                       # VALE3 começa no 3º pregão
    dados = montar_dados(precos, cdi=pd.Series(dtype=float), selic=0.15)
    assert dados.acoes == ["PETR4.SA", "VALE3.SA"]
    assert dados.setores == {"PETR4.SA": "Petróleo, Gás e Biocombustíveis", "VALE3.SA": "Materiais Básicos"}
    assert list(dados.retornos.columns) == ["PETR4.SA", "VALE3.SA", "BOVA11.SA"]
    assert dados.retornos.index[0] == precos.index[3]   # 1º dia com os 3 preços hoje e na véspera
    assert dados.retornos.notna().all().all()


def test_cdi_fora_do_ar_vira_serie_vazia(monkeypatch):
    def bcb_fora_do_ar(anos, hoje=None):
        raise market_data.ErroDadosMercado("SGS 12: 503")
    monkeypatch.setattr(market_data, "baixar_cdi_diario", bcb_fora_do_ar)
    assert cdi_ou_vazio(date(2026, 10, 15)).empty


def test_imprimir_mostra_os_3_criterios_de_cada_desafiante(dados_com_setor_dominante, capsys):
    _imprimir(avaliar(dados_com_setor_dominante))
    linhas = capsys.readouterr().out.splitlines()
    for nome in ("R3", "R2", "EWMA"):
        assert sum(linha.strip().startswith(f"{nome}: ") for linha in linhas) == 3, nome
