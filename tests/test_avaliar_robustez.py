"""Teste de robustez do passo 3 (PLAN §11): partes sem rede."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from core.backtest import ResultadoBacktest
from core.config import ConfigBacktest
from core.metrics import (drawdown, momentos_sharpe, retornos_de_valores, sharpe_probabilistico,
                          sharpe_referencia_deflacionado)
from core.robustez import anos_minimos_para_psr, psr_da_diferenca
from scripts.avaliar_robustez import (ARQUIVO_TENTATIVAS, comparar_r10, conferir_hashes, contar_tentativas,
                                      dinheiro_real, r10_passa, recortar, referencias, tabela_do_periodo, trecho)


def test_hash_que_nao_bate_com_o_sha256sums_para_o_script(tmp_path):
    # o pré-registro fixa os arquivos: a rotina do lago pode sobrescrever data/b3lake entre o commit e a rodada
    (tmp_path / "a.parquet").write_bytes(b"conteudo")
    (tmp_path / "SHA256SUMS").write_text(f"{hashlib.sha256(b'outro').hexdigest()} *a.parquet\n", encoding="ascii")
    with pytest.raises(SystemExit, match="a.parquet"):
        conferir_hashes(tmp_path, ["a.parquet"])
    (tmp_path / "SHA256SUMS").write_text(f"{hashlib.sha256(b'conteudo').hexdigest()} *a.parquet\n", encoding="ascii")
    assert conferir_hashes(tmp_path, ["a.parquet"]) == {"a.parquet": hashlib.sha256(b"conteudo").hexdigest()}


def test_tentativas_contam_as_linhas_sem_os_comentarios(tmp_path):
    arquivo = tmp_path / "tentativas.csv"
    arquivo.write_text('# regra\n# mais regra\ndata,script,configuracao\n2026-10-13,a.py,"x, y"\n2026-10-15,b.py,z\n',
                       encoding="utf-8")
    assert contar_tentativas(arquivo) == 2


def test_tentativas_ate_o_pre_registro_da_robustez_sao_21():
    # o arquivo só ganha linhas: as de até 19/10/2026 são as 21 que a rodada do passo 3 contou no N
    tentativas = pd.read_csv(ARQUIVO_TENTATIVAS, comment="#")
    assert int((tentativas["data"] <= "2026-10-19").sum()) == 21


DATAS = pd.bdate_range("2024-01-01", periods=10)


def _curva(valores: list[float], rebalanceios: list[int] = ()) -> ResultadoBacktest:
    """Curva diária com os rebalanceios nas posições dadas (1º pregão com os pesos novos)."""
    alvo = pd.DataFrame({"A": 1.0}, index=DATAS[list(rebalanceios)])
    return ResultadoBacktest(nome="curva", metodologia="walk-forward OOS", valores=pd.Series(valores, index=DATAS),
                             pesos_alvo=alvo, turnover=pd.Series(0.5, index=alvo.index), custos_totais=0.0)


def test_recorte_comeca_no_fechamento_do_inicio_e_deixa_fora_o_rebalanceio_desse_dia():
    # o rebalanceio da posição 2 vale a partir do pregão 2: quem começa no fechamento dele já está com os pesos novos
    curva = recortar(_curva(list(range(100, 110)), rebalanceios=[2, 5, 8]), DATAS[2], DATAS[7])
    assert curva.valores.index[0] == DATAS[2] and curva.valores.index[-1] == DATAS[7]
    assert curva.pesos_alvo.index.tolist() == [DATAS[5]] and curva.turnover.index.tolist() == [DATAS[5]]


SEM_CUSTO = ConfigBacktest(janela_treino=60, janela_teste=63, capital_inicial=100.0, custo_bps=0.0)


def _satelite_que_caiu_no_2o_pregao() -> tuple[ResultadoBacktest, ResultadoBacktest]:
    """BOVA11 sobe 1% ao dia; o satélite cai 30% no pregão 1 e fica parado (rebalanceios nos pregões 3 e 6)."""
    bova = _curva([100.0 * 1.01**i for i in range(len(DATAS))])
    return _curva([100.0] + [70.0] * (len(DATAS) - 1), rebalanceios=[3, 6]), bova


def test_stop_do_dinheiro_real_e_armado_de_novo_no_comeco_da_janela():
    # quem começou no pregão 0 viu a razão satélite/BOVA11 cair 31%: o stop dispara no rebalanceio do pregão 3 e daí
    # em diante tudo anda com o BOVA11. Quem começou no pregão 1 já começou depois da queda: o pico é ali, sem stop.
    satelite, bova = _satelite_que_caiu_no_2o_pregao()
    com_stop = dinheiro_real(satelite, bova, DATAS[0], DATAS[-1], SEM_CUSTO)
    sem_stop = dinheiro_real(satelite, bova, DATAS[1], DATAS[-1], SEM_CUSTO)
    assert com_stop.pct_change().iloc[-1] == pytest.approx(0.01)
    assert sem_stop.pct_change().iloc[-1] < 0.009


def test_referencia_que_ainda_nao_existia_fica_vazia_antes_da_1a_cotacao():
    # IVVB11 só desde 29/04/2014: antes disso a curva não pode valer como "parada em 0%"
    datas = pd.bdate_range("2014-01-01", periods=80)
    precos = pd.DataFrame({"BOVA11": np.linspace(50.0, 60.0, 80),
                           "IVVB11": [np.nan] * 70 + list(np.linspace(80.0, 81.0, 10))}, index=datas)
    curvas = referencias(precos, SEM_CUSTO)
    assert curvas["BOVA11"].valores.notna().all()
    ivvb = curvas["IVVB11"].valores
    assert ivvb[ivvb.index < datas[70]].isna().all() and ivvb[ivvb.index >= datas[70]].notna().all()


def _aleatoria(semente: int, media: float, datas: pd.DatetimeIndex) -> pd.Series:
    retornos = np.random.default_rng(semente).normal(media, 0.01, len(datas) - 1)
    return pd.Series(100 * np.concatenate([[1.0], np.cumprod(1 + retornos)]), index=datas)


def test_dsr_contra_o_bova11_deflaciona_o_ir_pelas_tentativas_do_csv_e_a_variancia_das_curvas():
    datas = pd.bdate_range("2020-01-01", periods=500)
    bova = _aleatoria(0, 0.0003, datas)
    curvas = {"A": _aleatoria(1, 0.0008, datas), "B": _aleatoria(2, 0.0004, datas), "C": _aleatoria(3, 0.0, datas)}
    tabela = tabela_do_periodo(curvas, bova, pd.Series(0.0, index=datas), n_tentativas=21)

    diferencas = {nome: retornos_de_valores(c) - retornos_de_valores(bova) for nome, c in curvas.items()}
    momentos = {nome: momentos_sharpe(d) for nome, d in diferencas.items()}
    sr0 = sharpe_referencia_deflacionado(float(np.var([m.sharpe for m in momentos.values()], ddof=1)), 21)
    assert tabela.at["A", "DSR"] == pytest.approx(sharpe_probabilistico(momentos["A"], sr0))
    assert tabela.at["A", "IR"] == pytest.approx(momentos["A"].sharpe * np.sqrt(252))
    assert tabela.at["A", "MinTRL (anos)"] == pytest.approx(anos_minimos_para_psr(momentos["A"]))
    assert tabela.at["A", "P aritmético"] == pytest.approx(sharpe_probabilistico(momentos["A"]))


def _comparacao(p: float = 0.97, mdd_r10: float = -0.30, giro_r10: float = 2.0) -> dict[str, float]:
    return {"P[R10 > R5]": p, "MDD R10": mdd_r10, "MDD R5": -0.25, "Giro R10": giro_r10, "Giro R5": 1.0}


def test_r10_passa_so_com_c1_c2_e_c3_em_todas_as_fases_e_horizontes():
    # na fronteira passa: P = 0,95, MDD exatamente 5 p.p. pior, giro exatamente o dobro
    fronteira = pd.DataFrame([_comparacao(p=0.95)] * 6)
    assert r10_passa(fronteira)
    for ruim in (_comparacao(p=0.94), _comparacao(mdd_r10=-0.31), _comparacao(giro_r10=2.01)):
        assert not r10_passa(pd.DataFrame([_comparacao()] * 5 + [ruim]))


def test_trecho_comeca_no_fechamento_anterior_e_termina_no_ultimo_pregao_dentro():
    datas = pd.bdate_range("2009-12-01", "2015-01-30")
    assert trecho(datas, "2010-01-01", "2014-12-31") == (pd.Timestamp("2009-12-31"), pd.Timestamp("2014-12-31"))
    # a curva começa depois do início do trecho (fase 42): começa onde ela começa
    assert trecho(datas[30:], "2010-01-01", "2014-12-31")[0] == datas[30]


def test_comparacao_r10_mede_os_ultimos_anos_das_duas_curvas():
    datas = pd.bdate_range("2020-01-01", periods=600)
    rebalanceios = pd.DataFrame({"A": 1.0}, index=datas[::63][1:])

    def curva(semente: int, giro: float) -> ResultadoBacktest:
        return ResultadoBacktest("c", "walk-forward OOS", _aleatoria(semente, 0.0004, datas), rebalanceios,
                                 pd.Series(giro, index=rebalanceios.index), 0.0)

    r5, r10 = curva(1, 0.5), curva(2, 0.8)
    linha = comparar_r10(r5, r10, anos=1)
    inicio, fim = trecho(datas, datas[-1] - pd.DateOffset(years=1) + pd.Timedelta(days=1), datas[-1])
    so_r5, so_r10 = recortar(r5, inicio, fim), recortar(r10, inicio, fim)
    assert linha["P[R10 > R5]"] == pytest.approx(psr_da_diferenca(so_r10.valores, so_r5.valores))
    assert (linha["MDD R10"], linha["MDD R5"]) == (drawdown(so_r10.valores)[1], drawdown(so_r5.valores)[1])
    assert (linha["Giro R10"], linha["Giro R5"]) == (so_r10.turnover_anual_medio, so_r5.turnover_anual_medio)


def test_trecho_todo_antes_da_curva_nao_existe():
    datas = pd.bdate_range("2021-02-08", "2026-10-16")
    assert trecho(datas, "2010-01-01", "2014-12-31") is None
