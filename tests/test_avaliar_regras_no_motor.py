"""Regras de risco fixas no motor dinâmico (item g): partes sem rede."""

import json

import numpy as np
import pandas as pd
import pytest

from core.backtest import ResultadoBacktest
from core.config import ConfigBacktest
from scripts.avaliar_regras_no_motor import (MISTURA, SATELITE, avaliar_regras, carregar, data_do_stop,
                                             imprimir_regras, satelite_com_stop, setores_dos_papeis,
                                             violacoes_por_regra)
from scripts.avaliar_universo_dinamico import REFERENCIAS

DATAS = pd.bdate_range("2024-01-01", periods=7)


def _curva(valores: list[float], rebalanceios: list[int] = ()) -> ResultadoBacktest:
    """Curva diária com os rebalanceios nas posições dadas (1º pregão com os pesos novos)."""
    alvo = pd.DataFrame({"A": 1.0}, index=DATAS[list(rebalanceios)])
    return ResultadoBacktest(nome="curva", metodologia="walk-forward OOS", valores=pd.Series(valores, index=DATAS),
                             pesos_alvo=alvo, turnover=pd.Series(0.0, index=alvo.index), custos_totais=0.0)


def test_stop_dispara_no_rebalanceio_em_que_a_razao_ja_caiu_25_por_cento():
    # a queda para 74 acontece no pregão do 2º rebalanceio (posição 3): na decisão, véspera, a razão caiu só 10%;
    # o stop sai no 3º (posição 5), não antes
    satelite = _curva([100.0, 100.0, 90.0, 74.0, 74.0, 74.0, 74.0], rebalanceios=[1, 3, 5])
    bova = _curva([100.0] * 7)
    assert data_do_stop(satelite, bova) == DATAS[5]


def test_satelite_vai_para_o_bova11_no_stop_pagando_venda_e_compra_e_fica_la():
    # stop no rebalanceio da posição 5: o satélite vale 74 na véspera, paga 2 x 10 bps e passa a andar com o BOVA11
    satelite = _curva([100.0, 100.0, 90.0, 74.0, 74.0, 70.0, 60.0], rebalanceios=[1, 3, 5])
    bova = _curva([100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 121.0])
    trocado = satelite_com_stop(satelite, bova, DATAS[5], custo_bps=10.0)
    assert trocado.valores.tolist() == pytest.approx([100.0, 100.0, 90.0, 74.0, 74.0, 74.0 * 0.998 * 1.1,
                                                      74.0 * 0.998 * 1.21])


def test_setor_do_papel_vem_pelo_emissor_do_universo():
    # "OLD:PDG 4" (ticker de antes de 1998) é da PDGR; GGPS (holding sem setor principal) fica de fora
    universo = pd.DataFrame({"month": pd.to_datetime(["1996-01-01", "2024-01-01", "2024-01-01", "2024-02-01"]),
                             "security": ["OLD:PDG 4", "PETR4", "GGPS3", "PETR4"],
                             "issuer": ["PDGR", "PETR", "GGPS", "PETR"]})
    setores = pd.Series({"PDGR": "Consumo Cíclico", "PETR": "Petróleo, Gás e Biocombustíveis"})
    assert setores_dos_papeis(universo, setores) == {"OLD:PDG 4": "Consumo Cíclico",
                                                      "PETR4": "Petróleo, Gás e Biocombustíveis"}


def test_violacoes_contadas_uma_vez_por_rebalanceio_e_regra():
    # 1º alvo: A e B com 25% (duas ações acima de 20%, uma contagem) e setor X com 50%; 2º alvo: D sem setor
    alvo = pd.DataFrame({"A": [0.25, 0.2], "B": [0.25, 0.2], "C": [0.5, 0.4], "D": [0.0, 0.2]},
                        index=DATAS[[1, 3]])
    resultado = ResultadoBacktest(nome="HRP", metodologia="walk-forward OOS", valores=pd.Series(dtype=float),
                                  pesos_alvo=alvo, turnover=pd.Series(dtype=float), custos_totais=0.0)
    setores = {"A": "X", "B": "X", "C": "Y"}
    assert violacoes_por_regra(resultado, setores) == {"peso por ação": 2, "teto de setor": 2, "papel sem setor": 1}


def _exportar_lago(pasta, papeis: list[str], setores: dict[str, str]) -> None:
    """Os três exports como o lago grava (``date``/``month`` em ``datetime.date``, preços float32) e o manifest."""
    datas = pd.bdate_range("2020-01-01", "2022-06-30")
    gerador = np.random.default_rng(7)
    nomes = [*papeis, *REFERENCIAS]
    precos = pd.DataFrame(10 * np.cumprod(1 + gerador.normal(0.0004, 0.01, (len(datas), len(nomes))), axis=0),
                          index=datas, columns=nomes).astype("float32").reset_index(names="date")
    precos["date"] = [d.date() for d in precos["date"]]
    precos.to_parquet(pasta / "universe_prices.parquet", index=False)
    meses = pd.date_range(datas[0], datas[-1], freq="MS")
    pd.DataFrame([(mes.date(), papel, papel) for mes in meses for papel in papeis],
                 columns=["month", "security", "issuer"]).to_parquet(pasta / "universe_monthly.parquet", index=False)
    pd.DataFrame({"issuer": papeis, "code_cvm": range(len(papeis)), "setor_ativ": "x",
                  "sector": [setores.get(p) for p in papeis]}).to_parquet(pasta / "sectors.parquet", index=False)
    manifest = {"generated_at": "2026-10-18T14:22:49", "data_from": "1994-07-01", "data_through": "2026-10-16"}
    (pasta / "manifest.json").write_bytes(json.dumps(manifest).encode("cp1252"))


def test_avaliacao_com_regras_roda_no_formato_do_lago_e_imprime_os_disparos(tmp_path, capsys):
    # H não tem setor no lago: não entra no satélite; o DSR da mistura com regras sai (N = 12)
    papeis = list("ABCDEFGH")
    _exportar_lago(tmp_path, papeis, {"A": "X", "B": "X", "C": "Y", "D": "Y", "E": "Z", "F": "Z", "G": "W"})
    cfg = ConfigBacktest(janela_treino=60, janela_teste=21, capital_inicial=1_000.0, custo_bps=10.0)
    avaliacao = avaliar_regras(carregar(tmp_path), cdi=pd.Series(dtype=float), selic=0.15, cfg_bt=cfg, horizontes=(1,))
    imprimir_regras(avaliacao)
    satelite = avaliacao.regras[1][SATELITE]
    assert satelite.pesos_alvo.get("H", pd.Series(0.0)).eq(0.0).all()
    assert satelite.valores.index.equals(avaliacao.base.resultados[1]["Kelly λ=2"].valores.index)
    assert pd.notna(avaliacao.tabelas[1].loc[MISTURA, "DSR"])
    saida = capsys.readouterr().out
    assert "PARTE 2" in saida and "Stop relativo do satélite:" in saida and "sha256 sectors.parquet" in saida
