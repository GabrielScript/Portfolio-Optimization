"""Avaliação pré-registrada com o universo ponto no tempo do lago (partes sem rede)."""

import numpy as np
import pandas as pd
import pytest

from core.backtest import ResultadoBacktest, estrategia_equiponderada, walk_forward_dinamico
from core.config import ConfigBacktest
from scripts.avaliar_universo_dinamico import (ALTERNATIVOS, CONTROLE, Medidas, decidir_objetivo, misturar,
                                               recortar_horizonte)

DATAS = pd.bdate_range("2020-01-01", "2023-06-30")
CFG = ConfigBacktest(janela_treino=60, janela_teste=21, capital_inicial=1_000.0, custo_bps=0.0)


def _precos(*papeis: str) -> pd.DataFrame:
    gerador = np.random.default_rng(7)
    retornos = gerador.normal(0.0004, 0.01, (len(DATAS), len(papeis)))
    return pd.DataFrame(10 * np.cumprod(1 + retornos, axis=0), index=DATAS, columns=list(papeis))


def _universo(papeis: list[str]) -> pd.DataFrame:
    meses = pd.date_range(DATAS[0], DATAS[-1], freq="MS")
    return pd.DataFrame([(mes, papel) for mes in meses for papel in papeis], columns=["month", "security"])


def test_recorte_deixa_os_ultimos_anos_fora_da_amostra_com_o_treino_antes():
    precos = _precos("A", "B")
    recorte = recortar_horizonte(precos, anos=1, janela_treino=CFG.janela_treino)
    bt = walk_forward_dinamico(recorte, _universo(["A", "B"]), estrategia_equiponderada(), CFG, "1/N")
    assert bt.valores.index[1] == pd.Timestamp("2022-07-01")   # 1º pregão depois de 30/06/2022 (fim − 1 ano)
    assert bt.valores.index[-1] == DATAS[-1]


def test_recorte_sem_historico_para_o_treino_e_erro():
    # 3,5 anos de dados: 3 anos fora da amostra deixam ~130 pregões antes, menos que os 180 de treino
    with pytest.raises(ValueError, match="Histórico insuficiente"):
        recortar_horizonte(_precos("A", "B"), anos=3, janela_treino=CFG.janela_treino * 3)


def _curva(nome: str, valores: list[float]) -> ResultadoBacktest:
    datas = pd.bdate_range("2024-01-01", periods=len(valores))
    return ResultadoBacktest(nome=nome, metodologia="walk-forward OOS", valores=pd.Series(valores, index=datas),
                             pesos_alvo=pd.DataFrame(), turnover=pd.Series(dtype=float), custos_totais=0.0)


NUCLEO = _curva("núcleo", [1_000.0, 2_000.0, 2_000.0, 2_000.0, 2_000.0])      # dobra no 1º bloco
SATELITE = _curva("satélite", [1_000.0, 1_000.0, 1_000.0, 2_000.0, 2_000.0])  # dobra no 2º bloco
CFG_MISTURA = ConfigBacktest(janela_treino=60, janela_teste=2, capital_inicial=1_000.0, custo_bps=0.0)


def test_mistura_volta_aos_pesos_alvo_a_cada_rebalanceio():
    # 50/50: 1.000 -> 1.500 no 1º bloco; volta a 50/50 e o satélite dobra: 1.500 x 1,5 = 2.250 (sem voltar: 2.000)
    mistura = misturar(NUCLEO, SATELITE, peso_nucleo=0.5, cfg=CFG_MISTURA, nome="50/50")
    assert mistura.valores.index.equals(NUCLEO.valores.index)
    assert mistura.valores.tolist() == pytest.approx([1_000.0, 1_500.0, 1_500.0, 2_250.0, 2_250.0])


def test_mistura_paga_o_custo_sobre_o_que_muda_entre_nucleo_e_satelite():
    # 1%: montagem 1.000 -> 990; no rebalanceio o núcleo está em 2/3, giro 1/3: 1.485 x (1 - 1/300) x 1,5
    cfg = ConfigBacktest(janela_treino=60, janela_teste=2, capital_inicial=1_000.0, custo_bps=100.0)
    mistura = misturar(NUCLEO, SATELITE, peso_nucleo=0.5, cfg=cfg, nome="50/50")
    assert mistura.valores.iloc[-1] == pytest.approx(1_485.0 * (1 - 1 / 300) * 1.5)


def _m(sharpe: float = 0.50, max_drawdown: float = -0.30, p_controle_supera: float = float("nan")) -> Medidas:
    return Medidas(sharpe=sharpe, max_drawdown=max_drawdown, p_controle_supera=p_controle_supera)


def _horizonte(controle: Medidas, **outras: Medidas) -> dict[str, Medidas]:
    """Controle + os 4 alternativos (os não passados perdem em tudo: Sharpe 0, MDD -60%, P[Kelly > X] = 0,99)."""
    medidas = {nome: _m(0.0, -0.60, 0.99) for nome in ALTERNATIVOS}
    medidas.update(outras)
    return {CONTROLE: controle, **medidas}


def test_regra_a_opcao_4_quando_um_alternativo_vence_em_sharpe_e_mdd_nos_dois_horizontes():
    mais_e_menos = _m(0.60, -0.20, 0.30)          # Mín. Vol: Sharpe maior e MDD melhor que o Kelly (0,50; -30%)
    h = {5: _horizonte(_m(), **{"Mín. Vol": mais_e_menos}), 10: _horizonte(_m(), **{"Mín. Vol": mais_e_menos})}
    assert decidir_objetivo(h) == "4"


def test_regra_a_opcao_1_quando_o_kelly_vence_todos_com_significancia_e_mdd_aceitavel():
    # alternativos com Sharpe 0 e P[Kelly > X] = 0,99; o melhor MDD deles (-28%) fica a 2 p.p. do Kelly (-30%)
    melhor_mdd = _m(0.0, -0.28, 0.99)
    h = {5: _horizonte(_m(), **{"Mín. Vol": melhor_mdd}), 10: _horizonte(_m(), **{"Mín. Vol": melhor_mdd})}
    assert decidir_objetivo(h) == "1"
