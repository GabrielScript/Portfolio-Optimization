"""Avaliação pré-registrada com o universo ponto no tempo do lago (partes sem rede)."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from core.backtest import ResultadoBacktest, estrategia_equiponderada, retornos_da_carteira, walk_forward_dinamico
from core.config import ConfigBacktest, ConfigOtimizacao
from scripts.avaliar_universo_dinamico import (ALTERNATIVOS, CONTROLE, MISTURAS, REFERENCIAS, Medidas, avaliar,
                                               carregar_lago, imprimir,
                                               ESTRATEGIAS, decidir_mistura, decidir_objetivo, medir, montar_tabela,
                                               misturar, recortar_horizonte, rodar_horizonte, rodar_referencias)

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


def test_kelly_vence_em_sharpe_sem_significancia_da_opcao_2():
    # Mín. Vol perde em Sharpe (0,40 < 0,50), mas P[Kelly > Mín. Vol] = 0,60 fica abaixo de 0,90
    perde_sem_significancia = _m(0.40, -0.30, 0.60)
    h = {5: _horizonte(_m(), **{"Mín. Vol": perde_sem_significancia}),
         10: _horizonte(_m(), **{"Mín. Vol": perde_sem_significancia})}
    assert decidir_objetivo(h) == "2"


def _horizonte_misturas(**misturas: Medidas) -> dict[str, Medidas]:
    """Kelly λ=2 (Sharpe 0,50; MDD -30%) + as 4 misturas (as não passadas: Sharpe 0,50, MDD -30%, reprovadas)."""
    medidas = {nome: _m() for nome in MISTURAS}
    medidas.update(misturas)
    return {CONTROLE: _m(), **medidas}


def test_mistura_aprovada_com_maior_sharpe_medio_vira_padrao():
    # duas aprovadas nos dois horizontes (Sharpe >= 0,45 e MDD >= 10 p.p. melhor); médias 0,47 e 0,48
    a, b = MISTURAS[0], MISTURAS[2]
    h = {5: _horizonte_misturas(**{a: _m(0.46, -0.15), b: _m(0.50, -0.20)}),
         10: _horizonte_misturas(**{a: _m(0.48, -0.15), b: _m(0.46, -0.20)})}
    assert decidir_mistura(h) == b


def test_mistura_sem_mdd_10_pp_melhor_ou_sharpe_kelly_menos_005_em_um_horizonte_da_none():
    # a: Sharpe 0,44 < 0,45 só em 10 anos; b: MDD só 9 p.p. melhor em 5 anos; c, d: reprovadas nos dois
    a, b = MISTURAS[0], MISTURAS[1]
    h = {5: _horizonte_misturas(**{a: _m(0.60, -0.10), b: _m(0.60, -0.21)}),
         10: _horizonte_misturas(**{a: _m(0.44, -0.10), b: _m(0.60, -0.10)})}
    assert decidir_mistura(h) is None


def _exportar_lago(pasta, precos: pd.DataFrame, universo: pd.DataFrame) -> None:
    """Os dois exports e o manifest como o lago grava: ``date``/``month`` em ``datetime.date``, preços float32."""
    tabela = precos.astype("float32").reset_index(names="date")
    tabela["date"] = [d.date() for d in tabela["date"]]
    tabela.to_parquet(pasta / "universe_prices.parquet", index=False)
    universo.assign(month=[m.date() for m in pd.to_datetime(universo["month"])],
                    issuer=universo["security"].str[:4]).to_parquet(pasta / "universe_monthly.parquet", index=False)
    manifest = {"generated_at": "2026-10-17T20:15:24", "data_from": "1994-07-01", "data_through": "2026-10-16",
                "sources": {"prices": "séries"}}
    # o manifest do lago sai em cp1252 (não é UTF-8 válido)
    (pasta / "manifest.json").write_bytes(json.dumps(manifest, ensure_ascii=False).encode("cp1252"))


def test_carga_do_lago_normaliza_o_formato_real_do_export(tmp_path):
    _exportar_lago(tmp_path, _precos("A", "B"), _universo(["A", "B"]))
    dados = carregar_lago(tmp_path)
    assert isinstance(dados.precos.index, pd.DatetimeIndex)
    assert (dados.precos.dtypes == "float64").all()
    assert dados.universo["month"].iloc[0] == pd.Timestamp("2020-01-01")


def test_carga_do_lago_guarda_o_sha256_dos_exports_e_as_datas_do_manifest(tmp_path):
    _exportar_lago(tmp_path, _precos("A", "B"), _universo(["A", "B"]))
    dados = carregar_lago(tmp_path)
    assert dados.sha256 == {nome: hashlib.sha256((tmp_path / nome).read_bytes()).hexdigest()
                            for nome in ("universe_prices.parquet", "universe_monthly.parquet")}
    assert dados.manifest == {"generated_at": "2026-10-17T20:15:24", "data_from": "1994-07-01",
                              "data_through": "2026-10-16"}


def test_referencia_sem_cotacao_na_primeira_decisao_fica_sem_historico():
    precos = _precos(*REFERENCIAS)
    precos.loc[:"2021-06-30", "DIVO11"] = np.nan        # 1ª decisão em 03/2020 (60 pregões de treino)
    referencias = rodar_referencias(precos, CFG)
    assert referencias["DIVO11"] is None
    assert referencias["BOVA11"].valores.index.equals(retornos_da_carteira(precos).index[CFG.janela_treino - 1:])


def test_rodar_horizonte_poe_estrategias_referencias_e_misturas_nas_mesmas_datas():
    papeis = ["A", "B", "C", "D", "E", "F"]
    precos = _precos(*papeis, *REFERENCIAS)
    resultados = rodar_horizonte(precos, _universo(papeis), ConfigOtimizacao(), CFG, cdi=None)
    assert list(resultados) == [*ESTRATEGIAS, *REFERENCIAS, *MISTURAS]
    datas = resultados[CONTROLE].valores.index
    assert all(r.valores.index.equals(datas) for r in resultados.values())


def _curva_de_retornos(nome: str, retornos: np.ndarray) -> ResultadoBacktest:
    return _curva(nome, list(1_000.0 * np.cumprod(np.r_[1.0, 1.0 + retornos])))


def test_p_controle_supera_e_o_psr_do_kelly_menos_a_estrategia():
    gerador = np.random.default_rng(3)
    kelly = gerador.normal(0.0005, 0.01, 500)
    pior = kelly - 0.002 + gerador.normal(0.0, 0.001, 500)       # perde 0,2% ao dia do Kelly, com ruído
    resultados = {CONTROLE: _curva_de_retornos(CONTROLE, kelly), "Mín. Vol": _curva_de_retornos("Mín. Vol", pior)}
    medidas = medir(resultados, rf_diaria=pd.Series(0.0, index=resultados[CONTROLE].valores.index))
    assert medidas["Mín. Vol"].p_controle_supera > 0.99
    assert np.isnan(medidas[CONTROLE].p_controle_supera)


def test_dsr_so_nas_11_da_lista_fechada():
    gerador = np.random.default_rng(5)
    nomes = [*ESTRATEGIAS, *REFERENCIAS, *MISTURAS]
    resultados = {nome: _curva_de_retornos(nome, gerador.normal(0.0005, 0.01, 300)) for nome in nomes}
    rf = pd.Series(0.0, index=resultados[CONTROLE].valores.index)
    tabela = montar_tabela(resultados, medir(resultados, rf), rf)
    assert tabela.loc[[*ESTRATEGIAS, *MISTURAS], "DSR"].notna().all()
    assert tabela.loc[list(REFERENCIAS), "DSR"].isna().all()


def test_sem_cdi_usa_selic_constante_e_avisa_na_impressao(tmp_path, capsys):
    papeis = ["A", "B", "C", "D", "E", "F"]
    _exportar_lago(tmp_path, _precos(*papeis, *REFERENCIAS), _universo(papeis))
    avaliacao = avaliar(carregar_lago(tmp_path), cdi=pd.Series(dtype=float), selic=0.15, cfg_bt=CFG, horizontes=(1,))
    imprimir(avaliacao)
    assert avaliacao.rf_diaria[1].iloc[-1] == pytest.approx(1.15 ** (1 / 252) - 1)
    assert "CDI indisponível: taxa livre de risco = Selic constante de 15,00%" in capsys.readouterr().out
