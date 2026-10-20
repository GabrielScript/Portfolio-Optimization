"""Diário do paper: um fechamento por pregão e revisão quando o lago muda um preço passado."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from core.diario import atualizar_diario, ler_diario, sha256_do_arquivo, texto_das_revisoes
from core.registro import registrar_carteira

DATAS = pd.bdate_range("2026-07-01", periods=20)
AGORA = datetime(2026, 7, 29, 8, 30)


@pytest.fixture
def precos() -> pd.DataFrame:
    """Índice de retorno total no formato que a rotina lê (float32 do parquet, convertido para float64)."""
    return pd.DataFrame({
        "A": np.linspace(10.0, 12.0, len(DATAS)).astype("float32"),
        "BOVA11": np.linspace(200.0, 204.0, len(DATAS)).astype("float32"),
    }, index=DATAS).astype("float64")


@pytest.fixture
def carteiras(tmp_path) -> pd.DataFrame:
    arquivo = tmp_path / "carteiras.csv"
    registrar_carteira(DATAS[0].date(), pd.Series({"A": 1.0}), pd.Series({"A": 10.0}), arquivo)
    return pd.read_csv(arquivo, parse_dates=["data"])


def test_primeira_rodada_grava_um_fechamento_por_pregao_desde_o_registro(tmp_path, carteiras, precos):
    arquivo = tmp_path / "diario.csv"
    revistos = atualizar_diario(arquivo, carteiras, precos, "abc", AGORA)
    diario = ler_diario(arquivo)
    assert (revistos, diario["fechamento"].tolist(), set(diario["tipo"])) == ([], list(DATAS), {"novo"})
    assert (diario["satelite"].iloc[0], diario["bova"].iloc[0]) == (1.0, 1.0)
    assert diario["satelite"].iloc[-1] == pytest.approx(1.2)


def test_rodar_de_novo_com_os_mesmos_precos_nao_grava_nada(tmp_path, carteiras, precos):
    arquivo = tmp_path / "diario.csv"
    atualizar_diario(arquivo, carteiras, precos, "abc", AGORA)
    antes = arquivo.read_bytes()
    assert (atualizar_diario(arquivo, carteiras, precos, "abc", AGORA), arquivo.read_bytes()) == ([], antes)


def test_pregao_novo_entra_como_novo_sem_revisao(tmp_path, carteiras, precos):
    arquivo = tmp_path / "diario.csv"
    atualizar_diario(arquivo, carteiras, precos.iloc[:-1], "abc", AGORA)
    revistos = atualizar_diario(arquivo, carteiras, precos, "def", AGORA)
    diario = ler_diario(arquivo)
    assert (revistos, len(diario), diario["fechamento"].iloc[-1], diario["precos_sha256"].iloc[-1]) == (
        [], len(DATAS), DATAS[-1], "def")


def test_lago_que_muda_preco_passado_vira_revisao_com_a_data_e_avisa_uma_vez(tmp_path, carteiras, precos):
    # provento lançado depois: o índice de retorno total de A sobe 1% do 11º pregão em diante
    arquivo = tmp_path / "diario.csv"
    atualizar_diario(arquivo, carteiras, precos, "abc", AGORA)
    revisto = precos.copy()
    revisto.loc[DATAS[10]:, "A"] *= 1.01
    primeira = atualizar_diario(arquivo, carteiras, revisto, "def", AGORA)
    segunda = atualizar_diario(arquivo, carteiras, revisto, "def", AGORA)
    revisoes = ler_diario(arquivo).query("tipo == 'revisao'")
    assert (primeira, segunda, revisoes["fechamento"].tolist()) == (list(DATAS[10:]), [], list(DATAS[10:]))
    assert revisoes["satelite"].iloc[-1] == pytest.approx(1.2 * 1.01)


def test_hash_dos_precos_e_o_sha256_do_arquivo(tmp_path):
    arquivo = tmp_path / "precos.parquet"
    arquivo.write_bytes(b"abc")
    assert sha256_do_arquivo(arquivo) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_texto_das_revisoes_diz_quantos_fechamentos_e_o_intervalo():
    assert texto_das_revisoes([DATAS[10], DATAS[11], DATAS[19]]) == (
        "lago mudou 3 fechamento(s) já gravado(s) no diário, de 15/07/2026 a 28/07/2026")


def test_lago_que_corrige_o_dia_do_pico_revisa_o_pico_dos_dias_seguintes(tmp_path, carteiras, precos):
    # um tick errado de +30% em A no 6º pregão fez o pico; o lago corrige só esse dia: a razão dos dias seguintes
    # não muda, mas o pico gravado neles (o que o stop relativo olha) muda
    arquivo = tmp_path / "diario.csv"
    errado = precos.copy()
    errado.loc[DATAS[5], "A"] *= 1.3
    atualizar_diario(arquivo, carteiras, errado, "abc", AGORA)
    revistos = atualizar_diario(arquivo, carteiras, precos, "def", AGORA)
    assert revistos == list(DATAS[5:])


def test_pregao_que_o_lago_poe_no_meio_do_passado_vira_revisao(tmp_path, carteiras, precos):
    # o diário viu os pregões sem o 8º; o lago repõe esse dia: muda a contagem de pregões e o retorno diário
    arquivo = tmp_path / "diario.csv"
    atualizar_diario(arquivo, carteiras, precos.drop(DATAS[7]), "abc", AGORA)
    revistos = atualizar_diario(arquivo, carteiras, precos, "def", AGORA)
    linha = ler_diario(arquivo).iloc[-1]
    assert (revistos, linha["fechamento"], linha["tipo"]) == ([DATAS[7]], DATAS[7], "revisao")
