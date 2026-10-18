"""Preços diários do lago da B3 (``lago/precos_diarios.parquet``), sem rede."""

import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import lago


@pytest.fixture
def arquivo(tmp_path):
    """O formato do export do lago: coluna ``date`` e uma coluna float32 por papel (ticker sem ``.SA``)."""
    caminho = tmp_path / "precos_diarios.parquet"
    pd.DataFrame({
        "date": pd.to_datetime(["2026-10-14", "2026-10-15", "2026-10-16"]).date,
        "PETR4": np.array([30.0, 31.0, 30.5], dtype="float32"),
        "BOVA11": np.array([200.0, 201.4, 202.3], dtype="float32"),
    }).to_parquet(caminho)
    return caminho


def test_fechamentos_do_lago_vem_com_o_ticker_do_app(arquivo):
    fechamentos = lago.ler_fechamentos(["PETR4.SA", "BOVA11.SA"], date(2026, 10, 1), date(2026, 10, 16), arquivo)
    assert list(fechamentos.columns) == ["PETR4.SA", "BOVA11.SA"]


def test_ticker_fora_do_lago_vem_como_coluna_vazia(arquivo):
    # como no Yahoo: o ``limpar_precos`` o descarta e avisa "sem dados" (SHOW3 parou de negociar)
    fechamentos = lago.ler_fechamentos(["PETR4.SA", "SHOW3.SA"], date(2026, 10, 1), date(2026, 10, 16), arquivo)
    assert (list(fechamentos.columns), fechamentos["SHOW3.SA"].isna().all()) == (["PETR4.SA", "SHOW3.SA"], True)


def test_fechamentos_do_lago_ficam_na_janela_pedida(arquivo):
    fechamentos = lago.ler_fechamentos(["PETR4.SA"], date(2026, 10, 15), date(2026, 10, 15), arquivo)
    assert list(fechamentos.index) == [pd.Timestamp("2026-10-15")]


def test_fechamentos_do_lago_vem_em_precisao_dupla(arquivo):
    # o arquivo guarda float32 (metade do tamanho); as contas do app são em float64
    fechamentos = lago.ler_fechamentos(["PETR4.SA", "BOVA11.SA"], date(2026, 10, 1), date(2026, 10, 16), arquivo)
    assert set(fechamentos.dtypes) == {np.dtype("float64")}


def test_versao_dos_dados_muda_quando_o_lago_republica_o_arquivo(monkeypatch, arquivo):
    # entra na chave do cache do app: uma republicação no mesmo dia não pode servir os preços antigos
    monkeypatch.setattr(lago, "ARQUIVO_PRECOS", arquivo)
    antes = lago.versao_dos_dados()
    os.utime(arquivo, ns=(arquivo.stat().st_atime_ns, arquivo.stat().st_mtime_ns + 1_000_000_000))
    assert lago.versao_dos_dados() != antes


def test_sem_arquivo_a_versao_e_zero_e_o_erro_claro_vem_da_leitura(monkeypatch, tmp_path):
    monkeypatch.setattr(lago, "ARQUIVO_PRECOS", tmp_path / "precos_diarios.parquet")
    assert lago.versao_dos_dados() == 0


def test_ler_setores_traz_o_setor_por_emissor_e_deixa_de_fora_quem_nao_tem(tmp_path):
    # formato do sectors.parquet do lago; setor nulo = holding sem setor principal, que o lago não adivinha
    caminho = tmp_path / "sectors.parquet"
    pd.DataFrame({"issuer": ["CESP", "GGPS"], "code_cvm": [2577, 25712],
                  "setor_ativ": ["Energia Elétrica", "Emp. Adm. Part. - Sem Setor Principal"],
                  "sector": ["Utilidade Pública", None]}).to_parquet(caminho)
    assert lago.ler_setores(caminho).to_dict() == {"CESP": "Utilidade Pública"}
