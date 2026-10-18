"""Análise de poder dos portões do S2 (partes sem rede)."""

import json

import pandas as pd

from scripts.poder_dos_portoes import PORTOES, carregar_painel, escolher_portao


def test_escolhe_o_primeiro_portao_da_lista_com_poder_e_falso_positivo_dentro():
    # o atual (IC >= 0,03) tem ~50% de poder em IC 0,03; o 2º cumpre os dois critérios; o 3º também, mas vem depois
    taxas = {0.0: {PORTOES[0].nome: 0.01, PORTOES[1].nome: 0.012, PORTOES[2].nome: 0.024},
             0.03: {PORTOES[0].nome: 0.50, PORTOES[1].nome: 0.86, PORTOES[2].nome: 0.92}}
    assert escolher_portao(taxas) == PORTOES[1].nome


def test_sem_portao_que_cumpra_os_dois_criterios_nenhum_e_escolhido():
    taxas = {0.0: {p.nome: 0.06 for p in PORTOES}, 0.03: {p.nome: 0.90 for p in PORTOES}}
    assert escolher_portao(taxas) is None


def test_painel_so_com_o_universo_do_mes_e_sem_o_mes_aberto(tmp_path):
    # B só entra no universo em fevereiro; outubro de 2026 ainda está aberto (lago até 16/10)
    meses = pd.to_datetime(["2026-08-01", "2026-09-01", "2026-10-01"])
    pd.DataFrame({"month": meses.date, "A": [0.01, 0.02, 0.03], "B": [0.04, 0.05, 0.06]}).to_parquet(
        tmp_path / "monthly_total_returns.parquet", index=False)
    pd.DataFrame({"month": pd.to_datetime(["2026-08-01", "2026-09-01", "2026-09-01", "2026-10-01"]).date,
                  "security": ["A", "A", "B", "A"], "issuer": ["A", "A", "B", "A"]}).to_parquet(
        tmp_path / "universe_monthly.parquet", index=False)
    (tmp_path / "manifest.json").write_bytes(json.dumps({"data_through": "2026-10-16"}).encode("cp1252"))
    painel = carregar_painel(tmp_path, inicio="2026-01-01")
    assert list(painel.index) == list(meses[:2])
    assert painel["A"].tolist() == [0.01, 0.02] and painel["B"].isna().tolist() == [True, False]
