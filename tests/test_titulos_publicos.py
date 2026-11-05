"""core.titulos_publicos: the Tesouro IPCA+ index of the strategies comparison, from Tesouro Transparente's CSV."""

import pandas as pd
import pytest

from core import titulos_publicos as tp

CSV = """Tipo Titulo;Data Vencimento;Data Base;Taxa Compra Manha;Taxa Venda Manha;PU Compra Manha;PU Venda Manha;PU Base Manha
Tesouro IPCA+;15/05/2035;02/01/2026;7,00;7,12;1000,00;990,00;990,00
Tesouro IPCA+;15/05/2045;02/01/2026;7,10;7,22;500,00;495,00;495,00
Tesouro IPCA+;15/05/2026;02/01/2026;6,00;6,10;3000,00;2990,00;2990,00
Tesouro Selic;01/03/2029;02/01/2026;0,06;0,07;20185,67;20170,66;20170,66
Tesouro IPCA+;15/05/2035;05/01/2026;7,00;7,12;1010,00;999,90;999,90
Tesouro IPCA+;15/05/2045;05/01/2026;7,10;7,22;505,00;499,95;499,95
Tesouro IPCA+;15/05/2026;05/01/2026;6,00;6,10;3010,00;3020,00;3020,00
Tesouro Selic;01/03/2029;05/01/2026;0,06;0,07;20190,00;20175,00;20175,00
"""


def test_the_csv_reads_with_brazilian_dates_and_decimals():
    tabela = tp.ler_precos_e_taxas(CSV.encode("latin-1"))
    assert set(tabela.columns) >= {"tipo", "vencimento", "data", "pu_base"}
    assert tabela["data"].min() == pd.Timestamp("2026-01-02")
    assert tabela.loc[tabela["tipo"] == "Tesouro Selic", "pu_base"].iloc[0] == pytest.approx(20170.66)


def test_the_index_averages_the_ipca_bonds_daily_returns_leaving_out_those_near_maturity():
    # 2035 and 2045 each rise 1% (base PU); the 2026 bond (< 1 year left) and Tesouro Selic are left out
    indice = tp.indice_tesouro_ipca(tp.ler_precos_e_taxas(CSV.encode("latin-1")))
    assert list(indice.index) == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")]
    assert indice.iloc[0] == pytest.approx(1.0) and indice.iloc[-1] == pytest.approx(1.01)
    assert indice.name == tp.NOME
