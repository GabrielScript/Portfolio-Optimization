"""The HRP table on the US screen (docs/superpowers/specs/2026-11-01-us-hrp-design.md, sections 4 and 6)."""

import pandas as pd
from streamlit.testing.v1 import AppTest

from tests.test_us_teste import _pasta_de_resultados, _tela_eua


def test_the_screen_shows_the_hrp_table_since_2026_when_its_file_exists(tmp_path):
    _pasta_de_resultados(tmp_path)
    pd.DataFrame({"nome": ["HRP amostral", "CDI"], "Retorno": [0.05, 0.11], "Vol": [0.12, 0.0], "Queda máx.": [-0.04, 0.0]}).to_parquet(
        tmp_path / "us-hrp-desde-2026.parquet", index=False)
    tela = AppTest.from_function(_tela_eua, args=(str(tmp_path),), default_timeout=60).run()
    assert not tela.exception
    tabelas = [tabela.value for tabela in tela.dataframe if "Retorno" in tabela.value.columns]
    assert tabelas[0]["nome"].tolist() == ["HRP amostral", "CDI"]
