"""The hybrid QPO's screen section and the loading of its derived files."""

import pandas as pd
from streamlit.testing.v1 import AppTest

TELA = """
import pandas as pd
from ui.qpo_hibrido import Mercado, ResultadoHibrido, secao

ic = pd.DataFrame({"mercado": ["b3", "b3"], "modelo": ["VQC", "Fourier"], "meses": [175, 175], "IC": [0.01, 0.03],
                   "t": [1.0, 3.0], "passa": [False, True]})
diferencas = pd.DataFrame({"mercado": ["b3"], "comparacao": ["VQC - Fourier"], "diferenca": [-0.02], "t": [-2.0]})
bancada = pd.DataFrame({"n": [12], "eixo": ["datas"], "solver": ["QAOA-hard"], "etapa": ["cru"], "viavel": [1.0],
                        "p_otimo": [0.01], "razao_media": [0.7], "razao_melhor": [1.0], "iqr_razao_media": [0.1]})
portao = pd.DataFrame({"braco": ["VQC-15"], "p_composto": [[0.1, 0.2, 0.3]], "p_conservadora": [[0.1, 0.2, 0.3]],
                       "p_referencia": [[0.9, 0.9, 0.9]], "dsr": [[0.1, 0.1, 0.1]], "passa": [PASSA]})
anos = pd.DataFrame({"Ano": ["2020"], "VQC-15": [0.1], "BOVA11": [0.02], "CDI": [0.03], "Selic": [0.03]})
carteiras = pd.DataFrame({"braco": ["VQC-15"], "papel": ["ABCD3"], "peso": [1 / 15], "data": [pd.Timestamp("2026-10-01")]})
mercado = Mercado(portao, anos, carteiras)
secao(ResultadoHibrido(ic, diferencas, bancada, {"b3": mercado, "us": mercado}))
"""


def _tela(passa: bool) -> AppTest:
    tela = AppTest.from_string(TELA.replace("PASSA", str(passa)), default_timeout=30).run()
    assert not tela.exception
    return tela


def _textos(tela: AppTest) -> str:
    return " ".join(bloco.value for bloco in [*tela.markdown, *tela.caption, *tela.warning, *tela.success])


def test_it_says_so_when_no_arm_passed_and_shows_the_quantum_edge_rule():
    tela = _tela(False)
    assert "Nenhum braço passou" in " ".join(aviso.value for aviso in tela.warning)
    assert "Informação, não recomendação" in _textos(tela)


def test_a_passing_arm_is_only_a_candidate():
    tela = _tela(True)
    assert "candidato" in " ".join(aviso.value for aviso in tela.warning)
    assert "VQC-15" in " ".join(aviso.value for aviso in tela.warning)


def test_the_tables_of_both_markets_are_on_the_screen():
    tela = _tela(False)
    # forecast, its differences, bench; then gate, years and picks per market
    assert len(tela.dataframe) == 3 + 2 * 3


def test_without_results_it_says_it_has_not_run():
    tela = AppTest.from_string("from ui.qpo_hibrido import secao\nsecao(None)", default_timeout=30).run()
    assert not tela.exception and "ainda não" in tela.info[0].value


def test_loading_needs_all_the_files(tmp_path):
    from ui.qpo_hibrido import carregar

    assert carregar(tmp_path) is None
    tabela = pd.DataFrame({"x": [1]})
    for nome in ("ic", "diferencas", "bancada"):
        tabela.to_parquet(tmp_path / f"qpo-hibrido-{nome}.parquet")
    for mercado in ("b3", "us"):
        for nome in ("portao", "anos", "carteiras"):
            tabela.to_parquet(tmp_path / f"qpo-hibrido-{nome}-{mercado}.parquet")
    assert set(carregar(tmp_path).mercados) == {"b3", "us"}
