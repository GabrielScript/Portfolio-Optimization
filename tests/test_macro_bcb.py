"""The Banco Central block of the market panel (the user's request of 2026-11-04): activity, foreign direct
investment, credit volume, its cost and its defaults, and the last Copom minutes, each with a short explanation."""

from datetime import date

import pandas as pd
import pytest

from core import macro_bcb


def _mensal(valores: list[float], fim: str = "2026-09-01") -> pd.Series:
    return pd.Series(valores, index=pd.date_range(end=fim, periods=len(valores), freq="MS"))


SERIES = {macro_bcb.SERIE_IBC_BR: _mensal([100.0] + [101.0] * 11 + [104.0]),          # 13 months: +4% on a year
          macro_bcb.SERIE_IDP: _mensal([1000.0] * 12),                                  # US$ 12 billion in 12 months
          macro_bcb.SERIE_CONCESSOES: _mensal([600_000.0] + [650_000.0] * 11 + [747_130.0]),
          macro_bcb.SERIE_JUROS_DO_CREDITO: _mensal([32.23, 32.36]),
          macro_bcb.SERIE_INADIMPLENCIA: _mensal([5.05, 4.98])}


def test_each_indicator_is_its_last_value_with_the_date_of_the_bcb_series():
    pedidos = []

    def baixar(serie, inicio, hoje):
        pedidos.append(serie)
        return SERIES[serie]
    indicadores = macro_bcb.indicadores(baixar, date(2026, 11, 4))
    assert set(pedidos) == set(SERIES)
    assert indicadores[macro_bcb.IBC_BR] == (pd.Timestamp("2026-09-01"), pytest.approx(4.0))
    assert indicadores[macro_bcb.IDP] == (pd.Timestamp("2026-09-01"), pytest.approx(12.0))
    assert indicadores[macro_bcb.CONCESSOES] == (pd.Timestamp("2026-09-01"), pytest.approx(747.13))
    assert indicadores[macro_bcb.CONCESSOES_12M] == (pd.Timestamp("2026-09-01"),
                                                    pytest.approx(100 * (747_130 / 600_000 - 1)))
    assert indicadores[macro_bcb.JUROS_DO_CREDITO] == (pd.Timestamp("2026-09-01"), 32.36)
    assert indicadores[macro_bcb.INADIMPLENCIA] == (pd.Timestamp("2026-09-01"), 4.98)
    assert set(macro_bcb.EXPLICACOES) >= set(indicadores) | {macro_bcb.ATA_DO_COPOM}


def test_a_series_that_fails_is_left_out_and_the_others_stay():
    def baixar(serie, inicio, hoje):
        if serie == macro_bcb.SERIE_IDP:
            raise macro_bcb.bcb.ErroBcb("SGS fora do ar")
        return SERIES[serie]
    indicadores = macro_bcb.indicadores(baixar, date(2026, 11, 4))
    assert macro_bcb.IDP not in indicadores and macro_bcb.INADIMPLENCIA in indicadores


def test_the_last_copom_minutes_come_from_the_bcb_site_with_a_full_link():
    resposta = {"conteudo": [{"DataReferencia": "2026-09-16T03:00:00Z", "Titulo": "281st Meeting - September 15-16, 2026",
                              "Url": "/content/copom/copomminutes/Ata_281_final.pdf"}]}
    ata = macro_bcb.ata_do_copom(lambda url: resposta)
    assert ata == macro_bcb.Ata(pd.Timestamp("2026-09-16"), "Ata da 281ª reunião do Copom",
                                "https://www.bcb.gov.br/content/copom/copomminutes/Ata_281_final.pdf")
    assert macro_bcb.ata_do_copom(lambda url: {"conteudo": []}) is None


def test_the_adviser_reads_the_new_indicators_and_the_copom_minutes():
    # the user, 2026-11-04: these indicators must reach the DeepSeek adviser too
    from core.assessor import macro_do_painel
    from core.market_data import TaxaSelic
    ata = macro_bcb.Ata(pd.Timestamp("2026-09-16"), "Ata da 281ª reunião do Copom", "https://www.bcb.gov.br/a.pdf")
    macro = macro_do_painel(date(2026, 11, 4), TaxaSelic(0.15, "BCB SGS 432"),
                            {macro_bcb.INADIMPLENCIA: (pd.Timestamp("2026-09-01"), 4.98)}, None, ata)
    textos = [texto for _, texto in macro]
    assert any(texto.startswith(f"{macro_bcb.INADIMPLENCIA} em 01/09/2026: 4,98") for texto in textos)
    assert ("2026-09-16", "Ata da 281ª reunião do Copom, publicada em 16/09/2026: https://www.bcb.gov.br/a.pdf") in macro
