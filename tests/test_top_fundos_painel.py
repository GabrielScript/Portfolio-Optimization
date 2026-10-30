"""The top-funds panel builder: the pre-registered quota export and the coverage it prints."""

import pytest

from scripts.top_fundos_painel import ESPERADOS, cobertura_por_ano
from tests.test_top_fundos import CONTAGENS, _contagens


def test_the_quota_export_hash_is_the_one_in_the_spec_and_attempt_52s_files_are_kept():
    assert ESPERADOS == {
        "fund_monthly_quota.parquet": "296fd03dcb36e250bcf12664ce77085414bbfa91b2a0903aad2e801a711dfbab",
        "fund_stock_holdings.parquet": "cd997a5d06f1b41b4988ef0da06518c1a032768380bd2ebee350af5e78b14feb",
        "fund_nav.parquet": "e12d47008b06a5a7eca999c0e00c3554bda862dfaceeacbb6f8893a4896121ee"}


def test_coverage_by_year_gives_median_counts_and_shares_and_the_months_dropped():
    tabela = cobertura_por_ano(_contagens(CONTAGENS))
    assert tabela.loc[2015].to_dict() == pytest.approx(
        {"months": 2, "qualifying": 30.0, "with_quotas": 0.7, "ranked": 0.5, "dropped": 1})
    assert tabela.loc[2016].to_dict() == pytest.approx(
        {"months": 2, "qualifying": 35.0, "with_quotas": 0.75, "ranked": 0.425, "dropped": 1})
