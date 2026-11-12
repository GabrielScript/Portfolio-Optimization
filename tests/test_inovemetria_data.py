"""Which patent applications an issuer owns at a month: its own CNPJ root and the companies its latest FRE known
before the month declares controlled, each application published before the month, counted once."""

import pandas as pd

from core.inovemetria.data import InnovationData, issuer_patents

T = pd.Timestamp


def _data(subsidiaries: list[tuple], patents: list[tuple]) -> InnovationData:
    return InnovationData(
        fundamentals=pd.DataFrame(),
        patents=pd.DataFrame(patents, columns=["application", "cnpj_root", "filed_on", "known_from", "fields"]),
        subsidiaries=pd.DataFrame(subsidiaries, columns=["issuer", "cnpj_root", "dt_refer", "known_from"]),
        roots=pd.Series({"WEGE": "84429695"}),
        activity=pd.Series(dtype=str),
        ipca=pd.Series(dtype=float))


def test_a_subsidiarys_patent_counts_only_after_the_fre_that_names_it():
    data = _data([("WEGE", "07175725", T("2024-12-31"), T("2025-06-02"))],
                 [("PI1", "07175725", T("2022-01-01"), T("2023-07-01"), [4])])
    assert issuer_patents(data, T("2025-06-01")).empty
    assert issuer_patents(data, T("2025-07-01"))["application"].tolist() == ["PI1"]


def test_the_controlled_set_is_the_latest_fre_before_the_month():
    data = _data([("WEGE", "07175725", T("2023-12-31"), T("2024-05-30")),      # FRE 2024 names it
                  ("WEGE", "02810190", T("2024-12-31"), T("2025-06-02"))],     # FRE 2025 names only another one
                 [("PI1", "07175725", T("2022-01-01"), T("2023-07-01"), [4])])
    assert issuer_patents(data, T("2025-01-01"))["application"].tolist() == ["PI1"]
    assert issuer_patents(data, T("2025-07-01")).empty


def test_an_application_of_parent_and_subsidiary_counts_once():
    data = _data([("WEGE", "07175725", T("2023-12-31"), T("2024-05-30"))],
                 [("PI1", "84429695", T("2022-01-01"), T("2023-07-01"), [4]),
                  ("PI1", "07175725", T("2022-01-01"), T("2023-07-01"), [4])])
    assert issuer_patents(data, T("2025-01-01"))["application"].tolist() == ["PI1"]
