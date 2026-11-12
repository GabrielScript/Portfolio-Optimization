"""Insider dealing, the weight a score earns and the mean of strategies
(docs/superpowers/specs/2026-11-12-informados-design.md)."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from core import informados

M = pd.Timestamp


def _filings(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["cnpj_root", "reference", "version", "delivered", "bought", "sold"]).assign(
        reference=lambda t: pd.to_datetime(t["reference"]), delivered=lambda t: pd.to_datetime(t["delivered"]))


def test_the_ratio_reads_only_filings_delivered_before_the_month() -> None:
    arquivos = _filings([("11111111", "2020-03-01", 1, "2020-04-10", 100.0, 0.0),
                         ("11111111", "2020-04-01", 1, "2020-05-10", 0.0, 300.0)])
    assert informados.net_purchase(arquivos, M("2020-05-01"))["11111111"] == pytest.approx(1.0)
    assert informados.net_purchase(arquivos, M("2020-06-01"))["11111111"] == pytest.approx(-0.5)
    assert informados.net_purchase(arquivos, M("2020-04-01")).empty      # delivered on the 10th: not known on the 1st


def test_a_later_version_replaces_the_earlier_one_only_once_it_is_delivered() -> None:
    arquivos = _filings([("11111111", "2020-03-01", 1, "2020-04-10", 100.0, 0.0),
                         ("11111111", "2020-03-01", 2, "2020-06-20", 0.0, 100.0)])
    assert informados.net_purchase(arquivos, M("2020-06-01"))["11111111"] == pytest.approx(1.0)
    assert informados.net_purchase(arquivos, M("2020-07-01"))["11111111"] == pytest.approx(-1.0)


def test_the_ratio_covers_six_reference_months_and_needs_a_trade() -> None:
    arquivos = _filings([("11111111", "2020-01-01", 1, "2020-02-10", 100.0, 0.0),
                         ("22222222", "2020-06-01", 1, "2020-07-10", 0.0, 0.0)])
    assert "11111111" in informados.net_purchase(arquivos, M("2020-07-01"))      # January is one of the six before July
    assert "11111111" not in informados.net_purchase(arquivos, M("2020-08-01"))  # and not of the six before August
    assert "22222222" not in informados.net_purchase(arquivos, M("2020-08-01"))  # a filing without a trade: no score


def test_the_score_goes_to_every_ticker_of_the_issuer() -> None:
    arquivos = _filings([("11111111", "2020-03-01", 1, "2020-04-10", 100.0, 300.0)])
    nota = informados.insider_score(arquivos, pd.Series({"AAAA": "11111111", "BBBB": "99999999"}))
    assert nota(M("2020-05-01"), ["AAAA3", "AAAA4", "BBBB3", "CCCC3"]).to_dict() == {"AAAA3": -0.5, "AAAA4": -0.5}


def _zip(movimentos: list[dict], documentos: list[dict], ano: int = 2020) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as arquivo:
        for nome, linhas in ((f"vlmo_cia_aberta_con_{ano}.csv", movimentos), (f"vlmo_cia_aberta_{ano}.csv", documentos)):
            arquivo.writestr(nome, pd.DataFrame(linhas).to_csv(sep=";", index=False).encode("latin1"))
    return buffer.getvalue()


def test_reading_keeps_cash_trades_of_persons_in_the_companys_own_shares(tmp_path) -> None:
    base = {"CNPJ_Companhia": "11.111.111/0001-11", "Data_Referencia": "2020-03-01", "Versao": "1",
            "Tipo_Empresa": "Companhia", "Tipo_Cargo": "Diretor ou Vinculado", "Tipo_Ativo": "Ações",
            "Tipo_Movimentacao": "Compra à vista", "Quantidade": "10", "Preco_Unitario": "5", "Volume": "50"}
    movimentos = [base,
                  {**base, "Tipo_Movimentacao": "Venda à vista", "Volume": "", "Quantidade": "4"},      # 4 x 5 = 20
                  {**base, "Tipo_Cargo": ""},                                    # the company's own treasury
                  {**base, "Tipo_Ativo": "Debêntures"},
                  {**base, "Tipo_Empresa": "Controladora"},
                  {**base, "Tipo_Movimentacao": "Saldo Inicial"}]
    documentos = [{"CNPJ_Companhia": "11.111.111/0001-11", "Data_Referencia": "2020-03-01", "Versao": "1",
                   "Data_Entrega": "2020-04-10"},
                  {"CNPJ_Companhia": "22.222.222/0001-22", "Data_Referencia": "2020-03-01", "Versao": "1",
                   "Data_Entrega": "2020-04-09"}]
    (tmp_path / "vlmo_2020.zip").write_bytes(_zip(movimentos, documentos))
    arquivos = informados.read_filings(tmp_path).set_index("cnpj_root")
    assert arquivos.loc["11111111", ["bought", "sold"]].tolist() == [50.0, 20.0]
    assert arquivos.loc["22222222", ["bought", "sold"]].tolist() == [0.0, 0.0]   # a filing with no trade is kept
    assert arquivos.loc["11111111", "delivered"] == M("2020-04-10")


def test_the_weight_is_earned_from_past_months_only_and_needs_three_years() -> None:
    meses = pd.date_range("2015-01-01", periods=60, freq="MS")
    base = pd.Series(0.04, index=meses)
    extra = pd.Series(0.01, index=meses)
    extra.iloc[40:] = 0.20                                           # the future is much better: it must not count
    assert informados.earned_weight(extra, base, meses[36]) == pytest.approx(0.25)
    assert informados.earned_weight(extra, base, meses[35]) == 0.0   # 35 months before it: not enough
    assert informados.earned_weight(-extra, base, meses[36]) == 0.0  # a score that has lost earns nothing
    assert informados.earned_weight(base * 3, base, meses[36]) == 1.0    # and never more than the Composite itself
    furos = extra.copy()
    furos.iloc[:10] = np.nan                                         # months without the extra score do not count
    assert informados.earned_weight(furos, base, meses[40]) == 0.0


def test_the_weighted_score_adds_nothing_for_a_name_without_the_extra_score() -> None:
    base = lambda mes, candidatos: pd.Series({"A": 3.0, "B": 2.0, "C": 1.0})                      # noqa: E731
    extra = lambda mes, candidatos: pd.Series({"C": 9.0, "B": 1.0, "Z": 5.0})                     # noqa: E731
    sem = informados.weighted_score(base, extra, lambda mes: 0.0)(M("2020-01-01"), ["A", "B", "C"])
    com = informados.weighted_score(base, extra, lambda mes: 1.0)(M("2020-01-01"), ["A", "B", "C"])
    assert list(sem.sort_values(ascending=False).index) == ["A", "B", "C"] and "Z" not in com
    assert com["A"] == pytest.approx(sem["A"])                       # A has no extra score: its own z, nothing added
    assert com["C"] > sem["C"] and com["B"] < sem["B"]


def test_the_mean_of_strategies_averages_their_books() -> None:
    um = lambda treino: pd.Series({"A": 0.5, "B": 0.5})              # noqa: E731
    outro = lambda treino: pd.Series({"B": 0.5, "C": 0.5})           # noqa: E731
    pesos = informados.mean_of([um, outro])(pd.DataFrame())
    assert pesos.to_dict() == pytest.approx({"A": 0.25, "B": 0.5, "C": 0.25})
    with pytest.raises(ValueError):
        informados.mean_of([])
