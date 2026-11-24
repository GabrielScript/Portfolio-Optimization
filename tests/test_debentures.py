"""The debenture credit scores (docs/superpowers/specs/2026-11-24-debentures-design.md): SND loading, the monthly
measure, CRD-N / CRD-V, the earned weight and the veto. Synthetic data only: the SND files stay on D:."""

import pandas as pd

from core.debentures import read_characteristics

HEADER = "\t".join(["Codigo do Ativo", "Empresa        ", "indice", "Serie", "CNPJ"])


def test_characteristics_parsed_by_column_name():
    texto = "\n".join(["PARA CONSULTAR ...", "", HEADER,
                       "\t".join(["AAJR11    ", "FILADELFIA S/A", "DI", " UNI", "31907982000123"]),
                       "\t".join(["BBBB12    ", "OUTRA S/A", "IPCA", " 001", "08213823000107"])])
    caracteristicas = read_characteristics(texto)
    assert caracteristicas["cod_ativo"].tolist() == ["AAJR11", "BBBB12"]
    assert caracteristicas["cnpj_root"].tolist() == ["31907982", "08213823"]
    assert caracteristicas["indice"].tolist() == ["DI", "IPCA"]


def _snd_folder(tmp_path, wrong: bool = False):
    import hashlib
    raw = tmp_path / "raw"
    (raw / "caracteristicas").mkdir(parents=True)
    trades = pd.DataFrame({"data": [__import__("datetime").date(2020, 3, 2)], "emissor": ["A"], "cod_ativo": ["AAJR11"],
                           "isin": ["X"], "quantidade": [1.0], "n_negocios": [1.0], "pu_min": [100.0],
                           "pu_med": [100.0], "pu_max": [100.0], "pct_pu_curva": [99.0]})
    trades.to_parquet(raw / "snd_trades.parquet", index=False)
    (raw / "caracteristicas" / "caracteristicas.txt").write_text(
        "\n".join(["preamble", HEADER, "\t".join(["AAJR11", "A S/A", "DI", "1", "31907982000123"])]), encoding="utf-8")
    nomes = ["snd_trades.parquet", "caracteristicas/caracteristicas.txt"]
    hashes = {nome: hashlib.sha256((raw / nome).read_bytes()).hexdigest() for nome in nomes}
    if wrong:
        hashes[nomes[1]] = "0" * 64
    sums = tmp_path / "SHA256SUMS"
    sums.write_text("".join(f"{valor} *{nome}\n" for nome, valor in hashes.items()), encoding="ascii")
    return raw, sums, hashes


def test_load_snd_stops_on_a_hash_mismatch(tmp_path):
    import pytest
    from core.debentures import load_snd
    raw, sums, _ = _snd_folder(tmp_path, wrong=True)
    with pytest.raises(SystemExit, match="caracteristicas"):
        load_snd(raw, sums)


def test_load_snd_reads_matching_files(tmp_path):
    from core.debentures import load_snd
    raw, sums, hashes = _snd_folder(tmp_path)
    trades, caracteristicas, sha256 = load_snd(raw, sums)
    assert pd.api.types.is_datetime64_any_dtype(trades["data"])
    assert caracteristicas["cnpj_root"].tolist() == ["31907982"]
    assert sha256 == hashes


def test_issuer_roots_maps_cvm_code_to_issuer():
    from core.debentures import issuer_roots
    code_roots = pd.DataFrame({"code_cvm": [10, 20, 30], "cnpj_root": ["11111111", "22222222", "33333333"]})
    sectors = pd.DataFrame({"issuer": ["AAAA", "BBBB", "CCCC"], "code_cvm": [10.0, 20.0, None]})
    assert issuer_roots(code_roots, sectors).to_dict() == {"AAAA": "11111111", "BBBB": "22222222"}


CHARS = pd.DataFrame({"cod_ativo": ["AAAA11", "AAAA21", "BBBB11"], "cnpj_root": ["11111111", "11111111", "22222222"],
                      "indice": ["DI", "IPCA", "DI"]})


def _trades(rows):
    return pd.DataFrame(rows, columns=["data", "cod_ativo", "quantidade", "pu_med", "pct_pu_curva"]).assign(
        data=lambda t: pd.to_datetime(t["data"]))


def test_measure_is_volume_weighted_di_only_within_bounds():
    from core.debentures import monthly_measure
    trades = _trades([("2020-03-02", "AAAA11", 1, 100.0, 98.0), ("2020-03-10", "AAAA11  ", 3, 100.0, 102.0),
                      ("2020-03-11", "AAAA11", 1, 100.0, 160.0), ("2020-03-12", "AAAA11", 1, 100.0, None),
                      ("2020-03-13", "AAAA21", 5, 100.0, 80.0)])
    medida = monthly_measure(trades, CHARS)
    assert medida.to_dict() == {(pd.Timestamp("2020-03-01"), "11111111"): 101.0}


def test_measure_skips_a_month_with_no_weight():
    from core.debentures import monthly_measure
    trades = _trades([("2020-03-02", "AAAA11", 0, 100.0, 98.0), ("2020-03-03", "AAAA11", None, 100.0, 99.0),
                      ("2020-04-02", "AAAA11", 2, 100.0, 97.0)])
    medida = monthly_measure(trades, CHARS)
    assert medida.to_dict() == {(pd.Timestamp("2020-04-01"), "11111111"): 97.0}


ROOTS = pd.Series({"AAAA": "11111111", "BBBB": "22222222"})


def _measure(values):
    return pd.Series({(pd.Timestamp(mes), raiz): valor for (mes, raiz), valor in values.items()})


def test_level_score_uses_the_previous_month():
    from core.debentures import credit_score
    nota = credit_score(_measure({("2020-03-01", "11111111"): 101.0}), ROOTS, "level")
    em_abril = nota(pd.Timestamp("2020-04-01"), ["AAAA3", "AAAA4", "ZZZZ3"])
    assert em_abril.to_dict() == {"AAAA3": 101.0, "AAAA4": 101.0}
    assert nota(pd.Timestamp("2020-03-01"), ["AAAA3"]).empty


def test_change_score_needs_both_months():
    from core.debentures import credit_score
    medida = _measure({("2020-01-01", "11111111"): 99.0, ("2020-04-01", "11111111"): 103.0,
                       ("2020-04-01", "22222222"): 100.0})
    nota = credit_score(medida, ROOTS, "change")
    assert nota(pd.Timestamp("2020-05-01"), ["AAAA3", "BBBB3"]).to_dict() == {"AAAA3": 4.0}


def test_unknown_kind_is_refused():
    import pytest
    from core.debentures import credit_score
    with pytest.raises(ValueError):
        credit_score(_measure({("2020-01-01", "11111111"): 99.0}), ROOTS, "spread")


def _ics(n, valor, inicio="2012-01-01"):
    return pd.Series(valor, index=pd.date_range(inicio, periods=n, freq="MS"))


def test_earned_weight_is_zero_before_36_months_and_clipped():
    from core.debentures import earned_weight
    depois = pd.Timestamp("2030-01-01")
    assert earned_weight(_ics(35, 0.01), _ics(35, 0.02), depois) == 0.0
    assert earned_weight(_ics(40, 0.01), _ics(40, 0.02), depois) == 0.5
    assert earned_weight(_ics(40, 0.05), _ics(40, 0.02), depois) == 1.0
    assert earned_weight(_ics(40, -0.01), _ics(40, 0.02), depois) == 0.0
    assert earned_weight(_ics(40, 0.01), _ics(40, 0.02), pd.Timestamp("2014-12-01")) == 0.0   # 35 months before it


BASE5 = pd.Series([0.9, 0.1, 0.5, 0.7, 0.3], index=["AAAA3", "BBBB3", "CCCC3", "DDDD3", "EEEE3"])
EXTRA4 = pd.Series([4.0, 1.0, 3.0, 2.0], index=["AAAA3", "BBBB3", "CCCC3", "DDDD3"])


def test_weighted_score_adds_nothing_for_a_name_without_extra():
    from core.debentures import weighted_score
    from core.views import escore_normal
    base = lambda mes, candidatos: BASE5.reindex(list(candidatos))      # noqa: E731
    extra = lambda mes, candidatos: EXTRA4.reindex(list(candidatos))    # noqa: E731
    mes, nomes = pd.Timestamp("2020-01-01"), list(BASE5.index)
    z_base, z_extra = escore_normal(BASE5), escore_normal(EXTRA4)
    nota = weighted_score(base, extra, lambda _: 0.5)(mes, nomes)
    assert nota["EEEE3"] == z_base["EEEE3"]
    assert nota["AAAA3"] == z_base["AAAA3"] + 0.5 * z_extra["AAAA3"]
    pd.testing.assert_series_equal(weighted_score(base, extra, lambda _: 0.0)(mes, nomes), z_base)


def test_veto_threshold_ties_and_minimum():
    from core.debentures import vetoed_names
    nomes = [f"N{i:02d}3" for i in range(1, 24)]
    m21, m19, m20 = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01")
    painel = pd.DataFrame(index=[m21, m19, m20], columns=nomes, dtype=float)
    painel.loc[m21, nomes[:21]] = range(1, 22)                    # linear 10th percentile = 3.0, two NaN names
    painel.loc[m19, nomes[:19]] = range(1, 20)
    painel.loc[m20, nomes[:20]] = range(1, 21)                    # 10th percentile = 2.9
    vetados = vetoed_names(painel)
    assert vetados[m21] == frozenset(nomes[:3])
    assert vetados[m19] == frozenset()
    assert vetados[m20] == frozenset(nomes[:2])


def test_veto_score_drops_only_vetoed_candidates():
    from core.debentures import veto_score
    base = lambda mes, candidatos: BASE5.reindex(list(candidatos))      # noqa: E731
    jan, fev, mar = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01")
    nota = veto_score(base, {jan: frozenset({"BBBB3", "ZZZZ3"}), fev: frozenset()})
    nomes = list(BASE5.index)
    pd.testing.assert_series_equal(nota(jan, nomes), BASE5.drop("BBBB3"))
    pd.testing.assert_series_equal(nota(fev, nomes), BASE5)
    pd.testing.assert_series_equal(nota(mar, nomes), BASE5)                  # a month without a veto map
