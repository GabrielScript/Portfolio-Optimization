"""Regra do pré-registro do S3 (``scripts/avaliar_views.py``), sem dados do lago."""

from scripts.avaliar_views import candidata


def test_candidata_exige_p_de_0_95_contra_o_bova11_nas_tres_fases():
    assert [candidata([0.95, 0.97, 0.99]), candidata([0.95, 0.94, 0.99])] == [True, False]
