"""Pieces of ``scripts/avaliar_zona.py`` (the Conservadora with and without the no-trade zone, pre-registered)."""


def test_the_zone_ships_only_with_p_at_least_095_and_a_cagr_not_lower_in_every_phase():
    from scripts.avaliar_zona import decidir_zona
    assert decidir_zona(p_com_zona=[0.96, 0.95, 0.99], cagr_com_zona=[0.17, 0.16, 0.17], cagr_sem_zona=[0.17, 0.15, 0.16])
    assert not decidir_zona([0.96, 0.949, 0.99], [0.17, 0.16, 0.17], [0.16, 0.15, 0.16])     # P below in one phase
    assert not decidir_zona([0.96, 0.95, 0.99], [0.17, 0.149, 0.17], [0.16, 0.15, 0.16])     # CAGR lower in one
