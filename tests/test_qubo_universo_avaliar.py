"""The full-universe QUBO reading's gates and reported lines (scripts/qubo_universo_avaliar.py)."""

import numpy as np
import pandas as pd

from scripts.qubo_universo_avaliar import strategy_gate, years_beating_all


def test_strategy_gate_needs_every_p_and_every_dsr_at_or_above_095_in_every_phase():
    passa = {"p_vs_conservadora": [0.96, 0.97, 0.95], "p_vs_bova": [0.99, 0.98, 0.96], "dsr": [0.95, 0.97, 0.99]}
    assert strategy_gate(**passa)
    for chave in passa:
        falha = {**passa, chave: [*passa[chave][:2], 0.949]}
        assert not strategy_gate(**falha), chave


def _curva(retornos_anuais: dict[int, float]) -> pd.Series:
    """Values whose calendar-year returns are the given ones (first value on the first year's 2 January)."""
    datas, valores, valor = [pd.Timestamp(f"{min(retornos_anuais)}-01-02")], [1.0], 1.0
    for ano, retorno in sorted(retornos_anuais.items()):
        valor *= 1.0 + retorno
        datas.append(pd.Timestamp(f"{ano}-12-30"))
        valores.append(valor)
    return pd.Series(valores, index=datas)


def test_years_beating_all_counts_calendar_years_where_the_arm_beats_every_reference():
    braco = _curva({2020: 0.10, 2021: 0.05, 2022: 0.20})
    bova = _curva({2020: 0.08, 2021: 0.07, 2022: 0.10})
    conservadora = _curva({2020: 0.09, 2021: 0.01, 2022: 0.25})
    assert years_beating_all(braco, [bova, conservadora]) == (1, 3)
    assert years_beating_all(braco, [bova]) == (2, 3)


def test_anneal_status_counts_disagreements_and_reads_the_agreement_gate():
    from core.qubo_universo import AnnealRecord
    from scripts.qubo_universo_avaliar import anneal_status

    log = {pd.Timestamp(f"2020-0{m}-02"): AnnealRecord((0, 1), -1.0, m != 4) for m in range(1, 10)}
    assert anneal_status(log) == ("Annealing (quantum-inspired, solver of record): 9 rebalances, 1 where the two "
                                  "independent runs disagreed (agreement 89% >= 80%: consistent)")
    log.update({pd.Timestamp("2021-01-04"): AnnealRecord((0, 1), -1.0, False)})
    assert anneal_status(log).endswith("(agreement 80% >= 80%: consistent)")
    log.update({pd.Timestamp("2021-04-01"): AnnealRecord((0, 1), -1.0, False)})
    assert anneal_status(log).endswith("(agreement 73% < 80%: NOT consistent)")

def test_picks_profile_reports_candidates_negative_alphas_and_the_fullest_sector():
    from core.qpo import QuboInstance
    from scripts.qubo_universo_avaliar import picks_profile

    def instancia(n: int, negativos: int) -> QuboInstance:
        alpha = np.r_[-np.ones(negativos), np.ones(n - negativos)]
        return QuboInstance(pd.Timestamp("2020-01-02"), tuple(f"N{i}" for i in range(n)), np.eye(n), alpha,
                            np.eye(n), 2)

    setores = {f"N{i}": "AB"[i % 2] for i in range(6)}
    registro = {pd.Timestamp("2020-01-02"): (instancia(4, 1), (0, 2)),
                pd.Timestamp("2020-04-02"): (instancia(6, 3), (0, 1))}
    assert picks_profile(registro, setores) == ("candidates per rebalance: min 4, median 5, max 6; share of negative "
                                                "alpha 38%; most picks in one sector: median 1, max 2")


def test_verdict_needs_the_strategy_gate_and_reproducible_picks():
    from scripts.qubo_universo_avaliar import verdict

    assert verdict(True, 0.80) == "CANDIDATE: pre-register a forward paper log (paper-v3)"
    assert verdict(True, 0.79) == "not a candidate (picks not reproducible): the QPO line is closed"
    assert verdict(False, 1.0) == "not a candidate: the QPO line is closed"


def test_with_retries_returns_after_transient_failures_and_reraises_when_they_persist():
    from core.market_data import ErroDadosMercado
    from scripts.qubo_universo_avaliar import with_retries

    chamadas = []

    def instavel():
        chamadas.append(1)
        if len(chamadas) < 3:
            raise ErroDadosMercado("BCB SGS 12: Expecting value")
        return "cdi"

    assert with_retries(instavel, tries=5, wait=lambda: None) == "cdi"
    assert len(chamadas) == 3

    def sempre_falha():
        raise ErroDadosMercado("BCB fora")

    try:
        with_retries(sempre_falha, tries=3, wait=lambda: None)
    except ErroDadosMercado as erro:
        assert "BCB fora" in str(erro)
    else:
        raise AssertionError("the persistent error was swallowed")
