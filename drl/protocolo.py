"""The walk-forward the runs are locked into. It is written here, in code, before any test number is looked at, and a
change to it is a new attempt (N increments in ``DECISOES.md``) — the whole point of a DRL grid is that its search space
is huge enough to manufacture a backtest if the splits are chosen after the fact.

Each window is P1's protocol on our tape: five sessions-years of training, one year of validation to pick the seed and
stop training, one year untouched as the test. The windows slide one year at a time."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ANO_TREINO, ANO_VALIDACAO = 5, 1
PRIMEIRO_TESTE, ULTIMO_TESTE = 2012, 2026


@dataclass(frozen=True)
class Janela:
    """The row range of one walk-forward window (positions in the session index, not dates)."""
    treino: slice
    validacao: slice
    teste: slice
    ano_teste: int


def janelas(sessoes: pd.DatetimeIndex, primeiro_teste: int = PRIMEIRO_TESTE,
            ultimo_teste: int = ULTIMO_TESTE) -> list[Janela]:
    """One window per test year, each with the five years before it as training, the year between as validation and
    the test year itself untouched. A year missing from the tape (or too short to hold 60 sessions) skips its window
    instead of shortening the others."""
    anos = sessoes.year
    saida: list[Janela] = []
    for ano in range(primeiro_teste, ultimo_teste + 1):
        inicio_treino, inicio_validacao, inicio_teste, fim_teste = ano - ANO_TREINO - ANO_VALIDACAO, ano - 1, ano, ano + 1
        treino = (anos >= inicio_treino) & (anos < inicio_validacao)
        validacao = (anos >= inicio_validacao) & (anos < inicio_teste)
        teste = (anos >= inicio_teste) & (anos < fim_teste)
        if not (treino.sum() > 200 and validacao.sum() > 60 and teste.sum() > 60):
            continue
        saida.append(Janela(_fatia(treino), _fatia(validacao), _fatia(teste), ano))
    return saida


def _fatia(mascara) -> slice:
    """The first and last+1 row of a boolean mask, as a slice of the session index."""
    posicoes = mascara.nonzero()[0]
    return slice(int(posicoes[0]), int(posicoes[-1]) + 1)


def eras(sessoes: pd.DatetimeIndex) -> dict[str, slice]:
    """The three eras the report is split by: the thin early tape, the decade before the universe widened, and the
    wide tape the agents are tested on. Fixed here so the split cannot be chosen to flatter a result."""
    anos = sessoes.year
    return {"2005-2012": _fatia(anos <= 2012), "2013-2019": _fatia((anos >= 2013) & (anos <= 2019)),
            "2020-2026": _fatia(anos >= 2020)}
