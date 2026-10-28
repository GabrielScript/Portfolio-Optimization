"""
crescimento_lucro.py - The profit-growth signal of the pre-registered IC test (DECISOES 157,
``docs/superpowers/specs/2026-10-27-crescimento-lucro-design.md``): the year-on-year change of net income and of gross
profit over the total assets of a year before (the JKP ``niq_at_chg1`` family; gross profitability's change, Novy-Marx
2013), point in time from the CVM statements (12-month flows, each value dated by its receipt at the CVM).
"""

from __future__ import annotations

import pandas as pd

from core.risco import emissor
from core.valor import NotaDoMes
from core.views import nota_media_dos_escores


def variacao_anual(fundamentos: pd.DataFrame, conta: str, mes: pd.Timestamp) -> pd.Series:
    """Issuer -> (``conta``(t) - ``conta``(t - 1 year)) / total assets(t - 1 year), where t is the latest period end
    of ``conta`` received before ``mes`` and t - 1 year the same quarter end a year earlier, also received before
    ``mes``. Out: no year-ago filing, total assets <= 0 a year before."""
    conhecidos = fundamentos[fundamentos["available_from"] < mes]
    valores = conhecidos.pivot_table(index=["issuer", "dt_refer"], columns="account", values="value", aggfunc="last")
    if conta not in valores.columns or "total_assets" not in valores.columns:
        return pd.Series(dtype=float)
    atuais = valores[conta].dropna().reset_index().sort_values("dt_refer").drop_duplicates("issuer", keep="last")
    anteriores = valores.reindex(pd.MultiIndex.from_arrays(
        [atuais["issuer"], atuais["dt_refer"] - pd.DateOffset(years=1)]))
    ativo = anteriores["total_assets"].to_numpy()
    positivo = ativo > 0
    variacao = pd.Series(atuais[conta].to_numpy() - anteriores[conta].to_numpy(), index=atuais["issuer"].to_numpy())
    variacao = variacao[positivo] / ativo[positivo]
    return variacao.dropna()


def nota_da_variacao(fundamentos: pd.DataFrame, conta: str) -> NotaDoMes:
    """``variacao_anual`` of ``conta`` on each ticker of the month's candidates (an issuer's classes share it)."""
    def _nota(mes: pd.Timestamp, candidatos) -> pd.Series:
        variacao = variacao_anual(fundamentos, conta, mes)
        papeis = [papel for papel in candidatos if emissor(papel) in variacao.index]
        return pd.Series([variacao[emissor(papel)] for papel in papeis], index=papeis, dtype=float)
    return _nota


def nota_de_crescimento(fundamentos: pd.DataFrame) -> NotaDoMes:
    """The pre-registered score: the mean of the normal scores of the net-income and gross-profit changes, with the
    ones available (``core.views.nota_media_dos_escores``, the S3 rule for value)."""
    return nota_media_dos_escores(nota_da_variacao(fundamentos, "net_income"),
                                  nota_da_variacao(fundamentos, "gross_profit"))
