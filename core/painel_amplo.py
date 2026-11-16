"""
painel_amplo.py - The 23 characteristics added to the 213 six in spec
``docs/superpowers/specs/2026-11-16-painel-amplo-design.md`` (section 3). Every value of month m reads only data known
before m: monthly returns before m, daily closes before m's first day, documents received before m (at most 18 months
old) and the decision month's market value (close of m-1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.qml_painel import angulos
from core.risco import emissor
from core.sinais import _meses_antes
from core.valor import Balancos, fundamentos_no_mes, valor_da_firma

NOVAS = ("momentum_6_1", "reversao_curta", "reversao_longa", "beta", "vol_idiossincratica", "max_diario",
         "vol_diaria", "tamanho", "lucro_preco", "valor_patrimonial", "receita_preco", "caixa_preco", "ebitda_ev",
         "roe", "roa", "lucratividade_bruta", "margem_operacional", "accruals", "alavancagem", "caixa_ativos",
         "crescimento_ativos", "crescimento_receita", "emissao_liquida")
MESES_DO_BETA, MINIMO_DO_BETA = 36, 24
SESSOES_DO_MAXIMO, SESSOES_DA_VOL, MINIMO_DA_VOL_DIARIA = 21, 63, 40


@dataclass(frozen=True)
class Dados:
    retornos: pd.DataFrame          # monthly total returns, month x security
    fechamentos: pd.DataFrame       # daily closes, date x security
    balancos: Balancos


def _acumulado(janela: pd.DataFrame) -> pd.Series:
    return (1.0 + janela).prod(min_count=len(janela)).sub(1.0).where(janela.notna().all())


def _precos(retornos: pd.DataFrame, mes: pd.Timestamp, papeis: Sequence[str]) -> pd.DataFrame:
    medidas = {"momentum_6_1": _acumulado(_meses_antes(retornos[papeis], mes, 5, pulando=1)),
               "reversao_curta": -_meses_antes(retornos[papeis], mes, 1).iloc[0],
               "reversao_longa": -_acumulado(_meses_antes(retornos[papeis], mes, 24, pulando=12))}
    janela = _meses_antes(retornos[papeis], mes, MESES_DO_BETA)
    mercado = janela.mean(axis=1)
    beta, idio = {}, {}
    for papel in papeis:
        serie = janela[papel]
        validos = serie.notna() & mercado.notna()
        if validos.sum() < MINIMO_DO_BETA:
            continue
        x = np.column_stack([np.ones(validos.sum()), mercado[validos].to_numpy()])
        y = serie[validos].to_numpy()
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        beta[papel] = coef[1]
        idio[papel] = -float(np.std(y - x @ coef, ddof=2))
    medidas["beta"], medidas["vol_idiossincratica"] = pd.Series(beta, dtype=float), pd.Series(idio, dtype=float)
    return pd.DataFrame(medidas).reindex(papeis)


def _diarios(fechamentos: pd.DataFrame, mes: pd.Timestamp, papeis: Sequence[str]) -> pd.DataFrame:
    antes = fechamentos.loc[fechamentos.index < mes, [p for p in papeis if p in fechamentos.columns]]
    retornos = antes.pct_change(fill_method=None)
    ultimos = retornos.iloc[-SESSOES_DO_MAXIMO:]
    vol = retornos.iloc[-SESSOES_DA_VOL:]
    return pd.DataFrame({"max_diario": -ultimos.max().where(ultimos.notna().sum() >= SESSOES_DO_MAXIMO // 2),
                         "vol_diaria": -vol.std().where(vol.notna().sum() >= MINIMO_DA_VOL_DIARIA)}).reindex(papeis)


def _razao(numerador: pd.Series, denominador: pd.Series) -> pd.Series:
    return (numerador / denominador).where(denominador > 0)


def _conta(tabela: pd.DataFrame, nome: str, zero: bool = False) -> pd.Series:
    valores = tabela[nome] if nome in tabela.columns else pd.Series(np.nan, index=tabela.index, dtype=float)
    return valores.fillna(0.0) if zero else valores


def _fundamentos(balancos: Balancos, mes: pd.Timestamp, papeis: Sequence[str]) -> pd.DataFrame:
    agora = fundamentos_no_mes(balancos.fundamentos, mes)
    antes = fundamentos_no_mes(balancos.fundamentos, mes - pd.DateOffset(months=12))
    caps = balancos.valor_mercado
    datas = pd.to_datetime(caps["month"])
    mercado = caps[datas == mes].set_index("issuer")["market_cap"]
    acoes = caps[datas == mes].set_index("issuer")["shares_today_units"]
    acoes_antes = caps[datas == mes - pd.DateOffset(months=12)].set_index("issuer")["shares_today_units"]
    emissores = agora.index
    mercado = mercado.reindex(emissores)
    lucro = _conta(agora, "net_income_parent").fillna(_conta(agora, "net_income"))
    ativos = _conta(agora, "total_assets")
    patrimonio = _conta(agora, "equity")
    ebitda = _conta(agora, "ebit") + _conta(agora, "depreciation", True).abs()
    ev = valor_da_firma(agora, mercado)
    divida = _conta(agora, "short_term_debt", True) + _conta(agora, "long_term_debt", True)
    caixa = _conta(agora, "cash", True) + _conta(agora, "short_term_investments", True)
    por_emissor = pd.DataFrame({
        "tamanho": -np.log(mercado.where(mercado > 0)),
        "lucro_preco": _razao(lucro, mercado),
        "valor_patrimonial": _razao(patrimonio - _conta(agora, "minority_interest", True), mercado),
        "receita_preco": _razao(_conta(agora, "revenue"), mercado),
        "caixa_preco": _razao(_conta(agora, "operating_cash_flow"), mercado),
        "ebitda_ev": _razao(ebitda, ev),
        "roe": _razao(lucro, patrimonio),
        "roa": _razao(lucro, ativos),
        "lucratividade_bruta": _razao(_conta(agora, "gross_profit"), ativos),
        "margem_operacional": _razao(_conta(agora, "ebit"), _conta(agora, "revenue")),
        "accruals": -_razao(lucro - _conta(agora, "operating_cash_flow"), ativos),
        "alavancagem": -_razao(divida, ativos),
        "caixa_ativos": _razao(caixa, ativos),
        "crescimento_ativos": -(_razao(ativos, _conta(antes, "total_assets").reindex(emissores)) - 1.0),
        "crescimento_receita": _razao(_conta(agora, "revenue"), _conta(antes, "revenue").reindex(emissores)) - 1.0,
        "emissao_liquida": -(_razao(acoes.reindex(emissores), acoes_antes.reindex(emissores)) - 1.0),
    }, index=emissores)
    return por_emissor.reindex([emissor(p) for p in papeis]).set_axis(list(papeis))


def caracteristicas(dados: Dados, mes: pd.Timestamp, papeis: Sequence[str]) -> pd.DataFrame:
    """Raw values of the ``NOVAS`` for ``papeis`` in month ``mes`` (NaN where not available)."""
    com_retorno = [p for p in papeis if p in dados.retornos.columns]
    tabela = pd.concat([_precos(dados.retornos, mes, com_retorno), _diarios(dados.fechamentos, mes, com_retorno),
                        _fundamentos(dados.balancos, mes, com_retorno)], axis=1)
    return tabela.reindex(list(papeis))[list(NOVAS)]


def angulos_do_mes(dados: Dados, mes: pd.Timestamp, papeis: Sequence[str]) -> pd.DataFrame:
    """The 213 angle of each new characteristic among ``papeis`` (missing = 0), columns in ``NOVAS`` order."""
    brutas = caracteristicas(dados, mes, papeis).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame({nome: angulos(brutas[nome]) for nome in NOVAS})
