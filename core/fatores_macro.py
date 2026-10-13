"""
fatores_macro.py - Painel mensal de surpresas macro e retrato de hoje (sem Streamlit).

Seis fatores (spec §5). Cada um é o valor de fim do mês t menos o de t-1:

    juros      Δ taxa prefixada de 1 ano (p.p.)
    inflacao   Δ mediana Focus do IPCA 12 meses à frente (p.p.)
    cambio     Δ ln PTAX de venda
    atividade  revisão do PIB 12 meses à frente com ano de referência fixo (p.p.)
    premio     Δ (taxa de 5 anos - taxa de 1 ano) (p.p.)
    mercado    retorno do ^BVSP no mês menos o CDI do mês. A ortogonalização
               aos fatores 1-5 depende da janela de estimação, por isso fica em
               ``estimar_gama`` + ``mercado_residual`` e não no painel.

Índices mensais são ``PeriodIndex`` (freq M, nome "mes"). O painel vai de
2005-12, a primeira variação do ponto de 5 anos, até o último mês completo
antes de ``hoje``. Antes de tirar a diferença, toda série é reindexada para
meses contínuos: um mês sem dado gera NaN em t e em t+1, nunca uma diferença
de dois meses.

Uma fonte que falha tira só os fatores que dependem dela; o motivo vai para
``PainelMacro.fatores_indisponiveis``.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TypeVar

import numpy as np
import pandas as pd

from core import bcb, focus, market_data, tesouro

FATORES = ("juros", "inflacao", "cambio", "atividade", "premio", "mercado")
FATORES_MACRO = FATORES[:5]
INICIO_PAINEL = pd.Period("2005-12", freq="M")
_FONTES_DO_FATOR = {
    "juros": ("tesouro",), "inflacao": ("focus_ipca",), "cambio": ("ptax",),
    "atividade": ("focus_pib",), "premio": ("tesouro",), "mercado": ("bvsp", "cdi"),
}
INICIO_DOWNLOAD = date(2005, 10, 1)      # o nível de 2005-11 entra na primeira variação
TICKER_MERCADO = "^BVSP"
_ERROS_DE_FONTE = (bcb.ErroBcb, focus.ErroFocus, tesouro.ErroTesouro, market_data.ErroDadosMercado)

T = TypeVar("T")


@dataclass(frozen=True)
class DadosMacro:
    """Insumos brutos. None = fonte indisponível, com o motivo em ``erros[fonte]``."""

    titulos: pd.DataFrame | None = None          # tesouro.ler_csv_tesouro
    focus_ipca_12m: pd.DataFrame | None = None   # focus.baixar_focus_ipca_12m
    focus_selic: pd.DataFrame | None = None      # focus.baixar_focus_anual("Selic")
    focus_cambio: pd.DataFrame | None = None     # focus.baixar_focus_anual("Câmbio")
    focus_pib: pd.DataFrame | None = None        # focus.baixar_focus_anual("PIB Total")
    ptax: pd.Series | None = None                # R$/US$, diária
    bvsp_mensal: pd.Series | None = None         # ^BVSP no fim de cada mês completo, PeriodIndex
    cdi_diario: pd.Series | None = None          # decimal ao dia
    selic_meta: pd.Series | None = None          # % a.a., SGS 432, últimos 120 dias
    ipca_12m: pd.Series | None = None            # %, SGS 13522, último ano
    erros: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PainelMacro:
    fatores: pd.DataFrame                  # PeriodIndex "mes" x fatores disponíveis, na ordem de FATORES
    cdi_mensal: pd.Series                  # decimal no mês, PeriodIndex "mes"; vazia sem CDI
    fatores_indisponiveis: dict[str, str]  # fator -> motivo
    avisos: tuple[str, ...]


# ---------------------------------------------------------------------------
# Painel (funções puras)
# ---------------------------------------------------------------------------

def _mensal_continuo(dados: pd.Series | pd.DataFrame, ultimo: pd.Period) -> pd.Series | pd.DataFrame:
    """Reindexa do primeiro mês com dado até ``ultimo``, sem buracos (mês ausente = NaN)."""
    if dados.empty:
        return dados
    return dados.reindex(pd.period_range(dados.index.min(), ultimo, freq="M", name="mes"))


def _fim_de_mes(serie_diaria: pd.Series) -> pd.Series:
    return serie_diaria.groupby(serie_diaria.index.to_period("M")).last().rename_axis("mes")


def cdi_mensal(cdi_diario: pd.Series) -> pd.Series:
    """CDI composto no mês: prod(1 + cdi_d) - 1. Índice: PeriodIndex ``mes``."""
    return ((1.0 + cdi_diario).groupby(cdi_diario.index.to_period("M")).prod() - 1.0).rename_axis("mes")


def revisao_pib_12m(medianas: pd.DataFrame) -> pd.Series:
    """
    Revisão com ano de referência fixo: ``w·ΔF(Y) + (1 - w)·ΔF(Y+1)``, em que Y é
    o ano do mês t, w = ``focus.peso_ano_corrente(t)`` e Δ é a mediana do mês t
    menos a do mês t-1 para a mesma referência. ``medianas``: saída de
    ``focus.mediana_mensal_por_ano`` com meses contínuos. NaN quando falta um
    termo de peso positivo.
    """
    variacao = medianas.diff()
    valores = []
    for mes in variacao.index:
        peso = focus.peso_ano_corrente(mes)
        termos = [(peso, mes.year)] + ([(1.0 - peso, mes.year + 1)] if peso < 1.0 else [])
        valores.append(sum(p * variacao.at[mes, ano] if ano in variacao.columns else np.nan
                           for p, ano in termos))
    return pd.Series(valores, index=variacao.index, dtype=float)


def _calcular_fatores(dados: DadosMacro, ultimo: pd.Period, cdi_m: pd.Series) -> dict[str, pd.Series]:
    calculados: dict[str, pd.Series] = {}
    if dados.titulos is not None:
        curva = _mensal_continuo(tesouro.curva_fim_de_mes(dados.titulos), ultimo)
        calculados["juros"] = curva["juros_1a"].diff()
        calculados["premio"] = (curva["juros_5a"] - curva["juros_1a"]).diff()
    if dados.focus_ipca_12m is not None:
        calculados["inflacao"] = _mensal_continuo(focus.mediana_mensal(dados.focus_ipca_12m), ultimo).diff()
    if dados.ptax is not None:
        calculados["cambio"] = np.log(_mensal_continuo(_fim_de_mes(dados.ptax), ultimo)).diff()
    if dados.focus_pib is not None:
        calculados["atividade"] = revisao_pib_12m(
            _mensal_continuo(focus.mediana_mensal_por_ano(dados.focus_pib), ultimo))
    if dados.bvsp_mensal is not None and not cdi_m.empty:
        retorno = _mensal_continuo(dados.bvsp_mensal, ultimo).pct_change(fill_method=None)
        calculados["mercado"] = retorno - cdi_m.reindex(retorno.index)
    return calculados


def _motivo(fator: str, erros: dict[str, str]) -> str:
    motivos = [f"{fonte}: {erros[fonte]}" for fonte in _FONTES_DO_FATOR[fator] if fonte in erros]
    return "; ".join(motivos) or "fonte sem dados"


def montar_painel(dados: DadosMacro, hoje: date) -> PainelMacro:
    """Painel de 2005-12 até o último mês completo antes de ``hoje``."""
    ultimo = pd.Period(hoje, freq="M") - 1
    meses = pd.period_range(INICIO_PAINEL, ultimo, freq="M", name="mes")
    cdi_m = (cdi_mensal(dados.cdi_diario).loc[:ultimo] if dados.cdi_diario is not None
             else pd.Series(dtype=float, index=pd.PeriodIndex([], freq="M", name="mes")))
    calculados = _calcular_fatores(dados, ultimo, cdi_m)
    indisponiveis = {f: _motivo(f, dados.erros) for f in FATORES if f not in calculados}
    fatores = pd.DataFrame({f: calculados[f].reindex(meses) for f in FATORES if f in calculados},
                           index=meses)
    avisos = tuple(f"Fator '{f}' indisponível ({motivo})." for f, motivo in indisponiveis.items())
    return PainelMacro(fatores, cdi_m, indisponiveis, avisos)


def estimar_gama(fatores: pd.DataFrame) -> pd.Series:
    """
    MQO de ``mercado`` em constante + fatores macro presentes, nos meses com
    todas as colunas preenchidas. Índice do resultado: "const" e os fatores.
    Quem chama escolhe a janela (o modelo usa a mesma janela das regressões).

    Levanta ValueError sem a coluna ``mercado`` ou com menos meses completos
    que coeficientes + 2.
    """
    if "mercado" not in fatores.columns:
        raise ValueError("Painel sem o fator 'mercado': não há o que ortogonalizar.")
    macro = [f for f in FATORES_MACRO if f in fatores.columns]
    completos = fatores[macro + ["mercado"]].dropna()
    if len(completos) < len(macro) + 3:
        raise ValueError(f"{len(completos)} meses completos para {len(macro) + 1} coeficientes.")
    regressores = np.column_stack([np.ones(len(completos)), completos[macro].to_numpy()])
    coeficientes, *_ = np.linalg.lstsq(regressores, completos["mercado"].to_numpy(), rcond=None)
    return pd.Series(coeficientes, index=["const", *macro])


def mercado_residual(fatores: pd.DataFrame, gama: pd.Series) -> pd.Series:
    """
    Fator 6: ``mercado - (const + Σ γ_k·f_k)``. ``gama`` pode vir de outra janela
    (na validação, γ estimado até t-1 e aplicado em t). NaN onde faltar algum termo.
    """
    macro = [nome for nome in gama.index if nome != "const"]
    explicado = gama["const"] + fatores[macro].mul(gama[macro], axis=1).sum(axis=1, min_count=len(macro))
    return (fatores["mercado"] - explicado).rename("mercado")


# ---------------------------------------------------------------------------
# Download (único trecho com rede)
# ---------------------------------------------------------------------------

def _tentar(obter: Callable[[], T]) -> tuple[T | None, str | None]:
    """(valor, None) ou (None, motivo) para erros de fonte. Qualquer outra exceção é bug e sobe."""
    try:
        return obter(), None
    except _ERROS_DE_FONTE as erro:
        return None, str(erro)


def _serie_bcb(codigo: int, inicio: date, hoje: date) -> pd.Series:
    serie = bcb.baixar_serie(codigo, inicio, hoje)
    if serie.empty:
        raise bcb.ErroBcb(f"BCB SGS {codigo} sem observações entre {inicio} e {hoje}.")
    return serie


def _bvsp_mensal(hoje: date) -> pd.Series:
    precos = market_data.carregar_precos_mensais([TICKER_MERCADO], INICIO_DOWNLOAD, hoje)
    serie = precos[TICKER_MERCADO].dropna()
    if serie.empty:
        raise market_data.ErroDadosMercado(f"Yahoo sem preços mensais de {TICKER_MERCADO}.")
    return serie


def _tarefas_de_download(hoje: date) -> dict[str, tuple[str, Callable[[], object]]]:
    """Campo de DadosMacro -> (chave da fonte em ``erros``, função que baixa)."""
    return {
        "titulos": ("tesouro", lambda: tesouro.baixar_prefixados(hoje)),
        "focus_ipca_12m": ("focus_ipca", lambda: focus.baixar_focus_ipca_12m(hoje)),
        "focus_selic": ("focus_selic", lambda: focus.baixar_focus_anual(focus.INDICADOR_SELIC, hoje)),
        "focus_cambio": ("focus_cambio", lambda: focus.baixar_focus_anual(focus.INDICADOR_CAMBIO, hoje)),
        "focus_pib": ("focus_pib", lambda: focus.baixar_focus_anual(focus.INDICADOR_PIB, hoje)),
        "ptax": ("ptax", lambda: bcb.baixar_ptax(INICIO_DOWNLOAD, hoje)),
        "bvsp_mensal": ("bvsp", lambda: _bvsp_mensal(hoje)),
        "cdi_diario": ("cdi", lambda: _serie_bcb(bcb.SERIE_CDI, INICIO_DOWNLOAD, hoje) / 100.0),
        "selic_meta": ("selic_meta",
                       lambda: _serie_bcb(bcb.SERIE_SELIC_META, hoje - timedelta(days=120), hoje)),
        "ipca_12m": ("ipca_12m",
                     lambda: _serie_bcb(bcb.SERIE_IPCA_12M, hoje - timedelta(days=400), hoje)),
    }


def baixar_dados_macro(hoje: date) -> DadosMacro:
    """
    Baixa todas as fontes em paralelo (são independentes; em série levava ~160 s
    porque o SGS responde devagar). Fonte que falha vira None + motivo em
    ``erros``; erro que não é de fonte (bug) sobe.
    """
    tarefas = _tarefas_de_download(hoje)
    with ThreadPoolExecutor(max_workers=len(tarefas)) as executor:
        futuros = {campo: executor.submit(_tentar, obter) for campo, (_, obter) in tarefas.items()}
    valores: dict[str, object] = {}
    erros: dict[str, str] = {}
    for campo, futuro in futuros.items():
        valor, motivo = futuro.result()
        valores[campo] = valor
        if motivo is not None:
            erros[tarefas[campo][0]] = motivo
    return DadosMacro(**valores, erros=erros)
