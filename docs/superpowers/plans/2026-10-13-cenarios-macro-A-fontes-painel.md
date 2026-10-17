# Cenários macro — Plano A (fontes + painel de fatores) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Skills exigidas pelo usuário (carregar antes de codar):** `rigorous-coding`, `coding-principles`,
> `python-patterns`, `python-design-patterns`, `code-reviewer` (revisão de cada tarefa, formato 🔴/🟠)
> e `developing-with-streamlit` (só nos planos com interface; este plano não toca em Streamlit).

**Goal:** Baixar Focus, Tesouro Direto, SGS e ^BVSP; montar o painel mensal dos 6 fatores de surpresa
macro (2005-12 em diante) e o retrato macro de hoje, tudo em `core/`, sem Streamlit e sem rede nos testes.

**Architecture:** Duas fontes novas (`core/focus.py`, `core/tesouro.py`) com erro próprio e parsing
validado. Uma função nova de preços mensais em `core/market_data.py`. Um módulo `core/fatores_macro.py`
que separa I/O (`baixar_dados_macro`, em paralelo, com isolamento de falha por fonte) de funções
puras (`montar_painel`, `estimar_gama`/`mercado_residual`, `montar_retrato`).

**Tech Stack:** Python 3.12, pandas 3.0.6, numpy 2.5.4, requests, yfinance (já instalados no `.venv`), pytest 9.1.

**Spec:** `docs/superpowers/specs/2026-10-12-cenarios-macro-design.md` (aprovado em 2026-10-13). Este é o
**Plano A de 3**:
- **A** (este): §3 fontes e fatores, §4, §5, a parte de fontes da §11, o item 1 da §12, o retrato da §10.1 (só o core) e o script manual da §12.
- **B:** modelo de fatores (§6), cenários (§7), piso no otimizador (§8) e `projetar_horizonte` (§15).
- **C:** validação (§9), loaders e cache em `ui/servicos.py`, aba "Cenários macro", seletor de horizonte e `requirements.txt`.

B e C serão escritos depois que A rodar com dados reais, porque dependem dos formatos que A fixa.

## Global Constraints

- Diretório: todo comando do plano roda a partir de
  `C:\Users\gabri\Desktop\AI Hedge Fund Brazil\ai-hedge-fund\Portfolio-Optimization`, nunca do diretório
  pai, que é outro repositório git.
- Ambiente: sempre o `.venv` do projeto (o Python global está quebrado: numpy 1.26 com cvxpy 1.9).
  Rodar com `.venv/Scripts/python -m pytest tests -q`.
- `core/` não importa Streamlit e não ganha dependência nova (requests, pandas, numpy e yfinance já existem).
- `hoje` é sempre argumento explícito. Nada em `core/` chama `date.today()`; o script manual pode chamar.
- Nenhum teste acessa a rede. Os dublês seguem o padrão de `tests/test_bcb.py`: `monkeypatch` em
  `<módulo>.requests.get` ou nas funções de download.
- Índices mensais são `pd.PeriodIndex` com `freq="M"` e nome `"mes"`.
- Unidades:
  - taxas e medianas Focus em % (como a fonte publica);
  - fatores `juros`, `inflacao`, `atividade` e `premio` em p.p.;
  - `cambio` em diferença de log;
  - `mercado` e CDI mensal em decimal.
- Cada fonte tem erro próprio (`ErroFocus`, `ErroTesouro`, `bcb.ErroBcb`, `market_data.ErroDadosMercado`).
  O erro sempre encadeia a causa com `from`. Erro que não é de fonte (bug) nunca é engolido.
- Nomes de código e mensagens em português, como no resto do pacote.
- Git:
  - trabalhar no branch `feat/cenarios-macro`; **nunca dar push na `main`** (o Streamlit Cloud faz deploy dela e o `requirements.txt` está velho);
  - toda mensagem de commit termina com a linha `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Windows: prints com acento ou λ precisam de `PYTHONIOENCODING=utf-8`.

## Decisões que este plano fixa (o spec não as deixava explícitas)

Formatos verificados ao vivo em 2026-10-13:

1. **Tesouro, taxa zero:** `Taxa Compra Manha` ≤ 0 significa "título fora de venda". Aconteceu com a
   NTN-F 2021 em jan–mar/2010, cuja `Taxa Venda Manha` de 0,06 também é lixo. Essas linhas são
   descartadas.
2. **Tesouro, mesmo vencimento:** quando LTN e NTN-F vencem na mesma data, fica a LTN, que é zero-cupom.
3. **Ponto de 5 anos:** "Sem extrapolação" em 5 anos equivale ao critério "título com prazo ≥ 4,5a" do
   spec. Nenhum mês tem prazo máximo entre 4,5 e 5 anos: até 2005-10 o máximo é 4,43 anos, e de
   2005-11 em diante passa de 5,9 anos.
4. **Retrato com 8 indicadores:**
   - Selic meta;
   - Selic 12m Focus;
   - IPCA acumulado em 12 meses (SGS 13522, citada na §3/§4);
   - IPCA 12m Focus;
   - PTAX;
   - câmbio 12m Focus (o cenário da §7.1 usa esta mediana como base);
   - PIB 12m Focus;
   - inclinação 5a−1a.

   Cada indicador traz `dias_maximos`: 7 para fontes diárias e 75 para o IPCA mensal. A interface
   avisa quando a idade passa disso.
5. **Download paralelo:** as fontes são baixadas em paralelo. Em série, o download levou 157 s (o SGS
   é lento e uma consulta deu timeout); em paralelo, 12 s.
6. **Fator 6:** a ortogonalização (`estimar_gama`, `mercado_residual`) fica em `fatores_macro`, mas
   quem escolhe a janela é o modelo (Plano B). O painel guarda o `mercado` bruto (^BVSP − CDI).
7. **Mês corrente:** fica fora de tudo que é mensal. O último mês do painel é `Period(hoje) − 1`.

## Review Focus

As cinco situações que o spec implica e que mais podem morder quem usa o app. Cada uma tem teste na
tarefa dona do código.

1. **O Olinda responde 200 com página HTML de manutenção.** Esperado: `ErroFocus` e o fator vira
   indisponível, sem traceback. (Tarefa 1: `test_resposta_que_nao_e_json_levanta_erro`.)
2. **O Tesouro renomeia uma coluna do CSV.** Esperado: `ErroTesouro` com o nome da coluna, e não um
   `KeyError` solto. (Tarefa 2: `test_coluna_renomeada_levanta_erro_com_o_nome`.)
3. **Um mês sem cotação ou sem título que cubra o prazo de 1 ano.** Esperado: NaN no mês e no seguinte,
   nunca uma "variação" de dois meses. (Tarefa 4: `test_juros_e_premio_sao_diferencas_de_fim_de_mes_e_mes_sem_cotacao_vira_nan`.)
4. **Mês corrente incompleto.** Esperado: fica fora dos preços mensais, do CDI mensal e do painel.
   (Tarefas 3 e 4: `test_ultimo_preco_de_cada_mes_completo`, `test_mercado_e_retorno_do_ibovespa_menos_cdi_composto`.)
5. **Bug dentro de uma fonte (ex.: `KeyError`).** Esperado: a exceção sobe e não vira "fonte
   indisponível" em silêncio. (Tarefa 6: `test_baixar_dados_macro_nao_engole_bug`.)

## File Structure

```
core/focus.py            (novo)  Focus OData: anuais e IPCA 12m; Expectativa12m; ponderação 12m; medianas mensais
core/tesouro.py          (novo)  CSV do Tesouro -> prefixados limpos; interpolação; curva de fim de mês e de hoje
core/market_data.py      (muda)  + carregar_precos_mensais
core/bcb.py              (muda)  + SERIE_IPCA_12M = 13522
core/fatores_macro.py    (novo)  DadosMacro, PainelMacro, montar_painel, estimar_gama, mercado_residual,
                                 baixar_dados_macro (paralelo), RetratoMacro, montar_retrato
scripts/conferir_painel_macro.py (novo) conferência manual com rede
tests/test_focus.py, tests/test_tesouro.py, tests/test_precos_mensais.py, tests/test_fatores_macro.py (novos)
```

---

### Task 0: Pré-requisitos e branch

**Files:** nenhum.

- [ ] **Step 1: Entrar no repositório certo e conferir que os 3 commits do usuário já foram feitos**

O diretório pai (`ai-hedge-fund`) também é um repositório git e enxerga `Portfolio-Optimization/` como
untracked. Rodar o `git switch` de lá criaria o branch no repositório errado.

Run:
```powershell
cd "C:\Users\gabri\Desktop\AI Hedge Fund Brazil\ai-hedge-fund\Portfolio-Optimization"
git rev-parse --show-toplevel   # esperado: termina em /Portfolio-Optimization
git log -4 --oneline
git status --short
```
Expected:
- o topo NÃO é `0d0e613`;
- há 3 commits novos acima de `0d0e613` (o WIP original, a revisão e o spec; as mensagens podem ter sido reescritas);
- o `git status --short` mostra só `?? docs/superpowers/plans/` (este plano).

Se o topo ainda for `0d0e613`, PARE e peça ao usuário para rodar os comandos da "Sessão 2026-10-12 (parte 3)" do `session_state.md`.

- [ ] **Step 2: Criar o branch**

```powershell
git switch -c feat/cenarios-macro
```

- [ ] **Step 3: Linha de base verde**

Run: `.venv/Scripts/python -m pytest tests -q`
Expected: `48 passed`

---

### Task 1: Fonte Focus (`core/focus.py`)

**Files:**
- Create: `core/focus.py`
- Test: `tests/test_focus.py`

**Interfaces:**
- Consumes: nada de tarefas anteriores.
- Produces:
  - `ErroFocus(RuntimeError)`
  - `Expectativa12m(NamedTuple)`: `mediana: float`, `desvio: float` (NaN se não houver) e `data: pd.Timestamp`
  - `INDICADOR_SELIC = "Selic"`, `INDICADOR_CAMBIO = "Câmbio"`, `INDICADOR_PIB = "PIB Total"`, `ENDPOINT_ANUAIS`, `ENDPOINT_IPCA_12M`
  - `baixar_focus_anual(indicador: str, hoje: date) -> pd.DataFrame`: colunas `data, ano_referencia(int), mediana, desvio`
  - `baixar_focus_ipca_12m(hoje: date) -> pd.DataFrame`: colunas `data, mediana, desvio`
  - `peso_ano_corrente(data: pd.Timestamp | pd.Period) -> float`, que vale `(13 - mês)/12`
  - `expectativa_12m(anual: pd.DataFrame, ate: pd.Timestamp) -> Expectativa12m`
  - `mediana_mensal_por_ano(anual) -> pd.DataFrame`: PeriodIndex `mes` × anos de referência
  - `mediana_mensal(serie_12m) -> pd.Series`: PeriodIndex `mes`

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_focus.py`:

```python
"""
Testes de core/focus.py (sem rede).

``FocusFalso`` imita o OData do Olinda (sondado ao vivo em 2026-10-12/13):
HTTP 400 se os parâmetros vierem por ``params`` ou se o ``$filter`` não estiver
codificado; resposta ``{"value": [...]}`` com só os campos do ``$select``.
"""

import re
from datetime import date
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd
import pytest
import requests

from core import focus

HOJE = date(2026, 10, 13)


class RespostaFalsa:
    def __init__(self, status_code: int, corpo=None):
        self.status_code = status_code
        self._corpo = corpo

    def json(self):
        if not isinstance(self._corpo, (dict, list)):
            raise requests.JSONDecodeError("corpo não é JSON", "", 0)
        return self._corpo

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error", response=self)


def _linha_anual(indicador: str, data: str, ano: int, mediana: float, desvio: float | None) -> dict:
    return {"Indicador": indicador, "IndicadorDetalhe": None, "Data": data, "DataReferencia": str(ano),
            "Media": mediana, "Mediana": mediana, "DesvioPadrao": desvio, "Minimo": None,
            "Maximo": None, "numeroRespondentes": None, "baseCalculo": 0}


def _linha_ipca(data: str, mediana: float, desvio: float, suavizada: str = "S") -> dict:
    return {"Indicador": "IPCA", "Data": data, "Suavizada": suavizada, "Media": mediana,
            "Mediana": mediana, "DesvioPadrao": desvio, "Minimo": None, "Maximo": None,
            "numeroRespondentes": None, "baseCalculo": 0}


class FocusFalso:
    """Servidor OData em memória; ``linhas[endpoint]`` são os registros completos."""

    def __init__(self, linhas: dict[str, list[dict]]):
        self.linhas = linhas
        self.urls: list[str] = []

    def __call__(self, url, params=None, timeout=None):
        self.urls.append(url)
        if params is not None:
            return RespostaFalsa(400, {"error": "params não suportado"})
        partes = urlsplit(url)
        consulta = dict(parte.split("=", 1) for parte in partes.query.split("&"))
        if re.search(r"[ ']", consulta.get("$filter", "")):
            return RespostaFalsa(400, {"error": "filtro sem codificação"})
        filtro = unquote(consulta["$filter"])
        if "baseCalculo eq 0" not in filtro:
            return RespostaFalsa(200, {"value": []})
        indicador = re.search(r"Indicador eq '([^']+)'", filtro).group(1)
        suavizada = re.search(r"Suavizada eq '([^']+)'", filtro)
        campos = consulta["$select"].split(",")
        endpoint = partes.path.rsplit("/", 1)[-1]
        valor = [{c: r[c] for c in campos if c in r}
                 for r in self.linhas.get(endpoint, [])
                 if r["Indicador"] == indicador
                 and (suavizada is None or r.get("Suavizada") == suavizada.group(1))]
        return RespostaFalsa(200, {"value": valor})


@pytest.fixture
def servidor(monkeypatch):
    falso = FocusFalso({
        focus.ENDPOINT_ANUAIS: [
            _linha_anual("Câmbio", "2026-10-09", 2026, 5.1535, 0.1612),
            _linha_anual("Câmbio", "2026-10-09", 2027, 5.26, 0.204),
            _linha_anual("Câmbio", "2026-10-14", 2026, 9.99, 0.1),     # depois de HOJE
            _linha_anual("Selic", "2026-10-09", 2027, 12.00, 0.7543),
            _linha_anual("Selic", "2026-10-09", 2026, 13.25, 0.2236),
            _linha_anual("PIB Total", "1999-07-01", 1999, -1.0, None),
        ],
        focus.ENDPOINT_IPCA_12M: [
            _linha_ipca("2026-10-08", 4.5221, 0.4855),
            _linha_ipca("2026-10-09", 4.4772, 0.495),
            _linha_ipca("2026-10-09", 9.99, 9.99, suavizada="N"),
        ],
    })
    monkeypatch.setattr(focus.requests, "get", falso)
    return falso


def test_filtro_vai_codificado_na_url_e_nao_por_params(servidor):
    """O FocusFalso devolve 400 para ``params`` ou filtro cru, como o Olinda real."""
    cambio = focus.baixar_focus_anual(focus.INDICADOR_CAMBIO, HOJE)
    assert len(cambio) == 2
    assert "%C3%A2" in servidor.urls[0]          # "â" de Câmbio, em UTF-8
    assert " " not in servidor.urls[0]


def test_anual_converte_tipos_ordena_e_corta_pesquisas_futuras(servidor):
    tabela = focus.baixar_focus_anual(focus.INDICADOR_SELIC, HOJE)
    assert list(tabela.columns) == ["data", "ano_referencia", "mediana", "desvio"]
    assert list(tabela["ano_referencia"]) == [2026, 2027]
    assert tabela["ano_referencia"].dtype.kind == "i"
    assert tabela["mediana"].tolist() == [13.25, 12.00]

    cambio = focus.baixar_focus_anual(focus.INDICADOR_CAMBIO, HOJE)
    assert cambio["data"].max() == pd.Timestamp("2026-10-09")


def test_desvio_nulo_das_pesquisas_antigas_vira_nan(servidor):
    pib = focus.baixar_focus_anual(focus.INDICADOR_PIB, HOJE)
    assert len(pib) == 1 and np.isnan(pib["desvio"].iloc[0])


def test_ipca_12m_usa_so_a_serie_suavizada(servidor):
    ipca = focus.baixar_focus_ipca_12m(HOJE)
    assert ipca["mediana"].tolist() == [4.5221, 4.4772]


def test_resposta_que_nao_e_json_levanta_erro(monkeypatch):
    monkeypatch.setattr(focus.requests, "get", lambda *a, **k: RespostaFalsa(200, "<html>manutenção</html>"))
    with pytest.raises(focus.ErroFocus):
        focus.baixar_focus_ipca_12m(HOJE)


def test_http_500_levanta_erro(monkeypatch):
    monkeypatch.setattr(focus.requests, "get", lambda *a, **k: RespostaFalsa(500, {}))
    with pytest.raises(focus.ErroFocus):
        focus.baixar_focus_anual(focus.INDICADOR_PIB, HOJE)


def test_campo_faltando_levanta_erro_com_o_nome_do_campo(monkeypatch):
    linha = _linha_ipca("2026-10-09", 4.4772, 0.495)
    del linha["DesvioPadrao"]
    monkeypatch.setattr(focus.requests, "get", FocusFalso({focus.ENDPOINT_IPCA_12M: [linha]}))
    with pytest.raises(focus.ErroFocus, match="DesvioPadrao"):
        focus.baixar_focus_ipca_12m(HOJE)


def test_sem_pesquisas_ate_hoje_levanta_erro(servidor):
    with pytest.raises(focus.ErroFocus):
        focus.baixar_focus_ipca_12m(date(2001, 1, 1))


@pytest.mark.parametrize("data, peso", [("2026-01-31", 1.0), ("2026-10-09", 0.25), ("2026-12-01", 1 / 12)])
def test_peso_do_ano_corrente(data, peso):
    assert focus.peso_ano_corrente(pd.Timestamp(data)) == pytest.approx(peso)
    assert focus.peso_ano_corrente(pd.Period(data, freq="M")) == pytest.approx(peso)


def _anual(linhas: list[tuple[str, int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame({"data": pd.to_datetime([l[0] for l in linhas]),
                         "ano_referencia": [l[1] for l in linhas],
                         "mediana": [l[2] for l in linhas], "desvio": [l[3] for l in linhas]})


def test_expectativa_12m_pondera_ano_corrente_e_seguinte():
    """Outubro: 3/12 do ano corrente + 9/12 do seguinte (Selic Focus de 2026-10-09)."""
    anual = _anual([("2026-10-09", 2026, 13.25, 0.2236), ("2026-10-09", 2027, 12.00, 0.7543)])
    exp = focus.expectativa_12m(anual, pd.Timestamp("2026-10-13"))
    assert exp.mediana == pytest.approx(0.25 * 13.25 + 0.75 * 12.00)
    assert exp.desvio == pytest.approx(0.25 * 0.2236 + 0.75 * 0.7543)
    assert exp.data == pd.Timestamp("2026-10-09")


def test_expectativa_12m_usa_a_ultima_pesquisa_ate_a_data():
    anual = _anual([("2026-09-30", 2026, 14.0, 0.1), ("2026-09-30", 2027, 13.0, 0.1),
                    ("2026-10-09", 2026, 13.25, 0.2), ("2026-10-09", 2027, 12.0, 0.2)])
    exp = focus.expectativa_12m(anual, pd.Timestamp("2026-10-01"))
    assert exp.data == pd.Timestamp("2026-09-30")
    assert exp.mediana == pytest.approx((4 / 12) * 14.0 + (8 / 12) * 13.0)


def test_expectativa_12m_em_janeiro_dispensa_o_ano_seguinte():
    anual = _anual([("2027-01-08", 2027, 11.0, 0.5)])
    exp = focus.expectativa_12m(anual, pd.Timestamp("2027-01-10"))
    assert exp.mediana == pytest.approx(11.0)


def test_expectativa_12m_sem_o_ano_seguinte_levanta_erro():
    anual = _anual([("2026-10-09", 2026, 13.25, 0.2)])
    with pytest.raises(focus.ErroFocus, match="2027"):
        focus.expectativa_12m(anual, pd.Timestamp("2026-10-13"))


def test_mediana_mensal_por_ano_pega_a_ultima_pesquisa_do_mes():
    anual = _anual([("2026-09-01", 2026, 1.0, 0.1), ("2026-09-30", 2026, 2.0, 0.1),
                    ("2026-09-30", 2027, 3.0, 0.1), ("2026-10-09", 2026, 4.0, 0.1)])
    mensal = focus.mediana_mensal_por_ano(anual)
    assert isinstance(mensal.index, pd.PeriodIndex)
    assert mensal.loc[pd.Period("2026-09", freq="M"), 2026] == 2.0
    assert mensal.loc[pd.Period("2026-09", freq="M"), 2027] == 3.0
    assert mensal.loc[pd.Period("2026-10", freq="M"), 2026] == 4.0
    assert np.isnan(mensal.loc[pd.Period("2026-10", freq="M"), 2027])


def test_mediana_mensal_da_serie_12m():
    serie = pd.DataFrame({"data": pd.to_datetime(["2026-09-02", "2026-09-29", "2026-10-09"]),
                          "mediana": [4.8, 4.6, 4.5], "desvio": [0.5, 0.5, 0.5]})
    mensal = focus.mediana_mensal(serie)
    assert mensal.to_dict() == {pd.Period("2026-09", freq="M"): 4.6, pd.Period("2026-10", freq="M"): 4.5}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_focus.py -q`
Expected: erro de coleta `ModuleNotFoundError: No module named 'core.focus'`

- [ ] **Step 3: Implementar**

Criar `core/focus.py`:

```python
"""
focus.py - Expectativas de mercado (Focus/BCB) via OData do Olinda, sem Streamlit.

Peculiaridades da API (sondadas ao vivo em 2026-10-12 e 2026-10-13):
- O ``$filter`` vai na URL já codificado com ``quote(v, safe="")``. Passar os
  parâmetros pelo ``params`` do requests devolve HTTP 400.
- O histórico inteiro de um indicador vem numa resposta só (PIB Total: 32.740
  linhas, 3,6 MB, ~2 s). Não há paginação.
- ``Data`` é a data da pesquisa (aaaa-mm-dd). ``DataReferencia`` é o ano, em
  texto. Pesquisas antigas têm ``Minimo``/``Maximo`` nulos; ``Mediana`` e
  ``DesvioPadrao`` existem desde o início (PIB: 1999-07; IPCA 12m: 2001-12).
- Indicadores anuais usados: "Selic", "Câmbio" (R$/US$ no fim do ano) e
  "PIB Total". A inflação 12 meses à frente vem de um endpoint próprio
  (``Suavizada='S'``).

``hoje`` é sempre explícito: pesquisas com ``Data > hoje`` são descartadas.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple
from urllib.parse import quote

import pandas as pd
import requests

URL_BASE = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"
ENDPOINT_ANUAIS = "ExpectativasMercadoAnuais"
ENDPOINT_IPCA_12M = "ExpectativasMercadoInflacao12Meses"
INDICADOR_SELIC = "Selic"
INDICADOR_CAMBIO = "Câmbio"
INDICADOR_PIB = "PIB Total"
_TIMEOUT_HTTP = 60
_CAMPOS_ANUAIS = {"Data": "data", "DataReferencia": "ano_referencia",
                  "Mediana": "mediana", "DesvioPadrao": "desvio"}
_CAMPOS_IPCA_12M = {"Data": "data", "Mediana": "mediana", "DesvioPadrao": "desvio"}


class ErroFocus(RuntimeError):
    """Falha ao obter ou interpretar expectativas do Focus."""


class Expectativa12m(NamedTuple):
    """Mediana e desvio-padrão dos analistas 12 meses à frente, na pesquisa de ``data``."""

    mediana: float
    desvio: float          # NaN quando a pesquisa não traz desvio
    data: pd.Timestamp


def _url(endpoint: str, filtro: str, campos: dict[str, str]) -> str:
    return (f"{URL_BASE}{endpoint}?$format=json&$select={','.join(campos)}"
            f"&$filter={quote(filtro, safe='')}")


def _consultar(endpoint: str, filtro: str, campos: dict[str, str]) -> pd.DataFrame:
    """Linhas do OData com os campos renomeados. Levanta ErroFocus em falha de rede ou de formato."""
    try:
        resposta = requests.get(_url(endpoint, filtro, campos), timeout=_TIMEOUT_HTTP)
        resposta.raise_for_status()
        registros = resposta.json()["value"]
    except (requests.RequestException, ValueError, KeyError, TypeError) as erro:
        raise ErroFocus(f"Focus {endpoint} ({filtro}): {erro}") from erro
    bruto = pd.DataFrame.from_records(registros)
    faltando = sorted(set(campos) - set(bruto.columns))
    if faltando:
        raise ErroFocus(f"Focus {endpoint} ({filtro}) sem os campos {faltando}.")
    return bruto[list(campos)].rename(columns=campos)


def _limpar(tabela: pd.DataFrame, hoje: date) -> pd.DataFrame:
    """Converte tipos, descarta linhas sem data/mediana e pesquisas posteriores a ``hoje``."""
    tabela = tabela.copy()
    tabela["data"] = pd.to_datetime(tabela["data"], format="%Y-%m-%d", errors="coerce")
    for coluna in ("mediana", "desvio"):
        tabela[coluna] = pd.to_numeric(tabela[coluna], errors="coerce")
    tabela = tabela.dropna(subset=["data", "mediana"])
    tabela = tabela[tabela["data"] <= pd.Timestamp(hoje)]
    if tabela.empty:
        raise ErroFocus(f"Focus sem pesquisas válidas até {hoje}.")
    return tabela


def baixar_focus_anual(indicador: str, hoje: date) -> pd.DataFrame:
    """
    Pesquisas diárias de um indicador anual, até ``hoje``.

    Colunas: ``data`` (Timestamp), ``ano_referencia`` (int), ``mediana`` e
    ``desvio`` (float, nas unidades do Focus: % a.a. para Selic e PIB, R$/US$
    para o câmbio). Ordenadas por (data, ano_referencia).
    """
    filtro = f"Indicador eq '{indicador}' and baseCalculo eq 0"
    tabela = _consultar(ENDPOINT_ANUAIS, filtro, _CAMPOS_ANUAIS)
    tabela["ano_referencia"] = pd.to_numeric(tabela["ano_referencia"], errors="coerce")
    tabela = _limpar(tabela.dropna(subset=["ano_referencia"]), hoje)
    tabela["ano_referencia"] = tabela["ano_referencia"].astype(int)
    return (tabela.sort_values(["data", "ano_referencia"])
            .drop_duplicates(["data", "ano_referencia"], keep="last")
            .reset_index(drop=True))


def baixar_focus_ipca_12m(hoje: date) -> pd.DataFrame:
    """IPCA 12 meses à frente (série suavizada), até ``hoje``. Colunas: ``data``, ``mediana``, ``desvio``."""
    filtro = "Indicador eq 'IPCA' and Suavizada eq 'S' and baseCalculo eq 0"
    tabela = _limpar(_consultar(ENDPOINT_IPCA_12M, filtro, _CAMPOS_IPCA_12M), hoje)
    return tabela.sort_values("data").drop_duplicates("data", keep="last").reset_index(drop=True)


def peso_ano_corrente(data: pd.Timestamp | pd.Period) -> float:
    """
    Peso do ano corrente numa projeção 12 meses à frente: w = (12 - k)/12, com
    k = meses completos do ano na data. Janeiro: k = 0, w = 1. Outubro: k = 9, w = 3/12.
    """
    return (13 - data.month) / 12.0


def expectativa_12m(anual: pd.DataFrame, ate: pd.Timestamp) -> Expectativa12m:
    """
    Projeção 12 meses à frente na última pesquisa com ``data <= ate``:
    ``w·F(Y) + (1 - w)·F(Y+1)``, Y = ano da pesquisa. O desvio usa a mesma
    ponderação, o que supõe correlação 1 entre os dois anos (conservador).

    Levanta ErroFocus se não houver pesquisa até ``ate`` ou se ela não tiver
    o ano Y (ou o Y+1, quando o peso dele é positivo).
    """
    anteriores = anual[anual["data"] <= ate]
    if anteriores.empty:
        raise ErroFocus(f"Nenhuma pesquisa Focus até {ate:%Y-%m-%d}.")
    data = anteriores["data"].max()
    pesquisa = anteriores[anteriores["data"] == data].set_index("ano_referencia")
    peso = peso_ano_corrente(data)
    anos = [data.year] + ([data.year + 1] if peso < 1.0 else [])
    faltando = [ano for ano in anos if ano not in pesquisa.index]
    if faltando:
        raise ErroFocus(f"Pesquisa Focus de {data:%Y-%m-%d} sem os anos {faltando}.")
    if peso == 1.0:
        linha = pesquisa.loc[data.year]
        return Expectativa12m(float(linha["mediana"]), float(linha["desvio"]), data)
    atual, seguinte = pesquisa.loc[data.year], pesquisa.loc[data.year + 1]
    return Expectativa12m(
        mediana=float(peso * atual["mediana"] + (1 - peso) * seguinte["mediana"]),
        desvio=float(peso * atual["desvio"] + (1 - peso) * seguinte["desvio"]),
        data=data,
    )


def mediana_mensal_por_ano(anual: pd.DataFrame) -> pd.DataFrame:
    """
    Mediana da última pesquisa de cada mês, por ano de referência.
    Índice: PeriodIndex mensal ``mes``. Colunas: anos de referência (int).
    """
    ordenada = anual.sort_values("data")
    mes = ordenada["data"].dt.to_period("M").rename("mes")
    return ordenada.groupby([mes, "ano_referencia"])["mediana"].last().unstack("ano_referencia")


def mediana_mensal(serie_12m: pd.DataFrame) -> pd.Series:
    """Mediana da última pesquisa de cada mês (IPCA 12m). Índice: PeriodIndex mensal ``mes``."""
    ordenada = serie_12m.sort_values("data")
    mes = ordenada["data"].dt.to_period("M").rename("mes")
    return ordenada.groupby(mes)["mediana"].last()
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_focus.py -q`
Expected: `17 passed`

- [ ] **Step 5: Commit**

```powershell
git add core/focus.py tests/test_focus.py
git commit -m "feat(core): fonte Focus (OData) com Expectativa12m" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Fonte Tesouro Direto (`core/tesouro.py`)

**Files:**
- Create: `core/tesouro.py`
- Test: `tests/test_tesouro.py`

**Interfaces:**
- Consumes: nada de tarefas anteriores.
- Produces:
  - `ErroTesouro(RuntimeError)`; `TIPO_LTN`, `TIPO_NTNF`, `URL_TESOURO`
  - `Curva(NamedTuple)`: `data_base: pd.Timestamp`, `juros_1a: float` e `juros_5a: float` (% a.a.; NaN fora dos prazos)
  - `ler_csv_tesouro(conteudo: bytes, hoje: date) -> pd.DataFrame`: colunas `tipo, vencimento, data_base, taxa, prazo_anos`
  - `baixar_prefixados(hoje: date) -> pd.DataFrame`, com o mesmo formato
  - `interpolar_taxa(prazos: np.ndarray, taxas: np.ndarray, alvo_anos: float) -> float`
  - `curva_fim_de_mes(titulos) -> pd.DataFrame`: PeriodIndex `mes`, colunas `juros_1a, juros_5a`
  - `curva_mais_recente(titulos) -> Curva`

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_tesouro.py`:

```python
"""
Testes de core/tesouro.py (sem rede).

O CSV falso segue o formato real (sondado em 2026-10-12/13): ``;``, vírgula
decimal, latin-1, datas dd/mm/aaaa e as 8 colunas do arquivo publicado.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import requests

from core import tesouro

HOJE = date(2026, 10, 13)
CABECALHO = ("Tipo Titulo;Data Vencimento;Data Base;Taxa Compra Manha;Taxa Venda Manha;"
             "PU Compra Manha;PU Venda Manha;PU Base Manha")


def _csv(linhas: list[tuple[str, str, str, str]], cabecalho: str = CABECALHO) -> bytes:
    """linhas: (tipo, vencimento, data_base, taxa_compra com vírgula)."""
    corpo = [f"{tipo};{venc};{base};{taxa};{taxa};900,00;900,00;900,00" for tipo, venc, base, taxa in linhas]
    return "\n".join([cabecalho, *corpo]).encode("latin-1")


LTN, NTNF, SELIC = tesouro.TIPO_LTN, tesouro.TIPO_NTNF, "Tesouro Selic"


def test_le_so_prefixados_converte_tipos_e_calcula_prazo():
    tabela = tesouro.ler_csv_tesouro(_csv([
        (LTN, "01/01/2028", "09/10/2026", "12,49"),
        (SELIC, "01/03/2027", "09/10/2026", "0,01"),
        (NTNF, "01/01/2031", "09/10/2026", "12,48"),
    ]), HOJE)
    assert sorted(tabela["tipo"]) == [LTN, NTNF]
    linha = tabela[tabela["tipo"] == LTN].iloc[0]
    assert linha["taxa"] == pytest.approx(12.49)
    assert linha["prazo_anos"] == pytest.approx((pd.Timestamp("2028-01-01") - pd.Timestamp("2026-10-09")).days / 365.25)


def test_taxa_zero_de_titulo_fora_de_venda_e_descartada():
    tabela = tesouro.ler_csv_tesouro(_csv([
        (NTNF, "01/01/2021", "04/03/2010", "0,00"),
        (LTN, "01/01/2012", "04/03/2010", "11,00"),
    ]), HOJE)
    assert tabela["vencimento"].tolist() == [pd.Timestamp("2012-01-01")]


def test_mesmo_vencimento_fica_a_ltn():
    tabela = tesouro.ler_csv_tesouro(_csv([
        (NTNF, "01/01/2027", "09/10/2026", "13,20"),
        (LTN, "01/01/2027", "09/10/2026", "13,28"),
    ]), HOJE)
    assert len(tabela) == 1 and tabela["tipo"].iloc[0] == LTN
    assert tabela["taxa"].iloc[0] == pytest.approx(13.28)


def test_datas_base_depois_de_hoje_ficam_de_fora():
    tabela = tesouro.ler_csv_tesouro(_csv([
        (LTN, "01/01/2028", "09/10/2026", "12,49"),
        (LTN, "01/01/2028", "14/10/2026", "99,00"),
    ]), HOJE)
    assert tabela["data_base"].max() == pd.Timestamp("2026-10-09")


def test_coluna_renomeada_levanta_erro_com_o_nome():
    cabecalho = CABECALHO.replace("Taxa Compra Manha", "Taxa Compra")
    with pytest.raises(tesouro.ErroTesouro, match="Taxa Compra Manha"):
        tesouro.ler_csv_tesouro(_csv([(LTN, "01/01/2028", "09/10/2026", "12,49")], cabecalho), HOJE)


def test_csv_sem_prefixados_levanta_erro():
    with pytest.raises(tesouro.ErroTesouro):
        tesouro.ler_csv_tesouro(_csv([(SELIC, "01/03/2027", "09/10/2026", "0,01")]), HOJE)


def test_interpolacao_linear_entre_dois_prazos():
    assert tesouro.interpolar_taxa(np.array([1.5, 0.5]), np.array([12.0, 10.0]), 1.0) == pytest.approx(11.0)


@pytest.mark.parametrize("prazos, alvo", [([0.2, 4.43], 5.0), ([1.2, 3.0], 1.0), ([], 1.0)])
def test_fora_do_intervalo_de_prazos_e_nan(prazos, alvo):
    taxas = np.full(len(prazos), 12.0)
    assert np.isnan(tesouro.interpolar_taxa(np.array(prazos), taxas, alvo))


def test_curva_fim_de_mes_usa_o_ultimo_dia_com_cotacao():
    titulos = tesouro.ler_csv_tesouro(_csv([
        (LTN, "01/07/2006", "01/09/2005", "19,00"),        # dia que não é o último do mês
        (LTN, "01/07/2006", "30/09/2005", "18,00"),        # prazo ~0,75a
        (NTNF, "01/01/2008", "30/09/2005", "17,00"),       # prazo ~2,25a
        (LTN, "01/07/2006", "31/10/2005", "17,50"),
        (NTNF, "01/01/2012", "31/10/2005", "16,00"),       # prazo ~6,17a
    ]), HOJE)
    curva = tesouro.curva_fim_de_mes(titulos)
    assert list(curva.index) == [pd.Period("2005-09", freq="M"), pd.Period("2005-10", freq="M")]
    p_curto = (pd.Timestamp("2006-07-01") - pd.Timestamp("2005-09-30")).days / 365.25
    p_longo = (pd.Timestamp("2008-01-01") - pd.Timestamp("2005-09-30")).days / 365.25
    esperado = 18.0 + (17.0 - 18.0) * (1.0 - p_curto) / (p_longo - p_curto)
    assert curva.loc[pd.Period("2005-09", freq="M"), "juros_1a"] == pytest.approx(esperado)
    assert np.isnan(curva.loc[pd.Period("2005-09", freq="M"), "juros_5a"])
    assert not np.isnan(curva.loc[pd.Period("2005-10", freq="M"), "juros_5a"])


def test_curva_mais_recente():
    titulos = tesouro.ler_csv_tesouro(_csv([
        (LTN, "01/01/2027", "09/10/2026", "13,28"),
        (LTN, "01/01/2028", "09/10/2026", "12,49"),
        (NTNF, "01/01/2033", "09/10/2026", "12,59"),
        (LTN, "01/01/2028", "08/10/2026", "99,00"),
    ]), HOJE)
    curva = tesouro.curva_mais_recente(titulos)
    assert curva.data_base == pd.Timestamp("2026-10-09")
    assert 12.49 < curva.juros_1a < 13.28
    assert 12.49 < curva.juros_5a < 12.59


class _Resposta:
    def __init__(self, status_code: int, conteudo: bytes = b""):
        self.status_code, self.content = status_code, conteudo

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error", response=self)


def test_baixar_prefixados_le_o_conteudo_baixado(monkeypatch):
    conteudo = _csv([(LTN, "01/01/2028", "09/10/2026", "12,49")])
    monkeypatch.setattr(tesouro.requests, "get", lambda url, timeout=None: _Resposta(200, conteudo))
    assert len(tesouro.baixar_prefixados(HOJE)) == 1


@pytest.mark.parametrize("falha", [requests.ConnectionError("sem rede"), _Resposta(503)])
def test_baixar_prefixados_traduz_falha_de_rede(monkeypatch, falha):
    def get(url, timeout=None):
        if isinstance(falha, Exception):
            raise falha
        return falha
    monkeypatch.setattr(tesouro.requests, "get", get)
    with pytest.raises(tesouro.ErroTesouro):
        tesouro.baixar_prefixados(HOJE)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_tesouro.py -q`
Expected: erro de coleta `ModuleNotFoundError: No module named 'core.tesouro'`

- [ ] **Step 3: Implementar**

Criar `core/tesouro.py`:

```python
"""
tesouro.py - Taxas prefixadas do Tesouro Direto (Tesouro Transparente), sem Streamlit.

Fonte: CSV ``PrecoTaxaTesouroDireto.csv`` (~14,6 MB; ~27 s para baixar), com
``;`` de separador, vírgula decimal e latin-1. Data Base de 2004-12-31 em diante.

Achados (sondados ao vivo em 2026-10-12 e 2026-10-13):
- Prefixados: "Tesouro Prefixado" (LTN, sem cupom) e "Tesouro Prefixado com
  Juros Semestrais" (NTN-F, com cupom). Até ~2010 os prazos longos só existem
  via NTN-F. O primeiro mês com prazo >= 5 anos é 2005-11; antes dele o
  prazo máximo não passa de 4,43 anos, e nenhum mês tem prazo máximo entre
  4,5 e 5 anos. Por isso "sem extrapolação" em 5 anos equivale ao critério
  "título com prazo >= 4,5a" do spec.
- ``Taxa Compra Manha`` vale 0 quando o título não está à venda (NTN-F 2021,
  jan-mar/2010, cuja ``Taxa Venda Manha`` de 0,06 também é lixo). Linhas com
  taxa <= 0 são descartadas.
- LTN e NTN-F podem vencer na mesma data. Fica a LTN, que é zero-cupom; a
  taxa da NTN-F é o yield de um título com cupom.

``hoje`` é sempre explícito: linhas com ``Data Base > hoje`` são descartadas.
"""

from __future__ import annotations

import io
from datetime import date
from typing import NamedTuple

import numpy as np
import pandas as pd
import requests

URL_TESOURO = ("https://www.tesourotransparente.gov.br/ckan/dataset/"
               "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
               "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv")
TIPO_LTN = "Tesouro Prefixado"
TIPO_NTNF = "Tesouro Prefixado com Juros Semestrais"
DIAS_POR_ANO = 365.25
PRAZO_CURTO_ANOS = 1.0
PRAZO_LONGO_ANOS = 5.0
_TIMEOUT_HTTP = 180
_COLUNAS = {"Tipo Titulo": "tipo", "Data Vencimento": "vencimento",
            "Data Base": "data_base", "Taxa Compra Manha": "taxa"}


class ErroTesouro(RuntimeError):
    """Falha ao obter ou interpretar o CSV do Tesouro Direto."""


class Curva(NamedTuple):
    """Taxas prefixadas (% a.a.) interpoladas em 1 e 5 anos num dia; NaN fora dos prazos."""

    data_base: pd.Timestamp
    juros_1a: float
    juros_5a: float


def ler_csv_tesouro(conteudo: bytes, hoje: date) -> pd.DataFrame:
    """
    Prefixados com ``data_base <= hoje``, um por (data_base, vencimento).

    Colunas: ``data_base``, ``vencimento`` (Timestamp), ``tipo`` (str),
    ``taxa`` (% a.a.) e ``prazo_anos`` (dias corridos / 365,25).
    """
    try:
        bruto = pd.read_csv(io.BytesIO(conteudo), sep=";", decimal=",", encoding="latin-1")
    except ValueError as erro:
        raise ErroTesouro(f"CSV do Tesouro ilegível: {erro}") from erro
    faltando = sorted(set(_COLUNAS) - set(bruto.columns))
    if faltando:
        raise ErroTesouro(f"CSV do Tesouro sem as colunas {faltando}.")

    tabela = bruto[list(_COLUNAS)].rename(columns=_COLUNAS)
    tabela = tabela[tabela["tipo"].isin((TIPO_LTN, TIPO_NTNF))].copy()
    for coluna in ("data_base", "vencimento"):
        tabela[coluna] = pd.to_datetime(tabela[coluna], format="%d/%m/%Y", errors="coerce")
    tabela["taxa"] = pd.to_numeric(tabela["taxa"], errors="coerce")
    tabela = tabela.dropna(subset=["data_base", "vencimento", "taxa"])
    tabela = tabela[(tabela["taxa"] > 0)
                    & (tabela["data_base"] <= pd.Timestamp(hoje))
                    & (tabela["vencimento"] > tabela["data_base"])]
    if tabela.empty:
        raise ErroTesouro(f"Nenhum prefixado com Data Base até {hoje}.")

    tabela["ltn_primeiro"] = tabela["tipo"] != TIPO_LTN
    tabela = (tabela.sort_values(["data_base", "vencimento", "ltn_primeiro"])
              .drop_duplicates(["data_base", "vencimento"], keep="first")
              .drop(columns="ltn_primeiro"))
    tabela["prazo_anos"] = (tabela["vencimento"] - tabela["data_base"]).dt.days / DIAS_POR_ANO
    return tabela.reset_index(drop=True)


def baixar_prefixados(hoje: date) -> pd.DataFrame:
    """Baixa o CSV inteiro (~27 s) e devolve ``ler_csv_tesouro``. Levanta ErroTesouro."""
    try:
        resposta = requests.get(URL_TESOURO, timeout=_TIMEOUT_HTTP)
        resposta.raise_for_status()
    except requests.RequestException as erro:
        raise ErroTesouro(f"Download do CSV do Tesouro: {erro}") from erro
    return ler_csv_tesouro(resposta.content, hoje)


def interpolar_taxa(prazos: np.ndarray, taxas: np.ndarray, alvo_anos: float) -> float:
    """Interpolação linear da taxa em ``alvo_anos``. NaN fora do intervalo de prazos (sem extrapolação)."""
    if len(prazos) == 0:
        return float("nan")
    ordem = np.argsort(prazos)
    prazos, taxas = np.asarray(prazos)[ordem], np.asarray(taxas)[ordem]
    if alvo_anos < prazos[0] or alvo_anos > prazos[-1]:
        return float("nan")
    return float(np.interp(alvo_anos, prazos, taxas))


def _curva_do_dia(data_base: pd.Timestamp, titulos_do_dia: pd.DataFrame) -> Curva:
    prazos = titulos_do_dia["prazo_anos"].to_numpy()
    taxas = titulos_do_dia["taxa"].to_numpy()
    return Curva(data_base, interpolar_taxa(prazos, taxas, PRAZO_CURTO_ANOS),
                 interpolar_taxa(prazos, taxas, PRAZO_LONGO_ANOS))


def curva_mais_recente(titulos: pd.DataFrame) -> Curva:
    """Curva na última Data Base disponível."""
    ultima = titulos["data_base"].max()
    return _curva_do_dia(ultima, titulos[titulos["data_base"] == ultima])


def curva_fim_de_mes(titulos: pd.DataFrame) -> pd.DataFrame:
    """
    Taxas de 1 e 5 anos (% a.a.) na última Data Base de cada mês.
    Índice: PeriodIndex mensal ``mes``. Colunas: ``juros_1a``, ``juros_5a``.
    Meses sem cotação não aparecem; quem consome reindexa.
    """
    mes = titulos["data_base"].dt.to_period("M")
    ultimo_dia = titulos["data_base"] == titulos.groupby(mes)["data_base"].transform("max")
    curvas = [_curva_do_dia(data_base, grupo)
              for data_base, grupo in titulos[ultimo_dia].groupby("data_base")]
    return pd.DataFrame(
        {"juros_1a": [c.juros_1a for c in curvas], "juros_5a": [c.juros_5a for c in curvas]},
        index=pd.PeriodIndex([c.data_base.to_period("M") for c in curvas], name="mes"),
    )
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_tesouro.py -q`
Expected: `15 passed`

- [ ] **Step 5: Commit**

```powershell
git add core/tesouro.py tests/test_tesouro.py
git commit -m "feat(core): fonte Tesouro Direto e curva prefixada de 1a/5a" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Preços mensais (`market_data.carregar_precos_mensais`)

**Files:**
- Modify: `core/market_data.py` (nova função logo antes de `baixar_cdi_diario`)
- Test: `tests/test_precos_mensais.py`

**Interfaces:**
- Consumes: `market_data.baixar_fechamentos(tickers, inicio, fim)`, que já existe.
- Produces: `carregar_precos_mensais(tickers: list[str], inicio: date, hoje: date) -> pd.DataFrame`.
  Índice: PeriodIndex `mes` com os meses < mês de `hoje`. Colunas: tickers. NaN onde a ação não tem preço no mês.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_precos_mensais.py`:

```python
"""Testes de market_data.carregar_precos_mensais (sem rede)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import market_data

HOJE = date(2026, 10, 13)


@pytest.fixture
def yahoo(monkeypatch):
    """AAA.SA sobe 1 por dia útil; BBB.SA só existe em setembro e falta no último pregão dele."""
    dias = pd.bdate_range("2026-08-03", "2026-10-13")
    precos = pd.DataFrame({"AAA.SA": np.arange(len(dias), dtype=float) + 10.0}, index=dias)
    precos["BBB.SA"] = np.nan
    setembro = (dias >= "2026-09-01") & (dias <= "2026-09-29")
    precos.loc[setembro, "BBB.SA"] = 50.0
    pedidos = []

    def baixar(tickers, inicio, fim):
        pedidos.append((list(tickers), inicio, fim))
        return precos[tickers]
    monkeypatch.setattr(market_data, "baixar_fechamentos", baixar)
    return precos, pedidos


def test_ultimo_preco_de_cada_mes_completo(yahoo):
    precos, pedidos = yahoo
    mensais = market_data.carregar_precos_mensais(["AAA.SA", "BBB.SA"], date(2026, 8, 1), HOJE)

    assert pedidos == [(["AAA.SA", "BBB.SA"], date(2026, 8, 1), HOJE)]
    assert isinstance(mensais.index, pd.PeriodIndex) and mensais.index.name == "mes"
    assert list(mensais.index) == [pd.Period("2026-08", freq="M"), pd.Period("2026-09", freq="M")]
    assert mensais.loc[pd.Period("2026-08", freq="M"), "AAA.SA"] == precos.loc["2026-08-31", "AAA.SA"]
    assert mensais.loc[pd.Period("2026-09", freq="M"), "AAA.SA"] == precos.loc["2026-09-30", "AAA.SA"]


def test_sem_filtro_de_cobertura_e_ultimo_valor_valido_do_mes(yahoo):
    mensais = market_data.carregar_precos_mensais(["AAA.SA", "BBB.SA"], date(2026, 8, 1), HOJE)
    assert np.isnan(mensais.loc[pd.Period("2026-08", freq="M"), "BBB.SA"])
    assert mensais.loc[pd.Period("2026-09", freq="M"), "BBB.SA"] == 50.0
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_precos_mensais.py -q`
Expected: FAIL com `AttributeError: module 'core.market_data' has no attribute 'carregar_precos_mensais'`

- [ ] **Step 3: Implementar**

Em `core/market_data.py`, inserir logo antes de `def baixar_cdi_diario(`:

```python
def carregar_precos_mensais(tickers: list[str], inicio: date, hoje: date) -> pd.DataFrame:
    """
    Último fechamento ajustado de cada mês COMPLETO (meses anteriores ao de ``hoje``).

    Sem filtro de cobertura e sem ffill, ao contrário de ``carregar_precos``: o
    modelo de fatores usa até 180 meses e precisa também das ações com
    histórico curto. Cada regressão usa só os meses em que a ação tem preço.
    Índice: PeriodIndex mensal ``mes``.
    """
    fechamentos = baixar_fechamentos(tickers, inicio, hoje)
    mensais = fechamentos.groupby(fechamentos.index.to_period("M")).last().rename_axis("mes")
    return mensais[mensais.index < pd.Period(hoje, freq="M")]
```

- [ ] **Step 4: Rodar e ver passar (inclusive os testes antigos de market_data)**

Run: `.venv/Scripts/python -m pytest tests/test_precos_mensais.py tests/test_bcb.py -q`
Expected: `16 passed`

- [ ] **Step 5: Commit**

```powershell
git add core/market_data.py tests/test_precos_mensais.py
git commit -m "feat(core): preços mensais sem filtro de cobertura" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Painel mensal de fatores (`core/fatores_macro.py`, parte 1)

**Files:**
- Create: `core/fatores_macro.py`
- Test: `tests/test_fatores_macro.py`

**Interfaces:**
- Consumes:
  - `tesouro.curva_fim_de_mes(titulos) -> DataFrame[juros_1a, juros_5a]` e `tesouro.TIPO_LTN/TIPO_NTNF` (Tarefa 2);
  - `focus.mediana_mensal(serie_12m)`, `focus.mediana_mensal_por_ano(anual)` e `focus.peso_ano_corrente(mes)` (Tarefa 1).
- Produces (o Plano B depende disto):
  - `FATORES = ("juros", "inflacao", "cambio", "atividade", "premio", "mercado")`, `FATORES_MACRO = FATORES[:5]` e `INICIO_PAINEL = Period("2005-12")`
  - `DadosMacro`, um dataclass congelado em que todo campo tem padrão None:
    - `titulos`, `focus_ipca_12m`, `focus_selic`, `focus_cambio` e `focus_pib` (DataFrames);
    - `ptax`, `bvsp_mensal`, `cdi_diario`, `selic_meta` e `ipca_12m` (Series);
    - `erros: dict[str, str]`, com as chaves de fonte `tesouro`, `focus_ipca`, `focus_selic`, `focus_cambio`, `focus_pib`, `ptax`, `bvsp`, `cdi`, `selic_meta` e `ipca_12m`.
  - `PainelMacro`, com os campos:
    - `fatores: DataFrame`: PeriodIndex `mes` de 2005-12 até `Period(hoje) - 1`, com as colunas disponíveis na ordem de `FATORES`;
    - `cdi_mensal: Series`, em decimal;
    - `fatores_indisponiveis: dict[str, str]`;
    - `avisos: tuple[str, ...]`.
  - `cdi_mensal(cdi_diario) -> Series`, `revisao_pib_12m(medianas) -> Series` e `montar_painel(dados: DadosMacro, hoje: date) -> PainelMacro`

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_fatores_macro.py`:

```python
"""
Testes de core/fatores_macro.py (sem rede). Os dados são sintéticos, com
resposta calculável à mão: o painel vai de 2005-12 a 2006-03 (hoje = 2006-04-15).
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import fatores_macro, tesouro
from core.fatores_macro import DadosMacro

HOJE = date(2006, 4, 15)


def _mes(texto: str) -> pd.Period:
    return pd.Period(texto, freq="M")


def _titulos(curvas: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Dois títulos por dia, com prazo exato de 1 e 5 anos: a curva interpolada devolve as taxas dadas."""
    linhas = []
    for dia, (taxa_1a, taxa_5a) in curvas.items():
        base = pd.Timestamp(dia)
        linhas += [{"data_base": base, "vencimento": base + pd.Timedelta(days=365), "tipo": tesouro.TIPO_LTN,
                    "taxa": taxa_1a, "prazo_anos": 1.0},
                   {"data_base": base, "vencimento": base + pd.Timedelta(days=1826), "tipo": tesouro.TIPO_NTNF,
                    "taxa": taxa_5a, "prazo_anos": 5.0}]
    return pd.DataFrame(linhas)


def _focus_anual(linhas: list[tuple[str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame({"data": pd.to_datetime([l[0] for l in linhas]),
                         "ano_referencia": [l[1] for l in linhas],
                         "mediana": [l[2] for l in linhas], "desvio": 0.5})


@pytest.fixture
def dados() -> DadosMacro:
    titulos = _titulos({
        "2005-11-16": (99.0, 99.0),                   # não é o último dia do mês: ignorado
        "2005-11-30": (18.0, 17.0), "2005-12-30": (17.5, 16.0),
        "2006-01-31": (17.0, 16.5),                   # fevereiro sem cotação
        "2006-03-31": (16.0, 15.0), "2006-04-10": (1.0, 1.0),   # abril: mês incompleto
    })
    ipca = pd.DataFrame({"data": pd.to_datetime(["2005-11-28", "2005-12-26", "2006-01-30",
                                                 "2006-02-20", "2006-02-24", "2006-03-27"]),
                         "mediana": [5.0, 4.8, 4.6, 9.9, 4.5, 4.7], "desvio": 0.4})
    pib = _focus_anual([("2005-11-28", 2005, 2.0), ("2005-11-28", 2006, 3.0),
                        ("2005-12-26", 2005, 2.2), ("2005-12-26", 2006, 3.6),
                        ("2006-01-30", 2006, 3.4), ("2006-01-30", 2007, 3.5),
                        ("2006-02-24", 2006, 3.0), ("2006-02-24", 2007, 3.5),
                        ("2006-03-27", 2006, 3.3), ("2006-03-27", 2007, 3.1)])
    ptax = pd.Series([2.30, 2.20, 2.35, 2.10, 2.15, 2.12],
                     index=pd.to_datetime(["2005-11-30", "2005-12-01", "2005-12-30",
                                           "2006-01-31", "2006-02-24", "2006-03-31"]))
    bvsp = pd.Series([30000.0, 33000.0, 36300.0, 36300.0, 34485.0],
                     index=pd.PeriodIndex(["2005-11", "2005-12", "2006-01", "2006-02", "2006-03"], freq="M"))
    dias = pd.bdate_range("2005-11-01", "2006-04-14")
    cdi = pd.Series(0.0005, index=dias)
    return DadosMacro(titulos=titulos, focus_ipca_12m=ipca, focus_pib=pib, ptax=ptax,
                      bvsp_mensal=bvsp, cdi_diario=cdi)


def test_painel_vai_de_2005_12_ao_ultimo_mes_completo(dados):
    painel = fatores_macro.montar_painel(dados, HOJE)
    assert list(painel.fatores.index) == [_mes("2005-12"), _mes("2006-01"), _mes("2006-02"), _mes("2006-03")]
    assert list(painel.fatores.columns) == list(fatores_macro.FATORES)
    assert painel.fatores_indisponiveis == {} and painel.avisos == ()


def test_juros_e_premio_sao_diferencas_de_fim_de_mes_e_mes_sem_cotacao_vira_nan(dados):
    f = fatores_macro.montar_painel(dados, HOJE).fatores
    assert f.at[_mes("2005-12"), "juros"] == pytest.approx(17.5 - 18.0)
    assert f.at[_mes("2006-01"), "juros"] == pytest.approx(17.0 - 17.5)
    assert f.at[_mes("2005-12"), "premio"] == pytest.approx((16.0 - 17.5) - (17.0 - 18.0))
    assert np.isnan(f.at[_mes("2006-02"), "juros"])        # fevereiro sem cotação
    assert np.isnan(f.at[_mes("2006-03"), "juros"])        # não vira diferença de dois meses


def test_inflacao_usa_a_ultima_pesquisa_do_mes(dados):
    f = fatores_macro.montar_painel(dados, HOJE).fatores
    assert f.at[_mes("2006-02"), "inflacao"] == pytest.approx(4.5 - 4.6)   # 24/02, não 20/02
    assert f.at[_mes("2006-03"), "inflacao"] == pytest.approx(4.7 - 4.5)


def test_cambio_e_variacao_do_log_da_ptax_de_fim_de_mes(dados):
    f = fatores_macro.montar_painel(dados, HOJE).fatores
    assert f.at[_mes("2005-12"), "cambio"] == pytest.approx(np.log(2.35 / 2.30))
    assert f.at[_mes("2006-02"), "cambio"] == pytest.approx(np.log(2.15 / 2.10))


def test_atividade_e_revisao_com_ano_de_referencia_fixo(dados):
    f = fatores_macro.montar_painel(dados, HOJE).fatores
    # dezembro/2005: w = 1/12 no ano 2005 e 11/12 em 2006
    assert f.at[_mes("2005-12"), "atividade"] == pytest.approx((1 / 12) * (2.2 - 2.0) + (11 / 12) * (3.6 - 3.0))
    # janeiro/2006: w = 1, só o ano 2006, comparado com a pesquisa de dezembro para 2006
    assert f.at[_mes("2006-01"), "atividade"] == pytest.approx(3.4 - 3.6)
    # março/2006: w = 10/12
    assert f.at[_mes("2006-03"), "atividade"] == pytest.approx((10 / 12) * (3.3 - 3.0) + (2 / 12) * (3.1 - 3.5))


def test_mercado_e_retorno_do_ibovespa_menos_cdi_composto(dados):
    painel = fatores_macro.montar_painel(dados, HOJE)
    dias_dez = len(pd.bdate_range("2005-12-01", "2005-12-31"))
    cdi_dez = 1.0005 ** dias_dez - 1.0
    assert painel.cdi_mensal[_mes("2005-12")] == pytest.approx(cdi_dez)
    assert painel.fatores.at[_mes("2005-12"), "mercado"] == pytest.approx(0.10 - cdi_dez)
    assert painel.cdi_mensal.index.max() == _mes("2006-03")   # abril, incompleto, fica de fora


def test_fonte_que_falha_tira_so_os_fatores_dela(dados):
    sem_tesouro = DadosMacro(**{**dados.__dict__, "titulos": None, "erros": {"tesouro": "timeout"}})
    painel = fatores_macro.montar_painel(sem_tesouro, HOJE)
    assert "juros" not in painel.fatores.columns and "premio" not in painel.fatores.columns
    assert painel.fatores_indisponiveis == {"juros": "tesouro: timeout", "premio": "tesouro: timeout"}
    assert len(painel.avisos) == 2
    assert painel.fatores["inflacao"].notna().any()


def test_sem_cdi_nao_ha_fator_mercado(dados):
    sem_cdi = DadosMacro(**{**dados.__dict__, "cdi_diario": None, "erros": {"cdi": "HTTP 503"}})
    painel = fatores_macro.montar_painel(sem_cdi, HOJE)
    assert painel.fatores_indisponiveis == {"mercado": "cdi: HTTP 503"}
    assert painel.cdi_mensal.empty


def test_revisao_pib_nan_quando_falta_o_ano_seguinte():
    medianas = pd.DataFrame({2026: [3.0, 3.1]}, index=pd.PeriodIndex(["2026-09", "2026-10"], freq="M"))
    assert np.isnan(fatores_macro.revisao_pib_12m(medianas)[_mes("2026-10")])
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_fatores_macro.py -q`
Expected: erro de coleta `ModuleNotFoundError: No module named 'core.fatores_macro'`

- [ ] **Step 3: Implementar**

Criar `core/fatores_macro.py`:

```python
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

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from core import focus, tesouro

FATORES = ("juros", "inflacao", "cambio", "atividade", "premio", "mercado")
FATORES_MACRO = FATORES[:5]
INICIO_PAINEL = pd.Period("2005-12", freq="M")
_FONTES_DO_FATOR = {
    "juros": ("tesouro",), "inflacao": ("focus_ipca",), "cambio": ("ptax",),
    "atividade": ("focus_pib",), "premio": ("tesouro",), "mercado": ("bvsp", "cdi"),
}


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
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_fatores_macro.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```powershell
git add core/fatores_macro.py tests/test_fatores_macro.py
git commit -m "feat(core): painel mensal dos fatores de surpresa macro" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Ortogonalização do fator mercado (`estimar_gama`, `mercado_residual`)

**Files:**
- Modify: `core/fatores_macro.py` (anexar no fim do arquivo)
- Test: `tests/test_fatores_macro.py` (anexar no fim)

**Interfaces:**
- Consumes: `FATORES_MACRO` e as colunas de `PainelMacro.fatores` (Tarefa 4).
- Produces (usado pelo modelo e pela validação no Plano B/C):
  - `estimar_gama(fatores: pd.DataFrame) -> pd.Series`: índice `["const", *fatores_macro_presentes]`. Levanta `ValueError` sem a coluna `mercado` ou com menos de `n_coef + 2` meses completos.
  - `mercado_residual(fatores: pd.DataFrame, gama: pd.Series) -> pd.Series`, de nome `"mercado"`, com NaN onde falta algum termo.

- [ ] **Step 1: Escrever o teste que falha**

Anexar a `tests/test_fatores_macro.py`:

```python
def _fatores_sinteticos(n: int = 120, semente: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    gerador = np.random.default_rng(semente)
    meses = pd.period_range("2010-01", periods=n, freq="M", name="mes")
    macro = pd.DataFrame(gerador.normal(size=(n, 5)), index=meses, columns=list(fatores_macro.FATORES_MACRO))
    verdadeiro = np.array([0.01, 0.5, -0.3, -0.2, 0.1, 0.4])
    ruido = gerador.normal(scale=0.05, size=n)
    mercado = verdadeiro[0] + macro.to_numpy() @ verdadeiro[1:] + ruido
    return macro.assign(mercado=mercado), verdadeiro


def test_mercado_residual_e_ortogonal_aos_fatores_macro_na_janela():
    fatores, verdadeiro = _fatores_sinteticos()
    gama = fatores_macro.estimar_gama(fatores)
    residuo = fatores_macro.mercado_residual(fatores, gama)
    regressores = np.column_stack([np.ones(len(fatores)), fatores[list(fatores_macro.FATORES_MACRO)]])
    np.testing.assert_allclose(regressores.T @ residuo.to_numpy(), 0.0, atol=1e-9)
    np.testing.assert_allclose(gama.to_numpy(), verdadeiro, atol=0.03)
    assert list(gama.index) == ["const", *fatores_macro.FATORES_MACRO]


def test_gama_de_uma_janela_aplicado_em_outra_e_nan_onde_falta_fator():
    fatores, _ = _fatores_sinteticos()
    gama = fatores_macro.estimar_gama(fatores.iloc[:60])
    fora = fatores.iloc[60:].copy()
    fora.iloc[0, 0] = np.nan
    residuo = fatores_macro.mercado_residual(fora, gama)
    linha = fora.iloc[1]
    esperado = linha["mercado"] - gama["const"] - sum(gama[f] * linha[f] for f in fatores_macro.FATORES_MACRO)
    assert residuo.iloc[1] == pytest.approx(esperado)
    assert np.isnan(residuo.iloc[0])


def test_gama_so_com_os_fatores_macro_presentes():
    fatores, _ = _fatores_sinteticos()
    gama = fatores_macro.estimar_gama(fatores.drop(columns=["juros", "premio"]))
    assert list(gama.index) == ["const", "inflacao", "cambio", "atividade"]


@pytest.mark.parametrize("cortar", ["mercado", "linhas"])
def test_gama_recusa_painel_sem_mercado_ou_curto(cortar):
    fatores, _ = _fatores_sinteticos()
    entrada = fatores.drop(columns=["mercado"]) if cortar == "mercado" else fatores.iloc[:7]
    with pytest.raises(ValueError):
        fatores_macro.estimar_gama(entrada)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_fatores_macro.py -q`
Expected: 5 FAIL com `AttributeError: module 'core.fatores_macro' has no attribute 'estimar_gama'`; os 9 anteriores passam.

- [ ] **Step 3: Implementar**

Anexar ao fim de `core/fatores_macro.py`:

```python
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
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_fatores_macro.py -q`
Expected: `14 passed`

- [ ] **Step 5: Commit**

```powershell
git add core/fatores_macro.py tests/test_fatores_macro.py
git commit -m "feat(core): ortogonalização do fator mercado (gama por janela)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Download paralelo com isolamento de falha (`baixar_dados_macro`)

**Files:**
- Modify: `core/bcb.py` (uma constante)
- Modify: `core/fatores_macro.py` (bloco de imports, constantes e seção nova no fim)
- Test: `tests/test_fatores_macro.py` (linha de import e seção nova no fim)

**Interfaces:**
- Consumes:
  - `tesouro.baixar_prefixados` e `focus.baixar_focus_ipca_12m`/`baixar_focus_anual` (Tarefas 1 e 2);
  - `market_data.carregar_precos_mensais` (Tarefa 3);
  - `bcb.baixar_ptax`, `bcb.baixar_serie`, `bcb.SERIE_CDI` e `bcb.SERIE_SELIC_META`, que já existem.
- Produces:
  - `bcb.SERIE_IPCA_12M = 13522`;
  - `fatores_macro.baixar_dados_macro(hoje: date) -> DadosMacro`, que nunca levanta erro de fonte e propaga qualquer outra exceção;
  - `INICIO_DOWNLOAD = date(2005, 10, 1)` e `TICKER_MERCADO = "^BVSP"`.

- [ ] **Step 1: Escrever o teste que falha**

Em `tests/test_fatores_macro.py`, trocar a linha `from core import fatores_macro, tesouro` por:

```python
from core import bcb, fatores_macro, focus, market_data, tesouro
```

e anexar ao fim:

```python
# ---------------------------------------------------------------------------
# Download: erros de fonte viram None + motivo; bugs não são engolidos
# ---------------------------------------------------------------------------

@pytest.fixture
def fontes_falsas(monkeypatch):
    def serie_bcb(codigo, inicio, hoje):
        return pd.Series([float(codigo)], index=pd.to_datetime([hoje]))
    monkeypatch.setattr(tesouro, "baixar_prefixados", lambda hoje: _titulos({"2026-10-09": (13.0, 12.5)}))
    monkeypatch.setattr(focus, "baixar_focus_ipca_12m",
                        lambda hoje: pd.DataFrame({"data": [pd.Timestamp("2026-10-09")],
                                                   "mediana": [4.48], "desvio": [0.5]}))
    monkeypatch.setattr(focus, "baixar_focus_anual",
                        lambda indicador, hoje: _focus_anual([("2026-10-09", 2026, 1.0), ("2026-10-09", 2027, 2.0)]))
    monkeypatch.setattr(bcb, "baixar_ptax", lambda inicio, hoje: pd.Series([5.0], index=pd.to_datetime(["2026-10-09"])))
    monkeypatch.setattr(bcb, "baixar_serie", serie_bcb)
    monkeypatch.setattr(market_data, "carregar_precos_mensais",
                        lambda tickers, inicio, hoje: pd.DataFrame(
                            {"^BVSP": [100.0]}, index=pd.PeriodIndex(["2026-09"], freq="M", name="mes")))


def test_baixar_dados_macro_guarda_o_erro_da_fonte_e_segue(fontes_falsas, monkeypatch):
    def falha(hoje):
        raise tesouro.ErroTesouro("HTTP 503")
    monkeypatch.setattr(tesouro, "baixar_prefixados", falha)

    dados = fatores_macro.baixar_dados_macro(date(2026, 10, 13))

    assert dados.titulos is None and dados.erros == {"tesouro": "HTTP 503"}
    assert dados.focus_pib is not None and dados.ptax is not None
    assert dados.cdi_diario.iloc[0] == pytest.approx(bcb.SERIE_CDI / 100.0)


def test_baixar_dados_macro_nao_engole_bug(fontes_falsas, monkeypatch):
    def bug(hoje):
        raise KeyError("coluna")
    monkeypatch.setattr(focus, "baixar_focus_ipca_12m", bug)
    with pytest.raises(KeyError):
        fatores_macro.baixar_dados_macro(date(2026, 10, 13))
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_fatores_macro.py -q`
Expected: 2 FAIL com `AttributeError: module 'core.fatores_macro' has no attribute 'baixar_dados_macro'`; os 14 anteriores passam.

- [ ] **Step 3: Implementar**

(a) Em `core/bcb.py`, logo abaixo de `SERIE_SELIC_META = 432 ...`:

```python
SERIE_IPCA_12M = 13522      # IPCA acumulado em 12 meses, %; mensal, datado do dia 1 do mês
```

(b) Em `core/fatores_macro.py`, trocar o bloco de imports por:

```python
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TypeVar

import numpy as np
import pandas as pd

from core import bcb, focus, market_data, tesouro
```

(c) Logo abaixo do dicionário `_FONTES_DO_FATOR`, acrescentar:

```python
INICIO_DOWNLOAD = date(2005, 10, 1)      # o nível de 2005-11 entra na primeira variação
TICKER_MERCADO = "^BVSP"
_ERROS_DE_FONTE = (bcb.ErroBcb, focus.ErroFocus, tesouro.ErroTesouro, market_data.ErroDadosMercado)

T = TypeVar("T")
```

(d) Anexar ao fim do arquivo:

```python
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
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_fatores_macro.py tests/test_bcb.py -q`
Expected: `30 passed`

- [ ] **Step 5: Commit**

```powershell
git add core/bcb.py core/fatores_macro.py tests/test_fatores_macro.py
git commit -m "feat(core): download macro paralelo com isolamento de falha por fonte" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Retrato macro de hoje (`montar_retrato`)

**Files:**
- Modify: `core/fatores_macro.py` (import, constantes, 3 classes e seção nova no fim)
- Test: `tests/test_fatores_macro.py` (anexar no fim)

**Interfaces:**
- Consumes: `focus.expectativa_12m`, `focus.Expectativa12m`, `tesouro.curva_mais_recente` e `DadosMacro`.
- Produces (consumido pela aba "Cenários macro" no Plano C e pelos cenários Focus no Plano B):
  - `Indicador`, dataclass congelado com `nome`, `valor`, `unidade`, `data: pd.Timestamp`, `fonte` e `dias_maximos: int`;
  - `ExpectativasFocus(NamedTuple)`: `selic`, `ipca`, `cambio` e `pib`, cada um `Expectativa12m | None`;
  - `RetratoMacro`, dataclass congelado com `indicadores: tuple[Indicador, ...]`, `expectativas: ExpectativasFocus` e `indisponiveis: dict[str, str]`;
  - `montar_retrato(dados: DadosMacro, hoje: date) -> RetratoMacro`.

- [ ] **Step 1: Escrever o teste que falha**

Anexar a `tests/test_fatores_macro.py`:

```python
# ---------------------------------------------------------------------------
# Retrato de hoje
# ---------------------------------------------------------------------------

@pytest.fixture
def dados_hoje() -> DadosMacro:
    anual = _focus_anual([("2026-10-09", 2026, 13.25), ("2026-10-09", 2027, 12.00)])
    return DadosMacro(
        titulos=_titulos({"2026-10-09": (13.0, 12.5)}),
        focus_ipca_12m=pd.DataFrame({"data": pd.to_datetime(["2026-10-08", "2026-10-09"]),
                                     "mediana": [4.52, 4.48], "desvio": [0.49, 0.50]}),
        focus_selic=anual, focus_cambio=anual, focus_pib=anual,
        ptax=pd.Series([5.10, 4.99], index=pd.to_datetime(["2026-10-08", "2026-10-09"])),
        selic_meta=pd.Series([13.75], index=pd.to_datetime(["2026-10-13"])),
        ipca_12m=pd.Series([4.58], index=pd.to_datetime(["2026-09-01"])),
    )


def test_retrato_tem_oito_indicadores_com_data_e_fonte(dados_hoje):
    retrato = fatores_macro.montar_retrato(dados_hoje, date(2026, 10, 13))
    por_nome = {i.nome: i for i in retrato.indicadores}
    assert len(por_nome) == 8 and retrato.indisponiveis == {}
    assert por_nome["Selic meta"].valor == 13.75
    assert por_nome["Selic em 12 meses (Focus)"].valor == pytest.approx(0.25 * 13.25 + 0.75 * 12.00)
    assert por_nome["IPCA 12 meses à frente (Focus)"].valor == 4.48
    assert por_nome["Câmbio (PTAX venda)"].data == pd.Timestamp("2026-10-09")
    assert por_nome["Inclinação 5a − 1a"].valor == pytest.approx(-0.5)
    assert por_nome["IPCA acumulado em 12 meses"].dias_maximos > por_nome["Selic meta"].dias_maximos
    assert retrato.expectativas.ipca.desvio == 0.50


def test_retrato_registra_o_motivo_de_cada_indicador_que_falta(dados_hoje):
    dados = DadosMacro(**{**dados_hoje.__dict__, "ptax": None, "erros": {"ptax": "HTTP 500"},
                          "titulos": _titulos({"2026-10-09": (13.0, 12.5)}).iloc[:1]})
    retrato = fatores_macro.montar_retrato(dados, date(2026, 10, 13))
    assert retrato.indisponiveis["Câmbio (PTAX venda)"] == "HTTP 500"
    assert "1 e 5 anos" in retrato.indisponiveis["Inclinação 5a − 1a"]
    assert len(retrato.indicadores) == 6
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_fatores_macro.py -q`
Expected: 2 FAIL com `AttributeError: module 'core.fatores_macro' has no attribute 'montar_retrato'`; os 16 anteriores passam.

- [ ] **Step 3: Implementar**

(a) Em `core/fatores_macro.py`, trocar `from typing import TypeVar` por:

```python
from typing import NamedTuple, TypeVar
```

(b) Logo abaixo de `_ERROS_DE_FONTE = (...)`, acrescentar:

```python
_DIAS_FONTE_DIARIA = 7
_DIAS_FONTE_MENSAL = 75     # o IPCA do mês m sai por volta do dia 10 de m+1, datado do dia 1 de m
```

(c) Logo abaixo da classe `PainelMacro`, acrescentar:

```python
@dataclass(frozen=True)
class Indicador:
    nome: str
    valor: float
    unidade: str
    data: pd.Timestamp
    fonte: str
    dias_maximos: int      # idade acima da qual a interface avisa que o dado está velho


class ExpectativasFocus(NamedTuple):
    """Focus 12 meses à frente; None quando a fonte ou a pesquisa falhou."""

    selic: focus.Expectativa12m | None
    ipca: focus.Expectativa12m | None
    cambio: focus.Expectativa12m | None
    pib: focus.Expectativa12m | None


@dataclass(frozen=True)
class RetratoMacro:
    indicadores: tuple[Indicador, ...]
    expectativas: ExpectativasFocus
    indisponiveis: dict[str, str]          # nome do indicador -> motivo
```

(d) Anexar ao fim do arquivo:

```python
# ---------------------------------------------------------------------------
# Retrato de hoje (funções puras)
# ---------------------------------------------------------------------------

class _SemDado(Exception):
    """Indicador do retrato sem valor; a mensagem é o motivo."""


def _expectativas(dados: DadosMacro, hoje: date) -> tuple[ExpectativasFocus, dict[str, str]]:
    ate = pd.Timestamp(hoje)
    falhas: dict[str, str] = {}

    def anual(chave: str, tabela: pd.DataFrame | None) -> focus.Expectativa12m | None:
        if tabela is None:
            falhas[chave] = dados.erros.get(chave, "fonte sem dados")
            return None
        try:
            return focus.expectativa_12m(tabela, ate)
        except focus.ErroFocus as erro:
            falhas[chave] = str(erro)
            return None

    ipca = None
    if dados.focus_ipca_12m is None:
        falhas["focus_ipca"] = dados.erros.get("focus_ipca", "fonte sem dados")
    else:
        ultima = dados.focus_ipca_12m.iloc[-1]
        ipca = focus.Expectativa12m(float(ultima["mediana"]), float(ultima["desvio"]), ultima["data"])
    expectativas = ExpectativasFocus(selic=anual("focus_selic", dados.focus_selic), ipca=ipca,
                                     cambio=anual("focus_cambio", dados.focus_cambio),
                                     pib=anual("focus_pib", dados.focus_pib))
    return expectativas, falhas


def _ultimo(serie: pd.Series | None) -> tuple[float, pd.Timestamp]:
    if serie is None or serie.empty:
        raise _SemDado("fonte sem dados")
    return float(serie.iloc[-1]), pd.Timestamp(serie.index[-1])


def _da_expectativa(expectativa: focus.Expectativa12m | None) -> tuple[float, pd.Timestamp]:
    if expectativa is None:
        raise _SemDado("pesquisa Focus indisponível")
    return expectativa.mediana, expectativa.data


def _inclinacao(titulos: pd.DataFrame | None) -> tuple[float, pd.Timestamp]:
    if titulos is None:
        raise _SemDado("fonte sem dados")
    curva = tesouro.curva_mais_recente(titulos)
    inclinacao = curva.juros_5a - curva.juros_1a
    if np.isnan(inclinacao):
        raise _SemDado(f"prazos de 1 e 5 anos indisponíveis em {curva.data_base:%Y-%m-%d}")
    return float(inclinacao), curva.data_base


def montar_retrato(dados: DadosMacro, hoje: date) -> RetratoMacro:
    """Retrato macro de hoje, com a data e a fonte de cada número."""
    expectativas, falhas = _expectativas(dados, hoje)
    candidatos: list[tuple[str, str, str, str, int, Callable[[], tuple[float, pd.Timestamp]]]] = [
        # (nome, unidade, fonte exibida, chave em dados.erros, dias máximos, leitura)
        ("Selic meta", "% a.a.", "BCB SGS 432", "selic_meta", _DIAS_FONTE_DIARIA,
         lambda: _ultimo(dados.selic_meta)),
        ("Selic em 12 meses (Focus)", "% a.a.", "Focus", "focus_selic", _DIAS_FONTE_DIARIA,
         lambda: _da_expectativa(expectativas.selic)),
        ("IPCA acumulado em 12 meses", "%", "BCB SGS 13522", "ipca_12m", _DIAS_FONTE_MENSAL,
         lambda: _ultimo(dados.ipca_12m)),
        ("IPCA 12 meses à frente (Focus)", "%", "Focus", "focus_ipca", _DIAS_FONTE_DIARIA,
         lambda: _da_expectativa(expectativas.ipca)),
        ("Câmbio (PTAX venda)", "R$/US$", "BCB SGS 1", "ptax", _DIAS_FONTE_DIARIA,
         lambda: _ultimo(dados.ptax)),
        ("Câmbio em 12 meses (Focus)", "R$/US$", "Focus", "focus_cambio", _DIAS_FONTE_DIARIA,
         lambda: _da_expectativa(expectativas.cambio)),
        ("PIB 12 meses à frente (Focus)", "%", "Focus", "focus_pib", _DIAS_FONTE_DIARIA,
         lambda: _da_expectativa(expectativas.pib)),
        ("Inclinação 5a − 1a", "p.p.", "Tesouro Direto", "tesouro", _DIAS_FONTE_DIARIA,
         lambda: _inclinacao(dados.titulos)),
    ]
    indicadores, indisponiveis = [], {}
    for nome, unidade, fonte, chave, dias, ler in candidatos:
        try:
            valor, data = ler()
        except _SemDado as motivo:
            indisponiveis[nome] = dados.erros.get(chave) or falhas.get(chave) or str(motivo)
            continue
        indicadores.append(Indicador(nome, valor, unidade, data, fonte, dias))
    return RetratoMacro(tuple(indicadores), expectativas, indisponiveis)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_fatores_macro.py -q`
Expected: `18 passed`

- [ ] **Step 5: Commit**

```powershell
git add core/fatores_macro.py tests/test_fatores_macro.py
git commit -m "feat(core): retrato macro de hoje com data e fonte por indicador" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Conferência ao vivo (script manual) e suíte completa

**Files:**
- Create: `scripts/conferir_painel_macro.py`

**Interfaces:**
- Consumes: `baixar_dados_macro`, `montar_retrato` e `montar_painel` (Tarefas 4–7).
- Produces: um script que sai com código 0 se todos os fatores existirem com ≥ 200 meses, e 1 caso contrário.

- [ ] **Step 1: Criar o script**

Criar `scripts/conferir_painel_macro.py`:

```python
"""
Conferência manual, COM REDE, do painel macro e do retrato de hoje (spec §12, "fora da suíte").

Uso: .venv/Scripts/python scripts/conferir_painel_macro.py [AAAA-MM-DD]

Imprime o retrato, as fontes que falharam, a cobertura de cada fator e os
últimos meses do painel. Compare o retrato com o boletim Focus da semana
(https://www.bcb.gov.br/publicacoes/focus) e com o Tesouro Direto.
Sai com código 1 se algum fator ficar indisponível ou com menos de 200 meses.
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import fatores_macro  # noqa: E402

MESES_MINIMOS = 200      # 2005-12 a 2026-09 são 250 meses; buracos isolados são tolerados


def main() -> int:
    hoje = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    inicio = time.perf_counter()
    dados = fatores_macro.baixar_dados_macro(hoje)
    print(f"Download: {time.perf_counter() - inicio:.1f} s")
    for fonte, motivo in dados.erros.items():
        print(f"  FONTE COM FALHA  {fonte}: {motivo}")

    retrato = fatores_macro.montar_retrato(dados, hoje)
    print("\nRetrato de hoje")
    for ind in retrato.indicadores:
        print(f"  {ind.nome:<32} {ind.valor:>9.4f} {ind.unidade:<7} {ind.data:%Y-%m-%d}  ({ind.fonte})")
    for nome, motivo in retrato.indisponiveis.items():
        print(f"  {nome:<32} INDISPONÍVEL: {motivo}")
    for nome, exp in retrato.expectativas._asdict().items():
        if exp is not None:
            print(f"  Focus 12m {nome:<7} mediana {exp.mediana:.4f}  desvio {exp.desvio:.4f}  pesquisa {exp.data:%Y-%m-%d}")

    painel = fatores_macro.montar_painel(dados, hoje)
    fatores = painel.fatores
    print(f"\nPainel: {fatores.index.min()} a {fatores.index.max()} ({len(fatores)} meses)")
    print(fatores.describe().T[["count", "mean", "std", "min", "max"]].to_string(float_format="{:.4f}".format))
    print("\nÚltimos 6 meses")
    print(fatores.tail(6).to_string(float_format="{:.4f}".format))
    for aviso in painel.avisos:
        print(f"  AVISO {aviso}")

    curtos = [f for f in fatores.columns if fatores[f].notna().sum() < MESES_MINIMOS]
    if painel.fatores_indisponiveis or curtos:
        print(f"\nFALHOU: indisponíveis={list(painel.fatores_indisponiveis)}, curtos={curtos}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Rodar ao vivo (com rede)**

Run (PowerShell):
```powershell
$env:PYTHONIOENCODING = "utf-8"; .venv/Scripts/python scripts/conferir_painel_macro.py
```
Expected (referência da execução de 2026-10-13, com os mesmos dados):
- `Download:` entre 10 e 40 s, sem `FONTE COM FALHA`. Uma falha transitória do SGS é aceitável: rodar de novo.
- Os 8 indicadores do retrato aparecem. Com a pesquisa Focus de 2026-10-09: Selic em 12 meses 12,3125; IPCA 12m Focus 4,4772; câmbio 12m Focus 5,2334; PIB 12m 1,5116; inclinação ≈ −0,15 p.p.
- Painel de `2005-12` ao mês anterior a hoje (250 meses em 2026-10). `juros`/`premio` com ~241 meses (os meses sem título que cubra o prazo de 1 ano ficam NaN); os demais com 250.
- Última linha: `OK` (código de saída 0).

Se aparecer `FALHOU`, NÃO siga: registre a saída e investigue com `superpowers:systematic-debugging`.

- [ ] **Step 3: Suíte completa**

Run: `.venv/Scripts/python -m pytest tests -q`
Expected: `100 passed` (48 antigos + 17 Focus + 15 Tesouro + 2 preços mensais + 18 fatores)

- [ ] **Step 4: Revisão da tarefa e commit**

Rodar a revisão no formato do `code-reviewer` (🔴/🟠) sobre `git diff main...HEAD -- core scripts`. Corrigir os 🔴 antes do commit.

```powershell
git add scripts/conferir_painel_macro.py
git commit -m "chore(scripts): conferência ao vivo do painel macro" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Registrar no `session_state.md`**

Acrescentar no topo do `session_state.md` (fica fora do git): "Plano A concluído em AAAA-MM-DD", a saída resumida do script (download, painel, retrato) e "próximo: escrever o Plano B".

---

## Fora deste plano (vai para B ou C)

- **Plano B:**
  - modelo de fatores por ação: janela de 180 meses, mínimo de 36, Vasicek por setor e risco residual. O γ do fator 6 é estimado na mesma janela, com `estimar_gama`/`mercado_residual`;
  - cenários Focus, estresse e "meu cenário", plausibilidade e matriz;
  - `PisoCenario` no otimizador;
  - `projetar_horizonte` (§15), em que a entrada do CDI é a Selic **composta** (`selic.valor`) e não `rf_aritmetica`. O nome do parâmetro de risco de estimação não pode colidir com o `kappa` de Charnes-Cooper em `_max_sharpe`.
- **Plano C:**
  - `core/validacao.py` (Clark-West e Newey-West em numpy, sem statsmodels) e o script de validação, com o esquema JSON definido uma vez;
  - loaders em `ui/servicos.py`. Antes de escrever o C, testar se `persist="disk"` funciona com a chave `dia`;
  - aba "Cenários macro" e seletor de horizonte;
  - `requirements.txt`: faixas testadas + `pyarrow`.
