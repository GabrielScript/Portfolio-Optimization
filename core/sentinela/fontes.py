"""Source readers of the Sentinela (spec, "Sources", amendments 1-3 and 5). Parsers are pure; the thin readers take
their HTTP or file access as arguments, so a failing source drops only its own items (the run reports it).

- CVM: the app's filings table (``lago/fatos_relevantes.parquet``, refreshed by the nightly upkeep) and, when
  present, the daily RAD/ENET table (``core.sentinela.cvm_diaria``).
- Market: the lake's daily panel (total-return close, R$ traded) and the B3 BDI lending tables.
- BCB SGS: Selic target (only the days it changed), IPCA, IBC-Br, credit, default, fiscal 4649/5793/4513/13762,
  PTAX (only a move beyond 2.5 sigma), Focus (weekly), the last Copom minutes.
- IBGE SIDRA: GDP 5932 (q/q), unemployment 6381 (moving quarter), industry 8888 (y/y).
- Tesouro Nacional RTN, table 1.1 (central government, R$ million): the monthly publication's spreadsheet, found
  on its page; the page renders through Firecrawl when Tesouro's own server fails ("Falha na integração THOT/3PRO",
  seen 05/11/2026); then the CKAN copy (lags about two months). Tesouro's series API (apiapex) was down
  (HTTP 503) on 05/11/2026.
"""

from __future__ import annotations

import io
import re
import unicodedata
from collections.abc import Callable, Iterable
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from core.sentinela.itens import MACRO, Item, Serie, itens_aluguel, itens_cvm, itens_mercado

URL_CKAN_RTN = "https://www.tesourotransparente.gov.br/ckan/api/3/action/package_show?id=resultado-do-tesouro-nacional"
URL_PAGINA_RTN = "https://www.tesourotransparente.gov.br/publicacoes/boletim-resultado-do-tesouro-nacional-rtn/{ano}/{mes}"
URL_SIDRA = "https://apisidra.ibge.gov.br/values/"
URL_FIRECRAWL_SCRAPE = "https://api.firecrawl.dev/v2/scrape"
_ANEXO = re.compile(r"https://thot-arquivos\.tesouro\.gov\.br/publicacao-anexo/\d+")
MESES_DO_RTN = 3
PRAZO_S = 60
AGENTE = {"User-Agent": "Mozilla/5.0 (Aleph Quant Sentinela)"}

SIDRA = {   # name -> (path, label); amendment 2
    "pib": ("t/5932/n1/all/v/6564/p/last%206/c11255/90707", "PIB, variação contra o trimestre anterior (%), IBGE"),
    "desocupacao": ("t/6381/n1/all/v/4099/p/last%206", "Taxa de desocupação, trimestre móvel (%), IBGE PNAD"),
    "industria": ("t/8888/n1/all/v/11602/p/last%206/c544/129314",
                  "Produção industrial, variação contra o mesmo mês do ano anterior (%), IBGE PIM"),
}
SGS_MENSAIS = {   # code -> (name, label)
    433: ("ipca", "IPCA no mês (%), BCB SGS 433"),
    24363: ("ibc_br", "IBC-Br, índice (2002 = 100), BCB SGS 24363"),
    20631: ("concessoes", "Crédito concedido no mês (R$ milhões), BCB SGS 20631"),
    21082: ("inadimplencia", "Inadimplência do crédito (%), BCB SGS 21082"),
    4649: ("primario_setor_publico", "Resultado primário do setor público consolidado no mês (R$ milhões), BCB SGS 4649"),
    5793: ("primario_12m_pib", "Resultado primário do setor público em 12 meses (% do PIB), BCB SGS 5793"),
    4513: ("divida_liquida_pib", "Dívida líquida do setor público (% do PIB), BCB SGS 4513"),
    13762: ("divida_bruta_pib", "Dívida bruta do governo geral (% do PIB), BCB SGS 13762"),
}
SGS_SELIC_META, SGS_PTAX = 432, 1
RTN = {   # row prefix of table 1.1 (normalized) -> (name, label)
    "3. receita liquida": ("rtn_receita_liquida", "Receita líquida do Governo Central no mês (R$ milhões), Tesouro RTN"),
    "4. despesa total": ("rtn_despesa_total", "Despesa total do Governo Central no mês (R$ milhões), Tesouro RTN"),
    "5. resultado primario governo central - acima da linha": (
        "rtn_resultado_primario", "Resultado primário do Governo Central no mês (R$ milhões), Tesouro RTN"),
}


class FonteIndisponivel(Exception):
    """A source failed; the message says which (Portuguese, shown in the run's report)."""


def _normalizar(texto: str) -> str:
    sem_acento = "".join(letra for letra in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(letra))
    return re.sub(r"\s+", " ", sem_acento).strip().casefold()


# ---- IBGE SIDRA -------------------------------------------------------------------------------------------------

def serie_sidra(linhas: list[dict]) -> tuple[pd.Series, dict[pd.Timestamp, str]]:
    """The values of a SIDRA answer (first row = header), dated by period (a quarter on its first month), and the
    official period names. Values that are not numbers ("...", "-") are dropped."""
    cabecalho, dados = linhas[0], linhas[1:]
    codigo = next(chave for chave, nome in cabecalho.items()
                  if chave.endswith("C") and any(palavra in nome for palavra in ("Trimestre", "Mês")))
    trimestral = "Trimestre (" in cabecalho[codigo]
    valores, nomes = {}, {}
    for linha in dados:
        try:
            valor = float(linha["V"])
        except (TypeError, ValueError):
            continue
        ano, parte = int(linha[codigo][:4]), int(linha[codigo][4:])
        dia = pd.Timestamp(ano, 3 * (parte - 1) + 1 if trimestral else parte, 1)
        valores[dia] = valor
        nomes[dia] = linha[codigo[:-1] + "N"]
    return pd.Series(valores, dtype=float).sort_index(), nomes


def series_ibge(get: Callable) -> tuple[list[Serie], list[str]]:
    series, falhas = [], []
    for nome, (caminho, rotulo) in SIDRA.items():
        try:
            resposta = get(URL_SIDRA + caminho, timeout=PRAZO_S, headers=AGENTE)
            valores, nomes = serie_sidra(resposta.json())
            series.append(Serie(nome, "ibge", rotulo, valores, nomes=nomes))
        except Exception as erro:       # noqa: BLE001 - one failing table drops only its own item
            falhas.append(f"IBGE {nome}: {type(erro).__name__}")
    return series, falhas


# ---- Tesouro Nacional RTN ---------------------------------------------------------------------------------------

def ler_rtn(conteudo: bytes) -> dict[str, pd.Series]:
    """Table 1.1 of the RTN spreadsheet: net revenue, total spending and the primary result (R$ million, 1 decimal)."""
    tabela = pd.read_excel(io.BytesIO(conteudo), sheet_name="1.1", header=None)
    rotulos = tabela.iloc[:, 0].map(lambda valor: _normalizar(valor) if isinstance(valor, str) else "")
    linha_das_datas = int(rotulos[rotulos.str.startswith("discrimina")].index[0])
    datas = pd.to_datetime(tabela.iloc[linha_das_datas, 1:], errors="coerce")
    series = {}
    for prefixo, (nome, _) in RTN.items():
        achadas = rotulos[rotulos.str.startswith(prefixo)]
        if achadas.empty:
            continue
        valores = pd.to_numeric(tabela.iloc[int(achadas.index[0]), 1:], errors="coerce")
        serie = pd.Series(valores.to_numpy(), index=datas.to_numpy()).dropna()
        serie = serie[serie.index.notna()].round(1)
        series[nome] = serie.sort_index()
    return series


def link_do_rtn(texto: str) -> str | None:
    achado = _ANEXO.search(texto or "")
    return achado.group(0) if achado else None


def paginas_do_rtn(hoje: date) -> list[str]:
    """The RTN pages of the last months, newest first (month m's RTN comes out late in month m + 1)."""
    paginas, mes = [], pd.Timestamp(hoje).to_period("M")
    for passo in range(1, MESES_DO_RTN + 1):
        periodo = mes - passo
        paginas.append(URL_PAGINA_RTN.format(ano=periodo.year, mes=periodo.month))
    return paginas


def _planilha(get: Callable, url: str) -> bytes | None:
    resposta = get(url, timeout=PRAZO_S, headers=AGENTE)
    return resposta.content if resposta.ok and resposta.content[:2] == b"PK" else None


def baixar_rtn(hoje: date, get: Callable, renderizar: Callable[[str], str | None]) -> tuple[bytes, str]:
    """(spreadsheet, where it came from): each page by Tesouro's server, then rendered (Firecrawl); then CKAN."""
    for pagina in paginas_do_rtn(hoje):
        for origem, ler in (("tesouro", lambda url: get(url, timeout=PRAZO_S, headers=AGENTE)),
                            ("firecrawl", renderizar)):
            try:
                resposta = ler(pagina)
                texto = resposta if isinstance(resposta, str) or resposta is None else (
                    resposta.text if resposta.ok else None)
                link = link_do_rtn(texto or "")
                conteudo = _planilha(get, link) if link else None
            except Exception:           # noqa: BLE001 - try the next route
                conteudo = None
            if conteudo:
                return conteudo, f"{origem} {pagina}"
    try:
        recursos = get(URL_CKAN_RTN, timeout=PRAZO_S, headers=AGENTE).json()["result"]["resources"]
        for recurso in recursos:
            if str(recurso.get("format", "")).upper() == "XLSX" and "mensal" in _normalizar(recurso.get("name", "")):
                conteudo = _planilha(get, recurso["url"])
                if conteudo:
                    return conteudo, "ckan"
    except Exception:                   # noqa: BLE001
        pass
    raise FonteIndisponivel("Tesouro RTN: página, Firecrawl e CKAN indisponíveis")


def renderizador_firecrawl(chave: str | None, post: Callable) -> Callable[[str], str | None]:
    """A page's rendered markdown through Firecrawl (one scrape, the user's plan credits); None without a key."""
    def renderizar(url: str) -> str | None:
        if not chave:
            return None
        resposta = post(URL_FIRECRAWL_SCRAPE, json={"url": url, "formats": ["markdown"], "maxAge": 0},
                        timeout=PRAZO_S, headers={"Authorization": f"Bearer {chave}"})
        return ((resposta.json() or {}).get("data") or {}).get("markdown") if resposta.ok else None
    return renderizar


def series_tesouro(hoje: date, get: Callable, renderizar: Callable[[str], str | None], cache: Path,
                   ) -> tuple[list[Serie], list[str]]:
    """The RTN series; the spreadsheet is downloaded at most once a day (``cache``, outside the repository)."""
    try:
        if cache.exists() and date.fromtimestamp(cache.stat().st_mtime) == hoje:
            conteudo = cache.read_bytes()
        else:
            conteudo, _ = baixar_rtn(hoje, get, renderizar)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(conteudo)
        series = ler_rtn(conteudo)
    except Exception as erro:           # noqa: BLE001
        return [], [f"Tesouro RTN: {erro}" if isinstance(erro, FonteIndisponivel) else f"Tesouro RTN: {type(erro).__name__}"]
    return [Serie(nome, "tesouro", rotulo, series[nome]) for nome, rotulo in RTN.values() if nome in series], []


# ---- BCB --------------------------------------------------------------------------------------------------------

def so_mudancas(serie: pd.Series) -> pd.Series:
    """Only the days the value changed (the first value counts as a change)."""
    serie = serie.dropna()
    return serie[serie.ne(serie.shift())]


def semanal(serie: pd.Series) -> pd.Series:
    """The last value of each week (Monday to Sunday)."""
    serie = serie.dropna().sort_index()
    semanas = serie.index.to_period("W-SUN")
    return serie.groupby(semanas).last().set_axis(serie.groupby(semanas).apply(lambda parte: parte.index[-1]))


def series_bcb(baixar: Callable[[int, date, date], pd.Series], hoje: date) -> tuple[list[Serie], list[str]]:
    inicio = hoje - timedelta(days=500)
    series, falhas = [], []
    for codigo, (nome, rotulo) in SGS_MENSAIS.items():
        try:
            series.append(Serie(nome, "bcb", rotulo, baixar(codigo, inicio, hoje)))
        except Exception as erro:       # noqa: BLE001
            falhas.append(f"BCB SGS {codigo}: {type(erro).__name__}")
    try:
        selic = so_mudancas(baixar(SGS_SELIC_META, inicio, hoje))
        series.append(Serie("selic_meta", "bcb", "Meta da Selic definida pelo Copom (% a.a.), BCB SGS 432", selic,
                            mensal=False))
    except Exception as erro:           # noqa: BLE001
        falhas.append(f"BCB SGS 432: {type(erro).__name__}")
    return series, falhas


def itens_ptax(baixar: Callable[[int, date, date], pd.Series], hoje: date) -> tuple[list[Item], list[str]]:
    """A PTAX move beyond 2.5 sigma of its 63 sessions, as a macro item."""
    try:
        ptax = baixar(SGS_PTAX, hoje - timedelta(days=200), hoje).dropna()
    except Exception as erro:           # noqa: BLE001
        return [], [f"BCB PTAX: {type(erro).__name__}"]
    itens = itens_mercado(pd.DataFrame({"PTAX (R$/US$)": ptax}), pd.DataFrame())
    return [Item(item.id.replace("mercado-", "bcb-", 1), "bcb", item.data, MACRO, item.texto + ", BCB SGS 1")
            for item in itens], []


def serie_focus(anual: pd.DataFrame, ano: int, nome: str, rotulo: str) -> Serie:
    do_ano = anual[anual["ano_referencia"] == ano].set_index("data")["mediana"]
    return Serie(nome, "bcb", rotulo, semanal(do_ano), mensal=False)


def item_da_ata(ata) -> list[Item]:
    """The last Copom minutes, as a macro item (``core.macro_bcb.Ata``)."""
    if ata is None:
        return []
    texto = f"Banco Central publicou a ata do Copom em {ata.data:%d/%m/%Y}: {ata.titulo}"
    return [Item(f"bcb-ata-{ata.data:%Y%m%d}", "bcb", f"{ata.data:%Y-%m-%d}", MACRO, texto[:600])]


# ---- CVM and market ---------------------------------------------------------------------------------------------

def fatos_cvm(tabelas: Iterable[pd.DataFrame], tickers: Iterable[str], desde: date) -> list[Item]:
    """CVM filings of ``tickers`` since ``desde`` from every table given (weekly open data, daily RAD/ENET); the
    same filing in two tables keeps one item (same link, same id)."""
    vistos, itens = set(), []
    for tabela in tabelas:
        if tabela is None or tabela.empty:
            continue
        recentes = tabela[pd.to_datetime(tabela["data"]) >= pd.Timestamp(desde)]
        for item in itens_cvm(recentes, tickers):
            if item.id not in vistos:
                vistos.add(item.id)
                itens.append(item)
    return itens


def painel_de_mercado(arquivo: Path, tickers: list[str], sessoes: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(total-return closes, R$ traded), dates x tickers, the last ``sessoes`` sessions of the lake's daily panel."""
    painel = pd.read_parquet(arquivo, columns=["date", "security", "tr_close", "volume_brl"],
                             filters=[("security", "in", list(tickers))])
    painel["date"] = pd.to_datetime(painel["date"])
    fechamentos = painel.pivot(index="date", columns="security", values="tr_close").sort_index().iloc[-sessoes:]
    volumes = painel.pivot(index="date", columns="security", values="volume_brl").sort_index().iloc[-sessoes:]
    return fechamentos.astype(float), volumes.astype(float)


def aluguel(pasta: Path, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(shares lent out, the borrower's average rate), dates x tickers, from the lake's BDI tables: open position
    market "Total" and the loan balance's "Registro" market (the largest)."""
    posicoes = pd.read_parquet(pasta / "bdi_BTBLendingOpenPosition.parquet",
                               columns=["date", "TckrSymb", "Market", "StockBalance"])
    posicoes = posicoes[(posicoes["Market"] == "Total") & posicoes["TckrSymb"].isin(tickers)]
    saldos = posicoes.pivot_table(index="date", columns="TckrSymb", values="StockBalance", aggfunc="last")
    taxas = pd.read_parquet(pasta / "bdi_BTBLoanBalance.parquet", columns=["date", "TckrSymb", "Market", "TkrAvrgRate"])
    taxas = taxas[(taxas["Market"] == "Registro") & taxas["TckrSymb"].isin(tickers)]
    taxas = taxas.pivot_table(index="date", columns="TckrSymb", values="TkrAvrgRate", aggfunc="last")
    for quadro in (saldos, taxas):
        quadro.index = pd.to_datetime(quadro.index)
    return saldos.sort_index(), taxas.sort_index()


def itens_de_mercado(arquivo_painel: Path, pasta_bdi: Path, tickers: list[str]) -> tuple[list[Item], list[str],
                                                                                       pd.DataFrame]:
    itens, falhas, fechamentos = [], [], pd.DataFrame()
    try:
        fechamentos, volumes = painel_de_mercado(arquivo_painel, tickers)
        itens += itens_mercado(fechamentos, volumes)
    except Exception as erro:           # noqa: BLE001
        falhas.append(f"preços do lago: {type(erro).__name__}")
    try:
        itens += itens_aluguel(*aluguel(pasta_bdi, tickers))
    except Exception as erro:           # noqa: BLE001
        falhas.append(f"BDI aluguel: {type(erro).__name__}")
    return itens, falhas, fechamentos
