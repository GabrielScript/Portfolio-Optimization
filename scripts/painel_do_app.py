"""
The deployed app's news/sentiment and filings panel files, from the lake's forward log and CVM IPE filings on D:
(``core.sentimento``): ``lago/sentimento_diario.parquet`` (counts and Jev means per Brasília day, numbers only) and
``lago/fatos_relevantes.parquet`` (the material CVM filings of the app's stocks in the last ``DIAS_DE_FATOS`` days,
public CVM data, with Jev's tone and strength numbers of each scored filing for the per-stock light) and
``lago/fundamentos.parquet`` (multiples and 12-month dividend yield of each stock, from the lake's CVM statements).
No headline or summary text goes to git: the raw text stays on D: (DECISOES 123); the local app reads the headlines
from there. The daily upkeep (``scripts/manutencao.py``) runs this and commits the files.

Usage: .venv/Scripts/python -X utf8 scripts/painel_do_app.py [lake sentiment folder, default D:/b3-datalake/sentiment]
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

PASTA_DO_LOG = Path("D:/b3-datalake/sentiment")
EXPORTS_DO_LAGO = RAIZ / "data" / "b3lake"            # the lake's exports in the app checkout (not in git)
DESTINO_DIARIO = RAIZ / "lago" / "sentimento_diario.parquet"
DESTINO_FATOS = RAIZ / "lago" / "fatos_relevantes.parquet"
DESTINO_FUNDAMENTOS = RAIZ / "lago" / "fundamentos.parquet"
DIAS_DE_FATOS = 90
PASTA_DO_CADASTRO = Path("D:/b3-datalake/raw/b3_pesquisa/IN")   # B3 instruments registers (BVBG.028), monthly
CACHE = RAIZ / "data" / "fatos_bdr_etf"                         # downloads and the register's names (not in git)
URL_EVENTUAIS = "https://dados.cvm.gov.br/dados/FI/DOC/EVENTUAL/DADOS/eventual_fi_{ano}.csv"
INICIO_DO_LOG = date(2026, 10, 23)          # the log's first run (DECISOES 123)


def gerar(pasta: Path, exports: Path, tickers: list[str], hoje: date, nomes: pd.DataFrame | None = None,
          fatos_de_etfs: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The panel tables: the daily tone since the log began, the recent material filings of ``tickers`` (and of the
    sponsored BDRs of ``nomes``, plus ``fatos_de_etfs``) and the stocks' fundamentals
    (``core.indicadores.tabela_de_fundamentos``, from the lake's exports in ``exports``)."""
    from core.fatos_bdr_etf import codigos_dos_bdrs
    from core.indicadores import tabela_de_fundamentos
    from core.sentimento import com_notas_dos_fatos, fatos_recentes, ler_log, ler_notas_dos_fatos, tom_diario
    itens, notas = ler_log(pasta, INICIO_DO_LOG)
    diario = tom_diario(itens, notas).reset_index()
    ipe = pd.read_parquet(pasta / "ipe" / "ipe.parquet", columns=["codigo_cvm", "nome_companhia", "categoria",
                                                                  "assunto", "data_entrega", "link_download"])
    setores = pd.read_parquet(exports / "sectors.parquet", columns=["issuer", "code_cvm"])
    bdrs = None if nomes is None else nomes[nomes["asset_type"] == "bdr"]
    extras = {} if bdrs is None else codigos_dos_bdrs(bdrs, setores, ipe)
    fatos = fatos_recentes(ipe, setores, tickers, hoje - timedelta(days=DIAS_DE_FATOS), extras=extras)
    if fatos_de_etfs is not None and not fatos_de_etfs.empty:
        fatos = pd.concat([fatos, fatos_de_etfs], ignore_index=True).sort_values(
            ["data", "ticker"], ascending=[False, True], kind="stable").reset_index(drop=True)
    fatos = com_notas_dos_fatos(fatos, ler_notas_dos_fatos(pasta, INICIO_DO_LOG))
    fundamentos = pd.read_parquet(exports / "fundamentals.parquet",
                                  columns=["issuer", "account", "dt_refer", "value", "available_from", "sector"])
    for coluna in ("dt_refer", "available_from"):
        fundamentos[coluna] = pd.to_datetime(fundamentos[coluna])
    tabela = tabela_de_fundamentos(fundamentos, pd.read_parquet(exports / "market_caps.parquet",
                                                                columns=["issuer", "month", "market_cap"]),
                                   pd.read_parquet(exports / "dividends.parquet",
                                                   columns=["security", "ex_date", "yield"]), tickers, hoje)
    return diario, fatos, tabela


def tickers_do_painel(nomes: pd.DataFrame) -> list[str]:
    """The tickers that get fundamentals and filings: the app's 181, then every other stock and unit of the lake's
    names (``ativos_tela.acoes_da_tela``); never a BDR or ETF, whose prefix could match a B3 issuer."""
    from assets import get_all_tickers
    from ativos_tela import acoes_da_tela
    return list(dict.fromkeys((*get_all_tickers(), *acoes_da_tela(nomes))))


def nomes_dos_etfs(tickers: list[str], pasta: Path = PASTA_DO_CADASTRO, cache: Path = CACHE) -> dict[str, str]:
    """ETF ticker -> full fund name from the newest B3 instruments register on the lake's disk, cached per register
    (parsing its two XML files takes minutes)."""
    import tempfile
    import zipfile
    from core.fatos_bdr_etf import nomes_completos_do_cadastro
    registros = sorted(pasta.glob("*/IN*.zip"))
    if not registros:
        return {}
    ultimo, guardado = registros[-1], cache / "nomes_dos_etfs.parquet"
    if guardado.exists():
        tabela = pd.read_parquet(guardado)
        if not tabela.empty and tabela["registro"].iloc[0] == ultimo.name:
            return dict(zip(tabela["ticker"], tabela["nome"]))
    nomes: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as temporaria:
        with zipfile.ZipFile(ultimo) as externo:
            externo.extractall(temporaria)
        for interno in Path(temporaria).glob("*.zip"):
            with zipfile.ZipFile(interno) as arquivo:
                arquivo.extractall(temporaria)
        for xml in sorted(Path(temporaria).glob("*.xml")):
            nomes |= nomes_completos_do_cadastro(xml, set(tickers) - set(nomes))
    cache.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"registro": ultimo.name, "ticker": list(nomes), "nome": list(nomes.values())}).to_parquet(guardado)
    return nomes


def eventuais_de_fundos(hoje: date, cache: Path = CACHE) -> pd.DataFrame:
    """The CVM's fund eventual documents of this year and the last (the 90 days can cross a year), downloaded once a
    day into ``cache``."""
    import requests
    cache.mkdir(parents=True, exist_ok=True)
    partes = []
    for ano in (hoje.year - 1, hoje.year):
        arquivo = cache / f"eventual_fi_{ano}.csv"
        if not arquivo.exists() or date.fromtimestamp(arquivo.stat().st_mtime) < hoje:
            resposta = requests.get(URL_EVENTUAIS.format(ano=ano), timeout=300)
            resposta.raise_for_status()
            arquivo.write_bytes(resposta.content)
        partes.append(pd.read_csv(arquivo, sep=";", encoding="latin-1", dtype=str,
                                  usecols=["CNPJ_FUNDO_CLASSE", "DENOM_SOCIAL", "DT_RECEB", "TP_DOC", "LINK_ARQ"]))
    return pd.concat(partes, ignore_index=True)


def fatos_dos_etfs(nomes: pd.DataFrame, hoje: date) -> pd.DataFrame | None:
    """The ETFs' CVM material facts of the last ``DIAS_DE_FATOS`` days; None (and a line on stdout) when the register
    or the CVM download fails: the panel then keeps the stocks' and BDRs' facts."""
    from core.fatos_bdr_etf import cnpjs_dos_etfs, fatos_dos_fundos
    try:
        etfs = nomes_dos_etfs(list(nomes.loc[nomes["asset_type"] == "fund_etf", "security"]))
        eventuais = eventuais_de_fundos(hoje)
    except Exception as erro:       # noqa: BLE001 - network or disk: the panel goes on without the ETFs' facts
        print(f"Fatos dos ETFs indisponíveis agora: {erro}")
        return None
    return fatos_dos_fundos(eventuais, cnpjs_dos_etfs(etfs, eventuais), hoje - timedelta(days=DIAS_DE_FATOS))


def main() -> int:
    from ativos_tela import ler_nomes
    pasta = Path(sys.argv[1]) if len(sys.argv) > 1 else PASTA_DO_LOG
    nomes, hoje = ler_nomes(), date.today()
    diario, fatos, fundamentos = gerar(pasta, EXPORTS_DO_LAGO, tickers_do_painel(nomes), hoje, nomes,
                                       fatos_dos_etfs(nomes, hoje))
    diario.to_parquet(DESTINO_DIARIO)
    fatos.to_parquet(DESTINO_FATOS)
    fundamentos.to_parquet(DESTINO_FUNDAMENTOS)
    print(f"{DESTINO_DIARIO.relative_to(RAIZ)}: {len(diario)} dias; {DESTINO_FATOS.relative_to(RAIZ)}: {len(fatos)} "
          f"fatos relevantes e comunicados; {DESTINO_FUNDAMENTOS.relative_to(RAIZ)}: {len(fundamentos)} ações")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
