"""Pipeline entry point: python -m b3lake.cli <step> [options]."""
import argparse
import datetime
import logging
import os
import shutil
import time
from pathlib import Path

import polars as pl
import requests

from b3lake import (b3events, bdi, bronze, cdi, ch, corrections, cvm_audit, cvm_cda, cvm_inf_diario, daily, derivatives,
                    download, edgar, export, focus, fundamentals, gold, index_portfolios, indices, ipca, migrations,
                    opcoesnet, option_chain, option_greeks, option_signals, ptax, sectors, tesouro, universe, usdata)
from b3lake.launch import LAUNCHES_CSV, STATE_DIR

LAKE_DIR = Path(os.environ.get("LAKE_DIR", "D:/b3-datalake"))
RAW_COTAHIST = LAKE_DIR / "raw" / "cotahist"
BRONZE_COTAHIST = LAKE_DIR / "bronze" / "cotahist"
MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
SQL_DIR = Path(__file__).resolve().parents[2] / "sql"


def _years(args) -> range:
    last = args.to or datetime.date.today().year
    return range(args.since, last + 1)


def cmd_download(args) -> None:
    RAW_COTAHIST.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        for year in _years(args):
            path = download.fetch_year(year, RAW_COTAHIST, session)
            logging.info("%s ok (%.1f MB)", path.name, path.stat().st_size / 1e6)


def cmd_bronze(args) -> None:
    """Rebuild the bronze partition of every year whose zip changed (by sha256)."""
    BRONZE_COTAHIST.mkdir(parents=True, exist_ok=True)
    manifest = bronze.read_manifest(BRONZE_COTAHIST)
    for year in _years(args):
        zip_path = RAW_COTAHIST / f"COTAHIST_A{year}.ZIP"
        if not zip_path.exists():
            logging.warning("%s missing; run download first", zip_path.name)
            continue
        known = manifest.get(year)
        if known and not args.force and known.zip_sha256 == bronze.sha256_of(zip_path):
            continue
        manifest[year] = bronze.build_year(zip_path, year, BRONZE_COTAHIST)
        bronze.write_manifest(BRONZE_COTAHIST, manifest)  # after each year: resumable
        logging.info("bronze %d: %d rows", year, manifest[year].rows)


def cmd_migrate(args) -> None:
    done = migrations.apply_all(ch.client(), MIGRATIONS_DIR)
    logging.info("migrations applied now: %s", done or "none (up to date)")


def cmd_load(args) -> None:
    """Load bronze years into ClickHouse (atomic REPLACE PARTITION), skipping unchanged ones."""
    client = ch.client()
    manifest = bronze.read_manifest(BRONZE_COTAHIST)
    loaded = {r[0]: r[1] for r in client.query("SELECT year, argMax(zip_sha256, loaded_at) FROM b3.load_log GROUP BY year").result_rows}
    for year in _years(args):
        entry = manifest.get(year)
        if entry is None or (loaded.get(year) == entry.zip_sha256 and not args.force):
            continue
        client.command("TRUNCATE TABLE b3.cotahist_staging")
        client.command(
            f"INSERT INTO b3.cotahist_staging SELECT {year} AS year, * "
            f"FROM file('lake/bronze/cotahist/year={year}/data.parquet', Parquet) "
            "SETTINGS use_hive_partitioning = 0"
        )
        rows = client.query("SELECT count() FROM b3.cotahist_staging").result_rows[0][0]
        if rows != entry.rows:
            raise RuntimeError(f"{year}: staging has {rows} rows, bronze manifest says {entry.rows}")
        client.command(f"ALTER TABLE b3.cotahist REPLACE PARTITION {year} FROM b3.cotahist_staging")
        client.insert("b3.load_log", [[year, entry.zip_sha256, rows]], column_names=["year", "zip_sha256", "rows"])
        logging.info("loaded %d: %d rows", year, rows)
    client.command("TRUNCATE TABLE b3.cotahist_staging")  # don't keep a stale copy on the SSD


def _run_sql_file(client, path: Path, **params) -> None:
    """Run a SQL file statement by statement, substituting {name} placeholders (no str.format: SQL has braces)."""
    sql = path.read_text(encoding="utf-8")
    for name, value in params.items():
        sql = sql.replace("{" + name + "}", str(value))
    for statement in migrations._statements(sql):
        client.command(statement)


def cmd_silver(args) -> None:
    """Rebuild silver tables year by year (staging + atomic REPLACE PARTITION)."""
    client = ch.client()
    tables = ("equity_daily", "option_daily")
    for year in _years(args):
        for table in tables:
            client.command(f"TRUNCATE TABLE b3.{table}_staging")
        _run_sql_file(client, SQL_DIR / "silver_year.sql", year=year)
        for table in tables:
            client.command(f"ALTER TABLE b3.{table} REPLACE PARTITION {year} FROM b3.{table}_staging")
        logging.info("silver %d done", year)
    for table in tables:
        client.command(f"TRUNCATE TABLE b3.{table}_staging")


def cmd_events(args) -> None:
    """Collect B3 corporate events (raw JSON per issuer).

    Default: every issuer that ever traded spot and has no file yet (the cemetery included).
    --refresh-active: re-fetch issuers that traded in the last 15 days, so new dividends, JCP and
    splits of the current year reach the lake (old files of dead companies never change).
    """
    client = ch.client()
    out_dir = LAKE_DIR / "raw" / "b3_events"
    all_rows = client.query(
        "SELECT DISTINCT issuer FROM b3.equity_daily WHERE market_type = 10 AND match(issuer, '^[A-Z0-9]{4}$') ORDER BY issuer"
    ).result_rows
    b3events.collect([r[0] for r in all_rows], out_dir, refresh=args.refresh)
    if args.refresh_active:
        active = client.query(
            "SELECT DISTINCT issuer FROM b3.equity_daily WHERE market_type = 10 AND match(issuer, '^[A-Z0-9]{4}$') "
            "AND date >= (SELECT max(date) FROM b3.equity_daily) - 15 ORDER BY issuer"
        ).result_rows
        logging.info("refreshing events of %d active issuers", len(active))
        b3events.collect([r[0] for r in active], out_dir, refresh=True)


def cmd_gold(args) -> None:
    """Adjusted survivorship-free series + event tables (1994-07-01 ->)."""
    client = ch.client()
    result = gold.build_gold(client, LAKE_DIR / "raw" / "b3_events", LAKE_DIR / "raw" / "cvm")
    gold.write_gold(client, result, LAKE_DIR / "gold")
    logging.info("gold written: %d rows", result["adjusted"].height)


def cmd_metrics(args) -> None:
    """Daily/monthly market indicators and the securities (cemetery) table."""
    _run_sql_file(ch.client(), SQL_DIR / "metrics.sql")
    logging.info("metrics rebuilt")


APP_DIR = Path(__file__).resolve().parents[3] / "ai-hedge-fund" / "Portfolio-Optimization"
APP_DATA_DIR = APP_DIR / "data" / "b3lake"  # ignored by the app's git
APP_PRICES_PATH = APP_DIR / "lago" / "precos_diarios.parquet"  # versioned: the app in production runs on it


def _copy_to_app(*names: str) -> None:
    """Copy exports from the lake disk into the app's ignored data folder (created when missing). Only for commands
    typed by hand: the nightly's `_build_*` steps never call it, the app receives their files from `publish`, after
    the gates."""
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(LAKE_DIR / "exports" / name, APP_DATA_DIR / name)


def cmd_export(args) -> None:
    """Small parquet exports on the lake disk; --publish also copies them into the Streamlit app."""
    out_dir = LAKE_DIR / "exports"
    summary = export.build_exports(ch.client(), out_dir)
    total = sum(v["mb"] for v in summary.values())
    logging.info("exports: %.1f MB in %s", total, out_dir)
    if args.publish:
        export.publish(out_dir, APP_DATA_DIR, APP_PRICES_PATH)
        logging.info("copied to %s and %s (not committed)", APP_DATA_DIR, APP_PRICES_PATH)


def _build_universe() -> None:
    """Point-in-time universe (rule D1) and the daily prices of every security that was ever in it: exports on D:
    only (the daily routine publishes them with the rest, after the gates)."""
    ipca_series = ipca.refresh_ipca(LAKE_DIR / "raw" / "bcb" / "ipca_433.json", ipca.fetch_sgs_433)
    client = ch.client()
    members = universe.universe_from_lake(client, ipca_series)
    prices = export.universe_prices(client, members).with_columns(pl.exclude("date").cast(pl.Float32))
    out_dir = LAKE_DIR / "exports"
    members.write_parquet(out_dir / export.UNIVERSE_FILE, compression="zstd")
    prices.write_parquet(out_dir / export.UNIVERSE_PRICES_FILE, compression="zstd", compression_level=10)
    logging.info("universe: %d security-months, %d securities priced (%.1f MB), IPCA up to %s",
                 members.height, prices.width - 1, (out_dir / export.UNIVERSE_PRICES_FILE).stat().st_size / 1e6,
                 ipca_series["month"].max())


def cmd_universe(args) -> None:
    """The universe exports (`_build_universe`) and a copy in the app's ignored data folder (not versioned)."""
    _build_universe()
    _copy_to_app(export.UNIVERSE_FILE, export.UNIVERSE_PRICES_FILE)


def _build_cdi() -> None:
    """Daily CDI (SGS 12) brought up to date on D: and exported (the daily routine publishes it after the gates)."""
    series = cdi.export_cdi(LAKE_DIR / "raw" / "bcb" / "cdi_12.json", LAKE_DIR / "exports", cdi.fetch_sgs_12,
                            datetime.date.today())
    logging.info("cdi: %d days, %s to %s", series.height, series["date"].min(), series["date"].max())


def _build_tesouro() -> None:
    """Tesouro Direto workbooks brought up to date on D: (past years once, the current year every run), then the daily
    quotes, the real curve and the zero-coupon returns exported (the daily routine publishes them after the gates)."""
    daily_quotes = tesouro.export_tesouro(LAKE_DIR / "raw" / "tesouro", LAKE_DIR / "exports", tesouro.fetch_workbook,
                                          datetime.date.today())
    logging.info("tesouro: %d quotes, %s to %s", daily_quotes.height, daily_quotes["date"].min(),
                 daily_quotes["date"].max())


def _build_focus() -> None:
    """BCB Focus expectations (annual IPCA, Selic, GDP, FX; 12-month IPCA) brought up to date on D: and exported with
    the day each row became public (the daily routine publishes them after the gates)."""
    focus.export_focus(LAKE_DIR / "raw" / "bcb", LAKE_DIR / "exports", focus.fetch_annual, focus.fetch_ipca12m)
    logging.info("focus: exported")


def cmd_focus(args) -> None:
    """The Focus exports (`_build_focus`); they reach the app with the nightly publish."""
    _build_focus()


def cmd_tesouro(args) -> None:
    """The Tesouro exports (`_build_tesouro`); they reach the app with the nightly publish."""
    _build_tesouro()


def _build_indices() -> None:
    """B3's official index levels (`b3lake.indices`: the statistics API's years since 1994 and the BVBG.087 Index
    Report of every session since 2017) on D:, exported (the daily routine publishes them after the gates).
    Incremental: past years and fetched reports are not asked again."""
    daily_levels = indices.export_indices(LAKE_DIR / "raw" / "b3_indices", LAKE_DIR / "exports", indices.fetch_year,
                                          requests.get, datetime.date.today())
    logging.info("indices: %d rows, %d indices, %s to %s", daily_levels.height, daily_levels["index"].n_unique(),
                 daily_levels["date"].min(), daily_levels["date"].max())


def _build_index_portfolios() -> None:
    """Today's portfolio and next-quarter preview of every index (`b3lake.index_portfolios`): B3 shows only today's,
    so the history is what the nightly keeps; exported with every kept day (the daily routine publishes it after the
    gates)."""
    raw = LAKE_DIR / "raw" / "b3_indices"
    with_stocks = [index for index in indices.INDICES if index != "VXBR"]           # a volatility index holds none
    index_portfolios.snapshot(raw, with_stocks, index_portfolios.fetch_page, datetime.date.today())
    joined = index_portfolios.export_portfolios(raw, LAKE_DIR / "exports")
    logging.info("index portfolios: %d rows, %d days", joined.height, joined["date"].n_unique())


def _build_bdi() -> None:
    """B3's daily bulletin tables (`b3lake.bdi`: lending, investor participation, open positions, forward, IOPV,
    margin scenarios, credited dividends...): the sessions of the last five weeks not kept yet, then one export per
    table on D: (research data, not copied to the app)."""
    sessions = bdi.recent_sessions(datetime.date.today(), bdi.get_json)
    written = bdi.collect(LAKE_DIR / "raw", bdi.TABLES, sessions, bdi.fetch_page)
    counts = bdi.export_bdi(LAKE_DIR / "raw", LAKE_DIR / "exports")
    logging.info("bdi: %d new table-days, %d sessions asked; rows per table %s", written, len(sessions), counts)


def cmd_bdi(args) -> None:
    _build_bdi()


def _build_option_chain() -> None:
    """The last 30 sessions' option chains of the liquid underlyings (`b3lake.option_chain`) for the app's simulator
    options sleeve (the local upkeep reads the copy the daily routine publishes after the gates)."""
    chain = option_chain.export_option_chain(ch.client(), LAKE_DIR / "exports")
    logging.info("option chain: %d rows, %d underlyings, %s to %s", chain.height, chain["underlying"].n_unique(),
                 chain["date"].min(), chain["date"].max())


def cmd_option_chain(args) -> None:
    _build_option_chain()
    _copy_to_app(option_chain.EXPORT_FILE)


def cmd_option_signals(args) -> None:
    """Monthly option signals per chain (`b3lake.option_signals`, the options veto's section 2.4) from the forward-based
    greeks of every year (run `greeks` first), exported and copied to the app's ignored data folder."""
    client = ch.client()
    sessions = option_greeks.sessions_from_lake(client)
    years = range(args.since, (args.to or datetime.date.today().year) + 1)
    signals = option_signals.export_signals(client, LAKE_DIR / "exports", years, sessions)
    logging.info("option signals: %d chain-months, %d chains, %s to %s", signals.height,
                 signals["underlying_isin"].n_unique(), signals["month"].min(), signals["month"].max())
    _copy_to_app(option_signals.EXPORT_FILE)


def cmd_indices(args) -> None:
    _build_indices()
    _copy_to_app(indices.DAILY_FILE, indices.REPORT_FILE)


def cmd_index_portfolios(args) -> None:
    _build_index_portfolios()
    _copy_to_app(index_portfolios.EXPORT_FILE)


def _build_ptax() -> None:
    """Daily PTAX (BCB SGS 1) brought up to date on D: and exported (the daily routine publishes it after the
    gates)."""
    series = ptax.export_ptax(LAKE_DIR / "raw" / "bcb" / "ptax_1.json", LAKE_DIR / "exports", ptax.fetch_sgs_1,
                              datetime.date.today())
    logging.info("ptax: %d days, %s to %s", series.height, series["date"].min(), series["date"].max())


def cmd_ptax(args) -> None:
    """The PTAX export (`_build_ptax`) and a copy in the app's ignored data folder (not versioned)."""
    _build_ptax()
    _copy_to_app(ptax.EXPORT_FILE)


def cmd_cda(args) -> None:
    """CVM CDA stock positions and net asset values of every fund (research export on the lake disk only)."""
    positions, navs = cvm_cda.export_cda(LAKE_DIR / "raw" / "cvm_cda", LAKE_DIR / "exports", datetime.date.today())
    logging.info("cda: %d positions, %d navs", positions, navs)


def cmd_derivatives(args) -> None:
    """B3 Price Reports (futures, options, open interest) of every session from 2018 into b3.derivative_daily, then
    the instruments files: monthly snapshots and up to --gap-files days for instruments still without terms."""
    client = ch.client()
    sessions = [d for d in option_greeks.sessions_from_lake(client) if d >= derivatives.FIRST_DAY]
    raw, today = LAKE_DIR / "raw" / "b3_pesquisa", datetime.date.today()
    rows = derivatives.load_price_reports(client, raw, sessions, requests.get, today, args.workers)
    terms = derivatives.load_instruments(client, raw, derivatives.snapshot_days(sessions), requests.get, today,
                                         args.workers)
    gaps = derivatives.load_instruments(client, raw, derivatives.missing_instrument_days(client, args.gap_files),
                                        requests.get, today, args.workers)
    logging.info("derivatives: %d price rows, %d + %d instrument rows", rows, terms, gaps)


def cmd_opcoesnet(args) -> None:
    """Today's opcoes.net.br chains (every underlying, or --tickers) saved on D: and loaded into b3.opcoesnet_daily,
    then the gap between the site's IV and ours on the series that traded on the same day (run `greeks` first)."""
    client, today = ch.client(), datetime.date.today()
    out_dir = LAKE_DIR / "raw" / "opcoesnet"
    opcoesnet.collect(out_dir, out_dir / "auth.json", today, args.tickers, deadline=getattr(args, "deadline", None))
    logging.info("opcoes.net: %d series loaded", opcoesnet.load_day(client, out_dir / f"{today:%Y-%m-%d}", today))
    for row in client.query(opcoesnet.CROSS_CHECK_SQL, parameters={"day": today}).named_results():
        logging.info("opcoes.net vs COTAHIST greeks: %s", row)


def cmd_greeks(args) -> None:
    """Implied volatility and greeks of every traded option-day into b3.option_greeks, year by year (the CDI export
    must be current: run `cdi` first)."""
    client = ch.client()
    rates = pl.read_parquet(LAKE_DIR / "exports" / cdi.EXPORT_FILE)
    sessions = option_greeks.sessions_from_lake(client)
    for year in range(max(args.since, option_greeks.FIRST_DAY.year), (args.to or datetime.date.today().year) + 1):
        logging.info("greeks %d: %d option-days", year, option_greeks.build_year(client, year, rates, sessions))


def cmd_inf_diario(args) -> None:
    """CVM daily fund report, one row per fund and month (research export on the lake disk only)."""
    rows = cvm_inf_diario.export_inf_diario(LAKE_DIR / "raw" / "cvm_inf_diario", LAKE_DIR / "exports",
                                            datetime.date.today())
    logging.info("inf_diario: %d fund-months", rows)


def cmd_cdi(args) -> None:
    """The CDI export (`_build_cdi`) and a copy in the app's ignored data folder (not versioned)."""
    _build_cdi()
    _copy_to_app(cdi.EXPORT_FILE)


def _cvm_registry() -> Path:
    """CVM's registry of listed companies (cancelled registrations included), fetched once."""
    cad = LAKE_DIR / "raw" / "cvm" / "cad_cia_aberta.csv"
    if not cad.exists():
        response = requests.get("https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv", timeout=120)
        response.raise_for_status()
        cad.write_bytes(response.content)
    return cad


def cmd_audit_cvm(args) -> None:
    """Compare applied share events with CVM FRE (companies' own filings, 2010+)."""
    cvm_dir = LAKE_DIR / "raw" / "cvm"
    cad = _cvm_registry()
    fre = cvm_audit.net_same_day(
        cvm_audit.load_fre_events(cvm_dir / "fre")
        .join(cvm_audit.cnpj_to_issuer(cad, LAKE_DIR / "raw" / "b3_events"), on="cnpj", how="inner")
        .select("issuer", "share_kind", "approved_on", "fre_multiplier")
    )  # same netting/dedupe as the gold step, so the audit compares like with like
    events = pl.read_parquet(LAKE_DIR / "gold" / "events.parquet").filter(pl.col("confirmed"))
    number = pl.col("security").str.extract(gold.MODERN_TICKER, 1).cast(pl.Int32, strict=False)
    ours = events.with_columns(
        issuer=pl.col("security").str.slice(0, 4),
        share_kind=pl.when(number == 3).then(pl.lit("ON")).when(number.is_between(4, 8)).then(pl.lit("PN")),
    ).filter(pl.col("share_kind").is_not_null()).select("issuer", "share_kind", "ex_date", "applied_multiplier")
    result = cvm_audit.classify_against_fre(fre, ours)
    result.write_parquet(LAKE_DIR / "exports" / "audit_fre.parquet")
    logging.info("FRE audit: %s", dict(result.group_by("status").len().iter_rows()))


def cmd_sectors(args) -> None:
    """The sectors export (`_build_sectors`) and a copy in the app's ignored data folder."""
    _build_sectors()
    _copy_to_app(sectors.SECTORS_FILE)


def _build_sectors() -> None:
    """Sector of each issuer (CVM registry mapped onto the app's sectors), with its coverage of the point-in-time
    universe printed before any use: export on D: (the daily routine publishes it after the gates)."""
    out_dir = LAKE_DIR / "exports"
    universe_path = out_dir / export.UNIVERSE_FILE
    if not universe_path.exists():
        raise FileNotFoundError(f"{universe_path} missing: run `universe` first (coverage is measured against it)")
    result = sectors.issuer_sectors(_cvm_registry(), LAKE_DIR / "raw" / "b3_events", sectors.SECTOR_MAP_CSV,
                                    sectors.ISSUER_CVM_CSV)
    members = pl.read_parquet(universe_path)
    recent = members.filter(pl.col("month") >= pl.lit(export.APP_PRICES_FROM).str.to_date())
    for label, frame in (("universe", members), (f"since {export.APP_PRICES_FROM}", recent)):
        covered, total = sectors.coverage(result, frame)
        logging.info("sectors, %s: %d of %d issuers (%.1f%%)", label, covered, total, 100 * covered / total)
    missing = sectors.unmapped(members.select("issuer").unique().join(result, on="issuer", how="left"))
    logging.info("without sector: %s", ", ".join(f"{i} ({s})" for i, s in missing.iter_rows()))
    result.write_parquet(out_dir / sectors.SECTORS_FILE, compression="zstd")


def cmd_fundamentals(args) -> None:
    """The fundamentals exports (`_build_fundamentals`) and a copy in the app's ignored data folder."""
    _build_fundamentals()
    _copy_to_app(fundamentals.FUNDAMENTALS_FILE, fundamentals.MARKET_CAPS_FILE)


def _last_caps_month() -> datetime.date | None:
    """Last month of the market-values export on D:, None before the first `fundamentals`."""
    path = LAKE_DIR / "exports" / fundamentals.MARKET_CAPS_FILE
    if not path.exists():
        return None
    return pl.scan_parquet(path).select(pl.col("month").max()).collect().item()


def _build_fundamentals() -> None:
    """CVM statements point-in-time (DFP 2010+, ITR 2011+): the curated accounts of each document from its receipt date
    and the market value of each issuer at the start of each month, exports on D: (the daily routine runs it when
    `daily.fundamentals_due` and publishes them after the gates). Reads the sectors export and gold's parquet."""
    out_dir = LAKE_DIR / "exports"
    issuers = pl.read_parquet(out_dir / sectors.SECTORS_FILE).filter(pl.col("code_cvm").is_not_null())
    zips = {kind: [] for kind in fundamentals.FIRST_YEAR}
    with requests.Session() as session:
        for kind, first in fundamentals.FIRST_YEAR.items():
            folder = LAKE_DIR / "raw" / "cvm" / kind
            folder.mkdir(parents=True, exist_ok=True)
            for year in range(first, datetime.date.today().year + 1):
                try:
                    zips[kind].append(fundamentals.fetch_cvm_year(kind, year, folder, session))
                except download.YearNotPublished as e:
                    logging.info("%s", e)
    documents = zips["dfp"] + zips["itr"]
    codes = [code for _, code, _ in fundamentals.ACCOUNTS.values()]
    statements = pl.concat([fundamentals.read_statements(path).filter(pl.col("account").is_in(codes))
                            for path in documents], how="vertical_relaxed")
    accounts = (fundamentals.curated_accounts(statements)
                .join(issuers.select("issuer", "code_cvm", "sector"), on="code_cvm", how="inner"))
    # the FRE zips are read as they are: `gold` uses them for event factors, so they are refreshed by hand, yearly
    fre = sorted((LAKE_DIR / "raw" / "cvm" / "fre").glob("fre_cia_aberta_*.zip"))
    quotes = (pl.scan_parquet(LAKE_DIR / "gold" / "equity_adjusted.parquet")
              .filter(pl.col("asset_type") == "stock", pl.col("date") >= datetime.date(2009, 1, 1))
              .select("date", "security", "issuer", "share_class", "close", "split_factor", "volume_brl").collect())
    factors = quotes.group_by("issuer", "date").agg(pl.col("split_factor").median())
    shares = fundamentals.fix_share_scale(
        fundamentals.share_counts(pl.concat([fundamentals.read_shares(path) for path in documents]),
                                  pl.concat([fundamentals.read_fre_shares(path) for path in fre])),
        issuers, factors, fundamentals.b3_share_counts(LAKE_DIR / "raw" / "b3_events"))
    logging.info("share counts rescaled: %s", dict(shares.group_by("scale").len().iter_rows()))
    today = datetime.date.today()
    months = pl.date_range(datetime.date(2010, 2, 1), today.replace(day=1), "1mo", eager=True).to_list()
    # banks file under other plans (their 2.03 is not the equity): the check uses only their total assets
    caps = fundamentals.plausible_market_caps(fundamentals.market_caps(shares, issuers, quotes, months), accounts)
    logging.info("market values rescaled by price-to-book or price-to-assets: %s", dict(caps.group_by("book_scale").len().iter_rows()))
    accounts.write_parquet(out_dir / fundamentals.FUNDAMENTALS_FILE, compression="zstd")
    caps.write_parquet(out_dir / fundamentals.MARKET_CAPS_FILE, compression="zstd")
    members = pl.read_parquet(out_dir / export.UNIVERSE_FILE)
    for label, frame in (("ebit", accounts.filter(pl.col("account") == "ebit").select("issuer")),
                         ("market cap", caps.select("issuer"))):
        recent = members.filter(pl.col("month") >= datetime.date(2013, 1, 1)).select("issuer").unique()
        covered = recent.join(frame.unique(), on="issuer", how="semi").height
        logging.info("fundamentals, %s: %d of %d universe issuers since 2013", label, covered, recent.height)
    logging.info("fundamentals: %d account rows of %d issuers, %d issuer-months of market value", accounts.height,
                 accounts["issuer"].n_unique(), caps.height)




def _lastmod(year: int) -> str | None:
    stamp = RAW_COTAHIST / f"COTAHIST_A{year}.lastmod"
    return stamp.read_text().strip() if stamp.exists() else None


class _LakeSteps:
    """daily.Lake over the commands above, all in this process (one log per run)."""

    def __init__(self) -> None:
        self.notes: list[str] = []  # goes to the run's line in runs.csv

    def ready(self) -> str | None:
        if not LAKE_DIR.is_dir():
            return "D:"
        return daily.ensure_clickhouse(daily.clickhouse_answers, daily.start_clickhouse_container)

    def refresh_bronze(self, years: range) -> dict[int, str]:
        span = argparse.Namespace(since=years.start, to=years.stop - 1, force=False)
        before = {year: _lastmod(year) for year in years}
        cmd_download(span)
        # an unchanged stamp means "not published yet" or "B3 did not answer the HEAD" (download keeps its copy)
        changed = [year for year in years if _lastmod(year) != before[year]]
        self.notes.append(f"new zip: {changed or 'none'}")
        cmd_bronze(span)
        manifest = bronze.read_manifest(BRONZE_COTAHIST)
        return {year: manifest[year].zip_sha256 for year in years if year in manifest}

    def rebuild_silver(self, years: list[int]) -> None:
        for year in years:
            span = argparse.Namespace(since=year, to=year, force=False)
            cmd_load(span)
            cmd_silver(span)

    def last_session(self) -> datetime.date:
        return ch.client().query("SELECT max(date) FROM b3.equity_daily").result_rows[0][0]

    def refresh_fundamentals(self, weekly: bool, through: datetime.date) -> None:
        """Sectors, CVM statements and market values when `daily.fundamentals_due`; a failure (CVM down) keeps the
        files of the last run and leaves a note."""
        if not daily.fundamentals_due(weekly, _last_caps_month(), through):
            return
        daily.optional_step("sectors", _build_sectors, self.notes)
        daily.optional_step("fundamentals", _build_fundamentals, self.notes)

    def build(self, after: datetime.date, through: datetime.date, weekly: bool) -> daily.GateFacts:
        cmd_events(argparse.Namespace(refresh=False, refresh_active=weekly))  # always: new issuers (IPOs)
        if weekly:
            self.notes.append("events: full refresh")
        else:
            flagged = daily.issuers_with_new_flags(ch.client(), after, through)
            b3events.collect(flagged, LAKE_DIR / "raw" / "b3_events", refresh=True)
            self.notes.append(f"events: {len(flagged)} issuers with a new ex-flag")
        # weekly: same inputs twice, a different fingerprint means the build is not reproducible (a second gold every
        # night doubled the peak memory on an 18 GB machine also running the Docker VM)
        runs = []
        for _ in range(2 if weekly else 1):
            cmd_gold(argparse.Namespace())
            runs.append(daily.gold_fingerprint(ch.client()))
        cmd_metrics(argparse.Namespace())
        cmd_audit_cvm(argparse.Namespace())
        cmd_export(argparse.Namespace(publish=False))
        # the paper-v2 decides on these; a failure (BCB down with no stored IPCA) keeps yesterday's files
        daily.optional_step("universe", _build_universe, self.notes)
        self.refresh_fundamentals(weekly, through)       # after the universe: the sectors' coverage reads it
        daily.optional_step("ptax", _build_ptax, self.notes)
        # the app's risk-free rate; a BCB outage exports the stored series and leaves a note
        daily.optional_step("cdi", _build_cdi, self.notes)
        # bond quotes and the real curve for the allocation tests; the CDN lags (its 2026 files end 21/08/2026)
        daily.optional_step("tesouro", _build_tesouro, self.notes)
        # market expectations (python-bcb, imported inside the step: a missing package fails only this step)
        daily.optional_step("focus", _build_focus, self.notes)
        # option IVs and greeks of the year (after the CDI: the rate of the new sessions), ~1 min
        daily.optional_step("greeks", lambda: cmd_greeks(argparse.Namespace(since=through.year, to=None)), self.notes)
        # futures, options on futures and open interest of the new sessions (B3 Price Report), ~1 min a session
        daily.optional_step("derivatives", lambda: cmd_derivatives(argparse.Namespace(workers=2, gap_files=20)),
                            self.notes)
        # B3's official index levels and today's index portfolios (only kept by running every session), ~2 min
        daily.optional_step("indices", _build_indices, self.notes)
        daily.optional_step("index-portfolios", _build_index_portfolios, self.notes)
        daily.optional_step("bdi", _build_bdi, self.notes)     # the bulletin keeps 21 days: missed nights refill
        daily.optional_step("option-chain", _build_option_chain, self.notes)   # the simulator's options sleeve
        client = ch.client()
        traded, traded_median = daily.traded_counts(client, through)
        facts = daily.GateFacts(
            proposed_links=pl.read_csv(LAKE_DIR / "gold" / "proposed_links.csv").height,
            traded=traded,
            traded_median=traded_median,
            big_moves=daily.big_moves(client, after, through),
            reviewed_moves=daily.reviewed_moves(corrections.load_corrections(corrections.CORRECTIONS_CSV)),
            gold_runs=(runs[0], runs[1]) if weekly else None,
        )
        logging.info("gate facts: links %d, traded %d (median %.0f), big moves %s, fingerprints equal %s",
                     facts.proposed_links, traded, traded_median, list(facts.big_moves),
                     runs[0] == runs[1] if weekly else "not checked tonight")
        return facts

    def publish(self) -> None:
        export.publish(LAKE_DIR / "exports", APP_DATA_DIR, APP_PRICES_PATH)
        logging.info("published to %s and %s", APP_DATA_DIR, APP_PRICES_PATH)


def _alert_text(result: daily.RunResult, log_path: Path) -> tuple[str, str]:
    session, reason = result.alert
    title = f"b3lake: pregão de {session:%d/%m} - {reason}"
    if reason == "missing":
        body = "O COTAHIST desse pregão ainda não entrou no lago (a B3 não publicou ou não respondeu)."
    elif reason.startswith("gates"):
        body = "Portão reprovado: o app não recebeu os dados novos. Revisar antes de publicar."
    else:
        body = f"A rotina falhou no passo '{reason}': o app não recebeu os dados novos."
    return title, f"{body} Log: {log_path}"


def _show_alert(title: str, body: str) -> None:
    try:
        daily.show_toast(title, body)
    except Exception:  # the alert cannot alert about itself: the log is all there is
        logging.exception("Windows alert failed")


def _daily_once(started: datetime.datetime, log_path: Path) -> None:
    state_path = STATE_DIR / "state.json"
    lake = _LakeSteps()
    result = daily.run_daily(started, daily.read_state(state_path), lake)
    daily.write_state(state_path, result.state)
    expected = result.facts.expected if result.facts else daily.expected_session(started)
    failure = result.facts.failure if result.facts else None
    detail = "; ".join(note for note in (failure, *lake.notes) if note)
    daily.append_run(STATE_DIR / "runs.csv", daily.RunRecord(started=started, finished=datetime.datetime.now(),
                                                             expected=expected, outcome=result.outcome,
                                                             detail=detail))
    logging.info("daily: %s, expected session %s (%s)", result.outcome.value, expected, detail or "-")
    if result.alert is not None:
        _show_alert(*_alert_text(result, log_path))
    _late_steps(result)


def _late_steps(result: daily.RunResult) -> None:
    """After the publish, the state and the history: opcoes.net.br's chains, the greeks' cross-check (headless browser,
    ~80 min for ~245 underlyings; its history is only kept by running every session). It ran inside the build until
    2026-10-30, when the task's 2 h limit killed the night before anything was published."""
    if result.outcome is not daily.Outcome.DONE:
        return
    deadline = time.monotonic() + OPCOESNET_BUDGET_SECONDS
    notes: list[str] = []
    daily.optional_step("opcoesnet", lambda: cmd_opcoesnet(argparse.Namespace(tickers=None, deadline=deadline)), notes)


def _report_interrupted(launch: str | None, log_path: Path) -> None:
    """Earlier launches that died without reporting (no log, no runs.csv line): written to runs.csv, one alert."""
    if launch is None:  # `b3lake daily` typed by hand: no launch history
        return
    interrupted = daily.report_interrupted(LAUNCHES_CSV, launch, STATE_DIR / "runs.csv")
    if interrupted:
        logging.warning("launches that never reported: %s", interrupted)
        _show_alert("b3lake: execução anterior interrompida",
                    f"{len(interrupted)} execução(ões) morreram sem registro ({interrupted[-1][1]}); ver runs.csv. "
                    f"Log: {log_path}")


def cmd_daily(args) -> None:
    """Scheduled routine: bring the last session in, check it, publish; Windows alert only when something fails."""
    started = datetime.datetime.now()
    log_path = STATE_DIR / "logs" / f"{started:%Y%m%d-%H%M%S}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")  # pythonw has no stderr: the file is the only log
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)  # launched by pythonw, main()'s basicConfig never ran: lines were lost
    try:
        with daily.run_lock(STATE_DIR / "daily.lock") as took_over:
            if took_over:
                logging.warning("took over the lock of a run that never finished")
                _show_alert("b3lake: a execução anterior não terminou",
                            f"Foi morta no limite de 2 h ou o PC desligou no meio; esta refaz a cadeia. Log: {log_path}")
            _report_interrupted(getattr(args, "launch", None), log_path)
            _daily_once(started, log_path)
    except daily.LockHeld:
        logging.info("another run holds the lock; leaving")
    except Exception:  # anything that escaped the routine itself (state file, history, lock)
        logging.exception("daily routine crashed")
        _show_alert("b3lake: a rotina diária quebrou", f"Veja o log {log_path}")


OPCOESNET_BUDGET_SECONDS = 90 * 60  # the task is killed at 2 h and a weekly build takes ~25 min
US_EXPORTS = LAKE_DIR / "exports" / "us"
SA_RAW_DIR = Path(os.environ.get("SA_RAW_DIR", "D:/dados_stocks_USA_datalake"))
US_APP_FILES = ("fred_series.parquet", "fred_catalog.parquet", "edgar_facts_first_filed.parquet",
                "edgar_companies.parquet", "edgar_tickers.parquet", usdata.MANIFEST)  # edgar_facts.parquet stays on D:


def _publish_us() -> None:
    """Copy the US exports the app reads into its ignored data folder (French files by pattern)."""
    (APP_DATA_DIR / "us").mkdir(parents=True, exist_ok=True)
    names = [*US_APP_FILES, *(path.name for path in US_EXPORTS.glob("french_*.parquet"))]
    for name in names:
        if (US_EXPORTS / name).exists():
            shutil.copy2(US_EXPORTS / name, APP_DATA_DIR / "us" / name)


def _fail_on(kind: str, failed: list[str]) -> None:
    if failed:
        raise SystemExit(f"{kind}: {len(failed)} failed and were not refreshed: {', '.join(failed)}")


def cmd_us_fred(args) -> None:
    """FRED series, current vintage (docs/us-data-design.md); keyless."""
    with requests.Session() as session:
        failed = usdata.export_fred(US_EXPORTS, session)
    _publish_us()
    _fail_on("FRED", failed)


def cmd_us_french(args) -> None:
    """Ken French factors and 49 industries, decimal returns; keyless."""
    with requests.Session() as session:
        failed = usdata.export_french(US_EXPORTS, session)
    _publish_us()
    _fail_on("Ken French", failed)


def cmd_us_edgar(args) -> None:
    """SEC companyfacts and submissions (~3 GB on D:, streamed), the first-filed facts and company attributes; needs
    SEC_USER_AGENT_EMAIL."""
    with requests.Session() as session:
        session.headers["User-Agent"] = edgar.user_agent()
        usdata.export_edgar(LAKE_DIR / "raw" / "us" / "sec", US_EXPORTS, session)
    _publish_us()


def cmd_us_prices(args) -> None:
    """Daily prices of today's listed US stocks (survivor-biased; on D: only, not copied to the app)."""
    missing = usdata.export_prices(LAKE_DIR / "raw" / "us" / "yahoo", US_EXPORTS)
    if missing:
        logging.warning("US prices: %d listed tickers without any close (first: %s)", len(missing), missing[:10])


def cmd_us_sa_bronze(args) -> None:
    """The stockanalysis pages under SA_RAW_DIR (read-only) as typed parquet under bronze/us_sa, with a manifest."""
    from b3lake import sa_bronze

    out_dir = LAKE_DIR / "bronze" / "us_sa"
    summary = sa_bronze.build_bronze(SA_RAW_DIR, out_dir, partition_size=args.partition_size, workers=args.workers)
    sa_bronze.dump_summary(summary | {"by_kind": sa_bronze.manifest_summary(out_dir)}, out_dir / "_summary.json")
    logging.info("stockanalysis bronze: %s", summary)


def cmd_us_tiingo(args) -> None:
    """Monthly prices of the dead names stockanalysis lacks, largest first, within the free plan; needs TIINGO_API_KEY.
    A later run resumes: a name on disk is never asked again."""
    from b3lake import us_tiingo

    raw = LAKE_DIR / "raw" / "us" / "tiingo"
    names = us_tiingo.names_to_fetch(raw / "supported_tickers.zip", SA_RAW_DIR / "_meta" / "universe_master.csv",
                                     LAKE_DIR / "bronze" / "us_sa" / "prices_monthly" / "*.parquet",
                                     US_EXPORTS / "edgar_facts_first_filed.parquet")
    logging.info("tiingo: %d candidates", names.height)
    names.write_parquet(raw / "candidates.parquet")
    with requests.Session() as session:
        fetch = us_tiingo.fetcher(session, os.environ[us_tiingo.KEY_ENV])
        summary = us_tiingo.collect(names["symbol"].to_list(), raw / "prices", fetch, time.sleep, args.limit)
    logging.info("tiingo: %s", summary)


def cmd_us_load(args) -> None:
    """The stockanalysis bronze and the US exports into ClickHouse database `us`, one table at a time."""
    from b3lake import us_load

    loads = [load for load in us_load.LOADS if not args.only or any(name in load.table for name in args.only)]
    logging.info("us load: %s", us_load.load_all(ch.client(), loads))


def cmd_us_audit(args) -> None:
    """Every audit check of `us_audit` on ClickHouse, written to a dated report; one failed check does not stop it."""
    from b3lake import us_audit

    client = ch.client()
    lines: list[str] = []
    errors = us_audit.run(client.query, us_audit.CHECKS, lines.append)
    gold = LAKE_DIR / "exports" / "us_stocks"
    if (gold / "manifest.json").exists():
        from b3lake import us_audit_gold

        us_audit_gold.report(gold, LAKE_DIR / "bronze" / "us_sa", lines.append)
    report = Path(__file__).resolve().parents[2] / "docs" / f"us-audit-{datetime.date.today():%Y-%m-%d}.txt"
    us_audit.write_report(report, lines)
    logging.info("us audit: %s, %d checks failed: %s", report, len(errors), sorted(errors))


def cmd_us_splits(args) -> None:
    """The splits of the names that split since 2010, from Yahoo, with the pages' last split as the fallback."""
    import yfinance  # only this step and us-prices need it

    from b3lake import us_splits

    pages = pl.read_parquet(LAKE_DIR / "bronze" / "us_sa" / "statistics" / "*.parquet")
    names = us_splits.names_with_recent_split(pages)
    logging.info("us splits: %d names", len(names))
    yahoo = us_splits.collect(names, lambda chunk: yfinance.download(
        chunk, period="max", interval="1mo", actions=True, auto_adjust=False, progress=False, threads=True), time.sleep)
    out = LAKE_DIR / "raw" / "us" / "yahoo" / "splits.parquet"
    us_splits.merged(yahoo, us_splits.last_splits(pages)).write_parquet(out)
    logging.info("us splits: %s names answered by Yahoo", yahoo["symbol"].n_unique())


def cmd_us_gold(args) -> None:
    """The app-shaped exports of the US lake under exports/us_stocks, copied to the app's ignored data folder."""
    from b3lake import us_export

    raw = LAKE_DIR / "raw" / "us"
    sources = us_export.Sources(LAKE_DIR / "bronze" / "us_sa", raw / "tiingo" / "prices", raw / "tiingo" / "candidates.parquet",
                                US_EXPORTS, raw / "yahoo" / "splits.parquet", raw / "french" / "Siccodes12.txt",
                                LAKE_DIR / "exports" / "ptax_daily.parquet")
    out_dir = LAKE_DIR / "exports" / args.out
    manifest = us_export.build(sources, out_dir, floor_today=args.floor)
    logging.info("us gold: %s", manifest["coverage"])
    target = APP_DATA_DIR / args.out
    target.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        shutil.copy2(path, target / path.name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="b3lake")
    sub = parser.add_subparsers(dest="step", required=True)
    p = sub.add_parser("download", help="fetch COTAHIST yearly zips (current year refreshed)")
    p.add_argument("--since", type=int, default=download.FIRST_YEAR)
    p.add_argument("--to", type=int, default=None)
    p.set_defaults(func=cmd_download)
    p = sub.add_parser("bronze", help="parse zips into typed parquet (changed years only)")
    p.add_argument("--since", type=int, default=download.FIRST_YEAR)
    p.add_argument("--to", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_bronze)
    sub.add_parser("migrate", help="apply pending ClickHouse schema migrations").set_defaults(func=cmd_migrate)
    p = sub.add_parser("load", help="load bronze years into ClickHouse (changed years only)")
    p.add_argument("--since", type=int, default=download.FIRST_YEAR)
    p.add_argument("--to", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_load)
    p = sub.add_parser("silver", help="rebuild silver tables from bronze in ClickHouse")
    p.add_argument("--since", type=int, default=download.FIRST_YEAR)
    p.add_argument("--to", type=int, default=None)
    p.set_defaults(func=cmd_silver)
    p = sub.add_parser("events", help="collect B3 corporate events (raw JSON per issuer)")
    p.add_argument("--refresh", action="store_true", help="re-fetch every issuer")
    p.add_argument("--refresh-active", action="store_true", help="re-fetch issuers that traded in the last 15 days")
    p.set_defaults(func=cmd_events)
    sub.add_parser("gold", help="build adjusted series and event tables").set_defaults(func=cmd_gold)
    sub.add_parser("metrics", help="rebuild indicator tables in ClickHouse").set_defaults(func=cmd_metrics)
    p = sub.add_parser("export", help="write small parquet exports for the app")
    p.add_argument("--publish", action="store_true", help="also copy into Portfolio-Optimization/data/b3lake")
    p.set_defaults(func=cmd_export)
    sub.add_parser("audit-cvm", help="audit share events against CVM FRE").set_defaults(func=cmd_audit_cvm)
    sub.add_parser("sectors", help="issuer sectors from the CVM registry, coverage of the universe").set_defaults(
        func=cmd_sectors)
    sub.add_parser("fundamentals", help="CVM DFP/ITR point-in-time accounts and monthly market value").set_defaults(
        func=cmd_fundamentals)
    sub.add_parser("universe", help="point-in-time universe (D1) + research prices with dead companies").set_defaults(
        func=cmd_universe)
    sub.add_parser("cdi", help="daily CDI (BCB SGS 12) for the app's risk-free rate").set_defaults(func=cmd_cdi)
    sub.add_parser("cda", help="CVM CDA fund stock positions (research, lake disk only)").set_defaults(func=cmd_cda)
    sub.add_parser("inf-diario", help="CVM daily fund report, monthly (research, lake disk only)").set_defaults(
        func=cmd_inf_diario)
    sub.add_parser("ptax", help="daily PTAX (BCB SGS 1, R$ per US$)").set_defaults(func=cmd_ptax)
    p = sub.add_parser("derivatives", help="B3 Price Report and instruments: futures, options, open interest")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--gap-files", type=int, default=20)
    p.set_defaults(func=cmd_derivatives)
    p = sub.add_parser("opcoesnet", help="opcoes.net.br daily chains (playwright-cli; cross-check of greeks)")
    p.add_argument("--tickers", nargs="*")
    p.set_defaults(func=cmd_opcoesnet)
    p = sub.add_parser("greeks", help="option implied volatility and greeks from COTAHIST (ClickHouse)")
    p.add_argument("--since", type=int, default=option_greeks.FIRST_DAY.year)
    p.add_argument("--to", type=int, default=None)
    p.set_defaults(func=cmd_greeks)
    sub.add_parser("tesouro", help="Tesouro Direto quotes, real curve and bond returns").set_defaults(func=cmd_tesouro)
    sub.add_parser("focus", help="BCB Focus expectations, stamped point in time").set_defaults(func=cmd_focus)
    sub.add_parser("indices", help="B3 official index levels (history API + BVBG.087)").set_defaults(func=cmd_indices)
    sub.add_parser("option-chain", help="recent option chains for the app's simulator").set_defaults(
        func=cmd_option_chain)
    p = sub.add_parser("option-signals", help="monthly option signals per chain (options veto research)")
    p.add_argument("--since", type=int, default=2004)
    p.add_argument("--to", type=int, default=None)
    p.set_defaults(func=cmd_option_signals)
    sub.add_parser("bdi", help="B3 daily bulletin tables (lending, investors, open positions...)").set_defaults(
        func=cmd_bdi)
    sub.add_parser("index-portfolios", help="today's B3 index portfolios and previews, kept daily").set_defaults(
        func=cmd_index_portfolios)
    sub.add_parser("us-fred", help="US macro series from FRED (current vintage)").set_defaults(func=cmd_us_fred)
    sub.add_parser("us-french", help="US factors and industries from Ken French").set_defaults(func=cmd_us_french)
    p = sub.add_parser("us-sa-bronze", help="stockanalysis raw pages as typed parquet (Dask, read-only on the raw)")
    p.add_argument("--partition-size", type=int, default=400)
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(func=cmd_us_sa_bronze)
    p = sub.add_parser("us-load", help="load the US lake into ClickHouse database us")
    p.add_argument("--only", nargs="*", help="tables whose name holds one of these words (default: all)")
    p.set_defaults(func=cmd_us_load)
    sub.add_parser("us-audit", help="run the US lake audit, write docs/us-audit-<date>.txt").set_defaults(func=cmd_us_audit)
    sub.add_parser("us-splits", help="stock splits of the US names (Yahoo, the pages as fallback)").set_defaults(
        func=cmd_us_splits)
    p = sub.add_parser("us-gold", help="the US lake exports for the app (exports/us_stocks)")
    p.add_argument("--floor", type=float, default=1_000_000.0, help="median daily dollar volume in today's money")
    p.add_argument("--out", default="us_stocks", help="the folder under exports/ and the app's data folder")
    p.set_defaults(func=cmd_us_gold)
    p = sub.add_parser("us-tiingo", help="monthly prices of dead US names from Tiingo (free plan, resumes)")
    p.add_argument("--limit", type=int, default=480, help="symbols this run may spend (the plan allows 500 a month)")
    p.set_defaults(func=cmd_us_tiingo)
    sub.add_parser("us-prices", help="US listed stocks' daily prices from Yahoo (survivor-biased)").set_defaults(
        func=cmd_us_prices)
    sub.add_parser("us-edgar", help="US company facts from SEC EDGAR (point in time by filing date)").set_defaults(
        func=cmd_us_edgar)
    sub.add_parser("daily", help="scheduled routine: last session in, gates, publish, alert on failure").set_defaults(
        func=cmd_daily)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
