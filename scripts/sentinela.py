"""
The Sentinela's hourly run (spec ``docs/superpowers/specs/2026-11-05-sentinela-design.md``; task "sentinela", every
hour from 08:20 to 19:20 on weekdays; B3 holidays are skipped here).

Usage: python -X utf8 scripts/sentinela.py [--forcar] [--sem-git] [--sem-email] [--semear] [--pasta DIR]

Each run, in the app checkout on ``main``:
1. reads every source by code (no model): CVM filings (the weekly table and the daily RAD/ENET search), the lake's
   prices and volumes, B3 BDI lending, BCB SGS (Selic target, IPCA, IBC-Br, credit, fiscal, PTAX, Focus, Copom
   minutes), IBGE SIDRA and Tesouro RTN; a failing source drops only its items and is reported;
2. ``core.sentinela.rodada.rodar``: one DeepSeek scan per Composite name with new items, judges in shadow, a decision
   for each checked ``grave``, emails;
3. trades the accepted actions in the fictitious "Composite + Sentinela" account once their close is in the price
   files, and rewrites the derived score table (Brier and log scores);
4. commits exactly the Sentinela's files and pushes ``main`` fast-forward (skipped while another git operation runs;
   the files wait for the next hour).
It never touches real money. Outcomes go to ``%LOCALAPPDATA%/paper-sentinela/runs.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core import alerta_email, bcb, cdi_lago, focus, macro_bcb, precos_do_app, simulador  # noqa: E402
from core.rebalanceio import custo_com_liquidez  # noqa: E402
from core.segredos import segredo  # noqa: E402
from core.sentinela import calendario, cvm_diaria, fontes, pontuacao, registro, rodada  # noqa: E402
from core.sentinela.itens import Item  # noqa: E402
from scripts.manutencao import (ARQUIVO_ATIVOS, ARQUIVO_CALENDARIO, ARQUIVO_LIQUIDEZ, ARQUIVO_RECENTES,  # noqa: E402
                                ARQUIVO_REGISTRO_V3, PASTA_DO_LAGO, PASTA_ESTADO, Bloqueio, commitar, conferir_checkout,
                                git_em, pendentes, publicar)

PASTA = Path("simulador")
INICIO = pd.Timestamp("2026-11-06")          # the account's first session: the first one the agent is live
ARQUIVO_FATOS = Path("lago/fatos_relevantes.parquet")
DIAS_DE_FATOS = 7                            # filings older than a week are history, not news, on the first run
DIAS_DO_PACOTE_2 = 120                       # the name's recent filings for stage 2
DIAS_DA_CVM_DIARIA = 2
HORAS = range(8, 20)
LOG_LOCAL = PASTA_ESTADO.parent / "paper-sentinela" / "runs.csv"    # the launcher's "sentinela" folder
LAGO_EM_D = Path("D:/b3-datalake/exports")
CACHE_RTN = PASTA_ESTADO / "sentinela_rtn.xlsx"


def _lago(raiz: Path) -> Path:
    """The lake's exports: the checkout's copy, else the lake itself (a worktree has no copy)."""
    return raiz / PASTA_DO_LAGO if (raiz / PASTA_DO_LAGO / "daily_liquid_panel.parquet").exists() else LAGO_EM_D


def _composta(raiz: Path) -> tuple[pd.DataFrame, dict[str, float], pd.Timestamp]:
    registro_v3 = pd.read_csv(raiz / ARQUIVO_REGISTRO_V3, parse_dates=["data"])
    ultima = registro_v3["data"].max()
    atual = registro_v3[registro_v3["data"] == ultima]
    return registro_v3, dict(zip(atual["ticker"], atual["peso"].astype(float))), ultima


def _empresas(raiz: Path, tickers: list[str]) -> dict[str, tuple[str, str]]:
    ativos = pd.read_parquet(raiz / ARQUIVO_ATIVOS).set_index("security")
    return {ticker: (str(ativos.at[ticker, "name"]) if ticker in ativos.index else ticker,
                     str(ativos.at[ticker, "sector"]) if ticker in ativos.index else "") for ticker in tickers}


def _cvm(raiz: Path, tickers: list[str], hoje: date, falhas: list[str]) -> tuple[list[Item], list[Item]]:
    """(this run's CVM items, the names' recent filings for stage 2)."""
    tabelas = []
    try:
        semanal = pd.read_parquet(raiz / ARQUIVO_FATOS)
        semanal["data"] = pd.to_datetime(semanal["data"])
        tabelas.append(semanal)
    except Exception as erro:                   # noqa: BLE001
        falhas.append(f"CVM semanal: {type(erro).__name__}")
    try:
        setores = pd.read_parquet(_lago(raiz) / "sectors.parquet")
        lista = cvm_diaria.consultar(requests.Session(), hoje - timedelta(days=DIAS_DA_CVM_DIARIA), hoje)
        tabelas.append(cvm_diaria.fatos_dos_nomes(lista, cvm_diaria.codigos(setores, tickers)))
    except Exception as erro:                   # noqa: BLE001
        falhas.append(f"CVM diária: {erro}" if isinstance(erro, cvm_diaria.ErroDaCvm) else
                      f"CVM diária: {type(erro).__name__}")
    novos = fontes.fatos_cvm(tabelas, tickers, hoje - timedelta(days=DIAS_DE_FATOS))
    recentes = fontes.fatos_cvm(tabelas, tickers, hoje - timedelta(days=DIAS_DO_PACOTE_2))
    return novos, recentes


def _macro(hoje: date, falhas: list[str]) -> list[Item]:
    from core.sentinela.itens import itens_macro
    series, erros = fontes.series_bcb(bcb.baixar_serie, hoje)
    falhas += erros
    try:
        ano = hoje.year
        for indicador, nome, rotulo in (("IPCA", "focus_ipca", f"Focus: mediana do IPCA de {ano} (%)"),
                                        ("Selic", "focus_selic", f"Focus: mediana da Selic no fim de {ano} (% a.a.)")):
            series.append(fontes.serie_focus(focus.baixar_focus_anual(indicador, hoje), ano, nome, rotulo))
    except Exception as erro:                   # noqa: BLE001
        falhas.append(f"BCB Focus: {type(erro).__name__}")
    ibge, erros = fontes.series_ibge(requests.get)
    falhas += erros
    renderizar = fontes.renderizador_firecrawl(segredo("FIRECRAWL_API_KEY"), requests.post)
    tesouro, erros = fontes.series_tesouro(hoje, requests.get, renderizar, CACHE_RTN)
    falhas += erros
    itens = itens_macro(series + ibge + tesouro)
    ptax, erros = fontes.itens_ptax(bcb.baixar_serie, hoje)
    falhas += erros
    try:
        itens += fontes.item_da_ata(macro_bcb.ata_do_copom())
    except Exception as erro:                   # noqa: BLE001
        falhas.append(f"BCB ata do Copom: {type(erro).__name__}")
    return itens + ptax


def _email():
    credenciais = alerta_email.credenciais()
    if credenciais is None:
        return None
    return lambda assunto, corpo: alerta_email.enviar(assunto, corpo, credenciais)


def _conta(raiz: Path, arquivos: rodada.Arquivos, registro_v3: pd.DataFrame, hoje: date) -> list[str]:
    papeis = sorted(set(registro_v3["ticker"]))
    precos = precos_do_app.ler_fechamentos([f"{papel}.SA" for papel in papeis], INICIO.date(), hoje,
                                           raiz / ARQUIVO_CALENDARIO, raiz / ARQUIVO_RECENTES)
    precos = precos.rename(columns=lambda papel: papel.removesuffix(".SA"))
    if precos.empty or precos.index[-1] < INICIO:
        return []
    foto = pd.read_parquet(raiz / ARQUIVO_LIQUIDEZ) if (raiz / ARQUIVO_LIQUIDEZ).exists() else None
    return rodada.atualizar_conta(arquivos, registro_v3, precos, cdi_lago.daily_selic(1, hoje),
                                  custo_com_liquidez(foto, simulador.CUSTOS, simulador.CUSTO_BPS), INICIO)


def _pontuacao(raiz: Path, arquivos: rodada.Arquivos, tickers: list[str]) -> None:
    log = registro.ler_log(arquivos.log)
    if log.empty:
        return
    fechamentos, _ = fontes.painel_de_mercado(_lago(raiz) / "daily_liquid_panel.parquet", tickers, 400)
    tabela = pontuacao.pontuar(log, fechamentos)
    if not tabela.empty:
        tabela.to_csv(arquivos.pontuacao, index=False, lineterminator="\n", encoding="utf-8")


def _publicar(raiz: Path, arquivos: rodada.Arquivos, agora: datetime) -> str:
    git = git_em(raiz)
    try:
        conferir_checkout(raiz, git)
    except Bloqueio as bloqueio:
        return f"git adiado: {bloqueio}"
    caminhos = [caminho.relative_to(raiz).as_posix() for caminho in
                (arquivos.vistos, arquivos.log, arquivos.acoes, arquivos.ordens, arquivos.pontuacao)
                if caminho.exists()]
    if not pendentes(git, caminhos):
        return "git: nada novo"
    commitar(git, caminhos, f"data: Sentinela até {agora:%d/%m/%Y %H:%M} (rodada horária)")
    try:
        return publicar(git)
    except Bloqueio as bloqueio:
        return f"push adiado: {bloqueio}"


def _anotar(agora: datetime, situacao: str, relatorio: rodada.Relatorio | None, falhas: list[str], git: str) -> None:
    LOG_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    novo = not LOG_LOCAL.exists()
    with LOG_LOCAL.open("a", encoding="utf-8", newline="") as saida:
        escritor = csv.writer(saida, lineterminator="\n")
        if novo:
            escritor.writerow(["hora", "situacao", "chamadas", "graves", "acoes", "adiados", "emails", "falhas", "git"])
        escritor.writerow([f"{agora:%Y-%m-%dT%H:%M:%S}", situacao, relatorio.chamadas if relatorio else 0,
                           ";".join(relatorio.graves) if relatorio else "", ";".join(relatorio.acoes) if relatorio else "",
                           ";".join(relatorio.adiados) if relatorio else "", relatorio.emails if relatorio else 0,
                           " | ".join(falhas + (relatorio.falhas if relatorio else [])), git])


def main(argv: list[str] | None = None) -> int:
    argumentos = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    argumentos.add_argument("--forcar", action="store_true", help="run outside B3 sessions and hours")
    argumentos.add_argument("--sem-git", action="store_true", help="no commit or push")
    argumentos.add_argument("--pasta", type=Path, default=None, help="where the Sentinela's files go (tests)")
    argumentos.add_argument("--sem-email", action="store_true", help="no email (a test run)")
    argumentos.add_argument("--semear", action="store_true",
                            help="mark every current item seen, no model call (the baseline before going live)")
    opcoes = argumentos.parse_args(argv if argv is not None else [arg for arg in sys.argv[1:] if arg.startswith("-")])
    agora = datetime.now().replace(microsecond=0)
    hoje = agora.date()
    if not opcoes.forcar and (not calendario.pregao(hoje) or agora.hour not in HORAS):
        print(f"Sentinela: fora do horário ({agora:%d/%m %H:%M})")
        return 0
    raiz = RAIZ
    arquivos = rodada.Arquivos(opcoes.pasta or raiz / PASTA)
    falhas: list[str] = []
    registro_v3, carteira, registrada = _composta(raiz)
    tickers = sorted(carteira)
    cvm, fatos = _cvm(raiz, tickers, hoje, falhas)
    mercado, erros, fechamentos = fontes.itens_de_mercado(_lago(raiz) / "daily_liquid_panel.parquet", _lago(raiz),
                                                          tickers)
    falhas += erros
    itens = cvm + mercado + _macro(hoje, falhas)
    entrada = rodada.Entrada(agora=agora, carteira=carteira, registrada=registrada, empresas=_empresas(raiz, tickers),
                             itens=itens, fatos=fatos, fechamentos=fechamentos,
                             pregoes=calendario.pregoes(hoje - timedelta(days=10), hoje + timedelta(days=60)))
    if opcoes.semear:
        marcados = rodada.semear(entrada, arquivos)
        git = "sem git" if opcoes.sem_git or opcoes.pasta else _publicar(raiz, arquivos, agora)
        print(f"Sentinela: {marcados} pares (ação, item) marcados como vistos, sem chamada; {git}")
        for falha in falhas:
            print(f"  fonte com falha: {falha}")
        return 0
    servicos = rodada.Servicos(segredo("DEEPSEEK_API_KEY"), segredo("LIQUID_API_KEY"), email=None if opcoes.sem_email else _email())
    relatorio = rodada.rodar(entrada, arquivos, servicos)
    try:
        executadas = _conta(raiz, arquivos, registro_v3, hoje)
    except Exception as erro:                   # noqa: BLE001 - the account waits for the next run
        executadas = []
        falhas.append(f"conta: {type(erro).__name__}: {erro}")
    try:
        _pontuacao(raiz, arquivos, tickers)
    except Exception as erro:                   # noqa: BLE001
        falhas.append(f"pontuação: {type(erro).__name__}")
    git = "sem git" if opcoes.sem_git or opcoes.pasta else _publicar(raiz, arquivos, agora)
    situacao = relatorio.erro or "ok"
    _anotar(agora, situacao, relatorio, falhas, git)
    print(f"Sentinela {agora:%d/%m %H:%M}: {situacao}; {len(itens)} itens lidos, {relatorio.chamadas} chamada(s), "
          f"graves {relatorio.graves or '-'}, ações {relatorio.acoes or '-'}, executadas {executadas or '-'}, "
          f"adiados {relatorio.adiados or '-'}, e-mails {relatorio.emails}; {git}")
    for falha in falhas + relatorio.falhas:
        print(f"  fonte com falha: {falha}")
    return 0 if relatorio.erro is None else 1


if __name__ == "__main__":
    sys.exit(main())
