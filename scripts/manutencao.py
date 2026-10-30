"""
Daily upkeep of the Alepha (task "paper manutencao", weekdays at 10:00, 16:00 and 22:30, through
``scripts/rodar_paper.py manutencao``), so nothing after v1.0 depends on the user remembering it.

Usage: .venv/Scripts/python -X utf8 scripts/manutencao.py

Each run, in the app checkout on ``main``, after the lake and the paper routines have finished:
- paper-v2 readings: with 126 and 252 sessions after the first record (the pre-registered report and decision of
  ``scripts/paper_v2.py``), and once a month as information only (DECISOES 121), it runs ``paper_v2.py --leitura`` and
  saves its output to ``paper-v2/leituras/``; a reading counts only when its own session count reaches the target;
- paper-v3 readings: the same at 126 and 252 sessions after its first record (``scripts/paper_v3.py --leitura``,
  ``paper-v3/leituras/``), no monthly one (DECISOES 166 and 188);
- the app's liquidity snapshot (``scripts/liquidez_do_app.py``) when the lake has a newer closed month;
- ``lago/precos_recentes.parquet``: the nightly's sessions after the committed price file (``core.precos_do_app``),
  so the deployed app is one session old, not one rebalance old (DECISOES 124);
- ``lago/ativos.parquet``: the lake's list of the market panel's stocks, BDRs and ETFs (name, type, B3 sector), when
  its rows changed;
- the live backtest curves of the Alepha screen (``scripts/curvas_ao_vivo.py``, DECISOES 188) when they are 5
  sessions behind the lake (about weekly; a ~3 minute run);
- the news/sentiment panel files (``scripts/painel_do_app.py``) from the lake's log on D:, numbers and public CVM
  filings only (DECISOES 124);
- the newest scored headlines (title, link, topic, tone) to the private repository ``alepha-manchetes`` through its
  clone next to the projects folder, which the deployed app reads with a token in Streamlit secrets (DECISOES 134);
- a commit of exactly those paths and a fast-forward push of ``main`` (never forced), which also publishes the paper
  routines' local commits to the deployed app.
It decides nothing about the portfolio. Outcomes go to ``%LOCALAPPDATA%/paper-manutencao/runs.csv``, which the 12h
watchdog reads; saved readings also raise a Windows notification.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # run as a script, only scripts/ is on it

Git = Callable[..., str]

LEITURAS = (126, 252)           # the pre-registered report and decision of paper_v2.py
RAMO = "main"                   # the branch the scheduled routines run on
# relative to the app checkout
ARQUIVO_CALENDARIO = Path("lago/precos_diarios.parquet")
ARQUIVO_REGISTRO = Path("paper-v2/carteiras.csv")
PASTA_LEITURAS = Path("paper-v2/leituras")
ARQUIVO_REGISTRO_V3 = Path("paper-v3/composto.csv")      # paper-v3's composite book (DECISOES 166)
PASTA_LEITURAS_V3 = Path("paper-v3/leituras")
ARQUIVO_LIQUIDEZ = Path("lago/liquidez.parquet")
ARQUIVO_ATIVOS = Path("lago/ativos.parquet")           # the market panel's tickers: name, type, B3 sector
ARQUIVO_RECENTES = Path("lago/precos_recentes.parquet")   # the sessions after the committed calendar (deployed app)
SOBREPOSICAO = 10           # committed sessions repeated in the small file (a stock may not trade on the last one)
# the live backtest curves of the Alepha screen (scripts/curvas_ao_vivo.py, DECISOES 188), rerun about weekly
ARQUIVOS_DAS_CURVAS = (Path("docs/dados/curvas-ao-vivo.parquet"), Path("docs/curvas-ao-vivo.txt"))
DEFASAGEM_DAS_CURVAS = 5
ARQUIVOS_DO_PAINEL = (Path("lago/sentimento_diario.parquet"), Path("lago/fatos_relevantes.parquet"),
                      Path("lago/fundamentos.parquet"))
PASTA_DO_LOG = Path("D:/b3-datalake/sentiment")           # the lake's news/social log and CVM IPE filings
PASTA_DO_LAGO = Path("data/b3lake")             # the lake's exports (outside git)
_SEM_JANELA = getattr(subprocess, "CREATE_NO_WINDOW", 0)
RAIZ = Path(__file__).resolve().parent.parent
_LOCAL = Path(os.environ.get("LOCALAPPDATA", Path.home()))
PASTA_ESTADO = _LOCAL / "paper-manutencao"     # runs.csv, outside the repository (the launcher's "manutencao")
TRAVA_DO_LAGO = _LOCAL / "b3lake" / "daily.lock"
ROTINAS_ANTES = ("b3lake daily", "paper trading", "paper v2")     # the upkeep waits for them to finish
# the private headlines store (DECISOES 134): cloned next to the projects folder, outside every repository
URL_DAS_MANCHETES = "https://github.com/GabrielScript/alepha-manchetes.git"
CLONE_DAS_MANCHETES = RAIZ.parent.parent / "alepha-manchetes"


class Bloqueio(Exception):
    """A state only the user should resolve (another branch, a git operation in course, ``main`` behind origin)."""


def pregoes_desde_o_registro(calendario: pd.DatetimeIndex, registro: pd.DataFrame) -> int:
    """Sessions of the lake's calendar after the first registered close (as ``paper_v2.py --leitura`` counts them)."""
    if registro.empty:
        return 0
    return int((calendario > registro["data"].min()).sum())


def leituras_devidas(pregoes: int, pasta: Path, hoje: date, progresso: bool = True) -> dict[str, int]:
    """The readings not yet saved in ``pasta`` whose target was reached, name -> sessions the reading itself must
    count: the pre-registered ones at 126 and 252 sessions and, with ``progresso``, this month's progress
    (information only)."""
    alvos = {f"leitura-{alvo}": alvo for alvo in LEITURAS} | ({f"progresso-{hoje:%Y-%m}": 1} if progresso else {})
    return {nome: alvo for nome, alvo in alvos.items()
            if pregoes >= alvo and not (pasta / f"{nome}.txt").exists()}


def pregoes_da_saida(texto: str) -> int | None:
    """The session count in the first line of ``paper_v2.py --leitura`` ("...: 126 pregões, ..."); None without it."""
    achado = re.search(r": (\d+) pregões", texto.splitlines()[0] if texto else "")
    return int(achado.group(1)) if achado else None


def salvar_leituras(texto: str, devidas: dict[str, int], pasta: Path, agora: datetime,
                    tentativas: Path) -> list[Path]:
    """Saves the reading in ``pasta`` as each due name whose target its own session count reached; the others go to
    ``tentativas`` (outside the repository) with the time, and the next run retries them. Returns the saved ones."""
    pregoes = pregoes_da_saida(texto)
    feitas = []
    for nome, alvo in devidas.items():
        if pregoes is not None and pregoes >= alvo:
            destino = pasta / f"{nome}.txt"
            feitas.append(destino)
        else:
            destino = tentativas / f"{nome}-{agora:%Y%m%d-%H%M%S}.txt"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(texto, encoding="utf-8", newline="\n")
    return feitas


def liquidez_nova(atual: Path, origem: Path) -> pd.DataFrame | None:
    """The snapshot of ``scripts/liquidez_do_app.py`` from the lake's exports in ``origem`` when its month is newer
    than the one in ``atual`` (or there is no snapshot yet); None otherwise."""
    from scripts.liquidez_do_app import instantaneo, ler_exports
    foto = instantaneo(*ler_exports(origem))
    if foto.empty:
        return None
    if atual.exists() and pd.Timestamp(foto["month"].iloc[0]) <= pd.Timestamp(
            pd.read_parquet(atual, columns=["month"])["month"].iloc[0]):
        return None
    return foto


def git_em(raiz: Path) -> Git:
    """``git -C raiz ...`` that never waits for a credential prompt (a hidden task would hang on it) and raises
    ``CalledProcessError`` or ``TimeoutExpired`` on failure."""
    ambiente = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}

    def git(*argumentos: str) -> str:
        return subprocess.run(["git", "-C", str(raiz), *argumentos], check=True, capture_output=True, text=True,
                              encoding="utf-8", env=ambiente, timeout=300, creationflags=_SEM_JANELA).stdout.rstrip()
    return git


def pendentes(git: Git, caminhos: list[str]) -> list[str]:
    """The paths of ``caminhos`` that differ from HEAD (new or changed): what a commit of them would take."""
    saida = git("status", "--porcelain", "--untracked-files=all", "--", *caminhos)
    return [linha[3:] for linha in saida.splitlines()]


def commitar(git: Git, caminhos: list[str], mensagem: str) -> list[str]:
    """Commits exactly ``caminhos`` (never ``add -A``: the root has loose untracked files); returns the paths the
    commit took, none when they did not change."""
    git("add", "--", *caminhos)
    feitos = git("diff", "--cached", "--name-only", "--", *caminhos).splitlines()
    if feitos:
        git("commit", "-q", "-m", mensagem, "--", *caminhos)
    return feitos


def publicar(git: Git) -> str:
    """Fast-forward push of ``main`` to ``origin``; ``Bloqueio`` when ``origin/main`` has commits ``main`` lacks (a
    routine neither merges nor forces: the user decides)."""
    git("fetch", "-q", "origin", RAMO)
    atras = int(git("rev-list", "--count", f"{RAMO}..origin/{RAMO}"))
    if atras:
        raise Bloqueio(f"origin/{RAMO} tem {atras} commit(s) que a {RAMO} local não tem: git pull à mão")
    frente = int(git("rev-list", "--count", f"origin/{RAMO}..{RAMO}"))
    if not frente:
        return f"{RAMO} em dia com origin"
    git("push", "-q", "origin", f"{RAMO}:{RAMO}")
    return f"push de {frente} commit(s)"


def conferir_checkout(raiz: Path, git: Git) -> None:
    """``Bloqueio`` unless the checkout is on ``main`` with no git operation in course (``index.lock``)."""
    ramo = git("rev-parse", "--abbrev-ref", "HEAD")
    if ramo != RAMO:
        raise Bloqueio(f"checkout em {ramo}, as rotinas rodam na {RAMO}: git switch {RAMO}")
    trava = Path(git("rev-parse", "--git-path", "index.lock"))
    if (trava if trava.is_absolute() else raiz / trava).exists():
        raise Bloqueio(f"operação do git em curso ({trava.as_posix()})")


@dataclass(frozen=True)
class Resultado:
    desfecho: str                       # "ok", or the watchdog's failure tokens: "falhou", "bloqueada", "lago rodando"
    nota: str
    avisos: list[tuple[str, str]]       # Windows notifications: (title, body)


@dataclass(frozen=True)
class Paper:
    """A forward log whose readings the upkeep saves: its registry, its readings folder, its name (commit prefix and
    attempts subfolder), the notification's prefix, whether it has the monthly progress reading and the start of the
    lines that sum a reading up in the notification."""
    registro: Path
    leituras: Path
    nome: str
    titulo: str
    progresso: bool
    resumo: str


PAPER_V2 = Paper(ARQUIVO_REGISTRO, PASTA_LEITURAS, "paper-v2", "Alepha: ", True, "Critério")
PAPER_V3 = Paper(ARQUIVO_REGISTRO_V3, PASTA_LEITURAS_V3, "paper-v3", "Alepha: paper-v3, ", False, "Decision")


def _titulo(nome: str, prefixo: str = PAPER_V2.titulo) -> str:
    if nome.startswith("leitura-"):
        return f"{prefixo}leitura de {nome.removeprefix('leitura-')} pregões"
    ano, mes = nome.removeprefix("progresso-").split("-")
    return f"{prefixo}progresso de {mes}/{ano}"


def _leituras(raiz: Path, git: Git, agora: datetime, ler_leitura: Callable[[], str], tentativas: Path,
              avisos: list[tuple[str, str]], paper: Paper = PAPER_V2) -> str:
    from scripts.paper_trading import ler_registro
    calendario = pd.DatetimeIndex(pd.read_parquet(raiz / ARQUIVO_CALENDARIO, columns=["date"])["date"])
    registro = ler_registro(raiz / paper.registro)
    devidas = leituras_devidas(pregoes_desde_o_registro(calendario, registro), raiz / paper.leituras, agora.date(),
                               paper.progresso)
    if paper is not PAPER_V2:           # paper-v2's attempts stay where they always were
        tentativas = tentativas / paper.nome
    notas = []
    if devidas:
        texto = ler_leitura()
        if not salvar_leituras(texto, devidas, raiz / paper.leituras, agora, tentativas):
            notas.append(f"{', '.join(devidas)} sem os pregões na saída, de novo na próxima rodada")
    # every saved reading not yet in HEAD, this run's or one whose commit failed before: commit, then notify
    salvas = pendentes(git, [caminho.relative_to(raiz).as_posix()
                             for caminho in sorted((raiz / paper.leituras).glob("*.txt"))])
    if salvas:
        commitar(git, salvas, f"{paper.nome}: {', '.join(Path(caminho).stem for caminho in salvas)}")
        for caminho in salvas:
            texto = (raiz / caminho).read_text(encoding="utf-8")
            resumo = "\n".join(linha for linha in texto.splitlines() if linha.startswith(paper.resumo))
            avisos.append((_titulo(Path(caminho).stem, paper.titulo), resumo or texto.splitlines()[0]))
        notas.append(", ".join(Path(caminho).stem for caminho in salvas))
    rotulo = "leituras" if paper is PAPER_V2 else f"leituras {paper.nome}"
    return f"{rotulo}: {'; '.join(notas) or 'nenhuma devida'}"


def _liquidez(raiz: Path, git: Git) -> str:
    foto = liquidez_nova(raiz / ARQUIVO_LIQUIDEZ, raiz / PASTA_DO_LAGO)
    if foto is not None:
        (raiz / ARQUIVO_LIQUIDEZ).parent.mkdir(parents=True, exist_ok=True)
        foto.to_parquet(raiz / ARQUIVO_LIQUIDEZ, index=False)
    # this run's snapshot or one whose commit failed before
    if not pendentes(git, [ARQUIVO_LIQUIDEZ.as_posix()]):
        return "liquidez: em dia"
    mes = f"{pd.read_parquet(raiz / ARQUIVO_LIQUIDEZ, columns=['month'])['month'].iloc[0]:%m/%Y}"
    commitar(git, [ARQUIVO_LIQUIDEZ.as_posix()], f"liquidez: snapshot de {mes} (manutenção)")
    return f"liquidez: {mes}"


def _no_head(raiz: Path, caminho: Path) -> bytes | None:
    """The bytes of ``caminho`` as committed in HEAD; None when HEAD does not have it."""
    feita = subprocess.run(["git", "-C", str(raiz), "show", f"HEAD:{caminho.as_posix()}"], capture_output=True,
                           timeout=300, creationflags=_SEM_JANELA)
    return feita.stdout if feita.returncode == 0 else None


def atualizar_recentes(raiz: Path, git: Git) -> str:
    """Writes ``lago/precos_recentes.parquet`` = the sessions of the nightly's ``lago/precos_diarios.parquet`` after
    the committed file's last one, plus its last ``SOBREPOSICAO`` sessions (``core.precos_do_app.emendar`` rescales the
    base there), and commits it; nothing when the nightly's file has no newer session. The big file is committed only
    by the paper routines' rebalances, so this is what keeps the deployed app one session old."""
    commitado = _no_head(raiz, ARQUIVO_CALENDARIO)
    if commitado is None:
        return "pregões recentes: sem o arquivo de preços no HEAD"
    datas = pd.read_parquet(io.BytesIO(commitado), columns=["date"])["date"]
    ultima = datas.max()
    inicio = datas.sort_values().iloc[-SOBREPOSICAO:].iloc[0]       # the overlap where the splice rescales the base
    precos = pd.read_parquet(raiz / ARQUIVO_CALENDARIO)
    if not (precos["date"] > ultima).any():
        return "pregões recentes: o arquivo do lago não passa do commitado"
    recentes = precos.loc[precos["date"] >= inicio].reset_index(drop=True)
    destino = raiz / ARQUIVO_RECENTES
    if not destino.exists() or not pd.read_parquet(destino).equals(recentes):
        # ~1,550 columns since every BDR and ETF: no pandas schema or statistics (about 1 KB per column per commit)
        recentes.to_parquet(destino, index=False, compression="zstd", store_schema=False, write_statistics=False)
    if not pendentes(git, [ARQUIVO_RECENTES.as_posix()]):
        return "pregões recentes: em dia"
    fim = f"{pd.Timestamp(recentes['date'].max()):%d/%m/%Y}"
    commitar(git, [ARQUIVO_RECENTES.as_posix()], f"data: pregões recentes até {fim} (manutenção)")
    return f"pregões recentes: até {fim}"


def atualizar_curvas(raiz: Path, git: Git, gerar: Callable[[], object]) -> str:
    """Reruns the live backtest curves of the Alepha screen (``gerar``: ``scripts/curvas_ao_vivo.py``, ~3 minutes)
    when the file is missing or ``DEFASAGEM_DAS_CURVAS`` sessions of the lake's calendar behind, and commits both of
    its files (also a pair a failed commit left behind)."""
    arquivo = raiz / ARQUIVOS_DAS_CURVAS[0]
    calendario = pd.read_parquet(raiz / ARQUIVO_CALENDARIO, columns=["date"])["date"]
    ultima = pd.read_parquet(arquivo, columns=["date"])["date"].max() if arquivo.exists() else None
    atrasada = ultima is None or int((calendario > ultima).sum()) >= DEFASAGEM_DAS_CURVAS
    if atrasada:
        gerar()
        ultima = pd.read_parquet(arquivo, columns=["date"])["date"].max()
    mudados = pendentes(git, [caminho.as_posix() for caminho in ARQUIVOS_DAS_CURVAS])
    if mudados:
        commitar(git, mudados, f"data: curvas ao vivo até {pd.Timestamp(ultima):%d/%m/%Y} (manutenção)")
    return (f"curvas ao vivo: até {pd.Timestamp(ultima):%d/%m/%Y}" if atrasada else
            f"curvas ao vivo: em dia (até {pd.Timestamp(ultima):%d/%m/%Y})")


def atualizar_ativos(raiz: Path, git: Git) -> str:
    """Copies the lake's list of the market panel's tickers (``data/b3lake/app_names.parquet``) to
    ``lago/ativos.parquet`` and commits it when its rows changed. The nightly writes only to the ignored folder: a
    versioned file it rewrote would leave the checkout dirty for the paper routines until this upkeep."""
    origem = raiz / PASTA_DO_LAGO / "app_names.parquet"
    if not origem.exists():
        return "ativos: sem a lista do lago"
    nomes, destino = pd.read_parquet(origem), raiz / ARQUIVO_ATIVOS
    if destino.exists() and pd.read_parquet(destino).equals(nomes):
        return "ativos: em dia"
    destino.parent.mkdir(exist_ok=True)
    nomes.to_parquet(destino, index=False)
    commitar(git, [ARQUIVO_ATIVOS.as_posix()], f"data: lista de ativos da tela, {len(nomes)} papéis (manutenção)")
    return f"ativos: {len(nomes)} papéis"


def atualizar_painel(raiz: Path, git: Git, pasta_do_log: Path | None,
                     gerar: Callable[[], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> str:
    """Rebuilds the panel files (``scripts/painel_do_app.py``: the daily news tone, the recent CVM filings and the
    per-stock fundamentals; numbers and public filings only) and commits them; nothing without the lake's log."""
    if pasta_do_log is None or not pasta_do_log.exists():
        return "painel: sem o log do lago"
    tabelas = gerar()
    for arquivo, tabela in zip(ARQUIVOS_DO_PAINEL, tabelas):
        destino = raiz / arquivo
        destino.parent.mkdir(parents=True, exist_ok=True)
        if not destino.exists() or not pd.read_parquet(destino).equals(tabela):
            tabela.to_parquet(destino)
    caminhos = [arquivo.as_posix() for arquivo in ARQUIVOS_DO_PAINEL]
    if not pendentes(git, caminhos):
        return "painel: em dia"
    diario, fatos, fundamentos = tabelas
    resumo = f"{len(diario)} dia(s), {len(fatos)} fato(s), {len(fundamentos)} ação(ões)"
    commitar(git, caminhos, f"data: painel de notícias, fatos relevantes e fundamentos ({resumo})")
    return f"painel: {resumo}"


LojaDeManchetes = tuple[Path, str, Path | None, Callable[[], pd.DataFrame]]     # clone, url, the lake's log, gerar


def atualizar_manchetes(clone: Path, url: str, pasta_do_log: Path | None, gerar: Callable[[], pd.DataFrame],
                        agora: datetime) -> str:
    """Publishes the newest scored headlines (``gerar``: ``core.sentimento.manchetes``, title, link, topic and tone)
    as ``manchetes.json`` in the private repository ``url`` through its clone ``clone`` (cloned on the first run with
    the user's stored git login), committed and pushed only when the headlines changed; nothing without the lake's
    log. ``Bloqueio`` when the repository cannot be reached (not created yet, or the login lost access)."""
    from core.manchetes_online import ARQUIVO, para_json
    if pasta_do_log is None or not pasta_do_log.exists():
        return "manchetes: sem o log do lago"
    if not (clone / ".git").exists():
        try:
            git_em(clone.parent)("clone", "-q", url, clone.name)
        except subprocess.CalledProcessError as erro:
            raise Bloqueio(f"sem acesso ao repositório privado {url}: {(erro.stderr or '').strip()[-200:]}") from None
    git = git_em(clone)
    git("pull", "-q", "--ff-only", "origin", RAMO)
    manchetes = gerar()
    texto = para_json(manchetes, agora)
    arquivo = clone / ARQUIVO
    antes = json.loads(arquivo.read_text(encoding="utf-8"))["manchetes"] if arquivo.exists() else None
    if antes != json.loads(texto)["manchetes"]:        # a new publish time alone is no change
        arquivo.write_text(texto + "\n", encoding="utf-8", newline="\n")
    if not pendentes(git, [ARQUIVO]):
        return "manchetes: em dia"
    commitar(git, [ARQUIVO], f"manchetes: {len(manchetes)} de {agora:%d/%m/%Y %H:%M}")
    publicar(git)
    return f"manchetes: {len(manchetes)} publicada(s)"


def _gerar_manchetes(pasta_do_log: Path) -> pd.DataFrame:
    from core.manchetes_online import QUANTAS_MANCHETES
    from core.sentimento import ler_log, manchetes
    return manchetes(*ler_log(pasta_do_log, date.today() - timedelta(days=3)), quantas=QUANTAS_MANCHETES)


def _gerar_painel(raiz: Path, pasta_do_log: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from ativos_tela import ler_nomes
    from scripts.painel_do_app import gerar, tickers_do_painel
    return gerar(pasta_do_log, raiz / PASTA_DO_LAGO, tickers_do_painel(ler_nomes(raiz / ARQUIVO_ATIVOS)), date.today())


def executar(raiz: Path, git: Git, trava_do_lago: Path, agora: datetime, ler_leitura: Callable[[], str],
             tentativas: Path = PASTA_ESTADO / "tentativas", pasta_do_log: Path | None = None,
             manchetes: LojaDeManchetes | None = None, ler_leitura_v3: Callable[[], str] | None = None,
             gerar_curvas: Callable[[], object] | None = None) -> Resultado:
    """One run: the due readings, the liquidity snapshot and the push, each on its own (a failure in one does not
    stop the others); ``ler_leitura`` runs ``paper_v2.py --leitura`` and returns its output; ``pasta_do_log``: the
    lake's news log (None: the panel step does nothing); ``manchetes``: the arguments of ``atualizar_manchetes`` but
    the time (None: no headlines step); ``ler_leitura_v3`` runs ``paper_v3.py --leitura`` (None: no paper-v3 step);
    ``gerar_curvas`` runs ``scripts/curvas_ao_vivo.py`` (None: no live curves step)."""
    if trava_do_lago.exists():
        return Resultado("lago rodando", "", [])
    try:
        conferir_checkout(raiz, git)
    except Bloqueio as bloqueio:
        return Resultado("bloqueada", str(bloqueio), [])
    except Exception as erro:           # noqa: BLE001 - git missing, a broken repository, a timeout
        return Resultado("falhou", f"checkout: {type(erro).__name__}: {erro}"[:300], [])
    avisos: list[tuple[str, str]] = []
    notas, desfecho = [], "ok"
    passos = [("leituras", lambda: _leituras(raiz, git, agora, ler_leitura, tentativas, avisos))]
    if ler_leitura_v3 is not None:
        passos.append(("leituras paper-v3",
                       lambda: _leituras(raiz, git, agora, ler_leitura_v3, tentativas, avisos, PAPER_V3)))
    passos += [("liquidez", lambda: _liquidez(raiz, git)),
              ("pregões recentes", lambda: atualizar_recentes(raiz, git)),
              ("ativos", lambda: atualizar_ativos(raiz, git)),          # before the panel, which reads the list
              ("painel", lambda: atualizar_painel(raiz, git, pasta_do_log, lambda: _gerar_painel(raiz, pasta_do_log)))]
    if gerar_curvas is not None:
        passos.append(("curvas ao vivo", lambda: atualizar_curvas(raiz, git, gerar_curvas)))
    if manchetes is not None:
        passos.append(("manchetes", lambda: atualizar_manchetes(*manchetes, agora)))
    passos.append(("push", lambda: publicar(git)))
    for nome, passo in passos:
        try:
            notas.append(passo())
        except Bloqueio as bloqueio:
            notas.append(f"{nome}: {bloqueio}")
            desfecho = "bloqueada" if desfecho == "ok" else desfecho
        except Exception as erro:       # noqa: BLE001 - the run goes on; the watchdog reports the line
            notas.append(f"{nome}: {type(erro).__name__}: {erro}"[:300])
            desfecho = "falhou"
    return Resultado(desfecho, "; ".join(notas), avisos)


def esperar(rodando: Callable[[], bool], dormir: Callable[[float], None], pausa: float = 180, passo: float = 30,
            limite: float = 2_700) -> None:
    """Waits ``pausa`` (late tasks start together after the PC wakes), then while ``rodando``, up to ``limite``."""
    dormir(pausa)
    esperado = 0.0
    while rodando() and esperado < limite:
        dormir(passo)
        esperado += passo


def ler_leitura(raiz: Path, script: str = "scripts/paper_v2.py") -> str:
    """The output of ``script --leitura`` (paper-v2's or paper-v3's) run in the checkout ``raiz`` (it runs the lake's
    ``universe`` first); ``RuntimeError`` with its error output when it fails."""
    # python.exe next to the task's pythonw.exe: pythonw has no stdout to pipe
    python = Path(sys.executable).with_name("python.exe")
    feita = subprocess.run([str(python if python.exists() else sys.executable), "-X", "utf8", script, "--leitura"],
                           cwd=raiz, capture_output=True, text=True, encoding="utf-8", timeout=7_200,
                           creationflags=_SEM_JANELA)
    if feita.returncode:
        raise RuntimeError(f"{Path(script).name} --leitura saiu com {feita.returncode}: {feita.stderr.strip()[-300:]}")
    return feita.stdout


def rodar_curvas(raiz: Path) -> None:
    """Runs ``scripts/curvas_ao_vivo.py`` in the checkout ``raiz`` on its lake copy; ``RuntimeError`` on failure."""
    python = Path(sys.executable).with_name("python.exe")
    feita = subprocess.run([str(python if python.exists() else sys.executable), "-X", "utf8",
                            "scripts/curvas_ao_vivo.py"], cwd=raiz, capture_output=True, text=True, encoding="utf-8",
                           timeout=3_600, creationflags=_SEM_JANELA)
    if feita.returncode:
        raise RuntimeError(f"curvas_ao_vivo.py saiu com {feita.returncode}: {feita.stderr.strip()[-300:]}")


def tarefas_rodando(nomes: tuple[str, ...]) -> int:
    """How many of the scheduled tasks ``nomes`` are running now (a missing task counts as not running)."""
    comando = ("$nomes = $env:TAREFAS -split '\\|'; "
               "@(Get-ScheduledTask -TaskName $nomes -ErrorAction SilentlyContinue | "
               "Where-Object State -eq 'Running').Count")
    ambiente = {**os.environ, "TAREFAS": "|".join(nomes)}
    feita = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", comando], check=True,
                           capture_output=True, text=True, encoding="utf-8", env=ambiente, timeout=120,
                           creationflags=_SEM_JANELA)
    return int(feita.stdout.strip() or 0)


def avisar(raiz: Path, titulo: str, corpo: str, mostrar: bool = False) -> str:
    """A Windows notification through ``scripts/avisar.ps1`` (``mostrar``: only print "title | body", for tests)."""
    ambiente = {**os.environ, "AVISO_TITULO": titulo, "AVISO_CORPO": corpo}
    comando = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
               str(raiz / "scripts" / "avisar.ps1"), *(["-SoMostrar"] if mostrar else [])]
    return subprocess.run(comando, check=True, capture_output=True, text=True, encoding="utf-8", env=ambiente,
                          timeout=120, creationflags=_SEM_JANELA).stdout.rstrip("\n")


def notificar(resultado: Resultado, avisar_com: Callable[[str, str], object]) -> Resultado:
    """Sends each notification of the run; a failed one makes the run "falhou" (the watchdog then says so)."""
    falhas = []
    for titulo, corpo in resultado.avisos:
        try:
            avisar_com(titulo, corpo)
        except Exception as erro:       # noqa: BLE001 - the other notifications and the runs.csv line still go
            falhas.append(f"aviso: {type(erro).__name__}: {erro}"[:200])
    if not falhas:
        return resultado
    return Resultado("falhou", "; ".join([resultado.nota, *falhas]), resultado.avisos)


def anotar(pasta: Path, agora: datetime, resultado: Resultado) -> None:
    """One line in ``pasta/runs.csv`` (the header of the launcher's "manutencao" routine), read by the watchdog."""
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo = pasta / "runs.csv"
    novo = not arquivo.exists()
    with arquivo.open("a", newline="", encoding="utf-8") as saida:
        escritor = csv.writer(saida, lineterminator="\n")
        if novo:
            escritor.writerow(["quando", "desfecho", "nota"])
        escritor.writerow([agora.isoformat(timespec="seconds"), resultado.desfecho, resultado.nota])


def main() -> int:
    agora = datetime.now()
    esperar(lambda: tarefas_rodando(ROTINAS_ANTES) > 0, time.sleep)
    resultado = executar(RAIZ, git_em(RAIZ), TRAVA_DO_LAGO, agora, lambda: ler_leitura(RAIZ), pasta_do_log=PASTA_DO_LOG,
                         manchetes=(CLONE_DAS_MANCHETES, URL_DAS_MANCHETES, PASTA_DO_LOG,
                                    lambda: _gerar_manchetes(PASTA_DO_LOG)),
                         ler_leitura_v3=lambda: ler_leitura(RAIZ, "scripts/paper_v3.py"),
                         gerar_curvas=lambda: rodar_curvas(RAIZ))
    resultado = notificar(resultado, lambda titulo, corpo: avisar(RAIZ, titulo, corpo))
    anotar(PASTA_ESTADO, agora, resultado)
    print(f"{resultado.desfecho}: {resultado.nota}".strip())
    return 0 if resultado.desfecho == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
