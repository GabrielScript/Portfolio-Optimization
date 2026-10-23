"""
Daily upkeep of the Fundo Alepha (task "paper manutencao", weekdays at 10:00, through ``scripts/rodar_paper.py
manutencao``), so nothing after v1.0 depends on the user remembering it.

Usage: .venv/Scripts/python -X utf8 scripts/manutencao.py

Each run, in the app checkout on ``main``, after the lake and the paper routines have finished:
- paper-v2 readings: with 126 and 252 sessions after the first record (the pre-registered report and decision of
  ``scripts/paper_v2.py``), and once a month as information only (DECISOES 121), it runs ``paper_v2.py --leitura`` and
  saves its output to ``paper-v2/leituras/``; a reading counts only when its own session count reaches the target;
- the app's liquidity snapshot (``scripts/liquidez_do_app.py``) when the lake has a newer closed month;
- a commit of exactly those paths and a fast-forward push of ``main`` (never forced), which also publishes the paper
  routines' local commits to the deployed app.
It decides nothing about the portfolio. Outcomes go to ``%LOCALAPPDATA%/paper-manutencao/runs.csv``, which the 12h
watchdog reads; saved readings also raise a Windows notification.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

Git = Callable[..., str]

LEITURAS = (126, 252)           # the pre-registered report and decision of paper_v2.py
RAMO = "main"                   # the branch the scheduled routines run on
# relative to the app checkout
ARQUIVO_CALENDARIO = Path("lago/precos_diarios.parquet")
ARQUIVO_REGISTRO = Path("paper-v2/carteiras.csv")
PASTA_LEITURAS = Path("paper-v2/leituras")
ARQUIVO_LIQUIDEZ = Path("lago/liquidez.parquet")
PASTA_DO_LAGO = Path("data/b3lake")             # the lake's exports (outside git)
_SEM_JANELA = getattr(subprocess, "CREATE_NO_WINDOW", 0)
RAIZ = Path(__file__).resolve().parent.parent
_LOCAL = Path(os.environ.get("LOCALAPPDATA", Path.home()))
PASTA_ESTADO = _LOCAL / "paper-manutencao"     # runs.csv, outside the repository (the launcher's "manutencao")
TRAVA_DO_LAGO = _LOCAL / "b3lake" / "daily.lock"
ROTINAS_ANTES = ("b3lake daily", "paper trading", "paper v2")     # the upkeep waits for them to finish


class Bloqueio(Exception):
    """A state only the user should resolve (another branch, a git operation in course, ``main`` behind origin)."""


def pregoes_desde_o_registro(calendario: pd.DatetimeIndex, registro: pd.DataFrame) -> int:
    """Sessions of the lake's calendar after the first registered close (as ``paper_v2.py --leitura`` counts them)."""
    if registro.empty:
        return 0
    return int((calendario > registro["data"].min()).sum())


def leituras_devidas(pregoes: int, pasta: Path, hoje: date) -> dict[str, int]:
    """The readings not yet saved in ``pasta`` whose target was reached, name -> sessions the reading itself must
    count: the pre-registered ones at 126 and 252 sessions and this month's progress (information only)."""
    alvos = {f"leitura-{alvo}": alvo for alvo in LEITURAS} | {f"progresso-{hoje:%Y-%m}": 1}
    return {nome: alvo for nome, alvo in alvos.items()
            if pregoes >= alvo and not (pasta / f"{nome}.txt").exists()}


def pregoes_da_saida(texto: str) -> int | None:
    """The session count in the first line of ``paper_v2.py --leitura`` ("...: 126 pregões, ..."); None without it."""
    achado = re.search(r": (\d+) pregões", texto.splitlines()[0] if texto else "")
    return int(achado.group(1)) if achado else None


def salvar_leituras(texto: str, devidas: dict[str, int], pasta: Path, agora: datetime) -> list[Path]:
    """Saves the reading as each due name whose target its own session count reached; the others go to
    ``tentativas/`` with the time (the run retries them). Returns the saved readings."""
    pregoes = pregoes_da_saida(texto)
    feitas = []
    for nome, alvo in devidas.items():
        if pregoes is not None and pregoes >= alvo:
            destino = pasta / f"{nome}.txt"
            feitas.append(destino)
        else:
            destino = pasta / "tentativas" / f"{nome}-{agora:%Y%m%d-%H%M%S}.txt"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(texto, encoding="utf-8", newline="\n")
    return feitas


def liquidez_nova(atual: Path, origem: Path) -> pd.DataFrame | None:
    """The snapshot of ``scripts/liquidez_do_app.py`` from the lake's exports in ``origem`` when its month is newer
    than the one in ``atual`` (or there is no snapshot yet); None otherwise."""
    from scripts.liquidez_do_app import instantaneo
    # the lake writes the manifest in cp1252; only the dates, which are ASCII, matter
    manifest = json.loads((origem / "manifest.json").read_bytes().decode("utf-8", errors="replace"))
    metricas = pd.read_parquet(origem / "monthly_metrics.parquet",
                               columns=["month", "security", "quoted_spread", "vol_daily", "median_dtv_brl"])
    foto = instantaneo(metricas, date.fromisoformat(manifest["data_through"]))
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


def commitar(git: Git, caminhos: list[str], mensagem: str) -> bool:
    """Commits exactly ``caminhos`` (never ``add -A``: the root has loose untracked files); False when they did not
    change."""
    git("add", "--", *caminhos)
    if not git("diff", "--cached", "--name-only", "--", *caminhos):
        return False
    git("commit", "-q", "-m", mensagem, "--", *caminhos)
    return True


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


def _titulo(nome: str) -> str:
    if nome.startswith("leitura-"):
        return f"Fundo Alepha: leitura de {nome.removeprefix('leitura-')} pregões"
    ano, mes = nome.removeprefix("progresso-").split("-")
    return f"Fundo Alepha: progresso de {mes}/{ano}"


def _leituras(raiz: Path, git: Git, agora: datetime, ler_leitura: Callable[[], str],
              avisos: list[tuple[str, str]]) -> str:
    calendario = pd.DatetimeIndex(pd.read_parquet(raiz / ARQUIVO_CALENDARIO, columns=["date"])["date"])
    arquivo = raiz / ARQUIVO_REGISTRO
    registro = pd.read_csv(arquivo, parse_dates=["data"]) if arquivo.exists() else pd.DataFrame({"data": []})
    devidas = leituras_devidas(pregoes_desde_o_registro(calendario, registro), raiz / PASTA_LEITURAS, agora.date())
    if not devidas:
        return "leituras: nenhuma devida"
    texto = ler_leitura()
    feitas = salvar_leituras(texto, devidas, raiz / PASTA_LEITURAS, agora)
    if not feitas:
        return f"leituras: {', '.join(devidas)} sem os pregões na saída, de novo na próxima rodada"
    commitar(git, [caminho.relative_to(raiz).as_posix() for caminho in feitas],
             f"paper-v2: {', '.join(caminho.stem for caminho in feitas)} ({pregoes_da_saida(texto)} pregões)")
    resumo = "\n".join(linha for linha in texto.splitlines() if linha.startswith("Critério"))
    avisos.extend((_titulo(caminho.stem), resumo or texto.splitlines()[0]) for caminho in feitas)
    return f"leituras: {', '.join(caminho.stem for caminho in feitas)}"


def _liquidez(raiz: Path, git: Git) -> str:
    foto = liquidez_nova(raiz / ARQUIVO_LIQUIDEZ, raiz / PASTA_DO_LAGO)
    if foto is None:
        return "liquidez: em dia"
    (raiz / ARQUIVO_LIQUIDEZ).parent.mkdir(parents=True, exist_ok=True)
    foto.to_parquet(raiz / ARQUIVO_LIQUIDEZ, index=False)
    mes = f"{foto['month'].iloc[0]:%m/%Y}"
    commitar(git, [ARQUIVO_LIQUIDEZ.as_posix()], f"liquidez: snapshot de {mes} (manutenção)")
    return f"liquidez: {mes}"


def executar(raiz: Path, git: Git, trava_do_lago: Path, agora: datetime,
             ler_leitura: Callable[[], str]) -> Resultado:
    """One run: the due readings, the liquidity snapshot and the push, each on its own (a failure in one does not
    stop the others); ``ler_leitura`` runs ``paper_v2.py --leitura`` and returns its output."""
    if trava_do_lago.exists():
        return Resultado("lago rodando", "", [])
    try:
        conferir_checkout(raiz, git)
    except Bloqueio as bloqueio:
        return Resultado("bloqueada", str(bloqueio), [])
    avisos: list[tuple[str, str]] = []
    notas, desfecho = [], "ok"
    passos = (("leituras", lambda: _leituras(raiz, git, agora, ler_leitura, avisos)),
              ("liquidez", lambda: _liquidez(raiz, git)),
              ("push", lambda: publicar(git)))
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


def ler_leitura(raiz: Path) -> str:
    """The output of ``scripts/paper_v2.py --leitura`` run in the checkout ``raiz`` (it runs the lake's ``universe``
    first); ``RuntimeError`` with its error output when it fails."""
    # python.exe next to the task's pythonw.exe: pythonw has no stdout to pipe
    python = Path(sys.executable).with_name("python.exe")
    feita = subprocess.run([str(python if python.exists() else sys.executable), "-X", "utf8", "scripts/paper_v2.py",
                            "--leitura"], cwd=raiz, capture_output=True, text=True, encoding="utf-8",
                           timeout=7_200, creationflags=_SEM_JANELA)
    if feita.returncode:
        raise RuntimeError(f"paper_v2.py --leitura saiu com {feita.returncode}: {feita.stderr.strip()[-300:]}")
    return feita.stdout


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
    resultado = executar(RAIZ, git_em(RAIZ), TRAVA_DO_LAGO, agora, lambda: ler_leitura(RAIZ))
    resultado = notificar(resultado, lambda titulo, corpo: avisar(RAIZ, titulo, corpo))
    anotar(PASTA_ESTADO, agora, resultado)
    print(f"{resultado.desfecho}: {resultado.nota}".strip())
    return 0 if resultado.desfecho == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
