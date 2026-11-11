"""
qml_holdout_diario.py - the unattended daily run of the two QML forward-holdout ledgers.

``scripts/qml_holdout.py`` keeps each account's ledger (``simulador/qml_holdout/`` for the
Ledoit-Wolf account, ``simulador/qml_holdout_pca/`` for the PCA one) but only moves it when someone
calls it by hand. This script is what a hidden daily task calls: for every account it runs ``rodar``
when the account already has an ``estado.json`` and ``iniciar`` when it does not; an account whose
pre-registered window has not started yet (the lake has no session on or after its start date) is
reported as "aguardando" and does not fail the run. At the end it commits exactly the ledgers that
exist and pushes, so the holdout leaves this branch.

Usage: .venv-qpo/Scripts/python -X utf8 scripts/qml_holdout_diario.py --lago <lake> [--contas ledoit-wolf,pca]
       [--sem-push]

It decides nothing about the portfolio: the model, the rules and the accounts all come from
``qml_holdout.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(RAIZ))

from scripts.qml_holdout import CONTAS, Conta, SessionNotStarted, iniciar, rodar  # noqa: E402

PASTAS_DO_LEDGER = ("simulador/qml_holdout", "simulador/qml_holdout_pca")
LOG = Path("logs") / "qml_holdout_diario.log"


@dataclass(frozen=True)
class Resultado:
    """What one account did in this run: the action taken, its marks, or why it is still waiting."""

    conta: Conta
    acao: str                      # "rodar", "iniciar" or "aguardando"
    marcas: int | None = None      # marks in the ledger after the run, when it ran
    mensagem: str = ""             # why the account did not run, when it waited


def _marcas(conta: Conta) -> int:
    """Rows in the account's marks file, header excluded; 0 when there is no file yet."""
    if not conta.marcas.exists():
        return 0
    with conta.marcas.open(encoding="utf-8") as arquivo:
        return max(sum(1 for _ in arquivo) - 1, 0)


def atualizar(lago: Path, contas: Sequence[Conta], rodar: Callable[[Path, Conta], object] = rodar,
              iniciar: Callable[[Path, Conta], object] = iniciar,
              hoje: date | None = None) -> list[Resultado]:
    """Run each account in order: ``rodar`` when its ledger exists, ``iniciar`` when it does not.

    An account whose pre-registered start date is still in the future waits without touching the lake. Only
    :class:`SessionNotStarted` from ``iniciar`` means "the window has not started": any other exception belongs
    to a broken lake or model and propagates.
    """
    hoje = hoje or datetime.now().date()
    resultados: list[Resultado] = []
    for conta in contas:
        if conta.estado.exists():
            rodar(lago, conta)
            resultados.append(Resultado(conta, "rodar", marcas=_marcas(conta)))
            continue
        if getattr(conta, "inicio", None) is not None and conta.inicio.date() > hoje:
            resultados.append(Resultado(conta, "aguardando",
                                        mensagem=f"starts on {conta.inicio:%d/%m/%Y}"))
            continue
        try:
            iniciar(lago, conta=conta)
        except SessionNotStarted as erro:
            resultados.append(Resultado(conta, "aguardando", mensagem=str(erro)))
        else:
            resultados.append(Resultado(conta, "iniciar", marcas=_marcas(conta)))
    return resultados


def _a_enviar(git: Callable[..., subprocess.CompletedProcess]) -> int:
    """Commits local to the branch that the upstream does not have; 1 when there is no upstream yet."""
    try:
        return int(git("rev-list", "--count", "@{u}..HEAD").stdout.strip() or 0)
    except subprocess.CalledProcessError:
        return 1


PUSH_TENTATIVAS = 3
PUSH_ESPERA_S = 60.0          # doubled after each failed push: 60 s, then 120 s


def _push(git: Callable[..., subprocess.CompletedProcess], espera: Callable[[float], None]) -> None:
    """``git push`` with retries; when every try fails, raise with what git printed (the exit code alone says
    nothing: the 2026-11-10 23:00 run logged only "exit status 128" and a manual push a minute later worked)."""
    for tentativa in range(1, PUSH_TENTATIVAS + 1):
        try:
            git("push")
            return
        except subprocess.CalledProcessError as erro:
            if tentativa == PUSH_TENTATIVAS:
                detalhe = (erro.stderr or erro.stdout or "").strip() or f"exit status {erro.returncode}"
                raise RuntimeError(f"git push failed {PUSH_TENTATIVAS} times: {detalhe}") from erro
            espera(PUSH_ESPERA_S * 2 ** (tentativa - 1))


def publicar(raiz: Path, mensagem: str, sem_push: bool = False,
             executar: Callable[..., subprocess.CompletedProcess] = subprocess.run,
             espera: Callable[[float], None] = time.sleep) -> bool:
    """Stage the ledgers that exist, commit them with ``mensagem`` and push unless ``sem_push``.

    A day with no new marks still pushes commits left behind by an earlier failed push. Returns False when the
    ledgers have nothing new and the branch has nothing to send: no empty commit, no push.
    """
    def git(*argumentos: str) -> subprocess.CompletedProcess:
        return executar(["git", *argumentos], check=True, capture_output=True, text=True, cwd=raiz)

    caminhos = [pasta for pasta in PASTAS_DO_LEDGER if (raiz / pasta).exists()]
    if not caminhos:
        return False
    git("add", "--", *caminhos)
    mudou = bool(git("status", "--porcelain", "--", *caminhos).stdout.strip())
    if mudou:
        git("commit", "-q", "-m", mensagem, "--", *caminhos)
    if sem_push or not _a_enviar(git):
        return mudou
    _push(git, espera)
    return True


def _nome(conta: Conta) -> str:
    """The account's registry name, or its folder name for an account built outside ``CONTAS``."""
    for nome, registrada in CONTAS.items():
        if registrada is conta:
            return nome
    return conta.pasta.name


def _linha(resultado: Resultado) -> str:
    if resultado.acao == "aguardando":
        return f"{_nome(resultado.conta)}: aguardando ({resultado.mensagem})"
    return f"{_nome(resultado.conta)}: {resultado.acao}, {resultado.marcas} marca(s)"


def _registrar(caminho: Path, linha: str) -> None:
    """Append the line to the log file and print it, so the run leaves a trace either way."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("a", encoding="utf-8") as arquivo:
        arquivo.write(linha + "\n")
    print(linha)


def _contas(texto: str) -> list[str]:
    """Comma-separated account names, each one from ``CONTAS``; argparse turns a bad name into exit code 2."""
    nomes = [nome.strip() for nome in texto.split(",") if nome.strip()]
    desconhecidas = [nome for nome in nomes if nome not in CONTAS]
    if desconhecidas:
        raise argparse.ArgumentTypeError(
            f"unknown account(s) {', '.join(desconhecidas)}: choose from {', '.join(CONTAS)}")
    if not nomes:
        raise argparse.ArgumentTypeError(f"no account given: choose from {', '.join(CONTAS)}")
    return nomes


def _analisar(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily run of the QML forward-holdout ledgers.")
    parser.add_argument("--lago", type=Path, required=True, help="the frozen lake the accounts read")
    parser.add_argument("--contas", type=_contas, default=list(CONTAS),
                        help="comma-separated account names (default: all of them)")
    parser.add_argument("--sem-push", action="store_true", help="commit the ledgers but do not push")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run every requested account, log each outcome, publish the ledgers; 0 on success, 1 on any failure.

    A failure on one account does not throw away the marks the earlier accounts already wrote: publication
    happens after the loop, even when the loop stopped on an error.
    """
    argumentos = _analisar(argv)
    log = RAIZ / LOG
    try:
        contas = [CONTAS[nome] for nome in argumentos.contas]
        resultados: list[Resultado] = []
        falha: Exception | None = None
        for conta in contas:
            try:
                resultados.extend(atualizar(argumentos.lago, [conta]))
            except Exception as erro:  # noqa: BLE001 - recorded, then re-raised as the exit code
                falha = erro
                break
        for resultado in resultados:
            _registrar(log, _linha(resultado))
        if falha is not None:
            _registrar(log, f"falhou: {falha}")
        resumo = "; ".join(f"{_nome(r.conta)} {r.acao}" for r in resultados) or "no account ran"
        mensagem = f"qml holdout {datetime.now():%Y-%m-%d}: {resumo}"
        publicou = publicar(RAIZ, mensagem, sem_push=argumentos.sem_push)
        _registrar(log, "publicado" if publicou else "nada a publicar")
        return 1 if falha is not None else 0
    except Exception as erro:  # noqa: BLE001 - the daily task must report, never crash
        _registrar(log, f"falhou: {erro}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
