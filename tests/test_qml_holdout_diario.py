"""Tests for the unattended daily run of the two QML forward-holdout ledgers."""

from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.qml_holdout import CONTA_PCA, SessionNotStarted
from scripts.qml_holdout_diario import Resultado, atualizar, main, publicar


def _conta(tmp_path: Path, nome: str, com_estado: bool, marcas: int = 0,
           inicio: str = "2026-11-09") -> SimpleNamespace:
    """A stand-in for ``Conta``: its folder, its estado path and a marks file with ``marcas`` rows."""
    pasta = tmp_path / nome
    pasta.mkdir(parents=True)
    if com_estado:
        (pasta / "estado.json").write_text("{}", encoding="utf-8")
    if marcas:
        linhas = ["data,valor", *[f"2026-11-{dia:02d},10000" for dia in range(9, 9 + marcas)]]
        (pasta / "marcas.csv").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return SimpleNamespace(pasta=pasta, estado=pasta / "estado.json", marcas=pasta / "marcas.csv",
                           inicio=pd.Timestamp(inicio))


def _fakes(chamadas: list[tuple[str, object, object]]):
    """The two injected run functions: they record the call and leave the fake account untouched."""
    def fake_rodar(lago, conta):
        chamadas.append(("rodar", lago, conta))

    def fake_iniciar(lago, conta):
        chamadas.append(("iniciar", lago, conta))

    return fake_rodar, fake_iniciar


def test_a_started_ledger_runs_and_a_missing_one_starts(tmp_path) -> None:
    """The estado file alone decides the command: present runs, absent starts the account."""
    iniciada = _conta(tmp_path, "iniciada", com_estado=True, marcas=2)
    nova = _conta(tmp_path, "nova", com_estado=False)
    chamadas: list = []
    fake_rodar, fake_iniciar = _fakes(chamadas)

    resultados = atualizar(Path("lago"), [iniciada, nova], rodar=fake_rodar, iniciar=fake_iniciar,
                           hoje=date(2026, 11, 10))

    assert [r.acao for r in resultados] == ["rodar", "iniciar"]
    assert chamadas == [("rodar", Path("lago"), iniciada), ("iniciar", Path("lago"), nova)]
    assert resultados[0].marcas == 2
    assert resultados[0].mensagem == ""


def test_a_folder_without_estado_still_starts_the_account(tmp_path) -> None:
    """An empty folder is not a started account: the decidir is ``estado.json``, not the directory."""
    pasta = tmp_path / "vazia"
    pasta.mkdir()
    conta = SimpleNamespace(pasta=pasta, estado=pasta / "estado.json", marcas=pasta / "marcas.csv",
                            inicio=pd.Timestamp("2026-11-09"))
    chamadas: list = []
    fake_rodar, fake_iniciar = _fakes(chamadas)

    resultados = atualizar(Path("lago"), [conta], rodar=fake_rodar, iniciar=fake_iniciar,
                           hoje=date(2026, 11, 10))

    assert [r.acao for r in resultados] == ["iniciar"]


def test_a_future_start_date_waits_without_touching_the_lake(tmp_path) -> None:
    """A start date still ahead is decided from the date alone: no lake load, no model training."""
    nova = _conta(tmp_path, "futura", com_estado=False, inicio="2026-12-01")

    def explode(*_argumentos):
        raise AssertionError("the lake must not be touched while the window has not started")

    resultados = atualizar(Path("lago"), [nova], rodar=explode, iniciar=explode,
                           hoje=date(2026, 11, 10))

    assert [r.acao for r in resultados] == ["aguardando"]
    assert "01/12/2026" in resultados[0].mensagem


def test_the_account_reaches_iniciar_as_a_keyword(tmp_path) -> None:
    """``iniciar(lago, capital, conta)``: passing the account positionally would make it the capital."""
    from scripts.qml_holdout import iniciar as iniciar_real

    nova = _conta(tmp_path, "nova", com_estado=False)
    recebidas: dict = {}

    def fake_iniciar(lago, capital=10_000.0, conta=CONTA_PCA):
        recebidas["capital"] = capital
        recebidas["conta"] = conta

    atualizar(Path("lago"), [nova], rodar=lambda *a: None, iniciar=fake_iniciar,
              hoje=date(2026, 11, 10))

    assert recebidas == {"capital": 10_000.0, "conta": nova}
    assert iniciar_real.__defaults__[0] == 10_000.0


def test_an_unstarted_window_waits_without_failing(tmp_path) -> None:
    """``SessionNotStarted`` means the lake has no session yet: "aguardando", not a failure."""
    nova = _conta(tmp_path, "nova", com_estado=False)
    chamadas: list = []
    fake_rodar, _ = _fakes(chamadas)

    def fake_iniciar(lago, conta):
        raise SessionNotStarted("no session on or after 10/11/2026")

    resultados = atualizar(Path("lago"), [nova], rodar=fake_rodar, iniciar=fake_iniciar,
                           hoje=date(2026, 11, 10))

    assert [r.acao for r in resultados] == ["aguardando"]
    assert "no session on or after 10/11/2026" in resultados[0].mensagem
    assert not nova.estado.exists()


def test_a_plain_value_error_from_iniciar_propagates(tmp_path) -> None:
    """A broken lake (bad prices, short history) raises ValueError too and must NOT read as "waiting"."""
    nova = _conta(tmp_path, "nova", com_estado=False)

    def fake_iniciar(lago, conta):
        raise ValueError("no usable price for ['PETR4']")

    with pytest.raises(ValueError, match="no usable price"):
        atualizar(Path("lago"), [nova], rodar=lambda *a: None, iniciar=fake_iniciar,
                  hoje=date(2026, 11, 10))


def test_any_other_failure_propagates(tmp_path) -> None:
    """A broken lake must fail the run."""
    nova = _conta(tmp_path, "nova", com_estado=False)

    def fake_iniciar(lago, conta):
        raise RuntimeError("lake offline")

    with pytest.raises(RuntimeError):
        atualizar(Path("lago"), [nova], rodar=lambda *a: None, iniciar=fake_iniciar,
                  hoje=date(2026, 11, 10))


class _Executar:
    """A fake ``subprocess.run`` that records the git commands and answers a scripted status."""

    def __init__(self, status: str, pendentes: str = "0") -> None:
        self.chamadas: list[list[str]] = []
        self.argumentos: list[dict] = []
        self._status = status
        self._pendentes = pendentes

    def __call__(self, argumentos, **kwargs):
        self.chamadas.append(list(argumentos))
        self.argumentos.append(kwargs)
        if argumentos[1] == "status":
            stdout = self._status
        elif argumentos[1] == "rev-list":
            commitou = any(chamada[1] == "commit" for chamada in self.chamadas)
            stdout = "1\n" if commitou else self._pendentes
        else:
            stdout = ""
        return SimpleNamespace(stdout=stdout, returncode=0)


def test_publicar_stages_only_the_ledgers_that_exist(tmp_path) -> None:
    """A ledger folder that was never created is not a path to stage."""
    (tmp_path / "simulador" / "qml_holdout_pca").mkdir(parents=True)
    executar = _Executar("?? simulador/qml_holdout_pca/marcas.csv\n")

    assert publicar(tmp_path, "msg", executar=executar) is True

    add = executar.chamadas[0]
    assert add[:2] == ["git", "add"]
    assert add[2:] == ["--", "simulador/qml_holdout_pca"]
    assert executar.argumentos[0] == {"check": True, "capture_output": True, "text": True, "cwd": tmp_path}


def test_publicar_with_nothing_staged_does_not_commit_or_push(tmp_path) -> None:
    """No empty commits: a clean status ends the publication (nothing pending upstream)."""
    (tmp_path / "simulador" / "qml_holdout").mkdir(parents=True)
    executar = _Executar("")

    assert publicar(tmp_path, "msg", executar=executar) is False

    assert [c[1] for c in executar.chamadas] == ["add", "status", "rev-list"]


def test_publicar_pushes_commits_left_behind_by_an_earlier_failed_push(tmp_path) -> None:
    """A quiet day still sends what a previous run committed but could not push."""
    (tmp_path / "simulador" / "qml_holdout").mkdir(parents=True)
    executar = _Executar("", pendentes="2\n")

    assert publicar(tmp_path, "msg", executar=executar) is True

    assert [c[1] for c in executar.chamadas] == ["add", "status", "rev-list", "push"]


def test_publicar_adds_commits_and_pushes_in_that_order(tmp_path) -> None:
    (tmp_path / "simulador" / "qml_holdout").mkdir(parents=True)
    executar = _Executar("A  simulador/qml_holdout/marcas.csv\n")

    assert publicar(tmp_path, "qml holdout 2026-11-10", executar=executar) is True

    assert [c[1] for c in executar.chamadas] == ["add", "status", "commit", "rev-list", "push"]
    commit = executar.chamadas[2]
    assert commit[:2] == ["git", "commit"]
    assert "qml holdout 2026-11-10" in commit


class _PushFalha(_Executar):
    """Like ``_Executar``, but the first ``falhas`` pushes fail as git does when the network drops."""

    def __init__(self, status: str, falhas: int) -> None:
        super().__init__(status)
        self._falhas = falhas

    def __call__(self, argumentos, **kwargs):
        if argumentos[1] == "push" and self._falhas:
            self._falhas -= 1
            self.chamadas.append(list(argumentos))
            raise subprocess.CalledProcessError(128, argumentos, stderr="fatal: unable to access 'https://...': "
                                                                         "Could not resolve host: github.com\n")
        return super().__call__(argumentos, **kwargs)


def test_publicar_retries_a_push_that_fails_once(tmp_path) -> None:
    """The 2026-11-10 23:00 run lost its push to a one-off failure; a manual push a minute later worked."""
    (tmp_path / "simulador" / "qml_holdout").mkdir(parents=True)
    executar = _PushFalha("A  simulador/qml_holdout/marcas.csv\n", falhas=1)

    assert publicar(tmp_path, "msg", executar=executar, espera=lambda segundos: None) is True

    assert [c[1] for c in executar.chamadas].count("push") == 2


def test_publicar_reports_what_git_said_when_every_push_fails(tmp_path) -> None:
    """The log must say why: 'returned non-zero exit status 128' alone cannot be diagnosed."""
    (tmp_path / "simulador" / "qml_holdout").mkdir(parents=True)
    executar = _PushFalha("A  simulador/qml_holdout/marcas.csv\n", falhas=5)

    with pytest.raises(RuntimeError, match="Could not resolve host"):
        publicar(tmp_path, "msg", executar=executar, espera=lambda segundos: None)


def test_publicar_sem_push_does_not_push(tmp_path) -> None:
    (tmp_path / "simulador" / "qml_holdout").mkdir(parents=True)
    executar = _Executar("A  simulador/qml_holdout/marcas.csv\n")

    assert publicar(tmp_path, "msg", sem_push=True, executar=executar) is True

    assert [c[1] for c in executar.chamadas] == ["add", "status", "commit"]


def _git(raiz: Path, *argumentos: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(raiz), *argumentos], check=True, capture_output=True, text=True)


def test_publicar_commits_once_in_a_real_repository(tmp_path) -> None:
    """A real (freshly initialised) repository: one commit with the message, no remote needed."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    ledger = tmp_path / "simulador" / "qml_holdout"
    ledger.mkdir(parents=True)
    (ledger / "marcas.csv").write_text("data,valor\n2026-11-09,10000\n", encoding="utf-8")

    assert publicar(tmp_path, "qml holdout 2026-11-10", sem_push=True) is True

    assert _git(tmp_path, "log", "--format=%s").stdout.strip() == "qml holdout 2026-11-10"
    assert _git(tmp_path, "rev-list", "--count", "HEAD").stdout.strip() == "1"
    assert _git(tmp_path, "status", "--porcelain").stdout == ""


def test_main_requires_the_lake() -> None:
    with pytest.raises(SystemExit) as erro:
        main([])
    assert erro.value.code == 2


def test_main_rejects_an_unknown_account(tmp_path) -> None:
    with pytest.raises(SystemExit) as erro:
        main(["--lago", str(tmp_path), "--contas", "nope"])
    assert erro.value.code == 2


def test_main_rejects_an_empty_account_list(tmp_path) -> None:
    with pytest.raises(SystemExit) as erro:
        main(["--lago", str(tmp_path), "--contas", ""])
    assert erro.value.code == 2


def _preparar(tmp_path, monkeypatch, fake_atualizar, fake_publicar):
    from scripts import qml_holdout_diario as diario

    monkeypatch.setattr(diario, "RAIZ", tmp_path)
    monkeypatch.setattr(diario, "atualizar", fake_atualizar)
    monkeypatch.setattr(diario, "publicar", fake_publicar)
    return diario


def test_main_passes_a_comma_list_and_sem_push_to_publicar(tmp_path, monkeypatch) -> None:
    from scripts.qml_holdout import CONTA_FIXA

    chamadas: dict = {}

    def fake_atualizar(lago, contas):
        chamadas.setdefault("lago", lago)
        chamadas.setdefault("contas", []).extend(contas)
        return [diario.Resultado(contas[0], "rodar", marcas=1)]

    def fake_publicar(raiz, mensagem, sem_push=False):
        chamadas["raiz"] = raiz
        chamadas["sem_push"] = sem_push
        chamadas["mensagem"] = mensagem
        return False

    diario = _preparar(tmp_path, monkeypatch, fake_atualizar, fake_publicar)

    codigo = diario.main(["--lago", str(tmp_path), "--contas", "ledoit-wolf,pca", "--sem-push"])

    assert codigo == 0
    assert chamadas["contas"] == [CONTA_FIXA, CONTA_PCA]
    assert chamadas["raiz"] == tmp_path
    assert chamadas["sem_push"] is True
    assert "ledoit-wolf rodar" in chamadas["mensagem"]
    assert "pca rodar" in chamadas["mensagem"]


def test_main_publishes_after_an_awaiting_account_and_logs_it(tmp_path, monkeypatch, capsys) -> None:
    """An account waiting for its window is a success, and the run still publishes what the others wrote."""
    from scripts import qml_holdout_diario as diario

    resultado = diario.Resultado(CONTA_PCA, "aguardando", mensagem="no session on or after 10/11/2026")
    publicados: list[str] = []

    def fake_publicar(raiz, mensagem, sem_push=False):
        publicados.append(mensagem)
        return False

    diario = _preparar(tmp_path, monkeypatch, lambda lago, contas: [resultado], fake_publicar)

    assert diario.main(["--lago", str(tmp_path)]) == 0
    assert len(publicados) == 1
    texto = (tmp_path / "logs" / "qml_holdout_diario.log").read_text(encoding="utf-8")
    assert "pca: aguardando" in texto
    assert "pca: aguardando" in capsys.readouterr().out


def test_main_publishes_the_first_account_when_the_second_fails(tmp_path, monkeypatch) -> None:
    """A failure on one account must not throw away the marks the earlier account already wrote."""
    from scripts.qml_holdout import CONTA_FIXA

    def fake_atualizar(lago, contas):
        if contas[0] is CONTA_PCA:
            raise RuntimeError("lake offline on pca")
        return [Resultado(contas[0], "rodar", marcas=3)]

    publicados: list[str] = []

    def fake_publicar(raiz, mensagem, sem_push=False):
        publicados.append(mensagem)
        return True

    diario = _preparar(tmp_path, monkeypatch, fake_atualizar, fake_publicar)

    assert diario.main(["--lago", str(tmp_path), "--contas", "ledoit-wolf,pca"]) == 1
    assert len(publicados) == 1
    assert "ledoit-wolf rodar" in publicados[0]
    texto = (tmp_path / "logs" / "qml_holdout_diario.log").read_text(encoding="utf-8")
    assert "lake offline on pca" in texto


def test_main_returns_one_when_atualizar_fails(tmp_path, monkeypatch) -> None:
    def fake_atualizar(lago, contas):
        raise RuntimeError("lake offline")

    def fake_publicar(raiz, mensagem, sem_push=False):
        return False

    diario = _preparar(tmp_path, monkeypatch, fake_atualizar, fake_publicar)

    assert diario.main(["--lago", str(tmp_path)]) == 1
    texto = (tmp_path / "logs" / "qml_holdout_diario.log").read_text(encoding="utf-8")
    assert "lake offline" in texto


def test_main_returns_one_when_publicar_fails(tmp_path, monkeypatch) -> None:
    from scripts.qml_holdout import CONTA_FIXA

    def fake_publicar(raiz, mensagem, sem_push=False):
        raise RuntimeError("push rejected")

    diario = _preparar(tmp_path, monkeypatch,
                       lambda lago, contas: [Resultado(CONTA_FIXA, "rodar", marcas=1)], fake_publicar)

    assert diario.main(["--lago", str(tmp_path)]) == 1
    texto = (tmp_path / "logs" / "qml_holdout_diario.log").read_text(encoding="utf-8")
    assert "push rejected" in texto
