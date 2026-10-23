"""The daily upkeep (``scripts/manutencao.py``): the paper-v2 readings at 126 and 252 sessions, the monthly progress
reading, the app's liquidity snapshot and the push of ``main``, each committed by explicit path."""

import csv
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest


def test_sessions_are_counted_after_the_first_record_in_the_lakes_calendar():
    from scripts.manutencao import pregoes_desde_o_registro
    calendario = pd.DatetimeIndex(pd.bdate_range("2026-10-15", "2026-10-23"))
    registro = pd.DataFrame({"data": pd.to_datetime(["2026-10-19", "2026-10-19", "2026-10-21"])})
    assert pregoes_desde_o_registro(calendario, registro) == 4          # 20, 21, 22 and 23/10
    assert pregoes_desde_o_registro(calendario, registro.iloc[:0]) == 0


def test_due_readings_are_the_missing_ones_whose_session_target_was_reached(tmp_path):
    from scripts.manutencao import leituras_devidas
    hoje = date(2026, 11, 3)
    assert leituras_devidas(0, tmp_path, hoje) == {}                                  # nothing to read yet
    assert leituras_devidas(125, tmp_path, hoje) == {"progresso-2026-11": 1}
    assert leituras_devidas(126, tmp_path, hoje) == {"leitura-126": 126, "progresso-2026-11": 1}
    (tmp_path / "leitura-126.txt").write_text("feita", encoding="utf-8")
    (tmp_path / "progresso-2026-11.txt").write_text("feita", encoding="utf-8")
    assert leituras_devidas(300, tmp_path, hoje) == {"leitura-252": 252}


SAIDA_125 = ("Paper-v2 de 19/10/2026 a 23/04/2027: 125 pregões, 3 registros; guarda: sem mudança\n"
             "tabela\nCritério principal (pré-registro, decide só em 252 pregões): saída final da Conservadora >= a do "
             "BOVA11\n")


def test_a_reading_is_saved_only_when_its_own_session_count_reaches_the_target(tmp_path):
    # the calendar said 126, but the lake's universe stopped a day short: the reading counted 125
    from scripts.manutencao import salvar_leituras
    agora = datetime(2027, 4, 26, 10, 5, 0)
    leituras, tentativas = tmp_path / "leituras", tmp_path / "estado" / "tentativas"
    feitas = salvar_leituras(SAIDA_125, {"leitura-126": 126, "progresso-2027-04": 1}, leituras, agora, tentativas)
    assert feitas == [leituras / "progresso-2027-04.txt"]
    assert (leituras / "progresso-2027-04.txt").read_text(encoding="utf-8") == SAIDA_125
    assert [caminho.name for caminho in leituras.iterdir()] == ["progresso-2027-04.txt"]   # nothing loose in git
    tentativa = tentativas / "leitura-126-20270426-100500.txt"
    assert tentativa.read_text(encoding="utf-8") == SAIDA_125                     # kept, retried next run


def test_an_output_without_the_session_count_saves_nothing_final(tmp_path):
    from scripts.manutencao import salvar_leituras
    feitas = salvar_leituras("Paper-v2: nenhum pregão depois do 1º registro (ou nenhum registro)\n",
                             {"progresso-2026-10": 1}, tmp_path / "leituras", datetime(2026, 10, 26, 10, 0, 0),
                             tmp_path / "tentativas")
    assert (feitas, (tmp_path / "leituras").exists(), [caminho.name for caminho in (tmp_path / "tentativas").iterdir()]
            ) == ([], False, ["progresso-2026-10-20261026-100000.txt"])


def _exports(pasta, ate: str) -> None:
    """The two lake exports the snapshot reads: ``manifest.json`` and three months of ``monthly_metrics``."""
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "manifest.json").write_text(json.dumps({"data_through": ate}), encoding="utf-8")
    pd.DataFrame({"month": [date(2026, 8, 1), date(2026, 9, 1), date(2026, 10, 1)], "security": ["PETR4"] * 3,
                  "quoted_spread": [0.004, 0.002, 0.010], "vol_daily": [0.02, 0.018, 0.05],
                  "median_dtv_brl": [1e9, 2e9, 3e9]}).to_parquet(pasta / "monthly_metrics.parquet", index=False)


def test_the_liquidity_snapshot_is_rebuilt_only_when_the_lake_closed_a_newer_month(tmp_path):
    from scripts.liquidez_do_app import instantaneo
    from scripts.manutencao import liquidez_nova
    origem, atual = tmp_path / "b3lake", tmp_path / "liquidez.parquet"
    _exports(origem, "2026-10-22")
    assert liquidez_nova(atual, origem)["month"].tolist() == [date(2026, 9, 1)]      # no snapshot yet
    instantaneo(pd.read_parquet(origem / "monthly_metrics.parquet"), date(2026, 10, 22)).to_parquet(atual)
    assert liquidez_nova(atual, origem) is None                                       # 09/2026 is the newest closed
    _exports(origem, "2026-11-03")
    assert liquidez_nova(atual, origem)["month"].tolist() == [date(2026, 10, 1)]


def test_no_closed_month_in_the_exports_means_no_new_snapshot(tmp_path):
    # a partial export whose metric months all start at or after the lake's last month: an empty snapshot
    from scripts.manutencao import liquidez_nova
    _exports(tmp_path / "b3lake", "2026-08-15")
    assert liquidez_nova(tmp_path / "sem.parquet", tmp_path / "b3lake") is None


def test_run_as_a_script_the_upkeep_still_imports_the_projects_modules(tmp_path):
    # `python scripts/manutencao.py` puts only scripts/ on sys.path (the tests' conftest adds the root)
    _exports(tmp_path / "b3lake", "2026-10-22")
    codigo = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import manutencao; "
              "print(manutencao.liquidez_nova(Path(sys.argv[2]) / 'sem.parquet', Path(sys.argv[2]) / 'b3lake')"
              "['month'].tolist())")
    feita = subprocess.run([sys.executable, "-X", "utf8", "-c", codigo, str(RAIZ / "scripts"), str(tmp_path)],
                           cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert (feita.returncode, feita.stdout.strip()) == (0, "[datetime.date(2026, 9, 1)]"), feita.stderr


def _rodar_git(*argumentos: str) -> str:
    return subprocess.run(["git", *argumentos], check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


def _clone(origem: Path, destino: Path) -> Path:
    _rodar_git("clone", "-q", str(origem), str(destino))
    for chave, valor in (("user.email", "teste@exemplo"), ("user.name", "teste"), ("core.autocrlf", "false")):
        _rodar_git("-C", str(destino), "config", chave, valor)
    return destino


@pytest.fixture
def app(tmp_path) -> Path:
    """A clone on ``main`` of a bare ``origin`` with one commit, plus an untracked file of the kind the real root
    has (``dask.zip``)."""
    origem = tmp_path / "origem.git"
    _rodar_git("init", "-q", "--bare", "-b", "main", str(origem))
    semente = _clone(origem, tmp_path / "semente")
    (semente / "README.md").write_text("v1\n", encoding="utf-8")
    _rodar_git("-C", str(semente), "add", "README.md")
    _rodar_git("-C", str(semente), "commit", "-q", "-m", "inicio")
    _rodar_git("-C", str(semente), "push", "-q", "origin", "main")
    raiz = _clone(origem, tmp_path / "app")
    (raiz / "dask.zip").write_text("solto", encoding="utf-8")
    return raiz


def test_the_commit_takes_exactly_the_given_paths_and_nothing_when_they_did_not_change(app):
    from scripts.manutencao import commitar, git_em
    git = git_em(app)
    (app / "README.md").write_text("mexido à mão\n", encoding="utf-8")
    (app / "lago").mkdir()
    (app / "lago" / "liquidez.parquet").write_bytes(b"foto")
    assert commitar(git, ["lago/liquidez.parquet"], "liquidez: 10/2026") == ["lago/liquidez.parquet"]
    assert git("show", "--name-only", "--format=%s", "HEAD").splitlines() == [
        "liquidez: 10/2026", "", "lago/liquidez.parquet"]
    assert git("status", "--porcelain").splitlines() == [" M README.md", "?? dask.zip"]   # untouched
    assert commitar(git, ["lago/liquidez.parquet"], "liquidez: 10/2026") == []


def test_a_local_commit_is_pushed_fast_forward_and_nothing_happens_when_in_sync(app):
    from scripts.manutencao import commitar, git_em, publicar
    git = git_em(app)
    assert publicar(git) == "main em dia com origin"
    (app / "paper-v2").mkdir()
    (app / "paper-v2" / "carteiras.csv").write_text("data,ticker,peso,preco\n", encoding="utf-8")
    commitar(git, ["paper-v2/carteiras.csv"], "paper-v2: rebalancear")
    assert publicar(git) == "push de 1 commit(s)"
    assert _rodar_git("-C", str(app.parent / "origem.git"), "rev-parse", "main") == git("rev-parse", "HEAD")


def test_a_main_that_diverged_from_origin_is_not_pushed_and_blocks(app):
    # someone pushed from another machine: no merge and no force by an unattended routine, the user decides
    from scripts.manutencao import Bloqueio, commitar, git_em, publicar
    outro = _clone(app.parent / "origem.git", app.parent / "outro")
    (outro / "README.md").write_text("v2\n", encoding="utf-8")
    _rodar_git("-C", str(outro), "commit", "-q", "-am", "de outra máquina")
    _rodar_git("-C", str(outro), "push", "-q", "origin", "main")
    git = git_em(app)
    (app / "nota.txt").write_text("local\n", encoding="utf-8")
    commitar(git, ["nota.txt"], "local")
    with pytest.raises(Bloqueio, match="origin/main"):
        publicar(git)
    assert _rodar_git("-C", str(app.parent / "origem.git"), "log", "-1", "--format=%s", "main") == "de outra máquina"


def _saida(pregoes: int) -> str:
    return (f"Paper-v2 de 19/10/2026 a 26/04/2027: {pregoes} pregões, 3 registros; guarda: sem mudança\n"
            "                 CAGR\nConservadora    0.10\n"
            "Critério principal (pré-registro, decide só em 252 pregões): saída final da Conservadora >= a do BOVA11\n"
            "Critério secundário (adendo de 23/10/2026, decide só em 252 pregões): vol anual da Conservadora <= a do "
            "BOVA11\n")


def _checkout(app: Path, pregoes: int, ate: str = "2027-04-26") -> Path:
    """The files the upkeep reads in the app checkout: the lake's calendar with ``pregoes`` sessions after the first
    paper-v2 record (19/10/2026), the registry and the lake's exports."""
    (app / "lago").mkdir(exist_ok=True)
    datas = pd.bdate_range("2026-10-19", periods=pregoes + 1)
    pd.DataFrame({"date": datas, "PETR4": 1.0}).to_parquet(app / "lago" / "precos_diarios.parquet", index=False)
    (app / "paper-v2").mkdir(exist_ok=True)
    (app / "paper-v2" / "carteiras.csv").write_text("data,ticker,peso,preco\n2026-10-19,PETR4,1.0,30.0\n",
                                                    encoding="utf-8")
    _exports(app / "data" / "b3lake", ate)
    return app


def _origem(app: Path, *argumentos: str) -> str:
    return _rodar_git("-C", str(app.parent / "origem.git"), *argumentos)


def test_a_run_saves_the_due_readings_and_the_new_snapshot_commits_them_and_pushes(app):
    from scripts.manutencao import executar, git_em
    _checkout(app, 126)
    lidas = []
    resultado = executar(app, git_em(app), app / "daily.lock", datetime(2027, 4, 27, 10, 0),
                         lambda: lidas.append(1) or _saida(126))
    assert (resultado.desfecho, lidas) == ("ok", [1])                    # one reading serves both due names
    publicados = _origem(app, "ls-tree", "-r", "--name-only", "main").splitlines()
    assert {"paper-v2/leituras/leitura-126.txt", "paper-v2/leituras/progresso-2027-04.txt",
            "lago/liquidez.parquet"} <= set(publicados)
    assert "data/b3lake/manifest.json" not in publicados                  # only the upkeep's own paths
    titulos = [titulo for titulo, _ in resultado.avisos]
    assert titulos == ["Fundo Alepha: leitura de 126 pregões", "Fundo Alepha: progresso de 04/2027"]
    assert "Critério principal" in resultado.avisos[0][1]


def _nada_lido() -> str:
    raise AssertionError("the reading must not run")


def test_with_the_lake_running_the_run_does_nothing(app):
    from scripts.manutencao import executar, git_em
    _checkout(app, 126)
    (app / "daily.lock").write_text("", encoding="utf-8")
    antes = _rodar_git("-C", str(app), "rev-parse", "HEAD")
    resultado = executar(app, git_em(app), app / "daily.lock", datetime(2027, 4, 27, 10, 0), _nada_lido)
    assert (resultado.desfecho, _rodar_git("-C", str(app), "rev-parse", "HEAD")) == ("lago rodando", antes)
    assert not (app / "lago" / "liquidez.parquet").exists()


def test_a_checkout_off_main_blocks_the_run_and_touches_nothing(app):
    # the scheduled routines run this checkout: on another branch (22/10/2026) the user has to switch back
    from scripts.manutencao import executar, git_em
    _checkout(app, 126)
    _rodar_git("-C", str(app), "switch", "-q", "-c", "feat/x")
    resultado = executar(app, git_em(app), app / "daily.lock", datetime(2027, 4, 27, 10, 0), _nada_lido)
    assert (resultado.desfecho, "feat/x" in resultado.nota) == ("bloqueada", True)
    assert not (app / "lago" / "liquidez.parquet").exists()


def test_a_git_operation_in_course_blocks_the_run(app):
    from scripts.manutencao import executar, git_em
    _checkout(app, 126)
    (app / ".git" / "index.lock").write_text("", encoding="utf-8")
    resultado = executar(app, git_em(app), app / "daily.lock", datetime(2027, 4, 27, 10, 0), _nada_lido)
    assert (resultado.desfecho, "index.lock" in resultado.nota) == ("bloqueada", True)
    assert not (app / "lago" / "liquidez.parquet").exists()


def test_a_failed_reading_does_not_stop_the_snapshot_and_the_push(app):
    from scripts.manutencao import executar, git_em
    _checkout(app, 126)

    def quebrada() -> str:
        raise subprocess.CalledProcessError(1, ["paper_v2.py", "--leitura"])
    resultado = executar(app, git_em(app), app / "daily.lock", datetime(2027, 4, 27, 10, 0), quebrada)
    assert (resultado.desfecho, "CalledProcessError" in resultado.nota) == ("falhou", True)
    assert "lago/liquidez.parquet" in _origem(app, "ls-tree", "-r", "--name-only", "main").splitlines()


def _sem_commit(git):
    """The real git, with ``commit`` failing (a late lock, a hook, the 300 s timeout)."""
    def quebrado(*argumentos: str) -> str:
        if argumentos[0] == "commit":
            raise subprocess.CalledProcessError(1, ["git", "commit"])
        return git(*argumentos)
    return quebrado


def test_a_reading_whose_commit_failed_is_committed_and_notified_by_the_next_run(app, tmp_path):
    from scripts.manutencao import executar, git_em
    _checkout(app, 126)
    primeira = executar(app, _sem_commit(git_em(app)), app / "daily.lock", datetime(2027, 4, 27, 10, 0),
                        lambda: _saida(126), tentativas=tmp_path / "tentativas")
    assert (primeira.desfecho, primeira.avisos) == ("falhou", [])
    segunda = executar(app, git_em(app), app / "daily.lock", datetime(2027, 4, 28, 10, 0), _nada_lido,
                       tentativas=tmp_path / "tentativas")
    publicados = _origem(app, "ls-tree", "-r", "--name-only", "main").splitlines()
    assert segunda.desfecho == "ok"
    assert {"paper-v2/leituras/leitura-126.txt", "paper-v2/leituras/progresso-2027-04.txt",
            "lago/liquidez.parquet"} <= set(publicados)
    assert [titulo for titulo, _ in segunda.avisos] == ["Fundo Alepha: leitura de 126 pregões",
                                                         "Fundo Alepha: progresso de 04/2027"]


def test_a_git_error_while_checking_the_checkout_is_a_failed_run_not_a_crash(app, tmp_path):
    from scripts.manutencao import executar, git_em
    _checkout(app, 126)
    real = git_em(app)

    def sem_rev_parse(*argumentos: str) -> str:
        if argumentos[0] == "rev-parse":
            raise subprocess.TimeoutExpired(["git", "rev-parse"], 300)
        return real(*argumentos)
    resultado = executar(app, sem_rev_parse, app / "daily.lock", datetime(2027, 4, 27, 10, 0), _nada_lido,
                         tentativas=tmp_path / "tentativas")
    assert (resultado.desfecho, resultado.nota.startswith("checkout: TimeoutExpired")) == ("falhou", True)


def test_the_run_waits_for_the_routines_still_running_but_not_forever():
    # PC woken after the schedule: the late tasks start together, so wait for the lake and the paper routines
    from scripts.manutencao import esperar
    estados, sonos = iter([True, True, False]), []
    esperar(lambda: next(estados), sonos.append, pausa=180, passo=30, limite=2_700)
    assert sonos == [180, 30, 30]
    sonos.clear()
    esperar(lambda: True, sonos.append, pausa=180, passo=30, limite=90)
    assert sonos == [180, 30, 30, 30]                                    # gives up after the limit


def test_each_run_appends_one_line_to_its_runs_csv_in_the_launchers_format(tmp_path):
    from scripts.manutencao import Resultado, anotar
    anotar(tmp_path, datetime(2026, 11, 3, 10, 0, 5), Resultado("ok", "liquidez: 10/2026; push de 1 commit(s)", []))
    anotar(tmp_path, datetime(2026, 11, 4, 10, 0, 5), Resultado("falhou", 'push: erro "x"', []))
    with (tmp_path / "runs.csv").open(encoding="utf-8", newline="") as entrada:
        linhas = list(csv.reader(entrada))
    assert linhas == [["quando", "desfecho", "nota"],
                      ["2026-11-03T10:00:05", "ok", "liquidez: 10/2026; push de 1 commit(s)"],
                      ["2026-11-04T10:00:05", "falhou", 'push: erro "x"']]


def _paper_v2_de_mentira(raiz: Path, corpo: str) -> Path:
    (raiz / "scripts").mkdir(parents=True, exist_ok=True)
    (raiz / "scripts" / "paper_v2.py").write_text(corpo, encoding="utf-8")
    return raiz


def test_the_reading_runs_paper_v2_with_leitura_in_the_checkout_and_returns_its_output(tmp_path):
    from scripts.manutencao import ler_leitura
    raiz = _paper_v2_de_mentira(tmp_path, "import sys\nprint('Paper-v2 de x a y: 126 pregões,', sys.argv[1:])\n")
    assert ler_leitura(raiz) == "Paper-v2 de x a y: 126 pregões, ['--leitura']\n"


def test_a_reading_that_fails_raises_with_its_error_output(tmp_path):
    from scripts.manutencao import ler_leitura
    raiz = _paper_v2_de_mentira(tmp_path, "raise SystemExit('universe do lago falhou')\n")
    with pytest.raises(RuntimeError, match="universe do lago falhou"):
        ler_leitura(raiz)


RAIZ = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(sys.platform != "win32", reason="the notification is Windows PowerShell")
def test_the_notification_gets_title_and_body_intact_with_accents_quotes_and_lines():
    from scripts.manutencao import avisar
    corpo = 'Critério principal: saída final da Conservadora >= a do "BOVA11"\nCritério secundário: vol <= a do BOVA11'
    assert avisar(RAIZ, "Fundo Alepha: leitura de 126 pregões", corpo, mostrar=True) == (
        f"Fundo Alepha: leitura de 126 pregões | {corpo}")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Task Scheduler")
def test_tasks_that_do_not_exist_count_as_not_running():
    # a machine without the lake's task (or a renamed one) must not stop the upkeep
    from scripts.manutencao import tarefas_rodando
    assert tarefas_rodando(("tarefa que não existe 1", "tarefa que não existe 2")) == 0


def test_every_notification_is_sent_and_a_failed_one_is_recorded_in_the_outcome():
    from scripts.manutencao import Resultado, notificar
    enviados = []

    def avisar(titulo: str, corpo: str) -> None:
        if titulo == "quebra":
            raise subprocess.CalledProcessError(1, ["powershell.exe"])
        enviados.append((titulo, corpo))
    feito = Resultado("ok", "leituras: leitura-126", [("Fundo Alepha: leitura de 126 pregões", "c"), ("quebra", "x")])
    resultado = notificar(feito, avisar)
    assert enviados == [("Fundo Alepha: leitura de 126 pregões", "c")]
    assert (resultado.desfecho, resultado.nota.startswith("leituras: leitura-126; aviso: CalledProcessError")) == (
        "falhou", True)
    assert notificar(Resultado("ok", "nada", []), avisar) == Resultado("ok", "nada", [])
