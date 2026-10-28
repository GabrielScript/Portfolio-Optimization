"""Vigia das 12h (``scripts/vigia.ps1``): lê o runs.csv do lago e o do paper e diz o que está errado."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="o vigia é PowerShell do Windows")

VIGIA = Path(__file__).resolve().parent.parent / "scripts" / "vigia.ps1"
HOJE = "2026-10-20"


def _pastas(tmp_path, lago: list[str], paper: list[str], v2: list[str] | None = None,
            manutencao: list[str] | None = None, ramo: str = "main",
            sentimento: list[str] | None = None, v3: list[str] | None = None) -> tuple[Path, Path, Path, Path, Path]:
    """runs.csv do lago (formato do b3lake), do paper (formato do ``anotar``), do paper-v2 (formato do ``_anotar``
    dele), da manutenção (``scripts/manutencao.py``) e do log de notícias do lago (``sentiment_runs.csv``), mais um
    checkout git no ``ramo``; sem ``v2``, ``manutencao`` ou ``sentimento``, uma rodada certa dela hoje e em cada dia
    em que o paper rodou (os testes do paper-v1 não dependem delas)."""
    dias = sorted({HOJE, *(linha[:10] for linha in paper)})
    if v2 is None:
        v2 = [f"{dia}T08:45:05,,aguardar,,abc1234," for dia in dias]
    if v3 is None:
        v3 = [f"{dia}T10:30:05,,aguardar,,abc1234," for dia in dias]
    if manutencao is None:
        manutencao = [f"{dia}T10:03:05,ok,leituras: nenhuma devida" for dia in dias]
    if sentimento is None:
        sentimento = [f"{dia}T13:10:05+00:00,{dia}T13:10:40+00:00,ok,infomoney 3" for dia in dias]
    pasta_lago, pasta_paper, pasta_v2 = tmp_path / "b3lake", tmp_path / "paper-trading", tmp_path / "paper-v2"
    pasta_manutencao = tmp_path / "paper-manutencao"
    for pasta, cabecalho, linhas in ((pasta_lago, "started,finished,expected,outcome,detail", lago),
                                     (pasta_paper, "inicio,fim,fechamento,desfecho,nota", paper),
                                     (pasta_v2, "quando,fechamento,desfecho,nota,codigo,guarda", v2),
                                     (tmp_path / "paper-v3", "quando,fechamento,desfecho,nota,codigo,guarda", v3),
                                     (pasta_manutencao, "quando,desfecho,nota", manutencao)):
        pasta.mkdir()
        (pasta / "runs.csv").write_text("\n".join([cabecalho, *linhas]) + "\n", encoding="utf-8")
    (pasta_lago / "sentiment_runs.csv").write_text(
        "\n".join(["started,finished,outcome,detail", *sentimento]) + "\n", encoding="utf-8")
    checkout = tmp_path / "app"
    subprocess.run(["git", "init", "-q", "-b", ramo, str(checkout)], check=True, capture_output=True)
    return pasta_lago, pasta_paper, pasta_v2, pasta_manutencao, checkout


def _vigiar(pasta_lago: Path, pasta_paper: Path, pasta_v2: Path, pasta_manutencao: Path, checkout: Path,
            *quando: str) -> subprocess.CompletedProcess:
    """``quando``: "-Hoje", dia (o padrão, 20/10/2026) ou "-Agora", data e hora da rodada (sem -Hoje)."""
    return subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
                           str(VIGIA), *(quando or ("-Hoje", HOJE)), "-PastaLago", str(pasta_lago),
                           "-PastaPaper", str(pasta_paper), "-PastaPaperV2", str(pasta_v2),
                           "-PastaPaperV3", str(pasta_v2.parent / "paper-v3"),
                           "-PastaManutencao", str(pasta_manutencao), "-Repositorio", str(checkout), "-SoMostrar",
                           "-SemEspera"],
                          capture_output=True, text=True, encoding="utf-8", timeout=120)


LAGO_EM_DIA = "2026-10-19T21:45:10,2026-10-19T22:10:02,2026-10-19,done,new zip: [2026]"
LAGO_HOJE = "2026-10-20T08:00:03,2026-10-20T08:00:04,2026-10-19,up_to_date,"
PAPER_HOJE = "2026-10-20T08:30:05,2026-10-20T08:30:31,2026-10-19,aguardar,código abc1234"


def test_tudo_em_dia_nao_avisa_nada(tmp_path):
    saida = _vigiar(*_pastas(tmp_path, [LAGO_EM_DIA, LAGO_HOJE], [PAPER_HOJE]))
    assert (saida.returncode, saida.stdout.strip()) == (0, "vigia: tudo em dia"), saida.stderr


def test_paper_sem_rodada_hoje_avisa(tmp_path):
    # o PC dormiu às 8h30 e a tarefa não rodou depois: só a linha de ontem
    ontem = "2026-10-19T08:30:05,2026-10-19T08:30:31,2026-10-19,aguardar,"
    saida = _vigiar(*_pastas(tmp_path, [LAGO_HOJE], [ontem]))
    assert (saida.returncode, saida.stdout.strip()) == (
        1, 'vigia: paper trading: nenhuma rodada hoje (tarefa "paper trading")'), saida.stderr


def test_paper_que_falhou_hoje_avisa_com_a_nota(tmp_path):
    falhou = "2026-10-20T08:30:05,2026-10-20T08:30:06,,falhou,importação da rotina: ImportError: pandas"
    saida = _vigiar(*_pastas(tmp_path, [LAGO_HOJE], [PAPER_HOJE, falhou]))
    assert saida.stdout.strip().splitlines()[0] == (
        "vigia: paper trading: falhou (importação da rotina: ImportError: pandas)"), saida.stderr


def test_lago_que_falhou_hoje_avisa_com_o_detalhe(tmp_path):
    falhou = "2026-10-20T08:00:03,2026-10-20T08:01:00,2026-10-19,failed,HTTP 503"
    saida = _vigiar(*_pastas(tmp_path, [LAGO_HOJE, falhou], [PAPER_HOJE]))
    assert (saida.returncode, saida.stdout.strip()) == (1, "vigia: lago: failed (HTTP 503)"), saida.stderr


def test_lago_sem_rodada_hoje_avisa(tmp_path):
    saida = _vigiar(*_pastas(tmp_path, [LAGO_EM_DIA], [PAPER_HOJE]))
    assert saida.stdout.strip() == 'vigia: lago: nenhuma rodada hoje (tarefa "b3lake daily")', saida.stderr


def test_paper_decidido_num_fechamento_mais_velho_que_o_do_lago_avisa(tmp_path):
    # a rotina rodou e "aguardou", mas o lago já publicou o fechamento de 19/10 e ela decidiu no de 16/10
    atrasado = "2026-10-20T08:30:05,2026-10-20T08:30:31,2026-10-16,aguardar,"
    saida = _vigiar(*_pastas(tmp_path, [LAGO_HOJE], [atrasado]))
    assert saida.stdout.strip() == (
        "vigia: paper trading: último fechamento decidido 2026-10-16, o lago já tem o de 2026-10-19"), saida.stderr


def test_vigia_atrasado_a_noite_nao_cobra_o_fechamento_que_o_lago_publicou_depois_do_paper(tmp_path):
    # PC desligado às 12h e ligado às 22h: o lago das 21h45 já tem o fechamento de hoje, mas o paper das 8h30 só
    # podia decidir no de ontem (o que o lago tinha às 8h00)
    noite = "2026-10-20T21:45:02,2026-10-20T22:05:40,2026-10-20,done,new zip: [2026]"
    saida = _vigiar(*_pastas(tmp_path, [LAGO_HOJE, noite], [PAPER_HOJE]))
    assert (saida.returncode, saida.stdout.strip()) == (0, "vigia: tudo em dia"), saida.stderr


@pytest.mark.parametrize("agora", ["2026-10-21T07:30:00", "2026-10-24T10:00:00"])
def test_vigia_que_roda_atrasado_noutro_dia_confere_o_dia_util_anterior(tmp_path, agora):
    # PC desligado às 12h de 20/10 (ter) e ligado na manhã seguinte, ou de 23/10 (sex) e ligado no sábado: as rotinas
    # de hoje ainda não rodaram; a rodada perdida é a do dia útil anterior, que estava em dia
    sexta = ["2026-10-23T08:00:03,2026-10-23T08:00:04,2026-10-22,up_to_date,"]
    paper_sexta = ["2026-10-23T08:30:05,2026-10-23T08:30:31,2026-10-22,aguardar,"]
    lago, paper = ([LAGO_HOJE], [PAPER_HOJE]) if agora.startswith("2026-10-21") else (sexta, paper_sexta)
    saida = _vigiar(*_pastas(tmp_path, lago, paper), "-Agora", agora)
    assert (saida.returncode, saida.stdout.strip()) == (0, "vigia: tudo em dia"), saida.stderr


def test_paper_v2_sem_rodada_hoje_avisa(tmp_path):
    ontem = "2026-10-19T08:45:05,2026-10-16,aguardar,,abc1234,"
    saida = _vigiar(*_pastas(tmp_path, [LAGO_HOJE], [PAPER_HOJE], [ontem]))
    assert (saida.returncode, saida.stdout.strip()) == (
        1, 'vigia: paper v2: nenhuma rodada hoje (tarefa "paper v2")'), saida.stderr


def test_paper_v2_que_falhou_hoje_avisa_com_a_nota(tmp_path):
    falhou = "2026-10-20T08:45:05,,falhou,importação da rotina: ImportError: arch,,"
    saida = _vigiar(*_pastas(tmp_path, [LAGO_HOJE], [PAPER_HOJE], [falhou]))
    assert (saida.returncode, saida.stdout.strip()) == (
        1, "vigia: paper v2: falhou (importação da rotina: ImportError: arch)"), saida.stderr


def test_upkeep_without_a_run_today_warns(tmp_path):
    ontem = "2026-10-19T10:03:05,ok,leituras: nenhuma devida"
    saida = _vigiar(*_pastas(tmp_path, [LAGO_HOJE], [PAPER_HOJE], manutencao=[ontem]))
    assert (saida.returncode, saida.stdout.strip()) == (
        1, 'vigia: manutenção: nenhuma rodada hoje (tarefa "paper manutencao")'), saida.stderr


def test_upkeep_still_running_without_a_line_today_is_not_a_problem_yet(tmp_path):
    # a reading day after a late wake: the upkeep waited for the other routines and is still reading at 12h48
    ontem = "2026-10-19T10:03:05,ok,leituras: nenhuma devida"
    saida = _vigiar(*_pastas(tmp_path, [LAGO_HOJE], [PAPER_HOJE], manutencao=[ontem]), "-Hoje", HOJE,
                    "-ManutencaoRodando")
    assert (saida.returncode, saida.stdout.strip()) == (0, "vigia: tudo em dia"), saida.stderr


def test_upkeep_that_failed_or_was_blocked_today_warns_with_the_note(tmp_path):
    bloqueada = "2026-10-20T10:03:05,bloqueada,push: origin/main tem 1 commit(s) que a main local não tem"
    saida = _vigiar(*_pastas(tmp_path, [LAGO_HOJE], [PAPER_HOJE], manutencao=[bloqueada]))
    assert (saida.returncode, saida.stdout.strip()) == (
        1, "vigia: manutenção: bloqueada (push: origin/main tem 1 commit(s) que a main local não tem)"), saida.stderr


def test_an_app_checkout_off_main_warns(tmp_path):
    # 22/10/2026: the checkout was left on a branch and paper-v1 did not register the day
    saida = _vigiar(*_pastas(tmp_path, [LAGO_HOJE], [PAPER_HOJE], ramo="feat/x"))
    assert (saida.returncode, saida.stdout.strip()) == (
        1, "vigia: checkout do app em feat/x, as rotinas rodam na main (git switch main)"), saida.stderr


def test_an_app_checkout_with_a_detached_head_warns(tmp_path):
    pastas = _pastas(tmp_path, [LAGO_HOJE], [PAPER_HOJE])
    checkout = pastas[-1]
    subprocess.run(["git", "-C", str(checkout), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                    "--allow-empty", "-m", "x"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(checkout), "switch", "-q", "--detach"], check=True, capture_output=True)
    saida = _vigiar(*pastas)
    assert (saida.returncode, saida.stdout.strip()) == (
        1, "vigia: checkout do app com HEAD destacado, as rotinas rodam na main (git switch main)"), saida.stderr


def test_the_news_log_without_a_good_run_on_the_day_warns(tmp_path):
    saida = _vigiar(*_pastas(tmp_path, [LAGO_EM_DIA, LAGO_HOJE], [PAPER_HOJE], sentimento=[]))
    assert saida.returncode == 1
    assert 'vigia: log de notícias: nenhuma rodada certa hoje (tarefa "b3lake sentimento")' in saida.stdout


def test_the_news_log_whose_runs_all_failed_on_the_day_warns_with_the_last_detail(tmp_path):
    falhas = [f"{HOJE}T{hora}:10:05+00:00,{HOJE}T{hora}:10:09+00:00,failed,every source failed: offline"
              for hora in ("09", "11")]
    saida = _vigiar(*_pastas(tmp_path, [LAGO_EM_DIA, LAGO_HOJE], [PAPER_HOJE], sentimento=falhas))
    assert saida.returncode == 1
    assert "vigia: log de notícias: failed (every source failed: offline)" in saida.stdout


def test_paper_v3_without_a_run_today_warns(tmp_path):
    ontem = "2026-10-19T10:30:05,2026-10-16,aguardar,,abc1234,"
    saida = _vigiar(*_pastas(tmp_path, [LAGO_HOJE], [PAPER_HOJE], v3=[ontem]))
    assert (saida.returncode, saida.stdout.strip()) == (
        1, 'vigia: paper v3: nenhuma rodada hoje (tarefa "paper v3")'), saida.stderr


def test_paper_v3_that_failed_today_warns_with_the_note(tmp_path):
    falhou = "2026-10-20T10:30:05,2026-10-19,aguardar,lake update failed: CalledProcessError,abc1234,"
    saida = _vigiar(*_pastas(tmp_path, [LAGO_HOJE], [PAPER_HOJE], v3=[falhou.replace("aguardar", "falhou")]))
    assert (saida.returncode, saida.stdout.strip()) == (
        1, "vigia: paper v3: falhou (lake update failed: CalledProcessError)"), saida.stderr
