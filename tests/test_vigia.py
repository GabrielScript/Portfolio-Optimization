"""Vigia das 12h (``scripts/vigia.ps1``): lê o runs.csv do lago e o do paper e diz o que está errado."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="o vigia é PowerShell do Windows")

VIGIA = Path(__file__).resolve().parent.parent / "scripts" / "vigia.ps1"
HOJE = "2026-10-20"


def _pastas(tmp_path, lago: list[str], paper: list[str]) -> tuple[Path, Path]:
    """runs.csv do lago (formato do b3lake) e do paper (formato do ``anotar``)."""
    pasta_lago, pasta_paper = tmp_path / "b3lake", tmp_path / "paper-trading"
    for pasta, cabecalho, linhas in ((pasta_lago, "started,finished,expected,outcome,detail", lago),
                                     (pasta_paper, "inicio,fim,fechamento,desfecho,nota", paper)):
        pasta.mkdir()
        (pasta / "runs.csv").write_text("\n".join([cabecalho, *linhas]) + "\n", encoding="utf-8")
    return pasta_lago, pasta_paper


def _vigiar(pasta_lago: Path, pasta_paper: Path, *quando: str) -> subprocess.CompletedProcess:
    """``quando``: "-Hoje", dia (o padrão, 20/10/2026) ou "-Agora", data e hora da rodada (sem -Hoje)."""
    return subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
                           str(VIGIA), *(quando or ("-Hoje", HOJE)), "-PastaLago", str(pasta_lago),
                           "-PastaPaper", str(pasta_paper), "-SoMostrar", "-SemEspera"],
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
