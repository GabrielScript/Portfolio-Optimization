"""Lançador da rotina do paper trading: falha antes do main() da rotina também vira linha no runs.csv."""

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LANCADOR = Path(__file__).resolve().parent.parent / "scripts" / "rodar_paper.py"


@pytest.fixture
def checkout(tmp_path) -> Path:
    """Um checkout de mentira com o lançador de verdade e uma rotina trocada por cada teste."""
    raiz = tmp_path / "app"
    (raiz / "scripts").mkdir(parents=True)
    shutil.copy(LANCADOR, raiz / "scripts" / "rodar_paper.py")
    return raiz


def _lancar(raiz: Path, rotina: str, local: Path) -> subprocess.CompletedProcess:
    (raiz / "scripts" / "paper_trading.py").write_text(rotina, encoding="utf-8")
    ambiente = {**os.environ, "LOCALAPPDATA": str(local)}
    return subprocess.run([sys.executable, "-X", "utf8", str(raiz / "scripts" / "rodar_paper.py")], env=ambiente,
                          capture_output=True, text=True, encoding="utf-8", timeout=60)


def _linhas(local: Path) -> list[dict[str, str]]:
    with (local / "paper-trading" / "runs.csv").open(encoding="utf-8", newline="") as entrada:
        return list(csv.DictReader(entrada))


def test_falha_na_importacao_da_rotina_vira_linha_falhou_no_runs_csv(checkout, tmp_path):
    # venv quebrado ou erro de sintaxe: o main() da rotina nem chega a rodar, então quem anota é o lançador
    local = tmp_path / "local"
    saida = _lancar(checkout, "raise ImportError('pandas quebrado')\n", local)
    linhas = _linhas(local)
    assert (saida.returncode, len(linhas), linhas[0]["desfecho"], linhas[0]["fechamento"]) == (1, 1, "falhou", "")
    assert "pandas quebrado" in linhas[0]["nota"]


def test_rotina_importada_roda_o_main_dela_e_devolve_o_codigo_de_saida(checkout, tmp_path):
    # caminho normal: quem anota no runs.csv é a própria rotina, o lançador não escreve nada
    local = tmp_path / "local"
    saida = _lancar(checkout, "def main():\n    return 7\n", local)
    assert (saida.returncode, (local / "paper-trading" / "runs.csv").exists()) == (7, False)


def test_falha_no_main_da_rotina_fora_do_try_dela_tambem_vira_linha_falhou(checkout, tmp_path):
    # o main() da rotina monta o log antes do try (pasta sem permissão, disco cheio): a falha sai dele sem anotar
    local = tmp_path / "local"
    saida = _lancar(checkout, "def main():\n    raise PermissionError('logs sem permissão')\n", local)
    linha = _linhas(local)[0]
    log = local / "paper-trading" / "logs" / linha["nota"].rsplit("log ", 1)[1].rstrip(")")
    assert (saida.returncode, linha["desfecho"], "logs sem permissão" in linha["nota"]) == (1, "falhou", True)
    assert "PermissionError" in log.read_text(encoding="utf-8")


def test_runs_csv_travado_fica_registrado_no_log_do_lancador(checkout, tmp_path):
    # runs.csv aberto no Excel: a linha não entra, mas o log diz o que houve (o vigia avisa o dia sem rodada)
    local = tmp_path / "local"
    (local / "paper-trading" / "runs.csv").mkdir(parents=True)        # escrever nele falha como com a trava
    saida = _lancar(checkout, "raise ImportError('pandas quebrado')\n", local)
    logs = list((local / "paper-trading" / "logs").glob("*-lancador.log"))
    texto = logs[0].read_text(encoding="utf-8")
    assert (saida.returncode, saida.stderr, "pandas quebrado" in texto, "runs.csv" in texto) == (1, "", True, True)


def test_paper_v2_que_falha_na_importacao_vira_linha_falhou_no_runs_csv_dele(checkout, tmp_path):
    # a tarefa "paper v2" chama o lançador com "paper_v2": a falha vai para a pasta e o cabeçalho do paper-v2
    local = tmp_path / "local"
    (checkout / "scripts" / "paper_v2.py").write_text("raise ImportError('arch quebrado')\n", encoding="utf-8")
    ambiente = {**os.environ, "LOCALAPPDATA": str(local)}
    saida = subprocess.run([sys.executable, "-X", "utf8", str(checkout / "scripts" / "rodar_paper.py"), "paper_v2"],
                           env=ambiente, capture_output=True, text=True, encoding="utf-8", timeout=60)
    with (local / "paper-v2" / "runs.csv").open(encoding="utf-8", newline="") as entrada:
        linhas = list(csv.DictReader(entrada))
    assert (saida.returncode, len(linhas), list(linhas[0])) == (
        1, 1, ["quando", "fechamento", "desfecho", "nota", "codigo", "guarda"])
    assert (linhas[0]["desfecho"], "arch quebrado" in linhas[0]["nota"]) == ("falhou", True)


def test_upkeep_that_fails_on_import_becomes_a_failed_line_in_its_own_runs_csv(checkout, tmp_path):
    # the "paper manutencao" task calls the launcher with "manutencao": the failure goes to the upkeep's folder
    local = tmp_path / "local"
    (checkout / "scripts" / "manutencao.py").write_text("raise ImportError('pandas quebrado')\n", encoding="utf-8")
    ambiente = {**os.environ, "LOCALAPPDATA": str(local)}
    saida = subprocess.run([sys.executable, "-X", "utf8", str(checkout / "scripts" / "rodar_paper.py"), "manutencao"],
                           env=ambiente, capture_output=True, text=True, encoding="utf-8", timeout=60)
    with (local / "paper-manutencao" / "runs.csv").open(encoding="utf-8", newline="") as entrada:
        linhas = list(csv.DictReader(entrada))
    assert (saida.returncode, len(linhas), list(linhas[0])) == (1, 1, ["quando", "desfecho", "nota"])
    assert (linhas[0]["desfecho"], "pandas quebrado" in linhas[0]["nota"]) == ("falhou", True)
