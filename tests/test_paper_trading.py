"""Rotina do paper trading (S5): decisão, registro e commit sem rede, sobre o formato real do export do lago."""

import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from assets import get_ticker_info, get_tickers_liquidos
from core import lago
from core.paper import Acao, Leitura
from core.registro import registrar_carteira
from core.risco import Regra, Violacao
from scripts import paper_trading
from scripts.paper_trading import (codigo_mudado_desde_o_paper, commitar_no_git, commitar_pendente, contar_recusas,
                                   definicoes_mudadas, executar, ler_precos, ler_registro, mudancas_desde_o_paper,
                                   rodar, satelite_do_dia, texto_da_leitura, versao_do_codigo)

DATAS = pd.bdate_range("2026-07-01", periods=70)


@pytest.fixture
def precos(tmp_path):
    """``lago/precos_diarios.parquet``: coluna ``date`` (datetime.date) e uma float32 por papel, sem ``.SA``."""
    caminho = tmp_path / "precos_diarios.parquet"
    pd.DataFrame({
        "date": DATAS.date,
        "A": np.linspace(10.0, 12.0, len(DATAS)).astype("float32"),
        "B": np.linspace(20.0, 19.0, len(DATAS)).astype("float32"),
        "BOVA11": np.linspace(200.0, 210.0, len(DATAS)).astype("float32"),
    }).to_parquet(caminho)
    return ler_precos(caminho)


def _escolher(pesos: dict[str, float]):
    return lambda data: (pd.Series(pesos), [])


def test_sem_registro_registra_o_satelite_escolhido_no_ultimo_fechamento(precos, tmp_path):
    rodada = rodar(precos, ler_registro(tmp_path / "carteiras.csv"), _escolher({"A": 0.6, "B": 0.4}))
    assert (rodada.data, rodada.acao, rodada.carteira.to_dict()) == (DATAS[-1], Acao.REBALANCEAR,
                                                                     {"A": 0.6, "B": 0.4})


def _registro(tmp_path, linhas: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Grava pelo ``registrar_carteira`` (o formato que a rotina lê de volta) e lê com ``ler_registro``."""
    arquivo = tmp_path / "carteiras.csv"
    for data in dict.fromkeys(d for d, _, _ in linhas):
        pesos = pd.Series({t: p for d, t, p in linhas if d == data})
        registrar_carteira(pd.Timestamp(data).date(), pesos, pd.Series(1.0, index=pesos.index), arquivo)
    return ler_registro(arquivo)


def _nao_escolher(data):
    raise AssertionError("não era dia de otimizar")


def test_rodar_de_novo_no_fechamento_ja_registrado_nao_registra(precos, tmp_path):
    registro = _registro(tmp_path, [(str(DATAS[-1].date()), "A", 0.6), (str(DATAS[-1].date()), "B", 0.4)])
    rodada = rodar(precos, registro, _nao_escolher)
    assert (rodada.acao, rodada.carteira is None) == (Acao.AGUARDAR, True)


def test_satelite_ja_no_bova11_pelo_stop_fica_parado(precos, tmp_path):
    # o último registro é o do stop (100% BOVA11), 70 pregões atrás: nada de otimizar de novo
    registro = _registro(tmp_path, [(str(DATAS[0].date()), "BOVA11", 1.0)])
    rodada = rodar(precos, registro, _nao_escolher)
    assert (rodada.acao, rodada.carteira is None) == (Acao.PARADO, True)


def test_stop_no_dia_devido_registra_100_por_cento_bova11_sem_otimizar(tmp_path):
    # o satélite (só A) cai 2% por pregão e o BOVA11 fica parado: no 63º pregão depois do registro a razão está
    # ~70% abaixo do pico, sem nenhum dia de −5%
    datas = pd.bdate_range("2026-07-01", periods=64)
    precos = pd.DataFrame({"A": 10.0 * 0.98 ** np.arange(64), "BOVA11": 200.0}, index=datas)
    registro = _registro(tmp_path, [(str(datas[0].date()), "A", 1.0)])
    rodada = rodar(precos, registro, _nao_escolher)
    assert (rodada.acao, rodada.carteira.to_dict()) == (Acao.STOP, {"BOVA11": 1.0})


def test_carteira_nova_recusada_pelas_regras_nao_entra_e_diz_por_que(precos, tmp_path):
    # a escolha viola o teto de setor: fica a carteira registrada (nada a gravar) e a rotina tenta no pregão seguinte
    violacao = Violacao(Regra.SETOR, "Utilidade Pública", 0.36, 0.30)
    rodada = rodar(precos, ler_registro(tmp_path / "carteiras.csv"), lambda data: (pd.Series({"A": 1.0}), [violacao]))
    assert (rodada.acao, rodada.carteira is None, "Utilidade Pública" in rodada.nota) == (Acao.REBALANCEAR, True,
                                                                                         True)


def test_recusa_vai_para_o_runs_csv_como_recusada_e_nao_como_rebalanceio(precos, tmp_path):
    # o pré-registro conta as recusas: o desfecho da rodada tem de separá-las de um rebalanceio que entrou
    violacao = Violacao(Regra.SETOR, "Utilidade Pública", 0.36, 0.30)
    recusada = rodar(precos, ler_registro(tmp_path / "carteiras.csv"), lambda data: (pd.Series({"A": 1.0}), [violacao]))
    entrou = rodar(precos, ler_registro(tmp_path / "carteiras.csv"), _escolher({"A": 1.0}))
    assert (recusada.desfecho, entrou.desfecho) == ("recusada", "rebalancear")


def test_recusas_contam_os_fechamentos_recusados_ate_o_fim_da_leitura(tmp_path, monkeypatch):
    # 2 rodadas no mesmo fechamento contam 1; recusa depois do fim da janela e rodada sem fechamento não contam
    monkeypatch.setattr(paper_trading, "PASTA_ESTADO", tmp_path)
    inicio = datetime(2026, 10, 19, 8, 30)
    for fechamento, desfecho in [("2026-10-16", "recusada"), ("2026-10-16", "recusada"), ("", "falhou"),
                                 ("2026-10-19", "aguardar"), ("2026-10-20", "recusada"), ("2026-10-23", "recusada")]:
        paper_trading.anotar(inicio, fechamento, desfecho, "")
    assert contar_recusas(tmp_path / "runs.csv", pd.Timestamp("2026-10-20")) == 2


def test_relatorio_traz_as_recusas_da_leitura(arquivo_precos, tmp_path, monkeypatch):
    monkeypatch.setattr(paper_trading, "PASTA_ESTADO", tmp_path)
    monkeypatch.setattr(paper_trading, "_cdi", lambda inicio, hoje: None)          # sem rede
    paper_trading.anotar(datetime(2026, 10, 1, 8, 30), str(DATAS[-2].date()), "recusada", "")
    registro = tmp_path / "carteiras.csv"
    registrar_carteira(DATAS[0].date(), pd.Series({"A": 1.0}), pd.Series({"A": 10.0}), registro)
    texto = paper_trading.relatorio(paper_trading.Rodada(DATAS[-1], Acao.AGUARDAR), arquivo_precos, registro)
    assert "Recusas pelas regras (runs.csv): 1" in texto


def _git(raiz: Path, *argumentos: str) -> str:
    return subprocess.run(["git", "-C", str(raiz), *argumentos], check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def registro_no_git(tmp_path) -> Path:
    """``paper/carteiras.csv`` com a 1ª carteira commitada num repositório git de verdade."""
    raiz = tmp_path / "repo"
    raiz.mkdir()
    _git(raiz, "init", "-q", "-b", "main")
    _git(raiz, "config", "user.email", "teste@exemplo.com")
    _git(raiz, "config", "user.name", "teste")
    arquivo = raiz / "paper" / "carteiras.csv"
    registrar_carteira(DATAS[0].date(), pd.Series({"A": 1.0}), pd.Series({"A": 10.0}), arquivo)
    _git(raiz, "add", "--", "paper/carteiras.csv")
    _git(raiz, "commit", "-q", "-m", "1º registro")
    return arquivo


def test_registro_gravado_cujo_commit_falhou_entra_no_git_na_rodada_seguinte(registro_no_git):
    # o commit da rodada anterior falhou depois de gravar (index.lock, por ex.): a linha nova está só no disco
    raiz = registro_no_git.parent.parent
    registrar_carteira(DATAS[5].date(), pd.Series({"B": 1.0}), pd.Series({"B": 20.0}), registro_no_git)
    feitos = (commitar_pendente(registro_no_git, raiz), commitar_pendente(registro_no_git, raiz))
    assert (feitos, _git(raiz, "status", "--porcelain"), _git(raiz, "rev-list", "--count", "HEAD")) == (
        (True, False), "", "2")


def test_registro_reescrito_fora_da_rotina_nao_e_commitado(registro_no_git):
    # o arquivo só cresce: uma linha já commitada que mudou é erro (o aviso chama alguém), não um commit pendente
    raiz = registro_no_git.parent.parent
    registro_no_git.write_bytes(registro_no_git.read_bytes().replace(b",1.0,", b",0.9,"))
    with pytest.raises(ValueError, match="reescrito"):
        commitar_pendente(registro_no_git, raiz)
    assert _git(raiz, "rev-list", "--count", "HEAD") == "1"


def test_versao_do_codigo_limpo_e_o_hash_curto_do_head(registro_no_git):
    raiz = registro_no_git.parent.parent
    assert versao_do_codigo(raiz) == _git(raiz, "rev-parse", "--short", "HEAD")


def test_versao_do_codigo_com_arquivo_versionado_alterado_vem_marcada_suja(registro_no_git):
    # arquivo não versionado (os soltos da raiz do app) não suja: só mudança no que o git guarda
    raiz = registro_no_git.parent.parent
    (raiz / "solto.txt").write_text("fora do git", encoding="utf-8")
    registro_no_git.write_text(registro_no_git.read_text(encoding="utf-8") + "x\n", encoding="utf-8")
    assert versao_do_codigo(raiz) == _git(raiz, "rev-parse", "--short", "HEAD") + " (sujo)"



def test_versao_do_codigo_ignora_os_precos_que_o_lago_publica_todo_dia(registro_no_git):
    # lago/precos_diarios.parquet é versionado nos marcos e a rotina do lago o reescreve a cada pregão: não é código
    raiz = registro_no_git.parent.parent
    (raiz / "lago").mkdir()
    (raiz / "lago" / "precos_diarios.parquet").write_bytes(b"v1")
    _git(raiz, "add", "--", "lago/precos_diarios.parquet")
    _git(raiz, "commit", "-q", "-m", "marco")
    (raiz / "lago" / "precos_diarios.parquet").write_bytes(b"v2")
    assert versao_do_codigo(raiz) == _git(raiz, "rev-parse", "--short", "HEAD")

@pytest.fixture
def repo_com_tag(registro_no_git) -> Path:
    """O repositório do registro com ``core/estimation.py`` commitado e a tag ``paper-v1`` nesse commit."""
    raiz = registro_no_git.parent.parent
    (raiz / "core").mkdir()
    (raiz / "core" / "estimation.py").write_text("JANELA = 1260\n", encoding="utf-8")
    _git(raiz, "add", "--", "core/estimation.py")
    _git(raiz, "commit", "-q", "-m", "código do paper")
    _git(raiz, "tag", "paper-v1")
    return raiz


def test_arquivo_do_caminho_do_satelite_mudado_desde_a_tag_e_apontado(repo_com_tag):
    (repo_com_tag / "core" / "estimation.py").write_text("JANELA = 2520\n", encoding="utf-8")
    assert mudancas_desde_o_paper(repo_com_tag, ["core/estimation.py", "assets.py"]) == ["core/estimation.py"]


def test_satelite_do_dia_e_o_kelly_da_tela_dentro_das_regras(tmp_path, monkeypatch):
    # 8 papéis em 4 setores, 5 anos e meio de pregões no formato do lago: pesos sem ".SA", até 20% cada, 30% por setor
    gerador = np.random.default_rng(7)
    datas = pd.bdate_range("2021-03-01", "2026-10-16")
    papeis = [f"PAP{i}3" for i in range(8)]
    valores = 10.0 * np.cumprod(1.0 + gerador.normal(0.0005, 0.015, (len(datas), 9)), axis=0)
    tabela = pd.DataFrame(valores.astype("float32"), columns=[*papeis, "BOVA11"])
    caminho = tmp_path / "precos_diarios.parquet"
    tabela.assign(date=datas.date)[["date", *tabela.columns]].to_parquet(caminho)
    monkeypatch.setattr(lago, "ARQUIVO_PRECOS", caminho)
    setores = {f"{papel}.SA": f"Setor {i % 4}" for i, papel in enumerate(papeis)}
    pesos, violacoes = satelite_do_dia(pd.Timestamp("2026-10-16"), 0.1475, list(setores), setores)
    por_setor = pesos.groupby(lambda papel: setores[f"{papel}.SA"]).sum()
    assert violacoes == [] and not any(papel.endswith(".SA") for papel in pesos.index)
    assert (pesos.sum(), pesos.max() <= 0.20 + 1e-6, por_setor.max() <= 0.30 + 1e-6) == (pytest.approx(1.0), True, True)


@pytest.fixture
def arquivo_precos(tmp_path, precos):
    caminho = tmp_path / "lago.parquet"
    precos.astype("float32").assign(date=precos.index.date).to_parquet(caminho)
    return caminho


def _sem_commit(arquivo, rodada):
    raise AssertionError("nada a commitar")


def test_lago_rodando_sai_sem_decidir_nem_gravar(arquivo_precos, tmp_path):
    # o daily.lock do lago existe: o arquivo de preços pode estar no meio da cópia
    trava = tmp_path / "daily.lock"
    trava.touch()
    registro = tmp_path / "carteiras.csv"
    rodada = executar(arquivo_precos, registro, trava, _escolher({"A": 1.0}), _sem_commit)
    assert (rodada, registro.exists()) == (None, False)


def test_carteira_nova_entra_com_o_fechamento_do_dia_e_e_commitada_uma_vez(arquivo_precos, tmp_path):
    # 2ª rodada no mesmo fechamento: nada entra, nada é commitado
    registro, commits = tmp_path / "carteiras.csv", []
    for _ in range(2):
        executar(arquivo_precos, registro, tmp_path / "sem.lock", _escolher({"A": 0.6, "B": 0.4}),
                 lambda arquivo, rodada: commits.append(rodada.data))
    linhas = ler_registro(registro)
    assert (commits, linhas["preco"].tolist()) == ([DATAS[-1]], pytest.approx([12.0, 19.0]))


def _leitura(pregoes: int, satelite: float, bova: float, stop=None) -> Leitura:
    return Leitura(inicio=DATAS[0], fim=DATAS[-1], pregoes=pregoes, satelite=satelite, bova=bova, cdi=None,
                   comparacao=None, stop=stop, registros=4, congelados=0, p_supera=0.6)


def test_texto_da_leitura_traz_o_veredito_so_com_252_pregoes():
    assert "Item 3 do critério de pronto: cumprido" in texto_da_leitura(_leitura(252, 0.12, 0.10))
    assert "Item 3 do critério de pronto: não cumprido" in texto_da_leitura(_leitura(252, 0.08, 0.10))
    assert "Item 3" not in texto_da_leitura(_leitura(251, 0.12, 0.10))


def test_definicao_do_satelite_com_corpo_mudado_e_apontada():
    antes = "ANOS = 5\n\ndef satelite(x):\n    return x * 2\n\ndef main():\n    return 0\n"
    depois = "ANOS = 5\n\ndef satelite(x):\n    return x * 3\n\ndef main():\n    return 1\n"
    assert definicoes_mudadas(antes, depois, ["ANOS", "satelite"]) == ["satelite"]


def test_constante_do_satelite_com_valor_mudado_e_apontada():
    assert definicoes_mudadas("ANOS_DE_PRECOS = 5\n", "ANOS_DE_PRECOS = 10\n", ["ANOS_DE_PRECOS"]) == ["ANOS_DE_PRECOS"]


def test_mudanca_numa_definicao_do_satelite_no_script_aponta_o_nome_e_nao_o_arquivo(repo_com_tag):
    # o script também tem o main() e o commit, que mudam sem mexer no satélite: só as definições do satélite contam
    script = repo_com_tag / "scripts" / "paper_trading.py"
    script.parent.mkdir()
    script.write_text("ANOS_DE_PRECOS = 5\n\ndef main():\n    return 0\n", encoding="utf-8")
    _git(repo_com_tag, "add", "--", "scripts/paper_trading.py")
    _git(repo_com_tag, "commit", "-q", "-m", "script")
    _git(repo_com_tag, "tag", "-f", "paper-v1")
    script.write_text("ANOS_DE_PRECOS = 10\n\ndef main():\n    return 1\n", encoding="utf-8")
    assert codigo_mudado_desde_o_paper(repo_com_tag) == ["scripts/paper_trading.py: ANOS_DE_PRECOS"]


def test_commit_do_registro_leva_a_versao_do_codigo_na_mensagem(registro_no_git):
    raiz = registro_no_git.parent.parent
    registrar_carteira(DATAS[5].date(), pd.Series({"B": 1.0}), pd.Series({"B": 20.0}), registro_no_git)
    rodada = paper_trading.Rodada(DATAS[5], Acao.REBALANCEAR, pd.Series({"B": 1.0}))
    commitar_no_git(registro_no_git, rodada, "abc1234 (sujo)", raiz=raiz)
    assert _git(raiz, "log", "-1", "--format=%s") == (
        f"paper: rebalancear no fechamento de {DATAS[5]:%d/%m/%Y} (código abc1234 (sujo))")


FIXTURE_DO_PAPER = Path(__file__).parent / "dados" / "precos_paper_v1.parquet"
SATELITE_DE_16_10_2026 = {"PETR4": 0.2, "CPLE3": 0.005133815417867128, "POMO4": 0.0025401517695010484,
                          "DIRR3": 0.028102690416543997, "CXSE3": 0.19184461280332027, "BBSE3": 0.1081553871966797,
                          "EMBJ3": 0.16935715781395497, "SBSP3": 0.09486619281037109, "CSMG3": 0.19999999177176178}


def test_satelite_do_dia_congelado_no_pre_registro(monkeypatch):
    # golden: os preços que o 1º registro leu (lago/precos_diarios.parquet do 260fd9a, universo + BOVA11, desde
    # 09/2021) e a Selic do fallback dão a carteira do paper/carteiras.csv de 16/10/2026. Falhou = o satélite medido
    # mudou (código, universo do assets.py ou versão de biblioteca): pré-registro novo e relógio do zero.
    monkeypatch.setattr(lago, "ARQUIVO_PRECOS", FIXTURE_DO_PAPER)
    universo = get_tickers_liquidos()
    info = get_ticker_info()
    pesos, violacoes = satelite_do_dia(pd.Timestamp("2026-10-16"), 0.1375, universo,
                                       {ticker: info[ticker]["setor"] for ticker in universo})
    assert (violacoes, pesos.to_dict()) == ([], pytest.approx(SATELITE_DE_16_10_2026, abs=1e-6))
