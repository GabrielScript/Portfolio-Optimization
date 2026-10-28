"""
Paper-v3: the forward log of attempt 51's composite (Conservadora + value + profit growth, DECISOES 164) against a
Conservadora decided on the same close. Pre-registration: ``docs/superpowers/specs/2026-10-28-paper-v3-composto-design.md``
(commit ``0cb942c``, before this file existed). Paper-v2 (``scripts/paper_v2.py``) is untouched and keeps its own clock.

Usage: .venv/Scripts/python -X utf8 scripts/paper_v3.py             (the routine; the "paper v3" task uses rodar_paper.py)
       .venv/Scripts/python -X utf8 scripts/paper_v3.py --leitura   (the reading of the records up to the last close)
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtest import CarteiraRecusada, retornos_da_carteira  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.crescimento_lucro import nota_de_crescimento  # noqa: E402
from core.execucao import Liquidez, refazer  # noqa: E402
from core.impostos import ISENCAO_DE_VENDAS  # noqa: E402
from core.lago import ler_setores  # noqa: E402
from core.paper import Acao, curva_do_registro, rebalanceio_devido  # noqa: E402
from core.qubo_universo import composite_score  # noqa: E402
from core.registro import registrar_carteira  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import conservadora  # noqa: E402
from core.valor import Balancos, NotaDoMes  # noqa: E402
from core.views import nota_media_dos_escores  # noqa: E402
from scripts.avaliar_conservadora import CONSERVADORA, Mensal  # noqa: E402
from scripts.avaliar_execucao import (CUSTOS_DA_B3, EM_PORCENTO, EM_REAIS, comprado_e_mantido, medidas,  # noqa: E402
                                      proventos_diarios)
from scripts.avaliar_robustez import _tabela  # noqa: E402
from scripts.avaliar_valor import notas  # noqa: E402
from scripts.paper_trading import (IMPORTS_DO_PROJETO, _git, definicoes_mudadas, ler_precos,  # noqa: E402
                                   ler_registro, mudancas_desde_o_paper, versao_do_codigo)
from scripts.paper_v2 import (ARQUIVOS_DO_LAGO, CAMINHO_DA_CONSERVADORA, CAPITAL_DA_LEITURA,  # noqa: E402
                              DadosDoLago, acao_do_dia, estrategia, ler_dados, liquidez_do_lago,
                              resultado_do_registro, treino_do_dia)

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DO_LAGO = RAIZ / "data" / "b3lake"            # the lake's app exports (outside git)
PASTA = RAIZ / "paper-v3"
ARQUIVO_PROVENIENCIA = PASTA / "proveniencia.csv"
PASTA_MARCOS = PASTA / "marcos"
_LOCAL = Path(os.environ.get("LOCALAPPDATA", Path.home()))
PASTA_ESTADO = _LOCAL / "paper-v3"                  # runs.csv, outside the repository
PROJETO_DO_LAGO = RAIZ.parents[1] / "b3-datalake"
RAMO_DA_ROTINA = "main"
TAG_DO_PAPER = "paper-v3"
COMPOSTO = "Composto-crescimento"                   # attempt 51's arm name
P_CONTRA_A_CONSERVADORA = "P[> Conservadora]"
ARQUIVOS_DA_CVM = ("fundamentals.parquet", "market_caps.parquet")
FALHAS_DO_LAGO = (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError)
# what defines the recorded arms: the modules of the decision path and, in this script, its decision definitions
CAMINHO_DO_COMPOSTO = [*CAMINHO_DA_CONSERVADORA, "core/composto.py", "core/valor.py", "core/crescimento_lucro.py",
                       "core/qubo_universo.py", "core/views.py", "scripts/avaliar_valor.py", "scripts/paper_v2.py"]
DEFINICOES_DO_COMPOSTO = [IMPORTS_DO_PROJETO, "ler_balancos", "nota_do_composto", "carteiras_do_dia", "falta_no_lago",
                          "decidir", "executar", "atualizar_lago", "ler_dados_do_composto"]
_SEM_JANELA = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True)
class DadosDoComposto:
    lago: DadosDoLago                   # paper-v2's five files
    balancos: Balancos                  # fundamentals.parquet, market_caps.parquet, issuer sectors
    sha256: dict[str, str]              # of the two CVM files


@dataclass(frozen=True)
class Arquivos:
    calendario: Path                    # lago/precos_diarios.parquet: the lake publishes every session
    composto: Path                      # the records, one per arm
    conservadora: Path
    trava_do_lago: Path                 # daily.lock


@dataclass(frozen=True)
class RodadaDupla:
    data: pd.Timestamp                  # last close of the calendar
    acao: Acao
    carteiras: dict[str, pd.Series] | None = None   # recorded this run; None = nothing enters the files
    nota: str = ""

    @property
    def desfecho(self) -> str:
        """For runs.csv: a due rebalance without portfolios is a refusal by the rules."""
        return "recusada" if self.acao is Acao.REBALANCEAR and self.carteiras is None else self.acao.value


@dataclass(frozen=True)
class Passos:
    """The routine's side effects, injected."""
    atualizar: Callable[[], None]       # the lake's universe and fundamentals up to the last close
    ler: Callable[[], DadosDoComposto]
    guardar: Callable[[RodadaDupla, DadosDoComposto], None]


ARQUIVOS = Arquivos(RAIZ / "lago" / "precos_diarios.parquet", PASTA / "composto.csv", PASTA / "conservadora.csv",
                    _LOCAL / "b3lake" / "daily.lock")


def ler_balancos(pasta: Path, arquivo_setores: Path) -> tuple[Balancos, dict[str, str]]:
    """The lake's CVM files as ``avaliar_valor.carregar_balancos`` reads them, without its frozen-hash check (the live
    files change every run), and their sha256 for the provenance."""
    sha256 = {nome: hashlib.sha256((pasta / nome).read_bytes()).hexdigest() for nome in ARQUIVOS_DA_CVM}
    fundamentos = pd.read_parquet(pasta / ARQUIVOS_DA_CVM[0])
    for coluna in ("dt_refer", "available_from"):
        fundamentos[coluna] = pd.to_datetime(fundamentos[coluna])
    valor_mercado = pd.read_parquet(pasta / ARQUIVOS_DA_CVM[1], columns=["issuer", "month", "market_cap"])
    valor_mercado["month"] = pd.to_datetime(valor_mercado["month"])
    return Balancos(fundamentos, valor_mercado, ler_setores(arquivo_setores).to_dict()), sha256


def ler_dados_do_composto(pasta: Path = PASTA_DO_LAGO) -> DadosDoComposto:
    balancos, sha256 = ler_balancos(pasta, pasta / "sectors.parquet")
    return DadosDoComposto(ler_dados(pasta), balancos, sha256)


def nota_do_composto(retornos_mensais: pd.DataFrame, proventos: pd.DataFrame, balancos: Balancos) -> NotaDoMes:
    """Attempt 51's score, from the same functions as ``qubo_universo_avaliar._value_score(..., com_crescimento=True)``:
    the Conservadora score, value (Magic Formula + Trending Value) and profit growth, mean of the normal scores."""
    valor = nota_media_dos_escores(*notas(balancos, Mensal(retornos_mensais, proventos, {})).values())
    return composite_score(lambda mes, candidatos: conservadora(retornos_mensais, proventos, mes, candidatos),
                           valor, nota_de_crescimento(balancos.fundamentos))


def carteiras_do_dia(dados: DadosDoComposto, data: pd.Timestamp) -> dict[str, pd.Series]:
    """Both arms' portfolios on the close of ``data`` from the same training window, only the positive weights;
    ``CarteiraRecusada`` if the fixed rules refuse either."""
    treino = treino_do_dia(dados.lago, data)
    nota = nota_do_composto(dados.lago.retornos_mensais, dados.lago.proventos, dados.balancos)
    pesos = {COMPOSTO: estrategia_por_nota(nota, dados.lago.setores)(treino),
             CONSERVADORA: estrategia(dados.lago)(treino)}
    return {braco: carteira[carteira > 0] for braco, carteira in pesos.items()}


def falta_no_lago(dados: DadosDoComposto, data: pd.Timestamp) -> str:
    """Why the lake's files cannot decide the close of ``data`` (empty when they can): prices short of it, or no
    market value for its month (the value score would be empty on every name, silently)."""
    if dados.lago.precos.index[-1] != data:
        return f"universe_prices through {dados.lago.precos.index[-1]:%d/%m/%Y}: not deciding"
    mes = data.to_period("M").to_timestamp()
    if not (pd.to_datetime(dados.balancos.valor_mercado["month"]) == mes).any():
        return f"market_caps without {mes:%m/%Y}: not deciding"
    return ""


def decidir(dados: DadosDoComposto, composto: pd.DataFrame, data: pd.Timestamp) -> RodadaDupla:
    """The due day's decision: frozen by the composite's registered curve, refused if either arm is refused."""
    curva = pd.Series(dtype=float) if composto.empty else curva_do_registro(composto, dados.lago.precos)
    acao = acao_do_dia(curva, devido=True)
    if acao is not Acao.REBALANCEAR:
        return RodadaDupla(data, acao)
    try:
        return RodadaDupla(data, acao, carteiras_do_dia(dados, data))
    except CarteiraRecusada as recusa:
        return RodadaDupla(data, acao, nota="refused: " + "; ".join(
            f"{v.regra.value} in {v.alvo} {v.valor:.2%} (limit {v.limite:.0%})" for v in recusa.violacoes))


def executar(arquivos: Arquivos, passos: Passos) -> RodadaDupla | None:
    """One run: None with the lake running. Off the due day it does not touch the lake; on it, update the lake, read
    the files, check they can decide the last close, decide, and record both arms on that close."""
    if arquivos.trava_do_lago.exists():
        return None
    calendario = ler_precos(arquivos.calendario).index
    data = calendario[-1]
    composto = ler_registro(arquivos.composto)
    ultima = None if composto.empty else composto["data"].max()
    if ultima is not None and ultima >= data:
        return RodadaDupla(data, Acao.AGUARDAR, nota="close already decided")
    if not rebalanceio_devido(ultima, calendario):
        return RodadaDupla(data, Acao.AGUARDAR)
    try:
        passos.atualizar()
    except FALHAS_DO_LAGO as erro:
        return RodadaDupla(data, Acao.AGUARDAR, nota=f"lake update failed: {type(erro).__name__}: {erro}"[:300])
    dados = passos.ler()
    if motivo := falta_no_lago(dados, data):
        return RodadaDupla(data, Acao.AGUARDAR, nota=motivo)
    rodada = decidir(dados, composto, data)
    if rodada.carteiras is not None:
        precos = dados.lago.precos.ffill().loc[data]
        registrar_carteira(data.date(), rodada.carteiras[COMPOSTO], precos, arquivos.composto)
        registrar_carteira(data.date(), rodada.carteiras[CONSERVADORA], precos, arquivos.conservadora)
        passos.guardar(rodada, dados)
    return rodada


def leitura(registros: dict[str, pd.DataFrame], precos: pd.DataFrame, dividendos: pd.DataFrame,
            liquidez: Liquidez) -> pd.DataFrame:
    """The pre-registered measure: each arm's record refeito with R$ 10 thousand, the final proof's costs and the
    individual's income tax, as paper-v2's reading, plus BOVA11 bought on the composite's first close and held; and
    P[Composto > Conservadora] on the exit values, descriptive (``dividendos`` with ``corporate_action``)."""
    retornos = retornos_da_carteira(precos)
    proventos = proventos_diarios(dividendos, retornos.index, retornos.columns)
    refeitas = {braco: refazer(resultado_do_registro(registro, precos), retornos, CAPITAL_DA_LEITURA, CUSTOS_DA_B3,
                               liquidez, isencao=ISENCAO_DE_VENDAS, proventos=proventos)
                for braco, registro in registros.items()}
    referencia = comprado_e_mantido(resultado_do_registro(registros[COMPOSTO], precos), "BOVA11")
    refeitas["BOVA11"] = refazer(referencia, retornos, CAPITAL_DA_LEITURA, CUSTOS_DA_B3, liquidez, isencao=0.0)
    tabela = medidas(refeitas)
    tabela.loc[COMPOSTO, P_CONTRA_A_CONSERVADORA] = psr_da_diferenca(refeitas[COMPOSTO].saida,
                                                                     refeitas[CONSERVADORA].saida)
    return tabela


def criterio(tabela: pd.DataFrame) -> bool:
    """The decision at 252 sessions: the composite's final exit value >= the Conservadora's (``leitura`` table)."""
    return bool(tabela.loc[COMPOSTO, EM_REAIS[0]] >= tabela.loc[CONSERVADORA, EM_REAIS[0]])


def atualizar_lago(projeto: Path = PROJETO_DO_LAGO, rodar: Callable[..., object] = subprocess.run) -> None:
    """The lake's ``universe`` (prices and point-in-time universe up to the last close) and ``fundamentals`` (CVM
    DFP/ITR, network; not part of ``daily``), both copied to the app; the lake's venv Python, as its scheduled task."""
    python_do_lago = str(projeto / ".venv" / "Scripts" / "python.exe")
    for comando in ("universe", "fundamentals"):
        rodar([python_do_lago, "-m", "b3lake.cli", comando], cwd=projeto, check=True, capture_output=True, text=True,
              encoding="utf-8", timeout=3600, creationflags=_SEM_JANELA)


def codigo_mudado_desde_o_paper(raiz: Path = RAIZ, tag: str = TAG_DO_PAPER) -> list[str]:
    """What changed since the ``tag`` in the decision path: whole modules and, in this script, the decision's
    definitions only (``main()`` and the records change without changing the measured arms)."""
    script = Path(__file__).resolve().relative_to(raiz).as_posix()
    antes = _git("show", f"{tag}:{script}", raiz=raiz)
    depois = (raiz / script).read_text(encoding="utf-8")
    return (mudancas_desde_o_paper(raiz, CAMINHO_DO_COMPOSTO, tag)
            + [f"{script}: {nome}" for nome in definicoes_mudadas(antes, depois, DEFINICOES_DO_COMPOSTO)])


def guardar_no_git(rodada: RodadaDupla, dados: DadosDoComposto, versao: str, raiz: Path = RAIZ,
                   pasta_do_lago: Path = PASTA_DO_LAGO) -> None:
    """Provenance, the seven files' milestone copy and a local commit (only on ``main``; the upkeep pushes)."""
    marco = PASTA_MARCOS / f"{rodada.data:%Y-%m-%d}"
    marco.mkdir(parents=True, exist_ok=True)
    nomes = (*ARQUIVOS_DO_LAGO, *ARQUIVOS_DA_CVM)
    for nome in nomes:
        shutil.copy2(pasta_do_lago / nome, marco / nome)
    sha256 = dados.lago.sha256 | dados.sha256
    novo = not ARQUIVO_PROVENIENCIA.exists()
    with ARQUIVO_PROVENIENCIA.open("a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo, lineterminator="\n")
        if novo:
            escritor.writerow(["data", "codigo", *nomes, "manifest_gerado", "manifest_ate"])
        escritor.writerow([f"{rodada.data:%Y-%m-%d}", versao, *(sha256[nome] for nome in nomes),
                           dados.lago.manifest["generated_at"], dados.lago.manifest["data_through"]])
    if _git("rev-parse", "--abbrev-ref", "HEAD", raiz=raiz) != RAMO_DA_ROTINA:
        return
    caminhos = [caminho.relative_to(raiz).as_posix()
                for caminho in (ARQUIVOS.composto, ARQUIVOS.conservadora, ARQUIVO_PROVENIENCIA, marco)]
    _git("add", "--", *caminhos, raiz=raiz)
    _git("commit", "-m", f"paper-v3: {rodada.acao.value} on the close of {rodada.data:%d/%m/%Y} (code {versao})",
         "--", *caminhos, raiz=raiz)


def _anotar(rodada: RodadaDupla | None, versao: str, aviso: str) -> None:
    PASTA_ESTADO.mkdir(parents=True, exist_ok=True)
    arquivo = PASTA_ESTADO / "runs.csv"
    novo = not arquivo.exists()
    with arquivo.open("a", newline="", encoding="utf-8") as saida:
        escritor = csv.writer(saida, lineterminator="\n")
        if novo:
            escritor.writerow(["quando", "fechamento", "desfecho", "nota", "codigo", "guarda"])
        escritor.writerow([datetime.now().isoformat(timespec="seconds"),
                           "" if rodada is None else f"{rodada.data:%Y-%m-%d}",
                           "lago rodando" if rodada is None else rodada.desfecho,
                           "" if rodada is None else rodada.nota, versao, aviso])


def _imprimir_leitura(aviso: str) -> int:
    registros = {COMPOSTO: ler_registro(ARQUIVOS.composto), CONSERVADORA: ler_registro(ARQUIVOS.conservadora)}
    if registros[COMPOSTO].empty:
        print("Paper-v3: no record yet")
        return 0
    atualizar_lago()        # off the due day universe_prices stops at the last rebalance
    precos = ler_dados(PASTA_DO_LAGO).precos
    primeira = registros[COMPOSTO]["data"].min()
    pregoes = int((precos.index > primeira).sum())
    if pregoes == 0:
        print("Paper-v3: no session after the first record")
        return 0
    dividendos = pd.read_parquet(PASTA_DO_LAGO / "dividends.parquet",
                                 columns=["security", "ex_date", "corporate_action", "yield"])
    tabela = leitura(registros, precos, dividendos, liquidez_do_lago())
    print(f"Paper-v3 from {primeira:%d/%m/%Y} to {precos.index[-1]:%d/%m/%Y}: {pregoes} pregões, "
          f"{registros[COMPOSTO]['data'].nunique()} records; guard: {aviso or 'no change'}")
    print(_tabela(tabela.assign(**{coluna: tabela[coluna].round().astype("int64") for coluna in EM_REAIS}),
                  list(EM_PORCENTO)))
    print(f"Decision (pre-registered, only at 252 sessions): composite final exit "
          f"{'>=' if criterio(tabela) else '<'} the Conservadora's")
    return 0


def main() -> int:
    versao = versao_do_codigo(RAIZ)
    try:
        aviso = "; ".join(codigo_mudado_desde_o_paper())
    except (OSError, subprocess.CalledProcessError, SyntaxError, ValueError) as erro:
        aviso = f"{TAG_DO_PAPER} guard unavailable: {str(erro)[:200]}"
    if "--leitura" in sys.argv[1:]:
        return _imprimir_leitura(aviso)
    rodada = executar(ARQUIVOS, Passos(atualizar_lago, ler_dados_do_composto,
                                       lambda feita, dados: guardar_no_git(feita, dados, versao)))
    _anotar(rodada, versao, aviso)
    if rodada is None:
        print("lake running: not deciding")
        return 0
    print(f"{rodada.data:%d/%m/%Y}: {rodada.desfecho} {rodada.nota}".strip())
    for braco, carteira in (rodada.carteiras or {}).items():
        print(f"\n{braco}\n{carteira.round(4).to_string()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
