"""
Paper trading do satélite (S5, item h do PLAN): rotina diária que decide no último fechamento do lago e registra o
satélite em ``paper/carteiras.csv``. Rede só para a Selic e o CDI do relatório.

Uso: .venv/Scripts/python -X utf8 scripts/paper_trading.py
     (agendada no Windows como "paper trading", seg-sex 8h30, depois da repescagem das 8h00 do lago)

Pré-registro (2026-10-18, escrito e commitado antes do 1º registro; depois desse commit o texto não muda):
- O que se mede: o satélite do padrão do dinheiro real até o critério de pronto (PLAN, "Regras de risco fixas";
  DECISOES 45): núcleo BOVA11 com 70% das ações, fixo e não registrado, e satélite com 30%. O arquivo guarda só o
  satélite: pesos-alvo que somam até 100% do dinheiro do satélite, tickers sem ".SA". O 1º registro começa o relógio
  de 6 a 12 meses do critério de pronto (ROTEIRO §2, item 3).
- Satélite pelo caminho da carteira de hoje da tela, sem orçamento nem plano de compra: lista líquida do `assets.py`,
  5 anos de preços do `lago/precos_diarios.parquet` até o último fechamento do arquivo (`carregar_precos`, cobertura
  de 95%), Ledoit-Wolf + Jorion, Kelly λ = 2, peso máximo de 20%, até 10 ativos, vol ex-ante de até 40%, teto de 30%
  por setor do `assets.py`, taxa livre de risco = Selic meta (SGS 432; o fallback de 13,75% do app se o BCB falhar,
  anotado no runs.csv). A carteira passa por `checar_carteira`; recusada = fica a registrada e a rotina tenta de novo
  no pregão seguinte, com a recusa contada.
- Calendário: decisão no fechamento da última data do arquivo do lago; a carteira registrada vale a partir do pregão
  seguinte (a curva troca de pesos no fechamento da data registrada). Rebalanceio com 63 pregões do lago ou mais
  depois do último registro. Rodar de novo no mesmo fechamento não registra nada. Lago rodando (`daily.lock`) = sai
  sem decidir; a próxima rodada decide.
- Regras (`core.paper.decidir_do_dia`): perda de 5% ou mais do satélite no último pregão = nenhuma ordem, nem o stop
  (tenta no pregão seguinte); stop relativo só em dia de rebalanceio: se a razão satélite/BOVA11 caiu 25% ou mais
  desde o pico, registra 100% BOVA11 e nunca mais rebalanceia.
- Custos e IR: a curva do registro é bruta; o relatório desconta 10 bps por lado sobre o giro de cada registro (a
  montagem paga 10 bps, o stop 20 bps) e nada no BOVA11. IR fora (S4).
- Leituras: com 126 pregões depois do 1º registro (cerca de 6 meses), só relatório; com 252 (cerca de 12 meses), a
  decisão. Medidas: retorno acumulado líquido do satélite e do BOVA11 nos mesmos pregões e o CDI do período; alfa de
  Jensen contra o BOVA11 com t de Newey-West, beta, TE e IR (`comparar_benchmark`, a partir de 60 pregões);
  P[satélite > BOVA11] = PSR da diferença diária; registros, recusas, pregões congelados e data do stop.
- Decisão em 252 pregões, fixa agora: o item 3 do critério de pronto está cumprido se (a) o stop não disparou e
  (b) o retorno acumulado líquido do satélite for maior ou igual ao do BOVA11 nos mesmos pregões. Senão, não está, e
  o satélite segue igual. O item 3 não substitui o item 1: com TE de 16-19%, um ano de alfa tem erro padrão de 16-19
  p.p., então (b) acontece em cerca de metade das vezes sem alfa nenhum; o paper prova a operação (dados, rotina,
  regras), não o alfa.
- Dado errado achado depois (conferido no lago): a linha registrada fica (o arquivo só cresce), o relatório marca o
  trecho e o DECISOES registra. Mudar o satélite ou as regras = pré-registro novo e relógio do zero.
- Expectativa registrada: 1º registro no fechamento de 16/10/2026; 4 rebalanceios em 12 meses; 0 recusas (como no
  item g); chance de stop em 12 meses de 14-25% sem alfa (PLAN, aceite do risk); 0 a 2 pregões congelados.
"""

from __future__ import annotations

import ast
import csv
import logging
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assets import BENCHMARK_ACOES, get_ticker_info, get_tickers_liquidos  # noqa: E402
from core import lago, market_data  # noqa: E402
from core.config import ConfigOtimizacao, Objetivo, limites_por_setor, taxa_anual_para_aritmetica  # noqa: E402
from core.estimation import estimar_momentos, retornos_simples  # noqa: E402
from core.optimization import otimizar  # noqa: E402
from core.paper import (CUSTO_BPS, LEITURA_FINAL, LEITURA_INTERMEDIARIA, NUCLEO, Acao, Leitura,  # noqa: E402
                        curva_do_registro, decidir_do_dia, e_o_stop, item_3_cumprido, ler_o_paper,
                        rebalanceio_devido)
from core.registro import registrar_carteira  # noqa: E402
from core.risco import LIMITES, Violacao, checar_carteira  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_REGISTRO = RAIZ / "paper" / "carteiras.csv"
ARQUIVO_DIARIO = RAIZ / "paper" / "diario.csv"     # core.diario: avisa quando o lago muda um fechamento já visto
ARQUIVO_PROVENIENCIA = RAIZ / "paper" / "proveniencia.csv"     # core.proveniencia: uma linha por registro
_LOCAL = Path(os.environ.get("LOCALAPPDATA", Path.home()))
PASTA_ESTADO = _LOCAL / "paper-trading"            # runs.csv e logs, fora do repositório (como os do lago)
TRAVA_DO_LAGO = _LOCAL / "b3lake" / "daily.lock"
RAMO_DA_ROTINA = "main"     # o registro só entra na main: trabalho novo do app vai num worktree, como no lago
TAG_DO_PAPER = "paper-v1"   # o código do pré-registro (0ccfeeb): o satélite medido é o que sai dele
# o que define o satélite registrado: os módulos que o caminho importa e, neste script, as definições da decisão
CAMINHO_DO_SATELITE = ["assets.py", "core/__init__.py", "core/bcb.py", "core/config.py", "core/estimation.py",
                       "core/hierarchical.py", "core/lago.py", "core/market_data.py", "core/metrics.py",
                       "core/optimization.py", "core/paper.py", "core/registro.py", "core/risco.py"]
IMPORTS_DO_PROJETO = "imports do projeto"     # de onde vêm otimizar, estimar_momentos etc. (não os da stdlib)
MODULOS_DO_PROJETO = {"assets", "core", "scripts"}
DEFINICOES_DO_SATELITE = [IMPORTS_DO_PROJETO, "ANOS_DE_PRECOS", "Escolha", "Rodada", "ler_precos", "ler_registro",
                          "satelite_do_dia", "_parado", "rodar", "executar", "_escolha_do_dia", "contar_recusas"]
ANOS_DE_PRECOS = 5
_SEM_JANELA = getattr(subprocess, "CREATE_NO_WINDOW", 0)     # pythonw: não abrir console a cada git

Escolha = Callable[[pd.Timestamp], tuple[pd.Series, list[Violacao]]]


@dataclass(frozen=True)
class Rodada:
    data: pd.Timestamp                  # último fechamento do arquivo do lago
    acao: Acao
    carteira: pd.Series | None = None   # a registrar nesta rodada (sem ".SA"); None = nada entra no arquivo
    nota: str = ""

    @property
    def desfecho(self) -> str:
        """O que vai para o runs.csv: rebalanceio devido sem carteira a registrar é a recusa pelas regras."""
        return "recusada" if self.acao is Acao.REBALANCEAR and self.carteira is None else self.acao.value


def ler_precos(arquivo: Path) -> pd.DataFrame:
    """Índice de retorno total do lago: data x papel (sem ".SA"), float64."""
    tabela = pd.read_parquet(arquivo)
    return tabela.set_index(pd.to_datetime(tabela.pop("date"))).astype("float64")


def ler_registro(arquivo: Path) -> pd.DataFrame:
    """O ``paper/carteiras.csv`` com a data como Timestamp; vazio (mesmas colunas) se o arquivo ainda não existe."""
    if not arquivo.exists():
        return pd.DataFrame({"data": pd.Series(dtype="datetime64[ns]"), "ticker": pd.Series(dtype=str),
                             "peso": pd.Series(dtype=float), "preco": pd.Series(dtype=float)})
    return pd.read_csv(arquivo, parse_dates=["data"])


def satelite_do_dia(data: pd.Timestamp, selic: float, universo: list[str],
                    setores: Mapping[str, str]) -> tuple[pd.Series, list[Violacao]]:
    """
    Kelly λ = 2 da carteira de hoje da tela no fechamento de ``data``, sem o filtro de orçamento: ``universo`` e
    ``setores`` no formato do app (com ".SA"), pesos devolvidos sem ".SA" e só os positivos, com o que o
    ``checar_carteira`` diz deles.
    """
    precos = market_data.carregar_precos([*universo, BENCHMARK_ACOES], ANOS_DE_PRECOS, hoje=data.date()).precos
    colunas = [ticker for ticker in universo if ticker in precos.columns]
    momentos = estimar_momentos(retornos_simples(precos[colunas]), shrink_mu=True)
    cfg = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=LIMITES.peso_maximo_acao, n_ativos_max=10,
                           vol_maxima=LIMITES.vol_maxima, aversao_risco=2.0,
                           taxa_livre_risco=taxa_anual_para_aritmetica(selic),
                           grupos=limites_por_setor(setores, LIMITES.teto_setor))
    pesos = otimizar(momentos.mu, momentos.cov, cfg).pesos
    violacoes = checar_carteira(pesos, setores, momentos.cov)
    return pesos[pesos > 0].rename(lambda ticker: ticker.removesuffix(lago.SUFIXO_B3)), violacoes


def _parado(registro: pd.DataFrame, ultima: pd.Timestamp | None) -> bool:
    """O stop já disparou se a última carteira registrada é a do stop."""
    return ultima is not None and e_o_stop(registro[registro["data"] == ultima].set_index("ticker")["peso"])


def rodar(precos: pd.DataFrame, registro: pd.DataFrame, escolher: Escolha) -> Rodada:
    data = precos.index[-1]
    ultima = None if registro.empty else registro["data"].max()
    curva = pd.Series(dtype=float) if ultima is None else curva_do_registro(registro, precos)
    bova = precos[NUCLEO].ffill().reindex(curva.index)
    acao = decidir_do_dia(curva, bova, rebalanceio_devido(ultima, precos.index), _parado(registro, ultima))
    if acao is Acao.STOP:              # sem checar_carteira: o BOVA11 não tem setor no assets.py
        return Rodada(data, acao, pd.Series({NUCLEO: 1.0}))
    if acao is not Acao.REBALANCEAR:
        return Rodada(data, acao)
    pesos, violacoes = escolher(data)
    if violacoes:
        return Rodada(data, acao, nota="recusada: " + "; ".join(
            f"{v.regra.value} em {v.alvo} {v.valor:.2%} (limite {v.limite:.0%})" for v in violacoes))
    return Rodada(data, acao, pesos)


def executar(arquivo_precos: Path, arquivo_registro: Path, trava_do_lago: Path, escolher: Escolha,
             commitar: Callable[[Path, Rodada], None]) -> Rodada | None:
    """Uma rodada da rotina: None com o lago rodando (a cópia do arquivo de preços não é atômica). Grava e commita só
    quando entra carteira."""
    if trava_do_lago.exists():
        return None
    precos = ler_precos(arquivo_precos)
    rodada = rodar(precos, ler_registro(arquivo_registro), escolher)
    if rodada.carteira is not None:
        registrar_carteira(rodada.data.date(), rodada.carteira, precos.ffill().loc[rodada.data], arquivo_registro)
        commitar(arquivo_registro, rodada)
    return rodada


def texto_da_leitura(leitura: Leitura, titulo: str = "Até hoje") -> str:
    comparacao = leitura.comparacao
    linhas = [
        f"{titulo}: {leitura.inicio:%d/%m/%Y} a {leitura.fim:%d/%m/%Y}, {leitura.pregoes} pregões, "
        f"{leitura.registros} registro(s)",
        f"  Satélite (líquido de {CUSTO_BPS:g} bps por lado): {leitura.satelite:+.2%} | BOVA11: {leitura.bova:+.2%} | "
        f"diferença: {leitura.satelite - leitura.bova:+.2%} | CDI: "
        + ("indisponível" if leitura.cdi is None else f"{leitura.cdi:+.2%}"),
        "  P[satélite > BOVA11]: " + ("indisponível" if math.isnan(leitura.p_supera) else f"{leitura.p_supera:.3f}"),
        (f"  Alfa {comparacao.alfa_anual:+.2%} a.a. (t = {comparacao.t_stat_alfa:.2f}), beta {comparacao.beta:.2f}, "
         f"TE {comparacao.tracking_error:.2%}, IR {comparacao.information_ratio:.2f}") if comparacao
        else "  Alfa, beta, TE e IR: a partir de 60 pregões",
        f"  Pregões congelados (perda de 5% ou mais): {leitura.congelados} | Stop: "
        + ("não disparou" if leitura.stop is None else f"{leitura.stop:%d/%m/%Y}"),
    ]
    veredito = item_3_cumprido(leitura)
    if veredito is not None:
        linhas.append(f"  Item 3 do critério de pronto: {'cumprido' if veredito else 'não cumprido'} "
                      "(regra pré-registrada: sem stop e satélite líquido >= BOVA11)")
    return "\n".join(linhas)


def relatorio(rodada: Rodada, arquivo_precos: Path, arquivo_registro: Path) -> str:
    """A decisão do dia e as leituras do pré-registro: até hoje e, quando chegam, as fixas de 126 e 252 pregões."""
    partes = [f"Fechamento de {rodada.data:%d/%m/%Y}: {rodada.acao.value}" + (f" ({rodada.nota})" if rodada.nota
                                                                                else "")]
    if rodada.carteira is not None:
        partes.append("  Registrada: " + ", ".join(f"{t} {p:.1%}" for t, p in rodada.carteira.items()))
    registro = ler_registro(arquivo_registro)
    if registro.empty:
        return "\n".join(partes)
    precos = ler_precos(arquivo_precos)
    cdi = _cdi(registro["data"].min().date(), rodada.data.date())
    leituras = [(ler_o_paper(registro, precos, cdi), "Até hoje")]
    for pregoes, titulo in ((LEITURA_INTERMEDIARIA, "Leitura de 6 meses"), (LEITURA_FINAL, "Leitura de 12 meses")):
        leitura = ler_o_paper(registro, precos, cdi, ate=pregoes)
        if leitura.pregoes == pregoes:
            leituras.append((leitura, titulo))
    for leitura, titulo in leituras:
        partes += [texto_da_leitura(leitura, titulo),
                   f"  Recusas pelas regras (runs.csv): {contar_recusas(PASTA_ESTADO / 'runs.csv', leitura.fim)}"]
    return "\n".join(partes)


def _cdi(inicio: date, hoje: date) -> pd.Series | None:
    try:
        return market_data.baixar_cdi_diario(max(1, math.ceil((hoje - inicio).days / 365.25)), hoje)
    except market_data.ErroDadosMercado as erro:
        logging.warning("CDI indisponível: %s", erro)
        return None


def _git(*argumentos: str, raiz: Path = RAIZ) -> str:
    # UTF-8 explícito: sem -X utf8, o Windows decodificaria com a página de código (cp1252 não lê o 0x8d do "Í")
    return subprocess.run(["git", "-C", str(raiz), *argumentos], check=True, capture_output=True, text=True,
                          encoding="utf-8", creationflags=_SEM_JANELA).stdout.strip()


def versao_do_codigo(raiz: Path = RAIZ) -> str:
    """Hash curto do HEAD do checkout que rodou, com " (sujo)" se algum arquivo versionado mudou desde ele (os soltos
    não versionados da raiz, os preços que o lago publica a cada pregão e o diário e a proveniência que a rotina
    grava não contam)."""
    sujo = _git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)lago/precos_diarios.parquet",
                ":(exclude)paper/diario.csv", ":(exclude)paper/proveniencia.csv", raiz=raiz)
    return _git("rev-parse", "--short", "HEAD", raiz=raiz) + (" (sujo)" if sujo else "")


def mudancas_desde_o_paper(raiz: Path, caminhos: list[str], tag: str = TAG_DO_PAPER) -> list[str]:
    """Arquivos de ``caminhos`` que diferem da ``tag`` no checkout (commitados ou não)."""
    return _git("diff", "--name-only", tag, "--", *caminhos, raiz=raiz).splitlines()


def _sem_docstrings(arvore: ast.AST) -> ast.AST:
    """A árvore sem as docstrings de funções e classes (texto, não decisão)."""
    for no in ast.walk(arvore):
        if (isinstance(no, ast.FunctionDef | ast.ClassDef) and no.body and isinstance(no.body[0], ast.Expr)
                and isinstance(no.body[0].value, ast.Constant) and isinstance(no.body[0].value.value, str)):
            no.body = no.body[1:] or [ast.Pass()]
    return arvore


def _definicoes(fonte: str) -> dict[str, str]:
    """Funções, classes e constantes do nível do módulo pela árvore sintática (comentário, espaço e docstring não
    contam)."""
    definicoes: dict[str, str] = {}
    arvore = _sem_docstrings(ast.parse(fonte))
    do_projeto = [no for no in arvore.body if isinstance(no, ast.ImportFrom) and no.module
                  and no.module.split(".")[0] in MODULOS_DO_PROJETO]
    definicoes[IMPORTS_DO_PROJETO] = ast.dump(ast.Module(body=do_projeto, type_ignores=[]))
    for no in arvore.body:
        if isinstance(no, ast.FunctionDef | ast.ClassDef):
            definicoes[no.name] = ast.dump(no)
        elif isinstance(no, ast.Assign):
            for alvo in no.targets:
                if isinstance(alvo, ast.Name):
                    definicoes[alvo.id] = ast.dump(no)
    return definicoes


def definicoes_mudadas(antes: str, depois: str, nomes: list[str]) -> list[str]:
    """Quais de ``nomes`` mudaram (ou sumiram) entre duas versões da fonte de um módulo."""
    velhas, novas = _definicoes(antes), _definicoes(depois)
    return [nome for nome in nomes if velhas.get(nome) != novas.get(nome)]


def codigo_mudado_desde_o_paper(raiz: Path = RAIZ, tag: str = TAG_DO_PAPER) -> list[str]:
    """O que mudou desde a ``tag`` no caminho do satélite: os módulos inteiros e, neste script, só as definições do
    satélite (o ``main()`` e o commit mudam sem mudar a estratégia medida)."""
    script = Path(__file__).resolve().relative_to(RAIZ).as_posix()
    antes = _git("show", f"{tag}:{script}", raiz=raiz)
    depois = (raiz / script).read_text(encoding="utf-8")
    return (mudancas_desde_o_paper(raiz, CAMINHO_DO_SATELITE, tag)
            + [f"{script}: {nome}" for nome in definicoes_mudadas(antes, depois, DEFINICOES_DO_SATELITE)])


@dataclass(frozen=True)
class SituacaoDoCodigo:
    versao: str             # versao_do_codigo, ou "?" se o git falhou
    mudancas: list[str]     # codigo_mudado_desde_o_paper: não vazia = o satélite medido deixou de ser o do paper
    erro: str = ""          # a guarda não conseguiu conferir (ex.: checkout sem a tag)


def situacao_do_codigo(raiz: Path = RAIZ) -> SituacaoDoCodigo:
    """Versão e mudanças desde a tag, sem nunca levantar: a guarda avisa, não derruba a decisão do dia."""
    try:
        versao = versao_do_codigo(raiz)
        return SituacaoDoCodigo(versao, codigo_mudado_desde_o_paper(raiz))
    except (OSError, subprocess.CalledProcessError, SyntaxError, ValueError) as erro:
        detalhe = getattr(erro, "stderr", None) or str(erro)
        return SituacaoDoCodigo("?", [], f"guarda do {TAG_DO_PAPER} indisponível: {detalhe.strip()[:200]}")


def commitar_no_git(arquivo: Path, rodada: Rodada, versao: str, raiz: Path = RAIZ,
                    junto: Sequence[Path] = ()) -> None:
    """Commit local do registro e dos arquivos ``junto`` (a proveniência e o marco dos preços), com a
    ``versao_do_codigo`` que decidiu (sem push: push na main é produção, feito à mão)."""
    caminhos = [caminho.relative_to(raiz).as_posix() for caminho in (arquivo, *junto)]
    _git("add", "--", *caminhos, raiz=raiz)
    _git("commit", "-m", f"paper: {rodada.acao.value} no fechamento de {rodada.data:%d/%m/%Y} (código {versao})",
         "--", *caminhos, raiz=raiz)


def commitar_registro(arquivo: Path, rodada: Rodada, versao: str, notas: Sequence[str], arquivo_precos: Path,
                      raiz: Path = RAIZ) -> None:
    """O commit da rodada que registrou: a carteira, a linha dela no ``paper/proveniencia.csv`` (código, hash dos
    preços, Selic das ``notas`` e versões) e, se versionado no app, o arquivo de preços lido (marco)."""
    from core.diario import sha256_do_arquivo         # aqui dentro: ver conferir_o_diario
    from core.proveniencia import registrar_proveniencia

    proveniencia = arquivo.parent / ARQUIVO_PROVENIENCIA.name
    selic = next((nota for nota in notas if nota.startswith("Selic ")), "")
    registrar_proveniencia(proveniencia, rodada.data.date(), versao, sha256_do_arquivo(arquivo_precos), selic,
                           datetime.now())
    junto = [proveniencia]
    if arquivo_precos.is_relative_to(raiz):
        junto.append(arquivo_precos)
    commitar_no_git(arquivo, rodada, versao, raiz=raiz, junto=junto)


def commitar_pendente(arquivo: Path, raiz: Path = RAIZ, junto: Sequence[Path] = ()) -> bool:
    """Commita o registro que ficou só no disco (o commit da rodada que o gravou falhou: sem isso, a rodada seguinte
    vê a data registrada, segura a carteira e a linha nunca entra no git), com os arquivos ``junto`` que existirem
    (a proveniência gravada na mesma rodada). True se havia o que commitar. Só linhas acrescentadas: uma linha já
    commitada que mudou é ValueError e nada entra."""
    caminho = arquivo.relative_to(raiz).as_posix()
    caminhos = [caminho, *(extra.relative_to(raiz).as_posix() for extra in junto if extra.exists())]
    if not _git("status", "--porcelain", "--", caminho, raiz=raiz):
        return False
    try:
        commitado = _git("show", f"HEAD:{caminho}", raiz=raiz)
    except subprocess.CalledProcessError:          # nem o 1º registro chegou ao git
        commitado = ""
    if not arquivo.read_text(encoding="utf-8").startswith(commitado):
        raise ValueError(f"{caminho} foi reescrito fora da rotina (o arquivo só cresce): conferir com git diff.")
    _git("add", "--", *caminhos, raiz=raiz)
    _git("commit", "-m", "paper: registro que ficou fora do git na rodada anterior", "--", *caminhos, raiz=raiz)
    return True


def conferir_o_diario(rodada: Rodada, arquivo_diario: Path, arquivo_registro: Path, arquivo_precos: Path,
                      agora: datetime, raiz: Path = RAIZ) -> list[pd.Timestamp]:
    """Acrescenta ao diário os fechamentos novos e as revisões que o lago fez no passado, commita se algo entrou e
    devolve os fechamentos revistos (vazio = o lago não mudou o que a rotina já tinha visto)."""
    # aqui dentro: um import do projeto no topo do script mudaria o que a guarda do paper-v1 compara
    from core.diario import atualizar_diario, sha256_do_arquivo

    revistos = atualizar_diario(arquivo_diario, ler_registro(arquivo_registro), ler_precos(arquivo_precos),
                                sha256_do_arquivo(arquivo_precos), agora)
    caminho = arquivo_diario.relative_to(raiz).as_posix()
    if arquivo_diario.exists() and _git("status", "--porcelain", "--", caminho, raiz=raiz):
        _git("add", "--", caminho, raiz=raiz)
        _git("commit", "-m", f"paper: diário do fechamento de {rodada.data:%d/%m/%Y}", "--", caminho, raiz=raiz)
    return revistos


def notas_do_diario(rodada: Rodada, agora: datetime, log: Path) -> list[str]:
    """O diário do dia sem nunca derrubar a rodada: falha dele ou passado mudado pelo lago viram nota e aviso."""
    try:
        from core.diario import texto_das_revisoes     # aqui dentro: ver conferir_o_diario

        revistos = conferir_o_diario(rodada, ARQUIVO_DIARIO, ARQUIVO_REGISTRO, lago.ARQUIVO_PRECOS, agora)
    except Exception as erro:
        logging.exception("diário do paper trading falhou")
        avisar("paper trading: o diário falhou", f"{str(erro)[:150]} Log: {log}")
        return [f"diário falhou: {str(erro)[:150]}"]
    if not revistos:
        return []
    texto = texto_das_revisoes(revistos)
    logging.warning("%s (conferir no lago; o registro não muda)", texto)
    avisar("paper trading: o lago mudou dados passados", f"{texto}. Log: {log}")
    return [texto]


def _escolha_do_dia(notas: list[str]) -> Escolha:
    universo = get_tickers_liquidos()
    info = get_ticker_info()
    setores = {ticker: info[ticker]["setor"] for ticker in universo}

    def escolher(data: pd.Timestamp) -> tuple[pd.Series, list[Violacao]]:
        selic = market_data.obter_selic_meta(data.date())
        notas.append(f"Selic {selic.valor:.2%} ({selic.fonte})")
        return satelite_do_dia(data, selic.valor, universo, setores)

    return escolher


def anotar(inicio: datetime, fechamento: str, desfecho: str, nota: str) -> None:
    """Uma linha do ``runs.csv`` da rotina (o histórico que se lê quando chega um aviso)."""
    arquivo = PASTA_ESTADO / "runs.csv"
    novo = not arquivo.exists()
    with arquivo.open("a", encoding="utf-8", newline="") as saida:
        escritor = csv.writer(saida, lineterminator="\n")
        if novo:
            escritor.writerow(["inicio", "fim", "fechamento", "desfecho", "nota"])
        escritor.writerow([inicio.isoformat(timespec="seconds"), datetime.now().isoformat(timespec="seconds"),
                           fechamento, desfecho, nota])


def contar_recusas(arquivo: Path, ate: pd.Timestamp) -> int:
    """Fechamentos até ``ate`` em que a carteira nova foi recusada pelas regras, pelo ``runs.csv`` (rodar de novo no
    mesmo fechamento conta uma vez); 0 sem o arquivo."""
    if not arquivo.exists():
        return 0
    rodadas = pd.read_csv(arquivo, dtype=str, keep_default_na=False)
    recusadas = pd.to_datetime(rodadas.loc[rodadas["desfecho"] == "recusada", "fechamento"].unique())
    return int((recusadas <= ate).sum())


# Aviso do Windows pelo WinRT, como o do lago: o texto vai em variáveis de ambiente, nunca no script nem no XML.
_SCRIPT_DO_AVISO = """
$null = [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime]
$null = [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime]
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml('<toast><visual><binding template="ToastGeneric"><text/><text/></binding></visual></toast>')
$textos = $xml.GetElementsByTagName('text')
$null = $textos.Item(0).AppendChild($xml.CreateTextNode($env:PAPER_AVISO_TITULO))
$null = $textos.Item(1).AppendChild($xml.CreateTextNode($env:PAPER_AVISO_CORPO))
$aviso = [Windows.UI.Notifications.ToastNotification]::new($xml)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($env:PAPER_AVISO_APP).Show($aviso)
"""
_APP_DO_POWERSHELL = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"


def avisar(titulo: str, corpo: str) -> None:
    ambiente = {**os.environ, "PAPER_AVISO_TITULO": titulo, "PAPER_AVISO_CORPO": corpo,
                "PAPER_AVISO_APP": _APP_DO_POWERSHELL}
    try:
        subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
                        _SCRIPT_DO_AVISO], env=ambiente, check=True, capture_output=True, timeout=60,
                       creationflags=_SEM_JANELA)
    except (OSError, subprocess.SubprocessError):     # o aviso não tem como avisar de si mesmo: fica o log
        logging.exception("aviso do Windows falhou")


def _configurar_log(inicio: datetime) -> Path:
    log = PASTA_ESTADO / "logs" / f"{inicio:%Y%m%d-%H%M%S}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    manipuladores: list[logging.Handler] = [logging.FileHandler(log, encoding="utf-8")]
    if sys.stderr is not None:                      # pythonw não tem stderr: o arquivo é o único log
        manipuladores.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=manipuladores)
    return log


def main() -> int:
    inicio = datetime.now()
    log = _configurar_log(inicio)
    try:
        ramo = _git("rev-parse", "--abbrev-ref", "HEAD")
        if ramo != RAMO_DA_ROTINA:
            raise RuntimeError(f"checkout do app em '{ramo}', não na {RAMO_DA_ROTINA}: o registro não pode ir para "
                               "outro ramo")
        notas: list[str] = []
        if commitar_pendente(ARQUIVO_REGISTRO, junto=[ARQUIVO_PROVENIENCIA]):
            logging.warning("registro que ficou fora do git na rodada anterior: commitado agora")
            notas.append("registro pendente commitado")
        codigo = situacao_do_codigo()
        notas.append(f"código {codigo.versao}")
        if codigo.erro:
            logging.warning(codigo.erro)
            notas.append(codigo.erro)
            avisar("paper trading: guarda do código indisponível", f"{codigo.erro[:150]} Log: {log}")
        if codigo.mudancas:     # o registro só cresce: uma carteira de outro código misturaria duas estratégias
            nota = f"satélite fora do {TAG_DO_PAPER}: " + ", ".join(codigo.mudancas)
            logging.error("%s; nada decidido (reverter o código ou pré-registro novo)", nota)
            anotar(inicio, "", "bloqueada", "; ".join([*notas, nota]))
            avisar("paper trading: bloqueada, código do satélite mudou", f"{nota[:150]} Log: {log}")
            return 1
        rodada = executar(lago.ARQUIVO_PRECOS, ARQUIVO_REGISTRO, TRAVA_DO_LAGO, _escolha_do_dia(notas),
                          lambda arquivo, rodada: commitar_registro(arquivo, rodada, codigo.versao, notas,
                                                                    lago.ARQUIVO_PRECOS))
        if rodada is None:
            logging.info("lago rodando (daily.lock): nada decidido")
            anotar(inicio, "", "lago rodando", "; ".join(notas))
            return 0
        notas += notas_do_diario(rodada, inicio, log)
        anotar(inicio, rodada.data.date().isoformat(), rodada.desfecho, "; ".join(filter(None, [*notas, rodada.nota])))
        logging.info("\n%s", relatorio(rodada, lago.ARQUIVO_PRECOS, ARQUIVO_REGISTRO))
        if rodada.desfecho == "recusada":
            avisar("paper trading: carteira recusada", f"{rodada.nota[:180]} Log: {log}")
        return 0
    except Exception as erro:         # qualquer falha da rotina vira aviso; o log tem o traceback
        logging.exception("rotina do paper trading falhou")
        anotar(inicio, "", "falhou", str(erro)[:300])
        avisar("paper trading: a rotina falhou", f"Veja o log {log}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
