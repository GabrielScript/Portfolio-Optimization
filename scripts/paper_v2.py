"""
Paper-v2 da Conservadora (S5, depois da prova final, DECISOES 82): rotina diária que, no dia devido, decide a
Conservadora no último fechamento do lago, como o motor decide, e registra em ``paper-v2/carteiras.csv``. Separado do
paper-v1 (o satélite Kelly, ``scripts/paper_trading.py``), que segue igual. Sem rede, fora o ``universe`` do lago.

Uso: .venv/Scripts/python -X utf8 scripts/paper_v2.py             (a rotina; agendar só com o OK do usuário)
     .venv/Scripts/python -X utf8 scripts/paper_v2.py --leitura   (a leitura do registro até o último fechamento)

Pré-registro (2026-10-20, escrito e commitado antes do 1º registro; depois desse commit o texto não muda):
- O que se mede: a "Conservadora (15 nomes, pesos iguais, com regras)" aprovada na prova final com execução realista
  (`scripts/avaliar_execucao.py`), como carteira inteira: pesos-alvo que somam 100%, tickers do lago. O 1º registro
  começa o relógio de 6 a 12 meses do critério de pronto para ela (ROTEIRO §2, item 3).
- Decisão (`conservadora_do_dia`, a mesma do `walk_forward_dinamico` com `estrategia_conservadora`, conferida contra
  ele nos arquivos da prova): no fechamento D, o treino são os 1.260 retornos até D do `universe_prices.parquet` do
  lago (até 5 pregões sem negócio repetem o preço; papel com buraco maior na janela fica fora), nos papéis do universo
  ponto no tempo (`universe_monthly.parquet`) do mês do pregão seguinte, na ordem do arquivo; a nota é a do mês de D,
  com `monthly_total_returns.parquet` e `dividends.parquet`; setores do `sectors.parquet`; `checar_carteira` com a
  cov de Ledoit-Wolf do treino. O pregão seguinte é o dia útil seguinte, sem 24/12, 25/12, 31/12 e 01/01.
- Calendário: rebalanceio com 63 pregões ou mais depois do último registro, no calendário do
  `lago/precos_diarios.parquet` (o lago publica todo pregão). No dia devido, a rotina roda o `universe` do lago (preços
  e universo até D) e confere que o `universe_prices` chega a D; se não chega, sai sem decidir e tenta no pregão
  seguinte. Perda de 5% ou mais da curva registrada no último pregão = nenhuma ordem (`pode_negociar`; tenta no
  pregão seguinte). Carteira recusada pelas regras = fica a registrada e a rotina tenta no pregão seguinte (o motor
  esperaria o próximo rebalanceio marcado; a Conservadora não teve recusa em 2012–2026). Sem stop (o backtest não
  tinha). Rodar de novo no mesmo fechamento não registra nada. Lago rodando (`daily.lock`) = sai sem decidir.
- Registro: `paper-v2/carteiras.csv` (data, ticker, peso, valor do índice de retorno total; só acrescenta), uma linha
  por registro em `paper-v2/proveniencia.csv` (código, sha256 dos cinco arquivos e o manifest do lago) e a cópia dos
  cinco em `paper-v2/marcos/<fechamento>/` (o lago reescreve os originais). Commit local só na main; push à mão.
- Leituras: com 126 pregões depois do 1º registro, relatório; com 252, a decisão. Medidas (`leitura`): a sequência
  registrada refeita por `core.execucao.refazer` com R$ 10 mil, os custos da prova final (emolumentos de 0,030%, meio
  spread e impacto com Y = 1, com a liquidez do mês anterior do `monthly_metrics.parquet`) e o IR de pessoa física
  (isenção até R$ 20 mil de vendas no mês, proventos na base), contra o BOVA11 comprado no mesmo fechamento do 1º
  registro e mantido (sem isenção), nos valores de saída: P[> BOVA11] aritmético e em log, CAGR, MDD, saída final,
  custo por ano e IR.
- Decisão em 252 pregões, fixa agora: o item 3 do critério de pronto está cumprido para a Conservadora se a saída
  final dela for maior ou igual à do BOVA11 nos mesmos pregões. Senão, não está, e ela segue igual. Como no paper-v1,
  um ano não separa alfa de sorte (o erro padrão de um ano de alfa é do tamanho do TE): o paper prova a operação
  (dados, rotina, regras), não o alfa.
- Dado errado achado depois (conferido no lago): a linha registrada fica, o relatório marca o trecho e o DECISOES
  registra. Mudar a estratégia ou as regras = pré-registro novo e relógio do zero.
- Expectativa registrada: 1º registro no fechamento de 19/10/2026 (rodado à mão; a rotina diária entra quando o
  usuário autorizar o agendamento); 4 rebalanceios em 12 meses; 0 recusas; 0 a 2 pregões congelados; giro perto de
  300% ao ano, como no motor.
"""

from __future__ import annotations

import csv
import hashlib
import json
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

from core.backtest import (DIAS_SEM_NEGOCIO_NO_TREINO, CarteiraRecusada, ErroUniverso, Estrategia,  # noqa: E402
                           ResultadoBacktest, retornos_da_carteira)
from core.execucao import refazer  # noqa: E402
from core.impostos import ISENCAO_DE_VENDAS  # noqa: E402
from core.lago import ler_setores  # noqa: E402
from core.paper import Acao, curva_do_registro, rebalanceio_devido  # noqa: E402
from core.registro import registrar_carteira  # noqa: E402
from core.risco import pode_negociar  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from scripts.avaliar_conservadora import CONSERVADORA  # noqa: E402
from scripts.avaliar_execucao import (CUSTOS_DA_B3, EM_PORCENTO, EM_REAIS, comprado_e_mantido,  # noqa: E402
                                      liquidez_da_tabela, medidas, proventos_diarios)
from scripts.avaliar_regras_no_motor import setores_dos_papeis  # noqa: E402
from scripts.avaliar_robustez import _tabela  # noqa: E402
from scripts.paper_trading import (IMPORTS_DO_PROJETO, Rodada, _git, definicoes_mudadas, ler_precos,  # noqa: E402
                                   ler_registro, mudancas_desde_o_paper, versao_do_codigo)

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DO_LAGO = RAIZ / "data" / "b3lake"            # exports do lago (fora do git); o `universe` grava aqui
ARQUIVO_CALENDARIO = RAIZ / "lago" / "precos_diarios.parquet"
PASTA = RAIZ / "paper-v2"
ARQUIVO_REGISTRO = PASTA / "carteiras.csv"
ARQUIVO_PROVENIENCIA = PASTA / "proveniencia.csv"
PASTA_MARCOS = PASTA / "marcos"
ARQUIVOS_DO_LAGO = ("universe_prices.parquet", "universe_monthly.parquet", "sectors.parquet",
                    "monthly_total_returns.parquet", "dividends.parquet")
_LOCAL = Path(os.environ.get("LOCALAPPDATA", Path.home()))
PASTA_ESTADO = _LOCAL / "paper-v2"                  # runs.csv, fora do repositório
TRAVA_DO_LAGO = _LOCAL / "b3lake" / "daily.lock"
PROJETO_DO_LAGO = RAIZ.parents[1] / "b3-datalake"
RAMO_DA_ROTINA = "main"
TAG_DO_PAPER = "paper-v2"
# o que define a Conservadora registrada: os módulos do caminho da decisão e, neste script, as definições dela
CAMINHO_DA_CONSERVADORA = ["core/backtest.py", "core/config.py", "core/estimation.py", "core/lago.py", "core/paper.py",
                           "core/registro.py", "core/risco.py", "core/sinais.py", "scripts/avaliar_regras_no_motor.py"]
DEFINICOES_DA_CONSERVADORA = [IMPORTS_DO_PROJETO, "JANELA_TREINO", "SEM_PREGAO", "DadosDoLago", "ler_dados",
                              "mes_da_carteira", "acao_do_dia", "estrategia", "treino_do_dia",
                              "conservadora_do_dia", "rodar"]
JANELA_TREINO = 1_260
SEM_PREGAO = {(12, 24), (12, 25), (12, 31), (1, 1)}      # (mês, dia) sem pregão na B3 todo ano
CAPITAL_DA_LEITURA = 10_000.0
_SEM_JANELA = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True)
class DadosDoLago:
    precos: pd.DataFrame                # índice de retorno total diário, data x papel (universo de sempre + ETFs)
    universo: pd.DataFrame              # month, security: o universo ponto no tempo
    setores: dict[str, str]
    retornos_mensais: pd.DataFrame      # mês (1º dia) x papel
    proventos: pd.DataFrame             # security, ex_date, yield
    sha256: dict[str, str]
    manifest: dict[str, str]


def ler_dados(pasta: Path = PASTA_DO_LAGO) -> DadosDoLago:
    """Os cinco arquivos do lago na forma do motor, com o sha256 de cada um e as datas do manifest."""
    sha256 = {nome: hashlib.sha256((pasta / nome).read_bytes()).hexdigest() for nome in ARQUIVOS_DO_LAGO}
    tabela = pd.read_parquet(pasta / "universe_prices.parquet")
    precos = tabela.set_index(pd.DatetimeIndex(pd.to_datetime(tabela.pop("date")), name=None)).astype("float64")
    emissores = pd.read_parquet(pasta / "universe_monthly.parquet", columns=["month", "security", "issuer"])
    emissores["month"] = pd.to_datetime(emissores["month"])
    setores = setores_dos_papeis(emissores, ler_setores(pasta / "sectors.parquet"))
    mensais = pd.read_parquet(pasta / "monthly_total_returns.parquet")
    mensais = mensais.set_index(pd.DatetimeIndex(pd.to_datetime(mensais.pop("month")), name=None)).astype("float64")
    proventos = pd.read_parquet(pasta / "dividends.parquet", columns=["security", "ex_date", "yield"])
    # o lago grava o manifest em cp1252; só as datas, que são ASCII, interessam
    manifest = json.loads((pasta / "manifest.json").read_bytes().decode("utf-8", errors="replace"))
    return DadosDoLago(precos, emissores[["month", "security"]], setores, mensais, proventos, sha256,
                       {campo: manifest.get(campo, "") for campo in ("generated_at", "data_through")})


def mes_da_carteira(data: pd.Timestamp) -> pd.Timestamp:
    """O mês (1º dia) do pregão seguinte a ``data``, o do universo da carteira decidida no fechamento dela."""
    seguinte = data + pd.offsets.BDay(1)
    while (seguinte.month, seguinte.day) in SEM_PREGAO:
        seguinte += pd.offsets.BDay(1)
    return seguinte.to_period("M").to_timestamp()


def acao_do_dia(curva: pd.Series, devido: bool) -> Acao:
    """Perda de 5% ou mais da curva registrada no último pregão congela; fora do dia devido, segura."""
    if not pode_negociar(curva.pct_change().dropna()):
        return Acao.CONGELADO
    return Acao.REBALANCEAR if devido else Acao.AGUARDAR


def estrategia(dados: DadosDoLago) -> Estrategia:
    return estrategia_conservadora(dados.retornos_mensais, dados.proventos, dados.setores)


def treino_do_dia(dados: DadosDoLago, data: pd.Timestamp) -> pd.DataFrame:
    """O treino que o ``decidir`` do ``walk_forward_dinamico`` passaria à estratégia no fechamento de ``data``;
    ``ErroUniverso`` sem a janela inteira ou sem papel elegível com ela."""
    precos = dados.precos.loc[:data]
    treino = precos.ffill(limit=DIAS_SEM_NEGOCIO_NO_TREINO).pct_change().iloc[1:].iloc[-JANELA_TREINO:]
    if len(treino) < JANELA_TREINO:
        raise ErroUniverso(f"{data:%Y-%m-%d}: só {len(treino)} retornos, o treino pede {JANELA_TREINO}")
    mes = mes_da_carteira(data)
    nomes = [papel for papel in dados.universo.loc[dados.universo["month"] == mes, "security"]
             if papel in precos.columns]
    treino = treino[nomes].dropna(axis=1)
    if treino.empty:
        raise ErroUniverso(f"{data:%Y-%m-%d}: nenhum papel elegível com a janela inteira ({len(nomes)} no mês)")
    return treino


def conservadora_do_dia(dados: DadosDoLago, data: pd.Timestamp) -> pd.Series:
    """A carteira que o motor decidiria no fechamento de ``data``, só os pesos positivos; ``CarteiraRecusada`` se as
    regras recusam."""
    pesos = estrategia(dados)(treino_do_dia(dados, data))
    return pesos[pesos > 0]


def resultado_do_registro(registro: pd.DataFrame, precos: pd.DataFrame) -> ResultadoBacktest:
    """O registro na forma do motor: cada carteira vale do pregão seguinte ao da data registrada (a compra é no
    fechamento dela); a do último fechamento, que ainda não valeu, fica fora."""
    pesos = registro.pivot_table(index="data", columns="ticker", values="peso", aggfunc="sum").fillna(0.0)
    datas = precos.index[precos.index >= pesos.index[0]]
    vale_de = {datas[datas.get_loc(data) + 1]: linha for data, linha in pesos.iterrows()
               if datas.get_loc(data) + 1 < len(datas)}
    alvos = pd.DataFrame(vale_de).T.rename_axis(index=None, columns=None)
    return ResultadoBacktest(CONSERVADORA, "paper-v2", curva_do_registro(registro, precos).reindex(datas), alvos,
                             pd.Series(dtype=float), 0.0)


def liquidez_do_lago(pasta: Path = PASTA_DO_LAGO):
    """A liquidez mensal do ``monthly_metrics`` do lago, na forma do arquivo da prova final."""
    metricas = pd.read_parquet(pasta / "monthly_metrics.parquet",
                               columns=["month", "security", "quoted_spread", "vol_daily", "median_dtv_brl"])
    return liquidez_da_tabela(pd.DataFrame({
        "month": metricas["month"], "security": metricas["security"].astype(str),
        "meio_spread": metricas["quoted_spread"].astype("float64") / 2,
        "vol_diaria": metricas["vol_daily"].astype("float64"),
        "volume_mediano": metricas["median_dtv_brl"].astype("float64")}))


def leitura(registro: pd.DataFrame, dados: DadosDoLago, dividendos: pd.DataFrame, liquidez) -> pd.DataFrame:
    """A medida do pré-registro: o registro refeito com R$ 10 mil, custos e IR, contra o BOVA11, nos valores de
    saída (``dividendos`` com ``corporate_action``, para a base dos proventos)."""
    resultado = resultado_do_registro(registro, dados.precos)
    retornos = retornos_da_carteira(dados.precos)
    proventos = proventos_diarios(dividendos, retornos.index, retornos.columns)
    refeitas = {CONSERVADORA: refazer(resultado, retornos, CAPITAL_DA_LEITURA, CUSTOS_DA_B3, liquidez,
                                      isencao=ISENCAO_DE_VENDAS, proventos=proventos),
                "BOVA11": refazer(comprado_e_mantido(resultado, "BOVA11"), retornos, CAPITAL_DA_LEITURA, CUSTOS_DA_B3,
                                  liquidez, isencao=0.0)}
    return medidas(refeitas)


def rodar(dados: DadosDoLago, registro: pd.DataFrame, data: pd.Timestamp) -> Rodada:
    """A decisão do dia devido no fechamento ``data`` (os dados já chegam a ele)."""
    curva = pd.Series(dtype=float) if registro.empty else curva_do_registro(registro, dados.precos)
    acao = acao_do_dia(curva, devido=True)
    if acao is not Acao.REBALANCEAR:
        return Rodada(data, acao)
    try:
        return Rodada(data, acao, conservadora_do_dia(dados, data))
    except CarteiraRecusada as recusa:
        return Rodada(data, acao, nota="recusada: " + "; ".join(
            f"{v.regra.value} em {v.alvo} {v.valor:.2%} (limite {v.limite:.0%})" for v in recusa.violacoes))


def atualizar_universo(projeto: Path = PROJETO_DO_LAGO) -> None:
    """O ``universe`` do lago: preços e universo ponto no tempo até o último fechamento, copiados para o app."""
    subprocess.run(["uv", "run", "python", "-m", "b3lake.cli", "universe"], cwd=projeto, check=True,
                   capture_output=True, text=True, encoding="utf-8", timeout=3600, creationflags=_SEM_JANELA)


def executar(arquivo_calendario: Path, arquivo_registro: Path, trava_do_lago: Path, atualizar: Callable[[], None],
             ler: Callable[[], DadosDoLago], guardar: Callable[[Rodada, DadosDoLago], None]) -> Rodada | None:
    """Uma rodada: None com o lago rodando. Fora do dia devido, nem lê o lago; no dia devido, ``atualizar`` o
    universo, ``ler`` os arquivos, conferir que chegam ao último fechamento e decidir; ``guardar`` só quando entra
    carteira."""
    if trava_do_lago.exists():
        return None
    calendario = ler_precos(arquivo_calendario).index
    data = calendario[-1]
    registro = ler_registro(arquivo_registro)
    ultima = None if registro.empty else registro["data"].max()
    if ultima is not None and ultima >= data:
        return Rodada(data, Acao.AGUARDAR, nota="fechamento já decidido")
    if not rebalanceio_devido(ultima, calendario):
        return Rodada(data, Acao.AGUARDAR)
    atualizar()
    dados = ler()
    if dados.precos.index[-1] != data:
        return Rodada(data, Acao.AGUARDAR, nota=f"universe_prices até {dados.precos.index[-1]:%d/%m/%Y}: sem decidir")
    rodada = rodar(dados, registro, data)
    if rodada.carteira is not None:
        registrar_carteira(data.date(), rodada.carteira, dados.precos.ffill().loc[data], arquivo_registro)
        guardar(rodada, dados)
    return rodada


def codigo_mudado_desde_o_paper(raiz: Path = RAIZ, tag: str = TAG_DO_PAPER) -> list[str]:
    """O que mudou desde a ``tag`` no caminho da Conservadora: os módulos inteiros e, neste script, só as definições
    da decisão (o ``main()`` e o registro mudam sem mudar a estratégia medida)."""
    script = Path(__file__).resolve().relative_to(raiz).as_posix()
    antes = _git("show", f"{tag}:{script}", raiz=raiz)
    depois = (raiz / script).read_text(encoding="utf-8")
    return (mudancas_desde_o_paper(raiz, CAMINHO_DA_CONSERVADORA, tag)
            + [f"{script}: {nome}" for nome in definicoes_mudadas(antes, depois, DEFINICOES_DA_CONSERVADORA)])


def guardar_no_git(rodada: Rodada, dados: DadosDoLago, versao: str, raiz: Path = RAIZ,
                   pasta_do_lago: Path = PASTA_DO_LAGO) -> None:
    """Proveniência, marco dos cinco arquivos e commit local (só na main; push à mão)."""
    marco = PASTA_MARCOS / f"{rodada.data:%Y-%m-%d}"
    marco.mkdir(parents=True, exist_ok=True)
    for nome in ARQUIVOS_DO_LAGO:
        shutil.copy2(pasta_do_lago / nome, marco / nome)
    novo = not ARQUIVO_PROVENIENCIA.exists()
    with ARQUIVO_PROVENIENCIA.open("a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo, lineterminator="\n")
        if novo:
            escritor.writerow(["data", "codigo", *ARQUIVOS_DO_LAGO, "manifest_gerado", "manifest_ate"])
        escritor.writerow([f"{rodada.data:%Y-%m-%d}", versao, *(dados.sha256[nome] for nome in ARQUIVOS_DO_LAGO),
                           dados.manifest["generated_at"], dados.manifest["data_through"]])
    if _git("rev-parse", "--abbrev-ref", "HEAD", raiz=raiz) != RAMO_DA_ROTINA:
        return
    caminhos = [caminho.relative_to(raiz).as_posix() for caminho in (ARQUIVO_REGISTRO, ARQUIVO_PROVENIENCIA, marco)]
    _git("add", "--", *caminhos, raiz=raiz)
    _git("commit", "-m", f"paper-v2: {rodada.acao.value} no fechamento de {rodada.data:%d/%m/%Y} (código {versao})",
         "--", *caminhos, raiz=raiz)


def _anotar(rodada: Rodada | None, versao: str, aviso: str) -> None:
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


def main() -> int:
    versao = versao_do_codigo(RAIZ)
    try:
        mudancas = codigo_mudado_desde_o_paper()
        aviso = "; ".join(mudancas) if mudancas else ""
    except (OSError, subprocess.CalledProcessError, SyntaxError, ValueError) as erro:
        aviso = f"guarda do {TAG_DO_PAPER} indisponível: {str(erro)[:200]}"
    if "--leitura" in sys.argv[1:]:
        registro = ler_registro(ARQUIVO_REGISTRO)
        atualizar_universo()        # fora do dia devido o universe_prices para no último rebalanceio
        dados = ler_dados()
        dividendos = pd.read_parquet(PASTA_DO_LAGO / "dividends.parquet",
                                     columns=["security", "ex_date", "corporate_action", "yield"])
        pregoes = int((dados.precos.index > registro["data"].min()).sum()) if not registro.empty else 0
        if pregoes == 0:
            print("Paper-v2: nenhum pregão depois do 1º registro (ou nenhum registro)")
            return 0
        tabela = leitura(registro, dados, dividendos, liquidez_do_lago())
        print(f"Paper-v2 de {registro['data'].min():%d/%m/%Y} a {dados.precos.index[-1]:%d/%m/%Y}: {pregoes} pregões, "
              f"{registro['data'].nunique()} registros; guarda: {aviso or 'sem mudança'}")
        print(_tabela(tabela.assign(**{coluna: tabela[coluna].round().astype("int64") for coluna in EM_REAIS}),
                      EM_PORCENTO))
        return 0
    rodada = executar(ARQUIVO_CALENDARIO, ARQUIVO_REGISTRO, TRAVA_DO_LAGO, atualizar_universo, ler_dados,
                      lambda feita, dados: guardar_no_git(feita, dados, versao))
    _anotar(rodada, versao, aviso)
    print("lago rodando: sem decidir" if rodada is None else
          f"{rodada.data:%d/%m/%Y}: {rodada.desfecho} {rodada.nota}".strip())
    if rodada is not None and rodada.carteira is not None:
        print(rodada.carteira.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
