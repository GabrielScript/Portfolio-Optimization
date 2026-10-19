"""
Robustez fora da amostra (PLAN §11, passo 3): janelas móveis, subperíodos, P[> BOVA11] em log-retorno, MinTRL, DSR
contra o BOVA11 e a janela de estimação de 10 anos (R10). Descritivo: não muda o paper trading (paper-v1) nem o
critério de pronto. Rede só para o CDI.

Uso: .venv/Scripts/python -X utf8 scripts/avaliar_robustez.py > docs/robustez-2026-10-19.txt

Pré-registro (2026-10-19, escrito e commitado antes de rodar; depois desse commit o texto não muda):
- Dados: `docs/dados/2026-10-19/universe_prices.parquet` (o de 18/10 mais o IVVB11 diário) e, de
  `docs/dados/2026-10-18/`, `universe_monthly.parquet` e `sectors.parquet`. O script confere os três pelo
  `SHA256SUMS` de cada pasta e para se algum não bater.
- Motor e regras do `scripts/avaliar_regras_no_motor.py`, sem mudança: universo ponto no tempo, treino de 1.260
  pregões (R5), rebalanceio a cada 63, 10 bps por lado, Jorion, taxa livre de risco = CDI na data (Selic constante
  se o BCB falhar, com aviso), peso máximo 20%, até 10 ativos, teto de vol 40%, teto de 30% por setor no otimizador
  (setor do lago), `checar_carteira` recusando, perda diária de 5% no satélite congelando o pregão seguinte.
- Diferenças conhecidas para o satélite do paper (`scripts/paper_trading.satelite_do_dia`), escritas como limitação:
  o paper usa a lista líquida atual do app (com viés de sobrevivência), o setor do `assets.py` e a Selic constante
  como taxa livre de risco; aqui o universo é ponto no tempo, o setor é o do lago e a taxa é o CDI na data.
- Fases: o rebalanceio cai em pregões diferentes conforme o começo. O teste roda 3 fases, cortando 0, 21 e 42
  pregões do começo do arquivo de preços; a fase 0 é o arquivo inteiro. Tudo é reportado nas 3; nenhuma escolhe nada.
- Curvas (lista fechada; as 3 entram no DSR):
  1. "Kelly λ=2 com regras": o satélite com as regras 1 e 2, sem o stop (mede o otimizador);
  2. "Dinheiro real": BOVA11 70% + a curva 1 com o stop relativo 30%, de volta aos 70/30 a cada 63 pregões, com
     10 bps sobre o que muda e a volta congelada pela perda diária do satélite (`misturar_congelando`). O stop vale
     até o fim do horizonte: em cada janela, subperíodo ou período, ele é armado de novo no começo (o pico da razão
     satélite/BOVA11 começa ali), como para quem começou a investir ali;
  3. "Núcleo Mín. Vol 70/30 com regras": o padrão do estudo na tela, sem o stop (como no `avaliar_regras_no_motor`).
  Referências fora do N: BOVA11 e IVVB11 (desde 29/04/2014: janela ou subperíodo que começa antes fica sem ele).
- Período fora da amostra: da 1ª decisão de cada fase (~jan/2010) a 16/10/2026.
- Janelas móveis (`core.robustez.janelas_moveis`): 36 e 60 meses, uma começando em cada fim de mês. Por curva,
  comprimento e fase: nº de janelas, fração com CAGR acima do BOVA11 (riqueza), mediana e percentil 10 do excesso de
  CAGR, fração com P[> BOVA11] ≥ 0,95 em média aritmética e em log-retorno, pior MDD. Janelas sobrepostas não são
  amostras independentes: os números descrevem, não testam.
- Subperíodos fixados aqui: 2010–2014, 2015–2019 e 2020 a 16/10/2026 (curva recortada do último pregão antes do
  subperíodo ao último dentro dele): CAGR, excesso de CAGR sobre o BOVA11, P aritmético e em log, MDD.
- Período inteiro: Sharpe (excesso sobre o CDI), IR contra o BOVA11 (diferença diária, anualizada por √252),
  P[> BOVA11] aritmético e em log, DSR do IR (SR0 de `sharpe_referencia_deflacionado` com N = linhas de
  `docs/tentativas.csv` e V = variância dos IR diários das 3 curvas na mesma fase) e MinTRL
  (`anos_minimos_para_psr`: anos para o P aritmético chegar a 0,95 se o IR observado for o verdadeiro). O critério
  de pronto continua o do ROTEIRO §2 (PSR aritmético ≥ 0,95 contra o BOVA11); o P em log fica ao lado, não no lugar.
- R10 (DECISOES 61b): walk-forward com treino de 2.520 pregões; a R5 é a mesma estratégia estimada só nos últimos
  1.260 (`com_janela`), então as duas decidem nas mesmas datas e no mesmo universo (papéis com 10 anos de histórico
  na data: esta R5 não é a do paper nem a da parte de cima). Curva comparada: a 1 (o satélite sem stop; o P da
  diferença não muda com a fração fixa da mistura). Horizontes: os últimos 5 e 10 anos até 16/10/2026, curva
  recortada, nas 3 fases. A R10 passa se, nos dois horizontes e nas 3 fases: C1 P[R10 > R5] ≥ 0,95 (PSR aritmético
  da diferença diária); C2 MDD da R10 no máximo 5 p.p. pior que o da R5; C3 giro anual da R10 ≤ 2 × o da R5. Passar
  não muda o paper-v1: vira candidata a um pré-registro novo (paper-v2), registrada no DECISOES.
- Expectativa registrada: em janelas de 60 meses o satélite fica acima do BOVA11 em 40–65% delas, com percentil 10
  do excesso de CAGR abaixo de −8 p.p. ao ano e P em log ≥ 0,95 em menos de 15% delas; o "Dinheiro real" fica a
  poucos pontos do BOVA11 em quase todas; o sinal do excesso muda entre os subperíodos; MinTRL acima de 10 anos e DSR
  abaixo de 0,95 nas 3 curvas; R10 não passa (P[R10 > R5] entre 0,3 e 0,8).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import (Congelamento, ResultadoBacktest, com_janela, estrategia_buy_and_hold_ativo,  # noqa: E402
                           estrategia_otimizada_nas_regras, retornos_da_carteira, walk_forward,
                           walk_forward_dinamico)
from core.config import (DIAS_UTEIS_ANO, ConfigBacktest, ConfigOtimizacao, Objetivo,  # noqa: E402
                         taxa_anual_para_aritmetica)
from core.lago import ler_setores  # noqa: E402
from core.metrics import (drawdown, momentos_sharpe, retornos_de_valores, sharpe, sharpe_probabilistico,  # noqa: E402
                          sharpe_referencia_deflacionado)
from core.robustez import (COLUNAS_JANELAS, anos_minimos_para_psr, janelas_moveis, medir,  # noqa: E402
                           pares_de_janelas, psr_da_diferenca, resumir_janelas)
from scripts.avaliar_regras_no_motor import (MISTURA, NUCLEO, PESO_NUCLEO, SATELITE, data_do_stop,  # noqa: E402
                                             misturar_congelando, satelite_com_stop, setores_dos_papeis)
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT, CFG_KELLY, _porcento  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
PASTA_PRECOS = RAIZ / "docs" / "dados" / "2026-10-19"
PASTA_UNIVERSO = RAIZ / "docs" / "dados" / "2026-10-18"
ARQUIVO_TENTATIVAS = RAIZ / "docs" / "tentativas.csv"
REAL = "Dinheiro real"
REFERENCIAS = ("BOVA11", "IVVB11")         # fora do N do DSR
P_MINIMO_R10 = 0.95                         # C1: P[R10 > R5]
PIORA_MAXIMA_MDD_R10 = 0.05                 # C2: MDD da R10 no máximo 5 p.p. pior
GIRO_MAXIMO_R10 = 2.0                       # C3: giro anual da R10 <= 2x o da R5
_TOLERANCIA = 1e-9                          # "pelo menos" e "no máximo" incluem a fronteira
FASES = (0, 21, 42)                         # pregões cortados do começo do arquivo
MESES_DAS_JANELAS = (36, 60)
SUBPERIODOS = {"2010–2014": ("2010-01-01", "2014-12-31"), "2015–2019": ("2015-01-01", "2019-12-31"),
               "2020–2026": ("2020-01-01", "2026-10-16")}
JANELA_R10, JANELA_R5 = 2_520, 1_260
HORIZONTES_R10 = (5, 10)
FORA_DO_UNIVERSO = ("BOVA11", "SMAL11", "DIVO11", "IVVB11")     # ETFs: só referência
PESO_NUCLEO_REAL = 0.70


def conferir_hashes(pasta: Path, nomes: Sequence[str]) -> dict[str, str]:
    """sha256 de cada arquivo contra o ``SHA256SUMS`` da pasta; para o script se algum não bater."""
    esperados = {}
    for linha in (pasta / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        valor, nome = linha.split(maxsplit=1)
        esperados[nome.lstrip("*")] = valor
    obtidos = {nome: hashlib.sha256((pasta / nome).read_bytes()).hexdigest() for nome in nomes}
    errados = [nome for nome in nomes if esperados.get(nome) != obtidos[nome]]
    if errados:
        raise SystemExit(f"Hash diferente do pré-registro em {pasta}: {', '.join(errados)}")
    return obtidos


def contar_tentativas(arquivo: Path = ARQUIVO_TENTATIVAS) -> int:
    """N do DSR: as configurações distintas já avaliadas fora da amostra (a regra está no cabeçalho do arquivo)."""
    return len(pd.read_csv(arquivo, comment="#"))


def recortar(resultado: ResultadoBacktest, inicio: pd.Timestamp, fim: pd.Timestamp) -> ResultadoBacktest:
    """
    A curva de quem começou no fechamento de ``inicio`` e parou no de ``fim``: os rebalanceios do trecho são os
    depois do ``inicio`` (o do próprio dia já vale no fechamento dele).
    """
    dentro = (resultado.pesos_alvo.index > inicio) & (resultado.pesos_alvo.index <= fim)
    return replace(resultado, valores=resultado.valores.loc[inicio:fim], pesos_alvo=resultado.pesos_alvo[dentro],
                   turnover=resultado.turnover[(resultado.turnover.index > inicio)
                                               & (resultado.turnover.index <= fim)])


def dinheiro_real(satelite: ResultadoBacktest, bova: ResultadoBacktest, inicio: pd.Timestamp, fim: pd.Timestamp,
                  cfg_bt: ConfigBacktest) -> pd.Series:
    """
    BOVA11 70% + satélite 30% para quem começou em ``inicio``: o stop relativo vale até o fim do horizonte, então é
    armado de novo no começo (o pico da razão satélite/BOVA11 começa ali), e a volta aos 70/30 congela pela perda
    diária do satélite.
    """
    efetivo, nucleo = recortar(satelite, inicio, fim), recortar(bova, inicio, fim)
    stop = data_do_stop(efetivo, nucleo)
    if stop is not None:
        efetivo = satelite_com_stop(efetivo, nucleo, stop, cfg_bt.custo_bps)
    congelamento = Congelamento(retornos=retornos_de_valores(efetivo.valores))
    return misturar_congelando(nucleo, efetivo, PESO_NUCLEO_REAL, cfg_bt, REAL, congelamento).valores


def referencias(precos: pd.DataFrame, cfg_bt: ConfigBacktest) -> dict[str, ResultadoBacktest]:
    """
    BOVA11 e IVVB11 comprados e mantidos nas datas do motor. Antes da 1ª cotação o retorno do papel é 0 (o
    ``retornos_da_carteira``): a curva fica vazia ali, e a janela que começa antes fica sem a referência.
    """
    retornos = retornos_da_carteira(precos[list(REFERENCIAS)])
    curvas = {}
    for ref in REFERENCIAS:
        curva = walk_forward(retornos, estrategia_buy_and_hold_ativo(ref), cfg_bt, ref)
        existia = curva.valores.index >= precos[ref].first_valid_index()
        curvas[ref] = replace(curva, valores=curva.valores.where(existia))
    return curvas


def tabela_do_periodo(curvas: Mapping[str, pd.Series], bova: pd.Series, rf_diaria: pd.Series,
                      n_tentativas: int) -> pd.DataFrame:
    """
    Uma linha por curva no mesmo trecho que ``bova``: Sharpe sobre o CDI, IR contra o BOVA11 (diferença diária,
    ×√252), P[> BOVA11] aritmético e em log, DSR do IR (N = ``n_tentativas``, V = variância dos IR diários das
    curvas) e MinTRL em anos.
    """
    momentos = {nome: momentos_sharpe(retornos_de_valores(valores) - retornos_de_valores(bova))
                for nome, valores in curvas.items()}
    sr0 = sharpe_referencia_deflacionado(float(np.var([m.sharpe for m in momentos.values()], ddof=1)), n_tentativas)
    linhas = {}
    for nome, valores in curvas.items():
        retornos = retornos_de_valores(valores)
        linhas[nome] = {
            "Sharpe": sharpe(retornos - rf_diaria.reindex(retornos.index).fillna(0.0)),
            "IR": momentos[nome].sharpe * np.sqrt(DIAS_UTEIS_ANO),
            "P aritmético": psr_da_diferenca(valores, bova),
            "P log": psr_da_diferenca(valores, bova, log=True),
            "DSR": sharpe_probabilistico(momentos[nome], sr0),
            "MinTRL (anos)": anos_minimos_para_psr(momentos[nome]),
        }
    return pd.DataFrame(linhas).T


def r10_passa(comparacoes: pd.DataFrame) -> bool:
    """C1, C2 e C3 em todas as linhas (uma por fase e horizonte)."""
    c1 = comparacoes["P[R10 > R5]"] >= P_MINIMO_R10 - _TOLERANCIA
    c2 = comparacoes["MDD R10"] >= comparacoes["MDD R5"] - PIORA_MAXIMA_MDD_R10 - _TOLERANCIA
    c3 = comparacoes["Giro R10"] <= GIRO_MAXIMO_R10 * comparacoes["Giro R5"] + _TOLERANCIA
    return bool((c1 & c2 & c3).all())


def trecho(datas: pd.DatetimeIndex, de: str | pd.Timestamp,
           ate: str | pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(fechamento anterior a ``de``, último pregão até ``ate``); a curva que começa depois começa onde começa, e a
    que começa depois de ``ate`` não tem o trecho (None)."""
    if datas[0] >= pd.Timestamp(ate):
        return None
    antes = datas[datas < pd.Timestamp(de)]
    return (antes[-1] if len(antes) else datas[0]), datas[datas <= pd.Timestamp(ate)][-1]


def comparar_r10(r5: ResultadoBacktest, r10: ResultadoBacktest, anos: int) -> dict[str, float]:
    """C1–C3 nos últimos ``anos`` anos: P[R10 > R5] aritmético, MDD e giro anual das duas."""
    fim = r5.valores.index[-1]
    inicio, fim = trecho(r5.valores.index, fim - pd.DateOffset(years=anos) + pd.Timedelta(days=1), fim)
    so_r5, so_r10 = recortar(r5, inicio, fim), recortar(r10, inicio, fim)
    return {"P[R10 > R5]": psr_da_diferenca(so_r10.valores, so_r5.valores),
            "MDD R10": drawdown(so_r10.valores)[1], "MDD R5": drawdown(so_r5.valores)[1],
            "Giro R10": so_r10.turnover_anual_medio, "Giro R5": so_r5.turnover_anual_medio}


# ---------------------------------------------------------------- rodada (com o CDI da rede)

@dataclass(frozen=True)
class Dados:
    precos: pd.DataFrame
    universo: pd.DataFrame
    setores: dict[str, str]
    sha256: dict[str, str]
    manifest: dict[str, str]


def carregar() -> Dados:
    """Os arquivos do pré-registro, conferidos pelo hash (para o script se algum mudou)."""
    sha256 = {**conferir_hashes(PASTA_PRECOS, ["universe_prices.parquet"]),
              **conferir_hashes(PASTA_UNIVERSO, ["universe_monthly.parquet", "sectors.parquet"])}
    tabela = pd.read_parquet(PASTA_PRECOS / "universe_prices.parquet")
    precos = tabela.set_index(pd.DatetimeIndex(pd.to_datetime(tabela.pop("date")), name=None)).astype("float64")
    emissores = pd.read_parquet(PASTA_UNIVERSO / "universe_monthly.parquet", columns=["month", "security", "issuer"])
    emissores["month"] = pd.to_datetime(emissores["month"])
    setores = setores_dos_papeis(emissores, ler_setores(PASTA_UNIVERSO / "sectors.parquet"))
    # o lago grava o manifest em cp1252; só as datas, que são ASCII, interessam
    manifest = json.loads((PASTA_PRECOS / "manifest.json").read_bytes().decode("utf-8", errors="replace"))
    return Dados(precos, emissores[["month", "security"]], setores, sha256,
                 {campo: manifest[campo] for campo in ("generated_at", "data_from", "data_through")})


def _acoes(precos: pd.DataFrame) -> pd.DataFrame:
    return precos.drop(columns=[ticker for ticker in FORA_DO_UNIVERSO if ticker in precos.columns])


def curvas_da_fase(dados: Dados, fase: int, cfg: ConfigOtimizacao, cfg_bt: ConfigBacktest,
                   cdi: pd.Series | None) -> dict[str, ResultadoBacktest]:
    """Satélite com regras (sem stop), núcleo Mín. Vol, a mistura do estudo e as referências, nas datas da fase."""
    precos = dados.precos.iloc[fase:]
    kelly = estrategia_otimizada_nas_regras(replace(cfg, aversao_risco=2.0), True, dados.setores, cdi)
    satelite = walk_forward_dinamico(_acoes(precos), dados.universo, kelly, cfg_bt, SATELITE, Congelamento())
    congelamento = Congelamento(retornos=retornos_de_valores(satelite.valores))
    minima = estrategia_otimizada_nas_regras(replace(cfg, objetivo=Objetivo.MIN_VOL), True, dados.setores, cdi)
    nucleo = walk_forward_dinamico(_acoes(precos), dados.universo, minima, cfg_bt, NUCLEO, congelamento)
    curvas = {SATELITE: satelite, NUCLEO: nucleo,
              MISTURA: misturar_congelando(nucleo, satelite, PESO_NUCLEO, cfg_bt, MISTURA, congelamento),
              **referencias(precos, cfg_bt)}
    if any(not curva.valores.index.equals(satelite.valores.index) for curva in curvas.values()):
        raise SystemExit(f"Fase {fase}: curvas com datas diferentes")
    return curvas


def janelas_da_fase(curvas: Mapping[str, ResultadoBacktest], cfg_bt: ConfigBacktest) -> pd.DataFrame:
    """As janelas de 36 e 60 meses: curvas fixas pelo ``janelas_moveis``; o dinheiro real montado em cada janela."""
    bova = curvas["BOVA11"]
    tabela = pd.DataFrame({nome: curvas[nome].valores for nome in (SATELITE, MISTURA, "IVVB11", "BOVA11")})
    partes = []
    for meses in MESES_DAS_JANELAS:
        reais = [[inicio, fim, REAL, *medir(dinheiro_real(curvas[SATELITE], bova, inicio, fim, cfg_bt),
                                             bova.valores.loc[inicio:fim])]
                 for inicio, fim in pares_de_janelas(tabela.index, meses)]
        partes += [janelas_moveis(tabela, "BOVA11", meses).assign(meses=meses),
                   pd.DataFrame(reais, columns=COLUNAS_JANELAS).assign(meses=meses)]
    return pd.concat(partes, ignore_index=True)


def subperiodos_da_fase(curvas: Mapping[str, ResultadoBacktest], cfg_bt: ConfigBacktest) -> pd.DataFrame:
    """CAGR, excesso, P e MDD em cada subperíodo; referência sem histórico no começo fica fora."""
    bova = curvas["BOVA11"]
    linhas = []
    for rotulo, (de, ate) in SUBPERIODOS.items():
        datas = trecho(bova.valores.index, de, ate)
        if datas is None:
            continue
        inicio, fim = datas
        trechos = {nome: curvas[nome].valores.loc[inicio:fim] for nome in (SATELITE, MISTURA, "IVVB11")}
        trechos[REAL] = dinheiro_real(curvas[SATELITE], bova, inicio, fim, cfg_bt)
        linhas += [[rotulo, nome, *medir(valores, bova.valores.loc[inicio:fim])]
                   for nome, valores in trechos.items() if valores.notna().all()]
    return pd.DataFrame(linhas, columns=["subperiodo", "curva", *COLUNAS_JANELAS[3:]])


def periodo_da_fase(curvas: Mapping[str, ResultadoBacktest], cfg_bt: ConfigBacktest, rf_diaria: pd.Series,
                    n_tentativas: int) -> pd.DataFrame:
    """As 3 curvas da lista fechada no período inteiro da fase (o stop do dinheiro real armado na 1ª decisão)."""
    bova = curvas["BOVA11"]
    inicio, fim = bova.valores.index[0], bova.valores.index[-1]
    lista = {SATELITE: curvas[SATELITE].valores, REAL: dinheiro_real(curvas[SATELITE], bova, inicio, fim, cfg_bt),
             MISTURA: curvas[MISTURA].valores}
    return tabela_do_periodo(lista, bova.valores, rf_diaria, n_tentativas)


def r10_da_fase(dados: Dados, fase: int, cfg: ConfigOtimizacao, cfg_bt: ConfigBacktest,
                cdi: pd.Series | None) -> list[dict[str, float]]:
    """R10 e a R5 recortada (``com_janela``) no mesmo walk-forward de 2.520 pregões: mesmas datas, mesmo universo."""
    acoes = _acoes(dados.precos.iloc[fase:])
    kelly = estrategia_otimizada_nas_regras(replace(cfg, aversao_risco=2.0), True, dados.setores, cdi)
    cfg_r10 = replace(cfg_bt, janela_treino=JANELA_R10)
    r10 = walk_forward_dinamico(acoes, dados.universo, kelly, cfg_r10, "R10", Congelamento())
    r5 = walk_forward_dinamico(acoes, dados.universo, com_janela(kelly, JANELA_R5), cfg_r10, "R5", Congelamento())
    return [{"fase": fase, "anos": anos, **comparar_r10(r5, r10, anos)} for anos in HORIZONTES_R10]


def _tabela(tabela: pd.DataFrame, porcento: Sequence[str]) -> str:
    formatada = tabela.copy().astype(object)
    for coluna in formatada.columns:
        if coluna in porcento:
            formatada[coluna] = tabela[coluna].map(lambda v: "—" if pd.isna(v) else _porcento(float(v)))
        elif pd.api.types.is_float_dtype(tabela[coluna]):
            formatada[coluna] = tabela[coluna].map(lambda v: "—" if pd.isna(v) else f"{float(v):.3f}")
    return formatada.to_string()


PORCENTO_JANELAS = ("acima_do_bova", "excesso_mediano", "excesso_p10", "p_aritmetica_095", "p_log_095", "pior_mdd")


def imprimir_fase(fase: int, curvas: Mapping[str, ResultadoBacktest], rf: pd.Series, n_tentativas: int) -> None:
    satelite = curvas[SATELITE]
    print(f"\n=== Fase {fase}: 1ª decisão {satelite.pesos_alvo.index[0]:%d/%m/%Y}, fim "
          f"{satelite.valores.index[-1]:%d/%m/%Y}, {len(satelite.pesos_alvo)} rebalanceios; recusas "
          f"{len(satelite.recusas)} (satélite) e {len(curvas[NUCLEO].recusas)} (núcleo); falhas "
          f"{len(satelite.falhas) + len(curvas[NUCLEO].falhas)} ===")
    print("\nPeríodo inteiro (contra o BOVA11):")
    print(_tabela(periodo_da_fase(curvas, CFG_BT, rf, n_tentativas), []))
    janelas = janelas_da_fase(curvas, CFG_BT)
    for meses in MESES_DAS_JANELAS:
        print(f"\nJanelas móveis de {meses} meses:")
        print(_tabela(resumir_janelas(janelas[janelas["meses"] == meses]), PORCENTO_JANELAS))
    print("\nSubperíodos:")
    print(_tabela(subperiodos_da_fase(curvas, CFG_BT).set_index(["subperiodo", "curva"]),
                  ["cagr", "excesso_cagr", "mdd"]))


def main() -> int:
    inicio_rodada = time.perf_counter()
    hoje = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    dados = carregar()
    selic = market_data.obter_selic_meta(hoje)
    try:
        cdi = market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=hoje)
    except market_data.ErroDadosMercado as erro:
        print(f"CDI: {erro} (Selic constante)")
        cdi = pd.Series(dtype=float)
    cdi_vigente = cdi if not cdi.empty else None
    cfg = replace(CFG_KELLY, taxa_livre_risco=taxa_anual_para_aritmetica(selic.valor))
    n_tentativas = contar_tentativas()
    print(f"Arquivos (sha256): {dados.sha256}")
    print(f"Manifest: {dados.manifest}; Selic meta {_porcento(selic.valor)} ({selic.fonte}); "
          f"CDI {'do BCB' if cdi_vigente is not None else 'indisponível'}; N do DSR = {n_tentativas}")
    comparacoes = []
    for fase in FASES:
        curvas = curvas_da_fase(dados, fase, cfg, CFG_BT, cdi_vigente)
        rf = market_data.serie_rf_diaria(curvas[SATELITE].valores.index, cdi_vigente, selic.valor)
        imprimir_fase(fase, curvas, rf, n_tentativas)
        comparacoes += r10_da_fase(dados, fase, cfg, CFG_BT, cdi_vigente)
        print(f"(fase {fase} pronta em {time.perf_counter() - inicio_rodada:.0f} s)", flush=True)
    tabela_r10 = pd.DataFrame(comparacoes).set_index(["fase", "anos"])
    print("\nR10 × R5 (satélite com regras, mesmo universo: papéis com 10 anos de histórico):")
    print(_tabela(tabela_r10, ["MDD R10", "MDD R5", "Giro R10", "Giro R5"]))
    print(f"R10 passa (C1–C3 nas 3 fases e nos 2 horizontes): {'sim' if r10_passa(tabela_r10) else 'não'}")
    print(f"(tempo total {time.perf_counter() - inicio_rodada:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
