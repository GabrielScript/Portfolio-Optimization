"""
S3: Black-Litterman com views dos sinais aprovados no S2 (PLAN §5, S3): o Kelly λ=2 com as regras, com o μ do treino
trocado pela média posterior de um prior de equilíbrio e das views da Conservadora e do valor (Fórmula Mágica +
Trending Value). Walk-forward contra o BOVA11, o Kelly atual e a Conservadora. Não muda o paper trading (paper-v1) nem
o critério de pronto. Rede só para o CDI e a Selic meta.

Uso: .venv/Scripts/python -X utf8 scripts/avaliar_views.py > docs/views-2026-10-20.txt

Pré-registro (2026-10-20, escrito e commitado antes de rodar; depois desse commit o texto não muda):
- Dados: os do `scripts/avaliar_valor.py` (os mesmos 7 arquivos, conferidos pelo hash), com as mesmas datas: 3 fases
  com a 1ª decisão em 02/01, 01/02 e 05/03/2012, até 16/10/2026.
- Prior (`core.views.retorno_equilibrio`): Π = δ Σ w, com δ = 2 (a aversão do Kelly: sem views, o Kelly sem
  restrições volta ao portfólio de mercado), Σ = Ledoit-Wolf do treino de 1.260 pregões (a do Kelly atual) e w = pesos
  de mercado nos papéis elegíveis com setor, pelo `market_caps.parquet` do mês da decisão (o universo do lago tem uma
  classe por emissor; papel sem valor de mercado, ~15% em 2012 e ~1% depois de 2018, fica com peso 0 e o Π dele sai
  só das covariâncias). μ em excesso: com os pesos somando 1, o Kelly não muda quando todos os μ sobem o mesmo rf.
- Views (`core.views.view_do_sinal`, P = I): Q = Π + σ z, com z = escore normal do posto da nota do mês
  (Φ⁻¹((posto - 0,5)/n)) e σ = vol anual do papel pela Σ; papel sem nota tem z = 0. Ω = τ σ² (1 - IC)/IC: o peso
  de cada view é o IC e, com Σ diagonal, posterior - Π = IC σ z (o alfa de Grinold, sem contrair duas vezes); τ =
  0,05 se cancela (Ω é proporcional a τ). Duas views: a Conservadora (`core.sinais.conservadora`) e o valor, a média
  dos escores normais da Fórmula Mágica e do Trending Value (`core.views.nota_media_dos_escores`; duas views contariam
  duas vezes o mesmo alfa, os dois são de valor). IC de cada view (`core.views.ic_ponto_no_tempo`): a média do IC de
  Spearman mensal (nota do mês x retorno do mês, universo do mês) de todos os meses antes da decisão, com pelo menos 36
  meses e média positiva; sem isso a view fica fora naquele mês. A Conservadora tem IC desde 1998 e entra desde a 1ª
  decisão; o valor, com fundamentos desde 2011, entra quando completar 36 meses de IC (03/2014, contado só em formas).
  Limitação escrita aqui: o IC é de 1 mês e pesa a vol anual, com rebalanceio a cada 63 pregões (o alfa de um mês
  decai até o rebalanceio; o desenho fica conservador no tamanho da view e otimista na persistência).
- Carteira (`core.views.estrategia_com_views`): o otimizador do Kelly atual sem mudança (`CFG_KELLY`: Kelly λ=2,
  peso máximo 20%, no máximo 10 papéis, vol até 40%, teto de 30% por setor), conferida por `checar_carteira` com a
  Σ do treino (violação = recusa: o motor mantém a carteira atual e conta). Com 10 papéis no máximo, a cardinalidade
  pesa mais que as views nos pesos pequenos.
- Walk-forward: o motor do `scripts/avaliar_robustez.py`, sem mudança, nos preços cortados do `avaliar_valor.py`.
  Curvas nas mesmas datas: Kelly com views, o "Kelly λ=2 com regras" (o satélite), a Conservadora (a candidata do
  S2.1; a versão sem alvo de vol, que ficou no teste do EGARCH) e as referências BOVA11 e IVVB11. Por fase: Sharpe
  sobre o CDI, IR contra o BOVA11, P[> BOVA11] aritmético e em log, DSR do IR (N = linhas de `docs/tentativas.csv`,
  já com esta: 30; V = variância dos IR diários das três estratégias) e MinTRL; P[> Kelly] e P[> Conservadora] do
  Kelly com views; CAGR, MDD, giro anual, recusas e falhas.
- Decisão fixa agora: candidata à prova final = P[> BOVA11] aritmético >= 0,95 nas 3 fases (a régua do S2). Não
  candidata = o S3 fica fora da prova final; não volta com outro δ, τ, mínimo de meses, combinação de views ou
  otimizador sem pré-registro novo, contado no N.
- Expectativa registrada: CAGR entre o do Kelly e o da Conservadora, MDD perto do Kelly (−55 a −62%), P[> BOVA11]
  entre 0,80 e 0,97 (abaixo de 0,95 em pelo menos uma fase), P[> Kelly] entre 0,6 e 0,9, P[> Conservadora] abaixo de
  0,5, DSR abaixo de 0,95, giro de 200% a 300% e nenhuma recusa.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, ResultadoBacktest, estrategia_otimizada_nas_regras, walk_forward_dinamico  # noqa: E402
from core.config import taxa_anual_para_aritmetica  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.poder import ic_por_mes  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import conservadora, estrategia_conservadora  # noqa: E402
from core.valor import Balancos, formula_magica_no_mes, painel_da_nota, trending_value  # noqa: E402
from core.views import Sinal, estrategia_com_views, ic_ponto_no_tempo, nota_media_dos_escores  # noqa: E402
from scripts import avaliar_valor as valor  # noqa: E402
from scripts.avaliar_conservadora import CONSERVADORA, Mensal, ics_da_conservadora, retornos_no_universo  # noqa: E402
from scripts.avaliar_regras_no_motor import SATELITE  # noqa: E402
from scripts.avaliar_robustez import (FASES, Dados, _acoes, _tabela, carregar, contar_tentativas,  # noqa: E402
                                      referencias, tabela_do_periodo)
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT, CFG_KELLY, _porcento  # noqa: E402

COM_VIEWS = "Kelly λ=2 com views (Black-Litterman) e regras"
ESTRATEGIAS = (COM_VIEWS, SATELITE, CONSERVADORA)
INICIO_DO_IC_DE_VALOR = pd.Timestamp("2010-01-01")     # antes dos 1ºs fundamentos: meses sem nota ficam sem IC
P_MINIMO_DA_PROVA = 0.95
_TOLERANCIA = 1e-9


def candidata(p_bova_por_fase: Sequence[float]) -> bool:
    """Candidata à prova final: P[> BOVA11] aritmético >= 0,95 nas 3 fases, a régua do S2."""
    return all(p >= P_MINIMO_DA_PROVA - _TOLERANCIA for p in p_bova_por_fase)


# ---------------------------------------------------------------- rodada (com o CDI da rede)

def sinais_das_views(balancos: Balancos, mensal: Mensal, universo: pd.DataFrame) -> dict[str, Sinal]:
    """As duas views com o IC mensal de cada uma: a Conservadora desde 1998, o valor desde os 1ºs fundamentos."""
    def nota_conservadora(mes, candidatos):
        return conservadora(mensal.retornos, mensal.proventos, mes, candidatos)

    nota_valor = nota_media_dos_escores(
        lambda mes, candidatos: formula_magica_no_mes(balancos, mes, candidatos),
        lambda mes, candidatos: trending_value(balancos, mensal.retornos, mensal.proventos, mes, candidatos))
    meses = pd.date_range(INICIO_DO_IC_DE_VALOR, valor.PERIODO_DO_IC[1], freq="MS")
    painel = painel_da_nota(nota_valor, universo, meses, mensal.retornos.columns)
    ics_valor = ic_por_mes(painel, retornos_no_universo(mensal.retornos, universo, meses))
    return {"Conservadora": Sinal(nota_conservadora, ics_da_conservadora(mensal, universo)),
            "Valor (Fórmula Mágica + Trending Value)": Sinal(nota_valor, ics_valor)}


def curvas_da_fase(dados: Dados, mensal: Mensal, balancos: Balancos, sinais: Sequence[Sinal], corte: int, fase: int,
                   cdi: pd.Series | None, cfg_kelly) -> dict[str, ResultadoBacktest]:
    """O Kelly com views, o Kelly com regras, a Conservadora e as referências nas mesmas datas da fase."""
    precos = dados.precos.iloc[corte + fase:]
    cfg = replace(cfg_kelly, aversao_risco=2.0)
    estrategias = {COM_VIEWS: estrategia_com_views(cfg, dados.setores, sinais, balancos.valor_mercado, rf_diaria=cdi),
                   SATELITE: estrategia_otimizada_nas_regras(cfg, True, dados.setores, cdi),
                   CONSERVADORA: estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores)}
    curvas = {nome: walk_forward_dinamico(_acoes(precos), dados.universo, estrategia, CFG_BT, nome, Congelamento())
              for nome, estrategia in estrategias.items()}
    curvas.update(referencias(precos, CFG_BT))
    if any(not curva.valores.index.equals(curvas[SATELITE].valores.index) for curva in curvas.values()):
        raise SystemExit(f"Fase {fase}: curvas com datas diferentes")
    return curvas


def medidas_da_fase(curvas: Mapping[str, ResultadoBacktest], rf: pd.Series, n_tentativas: int) -> pd.DataFrame:
    tabela = tabela_do_periodo({nome: curvas[nome].valores for nome in ESTRATEGIAS}, curvas["BOVA11"].valores, rf,
                               n_tentativas)
    tabela.loc[COM_VIEWS, "P[> Kelly]"] = psr_da_diferenca(curvas[COM_VIEWS].valores, curvas[SATELITE].valores)
    tabela.loc[COM_VIEWS, "P[> Conservadora]"] = psr_da_diferenca(curvas[COM_VIEWS].valores,
                                                                  curvas[CONSERVADORA].valores)
    linhas = {}
    for nome in (*ESTRATEGIAS, "IVVB11", "BOVA11"):
        valores = curvas[nome].valores.dropna()
        linhas[nome] = {"CAGR": cagr(valores), "MDD": drawdown(valores)[1],
                        "Giro anual": curvas[nome].turnover_anual_medio, "Recusas": len(curvas[nome].recusas),
                        "Falhas": len(curvas[nome].falhas)}
    return pd.DataFrame(linhas).T.astype({"Recusas": int, "Falhas": int}).join(tabela)


def imprimir_ics(sinais: Mapping[str, Sinal]) -> None:
    decisoes = pd.date_range(valor.PRIMEIRA_DECISAO, valor.PERIODO_DO_IC[1], freq="MS")
    for nome, sinal in sinais.items():
        usados = pd.Series({mes: ic_ponto_no_tempo(sinal.ics, mes) for mes in decisoes}, dtype=float)
        dentro = usados.dropna()
        primeira = f"{dentro.index[0]:%m/%Y}" if not dentro.empty else "nunca"
        print(f"View {nome}: entra em {primeira}; IC usado: "
              + ", ".join(f"{mes:%m/%Y} {usados.get(mes, float('nan')):.3f}"
                          for mes in pd.to_datetime(["2012-01-01", "2015-01-01", "2020-01-01", "2026-09-01"])))


def main() -> int:
    inicio_rodada = time.perf_counter()
    hoje = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    dados, mensal = carregar(), valor.carregar_mensal()
    balancos, sha256_cvm = valor.carregar_balancos()
    selic = market_data.obter_selic_meta(hoje)
    try:
        cdi = market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=hoje)
    except market_data.ErroDadosMercado as erro:
        print(f"CDI: {erro} (Selic constante)")
        cdi = pd.Series(dtype=float)
    cdi_vigente = cdi if not cdi.empty else None
    cfg = replace(CFG_KELLY, taxa_livre_risco=taxa_anual_para_aritmetica(selic.valor))
    n_tentativas = contar_tentativas()
    corte = valor.corte_dos_precos(dados.precos.index, valor.PRIMEIRA_DECISAO, CFG_BT.janela_treino)
    print(f"Arquivos (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm}")
    print(f"Manifest: {dados.manifest}; Selic meta {_porcento(selic.valor)} ({selic.fonte}); "
          f"CDI {'do BCB' if cdi_vigente is not None else 'indisponível'}; N do DSR = {n_tentativas}; "
          f"preços desde {dados.precos.index[corte]:%d/%m/%Y}")

    sinais = sinais_das_views(balancos, mensal, dados.universo)
    imprimir_ics(sinais)
    print(f"(ICs prontos em {time.perf_counter() - inicio_rodada:.0f} s)", flush=True)

    p_bova = []
    for fase in FASES:
        curvas = curvas_da_fase(dados, mensal, balancos, list(sinais.values()), corte, fase, cdi_vigente, cfg)
        rf = market_data.serie_rf_diaria(curvas[SATELITE].valores.index, cdi_vigente, selic.valor)
        tabela = medidas_da_fase(curvas, rf, n_tentativas)
        p_bova.append(float(tabela.loc[COM_VIEWS, "P aritmético"]))
        com_views = curvas[COM_VIEWS]
        print(f"\n=== Fase {fase}: 1ª decisão {com_views.pesos_alvo.index[0]:%d/%m/%Y}, fim "
              f"{com_views.valores.index[-1]:%d/%m/%Y}, {len(com_views.pesos_alvo)} rebalanceios; adiados "
              + ", ".join(f"{nome.split(' (')[0]} {len(curvas[nome].adiamentos)}" for nome in ESTRATEGIAS) + " ===")
        print(_tabela(tabela, ["CAGR", "MDD", "Giro anual"]))
        print(f"(fase {fase} pronta em {time.perf_counter() - inicio_rodada:.0f} s)", flush=True)

    print(f"\n{COM_VIEWS}: candidata à prova final (P[> BOVA11] >= 0,95 nas 3 fases): "
          f"{'sim' if candidata(p_bova) else 'não'}")
    print(f"(tempo total {time.perf_counter() - inicio_rodada:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
