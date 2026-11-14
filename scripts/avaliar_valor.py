"""
S2.2 e S2.3: a Fórmula Mágica de Greenblatt e o Trending Value de O'Shaughnessy no lago da B3, com os fundamentos da
CVM ponto no tempo (PLAN §5, S2, estratégias 2 e 3 da lista fechada). IC de cada sinal mês a mês e walk-forward das
carteiras contra o BOVA11, o Kelly e a Conservadora. Não muda o paper trading (paper-v1) nem o critério de pronto. Rede
só para o CDI e a Selic meta.

Uso: .venv/Scripts/python -X utf8 scripts/avaliar_valor.py > docs/valor-2026-10-20.txt

Pré-registro (2026-10-20, escrito e commitado antes de rodar; depois desse commit o texto não muda):
- Dados: de `docs/dados/2026-10-20-cvm/`, `fundamentals.parquet` (DFP e ITR desde 2010: contas fixas, consolidado
  primeiro, fluxos em 12 meses, cada valor com a data de recebimento do documento na CVM) e `market_caps.parquet`
  (valor de mercado de cada emissor na decisão de cada mês, com o fechamento do mês anterior); de
  `docs/dados/2026-10-20/`, `monthly_total_returns.parquet` e `dividends.parquet`; de `docs/dados/2026-10-19/`,
  `universe_prices.parquet`; de `docs/dados/2026-10-18/`, `universe_monthly.parquet` e `sectors.parquet`. O script
  confere os sete pelo `SHA256SUMS` de cada pasta e para se algum não bater.
- Fundamentos do mês m (`core.valor.fundamentos_no_mes`): para cada emissor e conta, o valor do fim de período mais
  recente entre os documentos recebidos antes do 1º dia de m, com esse fim de período a no máximo 18 meses de m (quem
  parou de entregar sai). Os arquivos da CVM só trazem a última versão de cada documento: um valor reapresentado vale
  da data da reapresentação, e antes dela o emissor usa o período anterior. Financeiro e emissor sem setor fora das
  duas (outro plano de contas, sem EBIT). As classes de um emissor têm a mesma nota de balanço; o IC conta cada papel
  do universo, e a carteira fica com uma classe por emissor (a primeira na ordem das colunas do motor, em empate).
- S2.2, Fórmula Mágica (`core.valor.formula_magica`): EV = valor de mercado + empréstimos de curto e longo prazo +
  participação dos minoritários (valor contábil) - caixa - aplicações financeiras (conta que falta vale 0); capital
  empregado = (ativo circulante - caixa - aplicações) - (passivo circulante - empréstimos de curto prazo) +
  imobilizado; nota = média dos postos percentuais de EBIT/EV e de EBIT/capital empregado, nos emissores com EBIT,
  ativo e passivo circulantes, EV > 0 e capital > 0. Utilidade Pública fica dentro (o PLAN só tira bancos e
  seguradoras; o site de Greenblatt tira as duas).
- S2.3, Trending Value (`core.valor.trending_value`): composto de valor = média dos postos percentuais, entre os
  papéis do universo do mês com momentum, de 6 rendimentos: lucro (o dos controladores, ou o lucro inteiro quando a
  conta não vem)/valor de mercado, patrimônio sem minoritários/valor de mercado, receita/valor de mercado, EBITDA/EV
  (EBIT + |depreciação e amortização da DVA|; vazio com EV <= 0), caixa operacional/valor de mercado e dividend yield
  de 12 meses do papel (o mesmo da Conservadora; sem recompras: a contagem de ações muda de fonte em 2020, FRE com
  tesouraria para composição sem, e a troca pareceria recompra). Rendimento que falta vale o posto 0,5; emissor sem
  lucro (só balanço patrimonial) fica fora. Momentum = retorno total de m-6 a m-1, os 6 presentes. Nota = 1 para os
  30% de maior composto (int(0,3 n), empate pela ordem) e 0 para os outros, mais o posto percentual do momentum
  dentro do grupo: as 15 maiores notas são as 15 baratas de maior momentum.
- Portão (o mesmo do S2.1): IC de Spearman de cada mês entre a nota e o retorno total do próprio mês, nos papéis do
  universo do mês com os dois. Aprovada = IC médio >= 0,02 e t de Student da média >= 2 de 2012-01 a 2026-09 (o
  primeiro ano inteiro com fluxos de 12 meses da CVM até o último mês fechado; 177 meses, menos que os 225 da análise
  de poder, `docs/poder-dos-portoes-2026-10-18.txt`: o portão fica mais difícil de passar). Sem a condição 2 do S2.1
  (IC > 0 numa década anterior que nenhum teste olhou): a CVM não tem DFP/ITR em dados abertos antes de 2010, então a
  aprovação aqui é mais fraca que a da Conservadora. Subperíodos 2012–2016, 2017–2021 e 2022–2026 só descritivos.
- Carteira (`core.valor.estrategia_da_nota`, igual à da Conservadora): as 15 maiores notas com pesos iguais, um papel
  por emissor, até 4 por setor do lago, papel sem setor fora, conferida por `checar_carteira` com a cov de Ledoit-Wolf
  do treino (violação = recusa: o motor mantém a carteira atual e conta). Na decisão de um pregão do mês m, a nota é a
  do mês m, a mesma do IC daquele mês. Limitações escritas aqui: o livro de Greenblatt compra 20 a 30 e segura um ano;
  O'Shaughnessy compra 25 do decil mais barato de um universo de milhares; aqui são 15 de ~100 a ~170 com nota,
  rebalanceio a cada 63 pregões.
- Walk-forward: o motor do `scripts/avaliar_robustez.py`, sem mudança, mas com o arquivo de preços começando no pregão
  que põe a 1ª decisão da fase 0 no 1º pregão de 2012 (`corte_dos_precos`: 23/11/2006; sem isso a 1ª decisão cairia em
  02/2010, sem fundamentos, e o motor para); fases 0, 21 e 42 cortadas a partir dali (1ª decisão em 02/01, 01/02 e
  05/03/2012). Treino de 1.260 pregões, rebalanceio a cada 63, 10 bps por lado, perda diária de 5% congelando o pregão
  seguinte. Curvas nas mesmas datas: Fórmula Mágica, Trending Value, a Conservadora (aprovada no S2.1, rodada de novo
  nestas datas), o "Kelly λ=2 com regras" e as referências BOVA11 e IVVB11. Por fase (2012 a 16/10/2026): Sharpe
  sobre o CDI, IR contra o BOVA11, P[> BOVA11] aritmético e em log, DSR do IR (N = linhas de `docs/tentativas.csv`,
  já com estas duas: 25; V = variância dos IR diários das quatro estratégias) e MinTRL; P[> Kelly] e
  P[> Conservadora] das duas novas; CAGR, MDD, giro anual, recusas e falhas.
- Decisão fixa agora, para cada uma: aprovada = sinal candidato a views do S3 e a entrar nas provas seguintes
  (EGARCH, prova final); candidata à prova final = aprovada e P[> BOVA11] aritmético >= 0,95 nas 3 fases. Reprovada =
  sai da lista; não volta com outro parâmetro sem pré-registro novo, contado no N.
- Expectativa registrada: IC médio de 2012 a 2026 entre 0,00 e 0,04 nas duas, com t entre 0 e 3 (valor funcionou na
  B3 em vários estudos, mas 2012–2020 foi ruim para valor no mundo); Fórmula Mágica com IC um pouco maior que o
  Trending Value (o momentum de 6 meses é fraco aqui); pelo menos uma das duas reprovada. No walk-forward, CAGR perto
  do BOVA11 ou acima, MDD parecido, P[> BOVA11] entre 0,5 e 0,95, P[> Conservadora] abaixo de 0,5, DSR abaixo de
  0,95, giro anual de 150% a 300% e nenhuma recusa.
"""

from __future__ import annotations

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
from core.backtest import Congelamento, ResultadoBacktest, estrategia_otimizada_nas_regras, walk_forward_dinamico  # noqa: E402
from core.config import taxa_anual_para_aritmetica  # noqa: E402
from core.lago import ler_setores  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.poder import Portao, ic_por_mes  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from core.valor import (Balancos, NotaDoMes, estrategia_da_nota, formula_magica_no_mes, painel_da_nota,  # noqa: E402
                        trending_value)
from scripts.avaliar_conservadora import (CONSERVADORA, Mensal, carregar_mensal, resumo_do_ic,  # noqa: E402
                                          retornos_no_universo)
from scripts.avaliar_regras_no_motor import SATELITE  # noqa: E402
from scripts.avaliar_robustez import (FASES, PASTA_UNIVERSO, Dados, _acoes, _tabela, carregar,  # noqa: E402
                                      conferir_hashes, contar_tentativas, referencias, tabela_do_periodo)
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT, CFG_KELLY, _porcento  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
PASTA_CVM = RAIZ / "docs" / "dados" / "2026-10-20-cvm"
FORMULA = "Fórmula Mágica (15 nomes, pesos iguais, com regras)"
TRENDING = "Trending Value (15 nomes, pesos iguais, com regras)"
NOVAS = (FORMULA, TRENDING)
ESTRATEGIAS = (FORMULA, TRENDING, CONSERVADORA, SATELITE)
PORTAO = Portao("IC ≥ 0,02 e t ≥ 2", ic_minimo=0.02, t_minimo=2.0)
PERIODO_DO_IC = (pd.Timestamp("2012-01-01"), pd.Timestamp("2026-09-01"))
SUBPERIODOS_DO_IC = {"2012–2016": ("2012-01-01", "2016-12-01"), "2017–2021": ("2017-01-01", "2021-12-01"),
                     "2022–2026": ("2022-01-01", "2026-09-01")}
PRIMEIRA_DECISAO = pd.Timestamp("2012-01-01")
P_MINIMO_DA_PROVA = 0.95
_TOLERANCIA = 1e-9


@dataclass(frozen=True)
class Decisao:
    aprovada: bool          # portão do IC de 2012-01 a 2026-09
    candidata: bool         # à prova final: aprovada e P[> BOVA11] >= 0,95 nas 3 fases


def decidir(ics: pd.Series, p_bova_por_fase: Sequence[float]) -> Decisao:
    """As regras fixas do pré-registro sobre o IC de cada mês e o P[> BOVA11] aritmético de cada fase."""
    aprovada = PORTAO.aprova(ics.loc[PERIODO_DO_IC[0]:PERIODO_DO_IC[1]].dropna())
    candidata = aprovada and all(p >= P_MINIMO_DA_PROVA - _TOLERANCIA for p in p_bova_por_fase)
    return Decisao(aprovada, candidata)


def corte_dos_precos(datas: pd.DatetimeIndex, primeira_decisao: pd.Timestamp, janela_treino: int) -> int:
    """Posição do 1º preço que põe a 1ª decisão do motor (pregão ``janela_treino`` dos retornos, que perdem o 1º
    preço) no 1º pregão desde ``primeira_decisao``; para se o arquivo não tem a janela inteira antes dela."""
    corte = int(datas.searchsorted(primeira_decisao)) - janela_treino - 1
    if corte < 0:
        raise SystemExit(f"Preços desde {datas[0]:%d/%m/%Y}: sem {janela_treino} pregões antes de "
                         f"{primeira_decisao:%d/%m/%Y}")
    return corte


# ---------------------------------------------------------------- rodada (com o CDI da rede)

def carregar_balancos(pasta: Path = PASTA_CVM) -> tuple[Balancos, dict[str, str]]:
    sha256 = conferir_hashes(pasta, ["fundamentals.parquet", "market_caps.parquet"])
    fundamentos = pd.read_parquet(pasta / "fundamentals.parquet")
    for coluna in ("dt_refer", "available_from"):
        fundamentos[coluna] = pd.to_datetime(fundamentos[coluna])
    valor_mercado = pd.read_parquet(pasta / "market_caps.parquet", columns=["issuer", "month", "market_cap"])
    valor_mercado["month"] = pd.to_datetime(valor_mercado["month"])
    setores = ler_setores(PASTA_UNIVERSO / "sectors.parquet").to_dict()
    return Balancos(fundamentos, valor_mercado, setores), sha256


def notas(balancos: Balancos, mensal: Mensal) -> dict[str, NotaDoMes]:
    return {FORMULA: lambda mes, candidatos: formula_magica_no_mes(balancos, mes, candidatos),
            TRENDING: lambda mes, candidatos: trending_value(balancos, mensal.retornos, mensal.proventos, mes,
                                                             candidatos)}


def ics_e_cobertura(nota_do_mes: NotaDoMes, mensal: Mensal, universo: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """IC de cada mês de 2012-01 a 2026-09 (a nota do mês contra o retorno do próprio mês, no universo dele) e o
    número de papéis com nota."""
    meses = pd.date_range(PERIODO_DO_IC[0], PERIODO_DO_IC[1], freq="MS")
    painel = painel_da_nota(nota_do_mes, universo, meses, mensal.retornos.columns)
    return ic_por_mes(painel, retornos_no_universo(mensal.retornos, universo, meses)), painel.notna().sum(axis=1)


def curvas_da_fase(dados: Dados, mensal: Mensal, notas_do_mes: Mapping[str, NotaDoMes], corte: int, fase: int,
                   cdi: pd.Series | None, cfg_kelly) -> dict[str, ResultadoBacktest]:
    """As duas novas, a Conservadora, o Kelly com regras e as referências nas mesmas datas da fase."""
    precos = dados.precos.iloc[corte + fase:]
    estrategias = {nome: estrategia_da_nota(nota, dados.setores) for nome, nota in notas_do_mes.items()}
    estrategias[CONSERVADORA] = estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores)
    estrategias[SATELITE] = estrategia_otimizada_nas_regras(replace(cfg_kelly, aversao_risco=2.0), True,
                                                            dados.setores, cdi)
    curvas = {nome: walk_forward_dinamico(_acoes(precos), dados.universo, estrategia, CFG_BT, nome, Congelamento())
              for nome, estrategia in estrategias.items()}
    curvas.update(referencias(precos, CFG_BT))
    if any(not curva.valores.index.equals(curvas[SATELITE].valores.index) for curva in curvas.values()):
        raise SystemExit(f"Fase {fase}: curvas com datas diferentes")
    return curvas


def medidas_da_fase(curvas: Mapping[str, ResultadoBacktest], rf: pd.Series, n_tentativas: int) -> pd.DataFrame:
    """CAGR, MDD, giro, recusas e falhas das seis curvas (a referência que começa depois, do 1º pregão dela), as
    medidas contra o BOVA11 das quatro estratégias e as das duas novas contra o Kelly e a Conservadora."""
    tabela = tabela_do_periodo({nome: curvas[nome].valores for nome in ESTRATEGIAS}, curvas["BOVA11"].valores, rf,
                               n_tentativas)
    for nome in NOVAS:
        tabela.loc[nome, "P[> Kelly]"] = psr_da_diferenca(curvas[nome].valores, curvas[SATELITE].valores)
        tabela.loc[nome, "P[> Conservadora]"] = psr_da_diferenca(curvas[nome].valores, curvas[CONSERVADORA].valores)
    linhas = {}
    for nome in (*ESTRATEGIAS, "IVVB11", "BOVA11"):
        valores = curvas[nome].valores.dropna()
        linhas[nome] = {"CAGR": cagr(valores), "MDD": drawdown(valores)[1],
                        "Giro anual": curvas[nome].turnover_anual_medio, "Recusas": len(curvas[nome].recusas),
                        "Falhas": len(curvas[nome].falhas)}
    return pd.DataFrame(linhas).T.astype({"Recusas": int, "Falhas": int}).join(tabela)


def main() -> int:
    inicio_rodada = time.perf_counter()
    hoje = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    dados, mensal = carregar(), carregar_mensal()
    balancos, sha256_cvm = carregar_balancos()
    selic = market_data.obter_selic_meta(hoje)
    try:
        cdi = market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=hoje)
    except market_data.ErroDadosMercado as erro:
        print(f"CDI: {erro} (Selic constante)")
        cdi = pd.Series(dtype=float)
    cdi_vigente = cdi if not cdi.empty else None
    cfg = replace(CFG_KELLY, taxa_livre_risco=taxa_anual_para_aritmetica(selic.valor))
    n_tentativas = contar_tentativas()
    corte = corte_dos_precos(dados.precos.index, PRIMEIRA_DECISAO, CFG_BT.janela_treino)
    print(f"Arquivos (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm}")
    print(f"Manifest: {dados.manifest}; Selic meta {_porcento(selic.valor)} ({selic.fonte}); "
          f"CDI {'do BCB' if cdi_vigente is not None else 'indisponível'}; N do DSR = {n_tentativas}; "
          f"preços desde {dados.precos.index[corte]:%d/%m/%Y}")

    notas_do_mes = notas(balancos, mensal)
    ics = {}
    periodos = {"2012–2026 (portão)": PERIODO_DO_IC,
                **{rotulo: tuple(map(pd.Timestamp, datas)) for rotulo, datas in SUBPERIODOS_DO_IC.items()}}
    for nome, nota_do_mes in notas_do_mes.items():
        ics[nome], cobertura = ics_e_cobertura(nota_do_mes, mensal, dados.universo)
        print(f"\n{nome}: IC de Spearman mensal (nota do mês x retorno total do mês, universo do mês); papéis com "
              f"nota por mês: mínimo {cobertura.min()}, mediana {cobertura.median():.0f}, máximo {cobertura.max()}")
        print(pd.DataFrame({rotulo: resumo_do_ic(ics[nome].loc[inicio:fim]) for rotulo, (inicio, fim)
                            in periodos.items()}).T.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"(fase de IC pronta em {time.perf_counter() - inicio_rodada:.0f} s)", flush=True)

    p_bova = {nome: [] for nome in NOVAS}
    for fase in FASES:
        curvas = curvas_da_fase(dados, mensal, notas_do_mes, corte, fase, cdi_vigente, cfg)
        rf = market_data.serie_rf_diaria(curvas[SATELITE].valores.index, cdi_vigente, selic.valor)
        tabela = medidas_da_fase(curvas, rf, n_tentativas)
        for nome in NOVAS:
            p_bova[nome].append(float(tabela.loc[nome, "P aritmético"]))
        primeira = curvas[FORMULA]
        print(f"\n=== Fase {fase}: 1ª decisão {primeira.pesos_alvo.index[0]:%d/%m/%Y}, fim "
              f"{primeira.valores.index[-1]:%d/%m/%Y}, {len(primeira.pesos_alvo)} rebalanceios; adiados "
              + ", ".join(f"{nome.split(' (')[0]} {len(curvas[nome].adiamentos)}" for nome in ESTRATEGIAS) + " ===")
        print(_tabela(tabela, ["CAGR", "MDD", "Giro anual"]))
        print(f"(fase {fase} pronta em {time.perf_counter() - inicio_rodada:.0f} s)", flush=True)

    for nome in NOVAS:
        decisao = decidir(ics[nome], p_bova[nome])
        print(f"\n{nome}: portão ({PORTAO.nome}, 2012-01 a 2026-09) "
              f"{'cumprido' if decisao.aprovada else 'não cumprido'}; "
              f"{'APROVADA' if decisao.aprovada else 'REPROVADA'}; candidata à prova final (P[> BOVA11] >= 0,95 "
              f"nas 3 fases): {'sim' if decisao.candidata else 'não'}")
    print(f"(tempo total {time.perf_counter() - inicio_rodada:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
