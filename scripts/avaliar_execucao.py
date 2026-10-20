"""
Prova final (PLAN §11): a Conservadora, com o Kelly de controle, refeita com execução realista (S4: custo por liquidez
e IR de pessoa física) em R$ 10 mil, R$ 100 mil e R$ 1 milhão, contra o BOVA11 depois do IR. Não muda o paper trading
(paper-v1) nem o critério de pronto. Rede só para o CDI e a Selic meta.

Uso: .venv/Scripts/python -X utf8 scripts/avaliar_execucao.py > docs/execucao-2026-10-20.txt

Pré-registro (2026-10-20, escrito e commitado antes de rodar; depois desse commit o texto não muda):
- Curvas do motor: as do `scripts/avaliar_valor.py`, sem mudança (`curvas_da_fase` dele, sem as duas notas de valor):
  a Conservadora (aprovada no S2.1, a única candidata que sobrou das S2.1 a S3), o "Kelly λ=2 com regras" (o do
  paper-v1, aqui só controle) e o BOVA11 comprado e mantido; fases 0, 21 e 42 a partir do corte que põe a 1ª decisão
  em 02/01/2012; treino de 1.260 pregões, rebalanceio a cada 63, 10 bps por lado no motor. Dados: os mesmos arquivos e
  hashes da rodada do valor, mais `docs/dados/2026-10-20-liquidez/liquidez.parquet` (conferido pelo `SHA256SUMS`).
- Âncora: antes de tudo, cada curva refeita por `core.execucao.refazer` com os 10 bps do motor, sem liquidez e sem IR,
  tem de ser a do motor (desvio relativo máximo <= 1e-9); se não for, o script para.
- Execução realista (`core.execucao.refazer`: a mesma sequência de carteiras do motor, refeita em reais): cada ordem
  paga emolumentos de 0,030% (tarifa da B3 para ações e ETFs à vista: negociação 0,005% + CCP 0,0224% + TTA 0,0026%,
  faixa de ADTV até R$ 3 milhões; página de tarifas da B3 lida em 20/10/2026), o meio spread cotado do mês anterior
  ao do fechamento da ordem e o impacto Y x vol diária x raiz(ordem / volume mediano diário) com Y = 1 (lei da raiz
  quadrada; "the numerical constant Y is of order unity", Tóth et al., Phys. Rev. X 1, 021006, 2011, eq. 1). Papel sem
  medida no mês fica com a mediana do mês; mês sem liquidez para o script. Sem corretagem.
- IR (`core.impostos.apurar_mes`): preço médio por papel; no último pregão de cada mês, 15% sobre o ganho líquido do
  mês menos o prejuízo acumulado; mês com vendas de ações até R$ 20 mil é isento (Lei 11.033/2004, art. 3º, I) e o
  prejuízo dele acumula; o BOVA11 não tem isenção. Provento reinvestido entra na base pelo `yield` de
  `dividends.parquet` no dia ex (sem a restituição de capital) e não paga IR de novo como ganho; o BOVA11 não recebe.
- Valor de saída: em cada fechamento, o patrimônio menos o IR de vender tudo ali (somado às vendas e ao resultado do
  mês). É a curva que conta; o patrimônio antes do IR de saída é só descritivo.
- Por fase e capital: P[> BOVA11] aritmético e em log dos valores de saída (`psr_da_diferenca`, a medida das provas
  anteriores), P[Conservadora > Kelly], CAGR e MDD da saída, saída final, custo por ano sobre o patrimônio médio, IR
  pago no caminho e o IR que a saída ainda cobraria; e o P aritmético das curvas do motor (10 bps, sem IR), para
  comparar. Descritivo, só com R$ 10 mil: a Conservadora e o BOVA11 desde a 1ª decisão de 2014 (P de 0,89 a 0,96 no
  motor nesse trecho, `docs/valor-2026-10-20.txt`).
- Decisão fixa agora: aprovada na prova final = P[> BOVA11] aritmético da saída da Conservadora >= 0,95 com R$ 10 mil
  nas 3 fases (o tamanho mais perto do da carteira real; R$ 100 mil, R$ 1 milhão e o trecho de 2014 não mudam a
  decisão). Aprovada: a Conservadora vira a candidata ao paper-v2 (registro novo, separado do paper-v1; agendar só com
  o OK do usuário). Reprovada: não há paper-v2, a Conservadora fica como pesquisa e não volta com outro custo, tamanho
  ou período sem pré-registro novo. O Kelly não tem decisão aqui. O N do DSR fica em 30 (`docs/tentativas.csv`): a
  prova não cria configuração nova, só muda o modelo de custo e põe o IR; sem DSR nesta rodada.
- Simplificações, escritas aqui: sem corretagem (a das corretoras grandes hoje); o IR do mês sai do patrimônio na
  proporção de cada posição, sem contar como venda, e é pago no último pregão do mês (o DARF vence no mês seguinte); a
  isenção conta só as vendas desta carteira (a lei soma todas as vendas de ações da pessoa); units contam como ações;
  JCP líquido de 15% como o lago guarda (17,5% desde 2026, LC 224/2025, não entra); sem a tributação de dividendos
  acima de R$ 50 mil por mês por empresa e sem o IRPFM (Lei 15.270/2025: fora do alcance destes tamanhos, menos talvez
  no de R$ 1 milhão no fim); com R$ 10 mil as ordens ficam no mercado fracionário (lotes de 100), cujo spread pode
  ser maior que o do lote padrão do lago; a saída não paga o custo das ordens de venda; o último pregão da amostra não
  apura o mês (a saída cobre).
- Expectativa registrada: com R$ 10 mil a Conservadora paga pouco ou nenhum IR no caminho (vendas do mês abaixo de
  R$ 20 mil por anos) e o BOVA11 paga 15% do ganho na saída, então o IR ajuda a Conservadora; P aritmético da saída
  entre 0,95 e 0,995 nas 3 fases (aprovada), perto do P do motor ou acima. Com R$ 1 milhão o IR morde (15% do ganho
  realizado a cada rebalanceio contra o diferimento do BOVA11): P entre 0,85 e 0,97. Custo por ano da Conservadora de
  0,2% a 0,6% com R$ 10 mil e de 0,3% a 0,8% com R$ 1 milhão; IR pago + IR na saída abaixo de 5% da saída final com
  R$ 10 mil e de 10% a 25% com R$ 1 milhão; o do BOVA11 de 7% a 10%. Kelly com P abaixo do da Conservadora. Desde
  2014, P entre 0,85 e 0,97.
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
from core.backtest import ResultadoBacktest, retornos_da_carteira  # noqa: E402
from core.config import DIAS_UTEIS_ANO, taxa_anual_para_aritmetica  # noqa: E402
from core.execucao import Custos, Liquidez, Refeita, refazer  # noqa: E402
from core.impostos import ISENCAO_DE_VENDAS  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from scripts.avaliar_conservadora import CONSERVADORA, PASTA_MENSAL, carregar_mensal  # noqa: E402
from scripts.avaliar_regras_no_motor import SATELITE  # noqa: E402
from scripts.avaliar_robustez import FASES, _tabela, carregar, conferir_hashes, contar_tentativas  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT, CFG_KELLY, _porcento  # noqa: E402
from scripts.avaliar_valor import PRIMEIRA_DECISAO, corte_dos_precos, curvas_da_fase  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
PASTA_LIQUIDEZ = RAIZ / "docs" / "dados" / "2026-10-20-liquidez"
CUSTOS_DA_B3 = Custos(emolumentos=0.0003, impacto=1.0)
CAPITAIS = (10_000.0, 100_000.0, 1_000_000.0)
CAPITAL_DA_PROVA = 10_000.0
INICIO_DESCRITIVO = pd.Timestamp("2014-01-01")
ESTRATEGIAS = (CONSERVADORA, SATELITE)
P_MINIMO_DA_PROVA = 0.95
_TOLERANCIA = 1e-9
_TOLERANCIA_DA_ANCORA = 1e-9
RESTITUICAO_DE_CAPITAL = "REST CAP DIN"     # devolve capital: não é provento, não entra na base
EM_PORCENTO = ("CAGR", "MDD", "Custo por ano")
EM_REAIS = ("Saída final (R$)", "IR pago (R$)", "IR na saída (R$)")


def proventos_diarios(dividendos: pd.DataFrame, datas: pd.DatetimeIndex, papeis: Sequence[str]) -> pd.DataFrame:
    """Pregão x papel: o provento do dia ex sobre o fechamento ex (``yield`` do lago; vários no mesmo dia compõem),
    sem a restituição de capital; 0 nos outros dias."""
    pagos = dividendos.loc[dividendos["corporate_action"] != RESTITUICAO_DE_CAPITAL]
    pagos = pagos.assign(ex_date=pd.to_datetime(pagos["ex_date"]), fator=1.0 + pagos["yield"])
    compostos = pagos.groupby(["ex_date", "security"])["fator"].prod() - 1.0
    return compostos.unstack("security").reindex(index=datas, columns=list(papeis)).fillna(0.0).rename_axis(
        index=None, columns=None)


def liquidez_da_tabela(tabela: pd.DataFrame) -> Liquidez:
    """A tabela do arquivo (``month``, ``security`` e as três medidas) em quadros mês (1º dia) x papel."""
    quadros = tabela.assign(month=pd.to_datetime(tabela["month"])).pivot(index="month", columns="security")
    return Liquidez(*(quadros[medida].rename_axis(index=None, columns=None)
                      for medida in ("meio_spread", "vol_diaria", "volume_mediano")))


def a_partir_de(curva: ResultadoBacktest, inicio: pd.Timestamp) -> ResultadoBacktest:
    """A mesma sequência de carteiras desde a 1ª decisão em ``inicio`` ou depois, começando em caixa no pregão
    anterior a ela (o fechamento da compra)."""
    primeira = curva.pesos_alvo.index[curva.pesos_alvo.index >= inicio][0]
    anterior = curva.valores.index[curva.valores.index.get_loc(primeira) - 1]
    return replace(curva, valores=curva.valores.loc[anterior:], pesos_alvo=curva.pesos_alvo.loc[primeira:],
                   turnover=curva.turnover.loc[primeira:])


def decidir_prova(p_por_fase: Sequence[float]) -> bool:
    """Aprovada na prova final: P[> BOVA11] aritmético dos valores de saída >= 0,95 em todas as fases."""
    return all(p >= P_MINIMO_DA_PROVA - _TOLERANCIA for p in p_por_fase)


def conferir_ancora(curva: ResultadoBacktest, retornos: pd.DataFrame, custo_bps: float) -> None:
    """Refeita com o custo do motor (``custo_bps`` por lado, sem liquidez e sem IR), a curva tem de ser a do motor;
    para o script se não for."""
    refeita = refazer(curva, retornos, float(curva.valores.iloc[0]), Custos(custo_bps / 1e4, 0.0), com_ir=False)
    desvio = float((refeita.valores / curva.valores - 1.0).abs().max())
    if not desvio <= _TOLERANCIA_DA_ANCORA:
        raise SystemExit(f"{curva.nome}: a curva refeita sem IR desvia {desvio:.2e} da do motor")


def comprado_e_mantido(recorte: ResultadoBacktest, ativo: str) -> ResultadoBacktest:
    """``ativo`` comprado na 1ª decisão de ``recorte`` e mantido, nas mesmas datas (a referência do recorte)."""
    primeira = recorte.pesos_alvo.index[:1]
    return ResultadoBacktest(ativo, "comprado e mantido", recorte.valores, pd.DataFrame({ativo: [1.0]}, index=primeira),
                             pd.Series([1.0], index=primeira), 0.0)


def medidas(refeitas: Mapping[str, Refeita]) -> pd.DataFrame:
    """Por curva refeita: P[> BOVA11] aritmético e em log dos valores de saída (o da prova é o aritmético), P[> Kelly]
    da Conservadora quando o Kelly está junto, CAGR e MDD da saída, saída final, custo por ano sobre o patrimônio médio,
    IR pago no caminho e o que a saída ainda cobraria."""
    bova = refeitas["BOVA11"].saida
    linhas = {}
    for nome, refeita in refeitas.items():
        anos = (len(refeita.valores) - 1) / DIAS_UTEIS_ANO
        linha = {"CAGR": cagr(refeita.saida), "MDD": drawdown(refeita.saida)[1],
                 "Saída final (R$)": float(refeita.saida.iloc[-1]),
                 "Custo por ano": float(refeita.custos.sum() / refeita.valores.mean() / anos),
                 "IR pago (R$)": float(refeita.impostos.sum()),
                 "IR na saída (R$)": float(refeita.valores.iloc[-1] - refeita.saida.iloc[-1])}
        if nome != "BOVA11":
            linha["P aritmético"] = psr_da_diferenca(refeita.saida, bova)
            linha["P log"] = psr_da_diferenca(refeita.saida, bova, log=True)
        if nome == CONSERVADORA and SATELITE in refeitas:
            linha["P[> Kelly]"] = psr_da_diferenca(refeita.saida, refeitas[SATELITE].saida)
        linhas[nome] = linha
    return pd.DataFrame(linhas).T


def refeitas(curvas: Mapping[str, ResultadoBacktest], retornos: pd.DataFrame, liquidez: Liquidez,
             proventos: pd.DataFrame, capital: float) -> dict[str, Refeita]:
    """As estratégias de ``curvas`` (ações: isenção até R$ 20 mil, proventos na base) e o BOVA11 (sem isenção)."""
    feitas = {nome: refazer(curvas[nome], retornos, capital, CUSTOS_DA_B3, liquidez, isencao=ISENCAO_DE_VENDAS,
                            proventos=proventos) for nome in ESTRATEGIAS if nome in curvas}
    feitas["BOVA11"] = refazer(curvas["BOVA11"], retornos, capital, CUSTOS_DA_B3, liquidez, isencao=0.0)
    return feitas


def _formatar(tabela: pd.DataFrame) -> str:
    em_reais = {coluna: tabela[coluna].round().astype("int64") for coluna in EM_REAIS}
    return _tabela(tabela.assign(**em_reais), EM_PORCENTO)


def _p_do_motor(curvas: Mapping[str, ResultadoBacktest]) -> str:
    return ", ".join(f"{nome.split(' (')[0]} {psr_da_diferenca(curvas[nome].valores, curvas['BOVA11'].valores):.3f}"
                     for nome in ESTRATEGIAS)


def main() -> int:
    inicio_rodada = time.perf_counter()
    hoje = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    dados, mensal = carregar(), carregar_mensal()
    sha256 = dados.sha256 | mensal.sha256 | conferir_hashes(PASTA_LIQUIDEZ, ["liquidez.parquet"])
    liquidez = liquidez_da_tabela(pd.read_parquet(PASTA_LIQUIDEZ / "liquidez.parquet"))
    dividendos = pd.read_parquet(PASTA_MENSAL / "dividends.parquet",
                                 columns=["security", "ex_date", "corporate_action", "yield"])
    selic = market_data.obter_selic_meta(hoje)
    try:
        cdi = market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=hoje)
    except market_data.ErroDadosMercado as erro:
        print(f"CDI: {erro} (Selic constante)")
        cdi = pd.Series(dtype=float)
    cdi_vigente = cdi if not cdi.empty else None
    cfg = replace(CFG_KELLY, taxa_livre_risco=taxa_anual_para_aritmetica(selic.valor))
    corte = corte_dos_precos(dados.precos.index, PRIMEIRA_DECISAO, CFG_BT.janela_treino)
    print(f"Arquivos (sha256): {sha256}")
    print(f"Manifest: {dados.manifest}; Selic meta {_porcento(selic.valor)} ({selic.fonte}); "
          f"CDI {'do BCB' if cdi_vigente is not None else 'indisponível'}; N = {contar_tentativas()} (sem DSR); "
          f"preços desde {dados.precos.index[corte]:%d/%m/%Y}; {CUSTOS_DA_B3}")

    p_da_prova = []
    for fase in FASES:
        curvas = curvas_da_fase(dados, mensal, {}, corte, fase, cdi_vigente, cfg)
        retornos = retornos_da_carteira(dados.precos.iloc[corte + fase:])
        for nome in (*ESTRATEGIAS, "BOVA11"):
            conferir_ancora(curvas[nome], retornos, CFG_BT.custo_bps)
        proventos = proventos_diarios(dividendos, retornos.index, retornos.columns)
        conservadora = curvas[CONSERVADORA]
        print(f"\n=== Fase {fase}: 1ª decisão {conservadora.pesos_alvo.index[0]:%d/%m/%Y}, fim "
              f"{conservadora.valores.index[-1]:%d/%m/%Y}; âncora conferida; P[> BOVA11] aritmético no motor "
              f"(10 bps, sem IR): {_p_do_motor(curvas)} ===")
        for capital in CAPITAIS:
            tabela = medidas(refeitas(curvas, retornos, liquidez, proventos, capital))
            if capital == CAPITAL_DA_PROVA:
                p_da_prova.append(float(tabela.loc[CONSERVADORA, "P aritmético"]))
            print(f"\n--- R$ {capital:,.0f} ---".replace(",", "."))
            print(_formatar(tabela))
        recorte = a_partir_de(conservadora, INICIO_DESCRITIVO)
        tabela = medidas(refeitas({CONSERVADORA: recorte, "BOVA11": comprado_e_mantido(recorte, "BOVA11")}, retornos,
                                  liquidez, proventos, CAPITAL_DA_PROVA))
        print(f"\n--- Descritivo: R$ 10.000 desde a 1ª decisão de 2014 ({recorte.pesos_alvo.index[0]:%d/%m/%Y}) ---")
        print(_formatar(tabela))
        print(f"(fase {fase} pronta em {time.perf_counter() - inicio_rodada:.0f} s)", flush=True)

    aprovada = decidir_prova(p_da_prova)
    print("\nConservadora com R$ 10.000: P[> BOVA11] aritmético da saída por fase "
          + ", ".join(f"{p:.3f}" for p in p_da_prova)
          + f"; prova final {'APROVADA (candidata ao paper-v2)' if aprovada else 'REPROVADA (sem paper-v2)'}")
    print(f"(tempo total {time.perf_counter() - inicio_rodada:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
