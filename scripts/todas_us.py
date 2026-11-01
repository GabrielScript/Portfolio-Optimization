"""
todas_us.py - The one reading of every strategy on US stocks (docs/superpowers/specs/2026-11-01-todas-as-estrategias-design.md, sections 2,
3 and 5), on the exports frozen for the hybrid QPO, after its US reading. Each configuration is read once per market.

    python -X utf8 scripts/todas_us.py D:/b3-datalake/exports/qml/us/exports docs/us-estrategias-2026-11-01.txt
        > D:/b3-datalake/exports/todas/us/leitura.txt

Each arm's result is saved per phase, so a crash resumes where it stopped. The cheap arms come first. Writes
docs/dados/todas-{anos,portao}-us.parquet.
"""

from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import us_teste as ut  # noqa: E402
from core.backtest import Estrategia, ResultadoBacktest, backtest_pesos_fixos, estrategia_equiponderada, estrategia_nucleo_satelite, walk_forward_dinamico  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.config import Objetivo  # noqa: E402
from core.crescimento_lucro import nota_de_crescimento  # noqa: E402
from core.estrategias_todas import BRACOS_NOVOS, no_grid_mensal_real  # noqa: E402
from core.estrategias_todas_us import (estrategia_grande, estrategia_reamostrada_grande, estrategia_views_grande, hrp_grande,  # noqa: E402
                                       selecao_e_max_sharpe_mensal)
from core.hierarchical import otimizar_hrp  # noqa: E402
from core.hrp_paper import otimizar_hrp_arvore_paper, otimizar_hrp_paper  # noqa: E402
from core.metrics import retornos_de_valores  # noqa: E402
from core.periodos import anos_em_reais  # noqa: E402
from core.poder import ic_por_mes  # noqa: E402
from core.qpo import CFG_MAX_SHARPE, K, exact_picks, qubo_selector  # noqa: E402
from core.qubo_universo import FULL_UNIVERSE, SECTOR_CAP, anneal_solver, composite_score  # noqa: E402
from core.reamostragem import ConfigReamostragem  # noqa: E402
from core.risco import Limites, stop_relativo  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import carteira_da_nota, conservadora, estrategia_conservadora, painel_de_notas  # noqa: E402
from core.valor import painel_da_nota  # noqa: E402
from core.vencedoras import CAIXA, estrategia_vencedoras  # noqa: E402
from core.views import Sinal, nota_media_dos_escores  # noqa: E402
from scripts.avaliar_conservadora import Mensal, retornos_no_universo  # noqa: E402
from scripts.avaliar_robustez import contar_tentativas  # noqa: E402
from scripts.avaliar_universo_dinamico import CFG_KELLY  # noqa: E402
from scripts.avaliar_valor import notas  # noqa: E402
from scripts.paper_v3 import nota_do_composto  # noqa: E402
from scripts.qml_avaliar_b3 import COMPOSTO, JUROS, SAIDA  # noqa: E402
from scripts.qml_avaliar_us import CURVAS_DA_B3  # noqa: E402
from scripts.todas_comum import esperar_a_janela_passar, resultado_salvo  # noqa: E402
from scripts.us_avaliar import CONSERVADORA, FASES, SPY, Dados, _texto, anos, carregar, medidas_da_fase  # noqa: E402

PASTA = Path("D:/b3-datalake/exports/todas/us")
AGG = "AGG"
TOP_20 = 20
FRACAO_30, FRACAO_50 = 0.30, 0.50
LIMITES_DO_STOP = Limites(queda_relativa_stop=0.30)     # the recorded row's 30%, not today's rule
RESAMPLED = ConfigReamostragem(100, 1, 0)               # blocks of 1 month, the B3's 21 sessions
TOLERANCIA_DO_ENCANAMENTO = 0.0006                      # the report has three decimals
PERIODOS = 12


def cagrs_do_relatorio(texto: str, nomes: tuple[str, ...]) -> dict[int, dict[str, float]]:
    """The phases' CAGR of ``nomes`` as the batch's report prints them."""
    cagrs: dict[int, dict[str, float]] = {}
    for fase, bloco in re.findall(r"=== Phase (\d+):(.*?)(?==== |\Z)", texto, flags=re.DOTALL):
        cagrs[int(fase)] = {nome: float(re.search(rf"^{re.escape(nome)}\s+(-?\d+\.\d+)", bloco, flags=re.MULTILINE).group(1)) for nome in nomes}
    return cagrs


def conferir_encanamento(publicado: Mapping[str, float], medidos: Mapping[str, float], fase: int) -> None:
    for nome, valor in publicado.items():
        if abs(medidos[nome] - valor) > TOLERANCIA_DO_ENCANAMENTO:
            raise SystemExit(f"phase {fase}: {nome} CAGR {medidos[nome]:.4f} != the batch's {valor:.3f}")


def data_do_stop_de_30(satelite: pd.Series, rebalanceios: pd.DatetimeIndex, referencia: pd.Series) -> pd.Timestamp | None:
    """``data_do_stop`` at the recorded row's 30%: the first rebalance whose satellite/SPY ratio, known at the previous close, fell that much."""
    for data in rebalanceios:
        if stop_relativo(satelite[satelite.index < data], referencia[referencia.index < data], LIMITES_DO_STOP):
            return data
    return None


def satelite_com_stop_de_30(satelite: pd.Series, referencia: pd.Series, data: pd.Timestamp, custo_bps: float) -> pd.Series:
    """``satelite_com_stop``: from that rebalance on, the whole satellite is in SPY, after paying to sell it all and buy SPY."""
    antes = satelite[satelite.index < data]
    depois = referencia[referencia.index >= data]
    trocado = antes.iloc[-1] * (1.0 - 2.0 * custo_bps / 1e4) * depois / referencia[referencia.index < data].iloc[-1]
    return pd.concat([antes, trocado])


def mistura_de_curvas(curvas: Mapping[str, pd.Series], pesos: Mapping[str, float], janela_teste: int, custo_bps: float) -> pd.Series:
    """Fixed weights over curves on the same dates, back to target every ``janela_teste`` months, paying on what changes."""
    quadro = pd.DataFrame({nome: retornos_de_valores(curva) for nome, curva in curvas.items()})
    configuracao = replace(ut.CONFIG, janela_teste=janela_teste, custo_bps=custo_bps, capital_inicial=1.0)
    resultado = backtest_pesos_fixos(quadro, pd.Series(pesos), configuracao, "mistura")
    return resultado.valores.set_axis(next(iter(curvas.values())).index)


def curva_do_t_bill(rf_mensal: pd.Series, datas: pd.DatetimeIndex) -> pd.Series:
    return pd.concat([pd.Series([1.0], index=datas[:1]), (1.0 + rf_mensal.reindex(datas[1:]).fillna(0.0)).cumprod()])


def com_caixa_do_t_bill(precos: pd.DataFrame, rf_mensal: pd.Series) -> pd.DataFrame:
    """The cash column for the arm that holds cash, the only one that sees it."""
    return precos.assign(**{CAIXA: curva_do_t_bill(rf_mensal, precos.index)})


def sinais_das_views(dados: Dados) -> list[Sinal]:
    """The two views with their monthly ICs from the panel's own months: the Conservadora and value."""
    def nota_conservadora(mes, candidatos):
        return conservadora(dados.retornos, dados.proventos, mes, candidatos)

    nota_valor = _nota_de_valor(dados)
    meses = pd.date_range(dados.retornos.index[0], dados.retornos.index[-1], freq="MS")
    no_universo = retornos_no_universo(dados.retornos, dados.universo, meses)
    ics_conservadora = ic_por_mes(painel_de_notas(dados.retornos, dados.proventos, dados.universo, meses), no_universo)
    ics_valor = ic_por_mes(painel_da_nota(nota_valor, dados.universo, meses, dados.retornos.columns), no_universo)
    return [Sinal(nota_conservadora, ics_conservadora), Sinal(nota_valor, ics_valor)]


def _nota_de_valor(dados: Dados):
    return nota_media_dos_escores(*notas(dados.balancos, Mensal(dados.retornos, dados.proventos, {})).values())


def estrategias_do_motor(dados: Dados, rf: pd.Series, log: dict) -> dict[str, Callable[[], Estrategia]]:
    """Each walk-forward arm, cheap ones first, built only when it runs: the views' ICs alone take minutes."""
    setores = dados.setores
    kelly_2 = CFG_KELLY
    minima = replace(kelly_2, objetivo=Objetivo.MIN_VOL)
    sharpe = replace(kelly_2, objetivo=Objetivo.MAX_SHARPE)

    def nota_conservadora(mes, candidatos):
        return conservadora(dados.retornos, dados.proventos, mes, candidatos)

    def por_max_sharpe(selecionar, nota=nota_conservadora) -> Estrategia:
        return selecao_e_max_sharpe_mensal(selecionar, nota, setores, CFG_MAX_SHARPE, rf)

    def qubo(nota) -> Estrategia:
        return por_max_sharpe(qubo_selector(setores, FULL_UNIVERSE, K, anneal_solver(setores, SECTOR_CAP, log)), nota)

    def nucleo(fracao: float, com_setores: Mapping[str, str] | None) -> Estrategia:
        return estrategia_nucleo_satelite(estrategia_grande(minima, rf, com_setores), estrategia_grande(kelly_2, rf, com_setores), fracao)

    novos = {nome: (lambda construir=construir: construir(setores, PERIODOS)) for nome, construir in BRACOS_NOVOS.items()}
    return {
        "1/N": estrategia_equiponderada,
        "Kelly λ=1": lambda: estrategia_grande(replace(kelly_2, aversao_risco=1.0), rf),
        "Kelly λ=2": lambda: estrategia_grande(kelly_2, rf),
        "Kelly λ=4": lambda: estrategia_grande(replace(kelly_2, aversao_risco=4.0), rf),
        "Máx. Sharpe": lambda: estrategia_grande(sharpe, rf),
        "Mín. Vol": lambda: estrategia_grande(minima, rf),
        "Núcleo Mín. Vol 70/30": lambda: nucleo(FRACAO_30, None),
        "Núcleo Mín. Vol 50/50": lambda: nucleo(FRACAO_50, None),
        "Kelly λ=2 com regras": lambda: estrategia_grande(kelly_2, rf, setores),
        "Mín. Vol com regras": lambda: estrategia_grande(minima, rf, setores),
        "Núcleo Mín. Vol 70/30 com regras": lambda: nucleo(FRACAO_30, setores),
        "Topk-MS": lambda: estrategia_grande(replace(CFG_MAX_SHARPE, n_ativos_max=K), rf, setores),
        "Conservadora-MS": lambda: por_max_sharpe(lambda data, treino, nota: list(carteira_da_nota(nota, setores, K, SECTOR_CAP).index)),
        "QPO-exact": lambda: por_max_sharpe(qubo_selector(setores, TOP_20, K, exact_picks)),
        **novos,
        "HRP": lambda: hrp_grande(otimizar_hrp),
        "HRP distância do paper": lambda: hrp_grande(otimizar_hrp_paper),
        "HRP-Árvore ward": lambda: hrp_grande(otimizar_hrp_arvore_paper),
        "Kelly λ=2 com views": lambda: estrategia_views_grande(kelly_2, setores, sinais_das_views(dados), dados.balancos.valor_mercado),
        "QUBO-universo": lambda: qubo(nota_conservadora),
        "QUBO-universo-valor": lambda: qubo(composite_score(nota_conservadora, _nota_de_valor(dados))),
        "QUBO-universo-crescimento": lambda: qubo(composite_score(nota_conservadora, _nota_de_valor(dados), nota_de_crescimento(dados.balancos.fundamentos))),
        "Kelly λ=2 reamostrado": lambda: estrategia_reamostrada_grande(kelly_2, RESAMPLED),
    }


def _rodar_braco(nome: str, construir: Callable[[], Estrategia], fase: int, precos: pd.DataFrame, dados: Dados) -> ResultadoBacktest:
    def rodar() -> ResultadoBacktest:
        esperar_a_janela_passar()
        inicio = time.perf_counter()
        resultado = walk_forward_dinamico(precos, dados.universo, no_grid_mensal_real(construir()), ut.CONFIG, nome)
        print(f"{nome} phase {fase}: {time.perf_counter() - inicio:.0f} s; first failures {resultado.falhas[:2]}", flush=True)
        return resultado
    return resultado_salvo(PASTA / f"{nome.replace('/', '-')}-fase{fase}.pkl", rodar)


def _vencedoras(fase: int, precos: pd.DataFrame, dados: Dados, rf: pd.Series) -> ResultadoBacktest:
    def rodar() -> ResultadoBacktest:
        esperar_a_janela_passar()
        estrategia = estrategia_vencedoras(dados.retornos, nota_de_crescimento(dados.balancos.fundamentos), dados.setores)
        return walk_forward_dinamico(com_caixa_do_t_bill(precos, rf), dados.universo, ut.no_grid_mensal(estrategia), ut.CONFIG, "Vencedoras-crescimento")
    return resultado_salvo(PASTA / f"Vencedoras-crescimento-fase{fase}.pkl", rodar)


def curvas_compostas(kelly: pd.Series, kelly_regras: ResultadoBacktest, spy: pd.Series, agg: pd.Series, t_bill: pd.Series) -> dict[str, pd.Series]:
    """Rows 15, 16, 20, 34 and 35: curves mixed with the references. The B3's freeze by the satellite's daily loss has no daily data here."""
    janela, custo = ut.CONFIG.janela_teste, ut.CONFIG.custo_bps

    def mistura(partes: Mapping[str, pd.Series], pesos: Mapping[str, float]) -> pd.Series:
        return mistura_de_curvas(partes, pesos, janela, custo)

    satelite = kelly_regras.valores.dropna()
    stop = data_do_stop_de_30(satelite, kelly_regras.pesos_alvo.index, spy)
    efetivo = satelite if stop is None else satelite_com_stop_de_30(satelite, spy, stop, custo)
    return {"Núcleo SPY 70/30": mistura({"nucleo": spy, "satelite": kelly}, {"nucleo": 0.70, "satelite": 0.30}),
            "Núcleo SPY 50/50": mistura({"nucleo": spy, "satelite": kelly}, {"nucleo": 0.50, "satelite": 0.50}),
            "SPY 70% + Kelly com regras e stop de 30%": mistura({"nucleo": spy, "satelite": efetivo}, {"nucleo": 0.70, "satelite": 0.30}),
            "SPY 50% + AGG 50%": mistura({SPY: spy, AGG: agg}, {SPY: 0.50, AGG: 0.50}),
            "SPY 50% + T-bill 50%": mistura({SPY: spy, "T-bill": t_bill}, {SPY: 0.50, "T-bill": 0.50})}


def main(argv: list[str]) -> int:
    inicio = time.perf_counter()
    PASTA.mkdir(parents=True, exist_ok=True)
    dados = carregar(Path(argv[0]))
    publicado = cagrs_do_relatorio(Path(argv[1]).read_text(encoding="utf-8"), (COMPOSTO, CONSERVADORA))
    n = contar_tentativas()
    rf = dados.fatores["rf"]
    log: dict = {}
    construtores = estrategias_do_motor(dados, rf, log)
    print(f"Coverage: {dados.cobertura}; DSR N {n}; last month {dados.retornos.index[-1]:%Y-%m}; {len(construtores)} walk-forward arms", flush=True)

    nomes_do_portao: list[str] = []
    p_composto: dict[str, list[float]] = {}
    p_conservadora: dict[str, list[float]] = {}
    p_spy: dict[str, list[float]] = {}
    dsr: dict[str, list[float]] = {}
    tabela_de_anos = None
    for fase in FASES:
        precos = ut.indice_total(dados.retornos).iloc[fase:]
        composto = walk_forward_dinamico(precos, dados.universo, ut.no_grid_mensal(estrategia_por_nota(nota_do_composto(dados.retornos, dados.proventos, dados.balancos), dados.setores)), ut.CONFIG, COMPOSTO)
        cons = walk_forward_dinamico(precos, dados.universo, ut.no_grid_mensal(estrategia_conservadora(dados.retornos, dados.proventos, dados.setores)), ut.CONFIG, CONSERVADORA)
        referencias = {COMPOSTO: composto.valores.dropna(), CONSERVADORA: cons.valores.dropna()}
        datas = referencias[COMPOSTO].index
        spy = ut.referencia_nas_datas(dados.etfs[SPY], datas)
        conferir_encanamento(publicado[fase], {nome: ut.medidas(curva, rf)["CAGR"] for nome, curva in referencias.items()}, fase)

        resultados = {nome: _rodar_braco(nome, construir, fase, precos, dados) for nome, construir in construtores.items()}
        resultados["Vencedoras-crescimento"] = _vencedoras(fase, precos, dados, rf)
        curvas = {nome: resultado.valores.dropna() for nome, resultado in resultados.items()}
        curvas |= curvas_compostas(curvas["Kelly λ=2"], resultados["Kelly λ=2 com regras"], spy,
                                   ut.referencia_nas_datas(dados.etfs[AGG], datas), curva_do_t_bill(rf, datas))
        nomes_do_portao = list(curvas)
        comparadas = {**curvas, **referencias}
        tabela = medidas_da_fase(comparadas, spy, rf, n, dados.fatores)
        for nome in nomes_do_portao:
            tabela.loc[nome, "P[> Composto]"] = psr_da_diferenca(curvas[nome], referencias[COMPOSTO])
            tabela.loc[nome, "P[> Conservadora]"] = psr_da_diferenca(curvas[nome], referencias[CONSERVADORA])
            p_composto.setdefault(nome, []).append(float(tabela.loc[nome, "P[> Composto]"]))
            p_conservadora.setdefault(nome, []).append(float(tabela.loc[nome, "P[> Conservadora]"]))
            p_spy.setdefault(nome, []).append(float(tabela.loc[nome, "P[> SPY]"]))
            dsr.setdefault(nome, []).append(float(tabela.loc[nome, "DSR"]))
        print(f"\n=== Phase {fase}: {datas[0]:%m/%Y} to {datas[-1]:%m/%Y}, {len(datas) - 1} months ===")
        print(_texto(tabela))
        for nome, resultado in {**resultados, COMPOSTO: composto, CONSERVADORA: cons}.items():
            print(f"{nome}: refusals {len(resultado.recusas)}, failures {len(resultado.falhas)}, annual turnover {resultado.turnover_anual_medio:.2f}")
        print(f"SPY: {ut.medidas(spy, rf)}")
        print(_texto(anos({**comparadas, SPY: spy}) * 100, 1))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)
        if fase == FASES[0]:
            vivas = pd.read_parquet(CURVAS_DA_B3)
            bova11 = vivas[vivas["fase"] == vivas["fase"].min()].set_index("date")["BOVA11"]
            em_reais = pd.DataFrame({f"{nome} EUA": ut.em_reais(curva, dados.ptax) for nome, curva in comparadas.items()})
            tabela_de_anos = anos_em_reais(em_reais, bova11, pd.read_parquet(JUROS).set_index("date"))

    portao = []
    print("\nGate, all >= 0.95 in every phase:")
    for nome in nomes_do_portao:
        passa = ut.passa_o_portao([*p_composto[nome], *p_conservadora[nome], *p_spy[nome]], dsr[nome])
        portao.append({"braco": nome, "p_composto": p_composto[nome], "p_conservadora": p_conservadora[nome],
                       "p_referencia": p_spy[nome], "dsr": dsr[nome], "passa": passa})
        print(f"{nome}: P[> Composite] {np.round(p_composto[nome], 3)}, P[> Conservadora] {np.round(p_conservadora[nome], 3)}, "
              f"P[> SPY] {np.round(p_spy[nome], 3)}, DSR {np.round(dsr[nome], 3)}: {'PASSES' if passa else 'does not pass'}")
    tabela_de_anos.to_parquet(SAIDA / "todas-anos-us.parquet", index=False)
    pd.DataFrame(portao).to_parquet(SAIDA / "todas-portao-us.parquet", index=False)
    print("\n" + tabela_de_anos.to_string(float_format=lambda valor: f"{valor:.1%}"))
    print(f"\nDone in {time.perf_counter() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
