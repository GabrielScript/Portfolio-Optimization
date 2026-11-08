"""
bdr_conjunto_avaliar.py - Conjunto-15's one reading (docs/superpowers/specs/2026-11-08-bdr-conjunto-design.md): B3
stocks and US BDRs in one ranking, 2021-01 to 16/10/2026, phases 0/21/42, the DSR at the pre-registered union N = 109.
Runs only after the pre-registration commit (`7cb16f5`). Reads the lake's gold table live (hashed); nothing goes to the
app.

    python -X utf8 scripts/bdr_conjunto_avaliar.py > docs/bdr-conjunto-2026-11-08.txt
"""

from __future__ import annotations

import hashlib
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, ResultadoBacktest, walk_forward_dinamico  # noqa: E402
from core.bdr_conjunto import nota_conjunta, simular  # noqa: E402
from core.cdi_lago import daily_cdi  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.mix_bdr import elegiveis_no_mes, escolher_eua, momentum_12_1  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import POR_SETOR, carteira_da_nota, estrategia_conservadora  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, conferir_hashes, referencias, tabela_do_periodo  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qml_avaliar_b3 import COMPOSTO, MACRO, conferir_composto  # noqa: E402
from scripts.qpo_avaliar import CONSERVADORA, _year_table, check_plumbing  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import P_MIN, _value_score, strategy_gate  # noqa: E402

GOLD = Path("D:/b3-datalake/gold/equity_adjusted.parquet")
INICIO_DA_LEITURA = pd.Timestamp("2021-01-04")
INICIO_DOS_DADOS = date(2019, 1, 1)
N_DSR = 109
NOMES = 15
CUSTO_BPS = 10.0
TOLERANCIA_CAGR = 0.003
BRACO = "Conjunto-15"
FIXA = "BDR-3 fixed (reported)"
SETOR_EUA = "EUA"
PRECOS_RUINS = {("AVGO34", "2021-09-17")}
EUA = ["NVDC34", "MUTC34", "GOGL34", "M2ST34", "MSFT34", "M1TA34", "AAPL34", "ITLC34", "A1MD34", "ORCL34", "NFLX34",
       "P2LT34", "AVGO34", "C2OI34", "W1DC34", "S1TX34", "D1EL34", "M2RV34", "AMZO34", "TSLA34", "COCA34", "DISB34",
       "NIKE34", "BERK34", "JPMC34", "GSGI34", "VISA34", "LILY34", "FCXO34", "EXXO34", "IVVB11", "NASD11"]


def ler_eua() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """DECISOES 298's slice of the gold table: ``tr_close`` and ``volume_brl`` of the US set, hashed."""
    tabela = pq.read_table(GOLD, columns=["date", "security", "tr_close", "volume_brl"],
                           filters=[("security", "in", EUA), ("date", ">=", INICIO_DOS_DADOS),
                                    ("date", "<=", LAST_SESSION.date())]).to_pandas()
    tabela["date"] = pd.to_datetime(tabela["date"])
    for papel, dia in PRECOS_RUINS:
        tabela = tabela[~((tabela["security"] == papel) & (tabela["date"] == pd.Timestamp(dia)))]
    tabela = tabela.sort_values(["security", "date"]).reset_index(drop=True)
    resumo = hashlib.sha256(pd.util.hash_pandas_object(tabela, index=False).to_numpy().tobytes()).hexdigest()
    precos = tabela.pivot(index="date", columns="security", values="tr_close").sort_index()
    volume = tabela.pivot(index="date", columns="security", values="volume_brl").reindex(precos.index)
    return precos, volume, resumo


def gravadora(estrategia, registro: dict):
    """The engine's strategy, recording each decision's eligible names."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        registro[treino.index[-1]] = list(treino.columns)
        return estrategia(treino)
    return _estrategia


def eua_elegiveis(precos_eua: pd.DataFrame, volume_eua: pd.DataFrame, data: pd.Timestamp) -> list[str]:
    """Section 2: eligible that month and with a 12-1 momentum at ``data``."""
    elegiveis = elegiveis_no_mes(volume_eua, data.to_period("M").to_timestamp())
    if not elegiveis:
        return []
    return momentum_12_1(precos_eua[elegiveis].loc[:data], data).dropna().index.tolist()


def _recorte(valores: pd.Series, inicio: pd.Timestamp) -> pd.Series:
    serie = valores.dropna().loc[inicio:]
    return serie / serie.iloc[0] * 10_000.0


def giro_dos_alvos(alvos: dict) -> float:
    """Target-to-target traded weight per year (drift between rebalances ignored)."""
    datas = sorted(alvos)
    total = sum(float(alvos[b].sub(alvos[a], fill_value=0.0).abs().sum()) for a, b in zip(datas, datas[1:]))
    return total / ((datas[-1] - datas[0]).days / 365.25)


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, sha256_cvm = _value_score(mensal, com_crescimento=True)
    sha256_cdi = conferir_hashes(MACRO, ["cdi_daily.parquet"])
    cdi = daily_cdi(ANOS_DE_CDI, LAST_SESSION.date(), path=MACRO / "cdi_daily.parquet")
    precos_eua, volume_eua, sha256_eua = ler_eua()
    precos_br = dados.precos.loc[:LAST_SESSION]
    # IVVB11 is also in the engine's price file (a reference): the joint table keeps the engine's column
    juntos = precos_br.join(precos_eua.drop(columns=[c for c in precos_eua.columns if c in precos_br.columns])
                            .reindex(precos_br.index), how="left")
    setores = dict(dados.setores) | {papel: SETOR_EUA for papel in EUA}
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm | sha256_cdi}; US slice {sha256_eua}; "
          f"US names in the lake {precos_eua.shape[1]} of {len(EUA)}; DSR N {N_DSR}")

    p_composto, p_conservadora, p_bova, dsr = [], [], [], []
    for fase in FASES:
        acoes = _acoes(precos_br.iloc[fase:])

        def rodar(estrategia, nome) -> ResultadoBacktest:
            return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

        conservadora = rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores), CONSERVADORA)
        check_plumbing(fase, cagr(conservadora.valores.dropna()))
        candidatos: dict = {}
        composto = rodar(gravadora(estrategia_por_nota(nota, dados.setores), candidatos), COMPOSTO)
        conferir_composto(fase, cagr(composto.valores.dropna()))
        refs = referencias(precos_br.iloc[fase:], CFG_BT)
        # the engine decides on the training window ending at session d - 1 and holds from d: the simulator
        # rebalances at the close of the decision session, so each target is keyed by it
        decisoes = sorted(candidatos)
        if len(decisoes) != len(composto.pesos_alvo):
            raise SystemExit("decisions and the engine's targets do not pair one to one")
        alvos_do_motor = dict(zip(decisoes, (linha for _, linha in composto.pesos_alvo.iterrows())))
        datas = [d for d in decisoes if d >= INICIO_DA_LEITURA]

        proprios = {d: alvos_do_motor[d][alvos_do_motor[d] > 1e-9] for d in datas}
        reproduzido = simular(proprios, juntos, CUSTO_BPS)
        motor = composto.valores.dropna().loc[datas[0]:]
        diferenca = cagr(reproduzido) - cagr(motor)
        print(f"\nPhase {fase} plumbing: simulator fed the Composite's picks CAGR {cagr(reproduzido):.2%} vs engine "
              f"{cagr(motor):.2%} (difference {diferenca:+.2%}, tolerance {TOLERANCIA_CAGR:.1%})")
        if abs(diferenca) > TOLERANCIA_CAGR:
            raise SystemExit("the simulator does not reproduce the engine's Composite: stopped before the arm")

        alvos, fixos, bdrs = {}, {}, []
        for d in datas:
            mes = d.to_period("M").to_timestamp()
            composto_hoje = nota(mes, candidatos[d]).dropna()
            eua = eua_elegiveis(precos_eua, volume_eua, d)
            pool = juntos[[*composto_hoje.index, *eua]]
            alvos[d] = carteira_da_nota(nota_conjunta(composto_hoje, pool, d), setores, NOMES, POR_SETOR)
            bdrs.append(sum(papel in EUA for papel in alvos[d].index))
            escolha_eua = escolher_eua(precos_eua, volume_eua, d, 3)
            br = carteira_da_nota(composto_hoje, dados.setores, NOMES - len(escolha_eua), POR_SETOR)
            fixos[d] = pd.Series(1.0 / NOMES, index=[*br.index, *escolha_eua])
            fixos[d] = fixos[d] / fixos[d].sum()
        curvas = {BRACO: simular(alvos, juntos, CUSTO_BPS), FIXA: simular(fixos, juntos, CUSTO_BPS)}
        comeco = curvas[BRACO].index[0]
        valores = {nome: _recorte(curva, comeco) for nome, curva in curvas.items()}
        valores |= {COMPOSTO: _recorte(composto.valores, comeco), CONSERVADORA: _recorte(conservadora.valores, comeco),
                    "BOVA11": _recorte(refs["BOVA11"].valores, comeco)}
        indice = valores[COMPOSTO].index
        valores["IVVB11"] = _recorte(precos_eua["IVVB11"].reindex(indice).ffill(), comeco)
        valores = {nome: serie.reindex(indice) for nome, serie in valores.items()}
        rf = market_data.serie_rf_diaria(indice, cdi, (1.0 + float(cdi.iloc[-1])) ** 252 - 1.0)
        comparadas = {nome: valores[nome] for nome in (BRACO, FIXA, COMPOSTO, CONSERVADORA)}
        tabela = tabela_do_periodo(comparadas, valores["BOVA11"], rf, N_DSR)
        for nome in (*comparadas, "BOVA11", "IVVB11"):
            tabela.loc[nome, "CAGR"] = cagr(valores[nome].dropna())
            tabela.loc[nome, "MDD"] = drawdown(valores[nome].dropna())[1]
            tabela.loc[nome, "vol"] = valores[nome].pct_change().std() * np.sqrt(252)
        for nome in (BRACO, FIXA):
            tabela.loc[nome, "P[> Composto]"] = psr_da_diferenca(valores[nome], valores[COMPOSTO])
            tabela.loc[nome, "P[> Conservadora]"] = psr_da_diferenca(valores[nome], valores[CONSERVADORA])
        p_composto.append(float(tabela.loc[BRACO, "P[> Composto]"]))
        p_conservadora.append(float(tabela.loc[BRACO, "P[> Conservadora]"]))
        p_bova.append(float(tabela.loc[BRACO, "P aritmético"]))
        dsr.append(float(tabela.loc[BRACO, "DSR"]))
        print(f"=== Phase {fase}: {indice[0]:%d/%m/%Y} to {indice[-1]:%d/%m/%Y}, {len(datas)} rebalances ===")
        print(tabela.to_string(float_format=lambda valor: f"{valor:.3f}"))
        print(f"BDRs held by {BRACO} per rebalance: mean {np.mean(bdrs):.1f}, min {min(bdrs)}, max {max(bdrs)}; "
              f"by date {bdrs}")
        print(f"Target-to-target turnover per year: {BRACO} {giro_dos_alvos(alvos):.2f}, {COMPOSTO} "
              f"{giro_dos_alvos(proprios):.2f}")
        print(_year_table({**comparadas, "BOVA11": valores["BOVA11"], "IVVB11": valores["IVVB11"]})
              .to_string(float_format=lambda valor: f"{valor:.1%}"))
        if fase == FASES[0]:
            ultima = datas[-1]
            print(f"{BRACO} latest picks ({ultima:%d/%m/%Y}): " + ", ".join(sorted(alvos[ultima].index)))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    passa = strategy_gate([*p_composto, *p_conservadora], p_bova, dsr)
    print(f"\nGate, all >= {P_MIN} in every phase:")
    print(f"{BRACO}: P[> Composite] {np.round(p_composto, 3)}, P[> Conservadora] {np.round(p_conservadora, 3)}, "
          f"P[> BOVA11] {np.round(p_bova, 3)}, DSR {np.round(dsr, 3)}: {'PASSES' if passa else 'does not pass'}")
    print(f"\nDone in {time.perf_counter() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    raise SystemExit(main())
