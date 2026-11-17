"""
teste_a_avaliar.py - The reading of Test A (amendment of 2026-11-17, ``docs/preregistro-sentimento-2026-10-23.md``).

    python -X utf8 scripts/teste_a_avaliar.py fase 0       # Composite with and without the veto, one phase per process
    python -X utf8 scripts/teste_a_avaliar.py eventos      # descriptive event study (no gate)

Inputs: ``docs/dados/teste-a-2026-11-17/documentos.parquet`` and ``respostas.jsonl`` from ``teste_a_pontuar.py``; the
frozen export and the engine of ``scripts/curvas_pos_idade.py`` (the Composite must hit its DECISOES 360 anchor first).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core.teste_a import (colunas_da_resposta, emissores_vetados, estrategia_com_veto,  # noqa: E402
                          excessos_apos_documentos, faixa_de_p, faixa_do_tom)

PASTA = RAIZ / "docs" / "dados" / "teste-a-2026-11-17"
HORIZONTES = (21, 63)
PERIODOS = (("2010-2019", 2010, 2019), ("2020-2025", 2020, 2025), ("2026", 2026, 2026))


def documentos_pontuados() -> pd.DataFrame:
    """One row per (filing, candidate issuer) with P(bad), P(good), intensity and kind from its state's answer."""
    docs = pd.read_parquet(PASTA / "documentos.parquet")
    respostas = pd.read_json(PASTA / "respostas.jsonl", lines=True)
    colunas = pd.DataFrame([colunas_da_resposta(r) for r in respostas["answers"]], index=respostas["chave"])
    pontuados = docs.join(colunas, on="chave", how="left")
    faltam = int(pontuados["p_ruim"].isna().sum())
    if faltam:
        raise SystemExit(f"{faltam} filings have no answer: finish teste_a_pontuar.py first")
    return pontuados


def _carregar():
    from core.crescimento_lucro import nota_de_crescimento  # noqa: PLC0415
    from core.qubo_universo import composite_score  # noqa: PLC0415
    from core.sinais import conservadora  # noqa: PLC0415
    from core.views import nota_media_dos_escores  # noqa: PLC0415
    from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor  # noqa: PLC0415
    from scripts.fermionica_grade_avaliar import memoized  # noqa: PLC0415
    from scripts.qpo_montar import FOLDER  # noqa: PLC0415

    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = avaliar_valor.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    balancos, _ = avaliar_valor.carregar_balancos(FOLDER.parent / "2026-10-24-cvm")
    nota = memoized(composite_score(lambda mes, c: conservadora(mensal.retornos, mensal.proventos, mes, c),
                                    nota_media_dos_escores(*avaliar_valor.notas(balancos, mensal).values()),
                                    nota_de_crescimento(balancos.fundamentos)))
    return dados, nota


def fase(numero: int) -> int:
    """Both curves of one phase, the anchor check, the gate's P and MDD, and how many picks the veto changed."""
    from core.backtest import Congelamento, walk_forward_dinamico  # noqa: PLC0415
    from core.composto import estrategia_por_nota  # noqa: PLC0415
    from core.metrics import cagr  # noqa: PLC0415
    from core.robustez import psr_da_diferenca  # noqa: PLC0415
    from scripts.avaliar_robustez import _acoes  # noqa: PLC0415
    from scripts.avaliar_universo_dinamico import CFG_BT  # noqa: PLC0415
    from scripts.curvas_pos_idade import ANCHOR  # noqa: PLC0415
    from scripts.qpo_montar import LAST_SESSION  # noqa: PLC0415

    docs = documentos_pontuados()
    dados, nota = _carregar()
    acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[numero:])
    pregoes = acoes.index
    vetos: dict[pd.Timestamp, set[str]] = {}

    def vetados_em(t: pd.Timestamp) -> set[str]:
        vetos[t] = emissores_vetados(docs, pregoes, t)
        return vetos[t]

    def rodar(estrategia, nome):
        return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

    sem = rodar(estrategia_por_nota(nota, dados.setores), "Composite")
    if abs(cagr(sem.valores.dropna()) - ANCHOR[numero]) > 5e-5:
        raise SystemExit(f"phase {numero}: anchor failed, CAGR {cagr(sem.valores.dropna()):.5f} vs {ANCHOR[numero]}")
    com = rodar(estrategia_com_veto(nota, dados.setores, vetados_em), "Composite + veto")
    curvas = pd.DataFrame({"Composite": sem.valores, "Composite + veto": com.valores}).rename_axis("date")
    curvas.reset_index().to_parquet(PASTA / f"curvas-fase{numero}.parquet", index=False)

    escolhidos = sem.pesos_alvo.gt(0)
    trocas = int(((escolhidos != com.pesos_alvo.reindex_like(escolhidos).fillna(0).gt(0)).sum(axis=1) > 0).sum())
    nomes_trocados = int((escolhidos & ~com.pesos_alvo.reindex_like(escolhidos).fillna(0).gt(0)).sum().sum())
    mdd = {nome: float((s / s.cummax() - 1).min()) for nome, s in curvas.items()}
    print(f"phase {numero}: Composite {cagr(sem.valores.dropna()):.4f} (anchor ok), "
          f"+ veto {cagr(com.valores.dropna()):.4f}; P[veto > base] {psr_da_diferenca(com.valores, sem.valores):.3f} "
          f"(log {psr_da_diferenca(com.valores, sem.valores, log=True):.3f}); MDD {mdd['Composite']:.3f} -> "
          f"{mdd['Composite + veto']:.3f}; rebalances changed {trocas} of {len(escolhidos)}, names dropped "
          f"{nomes_trocados}; window {curvas.index[0]:%Y-%m-%d}..{curvas.index[-1]:%Y-%m-%d}")
    return 0


def _t_por_data(quadro: pd.DataFrame) -> tuple[float, float, int]:
    """Mean excess, its t with the events averaged per entry date first (clustered by date), and the date count."""
    por_data = quadro.groupby("entrada")["excesso"].mean()
    if len(por_data) < 3:
        return float(quadro["excesso"].mean()), float("nan"), len(por_data)
    return float(quadro["excesso"].mean()), float(por_data.mean() / (por_data.std(ddof=1) / np.sqrt(len(por_data)))), \
        len(por_data)


def eventos() -> int:
    """Excess returns after the filings by tone x intensity cell and by P(bad) / P(good) band, three periods apart."""
    docs = documentos_pontuados()
    dados, _nota = _carregar()
    precos = dados.precos.ffill(limit=5)
    universo = dados.universo.assign(month=lambda q: pd.to_datetime(q["month"]))
    elegiveis = universo.groupby("month")["security"].apply(list).to_dict()
    docs["faixa_tom"] = [faixa_do_tom(r, b, i) for r, b, i in zip(docs["p_ruim"], docs["p_bom"], docs["intensidade"])]
    docs["faixa_ruim"] = docs["p_ruim"].map(faixa_de_p)
    docs["faixa_bom"] = docs["p_bom"].map(faixa_de_p)
    unicos = docs[["emissor", "data"]].drop_duplicates()
    linhas = []
    for horizonte in HORIZONTES:
        excesso = excessos_apos_documentos(precos, elegiveis, unicos, horizonte)
        juntos = docs.merge(excesso, on=["emissor", "data"], how="inner")
        juntos["ano"] = juntos["entrada"].dt.year
        for periodo, de, ate in PERIODOS:
            parte = juntos[(juntos["ano"] >= de) & (juntos["ano"] <= ate)]
            for eixo in ("faixa_tom", "faixa_ruim", "faixa_bom"):
                for faixa, grupo in parte.groupby(eixo):
                    media, t, datas = _t_por_data(grupo)
                    linhas.append({"horizonte": horizonte, "periodo": periodo, "eixo": eixo, "faixa": faixa,
                                   "documentos": len(grupo), "datas": datas, "excesso": media, "t": t})
    tabela = pd.DataFrame(linhas)
    tabela.to_parquet(PASTA / "eventos.parquet", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print(tabela.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    comando = sys.argv[1] if len(sys.argv) > 1 else ""
    if comando == "fase":
        raise SystemExit(fase(int(sys.argv[2])))
    if comando == "eventos":
        raise SystemExit(eventos())
    raise SystemExit(__doc__)
