"""
qml_g2_avaliar.py - the pre-registered G2 run of the QML arm (test branch only).

Spec: ``docs/superpowers/specs/2026-11-09-qml-g2-design.md`` (written before this script ran).

Arms:
  * Composite              - the benchmark, unchanged.
  * Composite Quantum      - the QML tilt inside the Composite's 15 names, Meio-Kelly, with the signal
                             looked up on the decision session itself (daily predictions; the rebalance
                             stays every 63 sessions).
  * Composite Quantum + BDR- the same engine over the Composite's 15 B3 names union the eligible US BDRs,
                             the BDRs capped as one "EUA" sector. Descriptive: ~12 years, needs IR >= 0.58.

    .venv-qpo/Scripts/python -X utf8 scripts/qml_g2_avaliar.py [--diario] [--bdr] [--fases 0]

Writes docs/qml-g2-<mode>-2026-11-09.txt and docs/dados/qml-g2-<mode>.csv.
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core.backtest import ResultadoBacktest  # noqa: E402
from core.cdi_lago import daily_cdi  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.config import DIAS_UTEIS_ANO  # noqa: E402
from core.qml import QMLConfig, QMLPredictor, qml_strategy  # noqa: E402
from core.qpo import CFG_MAX_SHARPE  # noqa: E402
from core.sinais import NOMES, POR_SETOR, carteira_da_nota  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, qubo_universo_avaliar  # noqa: E402
from scripts.avaliar_robustez import _acoes, referencias  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT, CFG_KELLY  # noqa: E402
from scripts.curvas_ao_vivo import ARQUIVOS, LAGO, congelar  # noqa: E402
from scripts.qml_12_3_avaliar import BDRS, precos_das_bdrs, universo_com_bdrs  # noqa: E402
from scripts.qml_avaliar import (ProbabilidadesCongeladas, _gravar_cache, _ler_cache, _linha,  # noqa: E402
                                 _probabilidades, _rodar, _tabela, seletor_do_composite)

CACHE_DIARIO = RAIZ / "docs" / "dados" / "qml-probabilidades-g2-diario.parquet"
CACHE_BDR = RAIZ / "docs" / "dados" / "qml-probabilidades-g2-bdr.parquet"
SETOR_BDR = "EUA"


def probabilidades_de_todas_as_sessoes(predictor: QMLPredictor, acoes: pd.DataFrame,
                                       cache: Path, janela: int = CFG_BT.janela_treino) -> pd.DataFrame:
    """P(up) for **every** session from the first decision on, causally and memoized by date.

    This is the G2 arm's change: the decision session's own prediction is available, so the signal is
    never up to 62 sessions stale. Only the dates from ``janela`` on are predicted (nothing before the
    first decision), and the cache is date-keyed like the original one.
    """
    datas = list(acoes.index[janela:])
    quadro = _ler_cache(cache, predictor.config)
    faltam = [data for data in datas if data not in quadro.index]
    if faltam:
        print(f"  QML-G2: {len(faltam)} sessions to predict (cache has {len(quadro.index)})", flush=True)
    for indice, data in enumerate(faltam, start=1):
        linha = predictor.predict(data)
        if not linha.empty:
            nova = linha.to_frame().T
            nova.index = pd.DatetimeIndex([data])
            quadro = pd.concat([quadro[quadro.index != data], nova])
        if indice % 100 == 0 or indice == len(faltam):
            _gravar_cache(quadro.sort_index(), cache, predictor.config)
            print(f"  QML-G2: {indice}/{len(faltam)} sessions predicted", flush=True)
    _gravar_cache(quadro.sort_index(), cache, predictor.config)
    return quadro.sort_index()


def seletor_composite_mais_bdr(nota, setores: Mapping[str, str], bdrs: Sequence[str] = BDRS,
                               nomes: int = NOMES, por_setor: int = POR_SETOR):
    """The Composite's 15 B3 names plus every BDR that has a close in the training window.

    The BDRs are not ranked here: the QML conviction and the fixed rules do the allocation, with all of
    them sharing the "EUA" sector cap. Kept deliberately simple so the arm is reproducible.
    """
    def _select(data: pd.Timestamp, treino: pd.DataFrame) -> list[str]:
        mes = data.to_period("M").to_timestamp()
        b3 = [papel for papel in treino.columns if papel not in bdrs]
        base = list(carteira_da_nota(nota(mes, b3).dropna(), setores, nomes, por_setor).index)
        eua = [papel for papel in bdrs if papel in treino.columns]
        return base + eua
    return _select


def _carregar(entradas: Path):
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = entradas
    qubo_universo_avaliar.FOLDER = qubo_universo_avaliar.PASTA_CVM = entradas
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(entradas)
    nota, _ = qubo_universo_avaliar._value_score(mensal, com_crescimento=True)
    ultima = dados.precos.index.max()
    cdi = daily_cdi(ANOS_DE_CDI, ultima.date(), path=entradas / "cdi_daily.parquet")
    return dados, mensal, nota, ultima, cdi


def main(argv: list[str]) -> int:
    inicio = time.perf_counter()
    diario = "--diario" in argv
    bdr = "--bdr" in argv
    fase = int(argv[argv.index("--fases") + 1]) if "--fases" in argv else 0
    lago = Path(argv[argv.index("--lago") + 1]) if "--lago" in argv else LAGO
    modo = "diario" if diario else ("bdr" if bdr else "base")
    saida_txt = RAIZ / "docs" / f"qml-g2-{modo}-2026-11-09-f{fase}.txt"
    saida_csv = RAIZ / "docs" / "dados" / f"qml-g2-{modo}-f{fase}.csv"
    with tempfile.TemporaryDirectory(prefix="qml-g2-") as temporaria:
        entradas = Path(temporaria)
        somas = congelar(lago, entradas, ARQUIVOS)
        dados, mensal, nota, ultima, cdi = _carregar(entradas)
        if bdr:
            bdrs = precos_das_bdrs(ultima.date()).reindex(dados.precos.index)
            setores = {**dados.setores, **{papel: SETOR_BDR for papel in bdrs.columns}}
            precos = dados.precos.join(bdrs)
            universo = universo_com_bdrs(dados.universo, bdrs)
        else:
            setores = dict(dados.setores)
            precos = dados.precos
            universo = dados.universo
        retornos = precos.ffill(limit=5).pct_change().iloc[1:]
        acoes = _acoes(precos).iloc[fase:]
        if len(acoes) <= CFG_BT.janela_treino:
            raise ValueError("not enough sessions for the training window")
        predictor = QMLPredictor(retornos, QMLConfig(), log=print)
        if diario:
            print(f"G2 daily: predicting every session from {acoes.index[CFG_BT.janela_treino]:%d/%m/%Y}",
                  flush=True)
            quadro = probabilidades_de_todas_as_sessoes(predictor, acoes, CACHE_DIARIO)
        else:
            cache = CACHE_BDR if bdr else None
            quadro = _probabilidades(predictor, acoes, CFG_BT, cache)
        fonte = ProbabilidadesCongeladas(quadro)
        cfg_ms = replace(CFG_MAX_SHARPE, taxa_livre_risco=float((1 + cdi.iloc[-1]) ** DIAS_UTEIS_ANO - 1))
        seletor = seletor_do_composite(nota, setores)
        seletor_bdr = seletor_composite_mais_bdr(nota, setores)
        universo_bench = dados.universo if bdr else universo   # the Composite never picks a BDR
        curvas: dict[str, ResultadoBacktest] = {
            "Composite": _rodar(acoes, universo_bench, estrategia_por_nota(nota, dados.setores),
                                "Composite", inicio, CFG_BT),
            "Composite Quantum": _rodar(
                acoes, universo, qml_strategy(fonte, CFG_KELLY, setores, cdi, selector=seletor, tilt=1.0),
                "Composite Quantum", inicio, CFG_BT),
            "Composite Quantum (Máx. Sharpe)": _rodar(
                acoes, universo,
                qml_strategy(fonte, cfg_ms, setores, cdi, selector=seletor, tilt=1.0),
                "Composite Quantum (Máx. Sharpe)", inicio, CFG_BT),
        }
        if bdr:
            curvas["Composite Quantum + BDR"] = _rodar(
                acoes, universo,
                qml_strategy(fonte, CFG_KELLY, setores, cdi, selector=seletor_bdr, tilt=1.0),
                "Composite Quantum + BDR", inicio, CFG_BT)
        bova = referencias(precos, CFG_BT)["BOVA11"]
        rf = cdi.reindex(next(iter(curvas.values())).valores.index).fillna(0.0)
        tabela = _tabela(curvas, bova.valores, rf, curvas["Composite"].valores)
        tabela.loc["BOVA11"] = _linha(bova.valores, bova.valores, rf)
        tabela.insert(0, "Fase", fase)
        with saida_txt.open("w", encoding="utf-8", newline="\n") as saida, contextlib.redirect_stdout(saida):
            print(f"G2 {modo} (pre-registered: docs/superpowers/specs/2026-11-09-qml-g2-design.md). "
                  f"Lake exports sha256: {somas}")
            print(f"First decision {acoes.index[CFG_BT.janela_treino]:%d/%m/%Y}; "
                  f"last session {acoes.index[-1]:%d/%m/%Y}; cache rows {len(quadro)}.")
            if bdr:
                print(f"BDR list: {len(BDRS)} names of ativos_eua, sector '{SETOR_BDR}'; "
                      f"eligible in this window: {len([p for p in BDRS if p in precos.columns])}.")
            print(tabela.to_string(float_format=lambda x: f"{x:.4f}"))
        tabela.to_csv(saida_csv, encoding="utf-8", lineterminator="\n")
        print(tabela.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n{saida_txt.relative_to(RAIZ)} and {saida_csv.relative_to(RAIZ)} written "
          f"({time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
