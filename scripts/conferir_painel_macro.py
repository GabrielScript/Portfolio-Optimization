"""
Conferência manual, COM REDE, do painel macro e do retrato de hoje (spec §12, "fora da suíte").

Uso: .venv/Scripts/python scripts/conferir_painel_macro.py [AAAA-MM-DD]

Imprime o retrato, as fontes que falharam, a cobertura de cada fator e os
últimos meses do painel. Compare o retrato com o boletim Focus da semana
(https://www.bcb.gov.br/publicacoes/focus) e com o Tesouro Direto.
Sai com código 1 se algum fator ficar indisponível ou com menos de 200 meses.
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import fatores_macro  # noqa: E402

MESES_MINIMOS = 200      # 2005-12 a 2026-09 são 250 meses; buracos isolados são tolerados


def main() -> int:
    hoje = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    inicio = time.perf_counter()
    dados = fatores_macro.baixar_dados_macro(hoje)
    print(f"Download: {time.perf_counter() - inicio:.1f} s")
    for fonte, motivo in dados.erros.items():
        print(f"  FONTE COM FALHA  {fonte}: {motivo}")

    retrato = fatores_macro.montar_retrato(dados, hoje)
    print("\nRetrato de hoje")
    for ind in retrato.indicadores:
        print(f"  {ind.nome:<32} {ind.valor:>9.4f} {ind.unidade:<7} {ind.data:%Y-%m-%d}  ({ind.fonte})")
    for nome, motivo in retrato.indisponiveis.items():
        print(f"  {nome:<32} INDISPONÍVEL: {motivo}")
    for nome, exp in retrato.expectativas._asdict().items():
        if exp is not None:
            print(f"  Focus 12m {nome:<7} mediana {exp.mediana:.4f}  desvio {exp.desvio:.4f}  pesquisa {exp.data:%Y-%m-%d}")

    painel = fatores_macro.montar_painel(dados, hoje)
    fatores = painel.fatores
    print(f"\nPainel: {fatores.index.min()} a {fatores.index.max()} ({len(fatores)} meses)")
    print(fatores.describe().T[["count", "mean", "std", "min", "max"]].to_string(float_format="{:.4f}".format))
    print("\nÚltimos 6 meses")
    print(fatores.tail(6).to_string(float_format="{:.4f}".format))
    for aviso in painel.avisos:
        print(f"  AVISO {aviso}")

    curtos = [f for f in fatores.columns if fatores[f].notna().sum() < MESES_MINIMOS]
    if painel.fatores_indisponiveis or curtos:
        print(f"\nFALHOU: indisponíveis={list(painel.fatores_indisponiveis)}, curtos={curtos}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
