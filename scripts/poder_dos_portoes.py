"""
Poder dos portões do S2 por simulação (PLAN, backtest): IC plantado no painel mensal do lago, sem rede.

Uso: .venv/Scripts/python -X utf8 scripts/poder_dos_portoes.py

Pré-registro (2026-10-18, escrito e commitado antes de rodar):
- Painel: `data/b3lake/monthly_total_returns.parquet` (retorno total de cada mês) dos papéis do
  `universe_monthly.parquet` daquele mês (regra D1, deslistadas incluídas), de 2008-01 (o S2 precisa de 36 meses de
  vol desde 2005) até o último mês fechado do lago. O painel entra só como ruído de corte transversal: o sinal é
  plantado (`core.poder.sinal_plantado`), nenhum sinal real é olhado.
- IC verdadeiro plantado: 0; 0,02; 0,03; 0,05. IC do mês constante ou sorteado em torno da média com desvio de 0,05 e
  de 0,10 (o IC de um sinal real varia de mês a mês). Janelas: a inteira, os últimos 120 e os últimos 60 meses.
  2.000 simulações por célula, semente fixa.
- Portões, em ordem: "IC ≥ 0,03 e t ≥ 2" (o do PLAN hoje), "IC ≥ 0,02 e t ≥ 2", "t ≥ 2" (t de Student do IC médio
  mensal).
- Critério do PLAN: poder ≥ 80% com IC 0,03 e falso positivo ≤ 5% com IC 0. Escolha: o portão do S2 é o 1º da lista
  que cumpre os dois na janela inteira com desvio de 0,10 (o caso mais duro dos três); nenhum = o S2 não pode ser
  decidido pelo IC nesta janela, e isso vai ao DECISOES antes de qualquer sinal.
- Expectativa registrada: o portão de hoje fica perto de 50% de poder com IC 0,03 (a média amostral passa do
  próprio valor verdadeiro metade das vezes), então o escolhido deve ser o 2º; em 60 meses nenhum deve ter poder.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.poder import Portao, taxa_de_aprovacao  # noqa: E402

PASTA_LAGO = Path(__file__).resolve().parent.parent / "data" / "b3lake"
INICIO = "2008-01-01"
ICS = (0.0, 0.02, 0.03, 0.05)
DESVIOS = (0.0, 0.05, 0.10)
JANELAS = {"inteira": None, "últimos 120 meses": 120, "últimos 60 meses": 60}
SIMULACOES = 2_000
SEMENTE = 20261018
PORTOES = (Portao("IC ≥ 0,03 e t ≥ 2", ic_minimo=0.03, t_minimo=2.0),
           Portao("IC ≥ 0,02 e t ≥ 2", ic_minimo=0.02, t_minimo=2.0),
           Portao("t ≥ 2", ic_minimo=-1.0, t_minimo=2.0))
PODER_MINIMO, FALSO_POSITIVO_MAXIMO = 0.80, 0.05
IC_DO_PODER, DESVIO_DA_ESCOLHA = 0.03, 0.10


def escolher_portao(taxas: Mapping[float, Mapping[str, float]]) -> str | None:
    """O 1º portão de ``PORTOES`` com poder ≥ 80% em IC 0,03 e falso positivo ≤ 5% em IC 0 (``taxas[ic][portão]``)."""
    for portao in PORTOES:
        if taxas[IC_DO_PODER][portao.nome] >= PODER_MINIMO and taxas[0.0][portao.nome] <= FALSO_POSITIVO_MAXIMO:
            return portao.nome
    return None


def carregar_painel(pasta: Path = PASTA_LAGO, inicio: str = INICIO) -> pd.DataFrame:
    """Retorno total de cada mês (linha) dos papéis do universo daquele mês; fora do universo, vazio. Sem o mês aberto."""
    retornos = pd.read_parquet(pasta / "monthly_total_returns.parquet")
    retornos = retornos.set_index(pd.DatetimeIndex(pd.to_datetime(retornos.pop("month")), name=None))
    universo = pd.read_parquet(pasta / "universe_monthly.parquet", columns=["month", "security"])
    universo["month"] = pd.to_datetime(universo["month"])
    dentro = universo.assign(dentro=True).pivot(index="month", columns="security", values="dentro")
    dentro = dentro.reindex(columns=retornos.columns).fillna(False).astype(bool)
    ate = json.loads((pasta / "manifest.json").read_bytes().decode("utf-8", errors="replace"))["data_through"]
    aberto = pd.Timestamp(ate).to_period("M").to_timestamp()
    meses = retornos.index[(retornos.index >= inicio) & (retornos.index < aberto)].intersection(dentro.index)
    painel = retornos.loc[meses].where(dentro.loc[meses])
    return painel.dropna(axis=1, how="all")


def simular(painel: pd.DataFrame, simulacoes: int = SIMULACOES,
            semente: int = SEMENTE) -> dict[tuple[str, float], dict[float, dict[str, float]]]:
    """(janela, desvio) -> IC verdadeiro -> portão -> taxa de aprovação."""
    gerador = np.random.default_rng(semente)
    taxas = {}
    for janela, meses in JANELAS.items():
        recorte = painel if meses is None else painel.iloc[-meses:]
        for desvio in DESVIOS:
            taxas[janela, desvio] = {ic: taxa_de_aprovacao(recorte, ic, PORTOES, simulacoes, gerador, desvio)
                                     for ic in ICS}
    return taxas


def imprimir(painel: pd.DataFrame, taxas: Mapping[tuple[str, float], Mapping[float, Mapping[str, float]]]) -> None:
    papeis = painel.notna().sum(axis=1)
    print(f"Painel: {painel.index[0]:%m/%Y} a {painel.index[-1]:%m/%Y}, {len(painel)} meses; papéis por mês: "
          f"mín. {papeis.min()}, mediana {papeis.median():.0f}, máx. {papeis.max()}")
    print(f"{SIMULACOES} simulações por célula, semente {SEMENTE}. Taxa de aprovação de cada portão:")
    for (janela, desvio), por_ic in taxas.items():
        tabela = pd.DataFrame(por_ic).T.rename_axis("IC verdadeiro")
        print(f"\n--- janela {janela} ({len(painel) if JANELAS[janela] is None else JANELAS[janela]} meses), "
              f"desvio do IC do mês {desvio:.2f} ---")
        print(tabela.to_string(float_format=lambda x: f"{x:.1%}"))
    escolhido = escolher_portao(taxas["inteira", DESVIO_DA_ESCOLHA])
    print(f"\nCritério: poder ≥ {PODER_MINIMO:.0%} em IC {IC_DO_PODER} e falso positivo ≤ {FALSO_POSITIVO_MAXIMO:.0%} "
          f"em IC 0, janela inteira, desvio {DESVIO_DA_ESCOLHA:.2f}.")
    print(f"Portão do S2: {escolhido or 'nenhum cumpre: o S2 não pode ser decidido pelo IC nesta janela'}")


def main() -> int:
    inicio = time.perf_counter()
    painel = carregar_painel()
    imprimir(painel, simular(painel))
    print(f"(tempo total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
