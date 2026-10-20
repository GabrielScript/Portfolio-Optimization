"""
diario.py - Diário do paper trading (``paper/diario.csv``): o valor do satélite e do BOVA11 em cada fechamento,
gravado no dia em que a rotina o viu, para avisar quando o lago muda um dado passado.

O ``paper/carteiras.csv`` guarda só as decisões; a curva sai dele e dos preços atuais do lago. Se o lago muda um preço
passado (provento lançado depois, correção de um evento), a curva de um fechamento já visto muda sem que nada no
registro mude. O diário só cresce: a cada rodada entram os fechamentos novos ("novo") e, para cada fechamento já
gravado cujo valor recalculado mudou, uma linha nova "revisao" com o valor de agora (a última linha de cada fechamento
é a referência da rodada seguinte, então a mesma mudança avisa uma vez só). Satélite e BOVA11 valem 1,0 no fechamento
do 1º registro; razão = satélite / BOVA11 e pico = maior razão até o fechamento (o stop relativo olha os dois).
"""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from core.paper import NUCLEO, curva_do_registro

COLUNAS = ["fechamento", "satelite", "bova", "razao", "pico", "precos_sha256", "gravado_em", "tipo"]
MEDIDAS = ["satelite", "bova"]      # razão e pico saem delas
# o mesmo arquivo de preços dá a mesma curva bit a bit; um provento ou correção mexe na 3ª casa ou antes
TOLERANCIA_RELATIVA = 1e-6


def ler_diario(arquivo: Path) -> pd.DataFrame:
    """O diário com o fechamento como Timestamp; vazio (mesmas colunas) se o arquivo ainda não existe."""
    if not arquivo.exists():
        return pd.DataFrame(columns=COLUNAS).astype({"fechamento": "datetime64[ns]"})
    return pd.read_csv(arquivo, parse_dates=["fechamento"], dtype={"precos_sha256": str})


def curvas_do_paper(carteiras: pd.DataFrame, precos: pd.DataFrame) -> pd.DataFrame:
    """Satélite (``curva_do_registro``, bruta, a do stop) e BOVA11 desde o 1º registro, os dois em 1,0 nele."""
    satelite = curva_do_registro(carteiras, precos)
    bova = precos[NUCLEO].ffill().reindex(satelite.index)
    bova = bova / bova.iloc[0]
    razao = satelite / bova
    return pd.DataFrame({"satelite": satelite, "bova": bova, "razao": razao, "pico": razao.cummax()})


def _gravar(arquivo: Path, curvas: pd.DataFrame, precos_sha256: str, agora: datetime, tipo: str) -> None:
    novo = not arquivo.exists()
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    with arquivo.open("a", encoding="utf-8", newline="") as saida:
        escritor = csv.writer(saida, lineterminator="\n")
        if novo:
            escritor.writerow(COLUNAS)
        for fechamento, linha in curvas.iterrows():
            escritor.writerow([fechamento.date().isoformat(), *(repr(float(linha[c])) for c in curvas.columns),
                               precos_sha256, agora.isoformat(timespec="seconds"), tipo])


def atualizar_diario(arquivo: Path, carteiras: pd.DataFrame, precos: pd.DataFrame, precos_sha256: str,
                     agora: datetime) -> list[pd.Timestamp]:
    """Grava os fechamentos novos e as revisões; devolve os fechamentos revistos (vazio = o lago não mudou o
    passado). ``precos_sha256``: hash do arquivo de preços lido, para saber depois de que versão saiu cada linha."""
    if carteiras.empty:
        return []
    curvas = curvas_do_paper(carteiras, precos)
    ultimas = ler_diario(arquivo).drop_duplicates("fechamento", keep="last").set_index("fechamento")
    agora_nos_vistos = curvas.reindex(ultimas.index)[MEDIDAS]
    mudou = ~np.isclose(agora_nos_vistos, ultimas[MEDIDAS].astype(float), rtol=TOLERANCIA_RELATIVA, atol=0.0,
                        equal_nan=True).all(axis=1)
    revistos = list(ultimas.index[mudou])
    _gravar(arquivo, curvas.reindex(revistos), precos_sha256, agora, "revisao")
    _gravar(arquivo, curvas[~curvas.index.isin(ultimas.index)], precos_sha256, agora, "novo")
    return revistos


def sha256_do_arquivo(caminho: Path) -> str:
    """Hash do arquivo de preços lido (o mesmo que o ``sha256sum``), em blocos de 1 MB."""
    resumo = hashlib.sha256()
    with caminho.open("rb") as entrada:
        for bloco in iter(lambda: entrada.read(1 << 20), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def texto_das_revisoes(revistos: list[pd.Timestamp]) -> str:
    """A nota do runs.csv e do aviso quando ``atualizar_diario`` achou fechamentos revistos."""
    return (f"lago mudou {len(revistos)} fechamento(s) já gravado(s) no diário, de {min(revistos):%d/%m/%Y} a "
            f"{max(revistos):%d/%m/%Y}")
