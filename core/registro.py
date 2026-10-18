"""
registro.py - Registro do paper trading (S5): a carteira de cada data, num CSV versionado que só cresce.

Uma linha por papel: ``data,ticker,peso,preco`` (peso = fração do patrimônio; preço do fechamento usado). O arquivo
só recebe linhas no fim e nunca é reescrito, então o histórico do git prova que nenhuma carteira foi trocada depois
de vista. Dele saem os retornos do satélite para ``core.risco.pode_negociar`` e ``core.risco.stop_relativo``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

CABECALHO = "data,ticker,peso,preco\n"


def ultima_data(arquivo: Path) -> date | None:
    """Data da última carteira registrada, ou None se o arquivo não existe ou está vazio."""
    if not arquivo.exists():
        return None
    linhas = arquivo.read_text(encoding="utf-8").splitlines()[1:]
    return date.fromisoformat(linhas[-1].split(",", 1)[0]) if linhas else None


def registrar_carteira(data: date, pesos: pd.Series, precos: pd.Series, arquivo: Path) -> None:
    """Acrescenta a carteira de ``data`` ao fim de ``arquivo``. Recusa data igual ou anterior à última registrada."""
    ultima = ultima_data(arquivo)
    if ultima is not None and data <= ultima:
        raise ValueError(f"Carteira de {data:%d/%m/%Y} não entra: a última registrada é de {ultima:%d/%m/%Y}.")
    if (pesos < 0).any() or pesos.sum() > 1.0 + 1e-6:
        raise ValueError(f"Pesos devem ser >= 0 e somar no máximo 100% (soma {pesos.sum():.4f}).")
    cotacoes = precos.reindex(pesos.index).astype(float)
    if not (cotacoes > 0).all():
        raise ValueError(f"Preço ausente ou inválido: {', '.join(cotacoes.index[~(cotacoes > 0)])}.")
    linhas = "".join(f"{data.isoformat()},{ticker},{float(peso)!r},{float(cotacoes[ticker])!r}\n"
                     for ticker, peso in pesos.items())
    novo = not arquivo.exists()
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    with arquivo.open("a", encoding="utf-8", newline="\n") as saida:
        saida.write((CABECALHO if novo else "") + linhas)
