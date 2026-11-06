"""Shadow judges (spec amendments 3 and 5, DECISOES 270 and 272): Liquid AI's d1.1 (the chosen judge) and d1-ultra
(logged beside it so recall can be re-read on real events), direct Liquid API, key ``LIQUID_API_KEY``. Each company item
of a model call gets one request per judge with two yes/no questions; macro items never go to a judge. Logged only
(``simulador/sentinela_juizes.csv``): nothing here changes a verdict, an order or money. A missing key or a failure
leaves the Sentinela running without them."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass

import requests

from core.sentinela.itens import MACRO, Item

URL = "https://api.liquid.ai/decisions/v1/systemone"
JUIZES = ("d1.1", "d1-ultra")
PRAZO_S = 60
PERGUNTAS = {
    "material": {"type": "noul", "instructions": (
        "Is this filing new information likely to move the company's share price, beyond routine disclosure? Routine "
        "means scheduled reports and calendars, recurring dividends or interest on equity, routine board or officer "
        "changes, internal reorganizations and debt issuance.")},
    "nova": {"type": "noul", "instructions": (
        "Is this item new information about the company, rather than a repeat or a re-presentation of facts it had "
        "already disclosed?")},
}
COLUNAS = ["juiz", "item", "ticker", "p_material", "p_nova", "segundos", "tokens", "custo", "modelo", "erro"]


@dataclass(frozen=True)
class Juizo:
    juiz: str
    item: str
    ticker: str
    p_material: float | None
    p_nova: float | None
    segundos: float
    tokens: int | None = None
    custo: float | None = None
    modelo: str | None = None
    erro: str | None = None

    def linha(self) -> dict:
        return asdict(self)


def _estado(item: Item, empresa: str, setor: str) -> str:
    return f"Empresa: {empresa or item.ticker} ({item.ticker}), setor {setor or 'sem setor'}. {item.texto}"


def _perguntar(juiz: str, item: Item, empresa: str, setor: str, chave: str, post: Callable) -> Juizo:
    inicio = time.perf_counter()
    try:
        resposta = post(URL, json={"model": juiz, "state": _estado(item, empresa, setor), "questions": PERGUNTAS},
                        timeout=PRAZO_S, headers={"Authorization": f"Bearer {chave}"})
        segundos = time.perf_counter() - inicio
        if not resposta.ok:
            return Juizo(juiz, item.id, item.ticker, None, None, segundos, erro=f"HTTP {resposta.status_code}")
        dados = resposta.json()
        respostas, uso = dados.get("answers") or {}, dados.get("usage") or {}
        return Juizo(juiz, item.id, item.ticker, float(respostas["material"]["noul"]), float(respostas["nova"]["noul"]),
                     segundos, uso.get("input_tokens"), uso.get("cost"), dados.get("model"))
    except (requests.RequestException, ValueError, KeyError, TypeError) as erro:
        return Juizo(juiz, item.id, item.ticker, None, None, time.perf_counter() - inicio,
                     erro=type(erro).__name__)


def julgar(itens: Iterable[Item], empresas: Mapping[str, tuple[str, str]], chave: str | None,
           post: Callable = requests.post) -> list[Juizo]:
    """Every company item x every judge; ``empresas``: ticker -> (company name, sector)."""
    if not chave:
        return []
    juizos = []
    for item in itens:
        if item.ticker == MACRO:
            continue
        empresa, setor = empresas.get(item.ticker, ("", ""))
        juizos += [_perguntar(juiz, item, empresa, setor, chave, post) for juiz in JUIZES]
    return juizos
