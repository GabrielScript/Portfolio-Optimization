"""Checks of the model's answers against the pack's verbatim items (spec, "Stage 1" and "Stage 2", amendment 4).

Reuses ``core.assessor``'s normalization and number matching only; ``core.assessor``'s own class set is untouched.
Any failure discards the answer: a scan becomes ``sem_evento`` and a decision ``manter``, with the reason kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.assessor import _LINK, _NUMERO_SOLTO, _leituras, _normalizar, _numero_presente, _numeros
from core.sentinela.pacote import Pacote

VEREDITOS = ("sem_evento", "atencao", "grave")
ACOES = ("manter", "reduzir_metade", "vender")
CLASSES_OFICIAIS = frozenset({"cvm", "mercado", "bcb", "ibge", "tesouro"})


@dataclass(frozen=True)
class Varredura:
    veredito: str
    citacoes: list[dict] = field(default_factory=list)
    resumo: str = ""
    descarte: str | None = None


@dataclass(frozen=True)
class Decisao:
    acao: str
    confianca: float
    citacoes: list[dict] = field(default_factory=list)
    motivo: str = ""
    descarte: str | None = None


class _Recusa(Exception):
    """A failed check; the message is the reason (Portuguese, shown on screen and in the log)."""


def _citacoes(bruto: dict, pacote: Pacote, exigidas: bool) -> list[dict]:
    citacoes = bruto.get("citacoes")
    if not isinstance(citacoes, list) or (exigidas and not citacoes):
        raise _Recusa("sem citações")
    evidencias = pacote.evidencias()
    for citacao in citacoes:
        if not isinstance(citacao, dict) or not isinstance(citacao.get("id"), str) \
                or not isinstance(citacao.get("trecho"), str):
            raise _Recusa("citação sem id e trecho")
        evidencia = evidencias.get(citacao["id"].strip())
        if evidencia is None:
            raise _Recusa(f"o item {citacao['id']} não está no pacote")
        trecho = _normalizar(citacao["trecho"])
        if not trecho or trecho not in _normalizar(evidencia.texto):
            raise _Recusa(f"o trecho não está no item {evidencia.id}: «{citacao['trecho'].strip()[:80]}»")
    return citacoes


def _texto_livre(texto: object, citacoes: list[dict], campo: str) -> str:
    if not isinstance(texto, str):
        raise _Recusa(f"sem {campo}")
    if _LINK.search(texto):
        raise _Recusa(f"{campo} com link")
    citados = [numero for citacao in citacoes for numero in _numeros(_normalizar(citacao["trecho"]))]
    for token in _NUMERO_SOLTO.findall(texto):
        if not _numero_presente(_leituras(token), citados):
            raise _Recusa(f"o número {token} do {campo} não está nas citações")
    return texto


def _cabecalho(bruto: object, pacote: Pacote) -> dict:
    if not isinstance(bruto, dict):
        raise _Recusa("a resposta não é um objeto JSON")
    if str(bruto.get("ticker", "")).strip().upper().removesuffix(".SA") != pacote.ticker:
        raise _Recusa(f"a resposta é sobre {bruto.get('ticker')!r}, não {pacote.ticker}")
    return bruto


def validar_varredura(bruto: object, pacote: Pacote, classes: frozenset[str] = CLASSES_OFICIAIS) -> Varredura:
    """``grave`` needs at least one citation of a new item of an official class (``classes``); else ``atencao``."""
    try:
        resposta = _cabecalho(bruto, pacote)
        veredito = resposta.get("veredito")
        if veredito not in VEREDITOS:
            raise _Recusa(f"veredito desconhecido {veredito!r}")
        citacoes = _citacoes(resposta, pacote, exigidas=veredito != "sem_evento")
        resumo = _texto_livre(resposta.get("resumo", ""), citacoes, "resumo")
    except _Recusa as recusa:
        return Varredura("sem_evento", descarte=str(recusa))
    if veredito == "grave":
        evidencias = pacote.evidencias()
        if not any(evidencias[citacao["id"].strip()].classe in classes for citacao in citacoes):
            veredito = "atencao"
    return Varredura(veredito, citacoes, resumo)


def validar_decisao(bruto: object, pacote: Pacote) -> Decisao:
    """``reduzir_metade`` and ``vender`` need citations; ``confianca`` must be a number in [0, 1]."""
    try:
        resposta = _cabecalho(bruto, pacote)
        acao = resposta.get("acao")
        if acao not in ACOES:
            raise _Recusa(f"ação desconhecida {acao!r}")
        confianca = resposta.get("confianca")
        if isinstance(confianca, bool) or not isinstance(confianca, (int, float)) or not math.isfinite(confianca) \
                or not 0.0 <= confianca <= 1.0:
            raise _Recusa(f"confiança inválida {confianca!r}")
        citacoes = _citacoes(resposta, pacote, exigidas=acao != "manter")
        motivo = _texto_livre(resposta.get("motivo", ""), citacoes, "motivo")
    except _Recusa as recusa:
        return Decisao("manter", 0.0, descarte=str(recusa))
    return Decisao(acao, float(confianca), citacoes, motivo)
