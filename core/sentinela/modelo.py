"""DeepSeek stages of the Sentinela (spec, "Stage 1" and "Stage 2"): one stock per call, the fixed instructions as the
system message (the cached prefix, ``deepseek-cache-economy``), the pack as the user message, a JSON answer, no tools.
Stage 1 scans at low effort; stage 2 decides at max effort, only for a ``grave`` that passed the checks. A failed
call is a discarded answer (``sem_evento`` / ``manter``), never a retry loop.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime

import requests

from core.assessor import MODELO, PRECOS, ErroDoModelo, Uso, _chamar, horario_de_pico
from core.sentinela.checagem import Decisao, Varredura, validar_decisao, validar_varredura
from core.sentinela.pacote import Pacote

ESFORCO_VARREDURA = "low"
ESFORCO_DECISAO = "max"
MAX_TOKENS_VARREDURA = 8_000
MAX_TOKENS_DECISAO = 32_000
LIMITES = {"varredura": 40, "decisao": 6}          # hard caps a day (spec, "Budget caps")

INSTRUCOES_VARREDURA = """You are the Sentinela of Aleph Quant: a skeptical watch agent over ONE stock of a rule-based \
B3 portfolio (the Composite). You read only the pack in the user message: new official items about this stock (CVM \
filings, B3 market data, BCB, IBGE and Tesouro releases) and a few context lines. You have no other source and no \
tools; never use facts from memory.

Your only job: say whether anything in the NEW items seriously compromises this stock's value: fraud, a judicial \
recovery or bankruptcy filing, a default, a regulator's sanction, the loss of a key license or contract, a large \
unexpected write-down or guidance cut, a controlling-shareholder event that hurts minorities, a price or volume shock \
explained by the items. Routine disclosure (scheduled results, calendars, recurring dividends or interest on equity, \
routine board or officer changes, presentations, debt issuance) and ordinary macro releases are "sem_evento". A macro \
item matters only when it bears directly on this company's business.

Verdicts: "sem_evento" (the expected answer most of the time), "atencao" (worth the investor's eye, not serious), \
"grave" (seriously compromises the stock). When unsure between two verdicts, pick the milder one.

Rules, checked by code (an answer that breaks one is discarded):
- cite only ids that are in the pack; each citation's "trecho" is copied verbatim from that item's text, a short span \
of at most 200 characters;
- every number in "resumo" appears in a cited "trecho"; no links;
- "sem_evento" may have no citations; "atencao" and "grave" need at least one; a "grave" must cite a new item, not \
only context lines.

Answer with one JSON object only, text in Brazilian Portuguese:
{"ticker": "<TICKER>", "veredito": "sem_evento" | "atencao" | "grave", "citacoes": [{"id": "...", "trecho": "..."}], \
"resumo": "at most 2 sentences"}"""

INSTRUCOES_DECISAO = """You are the Sentinela of Aleph Quant deciding, for a FICTITIOUS account only, whether to keep, \
halve or sell ONE stock of the Composite portfolio after the scan flagged a serious event. You read only the pack in \
the user message: the new official items, the scan's citations, the stock's recent CVM filings and context lines. No \
other source, no tools, never facts from memory.

Actions: "manter" (keep: the default when the evidence is ambiguous, already priced or temporary), "reduzir_metade" \
(sell half), "vender" (sell all). Sell only when the cited items show a lasting impairment of the company's value. A \
price fall alone is not a reason.

"confianca" is your probability, from 0 to 1, that the stock falls more than 1.5 standard deviations of its usual \
20-session move over the next 20 sessions (for "reduzir_metade" or "vender"), or that it does not (for "manter"). It \
is scored with the Brier and log scores against what happens, so overstating it costs you. Actions stated below 0.70 \
are not executed.

Rules, checked by code (an answer that breaks one becomes "manter"):
- cite only ids that are in the pack; each "trecho" is copied verbatim from that item's text;
- every number in "motivo" appears in a cited "trecho"; no links;
- "reduzir_metade" and "vender" need at least one citation.

Answer with one JSON object only, text in Brazilian Portuguese:
{"ticker": "<TICKER>", "acao": "manter" | "reduzir_metade" | "vender", "confianca": 0.0 to 1.0, "citacoes": \
[{"id": "...", "trecho": "..."}], "motivo": "at most 3 sentences"}"""

Post = Callable[..., requests.Response]


@dataclass
class Resposta:
    """The parsed JSON answer (None on failure, then ``erro`` says why) and the tokens it cost."""
    bruto: object
    uso: Uso
    erro: str | None = None
    falha_da_api: bool = False      # network or HTTP failure: the items are not marked seen and wait for the next run


def perguntar(instrucoes: str, texto: str, esforco: str, max_tokens: int, chave: str, agora: datetime,
              post: Post = requests.post) -> Resposta:
    uso = Uso()
    corpo = {"model": MODELO, "messages": [{"role": "system", "content": instrucoes},
                                           {"role": "user", "content": texto}],
             "thinking": {"type": "enabled"}, "reasoning_effort": esforco,
             "response_format": {"type": "json_object"}, "max_tokens": max_tokens}
    try:
        dados = _chamar(corpo, chave, post)
    except ErroDoModelo as erro:
        return Resposta(None, uso, str(erro), falha_da_api=True)
    uso.somar(dados.get("usage") or {}, horario_de_pico(agora), PRECOS[MODELO])
    escolha = (dados.get("choices") or [{}])[0]
    if escolha.get("finish_reason") == "length":
        return Resposta(None, uso, "resposta cortada no limite de tokens")
    try:
        return Resposta(json.loads((escolha.get("message") or {}).get("content") or ""), uso)
    except json.JSONDecodeError:
        return Resposta(None, uso, "a resposta não é JSON")


def varrer(pacote: Pacote, chave: str, agora: datetime, post: Post = requests.post) -> tuple[Varredura, Resposta]:
    resposta = perguntar(INSTRUCOES_VARREDURA, pacote.texto(), ESFORCO_VARREDURA, MAX_TOKENS_VARREDURA, chave, agora,
                         post)
    if resposta.erro:
        return Varredura("sem_evento", descarte=resposta.erro), resposta
    return validar_varredura(resposta.bruto, pacote), resposta


def texto_da_decisao(pacote: Pacote, varredura: Varredura) -> str:
    citacoes = "\n".join(f"{citacao['id']} | {citacao['trecho']}" for citacao in varredura.citacoes)
    return f"{pacote.texto()}\n\nCitações da varredura (veredito {varredura.veredito}):\n{citacoes}"


def decidir(pacote: Pacote, varredura: Varredura, chave: str, agora: datetime,
            post: Post = requests.post) -> tuple[Decisao, Resposta]:
    resposta = perguntar(INSTRUCOES_DECISAO, texto_da_decisao(pacote, varredura), ESFORCO_DECISAO,
                         MAX_TOKENS_DECISAO, chave, agora, post)
    if resposta.erro:
        return Decisao("manter", 0.0, descarte=resposta.erro), resposta
    return validar_decisao(resposta.bruto, pacote), resposta


@dataclass
class Orcamento:
    """Calls left today per stage; over the cap the items wait for the next session."""
    usados: dict[str, int] = field(default_factory=lambda: dict.fromkeys(LIMITES, 0))

    def gastar(self, etapa: str) -> bool:
        if self.usados.get(etapa, 0) >= LIMITES[etapa]:
            return False
        self.usados[etapa] = self.usados.get(etapa, 0) + 1
        return True

    @classmethod
    def do_dia(cls, usados: Mapping[str, int]) -> Orcamento:
        return cls({etapa: int(usados.get(etapa, 0)) for etapa in LIMITES})
