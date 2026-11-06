"""One hourly run of the Sentinela (spec, "Schedule"): the sources were already read by code into ``Entrada.itens``;
each Composite name with new items gets one low-effort scan (within the day's cap), its company items go to the shadow
judges, a checked ``grave`` gets one max-effort decision, an action is accepted or refused by the account's rules,
every call is logged, and the emails go out. ``atualizar_conta`` then trades the accepted actions whose close is in
the price files. Every outside service is injected (tests run offline)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from core.rebalanceio import CustoDaOrdem
from core.sentinela import avisos, conta, registro
from core.sentinela.checagem import Decisao, Varredura
from core.sentinela.itens import Item
from core.sentinela.juizes import julgar
from core.sentinela.modelo import Orcamento, Resposta, decidir, varrer
from core.sentinela.pacote import contexto_da_acao, montar_pacote, pacote_da_decisao
from core.sentinela.vistos import ler_vistos, marcar, novos_por_acao

HORA_DO_RESUMO = 19


@dataclass(frozen=True)
class Arquivos:
    pasta: Path

    @property
    def vistos(self) -> Path:
        return self.pasta / "sentinela_vistos.csv"

    @property
    def log(self) -> Path:
        return self.pasta / "sentinela_log.csv"

    @property
    def acoes(self) -> Path:
        return self.pasta / "sentinela_acoes.csv"

    @property
    def ordens(self) -> Path:
        return self.pasta / "sentinela_ordens.csv"

    @property
    def pontuacao(self) -> Path:
        return self.pasta / "sentinela_pontuacao.csv"


@dataclass
class Entrada:
    """What one run reads: ``carteira`` ticker -> Composite weight, ``empresas`` ticker -> (name, sector), ``itens``
    every source's items, ``fatos`` the names' recent CVM filings (stage 2), ``pregoes`` the B3 calendar."""
    agora: datetime
    carteira: Mapping[str, float]
    registrada: pd.Timestamp
    empresas: Mapping[str, tuple[str, str]]
    itens: list[Item]
    fatos: list[Item]
    fechamentos: pd.DataFrame
    pregoes: pd.DatetimeIndex


@dataclass
class Servicos:
    chave_deepseek: str | None
    chave_liquid: str | None
    post_deepseek: Callable = requests.post
    post_liquid: Callable = requests.post
    email: Callable[[str, str], object] | None = None


@dataclass
class Relatorio:
    chamadas: int = 0
    graves: list[str] = field(default_factory=list)
    acoes: list[str] = field(default_factory=list)
    adiados: list[str] = field(default_factory=list)
    emails: int = 0
    erro: str | None = None
    falhas: list[str] = field(default_factory=list)


def _linha(agora: datetime, ticker: str, etapa: str, resposta: Resposta | None = None, **campos) -> dict:
    linha = {"hora": agora, "ticker": ticker, "etapa": etapa} | campos
    if resposta is not None:
        linha |= {"entrada": resposta.uso.entrada, "cache": resposta.uso.cache, "saida": resposta.uso.saida,
                  "custo_usd": round(resposta.uso.custo_usd, 8), "erro": resposta.erro}
    return linha


def _citacoes(citacoes: list[dict]) -> str:
    return json.dumps(citacoes, ensure_ascii=False)


def _enviar(servicos: Servicos, relatorio: Relatorio, mensagem: tuple[str, str]) -> None:
    if servicos.email is None:
        return
    try:
        servicos.email(*mensagem)
        relatorio.emails += 1
    except Exception as erro:       # noqa: BLE001 - a failed email is noted; the run and its files stay
        relatorio.falhas.append(f"e-mail: {type(erro).__name__}")


def _decidir(entrada: Entrada, servicos: Servicos, pacote, varredura: Varredura, aceitas: list[dict],
             arquivos: Arquivos, linhas: list[dict], relatorio: Relatorio) -> tuple[Decisao, Mapping | None, dict]:
    ticker = pacote.ticker
    decisao_pacote = pacote_da_decisao(pacote, entrada.fatos)
    decisao, resposta = decidir(decisao_pacote, varredura, servicos.chave_deepseek, entrada.agora,
                                servicos.post_deepseek)
    relatorio.chamadas += 1
    linhas.append(_linha(entrada.agora, ticker, "decisao", resposta, acao=decisao.acao, confianca=decisao.confianca,
                         itens=";".join(item.id for item in decisao_pacote.itens), descarte=decisao.descarte,
                         texto=decisao.motivo, citacoes=_citacoes(decisao.citacoes)))
    if decisao.acao == "manter":
        return decisao, None, decisao_pacote.evidencias()
    acao = {"id": f"{ticker}-{entrada.agora:%Y-%m-%dT%H:%M}", "decidida_em": entrada.agora, "ticker": ticker,
            "acao": decisao.acao, "confianca": decisao.confianca}
    motivo = conta.recusa(acao, aceitas, entrada.registrada)
    executa = conta.data_de_execucao(entrada.agora, entrada.pregoes)
    if motivo is None and executa is None:
        motivo = "sem pregão no calendário"
    acao |= {"executa_em": executa, "estado": "aceita" if motivo is None else "recusada", "motivo": motivo}
    registro.acrescentar_acoes(arquivos.acoes, [acao])
    if motivo is None:
        aceitas.append(acao)
        relatorio.acoes.append(acao["id"])
    return decisao, acao, decisao_pacote.evidencias()


def _resumo(entrada: Entrada, arquivos: Arquivos, servicos: Servicos, relatorio: Relatorio) -> None:
    if entrada.agora.hour < HORA_DO_RESUMO:
        return
    log = registro.ler_log(arquivos.log)
    hoje = log[log["hora"].dt.date == entrada.agora.date()] if not log.empty else log
    if hoje.empty or (hoje["etapa"] == "resumo").any():
        return
    atencoes = hoje[(hoje["etapa"] == "varredura") & (hoje["veredito"] == "atencao")]
    adiados = list(hoje.loc[hoje["etapa"] == "adiado", "ticker"])
    if not atencoes.empty or adiados:
        _enviar(servicos, relatorio, avisos.mensagem_do_resumo(pd.Timestamp(entrada.agora), atencoes, adiados))
    registro.acrescentar_log(arquivos.log, [_linha(entrada.agora, "", "resumo")])


def rodar(entrada: Entrada, arquivos: Arquivos, servicos: Servicos) -> Relatorio:
    relatorio = Relatorio()
    if not servicos.chave_deepseek:
        relatorio.erro = "falta DEEPSEEK_API_KEY no .env do ai-hedge-fund"
        return relatorio
    log = registro.ler_log(arquivos.log)
    orcamento = Orcamento.do_dia(registro.usados_no_dia(log, entrada.agora.date()))
    aceitas = registro.aceitas(registro.ler_acoes(arquivos.acoes))
    novos = novos_por_acao(entrada.itens, sorted(entrada.carteira), ler_vistos(arquivos.vistos))
    linhas: list[dict] = []
    for ticker, itens in novos.items():
        empresa, setor = entrada.empresas.get(ticker, ("", ""))
        peso = float(entrada.carteira[ticker])
        pacote = montar_pacote(ticker, setor, peso, itens,
                               contexto_da_acao(ticker, entrada.fechamentos, entrada.registrada, peso))
        if not orcamento.gastar("varredura"):
            relatorio.adiados.append(ticker)
            linhas.append(_linha(entrada.agora, ticker, "adiado", itens=";".join(item.id for item in itens)))
            continue
        varredura, resposta = varrer(pacote, servicos.chave_deepseek, entrada.agora, servicos.post_deepseek)
        relatorio.chamadas += 1
        linhas.append(_linha(entrada.agora, ticker, "varredura", resposta, veredito=varredura.veredito,
                             itens=";".join(item.id for item in pacote.itens), descarte=varredura.descarte,
                             texto=varredura.resumo, citacoes=_citacoes(varredura.citacoes)))
        if resposta.falha_da_api:
            continue
        for juizo in julgar(pacote.itens, entrada.empresas, servicos.chave_liquid, servicos.post_liquid):
            linhas.append(_linha(entrada.agora, ticker, f"juiz:{juizo.juiz}", p=juizo.p_material, itens=juizo.item,
                                 texto=None if juizo.p_nova is None else f"p_nova={juizo.p_nova:.4f}",
                                 entrada=juizo.tokens, custo_usd=juizo.custo, erro=juizo.erro))
        marcar(arquivos.vistos, [(ticker, item.id) for item in pacote.itens], entrada.agora)
        if varredura.veredito != "grave":
            continue
        relatorio.graves.append(ticker)
        decisao, acao, evidencias = None, None, pacote.evidencias()
        if orcamento.gastar("decisao"):
            decisao, acao, evidencias = _decidir(entrada, servicos, pacote, varredura, aceitas, arquivos, linhas,
                                                 relatorio)
        _enviar(servicos, relatorio, avisos.mensagem_de_alerta(ticker, empresa, varredura, evidencias, decisao, acao))
    registro.acrescentar_log(arquivos.log, linhas)
    _resumo(entrada, arquivos, servicos, relatorio)
    return relatorio


def semear(entrada: Entrada, arquivos: Arquivos) -> int:
    """Marks every current item seen for every name, with no model call: the baseline of the first live session, so
    the backlog of releases from before the agent went live does not spend that day's budget. Returns the pairs."""
    novos = novos_por_acao(entrada.itens, sorted(entrada.carteira), ler_vistos(arquivos.vistos))
    pares = [(ticker, item.id) for ticker, itens in novos.items() for item in itens]
    marcar(arquivos.vistos, pares, entrada.agora)
    registro.acrescentar_log(arquivos.log, [_linha(entrada.agora, "", "semeado", itens=str(len(pares)))])
    return len(pares)


def atualizar_conta(arquivos: Arquivos, registro_composta: pd.DataFrame, precos: pd.DataFrame, cdi: pd.Series,
                    custo: CustoDaOrdem, inicio: pd.Timestamp) -> list[str]:
    """Appends the account's due rows (both portfolios) and marks the actions traded; returns their ids."""
    acoes = registro.ler_acoes(arquivos.acoes)
    ordens = conta.ler(arquivos.ordens)
    novas, feitas = conta.ordens_devidas(ordens, registro_composta, registro.pendentes(acoes), precos, cdi, custo,
                                         inicio)
    if not novas.empty:
        conta.acrescentar(arquivos.ordens, novas)
    executadas = acoes[acoes["id"].isin(feitas) & (acoes["estado"] == "aceita")]
    registro.acrescentar_acoes(arquivos.acoes, [linha | {"estado": "executada", "motivo": None}
                                                for linha in executadas.to_dict("records")])
    return feitas
