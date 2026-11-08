"""
recompra.py - Buyback announcements, signal 2 of the pre-registered test
(docs/superpowers/specs/2026-11-08-pead-recompra-design.md, section 4): which CVM IPE rows announce a buyback program
(by category and text only, no model reads them), their information and entry dates, the event study of stage A and
the arm of stage B.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

import pandas as pd

from core.backtest import Estrategia
from core.composto import estrategia_por_nota
from core.risco import emissor
from core.valor import NotaDoMes

CATEGORIAS = frozenset({"fato relevante", "comunicado ao mercado", "reuniao da administracao", "assembleia"})
PROGRAMA = re.compile(r"recompra|aquisicao de acoes de (sua )?propria emissao|acoes de (sua )?(propria )?emissao "
                      r"(da )?propria")
FIM = re.compile(r"encerramento|conclusao|termino|cancelamento|alienacao")
INICIO = re.compile(r"novo|nova|aprovacao|aprova|renovacao|institui|abertura|inicio")


def normalizar(texto: object) -> str:
    """Lower case without accents (NFKD); a missing field is empty."""
    if texto is None or (isinstance(texto, float) and texto != texto):
        return ""
    decomposto = unicodedata.normalize("NFKD", str(texto))
    return "".join(letra for letra in decomposto if not unicodedata.combining(letra)).lower()


def e_recompra(categoria: object, tipo: object, especie: object, assunto: object) -> bool:
    """A buyback-program row: one of ``CATEGORIAS``, ``Tipo | Especie | Assunto`` matching ``PROGRAMA``, and not the
    end of a program (``FIM`` without ``INICIO``)."""
    if normalizar(categoria) not in CATEGORIAS:
        return False
    texto = " | ".join(normalizar(campo) for campo in (tipo, especie, assunto))
    if not PROGRAMA.search(texto):
        return False
    return not FIM.search(texto) or bool(INICIO.search(texto))


ATRASO_MAXIMO = pd.Timedelta(days=30)      # a later delivery has no clear information date
SESSOES_ENTRE_EVENTOS = 63                 # one program, many filings


def eventos(linhas: pd.DataFrame, sessoes: pd.DatetimeIndex) -> pd.DataFrame:
    """Buyback rows (``issuer``, ``Data_Referencia``, ``Data_Entrega``) -> one event per (issuer, reference date):
    information date = the later of the reference and the earliest delivery (dropped when that delivery is more than
    ``ATRASO_MAXIMO`` after the reference), entry = the first session after it; per issuer, an entry less than
    ``SESSOES_ENTRE_EVENTOS`` sessions after the previous kept one is dropped."""
    documentos = (linhas.groupby(["issuer", "Data_Referencia"], as_index=False)["Data_Entrega"].min()
                  .sort_values(["issuer", "Data_Referencia"]))
    documentos = documentos[documentos["Data_Entrega"] <= documentos["Data_Referencia"] + ATRASO_MAXIMO]
    informacao = documentos[["Data_Referencia", "Data_Entrega"]].max(axis=1)
    posicao = sessoes.searchsorted(informacao.to_numpy(), side="right")
    validas = posicao < len(sessoes)
    candidatos = pd.DataFrame({"issuer": documentos["issuer"].to_numpy()[validas],
                               "informacao": informacao.to_numpy()[validas], "posicao": posicao[validas]})
    mantidos = []
    for _, grupo in candidatos.groupby("issuer", sort=True):
        ultima = None
        for linha in grupo.sort_values("posicao").itertuples(index=False):
            if ultima is None or linha.posicao - ultima >= SESSOES_ENTRE_EVENTOS:
                mantidos.append(linha)
                ultima = linha.posicao
    resultado = pd.DataFrame(mantidos, columns=["issuer", "informacao", "posicao"])
    resultado["entrada"] = sessoes[resultado["posicao"].to_numpy()]
    return resultado.drop(columns="posicao").reset_index(drop=True)


JANELA = 63                                # sessions after the entry


def retornos_anormais(eventos: pd.DataFrame, precos: pd.DataFrame, universo: pd.DataFrame,
                      janela: int = JANELA) -> pd.DataFrame:
    """Per event (``issuer``, ``entrada``): the mean return of the issuer's classes in the universe of the entry's month
    from the entry's close to ``janela`` sessions later, minus the equal-weighted mean of every name of that month's
    universe with a price at the entry (a price that stops is carried forward). Out: an end after the last session, or
    no class of the issuer in the month's universe."""
    carregados = precos.ffill()
    membros = universo.groupby("month")["security"].apply(list)
    linhas = []
    for evento in eventos.itertuples(index=False):
        inicio = precos.index.get_loc(evento.entrada)
        fim = inicio + janela
        mes = evento.entrada.to_period("M").to_timestamp()
        if fim >= len(precos.index) or mes not in membros.index:
            continue
        nomes = [papel for papel in membros[mes] if papel in precos.columns and pd.notna(precos[papel].iloc[inicio])]
        retornos = carregados[nomes].iloc[fim] / precos[nomes].iloc[inicio] - 1.0
        do_emissor = [papel for papel in nomes if emissor(papel) == evento.issuer]
        if not do_emissor:
            continue
        linhas.append({"issuer": evento.issuer, "entrada": evento.entrada,
                       "anormal": float(retornos[do_emissor].mean() - retornos.mean())})
    return pd.DataFrame(linhas, columns=["issuer", "entrada", "anormal"])


def carteira_mensal(anormais: pd.DataFrame) -> pd.Series:
    """Calendar month of entry (1st day) -> the mean abnormal return of the events entering in it."""
    meses = anormais["entrada"].dt.to_period("M").dt.to_timestamp()
    return anormais.groupby(meses)["anormal"].mean()


BONUS = 0.25                               # added to the Composite percentile, stage B


def estrategia_recompra(nota: NotaDoMes, eventos_: pd.DataFrame, setores: Mapping[str, str]) -> Estrategia:
    """The Composite rule on the Composite score's percentile (among the names with one) plus ``BONUS`` for each
    name whose issuer has an event with information date a < d inside the ``JANELA`` sessions ending at the decision
    session d."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        decisao, inicio = treino.index[-1], treino.index[-JANELA]
        recentes = set(eventos_.loc[(eventos_["informacao"] >= inicio) & (eventos_["informacao"] < decisao), "issuer"])

        def _inclinada(mes: pd.Timestamp, candidatos: list[str]) -> pd.Series:
            composto = nota(mes, candidatos)
            com_nota = composto.dropna().index
            bonus = pd.Series([BONUS if emissor(papel) in recentes else 0.0 for papel in com_nota], index=com_nota)
            return (composto[com_nota].rank(pct=True) + bonus).reindex(composto.index)
        return estrategia_por_nota(_inclinada, setores)(treino)
    return _estrategia
