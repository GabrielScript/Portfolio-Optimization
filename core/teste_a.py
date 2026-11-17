"""
teste_a.py - Test A of the sentiment pre-registration, as amended on 2026-11-17
(``docs/preregistro-sentimento-2026-10-23.md``): CVM material filings scored by a Liquid d1 model, used as a veto on
the Composite. A chosen name whose issuer delivered a filing judged bad (P >= 0.5) with intensity >= 2.0 in the 21
sessions before the decision date is dropped, and the next name by score takes its place under the same rules.

Pure helpers only: the scoring run is ``scripts/teste_a_pontuar.py`` and the reading ``scripts/teste_a_avaliar.py``.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd

CATEGORIAS = ("Fato Relevante", "Comunicado ao Mercado")    # the lake's MATERIAL_CATEGORIES
P_RUIM = 0.5
INTENSIDADE = 2.0
JANELA = 21
_DO_LAGO = ("INTENSITY_LEVELS", "FILING_QUESTIONS", "FILING_QUESTIONS_VERSION")


def perguntas_do_lago(fonte: Path) -> tuple[dict, str]:
    """``filings-v1`` exactly as the lake's ``jevscore.py`` defines it: only its literal assignments are evaluated, so
    the questions are never retyped here and the lake's imports are not needed."""
    arvore = ast.parse(Path(fonte).read_text(encoding="utf-8"))
    espaco: dict = {}
    for no in arvore.body:
        if isinstance(no, ast.Assign) and any(isinstance(alvo, ast.Name) and alvo.id in _DO_LAGO for alvo in no.targets):
            exec(compile(ast.Module(body=[no], type_ignores=[]), str(fonte), "exec"), {}, espaco)  # noqa: S102
    return espaco["FILING_QUESTIONS"], espaco["FILING_QUESTIONS_VERSION"]


def estado_do_documento(linha: Mapping) -> dict:
    """The lake's ``filing_request`` state: what the CVM form says the filing is, without the company's name or date."""
    return {"filing": {"category": _texto(linha["categoria"]), "subject": _texto(linha["assunto"]),
                       "type": _texto(linha["tipo"]), "species": _texto(linha["especie"])}}


def _texto(valor) -> str:
    """A CVM field as text: None (the lake's polars null), NaN or pd.NA (pandas) become ""."""
    return "" if valor is None or pd.isna(valor) else str(valor)


def documentos_do_universo(ipe: pd.DataFrame, setores: pd.DataFrame, universo: pd.DataFrame,
                           inicio: pd.Timestamp, fim: pd.Timestamp) -> pd.DataFrame:
    """Material filings delivered in [inicio, fim] by issuers that are Composite candidates in some month of the
    export, one row per (filing, candidate issuer): a CVM code shared by two prefixes counts for both. Columns: the
    issuer prefix, the CVM code, the delivery date, the four state fields and the state's key."""
    candidatos = set(universo["issuer"].dropna())
    codigos = (setores.dropna(subset=["code_cvm"])
               .assign(codigo_cvm=lambda quadro: pd.to_numeric(quadro["code_cvm"], errors="coerce"))
               .dropna(subset=["codigo_cvm"]))
    codigos = codigos[codigos["issuer"].isin(candidatos)][["issuer", "codigo_cvm"]].drop_duplicates()
    materiais = ipe[ipe["categoria"].isin(CATEGORIAS)].assign(
        data=lambda quadro: pd.to_datetime(quadro["data_entrega"], errors="coerce"),
        codigo_cvm=lambda quadro: pd.to_numeric(quadro["codigo_cvm"], errors="coerce"))
    materiais = materiais[(materiais["data"] >= inicio) & (materiais["data"] <= fim)]
    docs = materiais.merge(codigos, on="codigo_cvm", how="inner").rename(columns={"issuer": "emissor"})
    docs["chave"] = [chave_do_estado(estado_do_documento(linha)) for linha in docs.to_dict("records")]
    colunas = ["emissor", "codigo_cvm", "data", "categoria", "assunto", "tipo", "especie", "chave"]
    return docs[colunas].sort_values(["data", "emissor"], kind="stable").reset_index(drop=True)


def emissores_vetados(documentos: pd.DataFrame, pregoes: pd.DatetimeIndex, t: pd.Timestamp,
                      janela: int = JANELA) -> set[str]:
    """Issuers with a filing judged bad (``p_ruim`` >= 0.5) and strong (``intensidade`` >= 2.0) delivered from the
    ``janela``-th session before ``t`` up to, not including, ``t`` (the engine trades at t's close, and a filing dated
    t may arrive after it)."""
    posicao = int(pregoes.get_loc(t))
    inicio = pregoes[max(posicao - janela, 0)]
    na_janela = documentos[(documentos["data"] >= inicio) & (documentos["data"] < t)]
    fortes = na_janela[(na_janela["p_ruim"] >= P_RUIM) & (na_janela["intensidade"] >= INTENSIDADE)]
    return set(fortes["emissor"])


def estrategia_com_veto(nota, setores: Mapping[str, str], vetados_em: Callable[[pd.Timestamp], set[str]],
                        nomes: int | None = None, por_setor: int | None = None, limites=None):
    """``core.composto.estrategia_por_nota`` with the veto: at the decision (the training window's last session t),
    the names whose issuer prefix is in ``vetados_em(t)`` lose their score, so the next names by score take their
    places under the same sector cap and rules."""
    from core.composto import estrategia_por_nota  # noqa: PLC0415
    from core.risco import LIMITES  # noqa: PLC0415
    from core.sinais import NOMES, POR_SETOR  # noqa: PLC0415

    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        vetados = vetados_em(treino.index[-1])

        def nota_sem_vetados(mes, candidatos):
            notas = nota(mes, candidatos)
            return notas[[papel[:4] not in vetados for papel in notas.index]]

        return estrategia_por_nota(nota_sem_vetados, setores, NOMES if nomes is None else nomes,
                                   POR_SETOR if por_setor is None else por_setor,
                                   LIMITES if limites is None else limites)(treino)
    return _estrategia


def faixa_do_tom(p_ruim: float, p_bom: float, intensidade: float) -> str:
    """Tone x intensity cell: bad or good at P >= 0.5, "forte" at intensity >= 2.0 (the veto's cell is "ruim forte")."""
    forte = " forte" if intensidade >= INTENSIDADE else ""
    if p_ruim >= P_RUIM:
        return "ruim" + forte
    if p_bom >= P_RUIM:
        return "bom" + forte
    return "neutro"


def faixa_de_p(p: float) -> str:
    """The user's thresholds as bands: < 0.35, 0.35-0.65, 0.65-0.70, >= 0.70."""
    if p < 0.35:
        return "< 0.35"
    if p < 0.65:
        return "0.35-0.65"
    if p < 0.70:
        return "0.65-0.70"
    return ">= 0.70"


def excessos_apos_documentos(precos: pd.DataFrame, elegiveis: Mapping[pd.Timestamp, list[str]],
                             eventos: pd.DataFrame, horizonte: int) -> pd.DataFrame:
    """For each event (``emissor``, ``data``): entry at the close of the first session strictly after ``data``; the
    issuer's eligible classes' mean return over the next ``horizonte`` sessions minus the equal-weighted mean of the
    month's eligible names. Events without ``horizonte`` sessions after the entry, or with no eligible class, are
    dropped. Returns the events' columns plus ``entrada`` and ``excesso``, in the events' order."""
    pregoes = precos.index
    posicoes = pregoes.searchsorted(pd.DatetimeIndex(eventos["data"]), side="right")
    linhas = []
    for evento, posicao in zip(eventos.to_dict("records"), posicoes):
        if posicao + horizonte >= len(pregoes):
            continue
        entrada = pregoes[posicao]
        nomes = [papel for papel in elegiveis.get(entrada.to_period("M").to_timestamp(), []) if papel in precos.columns]
        retorno = (precos.iloc[posicao + horizonte][nomes] / precos.iloc[posicao][nomes] - 1.0).dropna()
        do_emissor = retorno[[papel[:4] == evento["emissor"] for papel in retorno.index]]
        if do_emissor.empty or retorno.empty:
            continue
        linhas.append({**evento, "entrada": entrada, "excesso": float(do_emissor.mean() - retorno.mean())})
    return pd.DataFrame(linhas)


def colunas_da_resposta(respostas: Mapping) -> dict:
    """P(bad) and P(good) from the tone's probabilities, the intensity score (expected level 0-3), the filing kind."""
    tom = respostas["filing_tone"]["probabilities"]
    return {"p_ruim": float(tom["bad"]), "p_bom": float(tom["good"]),
            "intensidade": float(respostas["filing_intensity"]["score"]),
            "tipo_do_fato": respostas["filing_kind"]["choice"]}


def chave_do_estado(estado: dict) -> str:
    """SHA-1 of the state's canonical JSON: identical states are asked once and share the answer."""
    return hashlib.sha1(json.dumps(estado, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
