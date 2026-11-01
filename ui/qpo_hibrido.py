"""
qpo_hibrido.py - Screen of the hybrid quantum forecast and QPO test (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md).
Past results, read from the derived files in ``docs/dados``; the app never imports torch or PennyLane. UI strings are Portuguese.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

PASTA = Path(__file__).resolve().parents[1] / "docs" / "dados"
PERCENTUAL = st.column_config.NumberColumn(format="percent")
TRES_CASAS = st.column_config.NumberColumn(format="%.3f")
DUAS_CASAS = st.column_config.NumberColumn(format="%.2f")
NOMES_DOS_MERCADOS = {"b3": ("B3", "BOVA11"), "us": ("Ações americanas", "SPY")}
COLUNAS_DO_IC = {"IC": st.column_config.NumberColumn(format="%.4f"), "t": DUAS_CASAS, "passa": "Passou"}
COLUNAS_DA_BANCADA = {"viavel": st.column_config.NumberColumn("Tiros viáveis", format="percent"),
                      "p_otimo": st.column_config.NumberColumn("Tiros no ótimo", format="percent"),
                      "razao_media": st.column_config.NumberColumn("Razão média", format="%.3f"),
                      "razao_melhor": st.column_config.NumberColumn("Razão do melhor", format="%.3f"),
                      "iqr_razao_media": st.column_config.NumberColumn("IQR da razão média", format="%.3f")}


@dataclass(frozen=True)
class Mercado:
    portao: pd.DataFrame        # one row per arm: lists with each phase's P and DSR, and whether it passed
    anos: pd.DataFrame          # by year in reais, with BOVA11, the CDI and the Selic
    carteiras: pd.DataFrame     # each arm's last rebalance


@dataclass(frozen=True)
class ResultadoHibrido:
    ic: pd.DataFrame
    diferencas: pd.DataFrame
    bancada: pd.DataFrame
    mercados: Mapping[str, Mercado]


def carregar(pasta: Path = PASTA) -> ResultadoHibrido | None:
    """The derived files, or None until every one exists."""
    def ler(nome: str) -> pd.DataFrame:
        return pd.read_parquet(pasta / f"qpo-hibrido-{nome}.parquet")

    try:
        return ResultadoHibrido(ler("ic"), ler("diferencas"), ler("bancada"),
                                {mercado: Mercado(ler(f"portao-{mercado}"), ler(f"anos-{mercado}"), ler(f"carteiras-{mercado}"))
                                 for mercado in NOMES_DOS_MERCADOS})
    except FileNotFoundError:
        return None


def _veredito(mercados: Mapping[str, Mercado]) -> str:
    candidatos = [f"{braco} ({NOMES_DOS_MERCADOS[mercado][0]})" for mercado, dados in mercados.items()
                  for braco in dados.portao.loc[dados.portao["passa"].astype(bool), "braco"]]
    if not candidatos:
        return ("Nenhum braço passou no portão, na B3 ou nos EUA. Nada aqui vira decisão de dinheiro real, e o Composto e a "
                "Conservadora continuam como estão.")
    return (f"Passou: {', '.join(candidatos)}. Passar faz do braço apenas candidato a um registro ao vivo; "
            "nenhuma decisão de dinheiro real sai desta tabela.")


def _portao(tabela: pd.DataFrame, referencia: str) -> pd.DataFrame:
    return pd.DataFrame({"Braço": tabela["braco"],
                         "P[> Composto] mínimo": tabela["p_composto"].map(min),
                         "P[> Conservadora] mínimo": tabela["p_conservadora"].map(min),
                         f"P[> {referencia}] mínimo": tabela["p_referencia"].map(min),
                         "DSR mínimo": tabela["dsr"].map(min), "Passou": tabela["passa"].astype(bool)})


def _mercado(mercado: str, dados: Mercado) -> None:
    nome, referencia = NOMES_DOS_MERCADOS[mercado]
    st.markdown(f"#### {nome}")
    portao = _portao(dados.portao, referencia)
    st.dataframe(portao, width="stretch", hide_index=True,
                 column_config={coluna: TRES_CASAS for coluna in portao.columns[1:-1]},
                 alt=f"Menor P e menor DSR de cada braço nas três fases em {nome}, e se passou no portão")
    st.dataframe(dados.anos, width="stretch", hide_index=True,
                 column_config={coluna: PERCENTUAL for coluna in dados.anos.columns[1:]},
                 alt=f"Retorno de cada ano em reais em {nome}: braços, Composto, Conservadora, BOVA11, CDI e Selic")
    data = pd.Timestamp(dados.carteiras["data"].max())
    st.caption(f"Informação, não recomendação: o que cada regra escolheu no último rebalanceio do teste, {data:%d/%m/%Y}.")
    st.dataframe(dados.carteiras.drop(columns="data"), width="stretch", hide_index=True,
                 column_config={"peso": PERCENTUAL}, alt=f"Papéis e pesos de cada braço em {nome}")


def secao(resultado: ResultadoHibrido | None) -> None:
    st.markdown("## QPO híbrido: previsão quântica e otimização")
    if resultado is None:
        st.info("O teste ainda não terminou: sem resultado para mostrar.")
        return
    st.warning(_veredito(resultado.mercados))
    st.caption("Nenhum dos quatro artigos prevê retornos: todos recebem o retorno esperado pronto. A previsão aqui é trabalho "
               "novo, e o circuito é comparado com o seu gêmeo clássico (Fourier), que usa os mesmos 729 termos. Só haveria "
               "ganho quântico se o circuito passasse no teste e também batesse esse gêmeo, com t de pelo menos 2.")
    st.markdown("### A previsão acerta a ordem das ações?")
    st.caption("IC de Spearman de cada mês contra o retorno dos 3 meses seguintes, t de Newey-West. Passa com IC de pelo menos "
               "0,02 e t de pelo menos 2.")
    st.dataframe(resultado.ic, width="stretch", hide_index=True, column_config=COLUNAS_DO_IC,
                 alt="IC médio, estatística t e resultado de cada modelo em cada mercado")
    st.dataframe(resultado.diferencas, width="stretch", hide_index=True,
                 column_config={"diferenca": st.column_config.NumberColumn("Diferença de IC", format="%+.4f"), "t": DUAS_CASAS},
                 alt="Diferença de IC entre o circuito e cada comparação, com estatística t")
    st.markdown("### As carteiras, com custo, em três fases")
    st.caption("Portão: P contra o Composto, a Conservadora e a referência, e DSR, todos de pelo menos 0,95 nas três fases. "
               "Rebalanceio a cada 3 meses, 10 bps por lado, antes de IR. Resultado passado, sem promessa de futuro.")
    for mercado, dados in resultado.mercados.items():
        _mercado(mercado, dados)
    st.markdown("### Os solvers quânticos, em instâncias pequenas e reais")
    st.caption("Simulados em CPU, sem hardware quântico. Razão 1 é o ótimo exato; \"reparado\" é o mesmo conserto aplicado "
               "também a amostras ao acaso. Um solver não cria retorno: no máximo empata com o ótimo exato.")
    st.dataframe(resultado.bancada, width="stretch", hide_index=True, column_config=COLUNAS_DA_BANCADA,
                 alt="Viabilidade, chance de ótimo e razão de aproximação de cada solver")
