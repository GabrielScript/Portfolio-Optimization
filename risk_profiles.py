"""
risk_profiles.py - Definição dos perfis de risco do investidor
TCC: Otimização de Carteiras de Investimentos
Autor: Gabriel Estrela Lopes

Baseado na Teoria Moderna do Portfólio de Markowitz
"""

from dataclasses import dataclass
from typing import Dict
import logging
import requests
import streamlit as st

logger = logging.getLogger(__name__)

# Fallback local caso a API do BCB esteja indisponível
_TAXA_SELIC_FALLBACK = 0.1475  # 14.75% a.a. (Março 2026)


@st.cache_data(ttl=86400)  # Cache de 24 horas
def obter_taxa_selic() -> float:
    """
    Busca a taxa Selic Meta atual via API pública do Banco Central do Brasil.
    
    Endpoint: SGS (Sistema Gerenciador de Séries Temporais)
    Série 432: Taxa de juros - Meta Selic definida pelo Copom (% a.a.)
    
    Returns:
        Taxa Selic anual em formato decimal (ex: 0.1475 para 14.75%)
    """
    try:
        url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        dados = resp.json()
        taxa = float(dados[0]["valor"]) / 100
        logger.info(f"Taxa Selic obtida via BCB: {taxa*100:.2f}%")
        return taxa
    except Exception as e:
        logger.warning(f"Falha ao buscar Selic via BCB: {e}. Usando fallback: {_TAXA_SELIC_FALLBACK*100:.2f}%")
        return _TAXA_SELIC_FALLBACK


# Taxa livre de risco (Selic atual, obtida dinamicamente)
TAXA_SELIC = obter_taxa_selic()


@dataclass
class PerfilRisco:
    """Classe que define um perfil de risco do investidor."""
    nome: str
    descricao: str
    volatilidade_maxima: float
    objetivo: str  # 'min_volatility', 'max_sharpe', 'max_return'
    cor_primaria: str
    cor_secundaria: str
    icone: str


PERFIS_RISCO: Dict[str, PerfilRisco] = {
    "Conservador": PerfilRisco(
        nome="Conservador",
        descricao="Prioriza segurança e estabilidade. Aceita retornos menores em troca de menor volatilidade.",
        volatilidade_maxima=0.20,
        objetivo="min_volatility",
        cor_primaria="#00D4AA",
        cor_secundaria="#004D40",
        icone="🛡️"
    ),
    "Moderado": PerfilRisco(
        nome="Moderado",
        descricao="Equilibra risco e retorno. Busca o melhor índice de Sharpe possível.",
        volatilidade_maxima=0.35,
        objetivo="max_sharpe",
        cor_primaria="#FFB74D",
        cor_secundaria="#E65100",
        icone="⚖️"
    ),
    "Agressivo": PerfilRisco(
        nome="Agressivo",
        descricao="Prioriza maximizar retornos. Tolera alta volatilidade em busca de ganhos maiores.",
        volatilidade_maxima=0.40,
        objetivo="max_return",
        cor_primaria="#FF5252",
        cor_secundaria="#B71C1C",
        icone="🚀"
    )
}


def get_perfil(nome: str) -> PerfilRisco:
    """Retorna o perfil de risco pelo nome."""
    return PERFIS_RISCO.get(nome, PERFIS_RISCO["Moderado"])


def get_nomes_perfis() -> list:
    """Retorna lista com nomes dos perfis disponíveis."""
    return list(PERFIS_RISCO.keys())


def get_parametros_otimizacao(perfil: PerfilRisco) -> dict:
    """Retorna os parâmetros para otimização baseado no perfil."""
    return {
        "objetivo": perfil.objetivo,
        "volatilidade_maxima": perfil.volatilidade_maxima,
        "taxa_livre_risco": TAXA_SELIC,
        "peso_maximo_ativo": 0.20,
        "peso_minimo_ativo": 0.0
    }
