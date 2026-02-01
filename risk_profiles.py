"""
risk_profiles.py - Definição dos perfis de risco do investidor
TCC: Otimização de Carteiras de Investimentos
Autor: Gabriel

Baseado na Teoria Moderna do Portfólio de Markowitz
"""

from dataclasses import dataclass
from typing import Dict

# Taxa livre de risco (Selic atual)
TAXA_SELIC = 0.1325  # 13.25% a.a.

@dataclass
class PerfilRisco:
    """Classe que define um perfil de risco do investidor."""
    nome: str
    descricao: str
    volatilidade_maxima: float  # Desvio padrão máximo anual
    objetivo: str  # 'min_volatility', 'max_sharpe', 'max_return'
    cor_primaria: str
    cor_secundaria: str
    icone: str

# Definição dos perfis de risco
PERFIS_RISCO: Dict[str, PerfilRisco] = {
    "Conservador": PerfilRisco(
        nome="Conservador",
        descricao="Prioriza segurança e estabilidade. Aceita retornos menores em troca de menor volatilidade.",
        volatilidade_maxima=0.20,  # 15% a.a.
        objetivo="min_volatility",
        cor_primaria="#00D4AA",  # Verde água
        cor_secundaria="#004D40",
        icone="🛡️"
    ),
    "Moderado": PerfilRisco(
        nome="Moderado",
        descricao="Equilibra risco e retorno. Busca o melhor índice de Sharpe possível.",
        volatilidade_maxima=0.35,  # 25% a.a.
        objetivo="max_sharpe",
        cor_primaria="#FFB74D",  # Laranja
        cor_secundaria="#E65100",
        icone="⚖️"
    ),
    "Agressivo": PerfilRisco(
        nome="Agressivo",
        descricao="Prioriza maximizar retornos. Tolera alta volatilidade em busca de ganhos maiores.",
        volatilidade_maxima=0.50,  # 40% a.a.
        objetivo="max_return",
        cor_primaria="#FF5252",  # Vermelho
        cor_secundaria="#B71C1C",
        icone="🚀"
    )
}

def get_perfil(nome: str) -> PerfilRisco:
    """
    Retorna o perfil de risco pelo nome.
    
    Args:
        nome: Nome do perfil ('Conservador', 'Moderado', 'Agressivo')
        
    Returns:
        Objeto PerfilRisco correspondente
    """
    return PERFIS_RISCO.get(nome, PERFIS_RISCO["Moderado"])

def get_nomes_perfis() -> list:
    """Retorna lista com nomes dos perfis disponíveis."""
    return list(PERFIS_RISCO.keys())

def get_parametros_otimizacao(perfil: PerfilRisco) -> dict:
    """
    Retorna os parâmetros para otimização baseado no perfil.
    
    Args:
        perfil: Objeto PerfilRisco
        
    Returns:
        Dict com parâmetros de otimização
    """
    return {
        "objetivo": perfil.objetivo,
        "volatilidade_maxima": perfil.volatilidade_maxima,
        "taxa_livre_risco": TAXA_SELIC,
        "peso_maximo_ativo": 0.20,  # Máximo 20% por ativo
        "peso_minimo_ativo": 0.0    # Mínimo 0% por ativo
    }
