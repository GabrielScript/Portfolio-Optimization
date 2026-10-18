"""
ordens.py - Plano de compra: pesos alvo + cotações -> quantidades inteiras de ações (sem UI).

Compra a partir de carteira zerada (não considera posições atuais). Regras:
- o custo estimado (bps sobre o valor comprado) é reservado antes da divisão,
  então ``investido + custo <= orçamento``;
- quantidades inteiras; cada posição tem 0 ou pelo menos ``quantidade_minima`` ações;
- nenhuma posição passa de ``peso_maximo`` do orçamento (o mesmo teto da otimização);
- a sobra não leva um grupo (ex.: setor) além do teto dele; o piso segue o alvo, que já
  respeita esse teto quando vem do otimizador;
- compra o piso de cada alvo; com a sobra, compra a ação mais abaixo do valor alvo,
  uma por vez (ou ``quantidade_minima`` de uma vez quando a posição está zerada),
  até nada caber no caixa e nos tetos. Uma posição pode passar do alvo: para quem quer
  investir o valor todo, dinheiro parado é pior que um desvio pequeno.

Uma ação pode ficar de fora; ``PlanoCompra.motivos_fora`` diz por quê: a quantidade
mínima passa do teto por ação, não cabe no que falta para o teto do grupo ou não cabe
no caixa que sobrou. ``acoes_que_cabem`` tira do
universo, antes da otimização, as que passariam do teto. Na B3, múltiplos de 100 vão
no lote padrão; o resto, no mercado fracionário (ticker com sufixo F).

Ponto flutuante: as comparações em reais usam a tolerância absoluta
``TOLERANCIA_REAIS``. Uma compra que cabe exatamente no caixa ou no teto (ex.:
teto 0,29 × R$ 100 = 28,999999999999996) não é recusada, e uma sobra dentro da
tolerância vira zero. Absoluta, e não relativa ao orçamento, para que
``investido + custo`` nunca passe do orçamento mais que essa tolerância.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
import pandas as pd

from core.config import LimiteGrupo
from core.optimization import encher_ate_os_tetos

LOTE_PADRAO = 100
TOLERANCIA_SOMA_PESOS = 1e-6
TOLERANCIA_REAIS = 1e-9


class MotivoFora(str, Enum):
    """Por que uma ação com peso alvo > 0 ficou sem nenhuma unidade comprada."""

    TETO = "teto"
    GRUPO = "grupo"
    CAIXA = "caixa"


@dataclass(frozen=True)
class RegrasCompra:
    orcamento: float
    custo_bps: float
    peso_maximo: float
    quantidade_minima: int = 2
    grupos: tuple[LimiteGrupo, ...] = ()        # ex.: setores; teto em fração do orçamento

    def __post_init__(self) -> None:
        if not self.orcamento > 0:
            raise ValueError("orcamento deve ser > 0.")
        if self.custo_bps < 0:
            raise ValueError("custo_bps não pode ser negativo.")
        if not 0 < self.peso_maximo <= 1:
            raise ValueError("peso_maximo deve estar em (0, 1].")
        if self.quantidade_minima < 1:
            raise ValueError("quantidade_minima deve ser >= 1.")


@dataclass(frozen=True)
class PlanoCompra:
    """``itens``: ticker x (peso_alvo, cotacao, quantidade, lote_padrao, fracionario, valor, peso_efetivo)."""

    itens: pd.DataFrame
    investido: float
    custo_estimado: float
    sobra: float
    motivos_fora: dict[str, MotivoFora]     # peso alvo > 0, mas nenhuma ação comprada

    @property
    def fora(self) -> tuple[str, ...]:
        return tuple(self.motivos_fora)


@dataclass(frozen=True)
class _Tetos:
    """Tetos em reais: por ação e por grupo. ``membros``: matriz grupos × ações (bool)."""

    por_acao: float
    por_grupo: np.ndarray
    membros: np.ndarray

    @classmethod
    def das_regras(cls, tickers: pd.Index, regras: RegrasCompra) -> _Tetos:
        membros = np.array([[t in grupo.tickers for t in tickers] for grupo in regras.grupos],
                           dtype=bool).reshape(len(regras.grupos), len(tickers))
        por_grupo = np.array([grupo.teto * regras.orcamento for grupo in regras.grupos])
        return cls(regras.peso_maximo * regras.orcamento, por_grupo, membros)

    def folga_dos_grupos(self, valores: np.ndarray) -> np.ndarray:
        """Por ação, quanto ainda cabe no grupo mais cheio dela (infinito se ela não tem grupo)."""
        folga = self.por_grupo - self.membros @ valores
        return np.where(self.membros, folga[:, None], np.inf).min(axis=0, initial=np.inf)


def limites_do_plano(grupos: tuple[LimiteGrupo, ...], alvo: pd.Series) -> tuple[LimiteGrupo, ...]:
    """
    Tetos de grupo para o plano: o de cada grupo sobe até o peso alvo dele, se for maior. O HRP
    ignora os tetos de grupo, e o plano nunca corta um alvo; só impede a sobra de passar do teto.
    No máximo 100%: um alvo reescalado pode somar 1 + ruído de ponto flutuante.
    """
    return tuple(replace(grupo, teto=min(1.0, max(grupo.teto, float(alvo[alvo.index.isin(grupo.tickers)].sum()))))
                 for grupo in grupos)


def redistribuir_sem_cotacao(pesos: pd.Series, com_cotacao: pd.Series | np.ndarray, peso_maximo: float,
                             grupos: tuple[LimiteGrupo, ...]) -> pd.Series:
    """
    Alvo do plano sem as ações sem cotação: o peso delas vai para as outras, na proporção do otimizador, só até o
    teto por ação e o teto de cada grupo. O que não couber fica em caixa (soma < 1). Um alvo que já passava de um
    teto (HRP) não é cortado, só deixa de receber.
    """
    restantes = pesos[np.asarray(com_cotacao, dtype=bool)]
    return encher_ate_os_tetos(restantes, min(1.0, float(pesos.sum())), peso_maximo, grupos)


def acoes_que_cabem(precos: pd.Series, regras: RegrasCompra) -> list[str]:
    """
    Tickers com preço > 0 cuja quantidade mínima cabe no teto por ação (``peso_maximo`` ×
    orçamento), na ordem de ``precos``. Uso: tirar do universo, ANTES de otimizar, as ações que
    o plano de compra nunca conseguiria comprar; senão o peso delas viraria caixa. Preço NaN sai.
    """
    teto = regras.peso_maximo * regras.orcamento
    cabem = (precos > 0) & (regras.quantidade_minima * precos <= teto + TOLERANCIA_REAIS)
    return precos.index[cabem].tolist()


def plano_de_compra(pesos: pd.Series, cotacoes: pd.Series, regras: RegrasCompra) -> PlanoCompra:
    alvo = _validar_pesos(pesos)
    precos = _validar_cotacoes(alvo.index, cotacoes)
    taxa = regras.custo_bps / 10_000
    disponivel = regras.orcamento / (1.0 + taxa)
    valores_alvo = alvo.to_numpy() * disponivel
    tetos = _Tetos.das_regras(alvo.index, regras)

    quantidades = np.floor(np.minimum(valores_alvo, tetos.por_acao) / precos)
    quantidades[quantidades < regras.quantidade_minima] = 0.0
    quantidades = _distribuir_sobra(quantidades, valores_alvo, precos, disponivel, tetos,
                                    regras.quantidade_minima)

    itens = _montar_itens(alvo, precos, quantidades.astype(int), regras.orcamento)
    investido = float(itens["valor"].sum())
    custo = investido * taxa
    sobra = _zerar_ruido(regras.orcamento - investido - custo)
    motivos = _motivos_fora(itens, tetos, regras.quantidade_minima)
    return PlanoCompra(itens, investido, custo, sobra, motivos)


def _motivos_fora(itens: pd.DataFrame, tetos: _Tetos, minimo: int) -> dict[str, MotivoFora]:
    """
    O guloso só para quando nada cabe; então a quantidade mínima de uma ação de fora passou do
    teto por ação, da folga do grupo dela ou do caixa que sobrou, nessa ordem de precedência. O
    teto por ação vem antes porque, com ele barrando, só um peso máximo maior resolve.
    """
    custos_minimos = minimo * itens["cotacao"].to_numpy()
    folgas = tetos.folga_dos_grupos(itens["valor"].to_numpy())
    motivos = {}
    for posicao in np.flatnonzero(itens["quantidade"].to_numpy() == 0):
        custo = custos_minimos[posicao]
        if custo > tetos.por_acao + TOLERANCIA_REAIS:
            motivos[itens.index[posicao]] = MotivoFora.TETO
        elif custo > folgas[posicao] + TOLERANCIA_REAIS:
            motivos[itens.index[posicao]] = MotivoFora.GRUPO
        else:
            motivos[itens.index[posicao]] = MotivoFora.CAIXA
    return motivos


def _zerar_ruido(valor: float) -> float:
    """Valor em reais dentro da tolerância vira 0,0 (evita "R$ -0.00" na tela)."""
    return 0.0 if abs(valor) <= TOLERANCIA_REAIS else valor


def _validar_pesos(pesos: pd.Series) -> pd.Series:
    """
    Pesos positivos. Uma soma acima de 1 aceita pela tolerância é reescalada
    para 1; senão o piso das quantidades poderia gastar além do orçamento.
    """
    if (pesos < 0).any() or pesos.sum() > 1.0 + TOLERANCIA_SOMA_PESOS:
        raise ValueError(f"pesos devem ser >= 0 e somar no máximo 100% (soma {pesos.sum():.4f}).")
    positivos = pesos[pesos > 0]
    soma = positivos.sum()
    return positivos / soma if soma > 1.0 else positivos


def _validar_cotacoes(tickers: pd.Index, cotacoes: pd.Series) -> np.ndarray:
    alinhadas = cotacoes.reindex(tickers).astype(float)
    invalidas = alinhadas.index[~(alinhadas > 0)].tolist()      # NaN também cai aqui
    if invalidas:
        raise ValueError(f"Cotação ausente ou inválida: {', '.join(invalidas)}.")
    return alinhadas.to_numpy()


def _distribuir_sobra(quantidades: np.ndarray, valores_alvo: np.ndarray, precos: np.ndarray,
                      disponivel: float, tetos: _Tetos, minimo: int) -> np.ndarray:
    """
    Guloso de enchimento por nível: compra a ação mais abaixo do valor alvo (déficit em reais, que
    pode ser negativo) entre as que cabem no caixa, no teto por ação e no teto do grupo, até nenhuma
    caber. Para um gasto fixo, igualar os déficits minimiza a soma dos quadrados dos desvios; escolher
    o passo que menos piora o desvio seria míope (o passo barato de uma ação já acima do alvo venceria
    sempre). Cada passo gasta pelo menos uma cotação > 0, então termina.
    """
    quantidades = quantidades.copy()
    while True:
        caixa = disponivel - float(quantidades @ precos)
        passos = np.where(quantidades == 0, minimo, 1)
        custo_passo = passos * precos
        valor_atual = quantidades * precos
        cabe_no_caixa = custo_passo <= caixa + TOLERANCIA_REAIS
        cabe_no_teto = valor_atual + custo_passo <= tetos.por_acao + TOLERANCIA_REAIS
        cabe_no_grupo = custo_passo <= tetos.folga_dos_grupos(valor_atual) + TOLERANCIA_REAIS
        viavel = cabe_no_caixa & cabe_no_teto & cabe_no_grupo
        if not viavel.any():
            return quantidades
        deficit = valores_alvo - valor_atual
        melhor = int(np.argmax(np.where(viavel, deficit, -np.inf)))    # empate: primeiro ticker
        quantidades[melhor] += passos[melhor]


def _montar_itens(alvo: pd.Series, precos: np.ndarray, quantidades: np.ndarray,
                  orcamento: float) -> pd.DataFrame:
    valores = quantidades * precos
    return pd.DataFrame({
        "peso_alvo": alvo.to_numpy(),
        "cotacao": precos,
        "quantidade": quantidades,
        "lote_padrao": quantidades // LOTE_PADRAO * LOTE_PADRAO,
        "fracionario": quantidades % LOTE_PADRAO,
        "valor": valores,
        "peso_efetivo": valores / orcamento,
    }, index=alvo.index)
