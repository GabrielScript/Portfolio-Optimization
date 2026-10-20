"""
impostos.py - IR de pessoa física sobre ganho em bolsa (S4), para a curva líquida de uma estratégia fora da amostra.

Regras (fontes primárias lidas em 20/10/2026): 15% sobre o ganho líquido do mês no mercado à vista; isenção do ganho
com ações quando as vendas de ações no mês somam até R$ 20 mil (Lei 11.033/2004, art. 3º, I); ETF de ações (BOVA11)
sem isenção. A Lei 15.270/2025 não muda o ganho em bolsa (os 10% na fonte valem para dividendos acima de R$ 50 mil por
mês de uma mesma empresa, e o IRPFM só acima de R$ 600 mil por ano).
"""

from __future__ import annotations

from dataclasses import dataclass

ALIQUOTA = 0.15
ISENCAO_DE_VENDAS = 20_000.0    # vendas de ações à vista no mês, em R$


@dataclass(frozen=True)
class Apuracao:
    imposto: float              # a pagar pelo mês (DARF)
    prejuizo_acumulado: float   # para compensar nos meses seguintes (>= 0)


def apurar_mes(vendas: float, resultado: float, prejuizo_anterior: float,
               isencao: float = ISENCAO_DE_VENDAS) -> Apuracao:
    """
    IR do mês sobre o ``resultado`` (ganhos menos perdas das vendas do mês, em R$): prejuízo do mês acumula, isento ou
    não; com ``vendas`` até ``isencao`` o ganho é isento e não gasta o prejuízo acumulado; acima, 15% sobre o ganho
    depois de compensar o prejuízo. ETF: ``isencao = 0``. (Fontes secundárias concordam que o prejuízo de mês isento
    compensa ganho tributável depois e que o acumulado não abate ganho isento.)
    """
    if resultado < 0.0:
        return Apuracao(0.0, prejuizo_anterior - resultado)
    if vendas <= isencao:
        return Apuracao(0.0, prejuizo_anterior)
    return Apuracao(ALIQUOTA * max(0.0, resultado - prejuizo_anterior), max(0.0, prejuizo_anterior - resultado))
