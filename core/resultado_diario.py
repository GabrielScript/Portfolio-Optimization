"""
resultado_diario.py - The 20:00 report (the user, 2026-11-13: "me manda os resultados do composite no simulador Aleph
Quant e do MetaTrader 5 pontualmente as 20h pelo email"): the pure part of ``scripts/resultado_diario.py``.

The simulator closes the day only in the 22:30 upkeep (the lake's prices arrive at 21:45), so at 20:00 its Composite
account is a preview: the replay of its ledger on the app's total-return prices up to the last session, with one more
row for today where each price moves by its official close move read on the MT5 terminal's daily bars. A dividend
going ex today is the only thing the preview misses. The MT5 demo book is read as it is (cash plus shares at today's
close) and its value kept day by day. Email text is Portuguese, like the app.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


def com_fechamento_de_hoje(precos: pd.DataFrame, base: pd.Series, hoje: pd.Series, dia: pd.Timestamp) -> pd.DataFrame:
    """``precos`` (dates x tickers, total-return, to the last session) with a row for ``dia``: each ticker's last price
    times ``hoje / base`` (its official closes today and on ``precos``' last session); NaN where either is missing."""
    ultimo = precos.ffill().iloc[-1]
    linha = ultimo * (hoje.reindex(precos.columns) / base.reindex(precos.columns))
    return pd.concat([precos, linha.to_frame(dia).T])


@dataclass(frozen=True)
class Linha:
    patrimonio: float
    dia: float
    dia_reais: float
    desde_inicio: float
    inicio: pd.Timestamp


def linha(curva: pd.Series, inicial: float) -> Linha:
    """Today's value (``curva``'s last), its change on the session before (on ``inicial`` when it is the first) and on
    ``inicial``."""
    hoje = float(curva.iloc[-1])
    antes = float(curva.iloc[-2]) if len(curva) > 1 else inicial
    return Linha(hoje, hoje / antes - 1.0, hoje - antes, hoje / inicial - 1.0, curva.index[0])


def anotar_patrimonio(historico: pd.DataFrame, dia: pd.Timestamp, valor: float) -> pd.DataFrame:
    """The book's day-by-day values with ``dia`` set to ``valor`` (a rerun replaces the day)."""
    resto = historico[pd.to_datetime(historico["data"]) != dia] if len(historico) else historico
    nova = pd.DataFrame({"data": [dia], "patrimonio": [float(valor)]})
    return (pd.concat([resto, nova], ignore_index=True) if len(resto) else nova).assign(
        data=lambda t: pd.to_datetime(t["data"])).sort_values("data", ignore_index=True)


def serie_do_historico(historico: pd.DataFrame) -> pd.Series:
    return pd.Series(historico["patrimonio"].to_numpy(dtype=float), index=pd.to_datetime(historico["data"]))


def _reais(valor: float) -> str:
    return "R$ " + f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _sinal_reais(valor: float) -> str:
    return ("+" if valor >= 0 else "-") + _reais(abs(valor))


def _pct(valor: float) -> str:
    return f"{valor:+.2%}".replace(".", ",")


def _bloco(titulo: str, l: Linha) -> str:
    return "\n".join([titulo, f"  Patrimônio: {_reais(l.patrimonio)}", f"  Dia: {_pct(l.dia)} ({_sinal_reais(l.dia_reais)})",
                      f"  Desde {l.inicio:%d/%m/%Y}: {_pct(l.desde_inicio)}"])


def mensagem(dia: pd.Timestamp, simulador: Linha, bova_dia: float, livro: Linha | None, notas: Sequence[str],
             sem_livro: str = "", inicial_sim: float = 10_000.0) -> tuple[str, str]:
    """Subject and body of the 20:00 email."""
    assunto = f"Aleph Quant {dia:%d/%m/%Y}: Composite {_reais(simulador.patrimonio)} ({_pct(simulador.dia)})"
    if livro is not None:
        assunto += f", MT5 {_reais(livro.patrimonio)} ({_pct(livro.dia)})"
    blocos = [f"Resultado do dia {dia:%d/%m/%Y}, depois do fechamento da B3.",
              _bloco(f"Simulador (Composite, conta fictícia de {_reais(inicial_sim)} desde {simulador.inicio:%d/%m/%Y})",
                     simulador)
              + f"\n  BOVA11 no dia: {_pct(bova_dia)}"
              + "\n  Prévia pelos fechamentos oficiais no MetaTrader 5; o número oficial do simulador sai na manutenção"
                " das 22:30 (difere só se alguma ação ficou ex-dividendo hoje)."]
    if livro is not None:
        blocos.append(_bloco("MetaTrader 5 (conta demo da XP, Composite, R$ 100.000,00 de início)", livro))
    else:
        blocos.append(f"MetaTrader 5: sem leitura ({sem_livro}).")
    if notas:
        blocos.append("Observações:\n" + "\n".join(f"  {nota}" for nota in notas))
    blocos.append("Nenhum dinheiro real foi movimentado. Isto não é recomendação de investimento.")
    return assunto, "\n\n".join(blocos)
