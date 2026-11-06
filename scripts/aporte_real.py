"""
The real-money ticket of 19/12/2026 (DECISOES 261 and 271): R$ 500 into the Composite's last record, whole shares
(``core.ordens.plano_de_compra``, at most 20% per stock, the plan's cost reserve), every order in B3's closing call
auction so the fill is the close the backtests and the fictitious accounts use. Lots of 100 go to the standard market,
the rest to the fractional market (ticker with an F). It prints the ticket and writes it to ``docs/aporte-real-<day>.md``;
the user places the orders at the broker (nothing here sends an order).

The amount follows the user's rule of 2026-11-06 (DECISOES 283, ``valor_do_aporte``) on the fictitious
"Composto + Sentinela" account's last close (R$ 10,000 at the start): no buy below R$ 11,000; it also prints the alpha against "Composto (base)".

Usage: python -X utf8 scripts/aporte_real.py [--saldo 12800] [--valor 500] [--dia 2026-12-19]
The quotes are the last closes in the app's price files; run it on the day, before the auction (16:55-17:00).
``--saldo`` replaces the account's balance, ``--valor`` the rule's amount.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core import precos_do_app, simulador  # noqa: E402
from core.ordens import PlanoCompra, RegrasCompra, plano_de_compra  # noqa: E402
from core.risco import LIMITES  # noqa: E402

ARQUIVO_REGISTRO = Path("paper-v3/composto.csv")
LOTE = 100
ORDEM = "compra no leilão de fechamento"
FAIXAS = ((13_000.0, 1_000.0), (12_500.0, 750.0), (11_000.0, 500.0))     # (balance from, amount): the user, 2026-11-06
CONTA = "Composto + Sentinela"
BASE = "Composto (base)"


def valor_do_aporte(saldo: float) -> float:
    """The real amount for the fictitious account's balance: the highest tier the balance reaches, 0 below them all."""
    return next((valor for piso, valor in FAIXAS if saldo >= piso), 0.0)


def alfa_contra_a_base(conta: float, base: float) -> float:
    """What the agent added: the agent's account over the Composite alone (same R$ 10,000, same days), minus 1."""
    return conta / base - 1


def saldos(dia: date, carregar=None) -> tuple[float, float | None]:
    """The agent's account and the Composite alone at their last close."""
    if carregar is None:
        from ui import servicos
        carregar = servicos.carregar_sentinela.__wrapped__
    painel = carregar(dia, 0)
    if painel is None or painel.curvas is None or painel.curvas.empty:
        raise SystemExit("A conta fictícia da Sentinela ainda não tem saldo; use --saldo")
    curvas = painel.curvas.ffill()
    return float(curvas[CONTA].iloc[-1]), (float(curvas[BASE].iloc[-1]) if BASE in curvas else None)


def saldo_da_conta(dia: date, carregar=None) -> float:
    """The fictitious "Composto + Sentinela" account at its last close."""
    return saldos(dia, carregar)[0]


def linhas_da_ordem(ticker: str, quantidade: int, preco: float) -> list[dict]:
    """One line per market: the standard lot (multiples of 100) and the fractional rest (ticker + F)."""
    padrao, resto = divmod(int(quantidade), LOTE)
    linhas = []
    for ativo, parte in ((ticker, padrao * LOTE), (f"{ticker}F", resto)):
        if parte:
            linhas.append({"Ativo": ativo, "Quantidade": parte, "Preço de referência (R$)": round(float(preco), 2),
                           "Valor (R$)": round(parte * float(preco), 2), "Ordem": ORDEM})
    return linhas


def boleta(pesos: pd.Series, cotacoes: pd.Series, orcamento: float) -> tuple[pd.DataFrame, PlanoCompra]:
    regras = RegrasCompra(orcamento=orcamento, custo_bps=simulador.CUSTO_BPS, peso_maximo=LIMITES.peso_maximo_acao,
                          quantidade_minima=1)
    plano = plano_de_compra(pesos, cotacoes.reindex(pesos.index), regras)
    linhas = [linha for ticker, item in plano.itens.iterrows() if item["quantidade"] > 0
              for linha in linhas_da_ordem(ticker, int(item["quantidade"]), float(item["cotacao"]))]
    colunas = ["Ativo", "Quantidade", "Preço de referência (R$)", "Valor (R$)", "Ordem"]
    tabela = pd.DataFrame(linhas, columns=colunas)
    tabela["Valor (R$)"] = tabela["Quantidade"] * tabela["Preço de referência (R$)"]
    return tabela, plano


def _br(valor: float) -> str:
    return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _markdown(dia: date, registrada: pd.Timestamp, ultimo: pd.Timestamp, tabela: pd.DataFrame, plano: PlanoCompra,
              orcamento: float) -> str:
    linhas = [f"# Aporte real de {dia:%d/%m/%Y}: R$ {_br(orcamento)} na Composite", "",
              f"Registro da Composite: {registrada:%d/%m/%Y}. Cotações de referência: fechamento de {ultimo:%d/%m/%Y}.",
              "Todas as ordens no leilão de fechamento da B3 (16h55-17h), tipo \"ordem no fechamento\" na corretora; "
              "lote de 100 no mercado padrão, o resto no fracionário (ticker com F). A quantidade é a de referência: se o "
              "preço do dia mudar muito, rode o script de novo antes do leilão.",
              "Confirme na corretora que o fracionário aceita \"ordem no fechamento\"; se não aceitar, use ordem "
              "limitada ao último preço pouco antes das 16h55 (o preço fica perto do fechamento, mas não igual).", "",
              "| Ativo | Quantidade | Preço de referência (R$) | Valor (R$) |", "|---|---:|---:|---:|"]
    linhas += [f"| {linha['Ativo']} | {linha['Quantidade']} | {_br(linha['Preço de referência (R$)'])} | "
               f"{_br(linha['Valor (R$)'])} |" for _, linha in tabela.iterrows()]
    linhas += ["", f"Investido: R$ {_br(plano.investido)}; reserva de custos: R$ {_br(plano.custo_estimado)}; "
                   f"sobra em caixa: R$ {_br(plano.sobra)}."]
    if plano.fora:
        linhas.append("Fora da compra (não cabem no valor ou no teto de 20% por ação): " + ", ".join(plano.fora) + ".")
    linhas += ["", "Informação para o investidor decidir; não é recomendação. A Sentinela só age na conta fictícia."]
    return "\n".join(linhas) + "\n"


def main(argv: list[str] | None = None) -> int:
    argumentos = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    argumentos.add_argument("--valor", type=float, default=None)
    argumentos.add_argument("--saldo", type=float, default=None)
    argumentos.add_argument("--dia", type=date.fromisoformat, default=date(2026, 12, 19))
    opcoes = argumentos.parse_args(argv)
    saldo, base = (opcoes.saldo, None) if opcoes.saldo is not None else saldos(date.today())
    valor = opcoes.valor if opcoes.valor is not None else valor_do_aporte(saldo)
    print(f"Conta fictícia {CONTA}: R$ {_br(saldo)} -> aporte real de R$ {_br(valor)}")
    if base:
        print(f"{BASE}: R$ {_br(base)}; alfa do agente: {alfa_contra_a_base(saldo, base):+.2%}")
    if valor <= 0:
        print("Abaixo de R$ 11.000: pela regra, nenhum aporte real.")
        return 0
    registro = pd.read_csv(RAIZ / ARQUIVO_REGISTRO, parse_dates=["data"])
    registrada = registro["data"].max()
    atual = registro[registro["data"] == registrada].set_index("ticker")["peso"]
    precos = precos_do_app.ler_fechamentos([f"{papel}.SA" for papel in atual.index], (registrada - pd.Timedelta(
        days=10)).date(), date.today()).rename(columns=lambda papel: papel.removesuffix(".SA")).ffill()
    tabela, plano = boleta(atual / atual.sum(), precos.iloc[-1], valor)
    texto = _markdown(opcoes.dia, registrada, precos.index[-1], tabela, plano, valor)
    destino = RAIZ / "docs" / f"aporte-real-{opcoes.dia:%Y-%m-%d}.md"
    destino.write_text(texto, encoding="utf-8", newline="\n")
    print(texto)
    print(f"(gravado em {destino.relative_to(RAIZ).as_posix()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
