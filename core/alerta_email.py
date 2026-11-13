"""
alerta_email.py - Email alerts of the fictitious account (DECISOES 240): each buy or sell the simulator records goes to
the user's Gmail through an app password (``GMAIL_USER`` and ``GMAIL_PASSWORD_APP`` in the environment or in the
``ai-hedge-fund/.env``; ``ALERTA_EMAIL_PARA`` changes the recipient, the sender by default). Message text is
Portuguese, like the app. The password is only ever sent to the SMTP login.
"""

from __future__ import annotations

import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

import pandas as pd

from core.segredos import ARQUIVO_ENV, segredo

SERVIDOR, PORTA = "smtp.gmail.com", 587
PRAZO_S = 30
RODAPE = ("Cada carteira começou com R$ 10.000 fictícios. Informação para acompanhar o simulador, não é recomendação de "
          "investimento.")


@dataclass(frozen=True)
class Credenciais:
    usuario: str
    senha: str
    destinatario: str


def credenciais(arquivo: Path = ARQUIVO_ENV) -> Credenciais | None:
    """The Gmail sender, app password and recipient (``core.segredos.segredo``: the .env first); None without a user
    or a password."""
    usuario, senha = segredo("GMAIL_USER", arquivo), segredo("GMAIL_PASSWORD_APP", arquivo)
    if not usuario or not senha:
        return None
    return Credenciais(usuario, senha, segredo("ALERTA_EMAIL_PARA", arquivo) or usuario)


def _reais(valor: float) -> str:
    return "R$ " + f"{valor:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


# each simulator portfolio's name in the email and what it is (the user, 2026-11-05: a bare "DeepSeek" was confusing)
CARTEIRAS = {
    "Composto": ("Composite", "a carteira principal do Aleph Quant, 15 ações em pesos iguais"),
    "Conservadora": ("Conservadora", "a regra mais simples, 15 ações em pesos iguais"),
    "DeepSeek": ("DeepSeek", "as ações que o assessor DeepSeek julgou favoráveis, entre as da Composite e da "
                             "Conservadora, em pesos iguais"),
    "Controle": ("Controle", "todas as ações que o DeepSeek avaliou, em pesos iguais, sem o filtro do modelo"),
    "Top-5": ("Composite top-5", "as 5 ações de maior nota da Composite, no máximo uma por setor, 20% cada"),
    "12/3": ("12/3 com BDRs", "as 12 primeiras da Composite mais as 3 BDRs americanas de maior momentum, 1/15 cada, "
                              "com 15% de Imposto de Renda sobre o lucro das BDRs"),
    "Fermiônica": ("Fermiônica", "15 ações escolhidas pela nota da Composite e pela baixa correlação entre elas, "
                                 "em pesos iguais"),
    "Composto 80/20": ("Composite 80/20", "80% a Composite e 20% em Bitcoin pelo ETF BITH11, de volta a 80/20 todo "
                                          "mês, com 15% de Imposto de Renda sobre o lucro do ETF"),
}


def _lista(nomes: list[str]) -> str:
    return nomes[0] if len(nomes) == 1 else ", ".join(nomes[:-1]) + " e " + nomes[-1]


def mensagem_das_ordens(ordens: pd.DataFrame) -> tuple[str, str]:
    """Subject and body: what the email is (the fictitious account, the close the orders were filled at), then each
    portfolio's name, what it is, its non-zero buys and sells with price and amount and its totals, and the portfolios
    without orders on that close."""
    feitas = ordens[ordens["ordem"] != 0]
    quando = ", ".join(f"{dia:%d/%m/%Y}" for dia in sorted(pd.to_datetime(feitas["data"]).dt.date.unique()))
    assunto = f"Aleph Quant (simulador): {len(feitas)} {'ordem' if len(feitas) == 1 else 'ordens'} no fechamento de {quando}"
    blocos = [f"Conta fictícia do simulador do Aleph Quant: estas ordens foram executadas ao preço de fechamento de "
              f"{quando}. Nenhum dinheiro real foi movimentado."]
    for carteira, linhas in feitas.groupby("carteira", sort=False):
        nome, descricao = CARTEIRAS.get(str(carteira), (str(carteira), ""))
        valores = linhas["ordem"].abs() * linhas["preco"].astype(float)
        itens = [f"  {'Compra' if linha.ordem > 0 else 'Venda'}: {abs(int(linha.ordem))} {linha.ticker} a "
                 f"{_reais(float(linha.preco))} = {_reais(valor)}" for linha, valor in zip(linhas.itertuples(), valores)]
        total = (f"  Total: compras {_reais(float(valores[linhas['ordem'] > 0].sum()))}, vendas "
                 f"{_reais(float(valores[linhas['ordem'] < 0].sum()))}, custos {_reais(float(linhas['custo'].sum()))}")
        blocos.append("\n".join([f"{nome}: {descricao}" if descricao else nome, *itens, total]))
    paradas = [nome for carteira, (nome, _) in CARTEIRAS.items() if carteira not in set(feitas["carteira"])]
    if paradas:
        blocos.append(f"Sem ordens nesta data: {_lista(paradas)}.")
    return assunto, "\n\n".join(blocos) + f"\n\n{RODAPE}\n"


def enviar(assunto: str, corpo: str, cred: Credenciais, smtp=smtplib.SMTP) -> None:
    """One plain-text email through Gmail's SMTP with STARTTLS (the app password's spaces are dropped)."""
    mensagem = EmailMessage()
    mensagem["From"], mensagem["To"], mensagem["Subject"] = cred.usuario, cred.destinatario, assunto
    mensagem.set_content(corpo)
    with smtp(SERVIDOR, PORTA, timeout=PRAZO_S) as conexao:
        conexao.starttls(context=ssl.create_default_context())
        conexao.login(cred.usuario, cred.senha.replace(" ", ""))
        conexao.send_message(mensagem)
