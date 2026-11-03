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
RODAPE = ("Simulador do Aleph Neumann (conta fictícia de R$ 10.000): Composite, Conservadora e a carteira do "
          "DeepSeek. Informação, não é recomendação de investimento.")


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


def mensagem_das_ordens(ordens: pd.DataFrame) -> tuple[str, str]:
    """Subject and body: the non-zero orders of each portfolio, buys and sells with price and amount."""
    feitas = ordens[ordens["ordem"] != 0]
    dias = sorted(pd.to_datetime(feitas["data"]).dt.date.unique())
    quando = ", ".join(f"{dia:%d/%m/%Y}" for dia in dias)
    assunto = f"Aleph Neumann: {len(feitas)} ordem(ns) no simulador em {quando}"
    blocos = []
    for carteira, linhas in feitas.groupby("carteira", sort=False):
        itens = [f"  {'COMPRA' if linha.ordem > 0 else 'VENDA'} {abs(int(linha.ordem))} {linha.ticker} a "
                 f"{_reais(float(linha.preco))} ({_reais(abs(linha.ordem) * float(linha.preco))})"
                 for linha in linhas.itertuples()]
        blocos.append("\n".join([str(carteira), *itens]))
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
