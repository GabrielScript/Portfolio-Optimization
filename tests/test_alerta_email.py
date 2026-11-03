"""The simulator's email alerts (``core.alerta_email``, DECISOES 240): every buy or sell of the fictitious account goes
to the user's Gmail through an app password."""

import pandas as pd
import pytest

from core import alerta_email as ae


def _ordens() -> pd.DataFrame:
    return pd.DataFrame({"data": pd.to_datetime(["2026-11-04"] * 3), "carteira": ["DeepSeek", "DeepSeek", "Composto"],
                         "ticker": ["PETR4", "VALE3", "ITSA4"], "ordem": [54, -10, 0],
                         "preco": [33.0, 61.25, 13.3], "custo": [1.78, 0.61, 0.0]})


def test_the_message_lists_each_buy_and_sell_per_portfolio_in_brazilian_numbers_and_skips_zero_orders():
    assunto, corpo = ae.mensagem_das_ordens(_ordens())
    assert assunto == "Aleph Neumann: 2 ordem(ns) no simulador em 04/11/2026"
    assert "DeepSeek\n  COMPRA 54 PETR4 a R$ 33,00 (R$ 1.782,00)\n  VENDA 10 VALE3 a R$ 61,25 (R$ 612,50)" in corpo
    assert "ITSA4" not in corpo and "Composto" not in corpo
    assert "não é recomendação de investimento" in corpo


def test_sending_logs_in_with_the_app_password_without_spaces_and_sends_to_the_user(monkeypatch):
    enviados = []

    class SMTP:
        def __init__(self, host, port, timeout):
            enviados.append(("conectar", host, port))

        def __enter__(self):
            return self

        def __exit__(self, *erro):
            return False

        def starttls(self, context=None):
            enviados.append(("tls",))

        def login(self, usuario, senha):
            enviados.append(("login", usuario, senha))

        def send_message(self, mensagem):
            enviados.append(("enviar", mensagem["From"], mensagem["To"], mensagem["Subject"], mensagem.get_content()))

    credenciais = ae.Credenciais("eu@gmail.com", "abcd efgh ijkl mnop", "eu@gmail.com")
    ae.enviar("Assunto", "Corpo\n", credenciais, smtp=SMTP)
    assert enviados == [("conectar", "smtp.gmail.com", 587), ("tls",), ("login", "eu@gmail.com", "abcdefghijklmnop"),
                        ("enviar", "eu@gmail.com", "eu@gmail.com", "Assunto", "Corpo\n")]


def test_credentials_come_from_the_env_and_are_none_without_the_password(tmp_path):
    arquivo = tmp_path / ".env"
    arquivo.write_text('GMAIL_USER=eu@gmail.com\nGMAIL_PASSWORD_APP="abcd efgh ijkl mnop" # app\n', encoding="utf-8")
    assert ae.credenciais(arquivo) == ae.Credenciais("eu@gmail.com", "abcd efgh ijkl mnop", "eu@gmail.com")
    (tmp_path / "vazio").write_text("GMAIL_USER=eu@gmail.com\n", encoding="utf-8")
    assert ae.credenciais(tmp_path / "vazio") is None


@pytest.fixture(autouse=True)
def _sem_variaveis(monkeypatch):
    for nome in ("GMAIL_USER", "GMAIL_PASSWORD_APP", "ALERTA_EMAIL_PARA"):
        monkeypatch.delenv(nome, raising=False)
