"""Headlines on the deployed app (``core.manchetes_online``, option (b) of DECISOES 126): the upkeep publishes the
newest scored headlines (title, link, topic, tone; no summary text) to a private GitHub repository, and the app reads
them with a read-only token from Streamlit secrets."""

from datetime import datetime

import pandas as pd
import pytest


def _manchetes() -> pd.DataFrame:
    return pd.DataFrame({"quando": pd.to_datetime(["2026-10-26 09:10", "2026-10-26 08:00"]),
                         "fonte": ["valor", "reddit"], "titulo": ["Copom mantém Selic", "Ibovespa sobe"],
                         "tema": ["Juros e inflação", "Empresas"], "tom": [-0.25, 0.5],
                         "link": ["https://valor.com.br/a", "javascript:alert(1)"]})


def test_the_published_headlines_come_back_with_their_types_the_publish_time_and_only_safe_links():
    from core.manchetes_online import de_json, para_json
    texto = para_json(_manchetes(), datetime(2026, 10, 26, 10, 5))
    lidas, gerado_em = de_json(texto)
    assert gerado_em == datetime(2026, 10, 26, 10, 5)
    assert list(lidas.columns) == ["quando", "fonte", "titulo", "tema", "tom", "link"]
    assert lidas["quando"].dtype.kind == "M" and lidas["tom"].dtype == float
    assert lidas["quando"].tolist() == _manchetes()["quando"].tolist()
    assert lidas["tom"].tolist() == pytest.approx([-0.25, 0.5])
    assert lidas["link"].tolist()[0] == "https://valor.com.br/a" and pd.isna(lidas["link"].tolist()[1])


class _Resposta:
    def __init__(self, status: int, texto: str = "") -> None:
        self.status_code, self.content = status, texto.encode("utf-8")     # GitHub's raw body: bytes, no charset


def test_the_download_sends_the_token_only_in_the_header_and_gives_none_on_any_failure():
    import requests
    from core.manchetes_online import baixar
    pedidos = []

    def get(url, headers, timeout):
        pedidos.append((url, headers, timeout))
        return _Resposta(200, '{"titulo": "Copom mantém a Selic"}')
    assert baixar("segredo", get=get) == '{"titulo": "Copom mantém a Selic"}'
    url, cabecalhos, prazo = pedidos[0]
    assert url == "https://api.github.com/repos/GabrielScript/alepha-manchetes/contents/manchetes.json"
    assert "segredo" not in url and cabecalhos["Authorization"] == "Bearer segredo"
    assert cabecalhos["Accept"] == "application/vnd.github.raw+json" and prazo == 10
    assert baixar("segredo", get=lambda url, headers, timeout: _Resposta(401, "Bad credentials")) is None

    def sem_rede(url, headers, timeout):
        raise requests.ConnectionError("sem rede")
    assert baixar("segredo", get=sem_rede) is None
