"""
Fetches B3's dated news on the Ibovespa previews, 2020 onwards (spec ``2026-11-08-ibov-recomposicao-design.md``,
data item 2), through the Firecrawl API: for each rebalance and each of the 1st and 3rd previews, a search on B3's
news and Bora Investir, then a structured extraction of each hit (publication date, preview number, the portfolio's
start, entrants, leavers). Raw answers are kept; nothing about prices is read.

Usage: python -X utf8 scripts/ibov_previas.py <path to .env with FIRECRAWL_API_KEY>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "docs" / "dados" / "ibov-recomposicao" / "previas"
API = "https://api.firecrawl.dev/v2"
MESES = {1: "janeiro", 5: "maio", 9: "setembro"}
ORDINAIS = {1: ("1ª", "primeira"), 3: ("3ª", "terceira")}
SITES = ("b3.com.br", "borainvestir.b3.com.br")
ESQUEMA = {
    "type": "object",
    "properties": {
        "e_previa_do_ibovespa": {"type": "boolean"},
        "numero_da_previa": {"type": "integer", "description": "1, 2 or 3; 0 if the final portfolio, not a preview"},
        "data_de_divulgacao": {"type": "string", "description": "YYYY-MM-DD the news or preview was published"},
        "data_base_do_fechamento": {"type": "string", "description": "YYYY-MM-DD of the session close it is based on"},
        "mes_de_vigencia": {"type": "string", "description": "YYYY-MM the new Ibovespa portfolio takes effect"},
        "entradas": {"type": "array", "items": {"type": "string"}, "description": "tickers entering the Ibovespa"},
        "saidas": {"type": "array", "items": {"type": "string"}, "description": "tickers leaving the Ibovespa"},
        "total_de_ativos": {"type": "integer"},
    },
    "required": ["e_previa_do_ibovespa", "numero_da_previa", "mes_de_vigencia", "entradas", "saidas"],
}
PROMPT = ("Extract only what this page says about the Ibovespa B3 (not other indices such as IBrX, SMLL or Ibovespa "
          "B3 BR+): which preview it is, its dates, the month the new portfolio takes effect, and the tickers it says "
          "enter and leave the Ibovespa. Leave lists empty if the page says there are none.")


def _post(caminho: str, corpo: dict, chave: str) -> dict:
    for tentativa in range(3):
        resposta = requests.post(f"{API}/{caminho}", json=corpo, timeout=120,
                                 headers={"Authorization": f"Bearer {chave}"})
        if resposta.ok:
            return resposta.json()
        time.sleep(5 * (tentativa + 1))
    return {"erro": resposta.status_code, "texto": resposta.text[:300]}


def rebalanceamentos() -> list[tuple[int, int]]:
    return [(ano, mes) for ano in range(2020, 2027) for mes in MESES if (ano, mes) <= (2026, 9)]


MES_DA_DIVULGACAO = {1: (12, "dezembro"), 5: (4, "abril"), 9: (8, "agosto")}


def consultas(ano: int, mes: int, numero: int) -> list[tuple[str, str]]:
    """(file suffix, query): the first pass by effective month, a second by publication month in B3's old titles."""
    curto, longo = ORDINAIS[numero]
    mes_pub, nome_pub = MES_DA_DIVULGACAO[mes]
    ano_pub = ano - 1 if mes == 1 else ano
    return [("", f"{curto} prévia carteira Ibovespa B3 que entra em vigor em {MESES[mes]} de {ano} entrada saída"),
            ("-b", f"B3 divulga {longo} prévia do Ibovespa {nome_pub} {ano_pub} registra a entrada")]


def main() -> int:
    env = dict(linha.strip().split("=", 1) for linha in open(sys.argv[1], encoding="utf-8")
               if "=" in linha and not linha.startswith("#"))
    chave = env["FIRECRAWL_API_KEY"].strip("\"'")
    SAIDA.mkdir(parents=True, exist_ok=True)
    vistos: dict[str, dict] = {}
    for ano, mes in rebalanceamentos():
        for numero, sufixo, consulta in [(n, sufixo, c) for n in ORDINAIS for sufixo, c in consultas(ano, mes, n)]:
            arquivo = SAIDA / f"{ano}-{mes:02d}-p{numero}{sufixo}.json"
            if arquivo.exists():
                continue
            busca = _post("search", {"query": consulta, "limit": 6, "includeDomains": list(SITES)}, chave)
            urls = [item["url"] for item in busca.get("data", {}).get("web", [])
                    if "previa" in item["url"].lower() or "carteira" in item["url"].lower()
                    or "ibovespa" in item["url"].lower()]
            extraidos = []
            for url in urls[:4]:
                if url not in vistos:
                    pagina = _post("scrape", {"url": url, "onlyMainContent": True,
                                              "formats": [{"type": "json", "schema": ESQUEMA, "prompt": PROMPT}]},
                                   chave)
                    vistos[url] = pagina.get("data", {}).get("json") or {"erro": pagina.get("erro")}
                extraidos.append({"url": url, "json": vistos[url]})
            arquivo.write_text(json.dumps({"consulta": consulta, "busca": busca, "paginas": extraidos},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{ano}-{mes:02d} p{numero}{sufixo}: {len(urls)} urls", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
