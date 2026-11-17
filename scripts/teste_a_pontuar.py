"""
teste_a_pontuar.py - Score the backfilled material filings of Test A (amendment of 2026-11-17) with the judge the
60-filing check chose (d1.1, Liquid direct API), with the lake's ``filings-v1`` questions.

    python -X utf8 scripts/teste_a_pontuar.py --contar     # sizes only, no call
    python -X utf8 scripts/teste_a_pontuar.py              # score (resumes from respostas.jsonl)

Writes ``docs/dados/teste-a-2026-11-17/documentos.parquet`` (one row per filing and candidate issuer) and appends one
JSON line per distinct state to ``respostas.jsonl``. A returned model id other than ``MODELO`` stops the run.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core.teste_a import chave_do_estado, documentos_do_universo, estado_do_documento, perguntas_do_lago  # noqa: E402

LAGO = Path("D:/b3-datalake/sentiment/ipe/ipe.parquet")
EXPORT = RAIZ.parent / "Portfolio-Optimization" / "docs" / "dados" / "2026-10-22"
JEVSCORE = RAIZ.parents[1] / "b3-datalake" / "src" / "b3lake" / "jevscore.py"
ENV = RAIZ.parent / ".env"
SAIDA = RAIZ / "docs" / "dados" / "teste-a-2026-11-17"
URL, MODELO = "https://api.liquid.ai/decisions/v1/systemone", "d1.1"
INICIO, FIM = pd.Timestamp("2009-12-01"), pd.Timestamp("2026-10-16")
TRABALHADORES, TENTATIVAS = 8, 4


class ModeloTrocado(RuntimeError):
    """The API answered with another model id: the amendment pins one model for every filing."""


def _chave_api(env: Path) -> str:
    linhas = (linha.strip().split("=", 1) for linha in env.read_text(encoding="utf-8").splitlines()
              if "=" in linha and not linha.lstrip().startswith("#"))
    return dict(linhas)["LIQUID_API_KEY"].strip("\"'")


def _documentos() -> pd.DataFrame:
    ipe = pd.read_parquet(LAGO, columns=["codigo_cvm", "categoria", "assunto", "tipo", "especie", "data_entrega"])
    setores = pd.read_parquet(EXPORT / "sectors.parquet", columns=["issuer", "code_cvm"])
    universo = pd.read_parquet(EXPORT / "universe_monthly.parquet")
    return documentos_do_universo(ipe, setores, universo, INICIO, FIM)


def _perguntar(sessao: requests.Session, chave_api: str, estado: dict, perguntas: dict) -> dict:
    for tentativa in range(TENTATIVAS):
        try:
            resposta = sessao.post(URL, timeout=90, headers={"Authorization": f"Bearer {chave_api}"},
                                   json={"model": MODELO, "state": estado, "questions": perguntas})
            if resposta.ok:
                corpo = resposta.json()
                if corpo.get("model", MODELO) != MODELO:
                    raise ModeloTrocado(f"asked {MODELO}, answered {corpo.get('model')}")
                return {"answers": corpo["answers"], "modelo": corpo.get("model", MODELO),
                        "tokens": corpo.get("usage", {}).get("input_tokens"), "custo": corpo.get("usage", {}).get("cost")}
            erro = f"HTTP {resposta.status_code}: {resposta.text[:120]}"
        except requests.RequestException as excecao:
            erro = str(excecao)[:120]
        time.sleep(2 ** tentativa)
    return {"erro": erro}


def main(argv: list[str] | None = None) -> int:
    argumentos = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    argumentos.add_argument("--contar", action="store_true", help="print the sizes and stop, no call")
    contar = argumentos.parse_args(argv).contar
    SAIDA.mkdir(parents=True, exist_ok=True)
    docs = _documentos()
    docs.to_parquet(SAIDA / "documentos.parquet", index=False)
    estados = {linha["chave"]: estado_do_documento(linha) for linha in docs.drop_duplicates("chave").to_dict("records")}
    arquivo = SAIDA / "respostas.jsonl"
    feitos = set()
    if arquivo.exists():
        for linha in arquivo.read_text(encoding="utf-8").splitlines():
            registro = json.loads(linha)
            if "answers" in registro:
                feitos.add(registro["chave"])
    pendentes = [chave for chave in estados if chave not in feitos]
    print(f"{len(docs)} filings ({docs['emissor'].nunique()} issuers), {len(estados)} distinct states, "
          f"{len(feitos)} answered, {len(pendentes)} to ask")
    if contar or not pendentes:
        return 0
    perguntas, versao = perguntas_do_lago(JEVSCORE)
    chave_api = _chave_api(ENV)
    trava, sessao = threading.Lock(), requests.Session()
    erros = 0
    with arquivo.open("a", encoding="utf-8") as saida, ThreadPoolExecutor(TRABALHADORES) as executor:
        futuros = {executor.submit(_perguntar, sessao, chave_api, estados[chave], perguntas): chave for chave in pendentes}
        for numero, futuro in enumerate(as_completed(futuros), start=1):
            chave = futuros[futuro]
            registro = {"chave": chave, "versao": versao, **futuro.result()}
            erros += "erro" in registro
            with trava:
                saida.write(json.dumps(registro, ensure_ascii=False) + "\n")
                saida.flush()
            if numero % 1000 == 0:
                print(f"  {numero}/{len(pendentes)} asked, {erros} errors", flush=True)
    print(f"done: {len(pendentes)} asked, {erros} errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
