"""
revisao_liquid.py - the code-review gate on Liquid AI's d1.1 (the project's chosen judge, DECISOES 270).

Jev is unavailable here (no provider key), and the Sentinela already judges with Liquid's System One API
(``core.sentinela.juizes``). This script asks d1.1 the same kind of questions a patch gate asks - is the
change correct, is there a defect, does it match the spec, could it reach production, can it read the
future - and turns the probabilities into ``auto`` / ``review`` / ``escalate``. Advice only: it never
applies the patch, runs the tests or opens a pull request.

    .venv/Scripts/python -X utf8 scripts/revisao_liquid.py --request "..." --diff arquivo.diff \
        [--tests testes.txt] [--model d1.1] [--relatorio docs/revisao-*.txt]
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import requests

URL = "https://api.liquid.ai/decisions/v1/systemone"
PRAZO_S = 120
LIMITE_DO_DIFF = 40_000          # characters shipped as evidence; the rest is reported as dropped
MODELO_PADRAO = "d1.1"
CHAVE_NO_ENV = "LIQUID_API_KEY"
RAIZ = Path(__file__).resolve().parent.parent

# Every question is a proposition whose probability d1.1 returns. "quero_alto" is the direction that argues
# for auto; the verdict is the worst clause.
PERGUNTAS: dict[str, tuple[str, bool]] = {
    "correto": ("The patch correctly implements what the request asks, with no defect that would produce a wrong "
                "result, a crash or a silent failure.", True),
    "defeito": ("The patch contains a defect that would produce a wrong result, a crash, or a silent failure in a "
                "real run.", False),
    "espec": ("The patch does what the specification it claims to follow says, including every control that "
              "specification promises.", True),
    "testes": ("The supplied test output, if any, exercises the behaviour the patch changes; without test output "
               "this is unlikely.", True),
    "producao": ("This patch could reach production or move real money without a further explicit decision by the "
                 "owner.", False),
    "futuro": ("This patch can read data from after the date a decision is taken (look-ahead), or reuse a stored "
               "value computed from such data.", False),
}
LIMIAR_ALTO = 0.80
LIMIAR_BAIXO = 0.20


@dataclass(frozen=True)
class Resposta:
    pergunta: str
    probabilidade: float | None
    erro: str | None = None


@dataclass(frozen=True)
class Revisao:
    modelo: str | None
    respostas: tuple[Resposta, ...]
    segundos: float
    tokens: int | None
    custo: float | None
    caracteres: int
    descartados: int

    def acao(self) -> str:
        """auto / review / escalate from the worst clause; a missing answer can never be auto."""
        if any(resposta.probabilidade is None for resposta in self.respostas):
            return "escalate"
        confiancas = []
        for resposta in self.respostas:
            prob = float(resposta.probabilidade)
            confiancas.append(prob if PERGUNTAS[resposta.pergunta][1] else 1.0 - prob)
        pior = min(confiancas)
        if pior >= LIMIAR_ALTO:
            return "auto"
        return "review" if pior >= LIMIAR_BAIXO else "escalate"

    def resumo(self) -> str:
        linhas = [f"{resposta.pergunta:<10} {resposta.probabilidade if resposta.probabilidade is not None else 'erro'}"
                  for resposta in self.respostas]
        return "\n".join(linhas)


def carregar_chave(env: Path | None = None) -> str | None:
    """The key from the environment, else from the project's .env (quotes stripped, as the Sentinela does)."""
    import os

    if os.environ.get(CHAVE_NO_ENV):
        return os.environ[CHAVE_NO_ENV].strip().strip('"').strip("'")
    caminho = env or (RAIZ.parent / ".env")
    if not caminho.exists():
        return None
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if linha.startswith(f"{CHAVE_NO_ENV}="):
            return linha.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def _estado(pedido: str, diff: str, testes: str) -> str:
    partes = [f"REQUEST FROM THE OWNER\n{pedido}", f"PATCH\n{diff}"]
    if testes.strip():
        partes.append(f"TEST OUTPUT\n{testes}")
    return "\n\n".join(partes)


def revisar(pedido: str, diff: str, testes: str = "", modelo: str = MODELO_PADRAO, chave: str | None = None,
            post: Callable = requests.post) -> Revisao:
    chave = chave if chave is not None else carregar_chave()
    if not chave:
        raise RuntimeError(f"{CHAVE_NO_ENV} not found: set it in the environment or in {RAIZ.parent / '.env'}")
    recortado = diff[:LIMITE_DO_DIFF]
    perguntas = {nome: {"type": "noul", "instructions": texto} for nome, (texto, _) in PERGUNTAS.items()}
    inicio = time.perf_counter()
    resposta = post(URL, json={"model": modelo, "state": _estado(pedido, recortado, testes[:10_000]),
                               "questions": perguntas},
                    timeout=PRAZO_S, headers={"Authorization": f"Bearer {chave}"})
    segundos = time.perf_counter() - inicio
    if not resposta.ok:
        raise RuntimeError(f"HTTP {resposta.status_code}: {resposta.text[:200]}")
    dados = resposta.json()
    respostas = tuple(Resposta(nome, float(dados["answers"][nome]["noul"]) if nome in dados.get("answers", {}) else None,
                               None if nome in dados.get("answers", {}) else "missing")
                      for nome in PERGUNTAS)
    uso = dados.get("usage") or {}
    return Revisao(dados.get("model", modelo), respostas, segundos, uso.get("input_tokens"), uso.get("cost"),
                   min(len(diff), LIMITE_DO_DIFF), max(0, len(diff) - LIMITE_DO_DIFF))


def dividir_por_arquivo(diff: str) -> dict[str, str]:
    """``git diff`` sections keyed by path, so a large change is reviewed whole instead of truncated."""
    secoes: dict[str, str] = {}
    atual, linhas = None, []
    for linha in diff.splitlines(keepends=True):
        if linha.startswith("diff --git "):
            if atual is not None:
                secoes[atual] = "".join(linhas)
            partes = linha.split()
            atual = partes[-1].removeprefix("b/") if len(partes) >= 4 else "desconhecido"
            linhas = [linha]
        elif atual is not None:
            linhas.append(linha)
    if atual is not None:
        secoes[atual] = "".join(linhas)
    return secoes


def revisar_arquivos(pedido: str, diff: str, testes: str = "", modelo: str = MODELO_PADRAO,
                     chave: str | None = None) -> dict[str, Revisao]:
    """One review per code file of the patch; documents are skipped (the rubric asks about an implementation)."""
    secoes = dividir_por_arquivo(diff) or {"diff": diff}
    codigo = {arquivo: secao for arquivo, secao in secoes.items() if arquivo.endswith(".py")}
    if not codigo:
        raise ValueError("the diff has no .py file to review")
    return {arquivo: revisar(pedido, secao, testes, modelo, chave) for arquivo, secao in codigo.items()}


def pior_acao(revisoes: Mapping[str, Revisao]) -> str:
    ordem = {"auto": 0, "review": 1, "escalate": 2}
    return max((revisao.acao() for revisao in revisoes.values()), key=ordem.__getitem__, default="escalate")


def _argumentos(args: Sequence[str]) -> tuple[str, str, str, str, Path | None]:
    def _valor(nome: str, padrao: str = "") -> str:
        if nome not in args:
            return padrao
        posicao = args.index(nome)
        if posicao + 1 == len(args):
            raise ValueError(f"{nome} needs a value")
        return args[posicao + 1]
    pedido = _valor("--request")
    if not pedido:
        raise ValueError("--request is required")
    diff = _valor("--diff")
    if not diff:
        raise ValueError("--diff is required")
    relatorio = _valor("--relatorio")
    return pedido, diff, _valor("--tests"), _valor("--model", MODELO_PADRAO), Path(relatorio) if relatorio else None


def main(argv: list[str]) -> int:
    pedido, caminho, caminho_testes, modelo, relatorio = _argumentos(argv)
    diff = Path(caminho).read_text(encoding="utf-8-sig")          # a BOM would hide the first "diff --git"
    testes = Path(caminho_testes).read_text(encoding="utf-8-sig") if caminho_testes else ""
    revisoes = revisar_arquivos(pedido, diff, testes, modelo)
    acao = pior_acao(revisoes)
    blocos = [f"Liquid gate ({modelo}) - action: {acao.upper()} over {len(revisoes)} file(s)\n"]
    total_tokens, total_custo, total_segundos = 0, 0.0, 0.0
    for arquivo, revisao in revisoes.items():
        total_tokens += revisao.tokens or 0
        total_custo += revisao.custo or 0.0
        total_segundos += revisao.segundos
        blocos.append(f"--- {arquivo}: {revisao.acao().upper()} "
                      f"({revisao.caracteres} chars, {revisao.descartados} dropped)\n{revisao.resumo()}")
    blocos.append(f"\n{len(revisoes)} calls, {total_tokens} input tokens, cost {total_custo:.6f}, "
                  f"{total_segundos:.1f}s. Thresholds: auto needs every clause at {LIMIAR_ALTO:.0%} of its wanted "
                  f"direction, per file; below {LIMIAR_BAIXO:.0%} escalates, and the worst file decides the patch.")
    texto = "\n".join(blocos)
    if relatorio is not None:
        relatorio.parent.mkdir(parents=True, exist_ok=True)
        relatorio.write_text(texto, encoding="utf-8")
    print(texto)
    return 0 if acao == "auto" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
