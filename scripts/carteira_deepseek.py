"""
The simulator's DeepSeek portfolio (DECISOES 240, the user's idea of 2026-11-03; since 2026-11-04, DECISOES 251): the
DeepSeek adviser judges the names of both rule portfolios, the Composite's and the Conservadora's, each once, for the
extremely aggressive investor, with the evidence rules of ``core.assessor`` (every claim
quotes evidence; news and posts count only when CVM, BCB, B3 or ANBIMA confirm them). The favorable names
("favoravel") share the portfolio equally (the app's 20% cap per stock still applies in the simulator); the rest get
zero. The record is dated the day after the opinion, so it trades at the next session's close, never at a close the
model may have seen. An opinion is due after a new Composite or Conservadora record and every 28 days. A control
portfolio (``deepseek/controle.csv``) holds the same list in equal weights from the same date, so the simulator shows
what the model's filter adds or costs.

Forward only: a language model cannot be backtested (its training data holds the past's prices and news).

Usage: .venv/Scripts/python -X utf8 scripts/carteira_deepseek.py            (when due)
       .venv/Scripts/python -X utf8 scripts/carteira_deepseek.py --forcar   (now)
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_COMPOSTO = RAIZ / "paper-v3" / "composto.csv"
ARQUIVO_CONSERVADORA = RAIZ / "paper-v3" / "conservadora.csv"
ARQUIVO = RAIZ / "deepseek" / "carteira.csv"      # its own folder: paper-v3/ stays as its guards expect
ARQUIVO_CONTROLE = RAIZ / "deepseek" / "controle.csv"
PASTA_PARECERES = RAIZ / "deepseek" / "pareceres"
DIAS_ENTRE_PARECERES = 28
FAVORAVEL = "favoravel"
COLUNAS = ["data", "ticker", "peso"]


def _ultimos_nomes(registro: pd.DataFrame) -> list[str]:
    ultimo = registro[registro["data"] == registro["data"].max()]
    return list(ultimo.loc[ultimo["peso"] > 0, "ticker"])


def universo_unido(composto: pd.DataFrame, conservadora: pd.DataFrame) -> list[str]:
    """The held names of both portfolios' last records, each once: the Composite's first, in record order."""
    return list(dict.fromkeys([*_ultimos_nomes(composto), *_ultimos_nomes(conservadora)]))


def pesos_do_controle(universo: list[str]) -> pd.Series:
    """The control: the whole list in equal weights, no model."""
    return pd.Series(1.0 / len(universo), index=universo, dtype=float)


def pesos_dos_verdes(parecer: dict, universo: list[str]) -> pd.Series:
    """Equal weights over the names of ``universo`` judged favorable; zero for the others (sold)."""
    posturas = {str(acao.get("ticker")): acao.get("postura") for acao in parecer.get("acoes") or []}
    verdes = [ticker for ticker in universo if posturas.get(ticker) == FAVORAVEL]
    return pd.Series({ticker: (1.0 / len(verdes) if ticker in verdes else 0.0) for ticker in universo}, dtype=float)


def registro_do_parecer(pesos: pd.Series, quando: date) -> pd.DataFrame:
    """The record's rows, dated the calendar day after the opinion."""
    return pd.DataFrame({"data": pd.Timestamp(quando + timedelta(days=1)), "ticker": list(pesos.index),
                         "peso": pesos.to_numpy(dtype=float)}, columns=COLUNAS)


def parecer_devido(composto: pd.DataFrame, feitos: pd.DataFrame, hoje: date,
                   base_julgada: pd.Timestamp | None = None) -> bool:
    """True with no opinion yet, 28 days after the last one, or when the Composite's last record is not the one the
    last opinion judged (``base_julgada``; without it, a record dated after the opinion's day). A record paper-v3
    writes after the opinion but dates before it is caught by ``base_julgada`` (code review, 2026-11-03)."""
    if feitos.empty:
        return True
    ultima = pd.Timestamp(feitos["data"].max()) - pd.Timedelta(days=1)      # the day of the last opinion
    if pd.Timestamp(hoje) >= ultima + pd.Timedelta(days=DIAS_ENTRE_PARECERES):
        return True
    base = pd.Timestamp(composto["data"].max())
    return base != base_julgada if base_julgada is not None else base > ultima


def ultima_base_julgada(pasta: Path = PASTA_PARECERES, campo: str = "base_do_composto") -> pd.Timestamp | None:
    """The record (``campo``: the Composite's or "base_da_conservadora") the newest saved opinion judged; None before
    the first one, or when that opinion did not judge it (the ones before the union)."""
    arquivos = sorted(pasta.glob("*.json")) if pasta.exists() else []
    if not arquivos:
        return None
    base = json.loads(arquivos[-1].read_text(encoding="utf-8")).get(campo)
    return None if base is None else pd.Timestamp(base)


def acrescentar(arquivo: Path, novo: pd.DataFrame) -> None:
    """Appends ``novo``, first dropping rows dated on or after it: a record not traded yet (an opinion redone the same
    day, or the switch to the union on 04/11) is replaced, never stacked on the same date."""
    antigo = ler_registro(arquivo)
    tabela = pd.concat([antigo[antigo["data"] < novo["data"].min()], novo], ignore_index=True)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(arquivo, index=False, lineterminator="\n", date_format="%Y-%m-%d")


def ler_registro(arquivo: Path = ARQUIVO) -> pd.DataFrame:
    if not arquivo.exists():
        return pd.DataFrame(columns=COLUNAS).astype({"data": "datetime64[ns]"})
    return pd.read_csv(arquivo, parse_dates=["data"])


def _salvar(resultado, universo: list[str], base: pd.Timestamp, base_conservadora: pd.Timestamp, agora: datetime,
            modelo: str) -> Path:
    from core import assessor
    parecer = resultado.validacao.parecer
    citadas = {afirmacao["evidencia"] for acao in parecer["acoes"]
               for campo in ("tese", "contra_tese_macro", "riscos", "nao_confirmadas") for afirmacao in acao[campo]}
    citadas |= {acao_conf for acao in parecer["acoes"] for campo in ("tese", "riscos")
                for afirmacao in acao[campo] if (acao_conf := (afirmacao.get("confirmacao") or {}).get("evidencia"))}
    registro = {"quando": agora.isoformat(timespec="seconds"), "base_do_composto": f"{base:%Y-%m-%d}",
                "base_da_conservadora": f"{base_conservadora:%Y-%m-%d}",
                "modelo": modelo, "esforco": assessor.ESFORCO, "universo": universo, "parecer": parecer,
                "cores": {acao["ticker"]: assessor.cor_da_postura(acao["postura"]) for acao in parecer["acoes"]},
                "descartes": resultado.validacao.descartes, "chamadas": resultado.chamadas,
                "uso": {"requisicoes": resultado.uso.requisicoes, "entrada": resultado.uso.entrada,
                        "saida": resultado.uso.saida, "custo_usd": round(resultado.uso.custo_usd, 6)},
                "evidencias_citadas": {i: vars(resultado.evidencias[i]) for i in sorted(citadas)
                                       if i in resultado.evidencias}}
    PASTA_PARECERES.mkdir(parents=True, exist_ok=True)
    arquivo = PASTA_PARECERES / f"{agora:%Y-%m-%d}.json"
    arquivo.write_text(json.dumps(registro, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    return arquivo


def executar(forcar: bool = False) -> int:
    from core import assessor
    from core.segredos import segredo
    from ui import parecer as entradas
    composto, feitos, hoje = pd.read_csv(ARQUIVO_COMPOSTO, parse_dates=["data"]), ler_registro(), date.today()
    conservadora = pd.read_csv(ARQUIVO_CONSERVADORA, parse_dates=["data"])
    base_conservadora = conservadora["data"].max()
    devido = (parecer_devido(composto, feitos, hoje, ultima_base_julgada())
              or ultima_base_julgada(campo="base_da_conservadora") != base_conservadora)
    if not forcar and not devido:
        print(f"DeepSeek: em dia (último parecer em {feitos['data'].max() - pd.Timedelta(days=1):%d/%m/%Y})")
        return 0
    chave = segredo("DEEPSEEK_API_KEY")
    if chave is None:
        print("DeepSeek: falta DEEPSEEK_API_KEY no ambiente ou no .env do ai-hedge-fund")
        return 1
    base = composto["data"].max()
    universo = universo_unido(composto, conservadora)
    alvo = pd.Series(1.0 / len(universo), index=[f"{ticker}.SA" for ticker in universo])
    pacote, fontes = entradas.pacote_e_fontes(alvo, base.date(), chave, hoje)
    for nome, variavel in (("chave_firecrawl", "FIRECRAWL_API_KEY"), ("chave_scrapecreators", "SCRAPECREATORS_API_KEY")):
        setattr(fontes, nome, segredo(variavel) or getattr(fontes, nome))       # the .env first, like the DeepSeek key
    agora = datetime.now(timezone.utc)
    resultado = assessor.pedir_parecer(pacote, universo, assessor.Ferramentas(fontes, {}), chave, agora,
                                       carteira=assessor.CARTEIRAS_UNIDAS)
    if resultado.validacao is None:
        print(f"DeepSeek: parecer indisponível ({resultado.erro}); nada registrado")
        return 1
    arquivo = _salvar(resultado, universo, base, agora, assessor.MODELO)
    pesos = pesos_dos_verdes(resultado.validacao.parecer, universo)
    novo = registro_do_parecer(pesos, agora.astimezone().date())
    ARQUIVO.parent.mkdir(parents=True, exist_ok=True)
    novo.to_csv(ARQUIVO, mode="a", header=not ARQUIVO.exists(), index=False, lineterminator="\n",
                date_format="%Y-%m-%d")
    verdes = [ticker for ticker, peso in pesos.items() if peso > 0]
    print(f"DeepSeek: {len(verdes)} de {len(universo)} favoráveis ({', '.join(verdes) or 'nenhuma: tudo no CDI'}); "
          f"custo ~US$ {resultado.uso.custo_usd:.4f}; parecer em {arquivo.relative_to(RAIZ).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(executar("--forcar" in sys.argv[1:]))
