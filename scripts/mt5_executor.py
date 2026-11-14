"""
The MT5 executor (docs/superpowers/specs/2026-11-12-mt5-demo-design.md, plan step 3): the Composite's last record
(``paper-v3/composto.csv``) traded by itself on XP's MetaTrader 5 demo account, in B3's closing call.

The Windows task "aleph mt5" runs it every weekday at 17:50 through the routines' launcher:
    <Python312>/pythonw.exe -X utf8 scripts/com_python_base.py scripts/rodar_paper.py mt5_executor
By hand:
    python -X utf8 scripts/mt5_executor.py --simular     (the orders it would send now; sends and writes nothing)
    python -X utf8 scripts/mt5_executor.py --leitura     (the book: shares, cash, value, fills)

Files: ``paper-mt5/`` (alvos.csv: the target of each record; execucoes.csv: each run that sent orders; ordens.csv:
each order; fills.csv: the book's deals, rewritten from the terminal on each run); a file ``paper-mt5/PARAR`` stops
everything. ``%LOCALAPPDATA%/paper-mt5/runs.csv``: one line per run, read by the routines' watcher.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time as relogio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core.mt5_book import (
    check_account,
    check_cash,
    check_names,
    check_quote,
    limit_price,
    orders_for,
    target_shares,
)
from core.mt5_corretora import Corretora, Cotacao, Terminal

INICIAL = 100_000.0                                      # the book's money (the user, 2026-11-12)
CHAMADA, ULTIMO_ENVIO = time(17, 55, 15), time(17, 58, 30)   # B3's closing call 17:55-18:00 since 03/11/2026
TENTATIVAS = 2                                           # the first send of a record and one resend next session
PACOTE_MT5 = RAIZ / ".venv-mt5"                          # MetaTrader5 installed with --no-deps (Windows only)
CABECALHO_RUNS = ["quando", "registro", "desfecho", "ordens", "nota"]


@dataclass(frozen=True)
class Pastas:
    registro: Path                  # paper-v3/composto.csv
    livro: Path                     # paper-mt5/
    runs: Path                      # %LOCALAPPDATA%/paper-mt5/runs.csv


PASTAS = Pastas(RAIZ / "paper-v3" / "composto.csv", RAIZ / "paper-mt5",
                Path(os.environ.get("LOCALAPPDATA", Path.home())) / "paper-mt5" / "runs.csv")


@dataclass(frozen=True)
class Resultado:
    desfecho: str                   # enviado | nada | fora da chamada | parado | recusado | simulado
    registro: str = ""
    ordens: int = 0
    nota: str = ""


def _ler(arquivo: Path) -> pd.DataFrame:
    return pd.read_csv(arquivo) if arquivo.exists() else pd.DataFrame()


def _anexar(arquivo: Path, linhas: list[dict]) -> None:
    if not linhas:
        return
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    novo = not arquivo.exists()
    with arquivo.open("a", encoding="utf-8", newline="") as saida:
        escritor = csv.DictWriter(saida, fieldnames=list(linhas[0]), lineterminator="\n")
        if novo:
            escritor.writeheader()
        escritor.writerows(linhas)


def ultimo_registro(arquivo: Path) -> tuple[str, pd.Series]:
    """The last record's date and weights."""
    tabela = pd.read_csv(arquivo)
    data = str(tabela["data"].max())
    return data, tabela.loc[tabela["data"] == data].set_index("ticker")["peso"].astype(float)


def caixa(fills: pd.DataFrame) -> float:
    """The book's cash: the start minus its buys, plus its sells, minus the commissions charged."""
    if fills.empty:
        return INICIAL
    sinal = fills["side"].map({"buy": -1.0, "sell": 1.0})
    return float(INICIAL + (sinal * fills["volume"] * fills["price"]).sum() - fills["commission"].sum())


def preco(cotacao: Cotacao | None) -> float:
    if cotacao is None:
        return 0.0
    if cotacao.last > 0:
        return cotacao.last
    return (cotacao.bid + cotacao.ask) / 2 if cotacao.bid > 0 and cotacao.ask > 0 else 0.0


def sincronizar_fills(c: Corretora, pastas: Pastas, execucoes: pd.DataFrame, agora: datetime) -> pd.DataFrame:
    """The book's deals since its first run, rewritten from the terminal (the terminal is the ledger's source)."""
    inicio = pd.Timestamp(execucoes["quando"].min()).normalize() if not execucoes.empty else pd.Timestamp(agora) - pd.Timedelta(days=1)
    fills = pd.DataFrame([vars(f) for f in c.execucoes(inicio, pd.Timestamp(agora) + pd.Timedelta(hours=1))],
                         columns=["name", "symbol", "side", "volume", "price", "commission", "time", "order"])
    if not fills.empty:
        pastas.livro.mkdir(parents=True, exist_ok=True)
        fills.to_csv(pastas.livro / "fills.csv", index=False, lineterminator="\n")
    return fills


def devido(registro: str, execucoes: pd.DataFrame, alvos: pd.DataFrame, detidas: pd.Series,
           hoje: str) -> tuple[str, pd.Series | None]:
    """("novo", None) for a record never sent; ("repetir", target) for a record sent once on an earlier session whose
    target is not held yet; ("nada", None) otherwise."""
    enviadas = execucoes.loc[execucoes["registro"].astype(str) == registro] if not execucoes.empty else execucoes
    if enviadas.empty:
        return "novo", None
    alvo = alvos.loc[alvos["registro"].astype(str) == registro].set_index("nome")["acoes"].astype(int)
    nomes = sorted(set(alvo.index) | set(detidas.index))
    falta = (alvo.reindex(nomes, fill_value=0) != detidas.reindex(nomes, fill_value=0)).any()
    ultima = str(pd.Timestamp(enviadas["quando"].max()).date())
    if falta and len(enviadas) < TENTATIVAS and ultima < hoje:
        return "repetir", alvo
    return "nada", None


def rodar(terminal: Terminal, pastas: Pastas, agora: Callable[[], datetime], esperar: Callable[[datetime], None],
          simular: bool = False) -> Resultado:
    if (pastas.livro / "PARAR").exists():
        return Resultado("parado", nota="paper-mt5/PARAR exists")
    registro, pesos = ultimo_registro(pastas.registro)
    with Corretora(terminal).conectar() as c:
        conta = c.conta()
        motivo = check_account(conta.is_demo, conta.server)
        if motivo:
            return Resultado("recusado", registro, nota=motivo)
        momento = agora()
        execucoes, alvos = _ler(pastas.livro / "execucoes.csv"), _ler(pastas.livro / "alvos.csv")
        fills = sincronizar_fills(c, pastas, execucoes, momento)
        detidas = c.acoes_da_carteira()
        tipo, alvo = devido(registro, execucoes, alvos, detidas, str(momento.date()))
        if simular:
            tipo, alvo = "novo", None
        if tipo == "nada":
            return Resultado("nada", registro, nota="the book already holds the last record")
        if not simular:
            if momento.time() > ULTIMO_ENVIO:
                return Resultado("fora da chamada", registro, nota=f"{momento:%H:%M:%S} is after {ULTIMO_ENVIO}")
            if momento.time() < CHAMADA:
                esperar(datetime.combine(momento.date(), CHAMADA))
                momento = agora()

        nomes = sorted(set(pesos.index) | set(detidas.index))
        ultimos = pd.Series({nome: preco(c.cotacao(nome)) for nome in nomes})
        sem = sorted(ultimos.index[ultimos <= 0])
        if sem:
            return Resultado("recusado", registro, nota=f"no quote for {sem}")
        dinheiro = caixa(fills)
        if alvo is None:
            valor = dinheiro + float((detidas.reindex(nomes, fill_value=0) * ultimos).sum())
            alvo = target_shares(pesos, ultimos.reindex(pesos.index), valor)
        ordens = orders_for(alvo, detidas)
        motivo = check_names(ordens, nomes) or check_cash(ordens, ultimos, dinheiro)
        if motivo:
            return Resultado("recusado", registro, nota=motivo)
        if simular:
            for o in ordens:
                print(f"{o.side:4} {o.volume:>6} {o.symbol:<8} ~R$ {o.volume * ultimos[o.name]:>10,.2f}")
            return Resultado("simulado", registro, len(ordens), f"cash {dinheiro:,.2f}, held {detidas.to_dict()}")

        quando = momento.isoformat(timespec="seconds")
        if tipo == "novo":
            _anexar(pastas.livro / "alvos.csv", [{"registro": registro, "nome": n, "acoes": int(q)}
                                                 for n, q in alvo.items() if q > 0])
        linhas, enviadas = [], 0
        for o in ordens:
            cotacao = c.cotacao(o.symbol)
            motivo = None if cotacao is None else check_quote(o.symbol, cotacao.bid, cotacao.ask, cotacao.time,
                                                              pd.Timestamp(agora()))
            ultimo = preco(cotacao)
            if cotacao is None or motivo or ultimo <= 0:
                linhas.append({"quando": quando, "registro": registro, "nome": o.name, "symbol": o.symbol,
                               "side": o.side, "volume": o.volume, "limite": "", "ok": False, "retcode": "",
                               "order": "", "nota": motivo or "no quote"})
                continue
            limite = limit_price(o.side, ultimo)
            envio = c.enviar(o, limite, f"aleph {registro}")
            enviadas += envio.ok
            linhas.append({"quando": quando, "registro": registro, "nome": o.name, "symbol": o.symbol, "side": o.side,
                           "volume": o.volume, "limite": limite, "ok": envio.ok, "retcode": envio.retcode,
                           "order": envio.order, "nota": envio.comment})
        _anexar(pastas.livro / "ordens.csv", linhas)
        _anexar(pastas.livro / "execucoes.csv", [{"quando": quando, "registro": registro, "tipo": tipo,
                                                  "ordens": len(ordens), "enviadas": enviadas,
                                                  "caixa": round(dinheiro, 2)}])
        return Resultado("enviado", registro, enviadas, f"{tipo}: {enviadas} of {len(ordens)} orders sent")


def leitura(terminal: Terminal, pastas: Pastas) -> None:
    with Corretora(terminal).conectar() as c:
        detidas = c.acoes_da_carteira()
        fills = sincronizar_fills(c, pastas, _ler(pastas.livro / "execucoes.csv"), datetime.now())
        dinheiro = caixa(fills)
        valores = pd.Series({n: q * preco(c.cotacao(n)) for n, q in detidas.items()}, dtype=float)
    total = dinheiro + float(valores.sum())
    print(f"Book: R$ {total:,.2f} ({total / INICIAL - 1:+.2%} on R$ {INICIAL:,.0f}); cash R$ {dinheiro:,.2f}; "
          f"{len(detidas)} names; {len(fills)} fills")
    if not detidas.empty:
        print(pd.DataFrame({"shares": detidas, "value": valores.round(2)}).to_string())


def _esperar(ate: datetime) -> None:
    relogio.sleep(max(0.0, (ate - datetime.now()).total_seconds()))


def _anotar(resultado: Resultado, inicio: datetime) -> None:
    _anexar(PASTAS.runs, [{"quando": inicio.isoformat(timespec="seconds"), "registro": resultado.registro,
                           "desfecho": resultado.desfecho, "ordens": resultado.ordens, "nota": resultado.nota[:300]}])


def ler_opcoes(argv: list[str] | None) -> argparse.Namespace:
    """The options from ``argv`` only, never from ``sys.argv``: the routines' launcher calls ``main()`` with its own
    arguments still there (``rodar_paper.py mt5_executor``), which argparse would reject with exit 2."""
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    argumentos.add_argument("--simular", action="store_true")
    argumentos.add_argument("--leitura", action="store_true")
    return argumentos.parse_args(argv if argv is not None else [])


def main(argv: list[str] | None = None) -> int:
    opcoes = ler_opcoes(argv)
    sys.path.insert(0, str(PACOTE_MT5))
    import MetaTrader5 as mt5
    if opcoes.leitura:
        leitura(mt5, PASTAS)
        return 0
    inicio = datetime.now()
    resultado = rodar(mt5, PASTAS, datetime.now, _esperar, simular=opcoes.simular)
    print(resultado)
    if not opcoes.simular:
        _anotar(resultado, inicio)
    return 0 if resultado.desfecho != "recusado" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
