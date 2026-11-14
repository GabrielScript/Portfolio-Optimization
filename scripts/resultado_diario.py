"""
The 20:00 report (the user, 2026-11-13): the Composite's simulator account (a preview of the 22:30 upkeep, built on
the official closes read on the MT5 terminal) and the MT5 demo book, by email (``core.alerta_email``).

The Windows task "aleph resultado" runs it every weekday at 20:00 through the routines' launcher:
    <Python312>/pythonw.exe -X utf8 scripts/com_python_base.py scripts/rodar_paper.py resultado_diario
By hand:
    python -X utf8 scripts/resultado_diario.py --simular     (prints the email; sends and writes nothing)

No session today (BOVA11 has no daily bar for today) sends nothing. ``paper-mt5/patrimonio.csv`` keeps the book's
value of each day; ``%LOCALAPPDATA%/paper-resultado/runs.csv`` one line per run.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core import alerta_email, cdi_lago, precos_do_app, simulador  # noqa: E402
from core import resultado_diario as rd  # noqa: E402
from core.mt5_book import check_account  # noqa: E402
from core.mt5_corretora import Corretora, Terminal  # noqa: E402

CARTEIRA = "Composto"
SESSOES = 10                                   # daily bars read per symbol: the last session and today among them
DEPOIS_DO_FECHAMENTO = time(18, 15)            # B3's closing call ends at 18:00
PACOTE_MT5 = RAIZ / ".venv-mt5"
ARQUIVO_ORDENS = RAIZ / "simulador" / "ordens.csv"
ARQUIVO_CALENDARIO = RAIZ / "lago" / "precos_diarios.parquet"
ARQUIVO_RECENTES = RAIZ / "lago" / "precos_recentes.parquet"
PASTA_MT5 = RAIZ / "paper-mt5"
RUNS = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "paper-resultado" / "runs.csv"


@dataclass(frozen=True)
class Resultado:
    desfecho: str                  # enviado | simulado | sem pregão | antes do fechamento | sem credenciais
    nota: str = ""
    assunto: str = ""
    corpo: str = ""


def _fechamento_de_hoje(c: Corretora, papel: str, fechamentos: pd.Series, hoje: pd.Timestamp,
                        notas: list[str]) -> float | None:
    if hoje in fechamentos.index:
        return float(fechamentos[hoje])
    cotacao = c.cotacao(papel)
    if cotacao is not None and cotacao.time.normalize() == hoje and cotacao.last > 0:
        notas.append(f"{papel}: sem barra diária no MT5, usado o último negócio do dia")
        return cotacao.last
    notas.append(f"{papel}: sem negócio hoje no MT5, mantido o fechamento anterior")
    return None


def linha_do_simulador(c: Corretora, hoje: pd.Timestamp, notas: list[str]) -> tuple[rd.Linha, float]:
    """The Composite account's preview for ``hoje`` and BOVA11's change on the day."""
    ordens = simulador.ler_ordens(ARQUIVO_ORDENS)
    ordens = ordens[ordens["carteira"] == CARTEIRA]
    papeis = sorted({*ordens["ticker"], *simulador.REFERENCIAS})
    precos = precos_do_app.ler_fechamentos([f"{p}.SA" for p in papeis], simulador.INICIO.date(),
                                           (hoje - pd.Timedelta(days=1)).date(), ARQUIVO_CALENDARIO, ARQUIVO_RECENTES)
    precos = precos.rename(columns=lambda papel: papel.removesuffix(".SA"))
    ultimo = precos.index[-1]
    base, fechamento = {}, {}
    for papel in papeis:
        fechamentos = c.fechamentos(papel, SESSOES)
        if ultimo not in fechamentos.index:
            notas.append(f"{papel}: sem o fechamento de {ultimo:%d/%m/%Y} no MT5, preço mantido")
            continue
        base[papel] = float(fechamentos[ultimo])
        hoje_px = _fechamento_de_hoje(c, papel, fechamentos, hoje, notas)
        if hoje_px is not None:
            fechamento[papel] = hoje_px
    anteriores = c.fechamentos("BOVA11", SESSOES).index
    anteriores = anteriores[anteriores < hoje]
    if len(anteriores) and anteriores[-1] != ultimo:
        notas.append(f"o simulador parou em {ultimo:%d/%m/%Y}: a variação do dia vai desde esse fechamento")
    estendidos = rd.com_fechamento_de_hoje(precos, pd.Series(base), pd.Series(fechamento), hoje)
    cdi = cdi_lago.daily_selic(hoje.year - simulador.INICIO.year + 1, hoje.date())
    curvas = simulador.curvas(ordens, estendidos, cdi)
    bova = curvas["BOVA11"]
    return rd.linha(curvas[CARTEIRA], simulador.CAPITAL_ACOES), float(bova.iloc[-1] / bova.iloc[-2] - 1.0)


def linha_do_livro(c: Corretora, hoje: pd.Timestamp, agora: datetime, notas: list[str],
                   gravar: bool) -> tuple[rd.Linha | None, str]:
    """The MT5 demo book: cash from its fills plus its shares at today's close, kept in ``paper-mt5/patrimonio.csv``."""
    from scripts import mt5_executor as ex
    conta = c.conta()
    motivo = check_account(conta.is_demo, conta.server)
    if motivo:
        return None, motivo
    pastas = ex.Pastas(ex.PASTAS.registro, PASTA_MT5, ex.PASTAS.runs)
    fills = ex.sincronizar_fills(c, pastas, ex._ler(PASTA_MT5 / "execucoes.csv"), agora)
    detidas = c.acoes_da_carteira()
    valor = ex.caixa(fills)
    for nome, acoes in detidas.items():
        preco = _fechamento_de_hoje(c, nome, c.fechamentos(nome, SESSOES), hoje, notas)
        if preco is None:
            cotacao = c.cotacao(nome)
            preco = ex.preco(cotacao)
        valor += acoes * preco
    arquivo = PASTA_MT5 / "patrimonio.csv"
    historico = pd.read_csv(arquivo) if arquivo.exists() else pd.DataFrame(columns=["data", "patrimonio"])
    historico = rd.anotar_patrimonio(historico, hoje, valor)
    if gravar:
        PASTA_MT5.mkdir(parents=True, exist_ok=True)
        historico.assign(data=historico["data"].dt.strftime("%Y-%m-%d"), patrimonio=historico["patrimonio"].round(2)
                         ).to_csv(arquivo, index=False, lineterminator="\n")
    return rd.linha(rd.serie_do_historico(historico), ex.INICIAL), ""


def rodar(terminal: Terminal, agora: datetime, simular: bool = False) -> Resultado:
    hoje = pd.Timestamp(agora.date())
    if not simular and agora.time() < DEPOIS_DO_FECHAMENTO:
        return Resultado("antes do fechamento", f"{agora:%H:%M} is before {DEPOIS_DO_FECHAMENTO}")
    with Corretora(terminal).conectar() as c:
        if hoje not in c.fechamentos("BOVA11", SESSOES).index:
            return Resultado("sem pregão", f"BOVA11 has no daily bar for {hoje:%d/%m/%Y}")
        notas: list[str] = []
        sim, bova_dia = linha_do_simulador(c, hoje, notas)
        try:
            livro, sem_livro = linha_do_livro(c, hoje, agora, notas, gravar=not simular)
        except Exception as erro:          # noqa: BLE001 - the simulator's half still goes out
            livro, sem_livro = None, f"{type(erro).__name__}: {erro}"
    assunto, corpo = rd.mensagem(hoje, sim, bova_dia, livro, notas, sem_livro, simulador.CAPITAL_ACOES)
    if simular:
        return Resultado("simulado", "", assunto, corpo)
    credenciais = alerta_email.credenciais()
    if credenciais is None:
        return Resultado("sem credenciais", "GMAIL_USER / GMAIL_PASSWORD_APP missing", assunto, corpo)
    alerta_email.enviar(assunto, corpo, credenciais)
    return Resultado("enviado", assunto, assunto, corpo)


def _anotar(resultado: Resultado, inicio: datetime) -> None:
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    novo = not RUNS.exists()
    with RUNS.open("a", encoding="utf-8", newline="") as saida:
        escritor = csv.writer(saida, lineterminator="\n")
        if novo:
            escritor.writerow(["quando", "desfecho", "nota"])
        escritor.writerow([inicio.isoformat(timespec="seconds"), resultado.desfecho, resultado.nota[:300]])


def main(argv: list[str] | None = None) -> int:
    """Options from ``argv`` only: the routines' launcher calls ``main()`` with its own arguments in ``sys.argv``."""
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    argumentos.add_argument("--simular", action="store_true")
    opcoes = argumentos.parse_args(argv if argv is not None else [])
    sys.path.insert(0, str(PACOTE_MT5))
    import MetaTrader5 as mt5
    inicio = datetime.now()
    resultado = rodar(mt5, inicio, simular=opcoes.simular)
    if opcoes.simular:
        print(resultado.assunto, resultado.corpo, sep="\n\n")
    else:
        _anotar(resultado, inicio)
    print(resultado.desfecho, resultado.nota)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
