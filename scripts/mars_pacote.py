"""
mars_pacote.py - the MARS data package (docs/superpowers/specs/2026-11-08-mars-design.md, section 4): for each test
year 2010-2026, the 50 most traded eligible names, their total-return prices, the five standardized features (price,
MACD, RSI, CCI, ADX), the Composite's score percentile and the CDI, with the train/validation/test cuts.

The two lake inputs are frozen first into docs/dados/2026-11-08-mars/ (median traded value per year and candidate;
high/close and low/close ratios of the chosen names) with SHA256SUMS; a second run reads only the frozen copies.

    python -X utf8 scripts/mars_pacote.py            # writes kan_run/mars/mars_pacote.npz
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.cdi_lago import daily_cdi  # noqa: E402
from core.mars_indicadores import adx, cci, macd_relativo, rsi  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI  # noqa: E402
from scripts.qml_avaliar_b3 import MACRO  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import _value_score  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
CONGELADOS = RAIZ / "docs" / "dados" / "2026-11-08-mars"
SAIDA = RAIZ / "kan_run" / "mars" / "mars_pacote.npz"
LAGO = Path("D:/b3-datalake")
ANOS = range(2010, 2027)
NOMES, COBERTURA, SESSOES_DE_VOLUME, AQUECIMENTO, JANELA, CORTE = 50, 0.95, 252, 60, 30, 5.0


@dataclass(frozen=True)
class Janelas:
    """Session positions in the price table: training start, validation start, test start, test end (exclusive)."""
    treino: int
    validacao: int
    teste: int
    fim: int


def janelas_do_ano(datas: pd.DatetimeIndex, ano: int) -> Janelas:
    """Training = the five calendar years before ano - 1 (from the 60-session warm-up on), validation = ano - 1,
    test = ano up to the last session."""
    def primeira(ano_: int) -> int:
        return int(datas.searchsorted(pd.Timestamp(ano_, 1, 1)))
    fim = int(datas.searchsorted(min(pd.Timestamp(ano + 1, 1, 1), LAST_SESSION + pd.Timedelta(days=1))))
    return Janelas(max(primeira(ano - 6), AQUECIMENTO), primeira(ano - 1), primeira(ano), fim)


def candidatos(precos: pd.DataFrame, universo: pd.DataFrame, ano: int, janelas: Janelas) -> list[str]:
    """Names in the January universe with a price on at least 95% of the training and validation sessions."""
    no_universo = set(universo.loc[universo["month"] == pd.Timestamp(ano, 1, 1), "security"])
    historico = precos.iloc[janelas.treino:janelas.teste]
    cobertos = historico.columns[historico.notna().mean() >= COBERTURA]
    return sorted(no_universo & set(cobertos))


def _sha256(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def congelar(precos: pd.DataFrame, universo: pd.DataFrame) -> None:
    """Freeze the median traded value of each year's candidates and the chosen names' high/low ratios."""
    CONGELADOS.mkdir(parents=True, exist_ok=True)
    datas = precos.index
    todos = sorted({nome for ano in ANOS for nome in candidatos(precos, universo, ano, janelas_do_ano(datas, ano))})
    painel = pd.read_parquet(LAGO / "exports" / "daily_liquid_panel.parquet", columns=["date", "security", "volume_brl"],
                             filters=[("security", "in", todos)])
    volume = painel.assign(date=pd.to_datetime(painel["date"])).pivot(index="date", columns="security",
                                                                       values="volume_brl")
    linhas = []
    for ano in ANOS:
        janelas = janelas_do_ano(datas, ano)
        antes = volume.loc[:datas[janelas.teste] - pd.Timedelta(days=1)].tail(SESSOES_DE_VOLUME)
        for nome in candidatos(precos, universo, ano, janelas):
            mediana = antes[nome].median() if nome in antes else np.nan
            linhas.append((ano, nome, float(mediana)))
    medianas = pd.DataFrame(linhas, columns=["ano", "security", "volume_mediano"])
    medianas.to_parquet(CONGELADOS / "volume_mediano.parquet", index=False)
    escolhidos = escolher(medianas)
    nomes = sorted({nome for lista in escolhidos.values() for nome in lista})
    partes = []
    for ano in range(2005, 2027):
        bruto = pd.read_parquet(LAGO / "bronze" / "cotahist" / f"year={ano}" / "data.parquet",
                                columns=["date", "ticker", "market_type", "high", "low", "close", "volume"])
        bruto = bruto[(bruto["market_type"].astype(int) == 10) & bruto["ticker"].isin(nomes)]   # cash market
        bruto = bruto.sort_values("volume").drop_duplicates(["date", "ticker"], keep="last")
        fechamento = bruto["close"].astype("float64")
        partes.append(pd.DataFrame({"date": pd.to_datetime(bruto["date"]), "security": bruto["ticker"],
                                    "razao_max": bruto["high"].astype("float64") / fechamento,
                                    "razao_min": bruto["low"].astype("float64") / fechamento}))
    faixas = pd.concat(partes, ignore_index=True)
    faixas = faixas[faixas["date"] <= LAST_SESSION]
    sem_faixa = sorted(set(nomes) - set(faixas["security"]))
    if sem_faixa:
        raise SystemExit(f"No COTAHIST high/low for {len(sem_faixa)} chosen names: {sem_faixa[:10]}")
    faixas.to_parquet(CONGELADOS / "faixas.parquet", index=False)
    somas = "".join(f"{_sha256(CONGELADOS / nome)}  {nome}\n" for nome in ("volume_mediano.parquet", "faixas.parquet"))
    (CONGELADOS / "SHA256SUMS").write_bytes(somas.encode("ascii"))


def escolher(medianas: pd.DataFrame) -> dict[int, list[str]]:
    """The 50 candidates of each year with the highest median traded value (ties by ticker)."""
    escolhidos = {}
    for ano, grupo in medianas.groupby("ano"):
        ordem = grupo.dropna(subset=["volume_mediano"]).sort_values(["volume_mediano", "security"],
                                                                     ascending=[False, True])
        escolhidos[int(ano)] = ordem["security"].head(NOMES).tolist()
    return escolhidos


def ler_congelados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The frozen inputs, checked against SHA256SUMS."""
    for linha in (CONGELADOS / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        esperado, nome = linha.split()
        if _sha256(CONGELADOS / nome) != esperado:
            raise SystemExit(f"{nome}: SHA256 differs from SHA256SUMS")
    return pd.read_parquet(CONGELADOS / "volume_mediano.parquet"), pd.read_parquet(CONGELADOS / "faixas.parquet")


def caracteristicas(fechamento: pd.Series, razao_max: pd.Series, razao_min: pd.Series, base: float) -> pd.DataFrame:
    """Price (log over the training start), MACD / price, RSI, CCI, ADX of one name, gaps carried forward."""
    preco = fechamento.ffill()
    maxima = preco * razao_max.reindex(preco.index).fillna(1.0).clip(lower=1.0)
    minima = preco * razao_min.reindex(preco.index).fillna(1.0).clip(upper=1.0)
    return pd.DataFrame({"preco": np.log(preco / base), "macd": macd_relativo(preco), "rsi": rsi(preco, JANELA),
                         "cci": cci(maxima, minima, preco, JANELA), "adx": adx(maxima, minima, preco, JANELA)})


def padronizar(bruto: np.ndarray, treino: slice) -> np.ndarray:
    """Each feature over the training window's mean and standard deviation (all names pooled), clipped to +/- 5."""
    amostra = bruto[treino].reshape(-1, bruto.shape[-1])
    media = np.nanmean(amostra, axis=0)
    desvio = np.nanstd(amostra, axis=0)
    return np.clip(np.nan_to_num((bruto - media) / np.where(desvio > 0, desvio, 1.0)), -CORTE, CORTE)


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    precos = dados.precos.loc[:LAST_SESSION]
    if not (CONGELADOS / "SHA256SUMS").exists():
        congelar(precos, dados.universo)
        print(f"Frozen into {CONGELADOS} ({time.perf_counter() - inicio:.0f} s)", flush=True)
    medianas, faixas = ler_congelados()
    escolhidos = escolher(medianas)
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, _ = _value_score(mensal, com_crescimento=True)
    cdi = daily_cdi(ANOS_DE_CDI, LAST_SESSION.date(), path=MACRO / "cdi_daily.parquet")
    max_por_nome = faixas.pivot(index="date", columns="security", values="razao_max")
    min_por_nome = faixas.pivot(index="date", columns="security", values="razao_min")
    saida: dict[str, np.ndarray] = {"anos": np.array(list(ANOS))}
    for ano in ANOS:
        janelas = janelas_do_ano(precos.index, ano)
        nomes = escolhidos[ano]
        if len(nomes) != NOMES:
            raise SystemExit(f"{ano}: {len(nomes)} names, not {NOMES}")
        corte = precos.iloc[:janelas.fim]
        bruto = np.stack([caracteristicas(corte[nome], max_por_nome.get(nome, pd.Series(dtype=float)),
                                          min_por_nome.get(nome, pd.Series(dtype=float)),
                                          float(corte[nome].iloc[janelas.treino:].dropna().iloc[0])).to_numpy()
                          for nome in nomes], axis=1)[janelas.treino:]
        datas = corte.index[janelas.treino:]
        meses = sorted({data.to_period("M").to_timestamp() for data in datas})
        por_mes = {mes: nota(mes, nomes).reindex(nomes).rank(pct=True) for mes in meses}
        nota_diaria = np.stack([por_mes[data.to_period("M").to_timestamp()].fillna(0.5).to_numpy() for data in datas])
        taxa = cdi.reindex(datas).ffill().fillna(0.0).to_numpy()
        ate_o_teste = slice(0, janelas.validacao - janelas.treino)
        saida |= {f"{ano}_nomes": np.array(nomes), f"{ano}_datas": datas.to_numpy().astype("datetime64[D]"),
                  f"{ano}_precos": corte[nomes].iloc[janelas.treino:].to_numpy(),
                  f"{ano}_caracteristicas": padronizar(bruto, ate_o_teste).astype("float32"),
                  f"{ano}_nota": nota_diaria.astype("float32"), f"{ano}_cdi": taxa,
                  f"{ano}_cortes": np.array([0, janelas.validacao - janelas.treino, janelas.teste - janelas.treino,
                                             janelas.fim - janelas.treino])}
        print(f"{ano}: {len(datas)} sessions, test {datas[janelas.teste - janelas.treino]:%d/%m/%Y} to "
              f"{datas[-1]:%d/%m/%Y}, first names {', '.join(nomes[:5])}", flush=True)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SAIDA, **saida)
    print(f"{SAIDA} sha256 {_sha256(SAIDA)} ({time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
