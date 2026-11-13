"""
Stage 1 of the reading of the app table's risk and return columns as point-in-time signals
(``docs/superpowers/specs/2026-11-13-indicadores-risco-design.md``, committed in ``ec8022c`` before any return-based
number): the monthly IC of the nine signals and of their mean normal score, raw and after the Composite's three
components, read once.

    python -X utf8 scripts/indicadores_risco_ic.py > docs/indicadores-risco-ic-2026-11-13.txt

House harness (``core.poder.ic_por_mes``): monthly Spearman IC of the month's score against the month's total return
in the month's universe, 2012-01 to 2026-09; gate IC >= 0.02 and t >= 2. Anchors first: the three components must
reproduce their published ICs or nothing else is read. No network.
"""
from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.crescimento_lucro import nota_de_crescimento
from core.indicadores_no_tempo import CONJUNTO, SINAIS, Fontes, residuo, tabela_no_mes
from core.poder import ic_por_mes
from core.sinais import conservadora
from core.valor import NotaDoMes
from core.views import nota_media_dos_escores
from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor
from scripts.avaliar_conservadora import resumo_do_ic, retornos_no_universo
from scripts.avaliar_robustez import conferir_hashes
from scripts.qpo_montar import FOLDER

PASTA_CVM = FOLDER.parent / "2026-10-24-cvm"
PASTA_MACRO = FOLDER.parent / "2026-10-21-macro"
PASTA_DE_SAIDA = FOLDER.parent / "2026-11-13-indicadores-risco"
ANCORAS = {"Conservadora": 0.085, "value": 0.070, "profit growth": 0.033}      # DECISOES 79, 161 and line 2261
TOLERANCIA_DA_ANCORA = 0.003
T_DE_BONFERRONI = 2.8           # ten readings at once: a t between the gate's 2 and this is reported as weak
Paineis = dict[str, pd.DataFrame]


def _painel(linhas: Mapping[pd.Timestamp, pd.Series], colunas: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(linhas).T.reindex(columns=colunas).astype(float)


def montar_paineis(fontes: Fontes, componentes: Mapping[str, NotaDoMes], universo: pd.DataFrame,
                   meses: pd.DatetimeIndex, colunas: pd.Index) -> tuple[Paineis, Paineis, Paineis]:
    """Month x ticker panels over the month's point-in-time universe: each signal, its residual on the components'
    normal scores, and each component."""
    elegiveis = universo["security"].groupby(pd.to_datetime(universo["month"])).apply(list)
    nomes = (*SINAIS, CONJUNTO)
    sinais = {nome: {} for nome in nomes}
    residuos = {nome: {} for nome in nomes}
    dos_componentes = {nome: {} for nome in componentes}
    for mes in meses:
        candidatos = elegiveis.get(mes, [])
        tabela = tabela_no_mes(fontes, mes, candidatos)
        notas = pd.DataFrame({nome: nota(mes, candidatos) for nome, nota in componentes.items()})
        for nome in componentes:
            dos_componentes[nome][mes] = notas[nome]
        for nome in nomes:
            sinais[nome][mes] = tabela[nome]
            residuos[nome][mes] = residuo(tabela[nome], notas.reindex(tabela.index))
    return ({nome: _painel(linhas, colunas) for nome, linhas in sinais.items()},
            {nome: _painel(linhas, colunas) for nome, linhas in residuos.items()},
            {nome: _painel(linhas, colunas) for nome, linhas in dos_componentes.items()})


def conferir_ancoras(dos_componentes: Paineis, retornos: pd.DataFrame) -> None:
    """Stops the reading unless each component reproduces its published IC: the harness is the registered one."""
    for nome, publicado in ANCORAS.items():
        resumo = resumo_do_ic(ic_por_mes(dos_componentes[nome], retornos))
        print(f"  anchor {nome}: IC {resumo['IC médio']:.4f} (t {resumo['t']:.2f}); published {publicado:.3f}")
        if abs(resumo["IC médio"] - publicado) > TOLERANCIA_DA_ANCORA:
            raise SystemExit(f"Anchor {nome} is off by more than {TOLERANCIA_DA_ANCORA}: nothing was read.")


def linha_do_sinal(nome: str, sinais: Paineis, residuos: Paineis, dos_componentes: Paineis,
                   retornos: pd.DataFrame) -> tuple[dict[str, float | str], pd.Series, pd.Series]:
    """The row of the summary table for one signal, with its monthly raw and residual ICs."""
    ics, do_residuo = ic_por_mes(sinais[nome], retornos).dropna(), ic_por_mes(residuos[nome], retornos).dropna()
    bruto, sobra = resumo_do_ic(ics), resumo_do_ic(do_residuo)
    passou = avaliar_valor.PORTAO.aprova(ics)
    novo = passou and sobra["IC médio"] > 0 and sobra["t"] >= avaliar_valor.PORTAO.t_minimo
    linha: dict[str, float | str] = {
        "months": bruto["meses"], "names": float(sinais[nome].notna().sum(axis=1).median()),
        "IC": bruto["IC médio"], "t": bruto["t"], "IC>0": bruto["meses com IC > 0"],
        **{rotulo: float(ics.loc[de:ate].mean()) for rotulo, (de, ate) in avaliar_valor.SUBPERIODOS_DO_IC.items()},
        "resid IC": sobra["IC médio"], "resid t": sobra["t"],
        **{f"rc {outro}": float(ic_por_mes(sinais[nome], painel).mean()) for outro, painel in dos_componentes.items()},
        "gate": "PASS" if passou else "fail",
        "stage 2": ("no" if not novo else "weak" if min(bruto["t"], sobra["t"]) < T_DE_BONFERRONI else "CANDIDATE")}
    return linha, ics, do_residuo


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = avaliar_valor.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    balancos, sha256_cvm = avaliar_valor.carregar_balancos(PASTA_CVM)
    sha256_cdi = conferir_hashes(PASTA_MACRO, ["cdi_daily.parquet"])
    cdi = pd.read_parquet(PASTA_MACRO / "cdi_daily.parquet")
    juro = pd.Series(cdi["cdi_pct"].to_numpy() / 100.0, index=pd.to_datetime(cdi["date"]))
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm | sha256_cdi}")

    fontes = Fontes(dados.precos, juro, mensal.retornos, balancos.fundamentos)
    componentes: dict[str, NotaDoMes] = {
        "Conservadora": lambda mes, candidatos: conservadora(mensal.retornos, mensal.proventos, mes, candidatos),
        "value": nota_media_dos_escores(*avaliar_valor.notas(balancos, mensal).values()),
        "profit growth": nota_de_crescimento(balancos.fundamentos)}
    meses = pd.date_range(*avaliar_valor.PERIODO_DO_IC, freq="MS")
    colunas = mensal.retornos.columns
    sinais, residuos, dos_componentes = montar_paineis(fontes, componentes, dados.universo, meses, colunas)
    retornos = retornos_no_universo(mensal.retornos, dados.universo, meses)
    print(f"\n=== Anchors ({meses[0]:%Y-%m} to {meses[-1]:%Y-%m}, {len(meses)} months; panels built in "
          f"{time.perf_counter() - inicio:.0f} s)")
    conferir_ancoras(dos_componentes, retornos)

    linhas, mensais = {}, {}
    for nome in sinais:
        linhas[nome], mensais[nome], mensais[f"{nome} resid"] = linha_do_sinal(nome, sinais, residuos,
                                                                              dos_componentes, retornos)
    print(f"\n=== Monthly Spearman IC of each signal (higher is better), raw and after the three components; gate "
          f"{avaliar_valor.PORTAO.nome}; stage 2 needs the gate and a residual IC > 0 with t >= 2 "
          f"(weak below t {T_DE_BONFERRONI})")
    print(pd.DataFrame(linhas).T.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n=== Mean monthly rank correlation among the signals")
    entre_si = pd.DataFrame({a: {b: float(ic_por_mes(sinais[a], sinais[b]).mean()) for b in sinais} for a in sinais})
    print(entre_si.to_string(float_format=lambda x: f"{x:.2f}"))
    PASTA_DE_SAIDA.mkdir(exist_ok=True)
    pd.DataFrame(mensais).rename_axis("month").to_csv(PASTA_DE_SAIDA / "ics.csv", float_format="%.6f")
    candidatos = [nome for nome, linha in linhas.items() if linha["stage 2"] != "no"]
    print(f"\nStage 2 candidates: {', '.join(candidatos) if candidatos else 'none'}")
    print(f"(total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
