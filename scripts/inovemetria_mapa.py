"""
The disruption map of the Inovemetria reading (spec "The disruption map"): one row per candidate of the last decision
month of the frozen data, every family's value and normal score, INOV, the chosen growth law and stage, the
forecast real growth, x/m, the share slope, applications in 60 months and the leading WIPO field. Descriptive.

    python -X utf8 scripts/inovemetria_mapa.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.inovemetria.data import issuer_patents, load  # noqa: E402
from core.inovemetria.signals import (PATENT_MONTHS, SNAPSHOT_HOLD, diffusion_table, families,  # noqa: E402
                                      field_counts, patent_components)
from core.risco import emissor  # noqa: E402
from core.views import escore_normal, nota_media_dos_escores  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.qpo_montar import FOLDER  # noqa: E402

MONTH = pd.Timestamp("2026-10-01")
OUT = Path(__file__).resolve().parent.parent / "docs" / "inovemetria-mapa-2026-11-12.csv"
# WIPO's 35 technology fields (Schmoch 2008), the numbering of BADEPI's CAMPO_TEC
WIPO = {1: "Electrical machinery, energy", 2: "Audio-visual", 3: "Telecommunications", 4: "Digital communication",
        5: "Basic communication", 6: "Computer technology", 7: "IT methods for management", 8: "Semiconductors",
        9: "Optics", 10: "Measurement", 11: "Biological materials analysis", 12: "Control", 13: "Medical technology",
        14: "Organic fine chemistry", 15: "Biotechnology", 16: "Pharmaceuticals", 17: "Polymers", 18: "Food chemistry",
        19: "Basic materials chemistry", 20: "Materials, metallurgy", 21: "Surface technology, coating",
        22: "Micro-structure, nano", 23: "Chemical engineering", 24: "Environmental technology", 25: "Handling",
        26: "Machine tools", 27: "Engines, pumps, turbines", 28: "Textile and paper machines",
        29: "Other special machines", 30: "Thermal processes", 31: "Mechanical elements", 32: "Transport",
        33: "Furniture, games", 34: "Other consumer goods", 35: "Civil engineering"}


def main() -> int:
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    data, _ = load()
    universo = dados.universo
    candidatos = universo.loc[pd.to_datetime(universo["month"]) == MONTH, "security"].tolist()
    sinais = families(data, dados.setores, mensal.retornos)
    valores = pd.DataFrame({k: f(MONTH, candidatos) for k, f in sinais.items()}).reindex(candidatos)
    mapa = valores.add_prefix("value_")
    for k in sinais:
        mapa[f"score_{k}"] = escore_normal(valores[k].dropna())
    mapa["INOV"] = nota_media_dos_escores(*sinais.values())(MONTH, candidatos)
    mapa.insert(0, "sector", [dados.setores.get(t) for t in mapa.index])
    curvas = diffusion_table(data, MONTH)
    emissores = [emissor(t) for t in mapa.index]
    for coluna in ("law", "stage", "forecast", "x_over_m"):
        mapa[coluna] = curvas[coluna].reindex(emissores).to_numpy() if coluna in curvas else None
    campos = field_counts(data, MONTH)
    p1, _ = patent_components(data, MONTH)
    fim = min(MONTH, SNAPSHOT_HOLD)
    recentes = issuer_patents(data, fim)
    recentes = recentes[recentes["known_from"] >= fim - pd.DateOffset(months=PATENT_MONTHS)]
    janela = campos.sum(axis=1)
    mapa["applications_60m"] = recentes.groupby("issuer").size().reindex(emissores).fillna(0).to_numpy()
    lider = campos.idxmax(axis=1).where(janela > 0).map(WIPO)
    mapa["leading_field"] = lider.reindex(emissores).to_numpy()
    mapa["patents_per_bn_assets"] = p1.reindex(emissores).to_numpy()
    mapa.sort_values("INOV", ascending=False).rename_axis("ticker").to_csv(OUT, float_format="%.4f",
                                                                           lineterminator="\n")
    print(f"{len(mapa)} candidates of {MONTH:%Y-%m} -> {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
