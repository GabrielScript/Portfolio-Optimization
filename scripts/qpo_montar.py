"""
qpo_montar.py - builds every QPO QUBO (docs/superpowers/specs/2026-10-27-qpo-design.md, section 8, step 1).

Runs the QPO-exact arm in the engine for each phase only to reach the real decisions (dates and training windows),
and saves each decision's QUBO with its exact solution in ``docs/qpo/qubos/fase{F}/{date}.npz`` plus
``docs/qpo/qubos/SHA256SUMS``. It prints no performance number: the curves are read once, by ``qpo_avaliar.py``,
after the pre-registration commit.

    python scripts/qpo_montar.py N_MAX
"""

from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, walk_forward_dinamico  # noqa: E402
from core.config import taxa_anual_para_aritmetica  # noqa: E402
from core.qpo import CFG_MAX_SHARPE, K, QuboInstance, exact_picks, qubo_selector, selection_then_max_sharpe  # noqa: E402
from core.qpo import solve_exact  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
FOLDER = RAIZ / "docs" / "dados" / "2026-10-22"
LAST_SESSION = pd.Timestamp("2026-10-16")       # as scripts/rerun_conservadora.py: the published Conservadora
RUN_DATE = date(2026, 10, 27)                    # Selic and CDI as of the pre-registration
SAIDA = RAIZ / "docs" / "qpo" / "qubos"


def export_phase(registro: Mapping[pd.Timestamp, tuple[QuboInstance, tuple[int, ...]]], fase: int,
                 pasta: Path) -> list[Path]:
    """One npz per decision: the QUBO, its inputs and the exact optimum and worst feasible energy."""
    destino = pasta / f"fase{fase}"
    destino.mkdir(parents=True, exist_ok=True)
    arquivos = []
    for data, (instancia, picks) in sorted(registro.items()):
        exata = solve_exact(instancia.q, instancia.k)
        arquivo = destino / f"{data:%Y-%m-%d}.npz"
        np.savez(arquivo, q=instancia.q, tickers=np.array(instancia.tickers), alpha=instancia.alpha,
                 cov=instancia.cov, k=instancia.k, exact_picks=np.array(picks), exact_energy=exata.energy,
                 exact_worst=exata.worst)
        arquivos.append(arquivo)
    return arquivos


def write_sha256sums(pasta: Path) -> Path:
    """``sha256  relative/path`` per npz, sorted (the format ``conferir_hashes`` and ``sha256sum -c`` read)."""
    linhas = [f"{hashlib.sha256(a.read_bytes()).hexdigest()}  {a.relative_to(pasta).as_posix()}"
              for a in sorted(pasta.rglob("*.npz"))]
    arquivo = pasta / "SHA256SUMS"
    arquivo.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return arquivo


def main(argv: list[str]) -> int:
    n_max = int(argv[0])
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    precos = dados.precos.loc[:LAST_SESSION]
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    selic = market_data.obter_selic_meta(RUN_DATE)
    cdi = market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=RUN_DATE)
    cfg = replace(CFG_MAX_SHARPE, taxa_livre_risco=taxa_anual_para_aritmetica(selic.valor))
    print(f"Files (sha256): {dados.sha256 | mensal.sha256}; N_MAX {n_max}, k {K}")
    for fase in FASES:
        registro: dict = {}
        estrategia = selection_then_max_sharpe(qubo_selector(dados.setores, n_max, K, exact_picks, registro),
                                               mensal.retornos, mensal.proventos, dados.setores, cfg, cdi)
        walk_forward_dinamico(_acoes(precos.iloc[fase:]), dados.universo, estrategia, CFG_BT, "QPO-exact",
                              Congelamento())
        arquivos = export_phase(registro, fase, SAIDA)
        print(f"phase {fase}: {len(arquivos)} QUBOs ({time.perf_counter() - inicio:.0f} s)", flush=True)
    print(f"{write_sha256sums(SAIDA)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
