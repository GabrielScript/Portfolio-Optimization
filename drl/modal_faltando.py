"""The cells of the pre-registered grid still missing from the local and Modal result files, fanned out on Modal.

    modal run drl/modal_faltando.py

One container per cell, four cores each. Writes to ``drl/saidas/resultados_modal.csv`` (the same file
``drl.modal_grade`` writes, so a cell is never repeated). The read is tolerant of the results file's schema drift:
``resultados.csv`` gained a ``modo`` column mid-run, and pandas will not parse a file whose rows disagree on width.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import modal

RAIZ = Path(__file__).resolve().parent.parent
SAIDAS = RAIZ / "drl" / "saidas"

app = modal.App("aleph-drl-faltando")

imagem = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("gymnasium>=1.0", "stable-baselines3>=2.4", "pandas", "pyarrow", "numpy<3")
    .add_local_file(str(RAIZ / "lago" / "precos_diarios.parquet"), remote_path="/root/lago/precos_diarios.parquet")
    .add_local_python_source("drl")
)


@app.function(image=imagem, cpu=4, timeout=3600, max_containers=31)
def uma_celula(config: tuple[str, str, float, int, int, int]) -> dict:
    """One cell of the grid: (algoritmo, recompensa, custo_bps, ano_teste, semente, timesteps)."""
    from drl import dados, executar, protocolo

    algoritmo, recompensa, custo, ano, semente, timesteps = config
    fechamentos = dados.painel(Path("/root/lago/precos_diarios.parquet"))
    elegiveis = dados.elegiveis(fechamentos)
    janelas = {j.ano_teste: j for j in protocolo.janelas(fechamentos.index)}
    janela = janelas[ano]
    nomes = executar.nomes_elegiveis(elegiveis, janela.teste.start, 30)
    try:
        return executar.corrida(fechamentos, elegiveis, janela, nomes, algoritmo=algoritmo, recompensa=recompensa,
                                custo_bps=custo, timesteps=timesteps, seed=semente, n_envs=4)
    except Exception as erro:                    # noqa: BLE001 - a broken cell must not take the grid down
        return {"algoritmo": algoritmo, "recompensa": recompensa, "custo_bps": custo, "ano_teste": ano,
                "semente": semente, "erro": f"{type(erro).__name__}: {erro}"}


def _feitos(arquivo: Path) -> set[tuple[str, str, float, int, int]]:
    """The (algoritmo, recompensa, custo, ano, semente) keys already computed, from both result files."""
    import pandas as pd

    chaves: set[tuple[str, str, float, int, int]] = set()
    for caminho in (SAIDAS / "resultados.csv", SAIDAS / "resultados_modal.csv", arquivo):
        if not caminho.exists():
            continue
        try:
            d = pd.read_csv(caminho, on_bad_lines="skip")
        except Exception:                        # noqa: BLE001 - a half-written file must not stop the grid
            continue
        for r in d.itertuples():
            chaves.add((r.algoritmo, r.recompensa, float(r.custo_bps), int(r.ano_teste), int(r.semente)))
    return chaves


@app.local_entrypoint()
def principal(timesteps: int = 150_000, secoes: str = ""):
    """Run every grid cell whose key is not in the result files yet. ``secoes`` filters, e.g. 'dsr_drawdown'."""
    import pandas as pd

    from drl import grade as g

    ARQUIVO = SAIDAS / "resultados_modal.csv"
    feitos = _feitos(ARQUIVO)
    celulas = [(*ar, custo, ano, semente, timesteps)
               for (ar, semente, custo, ano) in itertools.product(g.AGENTES, (0,), g.CUSTOS, g.JANELAS)
               if (ar[0], ar[1], float(custo), int(ano), int(semente)) not in feitos]
    if secoes:
        celulas = [c for c in celulas if c[1] in {s.strip() for s in secoes.split(",")}]
    print(f"{len(celulas)} missing cells", flush=True)
    if not celulas:
        return
    linhas = list(uma_celula.map(celulas))
    ok = [linha for linha in linhas if "erro" not in linha]
    ARQUIVO.parent.mkdir(parents=True, exist_ok=True)
    if ok:
        pd.DataFrame(ok).to_csv(ARQUIVO, mode="a", header=not ARQUIVO.exists(), index=False)
    for linha in linhas:
        if "erro" in linha:
            print("FAIL", linha, flush=True)
        else:
            print(f"{linha['algoritmo']}/{linha['recompensa']}/{linha['custo_bps']:.0f}bps/{linha['ano_teste']} "
                  f"ret={linha['retorno_anual']:.1%} sharpe={linha['sharpe']:.2f} giro={linha['giro_anual']:.1f}",
                  flush=True)
    print(f"{len(ok)}/{len(celulas)} cells ok -> {ARQUIVO}", flush=True)
