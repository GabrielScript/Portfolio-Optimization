"""Tests for the forward-holdout ledger helpers (no lake, no model)."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.qml_holdout import (comprar, primeira_sessao, rebalancear, rebalanceio_devido,
                                 valorizar)


def test_primeira_sessao_is_the_first_on_or_after_the_start() -> None:
    sessoes = pd.bdate_range("2026-11-02", periods=10)
    assert primeira_sessao(sessoes, pd.Timestamp("2026-11-09")) == pd.Timestamp("2026-11-09")
    assert primeira_sessao(sessoes, pd.Timestamp("2026-11-07")) == pd.Timestamp("2026-11-09")


def test_primeira_sessao_raises_when_the_window_has_not_started() -> None:
    sessoes = pd.bdate_range("2026-11-02", periods=3)
    with pytest.raises(ValueError):
        primeira_sessao(sessoes, pd.Timestamp("2027-01-01"))


def test_rebalanceio_devido_counts_sessions_not_days() -> None:
    sessoes = pd.bdate_range("2026-11-02", periods=200)
    assert not rebalanceio_devido(sessoes, sessoes[62], sessoes[0], 63)
    assert rebalanceio_devido(sessoes, sessoes[63], sessoes[0], 63)


def test_buying_charges_the_cost_and_keeps_the_weights() -> None:
    pesos = pd.Series({"AAAA3": 0.5, "BBBB3": 0.5})
    precos = pd.Series({"AAAA3": 10.0, "BBBB3": 20.0})
    cotas = comprar(pesos, precos, 10_000.0, custo_bps=10.0)
    assert valorizar(cotas, precos) == pytest.approx(10_000.0 * (1 - 10 / 10_000))
    assert cotas["AAAA3"] * 10.0 == pytest.approx(cotas["BBBB3"] * 20.0)


def test_buying_raises_on_a_missing_or_zero_price() -> None:
    pesos = pd.Series({"AAAA3": 1.0})
    with pytest.raises(ValueError):
        comprar(pesos, pd.Series({"AAAA3": 0.0}), 1_000.0)
    with pytest.raises(ValueError):
        comprar(pesos, pd.Series(dtype=float), 1_000.0)


def test_rebalancing_to_the_same_book_costs_nothing() -> None:
    pesos = pd.Series({"AAAA3": 0.5, "BBBB3": 0.5})
    precos = pd.Series({"AAAA3": 10.0, "BBBB3": 20.0})
    cotas = comprar(pesos, precos, 1_000.0, custo_bps=0.0)
    novas, giro = rebalancear(cotas, pesos, precos, custo_bps=10.0)
    assert giro == pytest.approx(0.0, abs=1e-9)
    assert valorizar(novas, precos) == pytest.approx(1_000.0)


def test_rebalancing_charges_only_on_the_turnover() -> None:
    precos = pd.Series({"AAAA3": 10.0, "BBBB3": 20.0})
    cotas = comprar(pd.Series({"AAAA3": 1.0, "BBBB3": 0.0}), precos, 1_000.0, custo_bps=0.0)
    alvo = pd.Series({"AAAA3": 0.5, "BBBB3": 0.5})
    novas, giro = rebalancear(cotas, alvo, precos, custo_bps=10.0)
    assert giro == pytest.approx(1.0)                       # half sold, half bought
    assert valorizar(novas, precos) == pytest.approx(1_000.0 * (1 - 10 / 10_000))


def test_rebalancing_charges_the_sale_of_a_name_that_leaves_the_target() -> None:
    """Review F1: a name sold in full is turnover too. Half in each name, then all in one: turnover 1.0, not 0.5."""
    precos = pd.Series({"AAAA3": 10.0, "BBBB3": 20.0})
    cotas = comprar(pd.Series({"AAAA3": 0.5, "BBBB3": 0.5}), precos, 1_000.0, custo_bps=0.0)
    novas, giro = rebalancear(cotas, pd.Series({"AAAA3": 1.0}), precos, custo_bps=10.0)
    assert giro == pytest.approx(1.0)
    assert valorizar(novas, precos) == pytest.approx(1_000.0 * (1 - 10 / 10_000))


def test_holdings_between_rebalances_do_not_depend_on_how_often_the_ledger_runs() -> None:
    """Shares come from the last rebalance; rebuilding them at the latest mark would rebalance for free."""
    from scripts.qml_holdout import cotas_em_carteira

    datas = pd.bdate_range("2026-11-09", periods=3)
    precos = pd.DataFrame({"AAAA3": [10.0, 20.0, 20.0], "BBBB3": [10.0, 10.0, 5.0]}, index=datas)
    ordens = pd.DataFrame({"data": datas[0], "ticker": ["AAAA3", "BBBB3"], "peso": [0.5, 0.5]})
    um_dia = pd.DataFrame({"data": datas[:1], "valor": [1_000.0]})
    dois_dias = pd.DataFrame({"data": datas[:2], "valor": [1_000.0, 1_500.0]})
    esperado = pd.Series({"AAAA3": 50.0, "BBBB3": 50.0})
    pd.testing.assert_series_equal(cotas_em_carteira(ordens, um_dia, precos), esperado, check_names=False)
    pd.testing.assert_series_equal(cotas_em_carteira(ordens, dois_dias, precos), esperado, check_names=False)
    assert valorizar(cotas_em_carteira(ordens, dois_dias, precos), precos.loc[datas[2]]) == pytest.approx(1_250.0)


def test_marking_refuses_a_holding_without_a_price() -> None:
    """A session with no trade must not count the position as zero."""
    cotas = pd.Series({"AAAA3": 10.0, "BBBB3": 10.0})
    with pytest.raises(ValueError):
        valorizar(cotas, pd.Series({"AAAA3": 10.0, "BBBB3": float("nan")}))
    with pytest.raises(ValueError):
        valorizar(cotas, pd.Series({"AAAA3": 10.0}))


def test_the_two_accounts_keep_separate_ledgers() -> None:
    """Each account owns its own ledger folder and its three files."""
    from scripts.qml_holdout import CONTA_FIXA, CONTA_PCA

    assert CONTA_FIXA.pasta != CONTA_PCA.pasta
    assert CONTA_PCA.pasta.name == "qml_holdout_pca"
    for conta in (CONTA_FIXA, CONTA_PCA):
        assert conta.estado == conta.pasta / "estado.json"
        assert conta.ordens == conta.pasta / "ordens.csv"
        assert conta.marcas == conta.pasta / "marcas.csv"


def test_only_the_pca_account_changes_the_model() -> None:
    """The frozen account keeps the legacy model; the PCA account pins seed and covariance."""
    from scripts.qml_holdout import CONTA_FIXA, CONTA_PCA

    assert CONTA_FIXA.covariance is None
    assert CONTA_FIXA.init_seed is None
    assert CONTA_PCA.covariance is not None
    assert callable(CONTA_PCA.covariance)
    assert CONTA_PCA.init_seed == 20261110
    assert CONTA_FIXA.estrategia == "QML-Composite (Meio-Kelly)"
    assert CONTA_PCA.estrategia != CONTA_FIXA.estrategia


def test_the_pca_account_starts_on_or_after_its_pre_registration() -> None:
    """The new account starts on 2026-11-10; the frozen one keeps its 2026-11-09 start."""
    from scripts.qml_holdout import CONTA_FIXA, CONTA_PCA

    assert CONTA_PCA.inicio == pd.Timestamp("2026-11-10")
    assert CONTA_FIXA.inicio == pd.Timestamp("2026-11-09")


def test_the_registry_names_the_two_accounts() -> None:
    """The CLI names map to exactly these two accounts."""
    from scripts.qml_holdout import CONTA_FIXA, CONTA_PCA, CONTAS

    assert CONTAS == {"ledoit-wolf": CONTA_FIXA, "pca": CONTA_PCA}


def test_the_commands_default_to_the_frozen_account() -> None:
    """Every command keeps the frozen account as its default, so nothing changes for it."""
    import inspect

    from scripts import qml_holdout
    from scripts.qml_holdout import CONTA_FIXA

    for nome in ("montar_contexto", "iniciar", "rodar", "relatorio"):
        assinatura = inspect.signature(getattr(qml_holdout, nome))
        assert assinatura.parameters["conta"].default is CONTA_FIXA


def test_each_account_writes_its_own_ledger(tmp_path, monkeypatch) -> None:
    """``iniciar`` writes the three files under the account's folder, and its name into the estado."""
    from scripts import qml_holdout

    datas = pd.bdate_range("2026-11-09", periods=2)
    precos = pd.DataFrame({"AAAA3": [10.0, 11.0]}, index=datas)

    def fake_contexto(lago, conta=qml_holdout.CONTA_FIXA):
        return qml_holdout.Contexto(precos, None, None, lambda dia: pd.Series({"AAAA3": 1.0}),
                                    None, 1260)

    monkeypatch.setattr(qml_holdout, "montar_contexto", fake_contexto)
    conta = qml_holdout.Conta(tmp_path / "conta", "test account", inicio=pd.Timestamp("2026-11-10"))

    estado = qml_holdout.iniciar(None, conta=conta)

    assert estado["estrategia"] == "test account"
    assert estado["inicio"] == "2026-11-10"
    for caminho in (conta.estado, conta.ordens, conta.marcas):
        assert caminho.exists()
        assert caminho.parent == tmp_path / "conta"
    ordens = pd.read_csv(conta.ordens)
    assert list(ordens.columns) == ["data", "ticker", "peso"]


def _conta_aberta(tmp_path, monkeypatch, precos, pesos_fn, passo):
    """An account opened on the first session at 50/50, then run with ``pesos_fn`` and a ``passo``-session step."""
    import json

    from scripts import qml_holdout

    abertura = {"pesos": lambda dia: pd.Series({"AAAA3": 0.5, "BBBB3": 0.5})}

    def fake_contexto(lago, conta=qml_holdout.CONTA_FIXA):
        return qml_holdout.Contexto(precos, None, None, lambda dia: abertura["pesos"](dia), None, 1260)

    monkeypatch.setattr(qml_holdout, "montar_contexto", fake_contexto)
    conta = qml_holdout.Conta(tmp_path / "conta", "test account", inicio=precos.index[0])
    qml_holdout.iniciar(None, conta=conta)
    estado = json.loads(conta.estado.read_text(encoding="utf-8"))
    estado["passo"] = passo
    conta.estado.write_text(json.dumps(estado), encoding="utf-8")
    abertura["pesos"] = pesos_fn
    return conta


def test_a_refused_rebalance_keeps_the_book_as_the_engine_does(tmp_path, monkeypatch) -> None:
    """Review F2: a refused or failed target must not stop the ledger; the engine keeps the current book."""
    import json

    from core.backtest import CarteiraRecusada
    from scripts import qml_holdout

    datas = pd.bdate_range("2026-11-09", periods=3)
    precos = pd.DataFrame({"AAAA3": [10.0, 20.0, 20.0], "BBBB3": [10.0, 10.0, 5.0]}, index=datas)

    def recusa(dia):
        raise CarteiraRecusada([])

    conta = _conta_aberta(tmp_path, monkeypatch, precos, recusa, passo=1)
    qml_holdout.rodar(None, conta=conta)

    marcas = pd.read_csv(conta.marcas, parse_dates=["data"]).set_index("data")["valor"]
    investido = 10_000.0 * (1 - 10 / 10_000)                   # the opening buy paid 10 bps
    assert marcas.loc[datas[1]] == pytest.approx(investido * 1.5)   # held, no cost on a kept book
    assert marcas.loc[datas[2]] == pytest.approx(investido * 1.25)
    estado = json.loads(conta.estado.read_text(encoding="utf-8"))
    assert estado["ultimo_rebalanceio"] == str(datas[2].date())    # the step moves on, as in the engine
    assert len(estado["falhas"]) == 2


def _conta_com_lago_movel(tmp_path, monkeypatch, precos):
    """An account opened 50/50 on the first session; ``lago['ate']`` sets how many sessions the lake shows,
    ``lago['alvo']`` the target by date. Returns (conta, lago)."""
    from scripts import qml_holdout

    lago = {"ate": 1, "alvo": {}}

    def fake_contexto(lago_, conta=qml_holdout.CONTA_FIXA):
        return qml_holdout.Contexto(precos.iloc[:lago["ate"]], None, None,
                                    lambda dia: lago["alvo"].get(dia, pd.Series({"AAAA3": 0.5, "BBBB3": 0.5})),
                                    None, 1260)

    monkeypatch.setattr(qml_holdout, "montar_contexto", fake_contexto)
    conta = qml_holdout.Conta(tmp_path / "conta", "test account", inicio=precos.index[0])
    qml_holdout.iniciar(None, conta=conta)
    return conta, lago


def test_a_stale_estado_does_not_rebalance_twice(tmp_path, monkeypatch) -> None:
    """Review F8: a crash after a rebalance row left ``ultimo_rebalanceio`` behind; the next run must take the last
    rebalance from the orders, not rebalance again (with cost) the following day."""
    import json

    from scripts import qml_holdout

    datas = pd.bdate_range("2026-11-09", periods=3)
    precos = pd.DataFrame({"AAAA3": [10.0, 10.0, 10.0], "BBBB3": [10.0, 10.0, 10.0]}, index=datas)
    conta, lago = _conta_com_lago_movel(tmp_path, monkeypatch, precos)
    estado = json.loads(conta.estado.read_text(encoding="utf-8"))
    estado["passo"] = 1
    conta.estado.write_text(json.dumps(estado), encoding="utf-8")
    lago["alvo"] = {datas[1]: pd.Series({"AAAA3": 1.0}), datas[2]: pd.Series({"BBBB3": 1.0})}
    lago["ate"] = 2
    qml_holdout.rodar(None, conta=conta)                      # rebalances on datas[1]

    estado = json.loads(conta.estado.read_text(encoding="utf-8"))
    estado["ultimo_rebalanceio"] = str(datas[0].date())       # what the crash left behind
    estado["passo"] = 2
    conta.estado.write_text(json.dumps(estado), encoding="utf-8")
    lago["ate"] = 3
    qml_holdout.rodar(None, conta=conta)                      # datas[2] is 1 session after datas[1]: not due

    ordens = pd.read_csv(conta.ordens, parse_dates=["data"])
    assert ordens["data"].max() == datas[1]


def test_a_crash_inside_a_run_leaves_the_ledger_as_it_was(tmp_path, monkeypatch) -> None:
    """Review F8: a run lands whole or not at all; half a run (an order without its mark) breaks every later run."""
    import json

    from scripts import qml_holdout

    datas = pd.bdate_range("2026-11-09", periods=3)
    precos = pd.DataFrame({"AAAA3": [10.0, 10.0, 10.0], "BBBB3": [10.0, 10.0, 10.0]}, index=datas)
    conta, lago = _conta_com_lago_movel(tmp_path, monkeypatch, precos)
    estado = json.loads(conta.estado.read_text(encoding="utf-8"))
    estado["passo"] = 1
    conta.estado.write_text(json.dumps(estado), encoding="utf-8")

    class Quebra(dict):
        def get(self, dia, padrao=None):
            if dia == datas[2]:
                raise RuntimeError("the process died here")
            return pd.Series({"AAAA3": 1.0})

    lago["alvo"] = Quebra()
    lago["ate"] = 3
    antes = {caminho: caminho.read_bytes() for caminho in (conta.estado, conta.ordens, conta.marcas)}
    with pytest.raises(RuntimeError):
        qml_holdout.rodar(None, conta=conta)
    assert {caminho: caminho.read_bytes() for caminho in antes} == antes


class _PredictorDeTeste:
    """Stands in for QMLPredictor: counts trainings and records the installed model."""

    def __init__(self, impressao: str) -> None:
        import torch

        self.config = None
        self.impressao = impressao
        self.treinos = 0
        self.instalado = None
        self.new_model = lambda: torch.nn.Linear(2, 1, dtype=torch.float64)

    block_of = staticmethod(lambda ano: ano - ano % 2)

    def training_fingerprint(self, ano):
        return self.impressao

    def model_for(self, ano):
        self.treinos += 1
        self.instalado = self.new_model()
        return self.instalado

    def install_model(self, ano, modelo):
        self.instalado = modelo


def test_a_block_model_is_trained_once_then_loaded_from_the_ledger(tmp_path) -> None:
    """Review F4: the frozen model is a file. The first rebalance of a block trains and saves it; every later
    run loads that file instead of retraining on whatever the lake holds then."""
    import torch

    from scripts.qml_holdout import congelar_bloco

    primeiro = _PredictorDeTeste("abc")
    assert congelar_bloco(primeiro, 2027, tmp_path) is None
    assert primeiro.treinos == 1
    assert (tmp_path / "modelo_2026.pt").exists() and (tmp_path / "modelo_2026.json").exists()

    depois = _PredictorDeTeste("abc")
    assert congelar_bloco(depois, 2026, tmp_path) is None
    assert depois.treinos == 0
    for nome, valor in primeiro.instalado.state_dict().items():
        assert torch.equal(depois.instalado.state_dict()[nome], valor)


def test_a_revised_lake_is_reported_and_the_saved_model_still_serves(tmp_path) -> None:
    """Review F4: drift in the training data is visible, and it does not swap the frozen model."""
    from scripts.qml_holdout import congelar_bloco

    primeiro = _PredictorDeTeste("abc")
    congelar_bloco(primeiro, 2026, tmp_path)
    revisto = _PredictorDeTeste("xyz")
    aviso = congelar_bloco(revisto, 2027, tmp_path)
    assert aviso is not None and "block 2026" in aviso
    assert revisto.treinos == 0 and revisto.instalado is not None


def test_the_strategy_source_freezes_the_block_of_the_prediction_date(tmp_path) -> None:
    """Review F4: the adapter the strategy calls freezes the block of the date it predicts (once per block) and
    passes drift warnings on."""
    from scripts.qml_holdout import _FonteDoPredictor, congelar_bloco

    primeiro = _PredictorDeTeste("abc")
    congelar_bloco(primeiro, 2026, tmp_path)

    revisto = _PredictorDeTeste("xyz")
    revisto.predict = lambda data: pd.Series({"AAAA3": 0.6})
    avisos: list[str] = []
    fonte = _FonteDoPredictor(revisto, tmp_path, avisos)
    fonte.predict(pd.Timestamp("2026-12-30"))
    fonte.predict(pd.Timestamp("2027-02-10"))
    assert revisto.treinos == 0 and revisto.instalado is not None
    assert len(avisos) == 1 and "block 2026" in avisos[0]


def test_the_rebalance_records_the_model_warnings_in_the_estado(tmp_path, monkeypatch) -> None:
    """Review F4: a warning raised while the target is computed lands in estado.json, not in a lost stdout."""
    import json

    from scripts import qml_holdout

    datas = pd.bdate_range("2026-11-09", periods=2)
    precos = pd.DataFrame({"AAAA3": [10.0, 10.0], "BBBB3": [10.0, 10.0]}, index=datas)
    conta, lago = _conta_com_lago_movel(tmp_path, monkeypatch, precos)
    estado = json.loads(conta.estado.read_text(encoding="utf-8"))
    estado["passo"] = 1
    conta.estado.write_text(json.dumps(estado), encoding="utf-8")
    contexto = qml_holdout.montar_contexto

    def com_aviso(lago_, conta=qml_holdout.CONTA_FIXA):
        ctx = contexto(lago_, conta)
        pesos = ctx.pesos

        def pesos_com_aviso(dia):
            ctx.avisos.append("block 2026: drift")
            return pesos(dia)

        ctx.pesos = pesos_com_aviso
        return ctx

    monkeypatch.setattr(qml_holdout, "montar_contexto", com_aviso)
    lago["ate"] = 2
    qml_holdout.rodar(None, conta=conta)
    estado = json.loads(conta.estado.read_text(encoding="utf-8"))
    assert estado["avisos"] == ["2026-11-10: block 2026: drift"]


def _main_block():
    """The one ``if __name__ == "__main__":`` block of the script, compiled."""
    import ast
    from pathlib import Path

    fonte = (Path(__file__).resolve().parent.parent / "scripts" / "qml_holdout.py").read_text(encoding="utf-8")
    blocos = [n for n in ast.parse(fonte).body
              if isinstance(n, ast.If) and ast.unparse(n.test) in {"__name__ == '__main__'", '__name__ == "__main__"'}]
    assert len(blocos) == 1, f"expected exactly one __main__ block, found {len(blocos)}"
    return compile(ast.Module(body=blocos[0].body, type_ignores=[]), "<qml_holdout>", "exec")


def _run_cli(argv):
    """Exec the CLI block with the commands stubbed, returning [(command, lake, account)]."""
    from pathlib import Path
    from types import SimpleNamespace

    from scripts import qml_holdout as qh

    chamadas = []

    def fake(nome):
        def _f(lago, conta=None):
            chamadas.append((nome, lago, conta))
            return nome
        return _f

    espaco = {"sys": SimpleNamespace(argv=["qml_holdout.py", *argv]), "Path": Path,
              "CONTAS": qh.CONTAS, "CONTA_FIXA": qh.CONTA_FIXA,
              "iniciar": fake("iniciar"), "rodar": fake("rodar"), "relatorio": fake("relatorio"),
              "print": lambda *a, **k: None}
    exec(_main_block(), espaco)
    return chamadas


def test_the_cli_runs_one_command_of_the_requested_account() -> None:
    """A duplicated __main__ block used to run the frozen account after every PCA call."""
    from scripts.qml_holdout import CONTA_PCA

    chamadas = _run_cli(["--lago", "LAKE", "--conta", "pca", "rodar"])
    assert [c[0] for c in chamadas] == ["rodar"]
    assert chamadas[0][2] is CONTA_PCA
    assert str(chamadas[0][1]) == "LAKE"


def test_the_cli_defaults_to_the_frozen_account() -> None:
    from scripts.qml_holdout import CONTA_FIXA

    chamadas = _run_cli(["relatorio"])
    assert [(c[0], c[2]) for c in chamadas] == [("relatorio", CONTA_FIXA)]


def test_the_cli_rejects_an_unknown_account() -> None:
    with pytest.raises(SystemExit):
        _run_cli(["--conta", "nope", "rodar"])


def test_the_account_choices_reach_the_model(monkeypatch) -> None:
    """montar_contexto passes the account's seed to the model and its covariance to the strategy."""
    import sys
    import types
    from pathlib import Path
    from types import SimpleNamespace

    from scripts.qml_holdout import CONTA_FIXA, CONTA_PCA, montar_contexto

    prisioneiro: dict = {}

    qml = types.ModuleType("core.qml")

    class QMLConfig:
        def __init__(self, init_seed=None) -> None:
            self.init_seed = init_seed

    class QMLPredictor:
        def __init__(self, retornos, config, log=None) -> None:
            self.config = config
            prisioneiro["seed"] = config.init_seed

        def predict(self, data):
            return None

    def qml_strategy(fonte, cfg, setores, rf=None, selector=None, tilt=1.0, base="rf",
                     limites=None, covariance=None):
        prisioneiro["covariance"] = covariance
        return lambda treino: pd.Series({"AAAA3": 1.0})

    qml.QMLConfig = QMLConfig
    qml.QMLPredictor = QMLPredictor
    qml.qml_strategy = qml_strategy

    dinamico = types.ModuleType("scripts.avaliar_universo_dinamico")
    dinamico.CFG_BT = SimpleNamespace(janela_treino=2)
    dinamico.CFG_KELLY = object()

    vivo = types.ModuleType("scripts.curvas_ao_vivo")
    vivo.ARQUIVOS = []
    vivo.congelar = lambda lago, entradas, arquivos: None

    avaliar = types.ModuleType("scripts.qml_avaliar")
    avaliar.seletor_do_composite = lambda nota, setores: None

    datas = pd.bdate_range("2026-11-02", periods=5)
    precos = pd.DataFrame({"AAAA3": [10.0, 10.5, 11.0, 11.5, 12.0]}, index=datas)
    dia = datas[3]
    dados = SimpleNamespace(precos=precos, setores={"AAAA3": "Teste"},
                            universo=pd.DataFrame({"month": [dia.to_period("M").to_timestamp()],
                                                   "security": ["AAAA3"]}))
    g2 = types.ModuleType("scripts.qml_g2_avaliar")
    g2._carregar = lambda entradas: (dados, None, None, None, None)

    for nome, modulo in (("core.qml", qml), ("scripts.avaliar_universo_dinamico", dinamico),
                         ("scripts.curvas_ao_vivo", vivo), ("scripts.qml_avaliar", avaliar),
                         ("scripts.qml_g2_avaliar", g2)):
        monkeypatch.setitem(sys.modules, nome, modulo)

    ctx = montar_contexto(Path("."), CONTA_PCA)
    ctx.pesos(dia)
    assert prisioneiro["seed"] == 20261110
    assert prisioneiro["covariance"] is CONTA_PCA.covariance

    ctx = montar_contexto(Path("."), CONTA_FIXA)
    ctx.pesos(dia)
    assert prisioneiro["seed"] is None
    assert prisioneiro["covariance"] is None
