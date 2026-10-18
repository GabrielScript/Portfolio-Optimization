"""Plano de compra: pesos alvo + cotações -> quantidades inteiras de ações."""

import numpy as np
import pandas as pd
import pytest

from core.config import LimiteGrupo
from core.ordens import (MotivoFora, RegrasCompra, acoes_que_cabem, limites_do_plano, plano_de_compra,
                         redistribuir_sem_cotacao)

SEM_CUSTO = dict(custo_bps=0.0, peso_maximo=1.0, quantidade_minima=2)


def _serie(**valores: float) -> pd.Series:
    return pd.Series(valores, dtype=float)


def test_exemplo_feito_a_mao_compra_o_piso_e_usa_a_sobra_onde_reduz_o_desvio():
    # alvo R$ 5.000 cada. Piso: A 151 (R$ 4.983), B 71 (R$ 4.970), caixa R$ 47.
    # +1 A (R$ 33) leva o desvio de A de 17 para -16: melhora. +1 B (R$ 70) não cabe.
    plano = plano_de_compra(_serie(A=0.5, B=0.5), _serie(A=33.0, B=70.0),
                            RegrasCompra(orcamento=10_000.0, **SEM_CUSTO))

    assert plano.itens.loc["A", "quantidade"] == 152
    assert plano.itens.loc["B", "quantidade"] == 71
    assert plano.investido == pytest.approx(152 * 33.0 + 71 * 70.0)
    assert plano.sobra == pytest.approx(10_000.0 - plano.investido)
    assert plano.fora == ()


def test_teto_de_peso_impede_a_compra_que_reduziria_o_desvio():
    # mesmo exemplo com teto de 50% = R$ 5.000: a 152ª ação de A (R$ 5.016) passaria do teto.
    plano = plano_de_compra(_serie(A=0.5, B=0.5), _serie(A=33.0, B=70.0),
                            RegrasCompra(orcamento=10_000.0, **SEM_CUSTO | {"peso_maximo": 0.5}))

    assert plano.itens.loc["A", "quantidade"] == 151


def test_sobra_que_paga_mais_uma_acao_e_investida_mesmo_passando_do_alvo():
    # alvo R$ 500 cada; piso 12 + 12 (R$ 960), caixa R$ 40. A 13ª de A deixa A R$ 20 acima do alvo
    # (antes, R$ 20 abaixo): o desvio não melhora, mas o caixa parado é pior para quem quer investir tudo.
    plano = plano_de_compra(_serie(A=0.5, B=0.5), _serie(A=40.0, B=40.0),
                            RegrasCompra(orcamento=1_000.0, **SEM_CUSTO))

    assert plano.itens.loc["A", "quantidade"] == 13
    assert plano.itens.loc["B", "quantidade"] == 12
    assert plano.sobra == 0.0


def test_quantidade_e_dividida_em_lote_padrao_e_fracionario():
    plano = plano_de_compra(_serie(A=0.5, B=0.5), _serie(A=33.0, B=70.0),
                            RegrasCompra(orcamento=10_000.0, **SEM_CUSTO))

    assert plano.itens.loc["A", "lote_padrao"] == 100
    assert plano.itens.loc["A", "fracionario"] == 52
    assert plano.itens.loc["B", "lote_padrao"] == 0
    assert plano.itens.loc["B", "fracionario"] == 71


def test_nunca_gasta_mais_que_o_orcamento_contando_o_custo():
    gerador = np.random.default_rng(3)
    for _ in range(50):
        pesos = pd.Series(gerador.dirichlet(np.ones(6)), index=list("ABCDEF"))
        cotacoes = pd.Series(gerador.uniform(2.0, 120.0, 6), index=list("ABCDEF"))
        regras = RegrasCompra(orcamento=float(gerador.uniform(100.0, 50_000.0)), custo_bps=30.0,
                              peso_maximo=1.0, quantidade_minima=2)

        plano = plano_de_compra(pesos, cotacoes, regras)

        assert plano.investido + plano.custo_estimado <= regras.orcamento + 1e-9
        assert plano.custo_estimado == pytest.approx(plano.investido * 30.0 / 10_000)
        assert plano.sobra == pytest.approx(regras.orcamento - plano.investido - plano.custo_estimado)


def test_nenhuma_posicao_tem_menos_que_a_quantidade_minima():
    gerador = np.random.default_rng(5)
    for _ in range(50):
        pesos = pd.Series(gerador.dirichlet(np.ones(8)), index=list("ABCDEFGH"))
        cotacoes = pd.Series(gerador.uniform(5.0, 90.0, 8), index=list("ABCDEFGH"))

        plano = plano_de_compra(pesos, cotacoes, RegrasCompra(orcamento=800.0, custo_bps=0.0,
                                                              peso_maximo=1.0, quantidade_minima=2))

        quantidades = plano.itens["quantidade"]
        assert ((quantidades == 0) | (quantidades >= 2)).all()


def test_acao_que_nao_cabe_no_teto_com_a_quantidade_minima_fica_de_fora():
    # R$ 500 e teto 20% = R$ 100 por ação. 2 de A custam R$ 115,72 > R$ 100.
    pesos = _serie(A=0.2, B=0.2, C=0.2, D=0.2, E=0.2)
    cotacoes = _serie(A=57.86, B=62.14, C=32.17, D=32.28, E=10.0)

    plano = plano_de_compra(pesos, cotacoes, RegrasCompra(orcamento=500.0, custo_bps=0.0,
                                                          peso_maximo=0.2, quantidade_minima=2))

    assert plano.fora == ("A", "B")
    assert plano.itens.loc["C", "quantidade"] == 3
    assert plano.itens.loc["E", "quantidade"] == 10
    assert (plano.itens["valor"] <= 0.2 * 500.0 + 1e-9).all()


def test_motivo_de_fora_e_o_teto_quando_2_acoes_passam_dele():
    # R$ 100 e teto 30% = R$ 30. Alvo de B = R$ 30; 2 de B = R$ 40: passa do teto, mas não do dobro do alvo.
    plano = plano_de_compra(_serie(A=0.3, B=0.3), _serie(A=1.0, B=20.0),
                            RegrasCompra(orcamento=100.0, custo_bps=0.0, peso_maximo=0.3, quantidade_minima=2))

    assert plano.fora == ("B",)
    assert plano.motivos_fora == {"B": MotivoFora.TETO}


def test_sobra_vai_primeiro_para_a_acao_mais_abaixo_do_alvo():
    # Alvos A R$ 50 e B R$ 5; piso 50 de A, caixa R$ 50. B (R$ 5 abaixo) leva 2 (R$ 20); depois cada compra
    # vai para quem está mais abaixo do alvo: A até 66, B a 3ª, A até 70. Gastar o caixa só em A (100 de A,
    # B de fora) daria desvios -50 e +5 (soma dos quadrados 2.525) contra -20 e -25 (1.025).
    plano = plano_de_compra(_serie(A=0.5, B=0.05), _serie(A=1.0, B=10.0),
                            RegrasCompra(orcamento=100.0, **SEM_CUSTO))

    assert plano.itens.loc["A", "quantidade"] == 70
    assert plano.itens.loc["B", "quantidade"] == 3
    assert plano.sobra == 0.0


def test_sobra_nao_passa_do_teto_do_grupo():
    # Alvos A R$ 25, B R$ 25 (grupo de 50% = R$ 50), C R$ 50 a R$ 60 (2 de C não cabem). Piso 2 + 2 (R$ 40),
    # caixa R$ 60. Sem o teto do grupo a sobra iria toda para A e B (5 + 5 = R$ 100); com ele, só +1 de A.
    grupo = LimiteGrupo("Energia", frozenset({"A", "B"}), 0.5)
    plano = plano_de_compra(_serie(A=0.25, B=0.25, C=0.5), _serie(A=10.0, B=10.0, C=60.0),
                            RegrasCompra(orcamento=100.0, **SEM_CUSTO, grupos=(grupo,)))

    assert plano.itens.loc["A", "quantidade"] == 3
    assert plano.itens.loc["B", "quantidade"] == 2


def test_motivo_de_fora_e_o_grupo_quando_o_teto_dele_ja_foi_atingido():
    # Grupo {A, B} com teto de 30% = R$ 30. Piso: 28 de A, 2 de C (R$ 60), caixa R$ 12. A sobe até 30 e enche
    # o grupo; sobram R$ 10, que pagam 2 de B (R$ 10), mas o grupo não tem mais folga.
    grupo = LimiteGrupo("Energia", frozenset({"A", "B"}), 0.3)
    plano = plano_de_compra(_serie(A=0.28, B=0.02, C=0.70), _serie(A=1.0, B=5.0, C=30.0),
                            RegrasCompra(orcamento=100.0, **SEM_CUSTO, grupos=(grupo,)))

    assert plano.sobra == pytest.approx(10.0)
    assert plano.motivos_fora == {"B": MotivoFora.GRUPO}


def test_limite_do_grupo_no_plano_nunca_fica_abaixo_do_peso_alvo_do_grupo():
    # O HRP ignora o teto do grupo: um alvo de 40% num grupo de teto 30% não pode ser cortado pelo plano.
    grupos = (LimiteGrupo("Energia", frozenset({"A", "B"}), 0.3), LimiteGrupo("Bancos", frozenset({"C"}), 0.3))

    limites = limites_do_plano(grupos, _serie(A=0.25, B=0.15, C=0.10, D=0.50))

    assert [(g.nome, g.teto) for g in limites] == [("Energia", pytest.approx(0.40)), ("Bancos", 0.3)]


def test_limite_do_plano_fica_em_100_por_cento_quando_o_alvo_do_grupo_passa_de_1_por_ponto_flutuante():
    # Universo de um setor só com teto de 100%: o alvo reescalado pode somar 1 + 1e-12.
    grupos = (LimiteGrupo("Energia", frozenset({"A", "B"}), 1.0),)

    limites = limites_do_plano(grupos, _serie(A=0.5, B=0.5 + 1e-12))

    assert limites[0].teto == 1.0


def test_motivo_de_fora_e_o_caixa_quando_2_acoes_nao_cabem_no_que_sobrou():
    # Piso: 85 de A (R$ 85), caixa R$ 15. Alvo de B = R$ 15; 2 de B = R$ 20: abaixo do dobro do alvo
    # e do teto, mas acima do caixa.
    plano = plano_de_compra(_serie(A=0.85, B=0.15), _serie(A=1.0, B=10.0),
                            RegrasCompra(orcamento=100.0, **SEM_CUSTO))

    assert plano.fora == ("B",)
    assert plano.motivos_fora == {"B": MotivoFora.CAIXA}


def test_motivo_e_o_teto_mesmo_quando_2_acoes_custam_o_dobro_do_alvo():
    # Teto R$ 30, alvo de B R$ 30, 2 de B = R$ 70. A fica no teto e sobram R$ 70 de caixa: B cabe no
    # caixa, mas não no teto. Passar do alvo não barra mais a compra (o guloso enche até nada caber).
    plano = plano_de_compra(_serie(A=0.3, B=0.3), _serie(A=1.0, B=35.0),
                            RegrasCompra(orcamento=100.0, custo_bps=0.0, peso_maximo=0.3, quantidade_minima=2))

    assert plano.fora == ("B",)
    assert plano.motivos_fora == {"B": MotivoFora.TETO}


def test_peso_efetivo_e_sobre_o_orcamento_e_fecha_com_custo_e_sobra():
    plano = plano_de_compra(_serie(A=0.6, B=0.4), _serie(A=21.37, B=8.9),
                            RegrasCompra(orcamento=3_000.0, custo_bps=10.0, peso_maximo=0.6,
                                         quantidade_minima=2))

    total = plano.itens["peso_efetivo"].sum() + (plano.custo_estimado + plano.sobra) / 3_000.0
    assert total == pytest.approx(1.0)
    assert list(plano.itens["peso_alvo"]) == [0.6, 0.4]


def test_pesos_zerados_nao_entram_no_plano():
    plano = plano_de_compra(_serie(A=1.0, B=0.0), _serie(A=10.0, B=5.0),
                            RegrasCompra(orcamento=1_000.0, **SEM_CUSTO))

    assert list(plano.itens.index) == ["A"]


def test_sobra_zero_nao_vira_negativa_por_ponto_flutuante():
    # 10 × R$ 10,00 + 10 bps (R$ 0,10) = R$ 100,10: sobra exatamente zero, e não −5,7e-15 ("R$ -0.00" na tela).
    plano = plano_de_compra(_serie(A=1.0), _serie(A=10.0),
                            RegrasCompra(orcamento=100.10, custo_bps=10.0, peso_maximo=1.0,
                                         quantidade_minima=2))

    assert plano.itens.loc["A", "quantidade"] == 10
    assert plano.sobra == 0.0


def test_orcamento_que_cabe_exatamente_com_o_custo_nao_e_recusado():
    # 2 × R$ 12,50 = R$ 25,00 + 16 bps (R$ 0,04) = R$ 25,04: cabe sem folga.
    plano = plano_de_compra(_serie(A=1.0), _serie(A=12.5),
                            RegrasCompra(orcamento=25.04, custo_bps=16.0, peso_maximo=1.0,
                                         quantidade_minima=2))

    assert plano.itens.loc["A", "quantidade"] == 2


def test_compra_que_bate_exatamente_no_teto_nao_e_recusada_por_ponto_flutuante():
    # teto = 0,29 × R$ 100, que em ponto flutuante dá 28,999999999999996; 2 × R$ 14,50 = R$ 29,00.
    plano = plano_de_compra(_serie(A=0.29), _serie(A=14.5),
                            RegrasCompra(orcamento=100.0, custo_bps=0.0, peso_maximo=0.29,
                                         quantidade_minima=2))

    assert plano.itens.loc["A", "quantidade"] == 2


def test_piso_respeita_o_teto_mesmo_com_peso_alvo_acima_dele():
    plano = plano_de_compra(_serie(A=0.6, B=0.4), _serie(A=10.0, B=10.0),
                            RegrasCompra(orcamento=10_000.0, custo_bps=0.0, peso_maximo=0.5,
                                         quantidade_minima=2))

    assert (plano.itens["valor"] <= 0.5 * 10_000.0 + 1e-9).all()


def test_pesos_que_somam_pouco_mais_de_100_por_cento_sao_reescalados_para_100():
    # soma 1 + 5e-7: dentro da tolerância aceita, mas sem reescalar o piso poderia gastar além do orçamento.
    plano = plano_de_compra(_serie(A=0.5, B=0.5 + 5e-7), _serie(A=3.33, B=7.77),
                            RegrasCompra(orcamento=1_000.0, **SEM_CUSTO))

    assert plano.itens["peso_alvo"].sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("cotacoes", [_serie(A=10.0), _serie(A=10.0, B=0.0), _serie(A=10.0, B=np.nan)])
def test_cotacao_ausente_ou_invalida_levanta_erro(cotacoes):
    with pytest.raises(ValueError, match="B"):
        plano_de_compra(_serie(A=0.5, B=0.5), cotacoes, RegrasCompra(orcamento=1_000.0, **SEM_CUSTO))


@pytest.mark.parametrize("pesos", [_serie(A=0.7, B=0.5), _serie(A=1.2, B=-0.2)])
def test_pesos_negativos_ou_somando_mais_de_100_por_cento_levantam_erro(pesos):
    with pytest.raises(ValueError, match="pesos"):
        plano_de_compra(pesos, _serie(A=10.0, B=10.0), RegrasCompra(orcamento=1_000.0, **SEM_CUSTO))


def test_acoes_que_cabem_tira_a_que_passa_do_teto_com_a_quantidade_minima():
    # R$ 500 e teto 20% = R$ 100: 2 de A custam R$ 115,72; 2 de B, R$ 20.
    regras = RegrasCompra(orcamento=500.0, custo_bps=10.0, peso_maximo=0.2, quantidade_minima=2)

    assert acoes_que_cabem(_serie(A=57.86, B=10.0), regras) == ["B"]


def test_acoes_que_cabem_mantem_a_que_bate_exatamente_no_teto():
    # teto = 0,29 × R$ 100, que em ponto flutuante dá 28,999999999999996; 2 × R$ 14,50 = R$ 29,00.
    regras = RegrasCompra(orcamento=100.0, custo_bps=0.0, peso_maximo=0.29, quantidade_minima=2)

    assert acoes_que_cabem(_serie(A=14.5), regras) == ["A"]


@pytest.mark.parametrize("preco", [np.nan, 0.0, -1.0])
def test_acoes_que_cabem_tira_a_sem_preco_valido(preco):
    regras = RegrasCompra(orcamento=1_000.0, custo_bps=0.0, peso_maximo=0.5, quantidade_minima=2)

    assert acoes_que_cabem(_serie(A=10.0, B=preco), regras) == ["A"]


@pytest.mark.parametrize("campo, valor", [("orcamento", 0.0), ("custo_bps", -1.0), ("peso_maximo", 0.0),
                                          ("quantidade_minima", 0)])
def test_regras_invalidas_levantam_erro(campo, valor):
    argumentos = {"orcamento": 1_000.0, **SEM_CUSTO, campo: valor}
    with pytest.raises(ValueError):
        RegrasCompra(**argumentos)


def test_sem_cotacao_reescala_sem_furar_teto_de_setor():
    # F (14%) sem cotação. Reescalar tudo por 1/0,86 levaria Utilidade Pública de 30% a 34,9%: o peso de F vai
    # só para quem tem folga (C, D, E e G, de 14% para 17,5%), e Utilidade Pública fica nos 30%.
    pesos = pd.Series({"A": 0.15, "B": 0.15, "C": 0.14, "D": 0.14, "E": 0.14, "G": 0.14, "F": 0.14})
    grupos = (LimiteGrupo("Utilidade Pública", frozenset({"A", "B"}), 0.30),
              LimiteGrupo("Outros", frozenset({"C", "D", "E", "G", "F"}), 0.80))
    alvo = redistribuir_sem_cotacao(pesos, pesos.index != "F", peso_maximo=0.20, grupos=grupos)
    assert alvo.to_dict() == pytest.approx({"A": 0.15, "B": 0.15, "C": 0.175, "D": 0.175, "E": 0.175, "G": 0.175})
