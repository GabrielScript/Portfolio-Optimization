"""
mars.py - MARS (Chen, Li, Wang, AAAI-26) on the B3, self-contained for a Kaggle kernel: the trading environment with
the paper's rule overlay, the risk score, the reward, the batched ensemble of 10 safety-critic DDPG agents and the
meta-adaptive controller, the yearly walk-forward. Every choice the paper leaves open is fixed in
docs/superpowers/specs/2026-11-08-mars-design.md, section 4.

    python mars.py treinar --pacote mars_pacote.npz --braco MARS-50 --semente 42 --estados st/ --anos 2022
    python mars.py encadear --pacote mars_pacote.npz --braco MARS-50 --semente 42 --estados st/ --saida out/
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Execucao:
    """The overlay: the largest trade per name as a share of V, the position cap, the cash buffer, cost per side."""
    passo_maximo: float = 0.10
    teto: float = 0.20
    colchao: float = 0.02
    custo: float = 0.001


def executar(x: np.ndarray, caixa: float, acao: np.ndarray, listado: np.ndarray, sai_amanha: np.ndarray,
             cfg: Execucao) -> tuple[np.ndarray, float, float, np.ndarray]:
    """One session's trades by the overlay's rules, in order: names leaving tomorrow are sold whole; no short; sales
    first; a buy stops at ``teto`` of V; buys shrink pro rata to keep ``colchao`` of V in cash after costs.

    ``x`` holds the value in each name, ``acao`` the target changes in [-1, 1]. Returns the new holdings, the new cash,
    the cost paid and the executed action in the same units as ``acao``."""
    valor = caixa + float(x.sum())
    desejo = np.clip(acao, -1.0, 1.0) * cfg.passo_maximo * valor
    desejo = np.where(listado, desejo, 0.0)
    desejo = np.where(sai_amanha, -x, desejo)
    vendas = np.minimum(np.maximum(-desejo, 0.0), x)
    compras = np.where(sai_amanha, 0.0, np.maximum(desejo, 0.0))
    compras = np.minimum(compras, np.maximum(cfg.teto * valor - x, 0.0))
    caixa_apos_vendas = caixa + float(vendas.sum()) * (1.0 - cfg.custo)
    custo_das_compras = float(compras.sum()) * (1.0 + cfg.custo)
    disponivel = caixa_apos_vendas - cfg.colchao * valor
    if custo_das_compras > 0.0 and custo_das_compras > disponivel:
        compras = compras * (max(disponivel, 0.0) / custo_das_compras)
    novo = x - vendas + compras
    custo = cfg.custo * float(vendas.sum() + compras.sum())
    nova_caixa = caixa_apos_vendas - float(compras.sum()) * (1.0 + cfg.custo)
    return novo, nova_caixa, custo, (compras - vendas) / (cfg.passo_maximo * valor)


VOL_DE_REFERENCIA = 0.5       # annual volatility that scores 1
PESOS_DO_RISCO = (0.4, 0.3, 0.3)   # volatility, concentration, leverage (the paper's)
PENALIDADE_VOL, PENALIDADE_DD, JANELA_PENALIDADE = 0.5, 2.0, 30


def risco_env(x: np.ndarray, caixa: float, acao: np.ndarray, cov: np.ndarray, cfg: Execucao) -> float:
    """C_env in [0, 1] of the action before the overlay (only the no-short clip): 0.4 x annual volatility / 0.5 +
    0.3 x concentration (HHI above 1/D, scaled to [0, 1]) + 0.3 x leverage (weights above 100%), each clipped."""
    valor = caixa + float(x.sum())
    troca = np.clip(acao, -1.0, 1.0) * cfg.passo_maximo * valor
    pesos = (x + np.maximum(troca, -x)) / valor
    nomes = len(pesos)
    volatilidade = min(1.0, float(np.sqrt(max(252.0 * pesos @ cov @ pesos, 0.0))) / VOL_DE_REFERENCIA)
    concentracao = float(np.clip((pesos @ pesos - 1.0 / nomes) / (1.0 - 1.0 / nomes), 0.0, 1.0))
    alavancagem = float(np.clip(pesos.sum() - 1.0, 0.0, 1.0))
    a, b, c = PESOS_DO_RISCO
    return a * volatilidade + b * concentracao + c * alavancagem


def penalidade(valores: np.ndarray) -> float:
    """rho_t = 0.5 x standard deviation of the last 30 daily returns + 2.0 x the largest drawdown inside the last 30
    values (daily units, a positive fraction)."""
    janela = valores[-JANELA_PENALIDADE:]
    if len(janela) < 2:
        return 0.0
    retornos = janela[1:] / janela[:-1] - 1.0
    queda = float(np.max(1.0 - janela / np.maximum.accumulate(janela)))
    return PENALIDADE_VOL * float(retornos.std()) + PENALIDADE_DD * queda


# ---------------------------------------------------------------- the environment over one year's panel

import copy  # noqa: E402
import math  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402

JANELA_COV = 60


@dataclass
class Painel:
    """One test year's data: prices carried forward inside each name's life, the state features, the CDI."""
    nomes: np.ndarray
    datas: np.ndarray
    precos: np.ndarray           # (T, D), NaN outside the name's life
    caracteristicas: np.ndarray  # (T, D, K)
    cdi: np.ndarray              # (T,), the session's rate
    cortes: np.ndarray           # training, validation, test, end (exclusive)

    def __post_init__(self) -> None:
        valido = ~np.isnan(self.precos)
        sessoes = np.arange(len(self.precos))[:, None]
        self.primeira = np.where(valido.any(0), valido.argmax(0), len(self.precos))
        self.ultima = np.where(valido.any(0), len(self.precos) - 1 - valido[::-1].argmax(0), -1)
        self.listado = (sessoes >= self.primeira) & (sessoes <= self.ultima)
        self.cheio = _carregar_adiante(self.precos)
        anterior = np.vstack([self.cheio[:1], self.cheio[:-1]])
        self.retornos = np.where(self.listado & ~np.isnan(anterior), self.cheio / anterior - 1.0, 0.0)
        self.retornos = np.nan_to_num(self.retornos)
        self.retornos[0] = 0.0
        self._covs = _covariancias_moveis(self.retornos, JANELA_COV)

    def sai_amanha(self, t: int) -> np.ndarray:
        return (self.ultima == t) & (self.ultima < len(self.precos) - 1)

    def cov(self, t: int) -> np.ndarray:
        return self._covs[t]

    def estado(self, t: int, x: np.ndarray, caixa: float) -> np.ndarray:
        valor = caixa + float(x.sum())
        return np.concatenate([[caixa / valor], x / valor, self.caracteristicas[t].ravel()]).astype(np.float32)


def _covariancias_moveis(retornos: np.ndarray, janela: int) -> np.ndarray:
    """np.cov (ddof 1) of the last ``janela`` rows up to each t, from running sums; zeros with 2 rows or fewer."""
    externos = np.cumsum(np.einsum("ti,tj->tij", retornos, retornos), axis=0)
    somas = np.cumsum(retornos, axis=0)
    zeros_d = np.zeros((1, retornos.shape[1]))
    externos = np.concatenate([np.zeros((1, *externos.shape[1:])), externos])
    somas = np.concatenate([zeros_d, somas])
    fim = np.arange(1, len(retornos) + 1)
    inicio = np.maximum(fim - janela, 0)
    n = (fim - inicio)[:, None, None].astype(float)
    s1 = somas[fim] - somas[inicio]
    s2 = externos[fim] - externos[inicio]
    covs = (s2 - np.einsum("ti,tj->tij", s1, s1) / n) / np.maximum(n - 1.0, 1.0)
    covs[n[:, 0, 0] <= 2] = 0.0
    return covs


def _carregar_adiante(precos: np.ndarray) -> np.ndarray:
    """Each column's last valid price carried forward (NaN before the first)."""
    linhas = np.where(~np.isnan(precos), np.arange(len(precos))[:, None], 0)
    np.maximum.accumulate(linhas, axis=0, out=linhas)
    return precos[linhas, np.arange(precos.shape[1])]


@dataclass
class Carteira:
    """Holdings by name (value), cash, and the path of values for the reward's penalty."""
    x: np.ndarray
    caixa: float
    valores: list[float]

    @classmethod
    def em_caixa(cls, nomes: int) -> Carteira:
        return cls(np.zeros(nomes), 1.0, [1.0])

    @property
    def valor(self) -> float:
        return self.caixa + float(self.x.sum())


def passo(painel: Painel, carteira: Carteira, t: int, acao: np.ndarray, cfg: Execucao, custo_unico: bool = False
          ) -> tuple[float, np.ndarray, float]:
    """Trade at session t's close by the overlay, then move to t + 1 (prices, CDI on cash). Returns the reward, the
    executed action and the risk score of the action before the overlay. The paper's reward subtracts the cost from
    a return that already nets it; ``custo_unico`` (MARS-R) subtracts it once."""
    risco = risco_env(carteira.x, carteira.caixa, acao, painel.cov(t), cfg)
    valor = carteira.valor
    x, caixa, custo, feita = executar(carteira.x, carteira.caixa, acao, painel.listado[t], painel.sai_amanha(t), cfg)
    movimento = np.where(painel.listado[t + 1] & painel.listado[t], painel.cheio[t + 1] / painel.cheio[t], 1.0)
    carteira.x = x * np.nan_to_num(movimento, nan=1.0)
    carteira.caixa = caixa * (1.0 + float(painel.cdi[t + 1]))
    carteira.valores.append(carteira.valor)
    retorno = carteira.valor / valor - 1.0
    segunda_vez = 0.0 if custo_unico else custo / valor
    return retorno - segunda_vez - penalidade(np.asarray(carteira.valores)), feita, risco


# ---------------------------------------------------------------- the agents

OCULTAS = (256, 128, 64)
N_AGENTES = 10
TETAS = np.linspace(0.10, 0.55, N_AGENTES)
LAMBDAS = np.linspace(5.5, 1.0, N_AGENTES)     # agent 0: lowest tolerance, largest penalty


@dataclass(frozen=True)
class Treino:
    episodios: int = 30
    lr_ator: float = 1e-4
    lr_critico: float = 1e-3
    tau: float = 0.005
    gama: float = 0.99
    lote: int = 128
    inicio: int = 1_000
    ruido: float = 0.1
    mac_passos: int = 200
    mac_lote: int = 256
    mac_lr: float = 1e-4
    lambda_meta: float = 0.5
    reparado: bool = False       # MARS-R: critic on the raw action, the cost once, the controller on current critics


class Conjunto(nn.Module):
    """N independent MLPs (256-128-64, ReLU) evaluated as batched matrix products; input (N, B, entrada)."""

    def __init__(self, n: int, entrada: int, saida: int) -> None:
        super().__init__()
        tamanhos = (entrada, *OCULTAS, saida)
        self.pesos = nn.ParameterList()
        self.vieses = nn.ParameterList()
        for de, para in zip(tamanhos[:-1], tamanhos[1:]):
            limite = 1.0 / math.sqrt(de)                      # nn.Linear's default range
            self.pesos.append(nn.Parameter(torch.empty(n, de, para).uniform_(-limite, limite)))
            self.vieses.append(nn.Parameter(torch.empty(n, 1, para).uniform_(-limite, limite)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ultima = len(self.pesos) - 1
        for camada, (peso, vies) in enumerate(zip(self.pesos, self.vieses)):
            x = torch.baddbmm(vies, x, peso)
            if camada < ultima:
                x = torch.relu(x)
        return x


def _mlp(entrada: int, saida: int) -> nn.Sequential:
    camadas: list[nn.Module] = []
    for de, para in zip((entrada, *OCULTAS), OCULTAS):
        camadas += [nn.Linear(de, para), nn.ReLU()]
    return nn.Sequential(*camadas, nn.Linear(OCULTAS[-1], saida))


class Mars(nn.Module):
    """The heterogeneous ensemble (actor, critic, safety critic per agent, target copies) and the controller."""

    def __init__(self, estado: int, acoes: int) -> None:
        super().__init__()
        self.ator = Conjunto(N_AGENTES, estado, acoes)
        self.critico = Conjunto(N_AGENTES, estado + acoes, 1)
        self.seguranca = Conjunto(N_AGENTES, estado + acoes, 1)
        self.ator_alvo = copy.deepcopy(self.ator)
        self.critico_alvo = copy.deepcopy(self.critico)
        self.mac = _mlp(estado, N_AGENTES)
        for alvo in (self.ator_alvo, self.critico_alvo):
            alvo.requires_grad_(False)

    def acoes(self, s: torch.Tensor) -> torch.Tensor:
        """(B, S) -> each agent's action, (N, B, D)."""
        return torch.tanh(self.ator(s.expand(N_AGENTES, *s.shape)))

    def pesos(self, s: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.mac(s), dim=-1)


class Memoria:
    """Preallocated replay of the shared trajectory (the paper stores the same transition in every agent's buffer)."""

    def __init__(self, capacidade: int, estado: int, acoes: int, dispositivo: torch.device) -> None:
        def vazio(*forma: int) -> torch.Tensor:
            return torch.zeros(capacidade, *forma, device=dispositivo)
        self.s, self.s2 = vazio(estado), vazio(estado)
        self.a, self.a_bruta = vazio(acoes), vazio(acoes)
        self.r, self.risco = vazio(1), vazio(1)
        self.q, self.c = vazio(N_AGENTES), vazio(N_AGENTES)
        self.n = 0

    def guardar(self, **campos: torch.Tensor) -> None:
        for nome, valor in campos.items():
            getattr(self, nome)[self.n] = valor
        self.n += 1


class Aprendiz:
    """Algorithm 1: decisions, environment interaction, the agents' updates, the controller's training."""

    def __init__(self, estado: int, acoes: int, cfg: Treino, dispositivo: torch.device, capacidade: int) -> None:
        self.cfg, self.dispositivo = cfg, dispositivo
        self.rede = Mars(estado, acoes).to(dispositivo)
        self.memoria = Memoria(capacidade, estado, acoes, dispositivo)
        self.opt_ator = torch.optim.Adam(self.rede.ator.parameters(), lr=cfg.lr_ator)
        self.opt_critico = torch.optim.Adam(self.rede.critico.parameters(), lr=cfg.lr_critico)
        self.opt_seguranca = torch.optim.Adam(self.rede.seguranca.parameters(), lr=cfg.lr_critico)
        self.opt_mac = torch.optim.Adam(self.rede.mac.parameters(), lr=cfg.mac_lr)
        self.teta = torch.tensor(TETAS, dtype=torch.float32, device=dispositivo).view(-1, 1, 1)
        self.lambda_ = torch.tensor(LAMBDAS, dtype=torch.float32, device=dispositivo).view(-1, 1, 1)

    def decidir(self, s: np.ndarray, pesos_fixos: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        """The aggregated action sum_i w_i a_i, the weights, and per-agent Q and C at the agents' own actions."""
        with torch.no_grad():
            estado = torch.from_numpy(s).to(self.dispositivo).unsqueeze(0)
            acoes = self.rede.acoes(estado)                                    # (N, 1, D)
            pesos = (self.rede.pesos(estado)[0] if pesos_fixos is None
                     else torch.as_tensor(pesos_fixos, dtype=torch.float32, device=self.dispositivo))
            agregada = (pesos.view(-1, 1, 1) * acoes).sum(0)[0]
            entrada = torch.cat([estado.expand(N_AGENTES, 1, -1), acoes], dim=-1)
            q = self.rede.critico(entrada).view(-1)
            c = self.rede.seguranca(entrada).view(-1)
        return agregada.cpu().numpy(), pesos.cpu().numpy(), q, c

    def acoes_do_critico(self, indices: torch.Tensor) -> torch.Tensor:
        """The action the critic is trained on. The paper's Algorithm 1 stores the executed one, but the actor is
        improved through the critic at its raw output, which the overlay cuts by a large factor whenever cash binds:
        the gradient is then taken where the critic saw no data. MARS-R trains the critic on the raw action (the
        overlay is part of the environment), so both sides meet on the same input."""
        return self.memoria.a_bruta[indices] if self.cfg.reparado else self.memoria.a[indices]

    def valores_do_mac(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-agent Q and C of the sampled states, (batch, N). The paper trains the controller on the values stored
        when each transition was collected, most of them from critics that had barely trained; MARS-R evaluates the
        current critics at the current actors' actions."""
        m = self.memoria
        if not self.cfg.reparado:
            return m.q[indices], m.c[indices]
        with torch.no_grad():
            s = m.s[indices]
            entrada = torch.cat([s.expand(N_AGENTES, *s.shape), self.rede.acoes(s)], dim=-1)
            return self.rede.critico(entrada).squeeze(-1).T, self.rede.seguranca(entrada).squeeze(-1).T

    def atualizar(self) -> None:
        cfg, m = self.cfg, self.memoria
        indices = torch.randint(m.n, (N_AGENTES, cfg.lote), device=self.dispositivo)
        s, s2, a, a_bruta = m.s[indices], m.s2[indices], self.acoes_do_critico(indices), m.a_bruta[indices]
        r, risco = m.r[indices], m.risco[indices]
        with torch.no_grad():
            alvo = r + cfg.gama * self.rede.critico_alvo(torch.cat([s2, torch.tanh(self.rede.ator_alvo(s2))], -1))
        perda_critico = ((self.rede.critico(torch.cat([s, a], -1)) - alvo) ** 2).mean(dim=(1, 2)).sum()
        self.opt_critico.zero_grad(set_to_none=True)
        perda_critico.backward()
        self.opt_critico.step()
        perda_seguranca = ((self.rede.seguranca(torch.cat([s, a_bruta], -1)) - risco) ** 2).mean(dim=(1, 2)).sum()
        self.opt_seguranca.zero_grad(set_to_none=True)
        perda_seguranca.backward()
        self.opt_seguranca.step()
        politica = torch.tanh(self.rede.ator(s))
        entrada = torch.cat([s, politica], -1)
        penalidade_segura = self.lambda_ * torch.relu(self.rede.seguranca(entrada) - self.teta)
        perda_ator = (-self.rede.critico(entrada) + penalidade_segura).mean(dim=(1, 2)).sum()
        self.opt_ator.zero_grad(set_to_none=True)
        perda_ator.backward()
        self.opt_ator.step()
        with torch.no_grad():
            for rede, alvo in ((self.rede.ator, self.rede.ator_alvo), (self.rede.critico, self.rede.critico_alvo)):
                for p, p_alvo in zip(rede.parameters(), alvo.parameters()):
                    p_alvo.lerp_(p, cfg.tau)

    def treinar_mac(self) -> None:
        cfg, m = self.cfg, self.memoria
        for _ in range(cfg.mac_passos):
            indices = torch.randint(m.n, (min(cfg.mac_lote, m.n),), device=self.dispositivo)
            pesos = self.rede.pesos(m.s[indices])
            q, c = self.valores_do_mac(indices)
            q_medio = (pesos * q).sum(-1)
            c_medio = (pesos * c).sum(-1)
            perda = -(q_medio.mean() / (q_medio.std() + 1e-6) - cfg.lambda_meta * c_medio.mean())
            self.opt_mac.zero_grad(set_to_none=True)
            perda.backward()
            self.opt_mac.step()

    def episodio(self, painel: Painel, inicio: int, fim: int, execucao: Execucao, gerador: np.random.Generator
                 ) -> None:
        carteira = Carteira.em_caixa(len(painel.nomes))
        d = self.dispositivo
        for t in range(inicio, fim - 1):
            s = painel.estado(t, carteira.x, carteira.caixa)
            agregada, _, q, c = self.decidir(s)
            bruta = np.clip(agregada + gerador.normal(0.0, self.cfg.ruido, agregada.shape), -1.0, 1.0)
            recompensa, feita, risco = passo(painel, carteira, t, bruta, execucao, self.cfg.reparado)
            s2 = painel.estado(t + 1, carteira.x, carteira.caixa)
            self.memoria.guardar(s=torch.from_numpy(s).to(d), s2=torch.from_numpy(s2).to(d),
                                 a=torch.as_tensor(feita, dtype=torch.float32, device=d),
                                 a_bruta=torch.as_tensor(bruta, dtype=torch.float32, device=d),
                                 r=torch.tensor([recompensa], dtype=torch.float32, device=d),
                                 risco=torch.tensor([risco], dtype=torch.float32, device=d), q=q, c=c)
            if self.memoria.n >= self.cfg.inicio:
                self.atualizar()
        self.treinar_mac()


def rodar_deterministico(aprendiz: Aprendiz, painel: Painel, carteira: Carteira, inicio: int, fim: int,
                         execucao: Execucao, pesos_fixos: np.ndarray | None = None) -> np.ndarray:
    """No noise, no learning: decisions at sessions inicio .. fim - 2; returns the weights used at each one."""
    usados = []
    for t in range(inicio, fim - 1):
        agregada, pesos, _, _ = aprendiz.decidir(painel.estado(t, carteira.x, carteira.caixa), pesos_fixos)
        passo(painel, carteira, t, agregada, execucao)
        usados.append(pesos)
    return np.asarray(usados)


def sharpe(valores: list[float] | np.ndarray, cdi: np.ndarray) -> float:
    """Annualized Sharpe of the daily excess over the CDI (``cdi[k]`` is the rate earned from value k to k + 1);
    0 when the excess does not move (all in cash), so a cash-only policy cannot win the validation."""
    serie = np.asarray(valores, dtype=float)
    excesso = serie[1:] / serie[:-1] - 1.0 - np.asarray(cdi, dtype=float)
    desvio = float(excesso.std())
    return float(excesso.mean() / desvio * np.sqrt(252.0)) if desvio > 1e-9 else 0.0


# ---------------------------------------------------------------- the yearly walk-forward

BRACOS = ("MARS-50", "MARS-50+score", "MARS-50R", "MARS-50R+score")     # R: the repaired learning (MARS-R)


def reparado(braco: str) -> bool:
    return braco.startswith("MARS-50R")


def com_nota(braco: str) -> bool:
    return braco.endswith("+score")


def carregar_painel(pacote: np.lib.npyio.NpzFile, ano: int, com_nota: bool) -> Painel:
    caracteristicas = pacote[f"{ano}_caracteristicas"]
    if com_nota:
        caracteristicas = np.concatenate([caracteristicas, pacote[f"{ano}_nota"][..., None]], axis=-1)
    return Painel(pacote[f"{ano}_nomes"], pacote[f"{ano}_datas"], pacote[f"{ano}_precos"].astype(float),
                  caracteristicas.astype(np.float32), pacote[f"{ano}_cdi"].astype(float), pacote[f"{ano}_cortes"])


def treinar_ano(painel: Painel, cfg: Treino, execucao: Execucao, semente: int, dispositivo: torch.device
                ) -> tuple[Aprendiz, int, float]:
    """30 episodes over the training window; after each, a deterministic validation year from cash; the snapshot
    with the best validation Sharpe is kept. Returns the learner holding it, its episode and its Sharpe."""
    torch.manual_seed(semente)
    gerador = np.random.default_rng(semente)
    treino, validacao, teste, _ = (int(c) for c in painel.cortes)
    estado = 1 + painel.caracteristicas.shape[1] * (1 + painel.caracteristicas.shape[2])
    aprendiz = Aprendiz(estado, len(painel.nomes), cfg, dispositivo, cfg.episodios * (validacao - treino))
    melhor, episodio_melhor, sharpe_melhor = None, -1, -np.inf
    for episodio in range(cfg.episodios):
        inicio = time.perf_counter()
        aprendiz.episodio(painel, treino, validacao, execucao, gerador)
        carteira = Carteira.em_caixa(len(painel.nomes))
        rodar_deterministico(aprendiz, painel, carteira, validacao, teste, execucao)
        nota = sharpe(carteira.valores, painel.cdi[validacao + 1:teste])
        if nota > sharpe_melhor:
            melhor, episodio_melhor, sharpe_melhor = copy.deepcopy(aprendiz.rede.state_dict()), episodio, nota
        print(f"  episode {episodio}: validation Sharpe {nota:.2f}, value {carteira.valor:.3f} "
              f"({time.perf_counter() - inicio:.0f} s)", flush=True)
    aprendiz.rede.load_state_dict(melhor)
    return aprendiz, episodio_melhor, sharpe_melhor


def trocar_nomes(carteira: Carteira, antigos: np.ndarray, novos: np.ndarray, execucao: Execucao) -> Carteira:
    """Holdings carried into the new year's 50: a name that stays keeps its value, one that leaves is sold."""
    posicao = {nome: valor for nome, valor in zip(antigos, carteira.x)}
    x = np.array([posicao.pop(nome, 0.0) for nome in novos])
    vendido = sum(posicao.values())
    return Carteira(x, carteira.caixa + vendido * (1.0 - execucao.custo), carteira.valores)


def nome_do_estado(braco: str, semente: int, ano: int) -> str:
    return f"{braco.replace('+', '_')}_s{semente}_{ano}.pt"


def treinar_e_guardar(pacote: np.lib.npyio.NpzFile, braco: str, semente: int, ano: int, pasta: Path,
                      cfg: Treino, dispositivo: torch.device) -> tuple[int, float, float]:
    """One (arm, seed, year) training; the chosen snapshot's actors and controller go to ``pasta``. The years train
    independently: only the test chains them (``encadear``)."""
    inicio = time.perf_counter()
    painel = carregar_painel(pacote, ano, com_nota=com_nota(braco))
    print(f"{braco} seed {semente} {ano}: {len(painel.nomes)} names, device {dispositivo}", flush=True)
    cfg = replace(cfg, reparado=reparado(braco))
    aprendiz, episodio, nota = treinar_ano(painel, cfg, Execucao(), semente * 10_000 + ano, dispositivo)
    estado = {chave: valor.cpu() for chave, valor in aprendiz.rede.state_dict().items()
              if chave.startswith(("ator.", "mac."))}
    segundos = time.perf_counter() - inicio
    pasta.mkdir(parents=True, exist_ok=True)
    torch.save({"estado": estado, "episodio": episodio, "sharpe": nota, "segundos": segundos},
               pasta / nome_do_estado(braco, semente, ano))
    print(f"{ano}: best episode {episodio} (validation Sharpe {nota:.2f}) in {segundos:.0f} s", flush=True)
    return episodio, nota, segundos


def encadear(pacote: np.lib.npyio.NpzFile, braco: str, semente: int, pasta: Path, anos: list[int] | None = None
             ) -> dict[str, np.ndarray]:
    """The test years in a row from the saved snapshots: holdings carry over, names leaving the 50 are sold; the
    same chain with uniform weights is MARS-Static. Decisions at each session's close, from the last session of
    the year before."""
    execucao, dispositivo = Execucao(), torch.device("cpu")
    carteiras: dict[str, Carteira | None] = {"mac": None, "estatica": None}
    datas, valores, estaticos, pesos, resumo = [], [], [], [], []
    nomes_antes = None
    for ano in anos or [int(ano) for ano in pacote["anos"]]:
        painel = carregar_painel(pacote, ano, com_nota=com_nota(braco))
        guardado = torch.load(pasta / nome_do_estado(braco, semente, ano), map_location="cpu", weights_only=True)
        estado = 1 + painel.caracteristicas.shape[1] * (1 + painel.caracteristicas.shape[2])
        aprendiz = Aprendiz(estado, len(painel.nomes), Treino(), dispositivo, 1)
        aprendiz.rede.load_state_dict(guardado["estado"], strict=False)
        teste, fim = int(painel.cortes[2]), int(painel.cortes[3])
        for chave, carteira in carteiras.items():
            carteiras[chave] = (Carteira.em_caixa(len(painel.nomes)) if carteira is None
                                else trocar_nomes(carteira, nomes_antes, painel.nomes, execucao))
        if not datas:
            datas.append(painel.datas[teste - 1])
            valores.append(carteiras["mac"].valor)
            estaticos.append(carteiras["estatica"].valor)
        antes = len(carteiras["mac"].valores)
        pesos.append(rodar_deterministico(aprendiz, painel, carteiras["mac"], teste - 1, fim, execucao))
        valores += carteiras["mac"].valores[antes:]
        antes = len(carteiras["estatica"].valores)
        rodar_deterministico(aprendiz, painel, carteiras["estatica"], teste - 1, fim, execucao,
                             np.full(N_AGENTES, 1.0 / N_AGENTES))
        estaticos += carteiras["estatica"].valores[antes:]
        datas += list(painel.datas[teste:fim])
        nomes_antes = painel.nomes
        resumo.append((ano, guardado["episodio"], guardado["sharpe"], guardado["segundos"]))
    return {"datas": np.asarray(datas).astype("datetime64[D]"), "valores": np.asarray(valores),
            "valores_estaticos": np.asarray(estaticos), "pesos": np.concatenate(pesos), "resumo": np.asarray(resumo)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etapa", choices=("treinar", "encadear"))
    parser.add_argument("--pacote", type=Path, required=True)
    parser.add_argument("--braco", choices=BRACOS, required=True)
    parser.add_argument("--semente", type=int, required=True)
    parser.add_argument("--estados", type=Path, required=True)
    parser.add_argument("--anos", type=int, nargs="*")
    parser.add_argument("--saida", type=Path)
    parser.add_argument("--episodios", type=int, default=Treino.episodios, help="only for a timing spike")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    pacote = np.load(args.pacote)
    if args.etapa == "treinar":
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for ano in args.anos or [int(ano) for ano in pacote["anos"]]:
            treinar_e_guardar(pacote, args.braco, args.semente, ano, args.estados,
                              Treino(episodios=args.episodios), dispositivo)
        return 0
    resultado = encadear(pacote, args.braco, args.semente, args.estados, args.anos)
    args.saida.mkdir(parents=True, exist_ok=True)
    destino = args.saida / f"mars_{args.braco.replace('+', '_')}_s{args.semente}.npz"
    np.savez_compressed(destino, **resultado)
    print(f"wrote {destino}: final value {resultado['valores'][-1]:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
