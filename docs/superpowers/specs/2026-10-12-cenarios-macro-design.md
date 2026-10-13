# Cenários macro para o otimizador B3 — design (spec 1: subprojetos B + E)

Data: 2026-10-12. Status: **APROVADO** pelo usuário em 2026-10-13 (texto inteiro, §1–15, com
κ = 1 na §15). Partes 1 a 7 aprovadas em conversa, uma a uma, em 2026-10-12. Revisado em 2026-10-13,
com o OK do usuário, depois do Plano A: download paralelo (§3) e retrato com 8 indicadores (§10).

## 1. Objetivo

O usuário pediu um otimizador "baseado no cenário macroeconômico atual". Ele quer que o
otimizador use indicadores econômicos (inclusive o IPCA), contábeis e notícias, com
acurácia e desempenho altos.

O pedido foi decomposto em subprojetos. Este spec cobre **B (dados macro) + E (cenários)**:

- mostrar o retrato macro de hoje (Selic, Focus, IPCA, câmbio, PIB, curva de juros);
- estimar a sensibilidade de cada ação a surpresas macro (modelo de fatores);
- mostrar quanto cada carteira rende ou perde em cenários (Focus, estresses históricos,
  "meu cenário");
- permitir um piso de retorno por cenário dentro do otimizador;
- validar fora da amostra, com critérios pré-registrados, se as sensibilidades explicam algo.

**Critério de sucesso:** o usuário vê, para a carteira otimizada e para as referências, o
resultado em cada cenário. Ele pode impor um piso. O app diz, com base em teste fora da
amostra, se as sensibilidades macro merecem confiança ("validado" ou "exploratório").

### Decisões do usuário (registradas)

1. Só ações. Cripto foi abandonado. A remoção do código de cripto é o subprojeto A, fora daqui.
2. Ordem (c): primeiro cenários. O μ condicional ao macro (subprojeto D) só entra se superar
   o baseline Jorion fora da amostra.
3. Cenários (iii): automáticos (Focus ± dispersão + estresses históricos) mais "meu cenário"
   com sliders.
4. Abordagem 1: modelo de fatores macro sobre **surpresas**. Foram rejeitados os análogos
   históricos/regimes e o VAR + Monte Carlo (este talvez volte no D).
5. A interface é **Streamlit**. A versão Cloud Run foi descartada pelo usuário: custo zero,
   código único e mesmo projeto. O `core/` continua sem Streamlit, então uma API futura
   continua possível.
6. Estresse fiscal de 2024 incluído.
7. Horizonte só para exibição (12/24/36/60 meses), afetando a faixa ±1σ em R$ e a
   P(ficar abaixo do CDI). Um horizonte que mexa nos pesos fica para o D. Detalhes na §15.
8. Risco de estimação na projeção de horizonte: **κ = 1** (2026-10-13).

### Premissas (não ditas pelo usuário; assumidas e não contestadas)

- O horizonte dos cenários é de 12 meses, independentemente do horizonte de exibição (§15).
- O cenário-base Focus já está no preço. Portanto base = μ de hoje (Jorion), e os cenários
  são choques **relativos ao consenso**.
- Nenhum cenário recebe probabilidade.

## 2. Escopo

**Dentro:** fontes Focus e Tesouro; painel mensal de fatores; modelo de fatores por ação;
cenários A/B/C; piso por cenário no otimizador; script de validação; aba "Cenários macro"
no Streamlit; seletor de horizonte de exibição (§15); testes sem rede.

**Fora (cada item terá spec próprio ou fica para depois):**
- A: remover cripto e criar peso mínimo por posição.
- C: dados contábeis CVM point-in-time, para substituir o Growth Score do Yahoo.
- D: μ condicional ao macro. Critério para entrar: superar o Jorion fora da amostra no
  mesmo walk-forward. Usa a mesma maquinaria de Focus point-in-time e de betas rolantes.
  Um horizonte que mexa nos pesos também fica para o D.
- F: notícias.
- Presets Arrojado/Moderado; deploy (sugestão: Streamlit Community Cloud).
- Piso dentro do walk-forward. Ele exige Focus point-in-time e betas por janela, que são do D.
- Migrar a aba "Carteira" existente para elementos nativos (sem CSS/Plotly).

**Garantia:** com o piso desligado, nada muda no otimizador, no backtest nem nas métricas.
O horizonte de exibição nunca muda pesos.

## 3. Arquitetura

```
core/bcb.py            (existe) SGS: CDI 12, PTAX 1, Selic meta 432, IPCA 12m 13522
core/focus.py          (novo)   Focus OData: anuais e IPCA 12 meses; ErroFocus
core/tesouro.py        (novo)   CSV do Tesouro Direto -> taxas prefixadas 1a e 5a; ErroTesouro
core/market_data.py    (muda)   + carregar_precos_mensais (sem filtro de cobertura de 95%)
core/fatores_macro.py  (novo)   painel mensal de surpresas (6 fatores) + retrato macro
core/modelo_fatores.py (novo)   regressão por ação, shrinkage de Vasicek, risco residual
core/cenarios.py       (novo)   cenários Focus/estresse/meu, plausibilidade, matriz
core/validacao.py      (novo)   Clark-West, IC transversal, replay; relatório
core/config.py         (muda)   PisoCenario; ConfigOtimizacao.pisos
core/optimization.py   (muda)   restrição de piso, pré-checagem, ErroPisoInviavel, pisos_ativos
core/metrics.py        (muda)   + projetar_horizonte: faixa e P(abaixo do CDI) em H meses (§15)
ui/servicos.py         (muda)   loaders cacheados do painel, modelo e matriz
ui/secoes.py           (muda)   card da faixa segue o horizonte de exibição (§15)
ui/secao_cenarios.py   (novo)   aba "Cenários macro"
app.py                 (muda)   abas dinâmicas "Carteira" | "Cenários macro"; seletor de horizonte
scripts/validar_modelo_fatores.py (novo) grava docs/validacao/AAAA-MM-DD.{json,md}
```

Regras:
- O `core/` não importa Streamlit e não ganha dependências novas.
- `hoje` é sempre um argumento explícito.
- Cada fonte tem um erro próprio. Fonte que falta reduz os fatores e gera aviso, sem
  derrubar o resto.
- `pyarrow`, já instalado, passa a ser declarado no `requirements.txt`.
- As fontes são baixadas em paralelo (`ThreadPoolExecutor`), porque são independentes. Em série
  levavam ~160 s (o SGS responde devagar); em paralelo, ~12 s (medido em 2026-10-13). Cada fonte
  continua isolada: a que falha vira motivo em `erros`, e um erro que não é de fonte sobe.

Fluxo:

```
fontes -> fatores_macro (painel mensal) -> modelo_fatores (betas)
       -> cenarios (vetores r_s por ação) -> matriz carteira x cenário
                                          -> otimizador (pisos)
```

## 4. Fontes de dados (verificadas ao vivo em 2026-10-12)

| Fonte | Acesso | Cobertura | Uso |
|---|---|---|---|
| Focus anual | `https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais`; `$filter` com URL-encode via `quote(v, safe="")` (passar por `params` do requests dá 400) | desde 2000 | Selic, Câmbio, PIB Total: `Mediana`, `DesvioPadrao`, `Data`, `DataReferencia`, `baseCalculo=0` |
| Focus IPCA 12m | `.../odata/ExpectativasMercadoInflacao12Meses`, `Indicador='IPCA'`, `Suavizada='S'`, `baseCalculo=0` | desde 2001 | fator 2, cenários |
| Tesouro Direto | CSV `PrecoTaxaTesouroDireto.csv` (Tesouro Transparente); `sep=';'`, `decimal=','`, latin-1; 14,6 MB, cerca de 27 s | Data Base 2004-12-31 em diante | "Tesouro Prefixado" (LTN) e "Tesouro Prefixado com Juros Semestrais" (NTN-F); `Taxa Compra Manha` |
| SGS | `core/bcb.py` | — | CDI (12), PTAX venda (1), Selic meta (432), IPCA 12m (13522, retrato) |
| Yahoo | yfinance | — | preços diários das ações, BOVA11 e ^BVSP, reamostrados para o fim do mês |

Achados sobre a cobertura:
- **Ponto de 5 anos:** até cerca de 2010 só existe via NTN-F, que paga cupom. A LTN vai no
  máximo até 2,5 anos.
- **Título com prazo ≥ 4,5 anos:** só a partir de **2005-11**. Por isso o painel completo
  começa em **2005-12**, que é a primeira variação mensal.
- **Descartadas:**
  - EMBI+ no Ipeadata (`JPM366_EMBI366`): termina em 2024-07.
  - Swaps DI-pré do SGS (7805/7806, 7816–7821): terminam em 2019.

## 5. Fatores (surpresas mensais; valor de fim de mês t menos fim de t−1)

| # | Fator | Definição | Unidade |
|---|---|---|---|
| 1 | Juros | Δ taxa prefixada de 1 ano. Interpolação linear da taxa por prazo (anos) entre os títulos prefixados do último dia útil do mês. | p.p. |
| 2 | Inflação | Δ mediana Focus IPCA 12m (última pesquisa do mês) | p.p. |
| 3 | Câmbio | Δ ln PTAX venda (último dia útil do mês) | log |
| 4 | Atividade | Revisão do PIB 12m à frente, com ano de referência fixo: `w·(F_t(Y) − F_{t−1}(Y)) + (1−w)·(F_t(Y+1) − F_{t−1}(Y+1))`, onde Y = ano corrente em t | p.p. |
| 5 | Prêmio de risco | Δ (taxa 5a − taxa 1a), com a mesma interpolação do fator 1 | p.p. |
| 6 | Mercado residual | resíduo da regressão de (^BVSP − CDI) mensal nos fatores 1–5 | retorno |

Detalhes:
- **Ponderação para 12 meses à frente** (fatores 4 e cenários): `w = (12 − k)/12`, onde k é
  o número de meses completos do ano na data da pesquisa. Em outubro, k = 9, então
  w = 3/12 para o ano corrente e 9/12 para o próximo. A mesma regra vale para a mediana e
  para o σ: `σ_12m = w·σ_Y + (1−w)·σ_{Y+1}`, que é conservador porque supõe correlação 1.
- **Juros 1a:** se a taxa de 1 ano no mês estiver fora do intervalo de prazos disponível, o
  valor é NaN. Não há extrapolação.
- **Prêmio 5a:** se não houver título com prazo ≥ 4,5a no mês, o fator 5 é NaN.
- **Datação:** pela data da pesquisa (Focus) ou pela Data Base (Tesouro). Isso serve para
  explicação (E). A defasagem de publicação do Focus só importa para previsão (D).
- **Ortogonalização do fator 6:** os coeficientes γ são estimados na mesma janela do modelo
  de fatores. Na validação, γ usa dados até t−1 e é aplicado em t. Nos estresses, aplica-se
  o γ de hoje aos meses do episódio, para manter coerência com os betas de hoje.
- **^BVSP é índice de preço:** o dividend yield, aproximadamente constante, vai para o
  intercepto.

## 6. Modelo de fatores

Por ação i, com retorno mensal simples em excesso ao CDI mensal composto:

`r_i,t − CDI_t = a_i + Σ_{k=1..5} b_ik·f_k,t + c_i·f_6,t + ε_i,t`

- **Janela:** os últimos até 180 meses que terminam no mês de estimação.
- **Mínimo de 36 meses válidos.** Com menos, a ação vai para `sem_cenario`.
- **Preços mensais:** vêm de `carregar_precos_mensais` (preço diário ajustado, último do
  mês, sem o filtro de cobertura de 95%).
- **Estimação:** MQO com `lstsq`. Se o número de condição da matriz de fatores passar do
  limite, o modelo gera aviso e continua.
- **Shrinkage de Vasicek, por coeficiente (b_1..b_5 e c):**
  `b̃ = (s²·b + se²·b̄)/(s² + se²)`.
  - `se` é o erro-padrão MQO da ação.
  - `s²` é a variância transversal dos betas MQO no universo.
  - `b̄` é a média do setor (campo `setor` em `assets.py`). Se o setor tiver menos de 4
    ações com betas, usa-se a média do universo.
  - Não há hiperparâmetro. O shrinkage fica mais forte com histórico curto.
- **Universo para o shrinkage:** todas as ações do universo carregado pelo app, não só as
  da carteira. Tickers que não são da B3 (cripto, até o subprojeto A removê-los) não são
  estimados e vão para `sem_cenario`.
- **Risco residual:** σ_ε mensal da ação. Para um horizonte de L meses, σ_ε·√L. A faixa da
  carteira é `√(Σ w_i²·σ_ε,i²·L)`, supondo resíduos independentes. A faixa **não** inclui a
  incerteza dos betas, e isso aparece na legenda.
- **BOVA11** é estimado pelo mesmo modelo (histórico desde 2008-12) para a linha de
  referência da matriz. O μ de BOVA11 é a média aritmética anualizada dos retornos
  diários na mesma janela de `anos` usada pelo app (padrão de 5), sem Jorion, e aparece
  marcado.

## 7. Cenários

Cada cenário é um vetor de choques nos 6 fatores. Para cada ação, ele vira um retorno `r_s,i`:

| Tipo | `r_s,i` | Horizonte |
|---|---|---|
| Focus e "meu cenário" | `μ_i + Σ_k b̃_ik·choque_k` (μ Jorion, aritmético anual) | 12 meses |
| Estresse histórico | `Σ_k b̃_ik·choque_k`, **sem μ** (impacto do estresse, prática de mercado) | duração da janela |

A variação do próprio CDI nos cenários Focus é ignorada, como simplificação documentada.

### 7.1 Focus (recalculado a cada pesquisa)

- **Valores para 12 meses à frente:** IPCA pela série de 12 meses; Selic, PIB e câmbio pela
  ponderação da seção 5.
- **Base:** choques 0, então r = μ.
- **Otimista e pessimista:** mediana ± 2σ dos analistas.
  - Otimista: Selic, IPCA e câmbio abaixo da mediana; PIB acima.
  - Pessimista: o inverso.
  - Prêmio de risco = 0 e mercado residual = 0.
- **Choques:**
  - f1 = Selic_cen − Selic_base (1:1, simplificação documentada);
  - f2 = IPCA_cen − IPCA_base;
  - f3 = ln(Câmbio_cen/Câmbio_base);
  - f4 = PIB_cen − PIB_base.
- Esses cenários são suaves de propósito: o σ dos analistas subestima a incerteza real, e o
  app avisa isso.

### 7.2 Estresses históricos

Para cada episódio há um intervalo de busca. O código acha nele a pior queda pico→vale do
^BVSP mensal: o par (p, v), com p < v, que maximiza `1 − P_v/P_p`. A janela são os meses
p+1..v, e `choque_k = Σ f_k,t` na janela (inclusive f6 realizado).

| Episódio | Intervalo de busca |
|---|---|
| Crise global 2008 | 2008-01 a 2009-06 |
| Recessão e crise fiscal 2015–16 | 2014-09 a 2016-03 |
| Covid 2020 | 2020-01 a 2020-12 |
| Aperto monetário 2021–22 | 2021-06 a 2022-12 |
| Estresse fiscal 2024 | 2024-08 a 2025-01 |

Por padrão os estresses são só exibidos. O usuário pode ligar um piso neles.

### 7.3 "Meu cenário"

| Slider | Faixa | Início |
|---|---|---|
| Selic em 12m | 6% a 20% | Focus |
| IPCA 12m | 2% a 12% | Focus |
| Câmbio em 12m | R$ 4,00 a R$ 8,00 | Focus |
| PIB 12m | −4% a +5% | Focus |
| Prêmio de risco (Δ inclinação 5a−1a) | −2 a +4 p.p. | 0 |

O choque é o valor escolhido menos a mediana Focus (câmbio em log). O mercado residual é 0.
O core valida as faixas e levanta `ValueError` fora delas.

### 7.4 Plausibilidade

- Mede a distância de Mahalanobis do vetor de choques (fatores 1–5) em relação à
  distribuição das variações acumuladas em 12 meses (janelas móveis mensais) desde 2005-12.
- Classificação:
  - até o percentil 90 da distância histórica: **comum**;
  - acima do p90 e até o máximo histórico: **raro**;
  - acima do máximo histórico: **sem precedente**.
- Vale para o "meu cenário" e também é exibida nos cenários A e B.

## 8. Integração com o otimizador: piso por cenário

- `core/config.py` ganha `PisoCenario(nome: str, piso: float, retornos: tuple[tuple[str, float], ...])`.
  O mapa ticker→r_s é hasheável, no molde do `LimiteGrupo`.
- `ConfigOtimizacao` ganha `pisos: tuple[PisoCenario, ...] = ()`.
- **Restrição** em `_restricoes` (`core/optimization.py`): `r_s'·w ≥ piso_s·escala`. É
  homogênea, então vale para Kelly, Máx Retorno, Mín Vol e Máx Sharpe com Charnes-Cooper
  (escala = κ). Os retornos vêm do mapa, alinhados aos `tickers` recebidos (o que inclui o
  subconjunto da etapa 2 da cardinalidade).
- **Ações sem cenário:** com qualquer piso ligado, ficam com peso 0. `_checar_capacidade`
  conta só as elegíveis.
- **Pré-checagem:** para cada piso, uma programação linear calcula `max r_s'w` sujeito só às
  restrições **lineares** de `_restricoes`: soma 1, teto de peso, grupos e peso 0 para
  ações sem cenário. O teto de vol, que é uma restrição cônica adicionada à parte em
  `_com_teto_vol`, fica de fora. Assim, um piso pode passar na programação linear e ainda
  ser inviável junto com o teto de vol. Nesse caso vale o fallback existente para mínima
  variância, que respeita o piso; o teto de vol é que cede. Se o máximo ficar abaixo do
  piso, o otimizador levanta
  `ErroPisoInviavel(ErroInviavel)` com os atributos `cenario` e `maximo_viavel`. A mensagem
  crua do solver nunca chega ao usuário. O mesmo máximo limita o slider na interface.
- **Interação com o que existe:**
  - Fallback do teto de vol: o piso continua valendo e o teto de vol é que cede, como hoje.
  - Cardinalidade: se a etapa 2 ficar inviável, volta para a etapa 1 com aviso
    (comportamento atual).
  - Fronteira eficiente: herda o piso e o rótulo diz isso.
  - HRP e 1/N: não usam o piso, mas aparecem na matriz.
  - Walk-forward: sem piso neste spec.
- **Saída:** `ResultadoOtimizacao.pisos_ativos: tuple[str, ...]` lista os pisos com folga
  ≤ 1e-6. A interface mostra o custo do piso: Δμ, Δvol e ΔSharpe em relação à mesma
  configuração sem piso.
- **Alternativas rejeitadas:**
  - max-min sobre os cenários: troca o objetivo Kelly;
  - CVaR sobre os cenários: exige probabilidades.

## 9. Validação (pré-registrada)

O que se valida são as **sensibilidades**, não os cenários: um cenário é condicional e não
tem como ser testado.

Regras:
- O estimador é o mesmo do app (seção 6).
- O conjunto de fatores é fixo: fator que falhar não é removido.
- O t de cada fator é um achado do TCC, não uma entrada do modelo.
- Os limiares abaixo são definidos antes de qualquer resultado.

Amostra: o painel começa em 2005-12, então as previsões fora da amostra vão de **2008-12**
até o último mês completo (cerca de 214 meses). Para cada mês t, os betas e γ são estimados
com dados até t−1, e o modelo explica o excesso de retorno de t usando os fatores
**realizados** em t.

- **Teste A (série temporal):** modelo completo contra CAPM (só ^BVSP − CDI). Os dois geram
  o mesmo espaço e o CAPM está aninhado no completo. Usa-se o **Clark-West (2007)**, com
  ações e meses empilhados e erro-padrão agrupado por mês. **Passa se t > 1,645.**
- **Teste B (corte transversal):** é o que importa para Focus e "meu cenário", em que o f6
  é 0. Todo mês, calcula-se a correlação de Spearman entre o impacto macro previsto
  (Σ_{k≤5} b̃_ik·f_k,t) e o excesso realizado menos a média do mês. **Passa se o IC médio
  for > 0 com t de Newey-West > 2.**
- **Teste C (replay, só relatório):** os betas usam dados até o mês anterior ao intervalo de
  busca. Aplicam-se os choques realizados e compara-se previsto contra realizado nas ações
  que existiam: correlação de ordem, erro médio e uma carteira 1/N dessas ações.
  - Episódios: 2015–16, Covid, 2021–22 e fiscal 2024.
  - **2008 fica de fora:** antes dele há só 25 meses de painel, abaixo do mínimo de 36.

**Veredito:** "validado" se A **e** B passarem; senão, "exploratório".
- O script `scripts/validar_modelo_fatores.py` grava `docs/validacao/AAAA-MM-DD.{json,md}`.
- O app lê o relatório mais recente.
- Sem relatório, o app trata como exploratório.
- Exploratório marca as células Focus e "meu cenário"; o piso continua disponível, com aviso.
- Os estresses não são afetados, porque dependem principalmente do fator mercado.
- O script roda de novo quando o painel ou o universo mudar.

## 10. Interface (Streamlit; skill `developing-with-streamlit`)

- **Abas:** `app.py` usa `st.tabs(["Carteira", "Cenários macro"], on_change="rerun",
  key="abas")`, e cada aba só calcula o próprio conteúdo se `.open` for verdadeiro. A aba
  "Carteira" é o app de hoje. A única mudança nela é o card da faixa, que segue o horizonte
  de exibição (§15).
- **A aba "Cenários macro" não depende de a outra ter sido aberta antes.** Ela chama as mesmas
  funções cacheadas de `ui/servicos.py` (preços, μ/Σ, otimização) com as entradas da
  sidebar. Por isso não há passagem de carteira por `st.session_state`, e a ordem de
  visita das abas não importa.
- **Loaders**, em `ui/servicos.py`:
  - `carregar_painel_macro(dia)`: `@st.cache_data(persist="disk", max_entries=7)`. A chave
    `dia` renova o painel uma vez por dia, e o persist sobrevive a reinícios.
  - Se o teste da seção 12 mostrar que o persist não se comporta assim, o plano B é um
    parquet manual em `.cache/macro/AAAA-MM-DD/`.
  - Modelo e matriz: `@st.cache_data(max_entries=8)`. Todo argumento entra na chave.
- **Aba "Cenários macro"**, só com elementos nativos:
  1. Retrato: `st.metric` em `st.container(horizontal=True)` com 8 indicadores:
     - Selic meta;
     - Selic em 12 meses (Focus);
     - IPCA acumulado em 12 meses (SGS 13522);
     - IPCA 12 meses à frente (Focus);
     - câmbio PTAX;
     - câmbio em 12 meses (Focus), que é a base do cenário da §7.1;
     - PIB 12 meses à frente (Focus);
     - inclinação 5a − 1a.

     A data de cada indicador aparece em `st.caption`. Cada indicador tem `dias_maximos` (7 para
     fontes diárias, 75 para o IPCA mensal); se a idade do dado passar disso, aparece
     `st.warning`.
  2. Matriz carteira × cenário: mapa de calor em `st.altair_chart` com `alt=`. Linhas:
     Otimizada, Otimizada com piso (se houver), 1/N, HRP, BOVA11. Abaixo, `st.dataframe`
     com números, faixa ±1σ e etiqueta "exploratório" quando a validação falhar. Legenda:
     "cenários em 12 meses (estresses: duração da janela), independentes do horizonte de
     exibição".
  3. "Meu cenário": 5 sliders em `@st.fragment` (recalcula só a matriz) e `st.badge` com a
     plausibilidade.
  4. Pisos: `st.form` com um piso por cenário, cada slider limitado ao máximo viável. A
     reotimização só roda no envio. Depois aparecem os pisos ativos e o custo
     (Δμ, Δvol, ΔSharpe). Se o piso for inviável, `st.error` mostra a sugestão.
  5. Detalhe por ação: `st.selectbox`, betas com t e barras com a contribuição de cada
     fator no cenário escolhido.
  6. Validação: `st.caption` com o veredito e a data.
- **Convenções da skill:** sem `use_container_width`, títulos em caixa de frase, ícones
  Material.

## 11. Erros

| Onde | Falha | Comportamento |
|---|---|---|
| Fontes (`ErroBcb`, `ErroFocus`, `ErroTesouro`) | rede, 4xx/5xx, formato | fator em `fatores_indisponiveis[fator] = motivo`; o modelo roda com os restantes; `st.warning` |
| ^BVSP | indisponível | sem f6: estresses indisponíveis; Focus e "meu cenário" seguem |
| Focus sem σ | — | só o cenário base, com aviso |
| Painel com mais de 7 dias | — | usa o painel e avisa com a data |
| Ação com menos de 36 meses | — | vai para `sem_cenario`; peso 0 com piso ligado |
| Regressão mal condicionada | — | aviso; segue com `lstsq` |
| Piso inviável | — | `ErroPisoInviavel` com `cenario` e `maximo_viavel`; a carteira sem piso continua visível |
| Relatório de validação ausente | — | "exploratório" com a legenda "validação não rodada" |

Todo objeto de resultado do core carrega `avisos: tuple[str, ...]`. A interface só exibe
esses avisos; não decide nada sozinha.

## 12. Testes (TDD; nenhum teste da suíte acessa a rede)

**Dublês:** `FocusFalso` (OData), `TesouroFalso` (CSV pequeno, latin-1, `;`), `SgsFalso` (já
existe) e Yahoo falso (padrão já existente). **Dados sintéticos com resposta conhecida:**
`r = a + B·f + ε`.

1. **Fatores:**
   - Juros 1a interpolado de uma curva conhecida, sem extrapolação.
   - Ponto de 5a a partir de LTN + NTN-F; mês sem título ≥ 4,5a deixa o fator 5 NaN.
   - Pesos 12m (outubro → 3/12 e 9/12).
   - Revisão do PIB com ano de referência fixo, inclusive na virada de janeiro.
   - Datação no fim do mês.
   - f6 ortogonal aos fatores 1–5.
2. **Modelo:**
   - Recupera B dentro de uma tolerância.
   - Vasicek: erro-padrão grande leva ao setor, pequeno mantém o beta próprio; setor com
     menos de 4 ações usa a média do universo.
   - Menos de 36 meses gera `sem_cenario`.
3. **Sem look-ahead:** alterar dados posteriores ao corte não muda betas nem γ, na validação
   e no replay.
4. **Cenários:**
   - Sinais de ±2σ.
   - O estresse acha o pico-vale de uma série sintética.
   - O choque é a soma das surpresas da janela.
   - Estresse sem μ.
   - As faixas de plausibilidade conferem com uma distribuição conhecida.
   - Sliders fora da faixa levantam `ValueError`.
5. **Piso:**
   - `pisos=()` dá pesos idênticos aos de hoje, e os 48 testes atuais continuam verdes.
   - Piso que não aperta dá os mesmos pesos.
   - Piso que aperta: `r_s'w ≥ piso` e o piso aparece em `pisos_ativos`.
   - Piso inviável: o erro traz `maximo_viavel` igual ao da programação linear.
   - Máx Sharpe com Charnes-Cooper respeita o piso.
   - Etapa 2 inviável volta para a etapa 1 com aviso.
   - Ação sem cenário fica com peso 0, e a checagem de capacidade conta só as elegíveis.
6. **Validação:**
   - Clark-West e Newey-West batem com valores calculados à mão.
   - Efeito macro verdadeiro: A e B passam. Sem efeito macro: não passam.
7. **Interface** (`AppTest.from_file("app.py", default_timeout=30)`, loaders substituídos por
   dublês). A aba é aberta com `at.session_state["abas"] = "Cenários macro"` antes do
   `.run()`, mecanismo verificado no Streamlit 1.65 em 2026-10-12:
   - Abrir a aba de cenários direto, sem passar pela "Carteira", funciona.
   - A aba renderiza sem `at.exception`.
   - Mexer num slider muda a matriz e o badge.
   - Piso inviável mostra `at.error`.
   - Painel velho mostra `at.warning`.
   - Sem relatório, aparece "exploratório".
8. **Cache:** `persist="disk"` com a chave `dia` baixa o painel uma vez por dia. Se falhar,
   aplica-se o plano B da seção 10.
9. **Horizonte de exibição (§15):**
   - `projetar_horizonte` bate com os valores do exemplo da §15 (calculados à mão).
   - Com κ = 0 e H = 12, a faixa difere de `μ ± σ` em no máximo 2,5 p.p. na grade de μ de
     −20% a 40% e σ de 5% a 30%.
   - Com κ = 1, a P é maior que com κ = 0 para todo H. Com κ = 0 e m > ln(1+CDI), a P cai
     com H.
   - Com κ = 1 e H = 12, a largura da faixa é √(1 + 1/T) vezes a de κ = 0.
   - O card chama `projetar_horizonte` sem passar `kappa` (usa o padrão 1).
   - Nenhuma função de `ui/servicos.py` tem o parâmetro `horizonte` (checado com
     `inspect.signature`).
   - AppTest: mudar o horizonte de 12 para 60 muda o card e deixa a tabela de pesos idêntica.

**Fora da suíte (manual, com rede):** um script monta o painel real e confere com os valores
do Focus de 2026-10-02 (IPCA 12m 4,59; Selic 2026 13,50; 2027 12,00). O
`.backup/review-2026-10-10/apptest.py` ao vivo continua como teste de fumaça.

**Ordem do TDD:** fontes → fatores → modelo → cenários → piso → validação → interface.

## 13. Critérios de aceitação

- `.venv/Scripts/python -m pytest tests -q` passa inteiro.
- Com o piso desligado, os pesos são idênticos aos de antes, garantidos pelos testes atuais.
- Com o painel em cache, a aba abre em menos de 2 s e o slider responde em menos de 0,5 s
  (medidos no AppTest com dublês).
- O app ao vivo abre sem exceção, e o relatório de validação real foi gerado uma vez.

## 14. Limitações conhecidas (vão para o TCC)

- **μ inflado:** o μ de hoje vem de 5 anos de Selic subindo de 2% para 15% e de rerating por
  privatização. As células Focus herdam esse μ. A correção é o D.
- **Viés de sobrevivência:** o universo são as ações de hoje, então a validação é
  condicionada à sobrevivência.
- **Cenários Focus suaves:** o σ dos analistas é estreito.
- **Simplificações:** Selic → juros 1a em 1:1; CDI dos cenários ignorado; resíduos
  independentes na faixa ±1σ; incerteza dos betas fora da faixa.
- **Curva e índice:** 5a mistura LTN e NTN-F (título com cupom); ^BVSP é índice de preço;
  granularidade mensal (o vale da Covid é o fim de março).
- **2008:** sem replay (painel curto), embora o cenário de estresse exista.
- **Horizonte de exibição:** supõe retornos iid lognormais, μ e σ constantes, CDI constante e
  nenhum IR. O μ inflado pesa mais em prazos longos: o erro dele cresce com h, e o ruído
  cresce com √h.
- **Fora do escopo:** notícias (F) e dados contábeis (C).

## 15. Horizonte de exibição (parte 8; aprovada em 2026-10-13 com κ = 1)

O app não tem horizonte de investimento. "Histórico (anos)" é a janela de estimação. Com
Kelly e retornos iid, os pesos não dependem do horizonte (Samuelson 1969; Merton 1969). Por
isso o horizonte entra **só na exibição**.

- **Seletor:** `st.select_slider("Horizonte de exibição", options=[12, 24, 36, 60],
  value=12, key="horizonte")` na sidebar, com rótulos "12 meses" etc. É escolha única e
  nunca devolve vazio.
- **Fora das chaves de cache:** o valor só é lido na renderização
  (`ui/secoes.cards_otimizacao`). Nenhuma função de `ui/servicos.py` recebe o horizonte,
  então mudá-lo não reotimiza nem baixa dados de novo.
- **O que muda:**
  - o card da faixa passa a dizer "Faixa de ±1σ em H meses", em R$ de ganho;
  - a linha de baixo do mesmo card mostra a P(patrimônio no fim de H meses ficar abaixo do
    que o CDI renderia), seguida de "estimativa de modelo; não é promessa".
- **O que não muda:** pesos, backtest, métricas e cenários.

**Modelo:** função pura `projetar_horizonte` em `core/metrics.py`, sem Streamlit. Entradas:
μ e σ anuais aritméticos da carteira (`ResultadoOtimizacao.retorno_esperado` e
`.volatilidade`), CDI anual composto, H em meses, T = "Histórico (anos)" (3 a 10) e
`kappa: float = 1.0`. O app sempre usa o padrão κ = 1; κ = 0 existe só para os testes e para
a comparação no TCC.

```
h   = H / 12
v   = ln(1 + σ² / (1 + μ)²)       # lognormal com a mesma média e variância anuais
m   = ln(1 + μ) − v / 2
s_h = √( v · h · (1 + κ · h / T) )

faixa (R$)       = orçamento · (exp(m·h ± s_h) − 1)    # ~68%, em torno da mediana
P(abaixo do CDI) = Φ( h · (ln(1 + CDI) − m) / s_h )
```

- **CDI:** a taxa vigente, constante no horizonte, da mesma fonte da rf do app. O Focus prevê
  Selic em queda, então a taxa constante superestima um pouco a P. É o lado conservador.
- **Mudança no card de 12 meses:** hoje a faixa é `orçamento · (μ ± σ)`, centrada na média.
  Na grade μ de −20% a 40% e σ de 5% a 30%:
  - com κ = 0, a fórmula nova fica a no máximo 2,34 p.p. dela (calculado em 2026-10-12);
  - com κ = 1 (o que o app usa), a faixa alarga por √(1 + 1/T) e fica a no máximo 4,84 p.p.
    (T = 3), 2,66 p.p. (T = 5) ou 1,20 p.p. (T = 10) da faixa de hoje (calculado em 2026-10-13).
    No exemplo abaixo, com H = 12: +11,6% a +57,0%, contra +13,0% a +55,0% hoje.

**Decisão (2026-10-13): risco de estimação κ = 1.**

| Exemplo: μ 34%, σ 21%, CDI 13,65%, T = 5 anos | κ = 0 (iid puro) | κ = 1 (iid + erro do μ), escolhido |
|---|---|---|
| Hipótese | μ e σ conhecidos | μ estimado em T anos: Var(m̂) ≈ v/T, que pesa h² |
| P(abaixo do CDI) em 12 meses | 16,4% | 18,6% |
| P(abaixo do CDI) em 60 meses | 1,4% | 6,1% |
| Faixa em 60 meses | +187% a +476% | +149% a +565% |

Com κ = 0, a P cai rápido com o horizonte e passa uma segurança falsa, porque o μ de hoje
está inflado (§14). Com κ = 1, a P de prazo longo reflete a incerteza do μ (Barberis 2000;
Jacquier, Kane e Marcus 2005). A aproximação ignora o shrinkage de Jorion, que reduz a
variância do μ, então κ = 1 é conservador.
