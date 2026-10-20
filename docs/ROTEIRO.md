# Roteiro do Otimizador B3: do protótipo a uma carteira realista e competitiva

_Atualizado em 2026-10-17, noite (S1-P.a na produção: preços diários do lago da B3 no app; SMAL11 e DIVO11 no
walk-forward; rotina diária do lago agendada no Windows; flags reabertas/atrasadas e janela do FRE corrigidas).
Produção: `main` = `4564459`
(https://portfolio-optimization-b3.streamlit.app/).
Branch de trabalho: `feat/s1p-precos`._

---

## 1. Onde estamos

| Parte | Situação |
|---|---|
| Motor de otimização (Kelly, Máx. Sharpe, Mín. Variância, HRP; tetos por ação, setor e vol) | Nível profissional, testado (295 testes) |
| Regras de risco fixas (`core/risco.py`: sem alavancagem, 20% por ação e emissor, 30% por setor, vol 40%) | Na tela desde 2026-10-18: a tela só aperta; carteira fora das regras não vira plano |
| Padrão da tela | Núcleo Mín. Vol 70% + satélite Kelly λ=2 30% (regra B do S1-P.b, §4.4) |
| Estimação (Ledoit-Wolf + Jorion) | Correta; μ = média histórica (o ponto fraco) |
| Walk-forward (fora da amostra, custos, BOVA11 na mesma data) | Honesto desde `8b6551b` |
| Métricas estatísticas e financeiras (Sharpe, Sortino, PSR/DSR, CAGR, MDD, CVaR, alfa) | Corretas, já auditadas |
| Dados de preço | **Lago de dados da B3 desde `4564459`**: retorno total do COTAHIST com proventos oficiais, JCP líquido e cisões (`lago/precos_diarios.parquet`, 380 papéis desde 2005), atualizado toda noite no lago e versionado no app nos marcos. O walk-forward da tela ainda usa a lista de hoje; o teste sem viés de sobrevivência (S1-P.b) já rodou fora da tela (§4.4) |
| Growth Score (indicadores contábeis) | Fora da tela desde `55fb9b5`; código guardado em `core/growth.py`, não testado |
| Execução (custo por liquidez, IR, carteira existente) | Não modelada; paper trading registrado desde 16/10/2026 (`paper/carteiras.csv`, rotina agendada) |

**Último resultado fora da amostra** (treino de 5 anos, teste de 3 anos, 11/2023 a 10/2026, teto de 30% por setor):

Com os preços do lago (2026-10-17, padrão da tela, 11/2023 a 10/2026):

| | Otimizada (Kelly) | BOVA11 | SMAL11 | DIVO11 |
|---|---|---|---|---|
| CAGR | 20,8% | 21,1% | 9,5% | 21,6% |
| Sharpe | 0,50 | 0,50 | −0,02 | 0,56 |
| Drawdown máximo | **−12,3%** | −16,2% | −26,2% | −16,1% |

Leitura: **empate estatístico com o índice** (erro do Sharpe ≈ ±0,5 num período só), com queda menor; o DIVO11
(dividendos) foi a melhor referência no período. Vantagem de retorno **não comprovada**. Com o Yahoo, o mesmo
protocolo dava 20,0% × 21,3% e Sharpe 0,42 × 0,47. Teste das janelas (outro protocolo, 7,6 anos): ver §4.2.

**Kelly × outros objetivos, sem viés de sobrevivência** (S1-P.b, 2026-10-18, §4.4): regra A = opção 3 (o Kelly
fica como opção, não como principal) e regra B aprova o **núcleo Mín. Vol 70% + satélite Kelly 30%**, que vira o
padrão recomendado: em 10 anos, Sharpe 0,35 × 0,36 do Kelly com drawdown de −46% × −63%.

---

## 2. Critério de "pronto" (fixado antes, não muda depois de ver resultado)

O otimizador só pode receber mais dinheiro quando tiver as três coisas abaixo:

1. **Alfa líquido** de custos e de IR contra o Ibovespa (BOVA11), **fora da amostra**, com **PSR ≥ 0,95**;
2. medido sobre **dados honestos** (com deslistadas e proventos oficiais, ou seja, depois do S1-P);
3. confirmado por **6 a 12 meses de paper trading** registrado (S5).

Até lá, a regra para dinheiro real: **BOVA11 no núcleo, otimizador como satélite pequeno**, só dinheiro de 5 anos
ou mais, rebalanceio trimestral.

---

## 3. Fluxo de tarefas

```mermaid
flowchart TD
    A["Produção 4564459<br/>preços do lago no app (feito)"] --> L["Lago de dados da B3 pronto<br/>(COTAHIST + eventos B3/CVM)"]
    L --> R["Resíduos corrigidos por regra geral<br/>(72 → 58 dias com retorno ≥ ±50%; trocas de ticker travadas)"]
    R --> RT["Rotina diária do lago no Windows<br/>+ FRE por regra (feito)"]
    RT --> D["S1-P.a: retorno total do lago no app<br/>+ SMAL11 e DIVO11 (feito)"]
    D --> E["Teste das janelas, tiro 2<br/>(feito: nenhuma passou)"]
    E --> F{"Passou nos 2 tiros?"}
    F -- sim --> F1["Nova janela vira padrão"]
    F -- não --> F2["Fica 5 anos"]
    F1 --> PB["S1-P.b: universo ponto no tempo<br/>com deslistadas (teste novo)"]
    F2 --> PB
    PB --> S5["S5: paper trading 6–12 meses<br/>(começou em 16/10/2026)"]
    PB --> CAP["Contagem histórica de ações<br/>(valor de mercado e giro)"]
    CAP --> BL["Black-Litterman sem views<br/>(alvo do Jorion = mercado)"]
    BL --> S1F["S1-F: fundamentos CVM point-in-time"]
    S1F --> S2["S2: laboratório de sinais"]
    S2 --> S3["S3: Black-Litterman com views"]
    S3 --> S4["S4: execução realista + IR + carteira existente<br/>+ fração Kelly ações × CDI"]
    S4 --> OK{"Alfa líquido fora da amostra<br/>com PSR ≥ 0,95?"}
    S5 --> OK
    OK -- sim --> SAT["Satélite pode crescer"]
    OK -- não --> NUC["Fica no núcleo BOVA11"]
```

---

## 4. Checklist

### 4.1 Feito

- [x] Otimizador convexo (cvxpy): Kelly λ, Máx. Sharpe exato (Charnes-Cooper), Mín. Variância, HRP e HRP-Árvore
- [x] Ledoit-Wolf + shrinkage de Jorion; teto de vol como cone de segunda ordem; cardinalidade em 2 etapas
- [x] Walk-forward fora da amostra com custos (10 bps por lado), falhas registradas, sem troca silenciosa por 1/N
- [x] PSR e DSR (Bailey e López de Prado); alfa de Jensen com estatística t; benchmark BOVA11 (retorno total)
- [x] Correções de métricas: Sortino (downside deviation), Calmar, alfa por regressão, curtose em excesso
- [x] Reamostragem (bagging) avaliada com pré-registro: não aprovada, ficou opcional e desligada
- [x] Plano de compra: quantidade de ações com cotação do dia; sobra de caixa ≤ 0,8% de R$ 300 a R$ 50 mil
- [x] Teto por setor (padrão 30%) na carteira, no plano de compra e no walk-forward
- [x] Walk-forward treina na mesma janela da carteira de hoje; BOVA11 começa na mesma data (`8b6551b`)
- [x] Growth Score fora da tela (`55fb9b5`); Reboot e conferência ao vivo feitos
- [x] Teste das janelas (código, pré-registro e tiro 1) na `main` (`f12ed03`)
- [x] Lago de dados da B3 (projeto `b3-datalake`, fora deste repositório): COTAHIST 1986–hoje, eventos da B3 +
      flags do COTAHIST + CVM FRE, retorno total com JCP líquido, 3.948 papéis com deslistados e em recuperação
      judicial; repositório privado `GabrielScript/b3-datalake`
- [x] Erros de ajuste do lago corrigidos por regra geral, não papel por papel (§4.3); trocas de ticker travadas numa
      tabela versionada; duas rodadas seguidas do lago dão tabelas idênticas
- [x] Teste Kelly × outros objetivos: Kelly fica como opção, não como principal (§1)
- [x] Tamanho do universo ponto no tempo (pisos R$ 1 mi e R$ 5 mi, 80% e 90% dos pregões) e triagem dos resíduos do
      ajuste nesse universo (§4.3)

### 4.2 Teste pré-registrado das janelas: encerrado, fica a janela de 5 anos

- [x] `pesos_ewma` e `covariancia_ewma` (EWMA com encolhimento Ledoit-Wolf), com TDD
- [x] `estimar_momentos(..., meia_vida_cov=...)`: Σ EWMA também alimenta o Jorion
- [x] `com_janela(estrategia, pregoes)`: janelas diferentes nas mesmas datas e no mesmo universo
- [x] `estrategia_otimizada(..., meia_vida_cov=...)`
- [x] `scripts/avaliar_janelas.py` com a regra de decisão `decidir` testada (`4b6394e`)
- [~] Revisão adversarial do pré-registro: parcial (só a sonda terminou; workflow parado por custo)
- [x] Pré-registro no docstring do script, commitado **antes de rodar** (`7d5c6cb`)
- [x] Tiro 1 (dados Yahoo), 2026-10-15: saída completa em `docs/janelas-tiro1-2026-10-15.txt`
- [x] Tiro 2 (dados do lago, 2026-10-17, pregão fechado de 16/10/2026, mesmo script): saída em
      `docs/janelas-tiro2-2026-10-16.txt`. **Nenhuma desafiante passou: fica a R5.** Mede só a correção de preços nas
      sobreviventes: o viés de sobrevivência fica para o teste novo do S1-P.b

Candidatas: R5 (controle, padrão atual), R3, R2 e EWMA (Σ com meia-vida de 6 meses e μ de 5 anos). Troca só se
P[candidata > R5] ≥ 0,983 (Bonferroni), drawdown não piorar mais de 5 p.p. e giro ≤ 2× nos **dois** tiros.
Expectativa registrada antes: "inconclusivo, fica 5 anos". **Errou** (ver abaixo).

**Tiro 1** (52 ações, fora da amostra de 19/02/2019 a 15/10/2026, 1.906 pregões, 10 bps por lado):

| | R5 (controle) | R3 | R2 | EWMA | 1/N | BOVA11 |
|---|---|---|---|---|---|---|
| CAGR | 18,5% | 28,9% | 33,0% | 25,0% | 14,8% | 10,7% |
| Sharpe | 0,43 | 0,74 | 0,85 | 0,61 | 0,31 | 0,16 |
| Drawdown máximo | −45,0% | −52,3% | −52,4% | −49,4% | −48,8% | −46,9% |
| Giro anual | 177% | 207% | 232% | 205% | 63% | 13% |
| P[> R5] | — | 0,982 | 0,994 | 0,991 | 0,274 | 0,085 |

Veredito pré-registrado: **EWMA passou nos 3 critérios** (C1 0,991 ≥ 0,9833; drawdown 4,4 p.p. pior, limite 5;
giro 1,16×). R2 passou no C1 e caiu no C2 (7,4 p.p. pior). R3 caiu no C1 (0,982, por 0,001) e no C2.
Margens da EWMA são estreitas (0,6 p.p. no C2). **O app não muda**: só troca se a EWMA passar também no tiro 2.

Leitura (hipóteses, não conclusões):
- A expectativa errou porque usou o erro do Sharpe (±0,4) como régua; a decisão usa o PSR da série diferença, que
  cancela o ruído comum de estratégias muito correlacionadas e tem mais poder.
- O Sharpe cresce com a "recência" (R2 > R3 > EWMA > R5). Mecanismo plausível: o universo é a lista líquida de
  hoje, escolhida pelo tamanho no fim do período; μ de janela curta reage mais rápido aos vencedores que essa
  escolha favorece. É exatamente o que o tiro 2 (com deslistadas) separa.
- Os saltos suspeitos do Yahoo ficaram como estavam (regra do pré-registro). O diagnóstico abaixo mostra que a
  aprovação da EWMA depende deles.

**Diagnóstico dos saltos** (não muda o veredito; saída em `docs/janelas-tiro1-saltos.txt`). Mesma rodada do tiro 1
com dados até 14/10 (o Yahoo mudou o 15/10 depois do tiro): mesmos MDD e mesmo giro total, ou seja, mesmos pesos.

| Dia (salto do Yahoo) | Peso R5 | Peso R3 | Peso R2 | Peso EWMA | Ganho/perda falso EWMA − R5 |
|---|---|---|---|---|---|
| 01/03/2021 PCAR3 +112% | 3,6% | 0% | 0% | **14,4%** | **+12,1 p.p.** |
| 28/06/2021 UGPA3 +119% | 0% | 0% | 0% | 0% | 0 |
| 23/08/2023 PCAR3 −67% | 4,7% | 6,1% | 0% | 0% | +3,1 p.p. |

- **P[EWMA > R5] cai de 0,991 para 0,956 sem esses 3 dias**: abaixo de 0,9833, a EWMA não passaria no C1. A
  aprovação do tiro 1 vem, em boa parte, de um único dia com preço errado. R2 (0,994 → 0,995) e R3 (0,982 → 0,989)
  quase não mudam. Essa conta só tira o efeito direto (o dia inteiro); não desfaz o efeito pelo treino.
- Os saltos também entram pelo treino: os dois saltos de PCAR3 em 2017 (+139%, +78%) inflam o μ por 5 anos; com eles
  na janela, a EWMA põe 8–9% em PCAR3 (média geral 3,4%). Com o salto de 2021 na janela, a R2 chegou a 10,7% de
  PCAR3 (média 2,8%): se isso ajudou ou atrapalhou a R2, esta saída não mostra.
- **Mecanismo (hipótese para o tiro 2)**: a EWMA usa μ de 5 anos e Σ de meia-vida de 6 meses. O μ lembra um salto por
  5 anos; o Σ esquece a variância dele em ~1 ano. O papel parece ter retorno alto e risco baixo, e a EWMA carrega. Com
  dados corretos isso vale para saltos reais também (ex.: AXIA3 em 2017, se for real).
- Nenhum salto cai dentro da janela do drawdown máximo (todas em fev–mar/2020): o C2 não depende deles.
- Prováveis eventos societários (a conferir na sonda do S1-P): PCAR3 01/03/2021 = cisão do Assaí; PCAR3 23/08/2023 =
  cisão do Éxito; AXIA3 22/08/2017 (+49%) pode ser real (anúncio da privatização da Eletrobras). UGPA3 28/06/2021,
  PRIO3 (2015, 2016) e PCAR3 (2017) sem explicação ainda. **O S1-P precisa tratar cisões**, não só proventos e
  desdobramentos.
- O 15/10/2026 do tiro 1 provavelmente era o preço com o pregão aberto (o Yahoo reescreveu esse dia depois): no
  tiro 2, passar a data de um pregão já fechado.
- Tiro 2 com "o mesmo script": o S1-P.a deve entrar por trás de `market_data.carregar_precos`; se precisar de outra
  porta de entrada, só a `carregar()` muda. Decidir isso no desenho do S1-P.a.
- R5 contra o BOVA11 aqui (18,5% × 10,7%) não contradiz o §1 (20,0% × 21,3%): outro protocolo (7,6 anos × 3; 52
  ações × 57) e as estratégias rodam sobre sobreviventes, o BOVA11 não. Mesmo assim P[R5 > BOVA11] ≈ 0,915 < 0,95.

**Tiro 2** (50 ações, fora da amostra de 20/03/2019 a 16/10/2026, 1.888 pregões; PCAR3 e SUZB3 saem por migração
de classe sem continuidade no lago, SIMH3 entra com a história da JSL):

| | R5 (controle) | R3 | R2 | EWMA | 1/N | BOVA11 |
|---|---|---|---|---|---|---|
| CAGR | 20,4% | 28,4% | 28,3% | 19,1% | 15,3% | 10,7% |
| Sharpe | 0,49 | 0,71 | 0,70 | 0,44 | 0,32 | 0,16 |
| Drawdown máximo | −44,3% | −54,1% | −58,1% | −49,1% | −50,0% | −46,9% |
| Giro anual | 162% | 204% | 242% | 189% | 63% | 13% |
| P[> R5] | — | 0,950 | 0,918 | **0,373** | 0,217 | 0,043 |

Veredito: **a EWMA caiu no C1** (0,991 no tiro 1 → 0,373): sem o salto falso de +112% da PCAR3 em 2021, a vantagem
some, como o diagnóstico do tiro 1 previa. R3 e R2 caem no C1 e no C2 (drawdown 9,8 e 13,7 p.p. pior). **A janela
do app continua 5 anos.** O Sharpe ainda cresce com a recência (R3/R2 > R5), mas com drawdown bem pior e sem
significância; a hipótese do viés de sobrevivência (lista de hoje favorece janela curta) só o S1-P.b testa.

### 4.3 S1-P.a: preços corrigidos no app (a partir do lago da B3)

- [x] Triagem dos resíduos do ajuste no universo ponto no tempo, com deslistadas (saída completa em
      `docs/universo-residuos-2026-10-16.txt`; lista em `docs/universo-saltos-2026-10-16.csv`; script
      `scripts/universe_residuals.py` no repositório do lago). Dos 3.449 saltos sem explicação, só **24** caem no
      universo mais largo (R$ 1 mi, 80%); dos 17 + 404 eventos da CVM sem match, só **8**. Efeito máximo num índice de
      pesos iguais do universo: +11 a +27 bps/ano. Olhando o preço de cada um: 6 já estão certos (cisões do Assaí, do
      Éxito, Bradespar, Telebrás PN), 5 da CVM não mexem no preço, 2 são reais (fraude da Americanas, tick da Ambipar)
      e **~12 estão errados** com causa identificada (BBDC3 2004, PMAM4 2008, ETER3 2007, LAME3/4 2021, JSLG3 2020,
      TELB3 1998, CST 6 1995, ELET6 1997, LIPR3 1998, CESP3 1999) + 4 a conferir (TLCP4 2005, BRPR3 2023, AMER3 2024,
      RCSL3 2023)
- [x] Erros corrigidos por regra geral (2026-10-17): classe irmã (evento confirmado numa classe vale para a irmã que
      salta pelo mesmo fator), FRE da CVM com veto do preço, flag grudada na classe, troca de ticker por regra estrita
      (81 ligações, como ITAU4 → ITUB4 e BRDT3 → VBBR3; amostra de 10 com notícia: 5 confirmadas, 0 contraditas) e
      tabela de correções manuais com evidência (JSLG3 → SIMH3, LAME → AMER3, RCSL3/4, PETZ3, quebras sem fonte).
      No universo: dias com retorno total ≥ ±50% caem de 72 para 58 (os erros sumiram, nenhum dia novo); saltos sem
      explicação, de 24 para 20. Saídas em `docs/universo-*-2026-10-17.*`
- [x] Trocas de ticker travadas em `corrections/corrections.csv`: a regra só propõe ligações novas, para revisão; e
      gold reproduzível (o Polars embaralhava a ordem das linhas em três escolhas; duas rodadas agora são idênticas)
- [x] Rotina diária do lago no Windows (`b3lake daily`, tarefa "b3lake daily" no Agendador): seg–sex 21h45 com novas
      tentativas até 23h45 e às 8h; portões (troca de ticker sem revisão, arquivo truncado, |retorno| ≥ 50% em papel
      líquido sem revisão, gold reproduzível na rodada semanal); aviso no Windows só quando falha. Aceite: o pregão
      de 16/10/2026 entrou em 28 min (hoje ~10 min: só os emissores com flag nova buscam eventos)
- [x] Erros pequenos em dia de evento, por regra: flag B/G reaberta em ≤ 10 dias = mesmo aviso (salvo quebra grande no
      preço); flag atrasada ≤ 30 dias depois de evento da B3; janela do FRE de 150 dias (bancos esperam o BACEN).
      Corrigidos: ITUB3/4 2009 (×1,15 + ×1,05 → ×1,10), SBSP3 02/01/2026 (+4,75% falso), VIVT3 28/04/2025 (+5,3%),
      CGRA3 02/01/2026, CEBR3/5/6 2021; nenhum outro retorno mudou
- [~] Portão de erro pequeno: medido (31 eventos pequenos suspeitos em papel líquido, 12 só com flag), sem regra que
      separe erro de bonificação real; fica como lista de revisão
- [x] Retorno total do lago por trás de `carregar_precos` (índice do `ret_total` ancorado no último fechamento;
      retorno desconhecido = índice parado); a tela mostra até que pregão vão os preços (`4564459`)
- [x] SMAL11 e DIVO11 ao lado do BOVA11 no walk-forward (mesmo cfg, fora do N do DSR; alfa contra o BOVA11)
- [x] Tiro 2 do teste das janelas (§4.2): fica a R5

### 4.4 S1-P.b: universo sem viés de sobrevivência (teste novo)

- [ ] IPCA (SGS 433) para o piso de liquidez: **R$ 1 mi/dia corrigido pelo IPCA** (decidido); o lago ainda não tem IPCA
      (a medição de 2026-10-16 baixou a SGS 433 pelo `core/bcb.py`)
- Tamanho medido (emissores, regra D1, reais de 09/2026): R$ 1 mi e 90% → mínimo de 36 (12/1995), ≥ 50 desde 2000,
  ≥ 119 desde 2010, **173 hoje**; R$ 5 mi e 90% → menos de 30 em 56 meses (todos de 1995 a 2000), ≥ 50 desde 2005,
  **137 hoje**. 80% × 90% muda de 1 a 8 papéis. Já passaram pelo universo de R$ 1 mi 570 papéis, 57% hoje deslistados
- [x] Universo ponto no tempo: em cada rebalanceio, só as ações líquidas naquela data (critério fixado no pré-registro)
- [x] Walk-forward com universo dinâmico: a deslistada sai no último fechamento
- [x] Pré-registro próprio contra o BOVA11; refazer o Kelly × objetivos e o teste núcleo + satélite sobre esses dados
      (congelado em `d60fa21`, rodado em 2026-10-18: `docs/universo-dinamico-2026-10-18.txt`)
- [x] Recalcular e publicar os números novos (caíram: o viés sumiu)

**Resultado (2026-10-18, lago `d47f2fa`, 132–140 papéis treinados por rebalanceio, sem teto de setor).** O BCB recusou
o CDI e a Selic na hora ("Requisição inválida"): taxa livre de risco = Selic constante de 13,75%, como o pré-registro
manda. Os pesos do Kelly, da Mín. Vol, do HRP e do 1/N não dependem da taxa (só os do Máx. Sharpe), mas o Sharpe
de cada um sobe Δ/σ com uma taxa Δ menor, mais para quem tem vol menor. **Regra B: robusta** (a mistura só ganha
folga). **Regra A: a ~2,9 p.p. de taxa da opção 4** (em 10 anos a diferença de Sharpe Kelly − Mín. Vol é 0,083 e
0,083 / (1/15,9% − 1/29,0%) = 2,9 p.p.); a Selic constante dá 12,9% a.a. na convenção aritmética, e o CDI médio de
10/2016 a 10/2026 ficou abaixo disso. O veredito pré-registrado (opção 3, pelo fallback) vale; a rodada com o CDI,
quando o BCB voltar, sai como robustez, não como decisão. Na prática a tela muda igual: o padrão vem da regra B.

| | 5 anos: Sharpe | 5 anos: MDD | 10 anos: Sharpe | 10 anos: MDD | 10 anos: CAGR |
|---|---|---|---|---|---|
| Kelly λ=2 (padrão da tela) | −0,31 | −38,8% | 0,36 | −63,1% | 21,1% |
| Mín. Vol | 0,04 | −13,7% | 0,28 | −39,4% | 17,4% |
| **Núcleo Mín. Vol 70% + satélite Kelly 30%** | −0,13 | −16,4% | 0,35 | −46,3% | 19,1% |
| BOVA11 | 0,06 | −21,0% | 0,09 | −46,9% | 13,1% |
| DIVO11 | 0,23 | −16,1% | 0,24 | −40,1% | 16,9% |

- **Regra A: opção 3.** O Kelly não é o objetivo principal: perde Sharpe para a Mín. Vol em 5 anos (−0,31 × 0,04).
- **Regra B: Núcleo Mín. Vol 70/30 aprovada** nos dois horizontes (Sharpe dentro de 0,05 do Kelly em 10 anos, MDD
  16,8 p.p. melhor em 10 anos e 22,4 p.p. em 5): vira o padrão recomendado da tela.
- Nenhuma estratégia tem alfa comprovado contra o BOVA11: em 5 anos todas as ativas perdem dele em Sharpe, menos a
  Mín. Vol (empate); em 10 anos P[Kelly > BOVA11] = 0,915, abaixo do PSR ≥ 0,95 do critério de pronto.

### 4.5 Black-Litterman sem views (aguarda decisão)

- [ ] Contagem histórica de ações (CVM FRE) para valor de mercado e giro; chave pronta: CNPJ → código CVM → emissor da B3
- [ ] Alvo do Jorion = retorno de equilíbrio de mercado Π = δΣw_cap (w_cap = nº de ações × preço, em cada data)
- [ ] Avaliar com protocolo pré-registrado contra o padrão atual

### 4.6 S1-F: fundamentos point-in-time

- [x] Demonstrações da CVM (DFP + ITR desde 2010) com **data de entrega** (sem look-ahead), fluxos em 12 meses e
      valor de mercado com a contagem de ações da composição do capital (2020+) e do FRE (antes): lago `b3-datalake`
      `72f06c1`, comando `fundamentals`; cópia fixada em `docs/dados/2026-10-20-cvm/`
- [x] Permite pôr indicadores contábeis no walk-forward pela primeira vez (S2.2 e S2.3, §4.7)

### 4.7 S2: laboratório de sinais (onde pode nascer o alfa)

- [ ] Sinais candidatos com evidência na literatura: **valor** (P/L, P/VP, EV/EBITDA), **qualidade** (ROE, ROIC,
      margens, dívida líquida/EBITDA), **dividend yield**, **momentum 12−1** (pulando o último mês), **baixa vol**
- [ ] Bancos e seguradoras com métricas próprias (não usar "receita" e "FCF" de empresa não financeira)
- [ ] 10+ anos, pré-registro, custos; conferir contra os fatores NEFIN
- [ ] Reescrever ou aposentar o Growth Score atual (P&D quase constante na B3, "TAM" sem evidência)
- [x] **S2.1, Conservadora de Blitz e van Vliet (2018)** (`scripts/avaliar_conservadora.py`, pré-registrado; saída
      em `docs/conservadora-2026-10-20.txt`): baixa vol de 36 meses + dividend yield de 12 meses + momentum 12−1, 15
      nomes com pesos iguais, 1 por emissor, até 4 por setor. **Aprovada e candidata à prova final.**

| 2010–2026, 3 fases (contra o BOVA11) | CAGR | MDD | IR | P aritmético (log) | DSR (N = 22) | MinTRL |
|---|---|---|---|---|---|---|
| Conservadora com regras | 16,3–17,3% | −40 a −42% | 0,58–0,63 | 0,99 (0,996–0,998) | 0,80–0,88 | 7–8 anos |
| Kelly λ=2 com regras (o satélite) | 13,0–14,3% | −62 a −63% | 0,32–0,37 | 0,90–0,93 | 0,35–0,64 | 20–27 anos |
| BOVA11 | 6,7–7,2% | −50% | — | — | — | — |

- IC mensal da nota: 0,081 (t 5,9) de 2008 a 2026 e 0,059 (t 3,1) de 1998 a 2007, acima da expectativa registrada
  (0,01–0,04); 2015–2019 é o trecho fraco (0,039, t 1,7). P[Conservadora > Kelly] 0,51–0,69; giro ~300% a.a.; 0
  recusas.
- Ainda não cumpre o item 1 do critério: DSR abaixo de 0,95, sem IR no teste e sem paper trading. Entra nas views do
  S3 e na prova final.
- Contas feitas depois da rodada (fora do pré-registro, não mudam a decisão): 1/N do universo elegível no mesmo
  motor fez 8,95% (P 0,68), então o excesso não vem só de pesos iguais contra o Ibovespa; com 30 bps por lado, 16,6%
  (P 0,988); cada componente sozinho tem IC de 0,06 a 0,07 (t 4,8–6,7), positivo também em 1998–2007. Setor: 97% das
  deslistadas têm setor, contra 99% das vivas (9 papéis sem setor desde 2009).

- [x] **S2.2 e S2.3, Fórmula Mágica de Greenblatt e Trending Value de O'Shaughnessy** (`scripts/avaliar_valor.py`,
      pré-registrado; saída em `docs/valor-2026-10-20.txt`): EBIT/EV + EBIT/capital empregado; 6 rendimentos de
      valor, os 30% mais baratos e o momentum de 6 meses dentro deles; 15 nomes com pesos iguais, sem Financeiro.
      **As duas aprovadas como sinal; nenhuma candidata à prova final.**

| 2012–2026, 3 fases (contra o BOVA11) | CAGR | MDD | IR | P aritmético (log) | DSR (N = 25) | P[> Conservadora] |
|---|---|---|---|---|---|---|
| Fórmula Mágica com regras | 10,2–12,1% | −53 a −55% | 0,06–0,27 | 0,59–0,85 (0,60–0,85) | 0,14–0,46 | 0,05–0,17 |
| Trending Value com regras | 11,1–13,5% | −52 a −57% | 0,17–0,27 | 0,75–0,85 (0,75–0,86) | 0,18–0,40 | 0,04–0,25 |
| Conservadora com regras (mesmas datas) | 16,7–18,5% | −41% | 0,47–0,63 | 0,96–0,99 (0,98–0,996) | 0,68–0,82 | — |
| Kelly λ=2 com regras (o satélite) | 12,2–14,4% | −61 a −63% | 0,18–0,31 | 0,75–0,88 | 0,26–0,40 | — |
| BOVA11 | 8,0–9,2% | −47% | — | — | — | — |

- IC mensal de 0,059 nas duas (t 5,4 na Fórmula Mágica e 5,5 no Trending Value), positivo nos três subperíodos
  (o mais fraco: Fórmula Mágica em 2017–2021, 0,037, t 1,9), acima da expectativa registrada (0,00–0,04; a
  expectativa de que uma das duas reprovasse errou). Com ~120 papéis com nota por mês, o sinal ordena bem o
  universo, mas a carteira de 15 nomes a cada 63 pregões captura menos que a Conservadora: MDD maior, giro de
  260–400% a.a. e P[> Kelly] 0,23–0,52. 0 recusas e 0 falhas.
- Entram como views do S3 (com a Conservadora) e nas provas seguintes como sinal, não como carteira própria.
- Conta feita depois da rodada (fora do pré-registro, não muda a decisão): o IC não vem da reversão de curto prazo
  pelo preço de m-1. Com o valor de mercado do fechamento de m-2, 0,061 (Fórmula Mágica) e 0,060 (Trending Value);
  o retorno de m-1 com sinal trocado tem IC de −0,033 aqui (a B3 tem continuação, não reversão, de 1 mês).

### 4.8 S3: Black-Litterman com views

- [ ] Views dos sinais aprovados no S2, confiança Ω pelo IC histórico de cada sinal
- [ ] Opcional: Plano B macro (views de setor por juros, câmbio e commodities)

### 4.9 S4: execução realista

- [ ] Custo por liquidez (proporcional a √(ordem / volume)), não 10 bps fixos
- [ ] Zona de não negociação (só rebalancear quando o desvio compensa o custo)
- [ ] IR: 15% sobre ganho; isenção de vendas até R$ 20 mil/mês; JCP e dividendos
- [ ] Partir da **carteira que já existe** (posições atuais) e rebalancear a partir dela
- [ ] **Fração Kelly ações × CDI**: quanto do dinheiro vai para ações, considerando a incerteza do prêmio de risco
      (hoje o app fica sempre 100% em ações)

### 4.10 S5: paper trading

- [x] Decidir quando o relógio começa (§7): depois do S1-P.b; 1º registro no fechamento de 16/10/2026
- [x] Carteira registrada a cada rebalanceio, com data e preço, sem ajuste retroativo (`paper/carteiras.csv`, rotina
      `scripts/paper_trading.py` agendada de segunda a sexta às 8h30)
- [ ] 6 a 12 meses; comparar com BOVA11 e CDI pelo critério da seção 2 (leituras fixas de 126 e 252 pregões)

### 4.11 Robustez fora da amostra (PLAN §11, passo 3): descritivo, o critério não muda

Pré-registro e script em `scripts/avaliar_robustez.py`, saída em `docs/robustez-2026-10-19.txt`: universo ponto no
tempo com as deslistadas, regras de risco no motor, 2010 a 16/10/2026 (16,7 anos), 3 fases de rebalanceio.

| Curva (contra o BOVA11) | IR | P aritmético | DSR (N = 21) | MinTRL | 60 meses acima do BOVA11 |
|---|---|---|---|---|---|
| Satélite Kelly λ=2 com regras, sem stop | 0,32–0,37 | 0,90–0,93 | 0,41–0,70 | 20–27 anos | 72–76% |
| Dinheiro real (BOVA11 70% + satélite com stop 30%) | 0,04–0,18 | 0,56–0,75 | 0,08–0,41 | 100+ anos | 47–54% |
| Núcleo Mín. Vol 70/30 com regras (estudo) | 0,42–0,44 | 0,95–0,96 | 0,56–0,76 | 14–16 anos | 96% |

- **O Núcleo Mín. Vol 70/30 com P ≥ 0,95 não cumpre o item 1 do critério.** As 3 fases são a mesma história deslocada
  em semanas; o P vem de 2010–2014 (0,97–0,98), e de 2015 em diante não é significativo (0,55–0,79); o DSR fica
  abaixo de 0,95 com um N que é piso; o IR sobre o ganho de capital (giro de 110–145% ao ano) não está no teste; e não
  é a estratégia do paper. O teste do critério continua o do S1-P.b mais o paper trading.
- **O stop relativo custa quase todo o excesso do satélite.** Sem stop, a mistura 70/30 teria o IR do satélite
  (0,32–0,37); com ele, 0,04–0,18. Numa conta feita depois da rodada (fora do pré-registro), o stop dispara no período
  inteiro entre 2011 e 2014 e em 59–100% das janelas de 5 anos, com mediana de 2 a 3 anos. No paper, depois do stop o
  satélite vira BOVA11 até o fim, e o paper deixa de medir o otimizador. O paper-v1 não muda; rever o stop pede um
  pré-registro novo (§7).
- **Janela de 10 anos (R10) não passa**: P[R10 > R5] de 0,20 a 0,48 (o C1 pede 0,95). Rende menos, mas com drawdown
  menor (−43% contra −58% em 10 anos) e giro ~30% menor.
- **IVVB11** ficou acima do BOVA11 em 94,5% das janelas de 5 anos desde 2014, num regime só (o real foi de ~2,2 a
  mais de 5 por dólar). Motiva o núcleo BOVA11 + IVVB11 (PLAN §11), que pede pré-registro próprio.

### 4.12 Núcleo BOVA11 + IVVB11 (PLAN §11, alavanca 1): não candidato

Pré-registro e script em `scripts/avaliar_nucleo.py`, saída em `docs/nucleo-2026-10-20.txt`: 50% BOVA11 + 50%
IVVB11 com pesos fixos, de volta ao alvo a cada 63 pregões, de 29/04/2014 (1ª cotação do IVVB11) a 16/10/2026, 3
fases.

| 2014–2026, 3 fases | CAGR | Vol | MDD | P[> BOVA11] em log | DSR (N = 23) | MinTRL |
|---|---|---|---|---|---|---|
| Núcleo BOVA11 50% + IVVB11 50% | 17,6–18,0% | 15,7% | −33 a −34% | 0,87–0,89 | 0,77–0,79 | 47–55 anos |
| IVVB11 sozinho | 20,9–21,1% | 19,7% | −31% | 0,82 | 0,77–0,79 | 51–53 anos |
| BOVA11 | 11,9–12,0% | 23,8% | −47% | — | — | — |

- **Não candidato**: a condição 1 (P em log ≥ 0,95 nas 3 fases) não passa; a 2 passa (excesso de CAGR positivo nos 3
  subperíodos, 2023–2026 por pouco: 0,8–1,9% a.a.). O núcleo do dinheiro real segue BOVA11.
- Diversifica: vol de 15,7% contra 23,8% e MDD de −34% contra −47% do BOVA11; correlação diária 0,05 e mensal −0,20;
  acima do BOVA11 em 99% das janelas de 5 anos e 83–90% das de 3. Tudo num regime de câmbio só, e o teste não é cego
  (o IVVB11 entrou na lista depois de ganhar do BOVA11 desde 2014).

### 4.13 Alvo de vol com EGARCH (PLAN §11, depois do S2): reprovado nas quatro

Pré-registro e script em `scripts/avaliar_vol_alvo.py` (módulo `core/vol_alvo.py`), saída em
`docs/vol-alvo-2026-10-20.txt`: em cada estratégia do S2.2/S2.3, um EGARCH(1,1) com assimetria e t de Student nos 504
retornos diários da curva prevê a vol dos 21 pregões seguintes; a exposição vira vol realizada da janela / vol
prevista (até 100%), o resto fica no CDI, reajuste a cada 21 pregões, de 2014 a 16/10/2026, 3 fases. Regra da R10:
P[com alvo > original] ≥ 0,95, MDD até 5 p.p. pior e giro até 2x, nas 3 fases.

| 2014–2026, 3 fases | P[com alvo > original] | MDD com alvo (original) | CAGR com alvo (original) | Exposição média |
|---|---|---|---|---|
| Conservadora | 0,31–0,59 | −32 a −33% (−41%) | 18,6–20,6% (18,1–20,7%) | 95% |
| Trending Value | 0,65–0,89 | −39 a −41% (−49 a −52%) | 17,4–18,7% (15,9–16,4%) | 91–96% |
| Fórmula Mágica | 0,56–0,66 | −41 a −45% (−53 a −55%) | 12,8–17,3% (11,6–16,2%) | 93–95% |
| Kelly λ=2 com regras | 0,14–0,52 | −52 a −56% (−61 a −63%) | 14,3–19,5% (14,9–18,5%) | 94–98% |
| BOVA11 | — | −47% | 12,0–12,9% | — |

- **Reprovado nas quatro** (C1 não passa em nenhuma fase); ficam as originais. O alvo corta 7 a 13 p.p. do MDD com
  pouca troca de retorno, mas não melhora a média de forma provável.
- O ajuste falhou (sem convergência ou previsão degenerada, mantendo a exposição) em 5 a 13% dos reajustes. A
  exposição média ficou acima da expectativa registrada (0,8–0,9): a vol de 2 anos já carrega as crises.
- No trecho de 2014 a 2026, a própria Conservadora tem P[> BOVA11] de 0,89 a 0,96 (0,96–0,99 desde 2012).

---

## 5. Realista × competitivo

| Tema | Hoje | Realista (mínimo para usar) | Competitivo (padrão de gestora quant) |
|---|---|---|---|
| Dados de preço | Yahoo, só sobreviventes (lago da B3 pronto, não ligado) | Lago da B3: proventos oficiais + deslistadas (S1-P.a/.b) | Base point-in-time auditada, eventos corporativos conferidos |
| Retorno esperado (μ) | Média histórica | Shrinkage para o mercado (BL sem views) | Sinais validados fora da amostra (S2/S3) |
| Covariância (Σ) | Ledoit-Wolf | LW ou EWMA, escolhida por teste | Modelo de fatores de risco (setor, estilo, macro) |
| Indicadores contábeis | Growth Score fora da tela | Desligado até validar | Fatores de valor e qualidade point-in-time (S1-F/S2) |
| Custos | 10 bps fixos | Custo por liquidez + zona de não negociação | Modelo de impacto calibrado com execuções reais |
| Impostos | Ignorados | IR 15% + isenção de R$ 20 mil/mês | Rebalanceio que otimiza o IR |
| Validação | WF de um caminho, PSR/DSR | Pré-registro + dados honestos + 2 tiros | Vários períodos, CPCV, paper trading, auditoria |
| Tamanho da aposta | 100% em ações | Núcleo BOVA11 + satélite | Fração Kelly com incerteza do prêmio |

---

## 6. Riscos e pendências técnicas abertos

- BCB SGS 12 (CDI) falhou em 2026-10-15: o app cai na Selic constante com aviso (Sharpe aproximado).
- A linha do `app.py` que passa os grupos ao plano de compra não tem teste de app próprio (coberta no core).
- O "retorno esperado" na tela é dentro da amostra (~29%) e otimista: rotular como estimativa, não previsão.
- t do alfa sem correção para autocorrelação (Newey-West); VaR paramétrico ao lado de CVaR histórico.
- O 1º carregamento faz dois downloads do Yahoo (janela e janela + teste).
- Ação sem cotação: a reescala dos pesos pode levar um setor acima do teto no plano.
- Cosmético: a legenda do walk-forward diz "ano(s)".
- Limite de N ações junto com o teto por setor pode cair no caminho alternativo (fallback) da otimização.
- Plano do dinheiro real (revisão de 19/10/2026): o teto por emissor não entra no preenchimento guloso (o plano que
  fura o emissor por arredondamento vai inteiro para o caixa; um `LimiteGrupo` por emissor em `RegrasCompra.grupos`
  resolveria nos dois planos); a conta "papéis que cabem no teto" e a descrição das violações estão repetidas em
  `app.py` e `core/ordens.py`; o plano não avisa quando o rebalanceio trimestral do paper está vencido; o app na
  nuvem lê o `paper/carteiras.csv` da `origin/main`, e o commit do registro é só local (o push fica com o usuário).
- `tests/test_ordens.py` importa constantes de `tests/test_paper_trading.py`, que lê a fixture do paper na importação.
- A linha "BOVA11 70% + Kelly λ=2 30% (dinheiro real)" da tabela sem viés da tela é a mistura sem regras e sem stop do
  S1-P.b; o teste de robustez mostra que o stop muda esse quadro. Trocar a linha pede uma tabela nova pré-registrada.
- Lago da B3: triagem do FRE no universo e portão de erro pequeno pendentes (§4.3); direitos de subscrição fora do
  retorno total (AMER3 2024); ligações por regra sem fonte aberta (5 de 10 na amostra); sem IPCA; sem rotina
  agendada (um pregão atrás); BDRs e ETFs só com preço cru (fora do universo e das auditorias); debêntures quase
  não aparecem no COTAHIST.
- Depois de todo push na `main`: Reboot do app no Streamlit Cloud.

## 7. Decisões que dependem do usuário

1. Rodar o teste pré-registrado núcleo + satélite (núcleo Mín. Variância ou BOVA11, satélite Kelly)?
2. Quando começa o relógio do paper trading: logo depois do S1-P.a ou depois do S1-P.b (recomendado: S1-P.b, para
   contar o tempo já na estratégia que vai ser usada)?
3. Quais exports do lago entram no repositório (o painel mensal tem 17,8 MB)? Decidir no desenho do S1-P.a.
4. Black-Litterman sem views: avaliar como alvo do Jorion?
5. Mostrar no app a fração Kelly ações × CDI (S4)?
6. Debêntures: ficam fora do escopo deste otimizador de ações (só entram se virarem um produto novo)?
7. Piso de liquidez do universo: manter o decidido em 2026-10-14 (**R$ 1 mi/dia e 90% dos pregões**) ou trocar pelo
   R$ 5 mi da arquitetura v1? A medição (§4.4) mostra que 80% × 90% não muda quase nada e que R$ 5 mi deixa o universo
   com menos de 30 empresas de 1995 a 2000 (recomendado: manter R$ 1 mi; o tamanho da ordem frente ao volume entra
   no S4 como teto por posição, que depende do capital).
8. "Comparar todas as estratégias possíveis": cada estratégia comparada entra no N do DSR (mais tentativas, mais chance
   de a melhor ganhar por sorte). Recomendado: lista fechada no pré-registro do S1-P.b.
9. Stop relativo do satélite (25% contra o BOVA11, até o fim do horizonte): no teste de robustez (§4.11) ele dispara
   em 2 a 3 anos na maioria das janelas de 5 anos e leva quase todo o excesso do satélite; no paper, depois dele o
   otimizador deixa de ser medido. Manter, ou fazer um pré-registro novo (paper-v2) com outra regra, como reentrada
   depois de N meses ou sem stop e só com o teto do satélite? Recomendado: o paper-v1 segue como está até a leitura
   de 126 pregões; a regra nova, se houver, entra num pré-registro próprio.

Decidido em 2026-10-17: terminar a B3 primeiro; depois, alocação entre classes (ações B3 × ETFs e BDRs de exterior ×
ETFs de cripto × CDI) com dados que já estão no lago; dados crus dos EUA e de cripto só se essa camada mostrar que vale.
