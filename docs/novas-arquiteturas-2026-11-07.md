# New architectures and quantum papers for the aggressive B3 portfolio (2026-11-07)

Research only. Nothing in the app, the engine or `docs/tentativas.csv` changed; N unchanged. Asked by the user: do any
of eight papers (new neural-net architectures, quantum portfolio optimization) help the aggressive investor on the
B3, and which Python implementations exist.

## The bar every idea has to clear

The project's limit is the expected-return signal (mu), not the optimizer: QPO-exact, the classical exact optimum of
the same QUBO, is the ceiling of any quantum solver and already fails (DECISOES 174, 180); the alpha t-stat is 1.25
(153). In the hybrid QPO reading (213) a neural net already lost to a linear model on the same data: rank IC Ridge
0.127, Composite score 0.114, MLP 0.109, VQC 0.104, Fourier 0.077. With about 143 eligible names a month (a few
tens of thousands of stock-months), extra capacity tends to fit noise. So an architecture matters only where it can
plausibly raise the rank IC of mu or improve the risk forecast, and every arm is pre-registered and counted in N.

## The eight papers

| # | Paper | Domain | Verdict for us |
|---|---|---|---|
| 1 | Wang et al. 2026, State space models as a transformer alternative: a survey (Visual Intelligence 4:29) | SSM / Mamba survey, mostly vision | Indirect. Lists MambaStock (single-stock price prediction, no license). Possible use: a volatility/regime forecast on the risk side, not stock picking. |
| 2 | Young et al. 2026, Hybrid real and complex-valued NN (J. Audio Speech Music Proc.) | Audio signal processing | Does not apply. Complex values suit phase data; our nearest analogue, Fourier features, was the weakest arm in 213 (IC 0.077). |
| 3 | Gong et al. 2026, Quantum computing for financial transformation, a review (arXiv 2604.08180) | Quantum finance review | Confirms our record. Its own synthesis: quantum optimisation "should be viewed as an exploratory and prototyping discipline. Existing results validate feasibility but not superiority." Its QAOA case study uses six US stocks, 2024-2025, and claims no advantage. |
| 4 | Morapakula et al. 2025, End-to-end portfolio optimization with hybrid quantum annealing (Adv. Quantum Tech.) | D-Wave CQM selection + convex weights + quarterly rebalance | Already tested and closed. It is our QPO pipeline (qubit per asset, cardinality, classical weights). Evidence: one Indian bank portfolio, 01/2023 to 02/2024, against NIFTY indexes, no walk-forward out-of-sample, no cost model; the authors "refrain from drawing conclusions" on quantum advantage. |
| 5 | Wilcox et al. 2026, Network architecture of general intelligence in the human connectome (Nat. Commun.) | Neuroscience | Not a portfolio paper. One transferable idea only (network position, below). |
| 6 | Zhu et al., Liquid neural networks for telecom from first principles | LTC / CfC tutorial, telecom | Indirect. Continuous-time cells handle irregularly spaced events; fits the CVM filing stream behind the Sentinela better than daily prices. |
| 7 | Liu et al. 2024/25, KAN: Kolmogorov-Arnold Networks (arXiv 2404.19756v5) | Learnable univariate edge functions | The best fit. A one-layer KAN is an additive model that nests both Ridge and the Composite score (proved below), so a test against them is clean. The paper itself: gains are shown "on small-scale AI + Science tasks" and KANs train about 10x slower than MLPs. |
| 8 | Chen, Liu, Lyu 2026, OptST: semantic tokenization for LLM recommendation (arXiv 2611.07523) | Recommender systems | Does not apply directly. One idea transfers: learned semantic IDs as data-driven peer groups (below). |

## Implementations checked (GitHub REST API, 2026-11-07)

| Repository | Stars | License | Last push | Note |
|---|---|---|---|---|
| KindXiaoming/pykan | 16,340 | MIT | 2025-01-19 | Reference KAN, symbolic extraction |
| Blealtan/efficient-kan | 4,662 | MIT | 2024-08-01 | Pure PyTorch, CPU |
| ZiyaoLi/fast-kan | 498 | Apache-2.0 | 2024-06-20 | RBF variant, faster |
| mlech26l/ncps | 2,358 | Apache-2.0 | 2024-08-14 | LTC, CfC, NCP wirings (PyTorch/TF) |
| raminmh/liquid_time_constant_networks | 1,870 | Apache-2.0 | 2024-06-03 | Original LTC code |
| state-spaces/mamba | 18,979 | Apache-2.0 | 2026-07-22 | Needs CUDA kernels: not usable on this laptop (Vega 8, no CUDA) |
| alxndrTL/mamba.py | 1,484 | MIT | 2026-05-03 | Pure PyTorch Mamba, runs on CPU |
| zshicode/MambaStock | 191 | none | 2024-03-05 | No license: read only, do not copy |
| dwave-examples/portfolio-optimization | 59 | Apache-2.0 | 2026-09-05 | D-Wave formulations; same design as our QPO |

Firecrawl's GitHub research index answered HTTP 410 (retired), so the repositories were read from the GitHub API;
Firecrawl web/developer search found the finance papers. PyTorch is in `../.venv-qpo` (CPU); the app venv has none.

Finance claims found on the web, all unverified against full text: KASPER (arXiv 2507.18983) reports a Sharpe of
12.02 on Yahoo Finance data with no cost model and no stated out-of-sample design (implausible, and Yahoo is no
source here); "KAN based Autoencoders for Factor Models" (arXiv 2408.02694) claims higher long-short Sharpe than an
MLP autoencoder, numbers not in the abstract; KAN for VIX forecasting (Expert Syst. Appl. 2025). No head-to-head of
KAN against the Gu-Kelly-Xiu networks on a public panel was found.

## A small theorem that shapes the KAN test

Setting: a one-layer KAN [d,1], f(x) = sum_i phi_i(x_i), phi_i = w_b silu + spline of degree p >= 1 on a grid of
[a,b] extended by p knots per side, inputs in [a,b]. Rank IC = per-cross-section Spearman correlation.

- (a) The class contains every affine map on [a,b]^d (Marsden's identity: degree-p B-splines on the extended knots
  reproduce polynomials of degree <= p on [a,b]; take w_b = 0). It nests the Ridge hypothesis class (not the fitted
  Ridge estimate).
- (b) On cross-sectional ranks in [0,1], it contains the Composite score S = (1/d) sum r_i.
- (c) With one feature, a strictly increasing phi leaves the rank IC unchanged on every cross-section; only a
  decreasing, tie-creating or non-monotone phi changes it.
- (d) With two or more features, strictly increasing per-feature transforms can change it: stocks (1,2), (2,0), (0,0)
  with returns (3,2,1) have rank IC 1 for x1 + x2 and 0.5 for x1^3 + x2.

So any KAN gain over the Composite must come from the shape of each factor's response curve (where (d) bites), or
from interactions in a deeper KAN, and the shallow KAN can always fall back to the Composite itself. Numerical
check: cubic B-splines on efficient-kan's default grid fit 1 and 0.7x - 0.2 with error below 2e-15; the example's
Spearman values are 1.0 and 0.5. Verification: two fresh-context adversarial verifiers (math-olympiad method, no tools, blind to the derivation and to each other): both HOLDS. Limits they recorded: the nesting holds only on [a,b]^d (splines vanish outside the extended grid, and pykan's update_grid_from_samples shrinks the grid to the training range); take w_s = 1 in the construction; (b) assumes the Composite averages a fixed set of d ranks (an average over the available factors only is not a fixed affine map); (c) is exact arithmetic (a nearly flat curve can create ties in float32).

## Ideas, ranked, each with a prior written now

1. **KAN-Composite (cheapest, strongest fit).** A [d,1] KAN on the Composite's own ranked factors, in the 213
   harness, against Ridge, the MLP and the Composite score; then read each learned curve (pykan symbolic
   extraction) and, if one is clearly non-linear (momentum saturating, value only in the top decile), freeze it as a
   fixed formula in the Composite. The frozen formula, not the net, is what would ship. Prior: IC within +/- 0.01
   of Ridge; about 15% that it beats Ridge with t > 2.
2. **Peripheral tilt (from paper 5's network view).** Build the correlation network of the universe each month and
   tilt the Composite's 15 toward the network periphery (low centrality); Pozzi, Di Matteo and Aste (2013) found
   peripheral portfolios did better out of sample. No neural net needed; differs from the graph/tree HRP we ran
   (that one sets weights, this one sets the ranking). Prior: about 20% it passes.
3. **Volatility-managed exposure for the aggressive profile.** Scale equity exposure by forecast market vol
   (Moreira-Muir), with a pure-PyTorch Mamba (mamba.py) or CfC forecaster against an EWMA/HAR baseline. Without
   leverage this only cuts exposure in turbulent months, so it helps the drawdown more than the return. Prior: the
   neural forecaster ties HAR (the literature's usual result); about 10% it adds anything over HAR.
4. **CfC on the CVM filing stream (Sentinela).** Irregular event times are the case continuous-time cells are built
   for: time since the last filing, decay of a material fact's effect. A forward log first, as with every text
   signal. Prior: low; the Sentinela's current judge is still the baseline to beat.
5. **Semantic peer groups (from paper 8).** Cluster companies by embeddings of their CVM filings into data-driven
   peer groups, used for the sector cap and for within-peer value ranks. Prior: small effect; mostly a better
   sector map.
6. **Quantum.** Nothing in papers 3 or 4 changes mu, so no new attempt. At most a quantum-inspired CVaR estimate as
   a teaching item, with no decision attached.

The honest ceiling for "turbocharging" the aggressive portfolio is unchanged: the levers that raise its risk and
return are concentration, looser per-name and sector caps, and leverage (rejected at the XP margin rate), not a new
optimizer. Ideas 1 and 2 are the only ones cheap enough to pre-register now (CPU, minutes, torch in
`../.venv-qpo`); both wait for the user's go.
