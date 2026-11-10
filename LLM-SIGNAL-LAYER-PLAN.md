# LLM signal layer for Aleph Quant — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the free NVIDIA Nemotron models into a document triage and event-extraction layer that produces a small number of pre-registered, testable features for the Aleph Quant universe — and decide, with a real gate, whether any of them earns a place in the Composite.

**Architecture:** Three stages. A cheap 8B reranker screens the day's filings and news per ticker (one forward pass per query-document pair, no generation). A large model reads only the handful of documents that survived, and returns strict JSON: event type, direction, magnitude bucket, confidence, and the passage it came from. The Composite stays untouched; the extracted feature enters as a veto/boost that is measured against the existing gate, never as a replacement for the scoring model.

**Tech Stack:** Python 3.12, the project's own CVM/Fato Relevante pipeline, the **NVIDIA NIM API** (`ai.api.nvidia.com`) as the default provider with **OpenRouter's free rerank endpoint** as the fallback, `httpx`, `pydantic` for the JSON contract, and the existing `core/backtest` for measurement.

**Spec:** this file is the design; the research it rests on is summarised in the Sources section.

## Global Constraints

- Everything written into the project stays in US English: code, comments, docs, commits.
- Never edit `core/market_data.py`, `core/paper.py`, `core/bcb.py` or `assets.py`.
- No point-in-time leakage: a document dated D is available to a decision made on D+1 at the earliest, and the pipeline must prove it with a test.
- Every number a model produces is a *feature*, never an order. No LLM output reaches the order path without passing the gate below.
- Zero cost: the free tiers only. No paid inference.
- Tests run one at a time and must be seen to fail before the code that fixes them.

## Review Focus

- A clock skew or a document published after the close being used the same session — the single most likely way this project fools itself.
- The model returning plausible JSON with a hallucinated passage; anything without a verbatim quote must be dropped.
- The free tier's daily request ceiling silently truncating the universe and producing a look-biased sample.
- A feature that only works because the LLM was trained on data after 2010; the walk-forward must re-scan, not reuse.
- Prompts logged by the free provider containing the user's positions.

---

## What the research says (10/11/2026)

| model | what it is | size | context | benchmark | license |
|---|---|---|---|---|---|
| Nemotron 3.5 Rerank 8B | multimodal cross-encoder reranker, returns relevance logits | 8.38B | 32,768 | NDCG@10 67.49 ViDoRe V3 image, 76.35 MIRACL-VISION | OpenMDW-1.1, commercial use allowed |
| Nemotron 3.5 Bolt 74B-A9B | MoE, vision+text in, built for local agentic work | 74B total / 9B active | 256k | Intelligence Index 21.4 | not stated in sources |
| Nemotron **3** Ultra 550B-A55B | MoE, hybrid Transformer-Mamba, reasoning | 550B / 55B active | 1M (65k output) | Intelligence Index 23 | not stated in sources |

Naming correction: the OpenRouter free endpoint is **Nemotron 3** Ultra, not 3.5 — NVIDIA's 3.5 wave (October 2026) is
Super VL, Bolt, Bolt E4B and the Retriever family (embed, rerank, parse). There is no "Nemotron 3.5 Ultra" in any source.

Three facts decide the design:

1. **The reranker is the only one that can cover the whole universe.** It emits logits in a single forward pass — there is no autoregressive decoding — so its cost is roughly constant per pair. The 34-language text evaluation includes Portuguese. It is also free on OpenRouter through a dedicated `POST /api/v1/rerank` endpoint ($0 in / $0 out, P50 latency 2.53 s), separate from chat completions, while the generative models are limited to about 50 requests/day on the free tier.
2. **Bolt 74B-A9B does not fit this laptop.** NVIDIA describes the 74B-A9B checkpoint as a 24 GB model; the machine has 20 GB of RAM, no NVIDIA GPU, and the integrated Vega 8 has no ROCm. Quantized to IQ2 it would fit in memory but run at a handful of tokens per second.
3. **Free tier is not free of terms.** OpenRouter's `:free` generative models charge $0 but around 20 requests/minute and 50 requests/day, and the provider may log prompts. NVIDIA's own trial API is more generous (40 RPM, 10k RPD reported) but has its own trial terms. The reranker's licence (OpenMDW-1.1) permits commercial use, so self-hosting it carries no legal question.

## Which API, and why the plan uses both

|  | NVIDIA NIM (build.nvidia.com) | OpenRouter `:free` |
|---|---|---|
| reranker endpoint | `POST https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-rerank-8b/reranking` | `POST https://openrouter.ai/api/v1/rerank` |
| key | `NVIDIA_API_KEY` (`nvapi-…`), obtained at build.nvidia.com | `OPENROUTER_API_KEY` (`sk-or-…`) |
| price | free trial credits, then metered | $0 in / $0 out |
| limits reported | ~40 RPM, 10k requests/day | ~20 RPM and 50 requests/day on the generative free tier; the rerank endpoint is separate |
| prompt handling | NVIDIA API Trial Terms | OpenRouter terms; the provider may log prompts |
| generative models | Bolt / Ultra per NVIDIA's catalogue | the only place the free `nemotron-3-ultra-550b-a55b:free` is reachable |

**Decision:** NIM is the default, because it is NVIDIA's own service, its ceiling is the most generous, and the key lives
in the project's existing `.env` beside the other data keys. OpenRouter is the fallback, used when NIM answers 402 or 429,
and it is the only route to the free Ultra model. The generative calls are capped by arithmetic, not by preference: at
50 requests/day on the free tier, only the documents that survive triage can be sent, which is exactly what Task 4 does.

Keys go in `ai-hedge-fund/.env` and are never committed. A missing key must raise at construction time, not degrade to an
empty score list — the failure this plan is most afraid of is a silent "no relevant documents" that looks like a finding.

## The brainstorm, including what was rejected

- **LLM predicts returns.** Rejected. No point-in-time feature archive, no way to distinguish learning from memorisation, and the project's own experience says the signal, not the solver, is what pays.
- **LLM as the portfolio allocator.** Rejected for the arithmetic reason already registered: seventeen years of monthly decisions is about 200 samples, four orders of magnitude short of what a policy needs.
- **Sentiment score per stock added to the Composite.** Rejected as the first move: the same feature has no seventeen-year history here, so it cannot be backtested — only shadow-logged.
- **Rerank as triage, extraction as structured events, consumed as a veto.** Kept. It is the only design where (a) the cost is affordable across the universe, (b) the output is a small set of typed facts rather than a vibe, and (c) the Composite is untouched, so the downside is bounded.

**Why this could beat the Composite where the DRL did not:** the DRL was given sixty days of price and asked to find alpha; the market had already priced that. This layer is given text the Composite does not read at all — Fato Relevante, earnings-call language, guidance changes — and its job is only to say *this document materially changes the case for this stock*. It adds information, not a new optimiser over the same information.

## Tasks

### Task 1: The reranking client

**Files:** Create `core/nvidia_rerank.py`; Test `tests/test_nvidia_rerank.py`

**Interfaces:**
- Produces: `pontuar(query: str, documentos: list[str], timeout: float = 30.0) -> list[float]` and `ordernar(query, documentos) -> list[tuple[str, float]]`

- [ ] **Step 1: Write the failing test**

```python
def test_a_failed_request_raises_instead_of_returning_empty_scores():
    with pytest.raises(RerankError):
        client.pontuar("query", ["doc"])   # against a server that returns 429
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_nvidia_rerank.py -v`, expect ImportError.

- [ ] **Step 3: Implement** the NIM call first — `POST https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-rerank-8b/reranking`
  with `Authorization: Bearer $NVIDIA_API_KEY`, returning one float per document in input order. On 402 or 429, retry once
  against `POST https://openrouter.ai/api/v1/rerank` with `Authorization: Bearer $OPENROUTER_API_KEY`. Raise `RerankError`
  on any other non-200, and raise `ConfiguracaoAusente` at construction when neither key is set, so a rate limit is never
  mistaken for "no relevant documents".

- [ ] **Step 3b: Test the fallback** — a client built with only `OPENROUTER_API_KEY` must skip NIM entirely; a client with
  neither key must fail before any HTTP call.

- [ ] **Step 4: Run tests to verify they pass.**

- [ ] **Step 5: Commit** — `feat(llm): reranker client`

### Task 2: The extraction contract

**Files:** Create `core/llm_evento.py`; Test `tests/test_llm_evento.py`

**Interfaces:**
- Produces: `Evento(ticker, data, tipo, direcao, magnitude, confianca, trecho)` as a frozen dataclass, and `validar(bruto: dict) -> Evento`

- [ ] **Step 1: Write the failing test** — an event whose `trecho` is not a substring of the source document must be rejected.

- [ ] **Step 2: Run it, see it fail.**

- [ ] **Step 3: Implement** `validar`: the quote must appear verbatim in the document (normalised whitespace); `direcao` in `{"alta","baixa","neutro"}`; `magnitude` in `{"material","imaterial"}`; `confianca` in [0, 1]; anything else raises `EventoInvalido`.

- [ ] **Step 4: Run it, see it pass.**

- [ ] **Step 5: Commit** — `feat(llm): strict event contract with a verbatim-quote guard`

### Task 3: The point-in-time guard

**Files:** Create `core/llm_pit.py`; Test `tests/test_llm_pit.py`

**Interfaces:**
- Produces: `disponivel_em(publicado: pd.Timestamp, sessao: pd.Timestamp) -> bool`, true only when the document was published at least one session before `sessao`.

- [ ] **Step 1: Write the failing test** — a document published at 18:30 on D is not available for the D close, only for D+1.

- [ ] **Step 2: Run it, see it fail.**

- [ ] **Step 3: Implement** using the exchange session calendar already in the project; a document available only after the close is stamped to the next session by its own test.

- [ ] **Step 4: Run it, see it pass.**

- [ ] **Step 5: Commit** — `feat(llm): point-in-time availability guard`

### Task 4: The daily sweep

**Files:** Create `scripts/llm_varredura.py`; Test `tests/test_llm_varredura.py`

**Interfaces:**
- Consumes: `pontuar`, `validar`, `disponivel_em`
- Produces: `varredura(data, tickers) -> list[Evento]`, writing to `lago/llm_eventos.parquet` with one row per event

- [ ] **Step 1: Write the failing test** — a run that hits the daily request ceiling must record which tickers were skipped, and must not mark them "no event".

- [ ] **Step 2: Run it, see it fail.**

- [ ] **Step 3: Implement** the two stages: rerank to the top 3 documents per ticker, then extract on those only. Persist `coberto` (boolean) beside every row.

- [ ] **Step 4: Run it, see it pass.**

- [ ] **Step 5: Commit** — `feat(llm): daily sweep that records its own coverage`

### Task 5: The gate

**Files:** Create `scripts/llm_portao.py`; Test `tests/test_llm_portao.py`

**Interfaces:**
- Consumes: `lago/llm_eventos.parquet`, the Composite's weights
- Produces: a table comparing the Composite with and without the veto, over the period the events actually cover

- [ ] **Step 1: Write the failing test** — with zero events the gated and ungated curves must be identical.

- [ ] **Step 2: Run it, see it fail.**

- [ ] **Step 3: Implement** the rule pre-registered here, before any event exists: exclude a ticker from the next rebalance when it has a `material` + `baixa` event with confidence ≥ 0.7 in the last 20 sessions. No other rule may be tried without a new decision entry.

- [ ] **Step 4: Run it, see it pass.**

- [ ] **Step 5: Commit** — `feat(llm): pre-registered veto gate`

## What this plan refuses to promise

The events start existing on the day the sweep first runs, so the gate cannot be measured on 2010-2026 until the CVM archive is replayed through the same pipeline — and that replay is only honest if the archive is a real point-in-time record rather than a back-filled one. Until that is verified, the LLM layer is a **shadow log**, exactly like every other signal this project has refused to promote without evidence.

## Sources

- NVIDIA. Nemotron 3.5 Bolt 74B-A9B 24GB technical report. https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-Technical-Report.pdf
- NVIDIA. Nemotron 3.5 Rerank 8B model card. https://build.nvidia.com/nvidia/nemotron-3_5-rerank-8b
- Artificial Analysis. "NVIDIA Releases Nemotron 3.5 Super and Nemotron 3.5 Bolt". https://artificialanalysis.ai/articles/nemotron-3-5-super-and-bolt
- NVIDIA blog. "NVIDIA Expands Nemotron Family". https://blogs.nvidia.com/blog/nemotron-3-5-super-vl/
- OpenRouter, `nvidia/nemotron-3-ultra-550b-a55b:free` endpoints. https://openrouter.ai/api/v1/models/nvidia/nemotron-3-ultra-550b-a55b:free/endpoints
