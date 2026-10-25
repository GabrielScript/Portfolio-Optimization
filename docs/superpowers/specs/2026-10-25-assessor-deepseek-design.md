# Assessor DeepSeek on the Alepha page — design (2026-10-25, session 49)

## Intent

The user wants an investment adviser inside the Streamlit app: DeepSeek V4.1 Flash 1017 reads the Alepha plan (the
registered Conservadora, 15 stocks), the stocks' numbers, the BCB macro data, CVM filings, recent news and posts, and
gives an impartial, skeptical opinion per stock for an aggressive ("arrojado") investor who accepts more volatility
for more return, but not much more. It must be cheap (prompt cache, few tokens), have tools to look things up, and
minimize hallucination. It must critique its own thesis before the verdict: three macro reasons, from the BCB data
only, that would invalidate it.

## Goals

- One button, one "parecer" (opinion) per click: per stock a thesis, three BCB-only counter-reasons written before
  the stance, risks, a stance (favorável / neutra / desfavorável / evidência insuficiente) and a confidence; a
  ranking of the 15; a short summary; the limits of the data.
- Every factual claim points to an evidence id and quotes it; the app checks the quote against the evidence and
  drops what does not match. Numbers in a claim must appear in the quoted evidence.
- Cost and cache shown under the opinion (tokens, cache hit rate, estimated US$).

## Non-goals and anti-goals

- Never reaches the weights, the orders, the plan or the paper routines (same boundary as the traffic light,
  DECISOES 132-133). The CVM disclaimer stays: a personal study tool, not a recommendation.
- Not a universe screener: only the 15 stocks of the registered plan are judged (the optimizer already chose them;
  judging the other 166 would be a second, untested strategy).
- No vector or graph store in v1 (see "Retrieval").
- No numeric risk premium or price target: a number the data does not hold would be invented.
- Failure = an opinion that states a number or fact not in the evidence, or a page that crashes without a key.

## Retrieval: deterministic evidence pack + tools (not vector/graph RAG)

The facts are tabular and few (15 stocks x ~12 numbers, ~10 macro series, ~15 lights, ~20 headlines): exact lookup
beats similarity search, and a fixed pack is cacheable. The pack is a list of evidence lines
`E12 | bcb | 2026-10-24 | Selic meta: 15,00% a.a.`, ids in a fixed order (macro, then stocks by ticker, then lights,
then headlines), classes `plano`, `bcb`, `mercado`, `cvm`, `noticia`, `web`, `social`.

Tools (fixed list, same order and text every call so the prefix caches; a tool without its key answers
`{"erro": "indisponível"}` instead of disappearing):

| Tool | Source | Class |
|---|---|---|
| `metricas_da_acao(ticker)` | the market panel's table (lake prices, CVM statements) | mercado |
| `fatos_relevantes(ticker)` | CVM IPE filings of the panel, with Jev tone | cvm |
| `serie_bcb(serie)` | BCB SGS, whitelist (Selic meta, IPCA 12m, PTAX, IBC-Br) | bcb |
| `buscar_web(consulta)` | DeepSeek Search API, `lang:pt`, last 30 days, 5 results | web |
| `ler_pagina(url)` | DeepSeek Fetch API, first 3,000 characters | web |
| `buscar_noticias(consulta)` | Firecrawl v2 search, news, last week | noticia |
| `buscar_posts(consulta)` | ScrapeCreators Threads search | social |

Tool results get evidence ids too (`E1001`...), so claims based on them are checked the same way. Web, news and post
text is untrusted data: the prompt says it never carries instructions.

## Model settings

- `deepseek-flash` (served as DeepSeek-V4.1-Flash-1017), OpenAI-format `/chat/completions` through `requests`.
- Thinking enabled, `reasoning_effort: "high"`. Temperature is ignored by the API in thinking mode, so it is not
  sent (recorded in DECISOES 136, the user asked for it to be configured).
- `response_format: json_object` (probed live 25/10 together with thinking and tools: works); the assistant message
  with its `reasoning_content` goes back on every tool round (the API answers 400 otherwise).
- Caps: 6 tool rounds (then `tool_choice: "none"`), 12 tool calls, `max_tokens` 64,000 per request (16,000 cut a
  real answer); at most 3 claims per list; empty content gets one retry; an answer cut at the token limit
  (`finish_reason: "length"`) or a structural failure gets one repair request; then the opinion is refused with the
  reason on screen.
- Cache: system message = rules + evidence pack, nothing variable before it; the question goes last.
- The public app spends the owner's key: the same pack, plan and day is cached for every visitor, and at most 8
  paid opinions a day per app process.

## Anti-hallucination checks (`validar_parecer`)

- Tickers only from the plan, each once; the ranking only from judged tickers.
- Claim = `{afirmacao, evidencia, citacao}`; `evidencia` must exist; `citacao` must be inside that evidence's text
  (spaces, case and accents normalized); every stand-alone number in `afirmacao` must be in `citacao` (digits
  inside a word, like PETR4 or 2T26, are names); no link in `afirmacao` or in free text.
- `contra_tese_macro`: exactly three claims, all with `bcb` evidence; otherwise the stance becomes "evidência
  insuficiente" with the reason shown.
- Invalid claims are dropped and counted on screen. Everything the model or a web page wrote is shown with Markdown
  escaped (no injected link, image or format).

## Online behaviour

Streamlit Cloud has no D:. The pack uses only what the deployed app already loads (lake parquet files in the repo,
BCB/Focus APIs, the headlines when their token exists). Keys come from `st.secrets["deepseek"]["api_key"]`,
`st.secrets["firecrawl"]["api_key"]`, `st.secrets["scrapecreators"]["api_key"]`, or the same-named environment
variables locally; without the DeepSeek key the section shows "Assessor indisponível" and nothing else. The user
puts the keys in the Streamlit secrets (I never type tokens). ANBIMA data lives only on D: and is out of v1.

## Left to the builder

Prompt wording, table layout, cost table per UTC hour (peak 01-04 and 06-10 UTC weekdays, double price).
