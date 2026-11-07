# Damodaran's YouTube content since 2023: what it holds and what the agent should use (2026-11-07)

The user's question of 07/11: does his content make sense for an agent that judges our stocks? This note is the
judgment made before the agent was specified (`docs/superpowers/specs/2026-11-07-damodaran-design.md`).

## What was fetched

- Channel `@AswathDamodaranonValuation` (id `UCLvnJL8htRR1T9cbSccaoVw`) through ScrapeCreators
  (`/v1/youtube/channel-videos`, `/v1/youtube/channel/playlists`, `/v1/youtube/video/transcript`), 448 credits.
- 427 videos published since 2023-01-01 (dates are YouTube's relative "n months ago", so approximate to the month;
  older years were not fetched), 426 with English captions (`wk6yec9pGAs` has none), about 4.2 million words,
  407 hours. 26 playlists (courses back to 2016).
- Kept outside git (size, and the transcripts are his work): `D:\data_damodaran\youtube\catalogo_videos.json`,
  `catalogo_playlists.json`, `transcricoes\<video id>.json` (title, date, length, views, description, text).

## What it holds (mentions per 10,000 words)

| Series | Videos | Words | Brazil | Country risk | Currency / inflation | Banks / insurance | Rates / ERP | Momentum / timing |
|---|---|---|---|---|---|---|---|---|
| Course sessions (MBA, undergraduate, corporate finance, philosophies) | 279 | 3.42 m | 3.4 | 4.1 | 6.9 | 9.2 | 8.8 | 0.7 |
| Data Updates (each January to March) | 29 | 148 k | 2.3 | 13.2 | 20.1 | 8.5 | 15.3 | 2.9 |
| Country risk editions | 5 | 39 k | 20.1 | 54.9 | 38.0 | 4.3 | 67.7 | 0.0 |
| Book chapters (Little Book, Corporate Life Cycle) | 34 | 155 k | 0.3 | 2.5 | 2.8 | 8.3 | 5.4 | 1.8 |
| Topical (blog-post videos: AI, IPOs, markets) | 79 | 443 k | 0.5 | 4.2 | 8.1 | 11.5 | 8.7 | 3.5 |

## Reading

1. **It is a method, not stock opinions.** Brazil appears as worked examples: the risk-free rate in reais, the
   country risk premium, and a company's exposure to country risk by where its revenue comes from (Embraer and Ambev,
   with his 2004 numbers). None of the videos values a name the Composite or the 12/3 holds today.
2. **It repeats.** About 80% of the words are his yearly courses; the 2023, 2024 and 2025 editions teach the same
   syllabus. The newest edition is enough for the method.
3. **It does not time markets.** Momentum and timing are nearly absent (0.7 a 10,000 words in the courses), and when
   they appear it is as a philosophy he does not follow. An agent in his image revalues on new numbers (quarterly
   statements, each January's data), not on news: the Composite's record cadence, not the Sentinela's hours.
4. **The current numbers matter more than the talk.** His yearly country risk and equity risk premium updates and
   his industry betas are already in `D:\data_damodaran` (2,073 spreadsheets, dated editions back to 2011); for
   Brazil in July 2026: Ba1, default spread 1.99%, country risk premium 3.10%, total ERP 7.30%.
5. **The model already knows the framework.** DeepSeek was trained on his books, blog and classes; feeding it 4
   million words adds little and costs much. What it needs is the rules applied the same way every time.

## Decision

- A curated core of 108 videos (888 thousand words; `docs/dados/damodaran-selecao.json`): the country risk
  editions, the Data Updates 2023-2026 and the newest course's sessions on risk-free rates, risk premiums, cost of
  capital, cash flows, growth and terminal value, financial service firms, cyclical and commodity firms, declining
  and distressed firms, the life cycle, narrative and pricing.
- From it, a method card of at most 40 rules (4,000 tokens), each with one verbatim passage and its video id checked
  by a test. The card and his current data go to the agent; the transcripts do not.
- The agent itself: a fictitious-account layer over the Composite and the 12/3 (the spec above).
