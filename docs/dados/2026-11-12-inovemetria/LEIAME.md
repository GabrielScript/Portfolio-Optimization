# Innovation data of 2026-11-12 (Inovemetria)

Inputs of the Inovemetria reading (`docs/superpowers/specs/2026-11-12-inovemetria-design.md`), built by the lake's
`b3lake innovation` command (branch `feat/innovation-data`) on 12/11/2026 and frozen by `scripts/inovemetria_montar.py`.

- `intangibles.parquet`: 24,262 rows
- `ipca.parquet`: 387 rows
- `issuer_roots.parquet`: 2,566 rows
- `patents.parquet`: 9,502 rows
- `subsidiaries.parquet`: 60,820 rows

- `intangibles.parquet`: CVM account 1.02.04.01 (intangibles without goodwill), consolidated first, each value
  dated by its document's receipt (`available_from`), only rows received up to 2026-10-24, with issuer and sector
  of `docs/dados/2026-10-22/sectors.parquet`.
- `patents.parquet`: INPI BADEPI v12.0 (snapshot 13/10/2026, dispatches to 30/12/2025) applications of listed
  companies and of companies they declared controlled; `known_from` = first publication (RPI 3.1, 1.1 or 1.3).
- `subsidiaries.parquet`: CVM FRE `participacao_sociedade`, "Controlada" or >= 50%, dated by the FRE receipt.
- `issuer_roots.parquet`: CVM code -> CNPJ root (CVM registry). `ipca.parquet`: BCB SGS 433, monthly %.

Known limits: one BADEPI snapshot (today's holder and today's WIPO fields, back-dated); holding companies inherit
their subsidiaries' patents (Itaúsa, Bradespar); goodwill filed inside 1.02.04.01 and concession assets count as
intangibles.
