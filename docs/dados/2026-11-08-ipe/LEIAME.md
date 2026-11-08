# CVM IPE buyback rows, frozen 2026-11-08

Built by `scripts/ipe_recompras_congelar.py` from the CVM IPE yearly index files 2010-2026 (dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE, kept in `../../cvm/`), rows accepted by `core.recompra.e_recompra` (spec `docs/superpowers/specs/2026-11-08-pead-recompra-design.md`, section 4). `issuer` = each issuer of the row's CVM code in `docs/dados/2026-10-24-cvm/fundamentals.parquet` (one row per issuer, spec addendum 1; empty when the code has none). The index keeps one row per document version.

IPE rows read 640,071; buyback rows x issuers 3,673; without an issuer in the fundamentals file 177 (48 CVM codes); with an unreadable date 0
