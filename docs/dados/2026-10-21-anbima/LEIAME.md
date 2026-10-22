# ANBIMA index histories of 2026-10-21

Inputs of `scripts/avaliar_acoes_titulos.py` (IMA-B) and `scripts/avaliar_filtros_macro.py` (IDA-DI).

- Source: ANBIMA Data, Índices page, the "Baixar histórico (XLS)" link of each index (data.anbima.com.br/indices),
  downloaded once on 2026-10-21 (one request per file, as a click on the page does):
  - `IMAB-HISTORICO.xls`: IMA-B, daily Number Index from 16/09/2003 to 20/10/2026 (server Last-Modified
    Tue, 20 Oct 2026 22:04:04 GMT);
  - `IDADI-HISTORICO.xls`: IDA-DI, daily Number Index from 05/01/2009 to 20/10/2026 (Last-Modified
    Wed, 21 Oct 2026 11:12:24 GMT).
- The files are .xlsx workbooks named .xls: the scripts read them from their bytes so the format is sniffed. The index
  name column changes spelling inside a file ("IMA-B", later "IMA - B") and is not read.
- ANBIMA regenerates these files every day: a download made on another day will not match the hashes in
  `SHA256SUMS`, and the scripts stop on a mismatch.
- **The workbooks are not in git.** ANBIMA's terms of use (www.anbima.com.br/pt_br/termos-de-uso.htm, read on
  2026-10-21) forbid using its databases and files without written authorization (8.2) and automated or mass
  collection (2.4); they allow reproducing the public screen content, unaltered, crediting the source (8.3). This
  repository is public, so only the hashes and this note are committed; the workbooks stay on the author's machine
  (and in the lake's raw folder on D:). Results derived from them credit ANBIMA as the source.
- ANBIMA's disclaimer (terms 9.2.1): ANBIMA's indices are unofficial estimates, used at the user's own risk;
  ANBIMA is not liable for any divergence, lag or inaccuracy in them.
