# Issuer CNPJ roots for the debenture reading (2026-11-24)

- `issuer_roots.parquet`: CVM code -> CNPJ root (first 8 digits), from CVM's open registry (`cad_cia_aberta`). A byte
  copy of `docs/dados/2026-11-12-inovemetria/issuer_roots.parquet` on branch `feat/informados` (commit `7330e2d`),
  the map the Inovemetria and insider readings used. Public data. Joined to the frozen export's
  `docs/dados/2026-10-22/sectors.parquet` (issuer -> CVM code).
