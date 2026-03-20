# U.S. Bank Deposit Stickiness & ALM Mismatch Pipeline

**[smkwray.github.io/bankalm](https://smkwray.github.io/bankalm/)**

## Site

The **bankALM** static site provides an interactive atlas of bank fragility scores, league tables, and bank detail pages — all generated from the pipeline's enriched panel.

Live at **[smkwray.github.io/bankalm](https://smkwray.github.io/bankalm/)** | served from `site/` via GitHub Pages.

- **Homepage** — hero metrics, risk index cards, backtest comparison
- **League Table** — sortable, filterable, paginated rankings for each index (Run Risk, ALM Mismatch, Deposit Competition, Composite Fragility)
- **Bank Detail** — per-bank score breakdown, profile metadata, methodology notes

To run locally:

```bash
cd site && python3 -m http.server 8000
```

## About

bankALM is an independent research project that screens U.S. bank fragility using only free public regulatory data. The live atlas is at **[smkwray.github.io/bankalm](https://smkwray.github.io/bankalm/)**.

It is not affiliated with or endorsed by the FDIC, FFIEC, Federal Reserve, or any financial institution. Scores are computed from public filings and should not be interpreted as supervisory ratings or investment advice.

---

A reproducible pipeline using only free public data to estimate, at the U.S. bank-quarter level:

1. **Deposit stickiness / run risk** — transparent score + experimental supervised overlay
2. **Effective maturity scenarios** — baseline, adverse, severe deposit-life scenario proxies
3. **Asset-liability mismatch** — 13 structural ratios (pre-hedge proxy)
4. **FFIEC repricing buckets** — loan/deposit/borrowing maturity + repricing gaps + duration-gap-lite
5. **Treasury buffer extensions** — shock scenarios, encumbrance-adjusted coverage
6. **Deposit competition pressure** — outside-option premium, pass-through gap, and rate-sensitive funding exposure
7. **Interpretable indices** — peer-group normalized, per-row versioned

## Quickstart

```bash
python3 -m venv ~/venvs/bankalm
source ~/venvs/bankalm/bin/activate
pip install -e .
pip install pytest

source ./.env
make universe    # download all data + build pipeline + generate reports
make test        # run the test suite
make status      # check data freshness, coverage, integrity
```

## Pipeline Scale

Counts depend on the current processed build. The default universe spans quarterly FDIC bank filings from 2007 forward and persists intermediates in Parquet.

## Failure Cohort Backtest

The validation layer now uses **quarter-aligned bank-quarter cohorts** rather than one-row-per-bank comparisons. For each report quarter, the pipeline labels whether a bank fails within `1Q`, `2Q`, or `4Q`, tracks whether that horizon is observable, and produces reproducible metrics tables by horizon and episode slice.

## Data Sources

| Source | What it provides |
|---|---|
| FDIC BankFind financials | Deposits, assets, capital, securities, borrowings |
| FDIC BankFind institutions | Identifiers, charter, location (current snapshot metadata) |
| FDIC Summary of Deposits | Branch geography, deposit concentration |
| FDIC failures list | Historical bank failure dates and institutions |
| FFIEC CDR Call Reports | Loan/deposit/borrowing maturity buckets, repricing data |
| FFIEC NIC | Holding-company structure and parent chains |
| SEC EDGAR | Company tickers, 10-Q/10-K filing parser for uninsured deposits |

## Validation

The repo ships:

- quarter-aligned failure cohort labels and metrics
- consistency checks on the bank-quarter panel
- historical stress validation scripts and reporting outputs

These artifacts support **exploratory public-data screening**, not a claim that bankALM is a decision-grade failure predictor.

## Package Structure

```
src/bankfragility/
  downloads/         FDIC API, Treasury yields
  staging/           Bank panel (5-way join), FFIEC repricing extraction
  features/          Deposit stickiness, ALM mismatch, Treasury extensions, deposit competition
  models/            Index construction, supervised stickiness overlay
  validation/        Failure backtest, 11 consistency checks
  entity/            SEC CIK mapping, 10-Q/10-K parser, NIC structure
  reporting/         League tables, drill-downs, peer-group summaries
```

## Make Targets

| Target | Description |
|---|---|
| `make universe` | Full download + build + reports |
| `make smoke` | Quick rebuild from existing data (no downloads) |
| `make test` | Run the test suite |
| `make status` | Check data freshness, coverage, integrity |
| `make validate` | Run 2023 SVB stress validation |
| `make download-all` | Download default raw data |
| `make download-treasury` | Download Treasury yield history |
| `make download-market-rates` | Optional FRED short-rate history (IORB / ON RRP) |
| `make build-all` | Build panel through indices + supervised |
| `make build-deposit-competition` | Build the post-Treasury deposit-competition feature stage |
| `make backtest-failures` | Run quarter-aligned failure cohort validation |
| `make reports` | Generate league tables and reports |
| `make clean` | Remove processed outputs (keeps raw) |

## Outputs

The pipeline produces per bank-quarter:

- **Raw features:** deposit composition, ALM ratios, Treasury coverage, deposit-competition pressure features, FFIEC repricing buckets
- **Scenario estimates:** deposit WAL under baseline/adverse/severe, stable-equivalent amounts
- **Indices:** `run_risk_index`, `deposit_stickiness_index`, `alm_mismatch_index`, `treasury_buffer_index`, `deposit_competition_pressure_index`, `deposit_competition_resilience_index`, `funding_fragility_index`
- **Supervised overlay:** experimental `SUPERVISED_OUTFLOW_SCORE` plus next-quarter observability and label fields
- **Failure backtest:** `FAIL_WITHIN_1Q`, `FAIL_WITHIN_2Q`, `FAIL_WITHIN_4Q` labels plus cohort metrics tables
- **Split publishable panels:**
  `data/processed/universe_core_panel.parquet` for the historically consistent full-history core output
  `data/processed/universe_enriched_panel.parquet` for the recent-history advanced-module output
- **Mart + site exports:** integrated `data/processed/universe_publishable_mart.parquet` plus generated `site/data/manifest.json` and `site/data/league.json`
- **Reports:** bank drill-downs, quarter league tables, peer-group summaries, scenario comparisons
- **Macro context report:** `data/reports/universe/treasury_regime_summary.parquet` with one row per quarter of the observed Treasury backdrop

## Configuration

| File | Purpose |
|---|---|
| `config/peer_groups.yaml` | Asset-size peer group thresholds (in FDIC thousands) |
| `config/index_weights.yaml` | Component weights for composite indices |
| `config/deposit_competition.yaml` | Outside-option benchmarks and transparent pressure-score weights |
| `config/stickiness_scenarios.yaml` | Deposit life parameters by scenario |
| `config/ffiec_repricing_map.yaml` | MDRM code → standard horizon mapping |
| `config/sec_overrides.yaml` | Manual CERT → SEC CIK/ticker overrides |
| `config/field_definitions.yaml` | Field definitions and known changes |

## Limitations

- No internal deposit behavior or management assumptions
- Annual SOD geography, not daily funding flows
- ALM model is pre-hedge (derivative overlay available via FFIEC RC-L)
- Deposit-life outputs are scenario assumptions, not empirical decay estimates
- ALM outputs are structural public-data proxies, not a bank's internal hedge-adjusted model
- Holding-company disclosures and institutions metadata are not historical as-of mappings
- Supervised overlay is backward-looking and experimental
- Treasury yield history is integrated into the recent-history enriched panel, not the full-history core panel
- Deposit-competition features and indices are part of the recent-history enriched surface unless optional market-rate history is supplied more broadly

The transparent score is the primary product. The supervised overlay is secondary. bankALM is a **transparent public-data fragility screen and scenario engine**, not a decision-grade probability model.

## Published Artifacts

Use the split outputs intentionally:

- `universe_core_panel.parquet`: full-history panel for historically consistent use. Excludes reliance on the recent-only advanced-module surface.
- `universe_enriched_panel.parquet`: recent-history panel beginning when FFIEC/derivative/Treasury enrichment is available, including deposit-competition fields.
- `universe_publishable_mart.parquet`: integrated internal mart used to derive the split panels and site exports.
- `universe_failure_backtest_metrics.csv`: quarter-aligned validation metrics by score, horizon, and episode slice.

The static site is generated from the recent-history enriched panel and includes manifest metadata describing both published panel variants, including the deposit-competition index where available.
