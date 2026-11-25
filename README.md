# bankALM

**[Browse the interactive atlas](https://smkwray.github.io/bankalm/)**

bankALM is a public-data pipeline and static atlas that screens U.S. banks for funding fragility. From quarterly FDIC (Federal Deposit Insurance Corporation) filings, it builds bank-quarter scores for deposit run risk, asset-liability mismatch (ALM), Treasury liquidity buffers, and deposit competition, and checks each score against later bank failures. The published build covers 5,237 banks from 2020 to 2024 on the site and 9,137 banks from 2007 to 2024 in the full-history panel.

The scores are exploratory screening indices and scenario proxies, not supervisory ratings, investment advice, or a decision-grade failure model. bankALM is not affiliated with the FDIC, the Federal Financial Institutions Examination Council (FFIEC), or the Federal Reserve.

## Backtest results

Each index was checked against FDIC failure records: for every bank-quarter, did the bank fail within the next four quarters? The table shows the area under the ROC curve (AUC, where 0.5 is chance) and the share of failing bank-quarters that fell in the top 20% of scores. The run-risk test covers 2,221 failing bank-quarters.

| Index | AUC | Failures in top 20% |
|---|---|---|
| Run risk | 0.78 | 59% |
| Composite fragility | 0.74 | 51% |
| Deposit competition | 0.72 | 54% |
| ALM mismatch | 0.56 | 26% |

Like the other indices, deposit competition now participates in the same quarter-aligned failure backtest. These figures rank banks against each other; they are not failure probabilities. The site has league tables, per-bank score breakdowns, and the methodology for each index. The published data ends in the fourth quarter of 2024.

## Method

Each index combines public-data ratios, ranked within the same quarter and asset-size peer group (under $1 billion, $1–10 billion, $10–100 billion, over $100 billion), using fixed weights, and is scaled from 0 to 100. Run risk weighs uninsured, brokered and large-account deposits, deposit cost, and recent deposit weakness. ALM mismatch compares long-term assets, securities and loans with stable funding. The Treasury buffer measures Treasury, agency and high-quality liquid assets against uninsured or runnable funding, including after a 100 basis point rate shock. Deposit competition measures how far the best safe short-term rate (interest on reserve balances (IORB), the overnight reverse repo (ON RRP) rate, or the 3-month Treasury yield) exceeds a bank's deposit cost, weighted by its rate-sensitive funding. The composite is 0.40 × run risk + 0.40 × ALM mismatch + 0.20 × (100 − Treasury buffer). An experimental supervised model trained on next-quarter deposit weakness is published alongside the transparent scores. Weights, peer groups, and scenario settings are in `config/`.

## Data sources

| Source | Provides |
|---|---|
| [FDIC BankFind](https://api.fdic.gov/banks) | Quarterly financials, institution metadata, Summary of Deposits (SOD) branch data, failed-bank list |
| FFIEC Central Data Repository (CDR) Call Reports | Maturity and repricing buckets, derivatives |
| FFIEC National Information Center (NIC) | Holding-company structure |
| SEC EDGAR | Tickers and 10-Q/10-K filings for uninsured-deposit disclosures |
| U.S. Treasury daily par yield curve | Treasury yields |
| FRED (Federal Reserve Economic Data), optional | IORB and ON RRP rates |

All sources are free. FRED requires an API key in `FRED_API_KEY`; without it, deposit competition uses Treasury yields only. Downloaded and processed data are written to `data/`, which is not tracked.

## Reproduce

Requires Python 3.11 or later. The Makefile's default interpreter path does not exist on other machines, so pass `PYTHON` explicitly.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export FRED_API_KEY=your_key            # optional

make universe PYTHON=python             # download all data, build indices, run backtest, write reports and site data
make smoke PYTHON=python                # rebuild from data already downloaded
make test PYTHON=python
make status PYTHON=python               # data freshness, coverage and integrity
cd site && python3 -m http.server 8000  # preview the atlas
```

`make help` lists the individual stages.

## Outputs

- `data/processed/universe_core_panel.parquet`: the full-history panel (2007 onward), limited to fields available consistently across that period.
- `data/processed/universe_enriched_panel.parquet`: the recent-history panel (2020 onward), adding FFIEC repricing, derivatives, Treasury and deposit-competition fields. The site is built from this panel.
- `data/processed/universe_failure_backtest_metrics.csv`: backtest metrics by index, horizon (1, 2 and 4 quarters), and episode (2007–2010, 2019–2021, 2022–2024, full sample).
- `data/reports/universe/`: league tables, bank drill-downs, peer-group summaries, and a quarterly Treasury yield summary.
- `site/data/`: the manifest and per-bank JSON files the atlas reads.

## Limitations

- Public filings only: no internal deposit behavior, management assumptions, or daily funding flows, and SOD branch geography is annual.
- ALM measures are pre-hedge structural public-data proxies, not a bank's internal hedge-adjusted model; an overlay from Call Report derivative data is available.
- Deposit-life outputs are scenario proxies under baseline, adverse and severe assumptions, not empirical decay estimates.
- Deposit competition is a transparent public-data pressure screen for exploratory comparison. Its backtest result above does not make it more than that: it is a transparent public-data pressure screen rather than a complete funding diagnostic.
- Repricing, Treasury and deposit-competition fields exist only in the recent-history panel.
- Holding-company and institution metadata are current snapshots, not historical as-of mappings.
- The supervised overlay is experimental and backward-looking.

## License

MIT; see [LICENSE](LICENSE).
