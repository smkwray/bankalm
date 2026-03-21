PYTHON ?= $(HOME)/venvs/bankalm/bin/python
QUARTERS ?= $(shell $(PYTHON) scripts/pipeline_status.py --source-window financials)
SOD_YEARS ?= $(shell $(PYTHON) scripts/pipeline_status.py --source-window sod-years)
FFIEC_QUARTERS ?= $(shell $(PYTHON) scripts/pipeline_status.py --source-window ffiec)
TREASURY_YEARS ?= $(shell $(PYTHON) scripts/pipeline_status.py --source-window treasury-years)
REPORT_QUARTER ?= $(shell $(PYTHON) scripts/pipeline_status.py --source-window report-quarter)

RAW     := data/raw
PROC    := data/processed
REPORTS := data/reports

.PHONY: help test clean download-all download-market-rates download-market-rates-if-key build-all build-deposit-competition reports universe validate smoke status backtest-failures

help:
	@echo "Main targets:"
	@echo "  universe      — download + build + report for the full bank universe"
	@echo "  download-all  — download raw data (FDIC, SOD, FFIEC, failures, Treasury)"
	@echo "  download-treasury  — download Treasury yield history"
	@echo "  download-market-rates — optional FRED market-rate history (IORB / ON RRP)"
	@echo "  build-all     — build panel → stickiness → ALM → treasury → deposit-competition → indices"
	@echo "  backtest-failures — run quarter-aligned failure cohort validation"
	@echo "  reports        — generate mart, league tables, summaries, and site data"
	@echo "  validate       — run 2023 stress validation report"
	@echo "  smoke          — quick build from existing raw data (no downloads)"
	@echo "  status         — check data freshness, coverage, and integrity"
	@echo "  test           — run all tests"
	@echo "  clean          — remove processed outputs (keeps raw downloads)"

# ──────────────────────── Tests ────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v

# ──────────────────────── Downloads ────────────────────────
download-financials:
	@for Q in $(QUARTERS); do \
		echo "Downloading financials $$Q..."; \
		$(PYTHON) scripts/download_fdic_bankfind.py financials \
			--fields-file config/fdic_financials_core_fields.txt \
			--filters "REPDTE:$$Q" \
			--sort-by CERT --limit 10000 --max-pages 1 --overwrite \
			--out $(RAW)/fdic/universe_financials_$$Q.parquet; \
	done

download-institutions:
	$(PYTHON) scripts/download_fdic_bankfind.py institutions \
		--fields CERT,FED_RSSD,RSSDHCR,NAME,BKCLASS,STALP,CITY \
		--sort-by CERT --limit 10000 --max-pages 0 --overwrite \
		--out $(RAW)/fdic/universe_institutions.parquet

download-sod:
	@for Y in $(SOD_YEARS); do \
		echo "Downloading SOD $$Y..."; \
		$(PYTHON) scripts/download_fdic_bankfind.py sod \
			--fields CERT,YEAR,DEPSUMBR,UNINUMBR,STALPBR,CNTYNAMB,BRNUM \
			--filters "YEAR:$$Y" \
			--sort-by CERT --limit 10000 --max-pages 0 --overwrite \
			--out $(RAW)/fdic/universe_sod_$$Y.parquet; \
	done

download-failures:
	$(PYTHON) scripts/download_fdic_bankfind.py failures \
		--sort-by FAILDATE --sort-order ASC --limit 10000 --max-pages 0 --overwrite \
		--out $(RAW)/fdic/failures.parquet

download-ffiec:
	@for Q in $(FFIEC_QUARTERS); do \
		ZIP="$(RAW)/ffiec/FFIEC CDR Call Bulk All Schedules $$Q.zip"; \
		if [ ! -f "$$ZIP" ]; then \
			echo "Downloading FFIEC CDR $$Q..."; \
			$(PYTHON) -c "\
from ffiec_data_collector import FFIECDownloader; \
from ffiec_data_collector.downloader import FileFormat; \
import shutil; from pathlib import Path; \
d = FFIECDownloader(); \
r = d.download_cdr_single_period('$$Q', format=FileFormat.TSV); \
dest = Path('$(RAW)/ffiec') / r.filename; \
dest.parent.mkdir(parents=True, exist_ok=True); \
shutil.move(str(r.file_path), str(dest)) if r.success and r.file_path.resolve() != dest.resolve() else None; \
print(f'  {dest}: {r.size_bytes:,} bytes') if r.success else print(f'  FAILED: {r.error_message}')"; \
		else \
			echo "Skipping FFIEC CDR $$Q (already downloaded)"; \
		fi; \
	done

download-treasury:
	@for Y in $(TREASURY_YEARS); do \
		echo "Downloading Treasury yields $$Y..."; \
		$(PYTHON) scripts/download_treasury_yields.py \
			--year $$Y --out $(RAW)/treasury/yields_$$Y.parquet; \
	done

download-market-rates:
	$(PYTHON) scripts/download_fred_series.py \
		--series-id IORB \
		--series-id RRPONTSYAWARD \
		--out $(RAW)/macro/fred_rates.parquet

download-market-rates-if-key:
	@if [ -n "$$FRED_API_KEY" ]; then \
		echo "FRED_API_KEY detected, downloading market rates..."; \
		$(PYTHON) scripts/download_fred_series.py \
			--series-id IORB \
			--series-id RRPONTSYAWARD \
			--out $(RAW)/macro/fred_rates.parquet; \
	else \
		echo "FRED_API_KEY not set, skipping market-rate download (deposit competition will use Treasury-only inputs)."; \
	fi

download-all: download-financials download-institutions download-sod download-ffiec download-failures download-treasury download-market-rates-if-key

# ──────────────────────── FFIEC Extraction ────────────────────────
extract-ffiec:
	@for ZIP in $(RAW)/ffiec/FFIEC\ CDR\ Call\ Bulk\ All\ Schedules\ *.zip; do \
		DATE=$$(basename "$$ZIP" .zip | grep -o '[0-9]\{8\}'); \
		OUT="$(PROC)/ffiec_repricing_$$DATE.parquet"; \
		if [ ! -f "$$OUT" ]; then \
			echo "Extracting FFIEC repricing $$DATE..."; \
			$(PYTHON) scripts/extract_ffiec_repricing.py \
				--zip "$$ZIP" \
				--map config/ffiec_repricing_map.yaml \
				--institutions $(RAW)/fdic/universe_institutions.parquet \
				--out "$$OUT"; \
		fi; \
	done

# ──────────────────────── Build Pipeline ────────────────────────
extract-derivatives:
	@for ZIP in $(RAW)/ffiec/FFIEC\ CDR\ Call\ Bulk\ All\ Schedules\ *.zip; do \
		DATE=$$(basename "$$ZIP" .zip | grep -o '[0-9]\{8\}'); \
		OUT="$(PROC)/derivatives_$$DATE.parquet"; \
		if [ ! -f "$$OUT" ]; then \
			echo "Extracting derivatives $$DATE..."; \
			$(PYTHON) scripts/extract_derivatives.py \
				--zip "$$ZIP" \
				--institutions $(RAW)/fdic/universe_institutions.parquet \
				--out "$$OUT"; \
		fi; \
	done

build-panel: extract-ffiec extract-derivatives
	$(PYTHON) scripts/build_bank_panel.py \
		--financials-glob '$(RAW)/fdic/universe_financials_*.parquet' \
		--institutions-glob '$(RAW)/fdic/universe_institutions.parquet' \
		--sod-glob '$(RAW)/fdic/universe_sod_*.parquet' \
		--ffiec-repricing-glob '$(PROC)/ffiec_repricing_*.parquet' \
		--derivatives-glob '$(PROC)/derivatives_*.parquet' \
		--out $(PROC)/universe_bank_panel.parquet

build-stickiness: build-panel
	$(PYTHON) scripts/build_deposit_stickiness_features.py \
		--input $(PROC)/universe_bank_panel.parquet \
		--scenario-config config/stickiness_scenarios.yaml \
		--out $(PROC)/universe_deposit_stickiness.parquet

build-alm: build-stickiness
	$(PYTHON) scripts/build_alm_mismatch_features.py \
		--input $(PROC)/universe_deposit_stickiness.parquet \
		--out $(PROC)/universe_alm_features.parquet

build-treasury: build-alm
	$(PYTHON) scripts/build_treasury_extensions.py \
		--input $(PROC)/universe_alm_features.parquet \
		--treasury-glob '$(RAW)/treasury/yields_*.parquet' \
		--out $(PROC)/universe_treasury_features.parquet

build-deposit-competition: build-treasury
	$(PYTHON) scripts/build_deposit_competition_features.py \
		--input $(PROC)/universe_treasury_features.parquet \
		--config config/deposit_competition.yaml \
		--market-rates $(RAW)/macro/fred_rates.parquet \
		--out $(PROC)/universe_deposit_competition_features.parquet

build-indices: build-deposit-competition
	$(PYTHON) scripts/build_indices.py \
		--stickiness $(PROC)/universe_deposit_stickiness.parquet \
		--alm $(PROC)/universe_alm_features.parquet \
		--treasury $(PROC)/universe_treasury_features.parquet \
		--deposit-competition $(PROC)/universe_deposit_competition_features.parquet \
		--peer-groups config/peer_groups.yaml \
		--weights config/index_weights.yaml \
		--out $(PROC)/universe_bank_indices.parquet

build-supervised: build-stickiness
	$(PYTHON) scripts/build_supervised_stickiness.py \
		--input $(PROC)/universe_deposit_stickiness.parquet \
		--outflow-percentile 5 \
		--out $(PROC)/universe_supervised_stickiness.parquet

build-sec-mapping:
	$(PYTHON) scripts/build_sec_mapping.py \
		--institutions $(RAW)/fdic/universe_institutions.parquet \
		--tickers-cache $(RAW)/sec/company_tickers.json \
		--overrides config/sec_overrides.yaml \
		--nic-structure $(PROC)/nic_structure.parquet \
		--out $(PROC)/universe_sec_mapping.parquet

build-all: build-indices build-supervised build-sec-mapping

backtest-failures: build-indices download-failures
	$(PYTHON) scripts/run_failure_backtest.py \
		--indices $(PROC)/universe_bank_indices.parquet \
		--failures $(RAW)/fdic/failures.parquet \
		--out $(PROC)/universe_failure_backtest.parquet

# ──────────────────────── Reports ────────────────────────
reports: build-all backtest-failures
	$(PYTHON) scripts/generate_reports.py \
		--indices $(PROC)/universe_bank_indices.parquet \
		--supervised $(PROC)/universe_supervised_stickiness.parquet \
		--validation-metrics $(PROC)/universe_failure_backtest_metrics.csv \
		--failures $(RAW)/fdic/failures.parquet \
		--mart-out $(PROC)/universe_publishable_mart.parquet \
		--core-panel-out $(PROC)/universe_core_panel.parquet \
		--enriched-panel-out $(PROC)/universe_enriched_panel.parquet \
		--site-dir site/data \
		--out-dir $(REPORTS)/universe/ \
		--quarter $(REPORT_QUARTER)

# ──────────────────────── Validation ────────────────────────
validate:
	$(PYTHON) scripts/run_validation_2023.py

# ──────────────────────── Full Pipeline ────────────────────────
universe: download-all build-all reports
	@echo "Universe pipeline complete."

# ──────────────────────── Smoke (quick rebuild, no downloads) ────────────────
smoke: build-all reports
	@echo "Smoke rebuild complete."

# ──────────────────────── Status ────────────────────────
status:
	$(PYTHON) scripts/pipeline_status.py

# ──────────────────────── Clean ────────────────────────
clean:
	rm -f $(PROC)/*.parquet
	rm -rf $(REPORTS)/
	rm -f site/data/*.json
	rm -rf site/data/banks
