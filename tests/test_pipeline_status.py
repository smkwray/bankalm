from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_pipeline_status():
    spec = importlib.util.spec_from_file_location("pipeline_status", ROOT / "scripts" / "pipeline_status.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_window_helper_rolls_forward_and_caps_sod():
    mod = load_pipeline_status()
    today = date(2026, 3, 20)

    assert mod.source_window("report-quarter", today=today) == "20251231"
    assert mod.source_window("financials", today=today)[-1] == "20251231"
    assert mod.source_window("ffiec", today=today)[0] == "20200331"
    assert mod.source_window("sod-years", today=today) == [str(y) for y in range(2010, 2025)]
    assert mod.source_window("treasury-years", today=today) == ["2022", "2023", "2024", "2025"]


def test_report_source_freshness_reports_lag_and_status(tmp_path, capsys):
    mod = load_pipeline_status()
    mod.ROOT = tmp_path

    for rel in [
        "data/raw/fdic/universe_financials_20241231.parquet",
        "data/raw/ffiec/FFIEC CDR Call Bulk All Schedules 20240930.zip",
        "data/raw/fdic/universe_sod_2023.parquet",
        "data/raw/treasury/yields_2024.parquet",
    ]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    mod.report_source_freshness(today=date(2026, 3, 20))
    out = capsys.readouterr().out

    assert "FDIC financials: latest 2024-12-31 / target 2025-12-31 / lag 4Q" in out
    assert "FFIEC CDR zips: latest 2024-09-30 / target 2025-12-31 / lag 5Q" in out
    assert "FDIC SOD: latest 2023 / target 2024 / lag 1Y" in out
    assert "Treasury yields: latest 2024 / target 2025 / lag 1Y" in out


def test_report_source_freshness_ignores_unparseable_files(tmp_path, capsys):
    mod = load_pipeline_status()
    mod.ROOT = tmp_path

    good = tmp_path / "data/raw/ffiec/FFIEC CDR Call Bulk All Schedules 20240930.zip"
    bad = tmp_path / "data/raw/ffiec/README.txt"
    good.parent.mkdir(parents=True, exist_ok=True)
    good.write_bytes(b"x")
    bad.write_text("note", encoding="utf-8")

    mod.report_source_freshness(today=date(2026, 3, 20))
    out = capsys.readouterr().out

    assert "FFIEC CDR zips: latest 2024-09-30 / target 2025-12-31 / lag 5Q" in out


def test_parse_date_from_filename_supports_real_ffiec_name():
    mod = load_pipeline_status()

    actual = mod._parse_date_from_filename("FFIEC CDR Call Bulk All Schedules 12312024.zip")

    assert actual == date(2024, 12, 31)


def test_main_checks_universe_sec_mapping_path(monkeypatch):
    mod = load_pipeline_status()
    calls: list[tuple[str, str]] = []

    def fake_check_files(*_args, **_kwargs):
        return 0

    def fake_check_parquet(label, path):
        calls.append((label, path))
        return None

    monkeypatch.setattr(mod, "check_files", fake_check_files)
    monkeypatch.setattr(mod, "check_parquet", fake_check_parquet)
    monkeypatch.setattr(mod, "report_source_freshness", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_load_site_freshness", lambda: None)
    monkeypatch.setattr(mod, "ROOT", ROOT)

    mod.main()

    assert ("SEC mapping", "data/processed/universe_sec_mapping.parquet") in calls
    assert not any(path == "data/processed/smoke_sec_mapping.parquet" for _, path in calls)
