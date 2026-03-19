from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd

from bankfragility.downloads.fdic_bankfind import (
    collect_pages,
    load_reference_fields,
    load_checkpoint_table,
    run_download,
    DownloadConfig,
)


class DummySession:
    pass


def test_load_reference_fields_includes_converted_aliases(tmp_path: Path) -> None:
    reference_csv = tmp_path / "reference_variablesanddefinitions.csv"
    reference_csv.write_text("Variable,Title,Definition\nDEPDOM,Domestic Deposits,desc\n", encoding="utf-8")
    converted_csv = tmp_path / "converted_variables.csv"
    converted_csv.write_text("Title,Old,New\nTime deposits,IDCD3LES,CD3LES\n", encoding="utf-8")

    fields = load_reference_fields(reference_csv=reference_csv, converted_csv=converted_csv)

    assert {"DEPDOM", "IDCD3LES", "CD3LES"} <= fields


def test_collect_pages_resumes_from_existing_checkpoints(tmp_path: Path) -> None:
    config = DownloadConfig(
        source="financials",
        filters="",
        fields=["CERT"],
        sort_by="CERT",
        sort_order="ASC",
        limit=2,
        max_pages=0,
        sleep=0.0,
        out=tmp_path / "financials.parquet",
        checkpoint_dir=tmp_path / "checkpoints",
        audit_out=None,
        strict_fields=False,
        overwrite=False,
    )

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame({"CERT": ["1001", "1002"]})
    existing.to_csv(config.checkpoint_dir / "page_00001.csv", index=False)

    seen_offsets: list[int] = []

    def fake_fetch_page(session: DummySession, source: str, params: dict[str, str | int]) -> pd.DataFrame:
        seen_offsets.append(int(params["offset"]))
        if int(params["offset"]) == 2:
            return pd.DataFrame({"CERT": ["1003"]})
        return pd.DataFrame()

    page_count = collect_pages(session=DummySession(), config=config, fetch_page_fn=fake_fetch_page)
    resumed = load_checkpoint_table(config.checkpoint_dir)

    assert page_count == 2
    assert seen_offsets == [2]
    assert resumed["CERT"].astype(str).tolist() == ["1001", "1002", "1003"]


def test_run_download_writes_audit_and_dedupes_rows(tmp_path: Path, monkeypatch) -> None:
    args = Namespace(
        source="financials",
        filters="",
        fields="CERT,DEPDOM",
        fields_file=None,
        sort_by="CERT",
        sort_order="ASC",
        limit=10,
        max_pages=0,
        sleep=0.0,
        checkpoint_dir=tmp_path / "checkpoints",
        audit_out=tmp_path / "audit.json",
        strict_fields=False,
        overwrite=False,
        out=tmp_path / "financials.parquet",
    )

    monkeypatch.setattr(
        "bankfragility.downloads.fdic_bankfind.load_reference_fields",
        lambda *args, **kwargs: {"CERT", "DEPDOM"},
    )

    def fake_fetch_page(session: DummySession, source: str, params: dict[str, str | int]) -> pd.DataFrame:
        return pd.DataFrame({"CERT": ["1001", "1001"], "DEPDOM": [1.0, 1.0]})

    monkeypatch.setattr("bankfragility.downloads.fdic_bankfind.fetch_page", fake_fetch_page)

    df = run_download(args=args, session=DummySession())

    assert len(df) == 1
    assert args.out.exists()
    assert args.audit_out.exists()
