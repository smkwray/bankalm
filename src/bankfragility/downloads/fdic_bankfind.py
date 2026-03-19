"""FDIC BankFind downloader with schema audit and checkpointed resume support."""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from bankfragility.tables import read_table, save_table

BASE_URL = "https://api.fdic.gov/banks"
SOURCES = {
    "institutions",
    "locations",
    "history",
    "summary",
    "failures",
    "sod",
    "financials",
    "demographics",
}


@dataclass(frozen=True)
class DownloadConfig:
    source: str
    filters: str
    fields: list[str]
    sort_by: str
    sort_order: str
    limit: int
    max_pages: int
    sleep: float
    out: Path
    checkpoint_dir: Path
    audit_out: Path | None
    strict_fields: bool
    overwrite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=sorted(SOURCES), help="FDIC API dataset name")
    parser.add_argument("--filters", default="", help="Elastic-style filter string")
    parser.add_argument("--fields", default="", help="Comma-separated field list")
    parser.add_argument("--fields-file", type=Path, help="Text file with one field per line")
    parser.add_argument("--sort-by", default="", help="Optional sort field")
    parser.add_argument("--sort-order", default="ASC", choices=["ASC", "DESC"])
    parser.add_argument("--limit", type=int, default=10_000, help="Rows per page")
    parser.add_argument("--max-pages", type=int, default=0, help="0 means no explicit cap")
    parser.add_argument("--sleep", type=float, default=0.15, help="Pause between requests")
    parser.add_argument("--checkpoint-dir", type=Path, help="Directory for per-page checkpoint files")
    parser.add_argument("--audit-out", type=Path, help="Optional JSON audit report path")
    parser.add_argument(
        "--strict-fields",
        action="store_true",
        help="Fail if requested fields are not in the reference dictionary or absent from the response",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard any existing checkpoint state and rebuild the output from scratch",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output .csv or .parquet path")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_checkpoint_dir(out_path: Path) -> Path:
    return out_path.parent / f".{out_path.stem}_fdic_pages"


def load_fields(args: argparse.Namespace) -> list[str]:
    fields: list[str] = []
    if args.fields_file:
        for line in args.fields_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields.append(line.upper())
    if args.fields:
        fields.extend([value.strip().upper() for value in args.fields.split(",") if value.strip()])
    deduped: list[str] = []
    seen: set[str] = set()
    for field in fields:
        if field not in seen:
            seen.add(field)
            deduped.append(field)
    return deduped


def load_reference_fields(
    reference_csv: Path | None = None,
    converted_csv: Path | None = None,
) -> set[str]:
    reference_csv = reference_csv or project_root() / "reference" / "reference_variablesanddefinitions.csv"
    converted_csv = converted_csv or project_root() / "reference" / "converted_variables.csv"
    valid_fields: set[str] = set()

    if reference_csv.exists():
        with reference_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                field = str(row.get("Variable", "")).strip().upper()
                if field:
                    valid_fields.add(field)

    if converted_csv.exists():
        with converted_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for key in ("Old", "New"):
                    field = str(row.get(key, "")).strip().upper()
                    if field:
                        valid_fields.add(field)

    return valid_fields


def fetch_page(
    session: requests.Session,
    source: str,
    params: dict[str, str | int],
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> pd.DataFrame:
    url = f"{BASE_URL}/{source}"
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = session.get(
                url,
                params={**params, "format": "csv"},
                headers={"Accept": "text/csv"},
                timeout=60,
            )
            response.raise_for_status()
            text = response.text.strip()
            if not text:
                return pd.DataFrame()
            df = pd.read_csv(io.StringIO(text))
            df.columns = [str(col).upper() for col in df.columns]
            return df
        except (requests.RequestException, IOError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait = retry_delay * (2 ** attempt)
                print(f"Retry {attempt + 1}/{max_retries} after {wait:.0f}s: {exc}", file=sys.stderr)
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def checkpoint_request_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "request.json"


def checkpoint_page_path(checkpoint_dir: Path, page_number: int) -> Path:
    return checkpoint_dir / f"page_{page_number:05d}.csv"


def list_checkpoint_pages(checkpoint_dir: Path) -> list[Path]:
    return sorted(checkpoint_dir.glob("page_*.csv"))


def checkpoint_payload(config: DownloadConfig) -> dict[str, Any]:
    return {
        "source": config.source,
        "filters": config.filters,
        "fields": config.fields,
        "sort_by": config.sort_by,
        "sort_order": config.sort_order,
        "limit": config.limit,
    }


def prepare_checkpoint_dir(config: DownloadConfig) -> None:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    request_path = checkpoint_request_path(config.checkpoint_dir)
    payload = checkpoint_payload(config)

    if config.overwrite:
        for path in list_checkpoint_pages(config.checkpoint_dir):
            path.unlink()
        if request_path.exists():
            request_path.unlink()

    if request_path.exists():
        existing = json.loads(request_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise SystemExit(
                "Existing checkpoint parameters do not match this request. "
                "Use --overwrite or a different --checkpoint-dir."
            )
    else:
        request_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def existing_progress(checkpoint_dir: Path, limit: int) -> tuple[int, bool]:
    pages = list_checkpoint_pages(checkpoint_dir)
    if not pages:
        return 0, False

    last_page = read_table(pages[-1])
    is_complete = len(last_page) < limit
    return len(pages), is_complete


def collect_pages(
    session: requests.Session,
    config: DownloadConfig,
    fetch_page_fn: Any = None,
) -> int:
    fetch_page_fn = fetch_page_fn or fetch_page
    prepare_checkpoint_dir(config)
    page_count, already_complete = existing_progress(config.checkpoint_dir, config.limit)
    if already_complete:
        return page_count
    if config.max_pages and page_count >= config.max_pages:
        return page_count

    offset = page_count * config.limit
    fetched_pages = page_count

    while True:
        params: dict[str, str | int] = {
            "limit": config.limit,
            "offset": offset,
        }
        if config.filters:
            params["filters"] = config.filters
        if config.fields:
            params["fields"] = ",".join(config.fields)
        if config.sort_by:
            params["sort_by"] = config.sort_by
            params["sort_order"] = config.sort_order

        page = fetch_page_fn(session=session, source=config.source, params=params)
        if page.empty:
            break

        fetched_pages += 1
        checkpoint_path = checkpoint_page_path(config.checkpoint_dir, fetched_pages)
        save_table(page, checkpoint_path)
        print(
            f"Fetched page {fetched_pages}: {len(page):,} rows (offset={offset:,})",
            file=sys.stderr,
        )

        if len(page) < config.limit:
            break
        if config.max_pages and fetched_pages >= config.max_pages:
            break

        offset += config.limit
        time.sleep(config.sleep)

    return fetched_pages


def load_checkpoint_table(checkpoint_dir: Path) -> pd.DataFrame:
    pages = list_checkpoint_pages(checkpoint_dir)
    if not pages:
        return pd.DataFrame()
    frames = [read_table(path) for path in pages]
    return pd.concat(frames, ignore_index=True, sort=False)


def build_audit(
    config: DownloadConfig,
    df: pd.DataFrame,
    known_fields: set[str],
) -> dict[str, Any]:
    requested = config.fields
    returned = [str(col).upper() for col in df.columns]
    unknown_requested = [field for field in requested if known_fields and field not in known_fields]
    missing_returned = [field for field in requested if field not in returned]
    return {
        "source": config.source,
        "filters": config.filters,
        "sort_by": config.sort_by,
        "sort_order": config.sort_order,
        "limit": config.limit,
        "checkpoint_dir": str(config.checkpoint_dir),
        "page_files": len(list_checkpoint_pages(config.checkpoint_dir)),
        "row_count": int(len(df)),
        "requested_fields": requested,
        "unknown_requested_fields": unknown_requested,
        "missing_returned_fields": missing_returned,
        "returned_fields": returned,
    }


def emit_field_warnings(audit: dict[str, Any], strict_fields: bool) -> None:
    problems: list[str] = []
    if audit["unknown_requested_fields"]:
        problems.append(
            "Unknown requested fields against reference dictionary: "
            + ", ".join(audit["unknown_requested_fields"])
        )
    if audit["missing_returned_fields"]:
        problems.append(
            "Requested fields missing from response: " + ", ".join(audit["missing_returned_fields"])
        )

    if not problems:
        return

    message = " | ".join(problems)
    if strict_fields:
        raise SystemExit(message)
    print(f"Field audit warning: {message}", file=sys.stderr)


def run_download(args: argparse.Namespace, session: requests.Session | None = None) -> pd.DataFrame:
    fields = load_fields(args)
    config = DownloadConfig(
        source=args.source,
        filters=args.filters,
        fields=fields,
        sort_by=args.sort_by,
        sort_order=args.sort_order,
        limit=args.limit,
        max_pages=args.max_pages,
        sleep=args.sleep,
        out=args.out,
        checkpoint_dir=args.checkpoint_dir or default_checkpoint_dir(args.out),
        audit_out=args.audit_out,
        strict_fields=bool(args.strict_fields),
        overwrite=bool(args.overwrite),
    )

    known_fields = load_reference_fields()
    session = session or requests.Session()
    collect_pages(session=session, config=config)

    df = load_checkpoint_table(config.checkpoint_dir)
    if df.empty:
        raise SystemExit("No rows returned. Check the source, filters, or field list.")

    df = df.drop_duplicates().reset_index(drop=True)
    audit = build_audit(config=config, df=df, known_fields=known_fields)
    emit_field_warnings(audit=audit, strict_fields=config.strict_fields)

    save_table(df, config.out)
    if config.audit_out:
        config.audit_out.parent.mkdir(parents=True, exist_ok=True)
        config.audit_out.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved {len(df):,} rows to {config.out}", file=sys.stderr)
    return df


def main() -> None:
    args = parse_args()
    run_download(args)


if __name__ == "__main__":
    main()
