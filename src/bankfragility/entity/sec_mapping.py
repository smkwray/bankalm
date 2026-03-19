"""Map bank holding companies to SEC CIK/ticker via the SEC EDGAR bulk file.

The SEC publishes ``company_tickers.json`` which contains CIK, ticker, and
company title for all SEC-registered entities.  This module downloads that
file, fuzzy-matches holding company names from the FDIC institutions data,
and produces a CERT → CIK → ticker mapping table.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from bankfragility.tables import save_table

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_USER_AGENT = "bankalm-research/1.0 research@example.com"

# Common suffixes to strip for fuzzy matching
_STRIP_SUFFIXES = re.compile(
    r",?\s*(inc\.?|corp\.?|corporation|company|co\.?|n\.?a\.?|national association"
    r"|bancorp\.?|bancshares|financial|group|holdings?|ltd\.?|llc|lp|plc)$",
    re.IGNORECASE,
)
_STRIP_PUNC = re.compile(r"[^a-z0-9\s]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--institutions", type=Path, required=True, help="Parquet with CERT, FED_RSSD, NAME, RSSDHCR")
    parser.add_argument("--tickers-cache", type=Path, help="Optional cached company_tickers.json")
    parser.add_argument("--overrides", type=Path, help="Optional YAML with manual CERT → CIK overrides")
    parser.add_argument("--nic-structure", type=Path, help="Optional NIC structure parquet for top-holder names")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def fetch_sec_tickers(cache_path: Path | None = None, session: requests.Session | None = None) -> list[dict[str, Any]]:
    """Download or load from cache the SEC company tickers list."""
    if cache_path and cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        session = session or requests.Session()
        r = session.get(SEC_TICKERS_URL, headers={"User-Agent": SEC_USER_AGENT}, timeout=60)
        r.raise_for_status()
        data = r.json()
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Normalize: the file is a dict keyed by string index
    if isinstance(data, dict):
        return list(data.values())
    return data


def _normalize_name(name: str) -> str:
    """Lowercase, strip suffixes and punctuation for fuzzy matching."""
    name = name.strip().lower()
    # Iteratively strip suffixes
    for _ in range(3):
        name = _STRIP_SUFFIXES.sub("", name).strip()
    name = _STRIP_PUNC.sub("", name)
    return " ".join(name.split())


def build_sec_lookup(tickers: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame from SEC tickers with normalized names."""
    df = pd.DataFrame(tickers)
    df.columns = [str(c).upper() for c in df.columns]
    df = df.rename(columns={"CIK_STR": "CIK"})
    df["SEC_NAME_NORM"] = df["TITLE"].apply(_normalize_name)
    return df


def match_institutions_to_sec(
    institutions: pd.DataFrame,
    sec_lookup: pd.DataFrame,
    nic_structure: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Match institution holding companies to SEC entities.

    Strategy (per docs/03_entity_resolution.md):
    1. Use NIC top-holder name if available (RSSDHCR → top holder → name → SEC match)
    2. Fall back to institution name exact match
    3. Fall back to distinctive-token fuzzy match
    """
    inst = institutions.copy()

    # Build candidate names from institution data
    name_col = "NAME" if "NAME" in inst.columns else "NAMEFULL"
    if name_col not in inst.columns:
        return inst

    inst["INST_NAME_NORM"] = inst[name_col].fillna("").apply(_normalize_name)

    # If NIC structure is available, look up the top-holder name for each institution
    # and use that for SEC matching (per spec: map RSSDHCR → top holder → SEC)
    if nic_structure is not None and not nic_structure.empty:
        rssd_col = "FED_RSSD" if "FED_RSSD" in inst.columns else "RSSDID"
        if rssd_col in inst.columns and "TOP_HOLDER_NAME" in nic_structure.columns:
            nic = nic_structure[["ID_RSSD", "TOP_HOLDER_NAME"]].dropna().drop_duplicates(subset=["ID_RSSD"])
            nic["ID_RSSD"] = pd.to_numeric(nic["ID_RSSD"], errors="coerce").astype("Int64")
            inst[rssd_col] = pd.to_numeric(inst[rssd_col], errors="coerce").astype("Int64")
            inst = inst.merge(nic, left_on=rssd_col, right_on="ID_RSSD", how="left")
            # Prefer top-holder name for SEC matching when available
            inst["HC_NAME_NORM"] = inst["TOP_HOLDER_NAME"].fillna("").apply(_normalize_name)
            has_hc = inst["HC_NAME_NORM"].str.len() > 0
            inst.loc[has_hc, "INST_NAME_NORM"] = inst.loc[has_hc, "HC_NAME_NORM"]

    # Build a lookup dict: normalized name → (CIK, ticker, title)
    exact_lookup: dict[str, dict[str, Any]] = {}
    for _, row in sec_lookup.iterrows():
        key = row["SEC_NAME_NORM"]
        if key and key not in exact_lookup:
            exact_lookup[key] = {
                "SEC_CIK": row["CIK"],
                "SEC_TICKER": row.get("TICKER", ""),
                "SEC_TITLE": row.get("TITLE", ""),
                "SEC_MATCH_TYPE": "exact",
            }

    # Stop words that carry no distinguishing power for bank names
    stop_words = {"bank", "of", "the", "and", "national", "first", "trust", "savings", "federal"}

    # Build token-overlap index for fuzzy matching (distinctive tokens only)
    sec_entries: list[tuple[set[str], set[str], dict[str, Any]]] = []
    for _, row in sec_lookup.iterrows():
        all_tokens = set(row["SEC_NAME_NORM"].split())
        distinctive = all_tokens - stop_words
        if len(distinctive) >= 1:
            sec_entries.append((all_tokens, distinctive, {
                "SEC_CIK": row["CIK"],
                "SEC_TICKER": row.get("TICKER", ""),
                "SEC_TITLE": row.get("TITLE", ""),
            }))

    def _best_fuzzy_match(norm: str, min_distinctive_overlap: int = 2, min_score: float = 0.6) -> dict[str, Any] | None:
        """Find SEC entry with best distinctive-token overlap."""
        all_tokens = set(norm.split())
        distinctive = all_tokens - stop_words
        if not distinctive:
            return None
        best_score = 0.0
        best_entry: dict[str, Any] | None = None
        for sec_all, sec_dist, entry in sec_entries:
            dist_overlap = len(distinctive & sec_dist)
            if dist_overlap < min_distinctive_overlap:
                continue
            # Score based on distinctive overlap / max distinctive set size
            denom = max(len(distinctive), len(sec_dist))
            score = dist_overlap / denom if denom > 0 else 0.0
            if score > best_score and score >= min_score:
                best_score = score
                best_entry = {**entry, "SEC_MATCH_TYPE": f"fuzzy_{best_score:.2f}"}
        return best_entry

    # Match
    results: list[dict[str, Any]] = []
    for _, row in inst.iterrows():
        norm = row["INST_NAME_NORM"]
        match = exact_lookup.get(norm)
        if not match:
            match = _best_fuzzy_match(norm)
        results.append(match or {"SEC_CIK": None, "SEC_TICKER": None, "SEC_TITLE": None, "SEC_MATCH_TYPE": None})

    match_df = pd.DataFrame(results, index=inst.index)
    return pd.concat([inst, match_df], axis=1)


def apply_overrides(
    mapping: pd.DataFrame,
    overrides: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Apply manual CERT → CIK/ticker overrides."""
    out = mapping.copy()
    for cert_str, override in overrides.items():
        cert = str(cert_str).strip()
        mask = out["CERT"].astype(str).str.strip() == cert
        if not mask.any():
            continue
        if "cik" in override:
            out.loc[mask, "SEC_CIK"] = override["cik"]
        if "ticker" in override:
            out.loc[mask, "SEC_TICKER"] = override["ticker"]
        out.loc[mask, "SEC_MATCH_TYPE"] = "override"
    return out


def build_sec_mapping(
    institutions: pd.DataFrame,
    tickers: list[dict[str, Any]],
    overrides: dict[str, dict[str, Any]] | None = None,
    nic_structure: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the full CERT → CIK → ticker mapping."""
    sec_lookup = build_sec_lookup(tickers)
    matched = match_institutions_to_sec(institutions, sec_lookup, nic_structure=nic_structure)

    if overrides:
        matched = apply_overrides(matched, overrides)

    # Select output columns
    keep = ["CERT"]
    for col in ["FED_RSSD", "RSSDID", "RSSDHCR", "NAME", "NAMEFULL",
                "SEC_CIK", "SEC_TICKER", "SEC_TITLE", "SEC_MATCH_TYPE"]:
        if col in matched.columns:
            keep.append(col)
    return matched[keep]


def run_mapping(args: argparse.Namespace) -> pd.DataFrame:
    import yaml
    institutions = pd.read_parquet(args.institutions)
    tickers = fetch_sec_tickers(cache_path=args.tickers_cache)
    overrides = None
    if args.overrides and args.overrides.exists():
        with open(args.overrides, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f).get("overrides", {})
    nic_structure = None
    if args.nic_structure and args.nic_structure.exists():
        nic_structure = pd.read_parquet(args.nic_structure)
    mapping = build_sec_mapping(institutions, tickers, overrides=overrides, nic_structure=nic_structure)
    save_table(mapping, args.out)
    matched = mapping["SEC_CIK"].notna().sum()
    print(f"Matched {matched}/{len(mapping)} institutions to SEC CIK. Saved to {args.out}", file=sys.stderr)
    return mapping


def main() -> None:
    args = parse_args()
    run_mapping(args)


if __name__ == "__main__":
    main()
