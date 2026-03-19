"""Parse NIC structure bulk CSV files for holding-company relationships.

Reads the Relationships and Active Attributes CSV downloads from the FFIEC
NIC Data Download page and builds a bank → top holding company mapping.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

from bankfragility.tables import save_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relationships-zip", type=Path, required=True)
    parser.add_argument("--attributes-zip", type=Path, help="Optional: active attributes for HC names")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_relationships(zip_path: Path) -> pd.DataFrame:
    """Load NIC relationships CSV from zip, filter to current controlling relationships."""
    zf = zipfile.ZipFile(zip_path)
    csv_name = [n for n in zf.namelist() if n.upper().endswith(".CSV")][0]
    with zf.open(csv_name) as f:
        df = pd.read_csv(f, dtype=str, encoding="latin-1")

    # Clean column names (remove leading #)
    df.columns = [c.lstrip("#").strip() for c in df.columns]

    # Filter to current, controlling relationships
    df["DT_END"] = pd.to_numeric(df["DT_END"], errors="coerce")
    df["CTRL_IND"] = pd.to_numeric(df["CTRL_IND"], errors="coerce")
    df["RELN_LVL"] = pd.to_numeric(df["RELN_LVL"], errors="coerce")

    current = df[
        (df["DT_END"] >= 20240101) &  # still active
        (df["CTRL_IND"] == 1)          # controlling relationship
    ].copy()

    current["ID_RSSD_PARENT"] = pd.to_numeric(current["ID_RSSD_PARENT"], errors="coerce").astype("Int64")
    current["ID_RSSD_OFFSPRING"] = pd.to_numeric(current["ID_RSSD_OFFSPRING"], errors="coerce").astype("Int64")

    return current[["ID_RSSD_PARENT", "ID_RSSD_OFFSPRING", "RELN_LVL", "PCT_EQUITY"]].drop_duplicates()


def load_attributes(zip_path: Path) -> pd.DataFrame:
    """Load NIC active attributes for entity names."""
    zf = zipfile.ZipFile(zip_path)
    csv_name = [n for n in zf.namelist() if n.upper().endswith(".CSV")][0]
    with zf.open(csv_name) as f:
        # Read only needed columns to save memory
        df = pd.read_csv(f, dtype=str, encoding="latin-1", usecols=lambda c: c.lstrip("#").strip() in (
            "ID_RSSD", "NM_LGL", "NM_SHORT", "ENTITY_TYPE", "BHC_IND",
        ))

    df.columns = [c.lstrip("#").strip() for c in df.columns]
    df["ID_RSSD"] = pd.to_numeric(df["ID_RSSD"], errors="coerce").astype("Int64")
    # Clean name fields
    for col in ["NM_LGL", "NM_SHORT"]:
        if col in df.columns:
            df[col] = df[col].str.strip()
    return df.drop_duplicates(subset=["ID_RSSD"], keep="last")


def build_top_holder_map(relationships: pd.DataFrame) -> pd.DataFrame:
    """Walk the relationship tree to find the ultimate top holder for each entity.

    Returns a DataFrame with columns: ID_RSSD, TOP_HOLDER_RSSD.
    """
    # Build parent lookup: offspring → parent
    parent_map: dict[int, int] = {}
    for _, row in relationships.iterrows():
        child = row["ID_RSSD_OFFSPRING"]
        parent = row["ID_RSSD_PARENT"]
        if pd.notna(child) and pd.notna(parent):
            parent_map[int(child)] = int(parent)

    # Walk up the tree for each entity
    all_entities = set(parent_map.keys()) | set(parent_map.values())
    results: list[dict[str, int]] = []
    for entity in all_entities:
        current = entity
        visited: set[int] = set()
        while current in parent_map and current not in visited:
            visited.add(current)
            current = parent_map[current]
        results.append({"ID_RSSD": entity, "TOP_HOLDER_RSSD": current})

    return pd.DataFrame(results)


def build_nic_mapping(
    relationships_zip: Path,
    attributes_zip: Path | None = None,
) -> pd.DataFrame:
    """Build the full NIC structure mapping."""
    rels = load_relationships(relationships_zip)
    top_map = build_top_holder_map(rels)

    if attributes_zip and attributes_zip.exists():
        attrs = load_attributes(attributes_zip)
        # Add top holder name
        holder_names = attrs[["ID_RSSD", "NM_LGL"]].rename(
            columns={"ID_RSSD": "TOP_HOLDER_RSSD", "NM_LGL": "TOP_HOLDER_NAME"}
        )
        top_map = top_map.merge(holder_names, on="TOP_HOLDER_RSSD", how="left")

    return top_map


def run_nic_mapping(args: argparse.Namespace) -> pd.DataFrame:
    mapping = build_nic_mapping(args.relationships_zip, args.attributes_zip)
    save_table(mapping, args.out)
    top_holders = mapping["TOP_HOLDER_RSSD"].nunique()
    print(
        f"Built NIC structure: {len(mapping):,} entities → {top_holders:,} top holders. Saved to {args.out}",
        file=sys.stderr,
    )
    return mapping


def main() -> None:
    args = parse_args()
    run_nic_mapping(args)


if __name__ == "__main__":
    main()
