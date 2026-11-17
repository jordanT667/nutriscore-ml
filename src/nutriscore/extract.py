"""Extract the modelling table from the Open Food Facts CSV export.

The export is a ~1.5 GB gzipped TSV (~10 GB uncompressed), so it is streamed
in chunks and only the needed columns are kept.

Usage:
    python -m src.nutriscore.extract
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "en.openfoodfacts.org.products.csv.gz"
OUT = ROOT / "data" / "interim" / "products_fr.parquet"
STATS = ROOT / "reports" / "results" / "extraction.json"

GRADES = ["a", "b", "c", "d", "e"]

# Inputs the model may use: all of them are available without a nutrition table.
TEXT_COLS = ["product_name", "ingredients_text"]
TAG_COLS = [
    "categories_tags",
    "ingredients_tags",
    "ingredients_analysis_tags",
    "additives_tags",
    "labels_tags",
    "food_groups_tags",
]
OTHER_COLS = ["pnns_groups_1", "pnns_groups_2", "nova_group", "additives_n"]

# Never used as model inputs: label, grouping key, filters, and the nutrition
# table (kept only for a reference model that shows the "with table" ceiling).
META_COLS = ["code", "brands_tags", "countries_tags", "created_t", "nutriscore_grade"]
NUTRIENT_COLS = [
    "energy-kcal_100g",
    "fat_100g",
    "saturated-fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
    "fruits-vegetables-legumes_100g",
]

USECOLS = META_COLS + TEXT_COLS + TAG_COLS + OTHER_COLS + NUTRIENT_COLS


def keep(chunk: pd.DataFrame) -> pd.DataFrame:
    """Rows sold in France, with a Nutri-Score grade and an ingredient list."""
    grade = chunk["nutriscore_grade"].str.lower()
    mask = (
        grade.isin(GRADES)
        & chunk["countries_tags"].fillna("").str.contains("en:france", regex=False)
        & chunk["ingredients_text"].fillna("").str.strip().str.len().gt(0)
    )
    out = chunk.loc[mask].copy()
    out["nutriscore_grade"] = grade[mask]
    return out


def main(chunksize: int = 200_000) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    STATS.parent.mkdir(parents=True, exist_ok=True)
    reader = pd.read_csv(
        RAW,
        sep="\t",
        usecols=USECOLS,
        dtype=str,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
        chunksize=chunksize,
        compression="gzip",
        encoding="utf-8",
        encoding_errors="replace",
    )
    parts, n_rows = [], 0
    for i, chunk in enumerate(reader):
        n_rows += len(chunk)
        parts.append(keep(chunk))
        if i % 10 == 0:
            print(f"chunk {i}: {n_rows:,} rows read, {sum(map(len, parts)):,} kept")

    df = pd.concat(parts, ignore_index=True)
    for col in NUTRIENT_COLS + ["nova_group", "additives_n", "created_t"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_parquet(OUT, index=False)

    stats = {
        "source": "Open Food Facts CSV export (en.openfoodfacts.org.products.csv.gz)",
        "rows_read": n_rows,
        "rows_kept": len(df),
        "filter": "nutriscore_grade in a-e, countries_tags contains en:france, non-empty ingredients_text",
        "grade_counts": df["nutriscore_grade"].value_counts().sort_index().to_dict(),
    }
    STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
