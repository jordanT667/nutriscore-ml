"""Clean the extracted table and build a brand-grouped train / validation / test split.

Usage:
    python -m src.nutriscore.prepare
"""

from __future__ import annotations

import json
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .extract import OUT as EXTRACTED
from .extract import ROOT, TAG_COLS

SEED = 42
PREPARED = ROOT / "data" / "interim" / "products_fr_split.parquet"
STATS = ROOT / "reports" / "results" / "preparation.json"


def normalize(text: str) -> str:
    """Lower case, no accents, single spaces: used for deduplication only."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def drop_leaky_tags(tags: str) -> str:
    """Remove tags that state the grade itself, such as en:nutriscore-grade-a."""
    if not isinstance(tags, str) or not tags:
        return ""
    return ",".join(t for t in tags.split(",") if "nutri" not in t and "nutrition" not in t)


def group_key(df: pd.DataFrame) -> pd.Series:
    """First brand tag; products without a brand form their own group."""
    brand = df["brands_tags"].fillna("").str.split(",").str[0].str.strip()
    return brand.where(brand != "", "nobrand:" + df["code"].astype(str))


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {"rows_in": len(df)}
    df = df.copy()
    for col in ["product_name", "ingredients_text"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    for col in TAG_COLS:
        df[col] = df[col].fillna("")
    df["labels_tags"] = df["labels_tags"].map(drop_leaky_tags)
    df["categories_tags"] = df["categories_tags"].map(drop_leaky_tags)

    # One row per product code, then one row per (name, ingredients) recipe.
    df = df.drop_duplicates("code")
    stats["rows_after_code_dedup"] = len(df)
    key = df["product_name"].map(normalize) + "|" + df["ingredients_text"].map(normalize)
    df = df.loc[~key.duplicated()].copy()
    stats["rows_after_recipe_dedup"] = len(df)

    df["group"] = group_key(df)
    return df.reset_index(drop=True), stats


def split(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15) -> pd.Series:
    """Assign train / val / test so that no brand appears in two splits."""
    idx = np.arange(len(df))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    rest, test = next(outer.split(idx, groups=df["group"]))
    inner = GroupShuffleSplit(
        n_splits=1, test_size=val_size / (1 - test_size), random_state=SEED
    )
    _, va = next(inner.split(rest, groups=df["group"].iloc[rest]))
    out = pd.Series("train", index=df.index)
    out.iloc[rest[va]] = "val"
    out.iloc[test] = "test"
    return out


def main() -> None:
    df, stats = clean(pd.read_parquet(EXTRACTED))
    df["split"] = split(df)
    df.to_parquet(PREPARED, index=False)
    stats["groups"] = int(df["group"].nunique())
    stats["split_sizes"] = df["split"].value_counts().to_dict()
    stats["grade_share_by_split"] = (
        pd.crosstab(df["split"], df["nutriscore_grade"], normalize="index").round(4).to_dict("index")
    )
    STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
