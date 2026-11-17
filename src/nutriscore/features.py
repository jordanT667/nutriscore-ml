"""Feature pipeline: everything a product page shows without its nutrition table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .extract import NUTRIENT_COLS

TAG_FEATURES = [
    "categories_tags",
    "ingredients_tags",
    "ingredients_analysis_tags",
    "additives_tags",
    "labels_tags",
    "food_groups_tags",
]
NUMERIC_FEATURES = ["nova_group", "additives_n", "n_ingredients"]
CATEGORICAL_FEATURES = ["pnns_groups_1", "pnns_groups_2"]
TEXT_FEATURES = ["ingredients_text", "product_name"]

# Columns the model must never see.
FORBIDDEN = set(NUTRIENT_COLS) | {"nutriscore_grade", "nutriscore_score", "group", "split", "code"}


def split_tags(text: str) -> list[str]:
    return [t for t in text.split(",") if t]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_ingredients"] = df["ingredients_tags"].map(lambda s: len(split_tags(s)))
    return df


def model_columns() -> list[str]:
    cols = TEXT_FEATURES + TAG_FEATURES + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert not FORBIDDEN & set(cols), "a forbidden column is used as a feature"
    return cols


def build_features(min_df: int = 5) -> ColumnTransformer:
    text = dict(
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
        min_df=min_df,
        dtype=np.float32,
    )
    transformers = [
        ("ingredients_words", TfidfVectorizer(ngram_range=(1, 2), max_features=150_000, **text), "ingredients_text"),
        ("name_words", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, **text), "product_name"),
    ]
    for col in TAG_FEATURES:
        transformers.append(
            (
                col,
                CountVectorizer(
                    tokenizer=split_tags,
                    token_pattern=None,
                    lowercase=False,
                    binary=True,
                    min_df=min_df,
                    dtype=np.float32,
                ),
                col,
            )
        )
    transformers.append(
        (
            "numeric",
            make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                StandardScaler(),
            ),
            NUMERIC_FEATURES,
        )
    )
    transformers.append(
        (
            "groups",
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value="unknown"),
                OneHotEncoder(handle_unknown="ignore", min_frequency=min_df),
            ),
            CATEGORICAL_FEATURES,
        )
    )
    return ColumnTransformer(transformers, sparse_threshold=1.0)
