"""Train the model behind the demo: only the fields a person can type in.

The main models also read Open Food Facts tags (categories, ingredient and additive tags,
NOVA group), which a visitor cannot provide. This model uses product name, ingredient list
and food group only, is tuned on the validation set and evaluated once on the same test set.

Usage:
    python -m src.nutriscore.demo_model
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from .extract import GRADES, ROOT
from .prepare import PREPARED, SEED
from .train import metrics

DEMO_COLUMNS = ["ingredients_text", "product_name", "pnns_groups_2"]
OUT = ROOT / "demo" / "model.joblib"
RESULTS = ROOT / "reports" / "results" / "demo_model.json"


def build_features(min_df: int = 5) -> ColumnTransformer:
    text = dict(strip_accents="unicode", lowercase=True, sublinear_tf=True, min_df=min_df, dtype=np.float32)
    return ColumnTransformer([
        ("ingredients_words", TfidfVectorizer(ngram_range=(1, 2), max_features=150_000, **text), "ingredients_text"),
        ("name_words", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, **text), "product_name"),
        ("groups", make_pipeline(SimpleImputer(strategy="constant", fill_value="unknown"),
                                 OneHotEncoder(handle_unknown="ignore", min_frequency=min_df)), ["pnns_groups_2"]),
    ], sparse_threshold=1.0)


def main() -> None:
    df = pd.read_parquet(PREPARED)
    y_all = df["nutriscore_grade"].map({g: i for i, g in enumerate(GRADES)}).to_numpy()
    parts = {s: df.index[df["split"] == s] for s in ("train", "val", "test")}
    X = {s: df.loc[i, DEMO_COLUMNS] for s, i in parts.items()}
    y = {s: y_all[i] for s, i in parts.items()}

    features = build_features().fit(X["train"])
    M = {s: features.transform(X[s]) for s in X}
    grid = {}
    best = None
    for c in (0.5, 2.0):
        model = LogisticRegression(C=c, max_iter=2000, random_state=SEED).fit(M["train"], y["train"])
        grid[f"C={c}"] = metrics(y["val"], model.predict(M["val"]))["macro_f1"]
        print(f"C={c}: val macro-F1 {grid[f'C={c}']:.4f}")
        if best is None or grid[f"C={c}"] > max(v for k, v in grid.items() if k != f"C={c}"):
            best = model
    test = metrics(y["test"], best.predict(M["test"]))
    groups = df.loc[parts["train"]].dropna(subset=["pnns_groups_2"])
    result = {
        "inputs": DEMO_COLUMNS,
        "validation_grid": grid,
        "test": {k: round(float(v), 4) for k, v in test.items()},
        "note": "Demo model only; the reported project results use all fields (see test_metrics.csv).",
    }
    RESULTS.write_text(json.dumps(result, indent=2), encoding="utf-8")
    OUT.parent.mkdir(exist_ok=True)
    joblib.dump({"features": features, "logreg": best,
                 "food_groups": sorted(groups["pnns_groups_2"].unique().tolist())}, OUT, compress=("xz", 6))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
