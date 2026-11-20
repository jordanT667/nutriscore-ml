"""Predict a Nutri-Score grade for one product, with the terms that drove it.

Uses the demo model (`demo_model.py`): a logistic regression on the product name, the
ingredient list and the food group only, the fields a visitor can type in. Its own test
score is in reports/results/demo_model.json.

Usage:
    python -m src.nutriscore.predict "sucre, farine de blé, huile de palme" --name "biscuits" --group "Biscuits and cakes"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .extract import GRADES, ROOT

MODEL = ROOT / "demo" / "model.joblib"
COLUMNS = ["ingredients_text", "product_name", "pnns_groups_2"]
PREFIXES = {"ingredients_words__": "ingredients: ", "name_words__": "name: ", "groups__pnns_groups_2_": "food group: "}


@dataclass
class Prediction:
    grade: str
    probabilities: dict[str, float]
    towards_grade: list[tuple[str, float]]  # terms pushing towards the predicted grade
    against_grade: list[tuple[str, float]]  # terms pushing away from it


@lru_cache(maxsize=1)
def load(path: Path = MODEL) -> dict:
    return joblib.load(path)


def food_groups() -> list[str]:
    return load()["food_groups"]


def readable(name: str) -> str:
    for prefix, label in PREFIXES.items():
        if name.startswith(prefix):
            return label + name[len(prefix):]
    return name


def predict(ingredients: str, name: str = "", group: str | None = None, top: int = 6) -> Prediction:
    bundle = load()
    features, model = bundle["features"], bundle["logreg"]
    row = pd.DataFrame([{"ingredients_text": ingredients, "product_name": name,
                         "pnns_groups_2": group if group else np.nan}])[COLUMNS]
    x = features.transform(row).tocsr()
    proba = model.predict_proba(x)[0]
    k = int(np.argmax(proba))

    # Contribution of each active feature to the chosen grade, relative to the average grade.
    names = features.get_feature_names_out()
    coef = model.coef_[k] - model.coef_.mean(axis=0)
    contrib = x.data * coef[x.indices]
    order = np.argsort(contrib)
    towards = [(readable(names[x.indices[i]]), float(contrib[i])) for i in order[::-1][:top] if contrib[i] > 0]
    against = [(readable(names[x.indices[i]]), float(contrib[i])) for i in order[:top] if contrib[i] < 0]
    return Prediction(GRADES[k].upper(), {g.upper(): float(p) for g, p in zip(GRADES, proba, strict=True)}, towards, against)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Estimate a Nutri-Score grade from an ingredient list.")
    parser.add_argument("ingredients")
    parser.add_argument("--name", default="")
    parser.add_argument("--group", default=None, help="Open Food Facts PNNS group, e.g. 'Biscuits and cakes'")
    args = parser.parse_args()
    p = predict(args.ingredients, args.name, args.group)
    print(f"Estimated grade: {p.grade}")
    print("  " + "  ".join(f"{g} {v:.0%}" for g, v in p.probabilities.items()))
    print("Pushes towards it:", ", ".join(t for t, _ in p.towards_grade))


if __name__ == "__main__":
    main()
