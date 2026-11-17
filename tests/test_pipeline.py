import numpy as np
import pandas as pd

from src.nutriscore.extract import NUTRIENT_COLS
from src.nutriscore.features import add_derived, build_features, model_columns
from src.nutriscore.prepare import clean, drop_leaky_tags, normalize, split
from src.nutriscore.train import CategoryBaseline, metrics


def toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    brands = [f"brand-{i}" for i in range(40)]
    words = ["sucre", "farine de ble", "huile de palme", "eau", "tomate", "sel", "lait"]
    rows = []
    for i in range(n):
        ing = ", ".join(rng.choice(words, 3, replace=False))
        rows.append({
            "code": str(i),
            "brands_tags": rng.choice(brands) if i % 10 else "",
            "countries_tags": "en:france",
            "created_t": 0,
            "nutriscore_grade": rng.choice(list("abcde")),
            "product_name": f"produit {i}",
            "ingredients_text": ing,
            "categories_tags": "en:snacks,en:biscuits" if "sucre" in ing else "en:meals",
            "ingredients_tags": ",".join("en:" + w.replace(" ", "-") for w in ing.split(", ")),
            "ingredients_analysis_tags": "en:vegan" if "lait" not in ing else "en:non-vegan",
            "additives_tags": "en:e330" if i % 2 else "",
            "labels_tags": "en:organic,en:nutriscore-grade-a" if i % 3 == 0 else "",
            "food_groups_tags": "en:sugary-snacks" if "sucre" in ing else "en:composite-foods",
            "pnns_groups_1": "Sugary snacks" if "sucre" in ing else "Composite foods",
            "pnns_groups_2": "Biscuits" if "sucre" in ing else "One-dish meals",
            "nova_group": rng.choice([1, 4, np.nan]),
            "additives_n": rng.integers(0, 5),
            **{c: rng.random() for c in NUTRIENT_COLS},
        })
    return pd.DataFrame(rows)


def test_grade_tags_are_removed():
    assert drop_leaky_tags("en:organic,en:nutriscore-grade-a,en:fr:nutriscore") == "en:organic"
    df, _ = clean(toy())
    assert not df["labels_tags"].str.contains("nutri").any()


def test_normalize_ignores_case_accents_and_punctuation():
    assert normalize("Sucre, Farine de BLÉ !") == normalize("sucre farine de ble")


def test_split_keeps_each_brand_in_one_split():
    df, _ = clean(toy())
    df["split"] = split(df)
    assert set(df["split"]) == {"train", "val", "test"}
    assert (df.groupby("group")["split"].nunique() == 1).all()


def test_no_nutrient_or_label_reaches_the_model():
    cols = model_columns()
    assert not set(cols) & set(NUTRIENT_COLS)
    assert "nutriscore_grade" not in cols
    df, _ = clean(toy())
    feats = build_features(min_df=1).fit(add_derived(df)[cols])
    names = feats.get_feature_names_out()
    assert not any("nutri" in n for n in names)


def test_category_baseline_backs_off_to_pnns_group():
    df, _ = clean(toy())
    y = np.where(df["pnns_groups_2"] == "Biscuits", 4, 1)
    base = CategoryBaseline(min_count=1).fit(df, y)
    new = df.head(2).copy()
    new["categories_tags"] = "en:never-seen"
    assert list(base.predict(new)) == list(np.where(new["pnns_groups_2"] == "Biscuits", 4, 1))


def test_metrics_on_perfect_prediction():
    y = np.array([0, 1, 2, 3, 4])
    m = metrics(y, y)
    assert m["macro_f1"] == 1 and m["within_one_grade"] == 1 and m["mean_grade_error"] == 0
