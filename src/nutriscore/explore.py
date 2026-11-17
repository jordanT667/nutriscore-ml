"""Descriptive statistics of the prepared table (no modelling decision depends on them).

Usage:
    python -m src.nutriscore.explore
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .extract import GRADES, ROOT
from .prepare import PREPARED

RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "reports" / "figures"
COLORS = ["#038141", "#85bb2f", "#fecb02", "#ee8100", "#e63e11"]  # Nutri-Score palette


def main() -> None:
    df = pd.read_parquet(PREPARED)
    family = df["pnns_groups_1"].fillna("unknown")
    share = pd.crosstab(family, df["nutriscore_grade"], normalize="index")[GRADES]
    sizes = family.value_counts()
    share = share.loc[sizes[sizes >= 1000].index]
    share = share.sort_values(["a", "b"], ascending=False)

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(share) + 1.2))
    left = pd.Series(0.0, index=share.index)
    for grade, color in zip(GRADES, COLORS):
        ax.barh(share.index, share[grade], left=left, color=color, label=grade.upper())
        left += share[grade]
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of products")
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "grades_by_family.png", dpi=150)
    plt.close(fig)

    n_ing = df["ingredients_tags"].fillna("").map(lambda s: len([t for t in s.split(",") if t]))
    stats = {
        "products": len(df),
        "brands_or_groups": int(df["group"].nunique()),
        "products_without_brand": int(df["group"].str.startswith("nobrand:").sum()),
        "grade_share": df["nutriscore_grade"].value_counts(normalize=True).sort_index().round(4).to_dict(),
        "products_per_family": sizes.to_dict(),
        "ingredients_per_product": {
            "median": float(n_ing.median()),
            "p90": float(n_ing.quantile(0.9)),
        },
        "missing_share": {
            c: round(float((df[c].fillna("") == "").mean()), 4)
            for c in ["product_name", "categories_tags", "ingredients_tags", "labels_tags", "additives_tags"]
        },
        "nova_group_missing_share": round(float(df["nova_group"].isna().mean()), 4),
    }
    (RESULTS / "exploration.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
