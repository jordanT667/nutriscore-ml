"""Train, select on validation, and evaluate once on test.

Usage:
    python -m src.nutriscore.train
"""

from __future__ import annotations

import json
import time

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from .extract import GRADES, NUTRIENT_COLS, ROOT
from .features import add_derived, build_features, model_columns
from .prepare import PREPARED, SEED

RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
N_BOOT = 1000


# ---------------------------------------------------------------- metrics
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    gap = np.abs(y_true - y_pred)
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "within_one_grade": float(np.mean(gap <= 1)),
        "mean_grade_error": float(np.mean(gap)),
        "quadratic_kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }


def group_bootstrap(y, preds: dict, groups, n=N_BOOT, seed=SEED) -> dict:
    """95% intervals by resampling brands (the unit of the split), not products."""
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(codes))]
    draws = {name: [] for name in preds}
    for _ in range(n):
        pick = rng.integers(0, len(codes), len(codes))
        idx = np.concatenate([members[g] for g in pick])
        for name, p in preds.items():
            draws[name].append(f1_score(y[idx], p[idx], average="macro"))
    return {k: np.array(v) for k, v in draws.items()}


# ---------------------------------------------------------------- baselines
class MajorityBaseline:
    def fit(self, df, y):
        self.value = np.bincount(y).argmax()
        return self

    def predict(self, df):
        return np.full(len(df), self.value)


class CategoryBaseline:
    """Most frequent grade of the product's most specific known category.

    Falls back to the PNNS group, then to the overall majority.
    """

    def __init__(self, min_count: int = 20):
        self.min_count = min_count

    @staticmethod
    def _specific(tags: str) -> list[str]:
        return [t for t in reversed(tags.split(",")) if t]

    def fit(self, df, y):
        self.default = np.bincount(y).argmax()
        frame = pd.DataFrame({"y": y, "pnns": df["pnns_groups_2"].fillna("unknown").values})
        self.pnns = frame.groupby("pnns")["y"].agg(lambda s: s.value_counts().idxmax()).to_dict()
        rows = [
            (t, label)
            for tags, label in zip(df["categories_tags"].values, y)
            for t in self._specific(tags)
        ]
        cat = pd.DataFrame(rows, columns=["tag", "y"])
        counts = cat.groupby("tag")["y"].agg(["size", lambda s: s.value_counts().idxmax()])
        counts.columns = ["size", "mode"]
        self.category = counts.loc[counts["size"] >= self.min_count, "mode"].to_dict()
        return self

    def predict(self, df):
        out = []
        for tags, pnns in zip(df["categories_tags"].values, df["pnns_groups_2"].fillna("unknown").values):
            label = next((self.category[t] for t in self._specific(tags) if t in self.category), None)
            if label is None:
                label = self.pnns.get(pnns, self.default)
            out.append(label)
        return np.array(out)


# ---------------------------------------------------------------- figures
def plot_confusion(y, p, title, path):
    cm = confusion_matrix(y, p, labels=range(5), normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
    labels = [g.upper() for g in GRADES]
    ax.set_xticks(range(5), labels)
    ax.set_yticks(range(5), labels)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scores(table: pd.DataFrame, path):
    t = table.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    err = np.vstack([t["macro_f1"] - t["macro_f1_low"], t["macro_f1_high"] - t["macro_f1"]])
    colors = ["#9aa5b1" if r else "#1f6feb" for r in t["reference_only"]]
    ax.barh(t["model"], t["macro_f1"], xerr=err, color=colors, capsize=3)
    ax.set_xlabel("Macro-F1 on test (95% brand bootstrap interval)")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


PREFIXES = {
    "ingredients_words__": "ingredients: ",
    "name_words__": "name: ",
    "categories_tags__": "category: ",
    "groups__pnns_groups_2_": "food group: ",
    "groups__pnns_groups_1_": "food family: ",
    "labels_tags__": "label: ",
    "additives_tags__": "additive: ",
    "ingredients_tags__": "ingredient tag: ",
    "ingredients_analysis_tags__": "analysis: ",
    "food_groups_tags__": "food group: ",
}


def readable(name: str) -> str:
    for prefix, label in PREFIXES.items():
        if name.startswith(prefix):
            name = label + name[len(prefix):]
            break
    return name.replace("en:", "")


def plot_top_terms(model, names, path, k=12):
    coef = model.coef_
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, cls, color in [(axes[0], 0, "#038141"), (axes[1], 4, "#e63e11")]:
        top = np.argsort(coef[cls])[-k:]
        ax.barh([readable(names[i]) for i in top], coef[cls][top], color=color)
        ax.set_title(f"Pushes towards grade {GRADES[cls].upper()}")
        ax.set_xlabel("Logistic regression coefficient")
        ax.tick_params(axis="y", labelsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    for d in (RESULTS, FIGURES, MODELS):
        d.mkdir(parents=True, exist_ok=True)
    df = add_derived(pd.read_parquet(PREPARED))
    y_all = df["nutriscore_grade"].map({g: i for i, g in enumerate(GRADES)}).to_numpy()
    parts = {s: df.index[df["split"] == s] for s in ("train", "val", "test")}
    X = {s: df.loc[i, model_columns()] for s, i in parts.items()}
    y = {s: y_all[i] for s, i in parts.items()}
    log = {"sizes": {s: int(len(i)) for s, i in parts.items()}, "validation": {}}

    models, val_scores = {}, {}

    def record(name, fitted, val_pred):
        models[name] = fitted
        val_scores[name] = metrics(y["val"], val_pred)
        print(f"{name:40s} val macro-F1 {val_scores[name]['macro_f1']:.4f}")

    # Baselines
    for name, base in [("Majority class", MajorityBaseline()), ("Category lookup", CategoryBaseline())]:
        base.fit(df.loc[parts["train"]], y["train"])
        record(name, base, base.predict(df.loc[parts["val"]]))

    # Shared sparse features, fitted on train only
    t0 = time.time()
    feats = build_features().fit(X["train"])
    M = {s: feats.transform(X[s]).tocsr() for s in X}
    log["n_features"] = int(M["train"].shape[1])
    print(f"features: {M['train'].shape} in {time.time() - t0:.0f}s")

    # Logistic regression: small grid on validation
    grid = [(c, w) for c in (0.5, 2.0, 8.0) for w in (None, "balanced")]
    lr_runs = {}
    for c, w in grid:
        m = LogisticRegression(C=c, class_weight=w, max_iter=2000, random_state=SEED)
        m.fit(M["train"], y["train"])
        score = metrics(y["val"], m.predict(M["val"]))["macro_f1"]
        lr_runs[f"C={c}, class_weight={w}"] = score
        print(f"  LR C={c} w={w}: {score:.4f}")
        if score >= max(lr_runs.values()):
            best_lr = m
    log["validation"]["logistic_regression_grid"] = lr_runs
    record("Logistic regression (text + tags)", best_lr, best_lr.predict(M["val"]))

    # LightGBM on the same features, early stopping on validation
    params = dict(
        objective="multiclass", num_class=5, learning_rate=0.1, num_leaves=127,
        min_child_samples=20, feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=1.0, seed=SEED, verbose=-1, num_threads=0,
    )
    booster = lgb.train(
        params,
        lgb.Dataset(M["train"], y["train"]),
        num_boost_round=3000,
        valid_sets=[lgb.Dataset(M["val"], y["val"])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
    )
    log["validation"]["lightgbm_best_iteration"] = int(booster.best_iteration)
    gbm = lambda m: booster.predict(m, num_iteration=booster.best_iteration).argmax(1)  # noqa: E731
    record("LightGBM (text + tags)", booster, gbm(M["val"]))

    # Reference only: what the nutrition table gives (not the task)
    def nutr(s):
        frame = df.loc[parts[s], NUTRIENT_COLS + ["pnns_groups_2"]].copy()
        frame["pnns_groups_2"] = frame["pnns_groups_2"].astype("category")
        return frame

    ref_train = nutr("train")
    cats = ref_train["pnns_groups_2"].cat.categories
    def nutr_aligned(s):
        f = nutr(s)
        f["pnns_groups_2"] = pd.Categorical(f["pnns_groups_2"], categories=cats)
        return f
    ref = lgb.train(
        {**params, "feature_fraction": 1.0, "num_leaves": 63},
        lgb.Dataset(ref_train, y["train"]),
        num_boost_round=3000,
        valid_sets=[lgb.Dataset(nutr_aligned("val"), y["val"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    ref_pred = lambda s: ref.predict(nutr_aligned(s), num_iteration=ref.best_iteration).argmax(1)  # noqa: E731
    record("Reference: nutrition table", ref, ref_pred("val"))

    # ------------------------------------------------ selection, then one test pass
    candidates = ["Category lookup", "Logistic regression (text + tags)", "LightGBM (text + tags)"]
    best = max(candidates, key=lambda n: val_scores[n]["macro_f1"])
    log["selected_on_validation"] = best

    test_pred = {
        "Majority class": models["Majority class"].predict(df.loc[parts["test"]]),
        "Category lookup": models["Category lookup"].predict(df.loc[parts["test"]]),
        "Logistic regression (text + tags)": best_lr.predict(M["test"]),
        "LightGBM (text + tags)": gbm(M["test"]),
        "Reference: nutrition table": ref_pred("test"),
    }
    groups = df.loc[parts["test"], "group"].to_numpy()
    boot = group_bootstrap(y["test"], test_pred, groups)

    rows = []
    for name, p in test_pred.items():
        lo, hi = np.percentile(boot[name], [2.5, 97.5])
        rows.append({
            "model": name, **metrics(y["test"], p),
            "macro_f1_low": lo, "macro_f1_high": hi,
            "val_macro_f1": val_scores[name]["macro_f1"],
            "reference_only": name.startswith("Reference"),
        })
    table = pd.DataFrame(rows)
    table.round(4).to_csv(RESULTS / "test_metrics.csv", index=False)

    diff = boot[best] - boot["Category lookup"]
    log["test_gain_over_category_lookup"] = {
        "model": best,
        "macro_f1_gain": float(np.mean(diff)),
        "interval_95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
    }

    # Per food family, for the selected model
    fam = df.loc[parts["test"], "pnns_groups_1"].fillna("unknown").to_numpy()
    per_family = []
    for f in pd.Series(fam).value_counts().index:
        mask = fam == f
        if mask.sum() >= 200:
            per_family.append({"pnns_groups_1": f, "n": int(mask.sum()),
                               **metrics(y["test"][mask], test_pred[best][mask])})
    pd.DataFrame(per_family).round(4).to_csv(RESULTS / "test_by_family.csv", index=False)

    log["validation"]["scores"] = val_scores
    (RESULTS / "training_log.json").write_text(json.dumps(log, indent=2, default=float), encoding="utf-8")

    plot_scores(table, FIGURES / "test_macro_f1.png")
    plot_confusion(y["test"], test_pred[best], f"{best}, test set", FIGURES / "confusion_best.png")
    plot_confusion(y["test"], test_pred["Category lookup"], "Category lookup, test set",
                   FIGURES / "confusion_category.png")
    plot_top_terms(best_lr, feats.get_feature_names_out(), FIGURES / "top_terms.png")

    joblib.dump({"features": feats, "logreg": best_lr}, MODELS / "logreg.joblib")
    booster.save_model(str(MODELS / "lightgbm.txt"))
    print(table.round(4).to_string(index=False))
    print(json.dumps(log["test_gain_over_category_lookup"], indent=2))


if __name__ == "__main__":
    main()
