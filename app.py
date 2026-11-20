"""Streamlit demo: estimate a Nutri-Score grade from an ingredient list.

Run locally:
    streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st

from src.nutriscore.predict import food_groups, predict

COLORS = {"A": "#038141", "B": "#85BB2F", "C": "#FECB02", "D": "#EE8100", "E": "#E63E11"}
DEMO_RESULTS = Path(__file__).parent / "reports" / "results" / "demo_model.json"

st.set_page_config(page_title="Nutri-Score estimator", page_icon="🥗", layout="centered")
st.title("Nutri-Score estimator")
st.write(
    "Estimates the Nutri-Score of a food product **without its nutrition table**, from its name, "
    "ingredient list and food group. Ingredient lists in French work best: the model was trained on "
    "French products from Open Food Facts."
)

with st.form("product"):
    name = st.text_input("Product name", "Biscuits fourrés au chocolat")
    ingredients = st.text_area(
        "Ingredient list (as printed on the pack)",
        "Farine de blé, sucre, huile de palme, cacao maigre en poudre 4,5 %, sirop de glucose, sel, émulsifiant : lécithine de soja",
        height=120,
    )
    group = st.selectbox("Food group", ["(not sure)"] + food_groups())
    submitted = st.form_submit_button("Estimate the grade")

if submitted and ingredients.strip():
    p = predict(ingredients, name, None if group == "(not sure)" else group)
    cols = st.columns(5)
    for col, (grade, prob) in zip(cols, p.probabilities.items(), strict=True):
        chosen = grade == p.grade
        col.markdown(
            f"<div style='text-align:center;padding:10px;border-radius:10px;background:{COLORS[grade]};"
            f"color:white;font-weight:700;font-size:{'34px' if chosen else '20px'};"
            f"opacity:{1 if chosen else 0.45}'>{grade}</div>"
            f"<div style='text-align:center;margin-top:4px'>{prob:.0%}</div>",
            unsafe_allow_html=True,
        )
    st.subheader(f"Estimated grade: {p.grade}")
    left, right = st.columns(2)
    left.markdown("**Pushes towards it**")
    left.write("\n".join(f"- {term}" for term, _ in p.towards_grade) or "—")
    right.markdown("**Pushes away from it**")
    right.write("\n".join(f"- {term}" for term, _ in p.against_grade) or "—")

st.divider()
if DEMO_RESULTS.exists():
    test = json.loads(DEMO_RESULTS.read_text(encoding="utf-8"))["test"]
    st.caption(
        f"This demo model reads only the three fields above. On held-out brands it reaches a macro-F1 of "
        f"{test['macro_f1']:.2f} and is within one grade for {test['within_one_grade']:.0%} of products. "
        "The full model of the project also reads Open Food Facts tags and scores higher."
    )
st.caption(
    "An estimate, not an official Nutri-Score. Data: Open Food Facts (ODbL). "
    "Code: [github.com/jordanT667/nutriscore-ml](https://github.com/jordanT667/nutriscore-ml)"
)
