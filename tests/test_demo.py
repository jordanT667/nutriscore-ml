import pytest

from src.nutriscore.predict import MODEL, food_groups, predict, readable

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="demo model not built")


def test_prediction_is_a_distribution_over_grades():
    p = predict("eau minérale naturelle", "eau de source", "Waters and flavored waters")
    assert p.grade in "ABCDE"
    assert set(p.probabilities) == set("ABCDE")
    assert sum(p.probabilities.values()) == pytest.approx(1.0)


def test_clear_cases_go_the_right_way():
    sweet = predict("sucre, farine de blé, huile de palme, chocolat, sirop de glucose",
                    "biscuits fourrés chocolat", "Biscuits and cakes")
    veg = predict("épinards, eau, sel", "épinards en branches", "Vegetables")
    assert sweet.grade in "DE" and veg.grade in "AB"


def test_explanations_name_input_terms():
    p = predict("sucre, farine de blé, huile de palme", "biscuits", "Biscuits and cakes")
    assert any("sucre" in term for term, _ in p.towards_grade)


def test_food_groups_are_listed():
    assert "Vegetables" in food_groups()


def test_streamlit_app_runs_and_answers():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert any("Estimated grade" in h.value for h in app.subheader)


def test_name_and_group_are_optional():
    p = predict("lait entier, sucre, cacao")
    assert p.grade in "ABCDE"
    assert sum(p.probabilities.values()) == pytest.approx(1.0)


def test_explanations_have_the_right_sign_and_size():
    p = predict("sucre, farine de blé, huile de palme, sel", "biscuits", "Biscuits and cakes", top=3)
    assert len(p.towards_grade) <= 3 and len(p.against_grade) <= 3
    assert all(w > 0 for _, w in p.towards_grade)
    assert all(w < 0 for _, w in p.against_grade)


def test_feature_names_are_made_readable():
    assert readable("ingredients_words__sucre") == "ingredients: sucre"
    assert readable("groups__pnns_groups_2_Vegetables") == "food group: Vegetables"
    assert readable("other") == "other"
