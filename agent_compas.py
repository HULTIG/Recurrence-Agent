import joblib
import shap
import pandas as pd

from agent_common import build_dummy_to_readable_map, encode_raw_profile, explain_contributions

ARTIFACTS_DIR = "results/baseline_compas"
MODEL_PATH = f"{ARTIFACTS_DIR}/xgboost_model.pkl"
X_TEST_PATH = f"{ARTIFACTS_DIR}/X_test.csv"

CATEGORICAL_COLUMNS = ["sex", "age_cat", "c_charge_degree"]

FEATURE_GROUPS = {
    "faixa_etaria": ["age", "age_cat_Greater than 45", "age_cat_Less than 25"],
}

_model = joblib.load(MODEL_PATH)
_reference_columns = pd.read_csv(X_TEST_PATH, nrows=0).columns.tolist()
_dummy_map = build_dummy_to_readable_map(_reference_columns, CATEGORICAL_COLUMNS)
_explainer = shap.TreeExplainer(_model)


def predict_and_explain(raw_profile: dict, top_n: int = 4) -> dict:
    X_row = encode_raw_profile(raw_profile, _reference_columns, CATEGORICAL_COLUMNS)

    pred = int(_model.predict(X_row)[0])
    proba = float(_model.predict_proba(X_row)[0][1])

    contributions = pd.Series(_explainer(X_row).values[0], index=_reference_columns)
    explicacao = explain_contributions(contributions, X_row.iloc[0], _dummy_map, FEATURE_GROUPS, top_n)

    return {
        "baseline_pred": pred,
        "baseline_proba": round(proba, 4),
        "explicacao": explicacao,
    }


if __name__ == "__main__":
    exemplo_raw = pd.read_csv("clean_data/compas-scores-two-years.csv").iloc[0].to_dict()
    resultado = predict_and_explain(exemplo_raw)
    print(resultado)