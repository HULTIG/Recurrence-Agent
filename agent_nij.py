import joblib
import shap
import pandas as pd

from agent_common import build_dummy_to_readable_map, encode_raw_profile, explain_contributions

ARTIFACTS_DIR = "results/baseline_nij"
MODEL_PATH = f"{ARTIFACTS_DIR}/xgboost_model.pkl"
X_TEST_PATH = f"{ARTIFACTS_DIR}/X_test.csv"

CATEGORICAL_COLUMNS = [
    "gender", "supervision_level_first", "education_level",
    "prison_offense", "residence_puma",
]

FEATURE_GROUPS = {
    "situacao_de_emprego": ["jobs_per_year", "percent_days_employed"],
    "historico_crimes_propriedade": ["prior_arrest_episodes_property", "prior_conviction_episodes_prop"],
    "historico_crimes_misdemeanor": ["prior_arrest_episodes_misd", "prior_conviction_episodes_misd"],
    "historico_crimes_droga": ["prior_arrest_episodes_drug", "prior_conviction_episodes_drug"],
}

_model = joblib.load(MODEL_PATH)
_reference_columns = pd.read_csv(X_TEST_PATH, nrows=0).columns.tolist()
_dummy_map = build_dummy_to_readable_map(_reference_columns, CATEGORICAL_COLUMNS)
_explainer = shap.TreeExplainer(_model)


def predict_and_explain(raw_profile: dict, top_n: int = 4) -> dict:
    """
    raw_profile: dict com os nomes ORIGINAIS das features do NIJ
    (ex: {"age_at_release": 34, "gender": "M", "supervision_level_first": "Standard", ...})

    Devolve: {"baseline_pred": 0/1, "baseline_proba": float, "explicacao": str}

    NOTA: baseline_pred e do XGBoost, nao do LLM fine-tuned. Ver agent_llm_nij.py
    (precisa de GPU) para a previsao "oficial" do agente.
    """
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
    # exemplo: usa a primeira linha do X_test como perfil de teste
    # (em producao, o raw_profile viria de um formulario/API com dados novos)
    exemplo_raw = pd.read_csv("clean_data/nij-challenge2021.csv").iloc[0].to_dict()
    resultado = predict_and_explain(exemplo_raw)
    print(resultado)