import joblib
import shap
import pandas as pd

ARTIFACTS_DIR = "results/baseline_compas"
MODEL_PATH = f"{ARTIFACTS_DIR}/xgboost_model.pkl"
X_TEST_PATH = f"{ARTIFACTS_DIR}/X_test.csv"
TEST_META_PATH = f"{ARTIFACTS_DIR}/test_meta.csv"
OUT_EXPLANATIONS_CSV = f"{ARTIFACTS_DIR}/explanations_compas.csv"

CATEGORICAL_COLUMNS = [
    "sex", "age_cat", "c_charge_degree",
]

FEATURE_GROUPS = {
    "faixa_etaria": ["age", "age_cat_Greater than 45", "age_cat_Less than 25"],
}

TOP_N = 4


def build_dummy_to_readable_map(columns, categorical_columns):
    mapping = {}
    for col in columns:
        matched = False
        for cat in categorical_columns:
            prefix = f"{cat}_"
            if col.startswith(prefix):
                valor = col[len(prefix):]
                mapping[col] = (cat, valor)
                matched = True
                break
        if not matched:
            mapping[col] = (col, None)
    return mapping


def merge_correlated_shap(contributions, row, groups):
    merged_contrib = contributions.copy()
    merged_values = {}

    for grupo_nome, features in groups.items():
        features_presentes = [f for f in features if f in merged_contrib.index]
        if not features_presentes:
            continue
        soma = merged_contrib[features_presentes].sum()
        merged_contrib = merged_contrib.drop(features_presentes)
        merged_contrib[grupo_nome] = soma
        merged_values[grupo_nome] = {f: row[f] for f in features_presentes}

    return merged_contrib, merged_values


def explain_row(row_idx, X_test, shap_values, dummy_map, groups, top_n=TOP_N):
    contributions = pd.Series(shap_values[row_idx].values, index=X_test.columns)
    row = X_test.iloc[row_idx]

    merged_contrib, merged_values = merge_correlated_shap(contributions, row, groups)
    top_features = merged_contrib.abs().sort_values(ascending=False).head(top_n)

    partes = []
    for col in top_features.index:
        direcao = "aumenta" if merged_contrib[col] > 0 else "reduz"

        if col in merged_values:
            detalhes = ", ".join(f"{k}={v}" for k, v in merged_values[col].items())
            partes.append(f"{col} [{detalhes}] ({direcao} o risco)")
        elif col in dummy_map and dummy_map[col][1] is not None:
            nome_original, valor_categorico = dummy_map[col]
            valor_real = row[col]
            if valor_real == 1:
                partes.append(f"{nome_original} = {valor_categorico} ({direcao} o risco)")
            else:
                partes.append(f"{nome_original} != {valor_categorico} ({direcao} o risco)")
        else:
            valor_real = row[col]
            partes.append(f"{col} = {valor_real} ({direcao} o risco)")

    return "; ".join(partes)


def main():
    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    test_meta = pd.read_csv(TEST_META_PATH)

    dummy_map = build_dummy_to_readable_map(X_test.columns, CATEGORICAL_COLUMNS)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    print("=== Exemplo: primeiras 5 explicacoes (COMPAS) ===\n")
    for i in range(5):
        explicacao = explain_row(i, X_test, shap_values, dummy_map, FEATURE_GROUPS)
        print(f"[{i}] y_true={test_meta.iloc[i]['y_true']} | race_group={test_meta.iloc[i]['race_group']}")
        print(f"    Justificacao: {explicacao}\n")

    all_explanations = [
        explain_row(i, X_test, shap_values, dummy_map, FEATURE_GROUPS)
        for i in range(len(X_test))
    ]
    result_df = test_meta.copy()
    result_df["explicacao_shap"] = all_explanations
    result_df.to_csv(OUT_EXPLANATIONS_CSV, index=False)
    print(f"Explicacoes completas guardadas em: {OUT_EXPLANATIONS_CSV}")


if __name__ == "__main__":
    main()