import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

COMPAS_PATH = "clean_data/compas-scores-two-years.csv"
SHARED_PATH = "schema/canonical_shared_schema.csv"
ARTIFACTS_DIR = "results/baseline_compas"

NON_FEATURE_COLUMNS = [
    "id",
    "race",
    "two_year_recid",
    "event", "start", "end",
    "is_recid", "is_violent_recid",
    "r_charge_degree", "r_days_from_arrest", "r_days_from_arrest_missing",
    "vr_charge_degree",
    "decile_score", "score_text", "type_of_assessment",
    "v_decile_score", "v_score_text", "v_type_of_assessment",
]

CATEGORICAL_COLUMNS = [
    "sex", "age_cat", "c_charge_degree",
]


def build_features(compas: pd.DataFrame) -> pd.DataFrame:
    X = compas.drop(columns=NON_FEATURE_COLUMNS)
    X = pd.get_dummies(X, columns=CATEGORICAL_COLUMNS, drop_first=True)
    return X


def fairness_report(y_true, y_pred, race_group):
    report = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "race_group": race_group})
    rows = []
    for group, sub in report.groupby("race_group"):
        false_positives = ((sub["y_true"] == 0) & (sub["y_pred"] == 1)).sum()
        negatives = (sub["y_true"] == 0).sum()
        fpr = false_positives / negatives if negatives else float("nan")
        rows.append({"race_group": group, "n": len(sub), "false_positive_rate": fpr})
    return pd.DataFrame(rows)


def main():
    import os
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    compas = pd.read_csv(COMPAS_PATH)
    shared = pd.read_csv(SHARED_PATH)
    compas_shared = shared[shared["dataset_origin"] == "COMPAS"].set_index("id")

    target = compas_shared.loc[compas["id"], "target_2yr_recid"].reset_index(drop=True)
    race_group = compas_shared.loc[compas["id"], "race_group"].reset_index(drop=True)

    X = build_features(compas)

    X_train, X_test, y_train, y_test, _, race_test = train_test_split(
        X, target, race_group, test_size=0.2, stratify=target, random_state=42
    )

    models = {
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            eval_metric="logloss", random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=42, verbosity=-1,
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(f"\n=== {name} (COMPAS) ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1: {f1_score(y_test, y_pred):.4f}")
        print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
        print(fairness_report(y_test, y_pred, race_test).to_string(index=False))

        # guarda o modelo treinado para reutilizacao (ex: SHAP, sem retreinar)
        joblib.dump(model, f"{ARTIFACTS_DIR}/{name.lower()}_model.pkl")

    # guarda o X_test (ja com one-hot encoding) e metadados, para o script de explicacao
    X_test.reset_index(drop=True).to_csv(f"{ARTIFACTS_DIR}/X_test.csv", index=False)
    pd.DataFrame({
        "y_true": y_test.reset_index(drop=True),
        "race_group": race_test.reset_index(drop=True),
    }).to_csv(f"{ARTIFACTS_DIR}/test_meta.csv", index=False)

    print(f"\nModelos e X_test guardados em: {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()