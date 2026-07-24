import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

NIJ_PATH = "clean_data/nij-challenge2021.csv"
SHARED_PATH = "schema/canonical_shared_schema.csv"

# Columns that are targets, leak the target, or are not real features
NON_FEATURE_COLUMNS = [
    "id",
    "race",  # kept only for the fairness audit, never as a feature
    "recidivism_within_3years",
    "recidivism_arrest_year1",
    "recidivism_arrest_year2",
    "recidivism_arrest_year3",
    "training_sample",  # original NIJ split flag, not a feature
]

CATEGORICAL_COLUMNS = [
    "gender", "supervision_level_first", "education_level",
    "prison_offense", "residence_puma",
]


def build_features(nij: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    X = nij.drop(columns=NON_FEATURE_COLUMNS)
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
    nij = pd.read_csv(NIJ_PATH)
    shared = pd.read_csv(SHARED_PATH)
    nij_shared = shared[shared["dataset_origin"] == "NIJ"].set_index("id")

    target = nij_shared.loc[nij["id"], "target_2yr_recid"].reset_index(drop=True)
    race_group = nij_shared.loc[nij["id"], "race_group"].reset_index(drop=True)

    X = build_features(nij, target)

    train_mask = (nij["training_sample"] == 1).values
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = target[train_mask], target[~train_mask]
    race_test = race_group[~train_mask]

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

        print(f"\n=== {name} (NIJ) ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1: {f1_score(y_test, y_pred):.4f}")
        print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
        print(fairness_report(y_test, y_pred, race_test).to_string(index=False))


if __name__ == "__main__":
    main()