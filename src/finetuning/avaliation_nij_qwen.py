import gc
import torch
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from finetune_common_functions import generate_predictions

NIJ_PATH = "clean_data/nij-challenge2021.csv"
SHARED_PATH = "schema/canonical_shared_schema.csv"

OUT_DIR = "results/qwen25_3b_nij_lora"
CHAT_TEMPLATE = "qwen-2.5"
MAX_SEQ_LENGTH = 768

PREDS_CSV = "results/nij_qwen25_results/preds_nij_qwen25_3b.csv"
METRICS_CSV = "results/nij_qwen25_results/metrics_nij_qwen25_3b.csv"

NON_FEATURE_COLUMNS = [
    "id", "race",
    "recidivism_within_3years", "recidivism_arrest_year1",
    "recidivism_arrest_year2", "recidivism_arrest_year3",
    "training_sample",
]


def load_test_data():
    nij = pd.read_csv(NIJ_PATH)
    shared = pd.read_csv(SHARED_PATH)
    nij_shared = shared[shared["dataset_origin"] == "NIJ"].set_index("id")

    target = nij_shared.loc[nij["id"], "target_2yr_recid"].reset_index(drop=True)
    race_group = nij_shared.loc[nij["id"], "race_group"].reset_index(drop=True)

    feature_columns = [c for c in nij.columns if c not in NON_FEATURE_COLUMNS]

    train_mask = (nij["training_sample"] == 1).values
    test_df = nij[~train_mask].reset_index(drop=True)
    test_target = target[~train_mask].reset_index(drop=True)
    test_race = race_group[~train_mask].reset_index(drop=True)

    return test_df, test_target, test_race, feature_columns


def main():
    test_df, test_target, test_race, feature_columns = load_test_data()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=OUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)
    FastLanguageModel.for_inference(model)

    preds = generate_predictions(
        model, tokenizer, test_df, feature_columns,
        max_input_length=MAX_SEQ_LENGTH,
    )

    # --- guarda as previsoes linha a linha, incluindo as descartadas (pred=None) ---
    preds_df = pd.DataFrame({
        "y_true": test_target,
        "y_pred": preds,
        "race_group": test_race,
    })
    preds_df.to_csv(PREDS_CSV, index=False)
    print(f"Previsoes guardadas em: {PREDS_CSV}")

    # --- filtra so as validas para calcular metricas ---
    valid = preds_df.dropna(subset=["y_pred"])
    y_true = valid["y_true"]
    y_pred = valid["y_pred"]
    dropped = len(preds_df) - len(valid)

    print(classification_report(y_true, y_pred, target_names=["Nao reincide", "Reincide"]))

    print("\n=== Accuracy por race_group ===")
    race_breakdown = valid.groupby("race_group").apply(
        lambda g: pd.Series({
            "accuracy": (g.y_true == g.y_pred).mean(),
            "n": len(g),
        })
    )
    print(race_breakdown)

    # --- guarda o resumo de metricas globais ---
    metrics_summary = pd.DataFrame([{
        "model": OUT_DIR,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "dropped": dropped,
        "total": len(preds_df),
    }])
    metrics_summary.to_csv(METRICS_CSV, index=False)
    print(f"\nMetricas guardadas em: {METRICS_CSV}")

    race_breakdown.to_csv(METRICS_CSV.replace(".csv", "_por_race.csv"))

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()