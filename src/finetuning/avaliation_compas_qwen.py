import gc
import torch
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from finetune_common_functions import generate_predictions

COMPAS_PATH = "clean_data/compas-scores-two-years.csv"
SHARED_PATH = "schema/canonical_shared_schema.csv"

OUT_DIR = "results/qwen25_3b_compas_lora"
CHAT_TEMPLATE = "qwen-2.5"
MAX_SEQ_LENGTH = 256

PREDS_CSV = "results/compas_qwen25_results/preds_compas_qwen25_3b.csv"
METRICS_CSV = "results/compas_qwen25_results/metrics_compas_qwen25_3b.csv"

NON_FEATURE_COLUMNS = [
    "id", "race", "two_year_recid",
    "event", "start", "end",
    "is_recid", "is_violent_recid",
    "r_charge_degree", "r_days_from_arrest", "r_days_from_arrest_missing",
    "vr_charge_degree",
    "decile_score", "score_text", "type_of_assessment",
    "v_decile_score", "v_score_text", "v_type_of_assessment",
]


def load_test_data():
    compas = pd.read_csv(COMPAS_PATH)
    shared = pd.read_csv(SHARED_PATH)
    compas_shared = shared[shared["dataset_origin"] == "COMPAS"].set_index("id")

    target = compas_shared.loc[compas["id"], "target_2yr_recid"].reset_index(drop=True)
    race_group = compas_shared.loc[compas["id"], "race_group"].reset_index(drop=True)

    feature_columns = [c for c in compas.columns if c not in NON_FEATURE_COLUMNS]

    train_idx, test_idx = train_test_split(
        compas.index, test_size=0.2, stratify=target, random_state=42
    )
    test_df = compas.loc[test_idx].reset_index(drop=True)
    test_target = target.loc[test_idx].reset_index(drop=True)
    test_race = race_group.loc[test_idx].reset_index(drop=True)

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

    # opcional: guarda tambem o breakdown por race num csv separado
    race_breakdown.to_csv(METRICS_CSV.replace(".csv", "_por_race.csv"))

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()