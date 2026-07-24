import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from finetune_common_functions import build_chat_examples, generate_predictions

COMPAS_PATH = "clean_data/compas-scores-two-years.csv"
SHARED_PATH = "schema/canonical_shared_schema.csv"
MODEL_NAME = "unsloth/gemma-4-E4B-it"
OUT_DIR = "results/gemma4_e4b_compas_lora"

MAX_SEQ_LENGTH = 1024

# Same exclusions as train_baseline_compas.py: no id, no race (fairness
# audit only), no target/leakage columns (event outcome fields leak the
# target, COMPAS's own scores are a comparison baseline, not a feature)
NON_FEATURE_COLUMNS = [
    "id", "race", "two_year_recid",
    "event", "start", "end",
    "is_recid", "is_violent_recid",
    "r_charge_degree", "r_days_from_arrest", "r_days_from_arrest_missing",
    "vr_charge_degree",
    "decile_score", "score_text", "type_of_assessment",
    "v_decile_score", "v_score_text", "v_type_of_assessment",
]


def load_data():
    compas = pd.read_csv(COMPAS_PATH)
    shared = pd.read_csv(SHARED_PATH)
    compas_shared = shared[shared["dataset_origin"] == "COMPAS"].set_index("id")

    target = compas_shared.loc[compas["id"], "target_2yr_recid"].reset_index(drop=True)
    race_group = compas_shared.loc[compas["id"], "race_group"].reset_index(drop=True)

    feature_columns = [c for c in compas.columns if c not in NON_FEATURE_COLUMNS]

    # Same 80/20 stratified split as train_baseline_compas.py, so the
    # comparison against the XGBoost/LightGBM numbers is apples-to-apples
    train_idx, test_idx = train_test_split(
        compas.index, test_size=0.2, stratify=target, random_state=42
    )
    train_df = compas.loc[train_idx].reset_index(drop=True)
    test_df = compas.loc[test_idx].reset_index(drop=True)
    train_target = target.loc[train_idx].reset_index(drop=True)
    test_target = target.loc[test_idx].reset_index(drop=True)
    test_race = race_group.loc[test_idx].reset_index(drop=True)

    return train_df, test_df, train_target, test_target, test_race, feature_columns


def main():
    train_df, test_df, train_target, test_target, test_race, feature_columns = load_data()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    train_examples = build_chat_examples(train_df, feature_columns, train_target)
    train_dataset = Dataset.from_list(train_examples)

    def formatting_func(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        formatting_func=formatting_func,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=3,  # COMPAS train set is much smaller than NIJ's
            learning_rate=2e-4,
            logging_steps=20,
            output_dir=OUT_DIR,
            max_seq_length=MAX_SEQ_LENGTH,
        ),
    )
    trainer.train()

    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    FastLanguageModel.for_inference(model)
    preds = generate_predictions(model, tokenizer, test_df, feature_columns)

    valid = [(y, p) for y, p in zip(test_target, preds) if p is not None]
    y_true = [y for y, p in valid]
    y_pred = [p for y, p in valid]
    dropped = len(preds) - len(valid)

    print(f"COMPAS fine-tuned Gemma 4 E4B: accuracy={accuracy_score(y_true, y_pred):.4f}")
    print(f"F1={f1_score(y_true, y_pred):.4f}")
    print(f"Respostas nao interpretaveis descartadas: {dropped}/{len(preds)}")


if __name__ == "__main__":
    main()