import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from finetune_common_functions import build_chat_examples, generate_predictions

NIJ_PATH = "clean_data/nij-challenge2021.csv"
SHARED_PATH = "schema/canonical_shared_schema.csv"
MODEL_NAME = "unsloth/gemma-4-E4B-it"
OUT_DIR = "results/gemma4_e4b_nij_lora"

MAX_SEQ_LENGTH = 1024

# Same exclusions as train_baseline_nij.py: no id, no race (fairness
# audit only), no target/leakage columns, no split flag
NON_FEATURE_COLUMNS = [
    "id", "race",
    "recidivism_within_3years", "recidivism_arrest_year1",
    "recidivism_arrest_year2", "recidivism_arrest_year3",
    "training_sample",
]


def load_data():
    nij = pd.read_csv(NIJ_PATH)
    shared = pd.read_csv(SHARED_PATH)
    nij_shared = shared[shared["dataset_origin"] == "NIJ"].set_index("id")

    target = nij_shared.loc[nij["id"], "target_2yr_recid"].reset_index(drop=True)
    race_group = nij_shared.loc[nij["id"], "race_group"].reset_index(drop=True)

    feature_columns = [c for c in nij.columns if c not in NON_FEATURE_COLUMNS]

    train_mask = (nij["training_sample"] == 1).values
    train_df = nij[train_mask].reset_index(drop=True)
    test_df = nij[~train_mask].reset_index(drop=True)
    train_target = target[train_mask].reset_index(drop=True)
    test_target = target[~train_mask].reset_index(drop=True)
    test_race = race_group[~train_mask].reset_index(drop=True)

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
            num_train_epochs=1,
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
    # Evaluate on a subsample first if the full test set is slow to
    # generate over, adjust test_df.sample(n=...) as needed
    preds = generate_predictions(model, tokenizer, test_df, feature_columns)

    valid = [(y, p) for y, p in zip(test_target, preds) if p is not None]
    y_true = [y for y, p in valid]
    y_pred = [p for y, p in valid]
    dropped = len(preds) - len(valid)

    print(f"NIJ fine-tuned Gemma 4 E4B: accuracy={accuracy_score(y_true, y_pred):.4f}")
    print(f"F1={f1_score(y_true, y_pred):.4f}")
    print(f"Respostas nao interpretaveis descartadas: {dropped}/{len(preds)}")


if __name__ == "__main__":
    main()