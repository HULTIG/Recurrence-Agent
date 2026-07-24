import pandas as pd

INSTRUCTION = (
    "Analisa o seguinte perfil de uma pessoa em liberdade condicional/supervisao "
    "e responde apenas 'Sim' ou 'Nao' a pergunta: esta pessoa vai reincidir dentro "
    "do periodo de acompanhamento?\n\nPerfil:\n"
)


def row_to_profile_text(row: pd.Series, feature_columns: list) -> str:
    lines = [f"{col}: {row[col]}" for col in feature_columns]
    return INSTRUCTION + "\n".join(lines)


def build_chat_examples(df: pd.DataFrame, feature_columns: list, target: pd.Series) -> list:
    examples = []
    for idx, row in df.iterrows():
        prompt = row_to_profile_text(row, feature_columns)
        completion = "Sim" if target.loc[idx] == 1 else "Nao"
        examples.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
        })
    return examples


def parse_completion_to_label(text: str):
    text = text.strip().lower()
    if text.startswith("sim"):
        return 1
    if text.startswith("n") and "ao" in text[:4]:  # "nao" / "não"
        return 0
    return None  # model produced something unparseable, drop from metrics


def generate_predictions(model, tokenizer, df: pd.DataFrame, feature_columns: list, max_new_tokens: int = 4):
    predictions = []
    for _, row in df.iterrows():
        prompt = row_to_profile_text(row, feature_columns)
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        output = model.generate(
            input_ids=inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
        completion = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
        predictions.append(parse_completion_to_label(completion))
    return predictions