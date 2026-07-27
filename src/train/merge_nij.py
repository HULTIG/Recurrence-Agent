import pandas as pd

preds = pd.read_csv("results/nij_qwen25_results/preds_nij_qwen25_3b.csv")
explicacoes = pd.read_csv("results/baseline_nij/explanations_nij.csv")

final = pd.concat([preds, explicacoes[["explicacao_shap"]]], axis=1)
final.to_csv("results/agente_nij_com_justificacao.csv", index=False)
print(f"Guardado: results/agente_nij_com_justificacao.csv ({len(final)} linhas)")
print(final.head())