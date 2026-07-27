import pandas as pd

preds = pd.read_csv("results/compas_qwen25_results/preds_compas_qwen25_3b.csv")
explicacoes = pd.read_csv("results/baseline_compas/explanations_compas.csv")

final = pd.concat([preds, explicacoes[["explicacao_shap"]]], axis=1)
final.to_csv("results/agente_compas_com_justificacao.csv", index=False)
print(f"Guardado: results/agente_compas_com_justificacao.csv ({len(final)} linhas)")
print(final.head())