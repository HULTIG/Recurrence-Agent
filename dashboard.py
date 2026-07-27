"""
Dashboard interativo - Agente de Previsao de Reincidencia
===========================================================
Corre com: streamlit run dashboard.py

Estrutura de pastas esperada (ajusta os caminhos abaixo se a tua for diferente):
  results/nij_qwen25_results/preds_nij_qwen25_3b.csv
  results/compas_qwen25_results/preds_compas_qwen25_3b.csv
  results/baseline_nij/preds_xgboost.csv, preds_lightgbm.csv, X_test.csv, explanations_nij.csv
  results/baseline_compas/preds_xgboost.csv, preds_lightgbm.csv, X_test.csv, explanations_compas.csv
  clean_data/nij-challenge2021.csv
  clean_data/compas-scores-two-years.csv
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

st.set_page_config(page_title="Agente de Reincidencia", layout="wide")

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
PATHS = {
    "nij": {
        "llm_preds": "results/nij_qwen25_results/preds_nij_qwen25_3b.csv",
        "baseline_dir": "results/baseline_nij",
        "explanations": "results/baseline_nij/explanations_nij.csv",
        "raw_data": "clean_data/nij-challenge2021.csv",
    },
    "compas": {
        "llm_preds": "results/compas_qwen25_results/preds_compas_qwen25_3b.csv",
        "baseline_dir": "results/baseline_compas",
        "explanations": "results/baseline_compas/explanations_compas.csv",
        "raw_data": "clean_data/compas-scores-two-years.csv",
    },
}

CATEGORICAL_COLUMNS = {
    "nij": ["gender", "supervision_level_first", "education_level", "prison_offense", "residence_puma"],
    "compas": ["sex", "age_cat", "c_charge_degree"],
}

# configuracao do LLM fine-tuned por dataset (so usado se houver GPU/CUDA disponivel)
LLM_CONFIG = {
    "nij": {
        "out_dir": "results/qwen25_3b_nij_lora",
        "chat_template": "qwen-2.5",
        "max_seq_length": 768,
    },
    "compas": {
        "out_dir": "results/qwen25_3b_compas_lora",
        "chat_template": "qwen-2.5",
        "max_seq_length": 256,
    },
}


@st.cache_resource(show_spinner="A carregar o LLM fine-tuned (pode demorar na primeira vez)...")
def try_load_llm(dataset_key):
    """Tenta carregar o LLM fine-tuned. Devolve (model, tokenizer) ou (None, None)
    se nao houver GPU/CUDA disponivel ou o Unsloth nao estiver instalado."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None

        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        cfg = LLM_CONFIG[dataset_key]
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["out_dir"],
            max_seq_length=cfg["max_seq_length"],
            load_in_4bit=True,
        )
        tokenizer = get_chat_template(tokenizer, chat_template=cfg["chat_template"])
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except Exception:
        return None, None


def llm_predict(model, tokenizer, perfil: dict, max_seq_length: int):
    """Gera a previsao Sim/Nao do LLM para um perfil (dict com nomes originais das features)."""
    linhas = [f"{col}: {val}" for col, val in perfil.items()]
    instrucao = (
        "Analisa o seguinte perfil de uma pessoa em liberdade condicional/supervisao "
        "e responde apenas 'Sim' ou 'Nao' a pergunta: esta pessoa vai reincidir dentro "
        "do periodo de acompanhamento?\n\nPerfil:\n"
    )
    prompt = instrucao + "\n".join(linhas)

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        truncation=True, max_length=max_seq_length,
    ).to(model.device)

    output = model.generate(
        input_ids=inputs, max_new_tokens=4, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    completion = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True).strip().lower()

    if completion.startswith("sim"):
        return 1, completion
    if completion.startswith("n") and "ao" in completion[:4]:
        return 0, completion
    return None, completion

FEATURE_GROUPS = {
    "nij": {
        "situacao_de_emprego": ["jobs_per_year", "percent_days_employed"],
        "historico_crimes_propriedade": ["prior_arrest_episodes_property", "prior_conviction_episodes_prop"],
        "historico_crimes_misdemeanor": ["prior_arrest_episodes_misd", "prior_conviction_episodes_misd"],
        "historico_crimes_droga": ["prior_arrest_episodes_drug", "prior_conviction_episodes_drug"],
    },
    "compas": {
        "faixa_etaria": ["age", "age_cat_Greater than 45", "age_cat_Less than 25"],
    },
}


# ---------------------------------------------------------------------------
# Utilitarios de carregamento (com cache para nao reler ficheiros a cada clique)
# ---------------------------------------------------------------------------
@st.cache_data
def safe_read_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_baseline_model(baseline_dir, model_name="xgboost"):
    path = f"{baseline_dir}/{model_name}_model.pkl"
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def compute_fpr_by_group(df, group_col="race_group"):
    """Calcula a taxa de falsos positivos por grupo, a partir de y_true/y_pred."""
    rows = []
    for group, sub in df.groupby(group_col):
        fp = ((sub["y_true"] == 0) & (sub["y_pred"] == 1)).sum()
        neg = (sub["y_true"] == 0).sum()
        fpr = fp / neg if neg > 0 else np.nan
        rows.append({"grupo": group, "n": len(sub), "fpr": fpr})
    return pd.DataFrame(rows)


def compute_metrics_row(df, label):
    return {
        "modelo": label,
        "accuracy": accuracy_score(df["y_true"], df["y_pred"]),
        "f1": f1_score(df["y_true"], df["y_pred"]),
        "precision": precision_score(df["y_true"], df["y_pred"]),
        "recall": recall_score(df["y_true"], df["y_pred"]),
        "n": len(df),
    }


def build_dummy_to_readable_map(columns, categorical_columns):
    mapping = {}
    for col in columns:
        matched = False
        for cat in categorical_columns:
            prefix = f"{cat}_"
            if col.startswith(prefix):
                mapping[col] = (cat, col[len(prefix):])
                matched = True
                break
        if not matched:
            mapping[col] = (col, None)
    return mapping


def merge_correlated_shap(contributions, row, groups):
    merged = contributions.copy()
    merged_values = {}
    for nome, feats in groups.items():
        presentes = [f for f in feats if f in merged.index]
        if not presentes:
            continue
        merged[nome] = merged[presentes].sum()
        merged = merged.drop(presentes)
        merged_values[nome] = {f: row[f] for f in presentes}
    return merged, merged_values


def explain_contributions(contributions, row, dummy_map, groups, top_n=4):
    merged, merged_values = merge_correlated_shap(contributions, row, groups)
    top = merged.abs().sort_values(ascending=False).head(top_n)

    partes = []
    for col in top.index:
        direcao = "aumenta" if merged[col] > 0 else "reduz"
        if col in merged_values:
            detalhes = ", ".join(f"{k}={v}" for k, v in merged_values[col].items())
            partes.append(f"**{col}** [{detalhes}] — {direcao} o risco")
        elif dummy_map.get(col, (col, None))[1] is not None:
            nome_original, valor_categorico = dummy_map[col]
            if row[col] == 1:
                partes.append(f"**{nome_original}** = {valor_categorico} — {direcao} o risco")
            else:
                partes.append(f"**{nome_original}** != {valor_categorico} — {direcao} o risco")
        else:
            partes.append(f"**{col}** = {row[col]} — {direcao} o risco")
    return partes


# ---------------------------------------------------------------------------
# Sidebar - selecao de dataset
# ---------------------------------------------------------------------------
st.sidebar.title("Agente de Reincidencia")
dataset_key = st.sidebar.radio("Dataset", ["nij", "compas"], format_func=lambda x: "NIJ" if x == "nij" else "COMPAS")
paths = PATHS[dataset_key]

st.title(f"Agente de Previsao de Reincidencia — {dataset_key.upper()}")
st.caption(
    "Previsao via LLM fine-tuned (Qwen2.5-3B); justificacao via atribuicao de "
    "features (SHAP) sobre um modelo baseline treinado no mesmo conjunto de dados."
)

tab_overview, tab_compare, tab_fairness, tab_explore, tab_simulate = st.tabs(
    ["Visao Geral", "Comparacao de Modelos", "Fairness", "Explorar Previsoes", "Simular Perfil Novo"]
)

# ---------------------------------------------------------------------------
# Carregamento comum
# ---------------------------------------------------------------------------
llm_preds = safe_read_csv(paths["llm_preds"])
explanations = safe_read_csv(paths["explanations"])
xgb_preds = safe_read_csv(f"{paths['baseline_dir']}/preds_xgboost.csv")
lgbm_preds = safe_read_csv(f"{paths['baseline_dir']}/preds_lightgbm.csv")

# ---------------------------------------------------------------------------
# TAB 1 - Visao geral
# ---------------------------------------------------------------------------
with tab_overview:
    if llm_preds is None:
        st.warning(f"Nao encontrei `{paths['llm_preds']}`. Confirma o caminho na secao PATHS do script.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Casos de teste", len(llm_preds))
        col2.metric("Taxa real de reincidencia", f"{llm_preds['y_true'].mean():.1%}")
        col3.metric("Respostas nao interpretaveis", int(llm_preds["y_pred"].isna().sum()))

        st.subheader("Distribuicao por grupo racial (conjunto de teste)")
        race_counts = llm_preds["race_group"].value_counts().reset_index()
        race_counts.columns = ["race_group", "n"]
        fig = px.bar(race_counts, x="race_group", y="n", text="n")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 - Comparacao de modelos
# ---------------------------------------------------------------------------
with tab_compare:
    rows = []
    if llm_preds is not None:
        valid = llm_preds.dropna(subset=["y_pred"])
        rows.append(compute_metrics_row(valid, "LLM fine-tuned (Qwen2.5-3B)"))
    if xgb_preds is not None:
        rows.append(compute_metrics_row(xgb_preds, "XGBoost (baseline)"))
    if lgbm_preds is not None:
        rows.append(compute_metrics_row(lgbm_preds, "LightGBM (baseline)"))

    if not rows:
        st.warning(
            "Nao encontrei previsoes de nenhum modelo. Confirma se ja correste "
            "o passo de guardar `preds_xgboost.csv`/`preds_lightgbm.csv` nos scripts de baseline."
        )
    else:
        metrics_df = pd.DataFrame(rows)
        st.dataframe(metrics_df.style.format({"accuracy": "{:.4f}", "f1": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}"}))

        fig = go.Figure()
        for metric in ["accuracy", "f1", "precision", "recall"]:
            fig.add_trace(go.Bar(name=metric, x=metrics_df["modelo"], y=metrics_df[metric]))
        fig.update_layout(barmode="group", yaxis_range=[0, 1], title="Comparacao de metricas por modelo")
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Baseline de referencia: um classificador que preveja sempre a classe "
            "maioritaria teria accuracy proxima da proporcao real de nao-reincidencia "
            "no conjunto de teste (ver aba Visao Geral)."
        )

# ---------------------------------------------------------------------------
# TAB 3 - Fairness
# ---------------------------------------------------------------------------
with tab_fairness:
    st.subheader("Taxa de Falsos Positivos (FPR) por grupo racial")
    st.caption(
        "FPR = proporcao de pessoas que NAO reincidiram mas foram classificadas "
        "como 'vai reincidir'. Disparidades aqui sao o tipo de vies que motivou "
        "a investigacao original da ProPublica sobre o COMPAS."
    )

    fpr_frames = []
    if llm_preds is not None:
        valid = llm_preds.dropna(subset=["y_pred"])
        fpr_llm = compute_fpr_by_group(valid)
        fpr_llm["modelo"] = "LLM fine-tuned"
        fpr_frames.append(fpr_llm)
    if xgb_preds is not None:
        fpr_xgb = compute_fpr_by_group(xgb_preds)
        fpr_xgb["modelo"] = "XGBoost"
        fpr_frames.append(fpr_xgb)
    if lgbm_preds is not None:
        fpr_lgbm = compute_fpr_by_group(lgbm_preds)
        fpr_lgbm["modelo"] = "LightGBM"
        fpr_frames.append(fpr_lgbm)

    if not fpr_frames:
        st.warning("Sem dados suficientes para calcular FPR por grupo.")
    else:
        fpr_all = pd.concat(fpr_frames, ignore_index=True)
        fig = px.bar(fpr_all, x="grupo", y="fpr", color="modelo", barmode="group",
                     labels={"fpr": "Taxa de Falsos Positivos", "grupo": "Grupo racial"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fpr_all)

        if dataset_key == "nij":
            st.markdown("---")
            st.subheader("Achado adicional: correlacao `gang_affiliated` / `gender_M`")
            st.caption(
                "Estas duas features estao correlacionadas (~0.74) no conjunto de teste. "
                "Mantidas separadas na explicacao SHAP (nao fundidas), por ser um possivel "
                "indicio de proxy indireto de genero atraves de outra variavel."
            )

# ---------------------------------------------------------------------------
# TAB 4 - Explorar previsoes existentes
# ---------------------------------------------------------------------------
with tab_explore:
    if llm_preds is None or explanations is None:
        st.warning("Preciso de `preds_*.csv` e `explanations_*.csv` para esta aba.")
    else:
        merged = pd.concat([llm_preds.reset_index(drop=True), explanations[["explicacao_shap"]].reset_index(drop=True)], axis=1)

        col1, col2 = st.columns(2)
        with col1:
            filtro_race = st.multiselect("Filtrar por grupo racial", merged["race_group"].unique().tolist())
        with col2:
            filtro_acerto = st.selectbox("Filtrar por acerto", ["Todos", "Apenas acertos", "Apenas erros"])

        filtered = merged.copy()
        if filtro_race:
            filtered = filtered[filtered["race_group"].isin(filtro_race)]
        if filtro_acerto == "Apenas acertos":
            filtered = filtered[filtered["y_true"] == filtered["y_pred"]]
        elif filtro_acerto == "Apenas erros":
            filtered = filtered[filtered["y_true"] != filtered["y_pred"]]

        st.write(f"{len(filtered)} casos correspondem ao filtro")
        st.dataframe(filtered.head(200), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 5 - Simular perfil novo (interativo)
# ---------------------------------------------------------------------------
with tab_simulate:
    st.subheader("Simular um perfil novo")
    st.caption(
        "A previsao usa o modelo baseline (XGBoost) e a justificacao vem da mesma "
        "atribuicao SHAP usada no resto do dashboard. A previsao 'oficial' do agente "
        "(LLM fine-tuned) requer GPU com CUDA disponivel — ver nota abaixo."
    )

    raw_data = safe_read_csv(paths["raw_data"])
    baseline_model = load_baseline_model(paths["baseline_dir"], "xgboost")
    x_test_ref = safe_read_csv(f"{paths['baseline_dir']}/X_test.csv")

    llm_model, llm_tokenizer = try_load_llm(dataset_key)
    if llm_model is not None:
        st.success("LLM fine-tuned carregado — a previsao final vem do LLM.")
    else:
        st.info(
            "GPU/CUDA nao detetada (ou Unsloth nao instalado) — a mostrar apenas "
            "a previsao do modelo baseline. Corre este dashboard numa maquina com "
            "GPU para ativar a previsao 'oficial' do LLM fine-tuned."
        )

    if raw_data is None or baseline_model is None or x_test_ref is None:
        st.warning(
            "Faltam ficheiros para esta aba (dataset original, modelo baseline "
            "ou X_test.csv de referencia). Confirma os caminhos em PATHS."
        )
    else:
        exclude_cols_by_dataset = {
            "nij": ["id", "race", "recidivism_within_3years", "recidivism_arrest_year1",
                    "recidivism_arrest_year2", "recidivism_arrest_year3", "training_sample"],
            "compas": ["id", "race", "two_year_recid", "event", "start", "end",
                       "is_recid", "is_violent_recid", "r_charge_degree",
                       "r_days_from_arrest", "r_days_from_arrest_missing", "vr_charge_degree",
                       "decile_score", "score_text", "type_of_assessment",
                       "v_decile_score", "v_score_text", "v_type_of_assessment"],
        }
        feature_cols = [c for c in raw_data.columns if c not in exclude_cols_by_dataset[dataset_key]]

        st.markdown("**Preenche o perfil** (ou usa um caso existente como ponto de partida):")
        usar_exemplo = st.checkbox("Preencher com um caso existente do dataset")
        exemplo_idx = None
        if usar_exemplo:
            exemplo_idx = st.number_input("Indice da linha (0 a N)", min_value=0, max_value=len(raw_data) - 1, value=0)

        perfil = {}
        cols_ui = st.columns(3)
        for i, col in enumerate(feature_cols):
            default = raw_data.iloc[exemplo_idx][col] if usar_exemplo else None
            with cols_ui[i % 3]:
                if raw_data[col].dtype == object or str(raw_data[col].dtype) == "str":
                    opcoes = sorted(raw_data[col].dropna().unique().tolist())
                    idx_default = opcoes.index(default) if default in opcoes else 0
                    perfil[col] = st.selectbox(col, opcoes, index=idx_default, key=f"in_{col}")
                elif "float" in str(raw_data[col].dtype):
                    perfil[col] = st.number_input(col, value=float(default) if default is not None else 0.0, key=f"in_{col}")
                else:
                    perfil[col] = st.number_input(col, value=int(default) if default is not None else 0, step=1, key=f"in_{col}")

        if st.button("Prever e Justificar"):
            categorical_cols = CATEGORICAL_COLUMNS[dataset_key]
            reference_cols = x_test_ref.columns.tolist()

            encoded = {}
            for col in reference_cols:
                matched = False
                for cat in categorical_cols:
                    prefix = f"{cat}_"
                    if col.startswith(prefix):
                        valor_dummy = col[len(prefix):]
                        encoded[col] = 1 if str(perfil.get(cat)) == valor_dummy else 0
                        matched = True
                        break
                if not matched:
                    encoded[col] = perfil.get(col, 0)

            X_row = pd.DataFrame([encoded], columns=reference_cols)

            pred = int(baseline_model.predict(X_row)[0])
            proba = float(baseline_model.predict_proba(X_row)[0][1])

            import shap
            explainer = shap.TreeExplainer(baseline_model)
            contributions = pd.Series(explainer(X_row).values[0], index=reference_cols)

            dummy_map = build_dummy_to_readable_map(reference_cols, categorical_cols)
            partes = explain_contributions(contributions, X_row.iloc[0], dummy_map, FEATURE_GROUPS[dataset_key])

            st.markdown("---")

            # tenta a previsao do LLM, se o modelo estiver carregado
            llm_pred, llm_raw = (None, None)
            if llm_model is not None:
                cfg = LLM_CONFIG[dataset_key]
                llm_pred, llm_raw = llm_predict(llm_model, llm_tokenizer, perfil, cfg["max_seq_length"])

            if llm_pred is not None:
                resultado_txt = "VAI reincidir" if llm_pred == 1 else "NAO vai reincidir"
                cor = "red" if llm_pred == 1 else "green"
                st.markdown(f"### Previsao do Agente (LLM fine-tuned): :{cor}[{resultado_txt}]")
                st.caption(f"Resposta bruta do modelo: `{llm_raw}`")
            elif llm_model is not None:
                st.warning(
                    f"O LLM devolveu uma resposta nao interpretavel (`{llm_raw}`). "
                    "A mostrar a previsao do baseline como alternativa."
                )
            resultado_baseline_txt = "VAI reincidir" if pred == 1 else "NAO vai reincidir"
            cor_baseline = "red" if pred == 1 else "green"
            label_baseline = "Previsao do Agente (baseline)" if llm_pred is None else "Previsao do baseline (referencia)"
            st.markdown(f"#### {label_baseline}: :{cor_baseline}[{resultado_baseline_txt}]  (probabilidade: {proba:.1%})")

            st.markdown("**Justificacao (baseada em atribuicao de features do modelo baseline):**")
            for p in partes:
                st.markdown(f"- {p}")

            if llm_model is None:
                st.caption(
                    "Nota: sem GPU disponivel, a previsao mostrada acima e do modelo "
                    "baseline (XGBoost), nao do LLM fine-tuned."
                )