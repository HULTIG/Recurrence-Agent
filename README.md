# Recurrence Agent — Previsão de Reincidência com LLM Fine-Tuned

Este projeto explora o fine-tuning de um LLM open-source (Qwen2.5-3B-Instruct) para prever
reincidência criminal a partir de dois datasets amplamente estudados na literatura de justiça
criminal, o **NIJ Recidivism Forecasting Challenge (2021)** e o **COMPAS (ProPublica)**, e
compara os resultados com baselines clássicos (XGBoost, LightGBM). Além da classificação em si,
o projeto inclui um mecanismo de **justificação das previsões**, baseado em atribuição de
features (SHAP) sobre os modelos baseline, para tornar as decisões do agente auditáveis.

## Objetivo

Construir um agente que, dado o perfil de uma pessoa em liberdade condicional/supervisão,
preveja se vai reincidir dentro do período de acompanhamento, e que seja capaz de **justificar
essa previsão** com base nas features que mais pesaram na decisão, de forma fiel (não apenas uma
racionalização gerada a posteriori pelo próprio LLM).

## Datasets

| Dataset | Ficheiro | Nº exemplos (treino / teste) | Features | Split |
|---|---|---|---|---|
| NIJ Challenge 2021 | `clean_data/nij-challenge2021.csv` | 18,028 / 7,807 | 48 | Split oficial (`training_sample`) |
| COMPAS (ProPublica) | `clean_data/compas-scores-two-years.csv` | ~4,937 / 1,235 | 11 | 80/20 estratificado (`random_state=42`) |

Ambos os datasets partilham um schema comum (`schema/canonical_shared_schema.csv`) com o alvo
binário `target_2yr_recid` e o `race_group`, usado exclusivamente para auditoria de fairness
(a variável `race` é sempre excluída das features de treino).

**Balanceamento de classes confirmado:**
- NIJ: ~52% "Não reincide" / ~48% "Reincide"
- COMPAS: taxa de reincidência próxima da distribuição documentada na literatura

## Arquitetura da solução

```
Perfil da pessoa (features tabulares)
        │
        ├──► LLM fine-tuned (Qwen2.5-3B-Instruct, LoRA) ──► "Sim" / "Não" (reincide?)
        │
        └──► Baseline (XGBoost/LightGBM, já treinado) ──► valores SHAP ──► justificação
                                                                  │
                                                                  ▼
                                                  Texto de justificação (features reais)
```

A previsão vem do LLM fine-tuned; a justificação vem da atribuição de importância de features
(SHAP) calculada sobre o modelo baseline treinado no mesmo conjunto de features — isto evita o
problema conhecido de LLMs gerarem explicações fluentes mas não fiéis ao que realmente
influenciou a decisão (*unfaithful post-hoc rationalization*).

## Modelo e fine-tuning

- **Modelo base:** `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (4-bit, via [Unsloth](https://github.com/unslothai/unsloth))
- **Método:** LoRA (r=16 no COMPAS, r=8 no NIJ — ajustado para caber em 8GB VRAM)
- **Hardware de treino:** NVIDIA RTX 4060 (8GB VRAM)
- **Formato de entrada:** cada perfil é convertido em texto (`feature: valor` por linha) e
  apresentado como pergunta de chat; o modelo responde apenas `"Sim"` ou `"Nao"`.
- **`MAX_SEQ_LENGTH`:** 768 (NIJ) / 256 (COMPAS) — calibrado por medição real do comprimento dos
  prompts tokenizados, não por estimativa (ver nota abaixo).

> **Nota técnica importante:** a primeira versão do treino usava `MAX_SEQ_LENGTH=512` para o NIJ,
> o que truncava a resposta ("Sim"/"Nao") em praticamente todos os exemplos de treino — o modelo
> nunca via a label real. Isto foi detetado através da inspeção das saídas cruas do modelo
> (`RAW COMPLETION`) e corrigido medindo o comprimento real dos prompts tokenizados antes de
> escolher `MAX_SEQ_LENGTH`. Qualquer novo dataset/modelo adicionado a este projeto deve repetir
> essa medição antes de treinar.

## Resultados

### Qwen2.5-3B fine-tuned

| Dataset | Accuracy | F1 | Respostas descartadas |
|---|---|---|---|
| NIJ | 0.6931 | 0.6386 | 0/7807 |
| COMPAS | 0.7028 | 0.6384 | 0/1235 |

### Baselines clássicos (XGBoost / LightGBM)

Ver `results/baseline_nij/` e `results/baseline_compas/` para accuracy, F1, AUC e o
`fairness_report` (taxa de falsos positivos por `race_group`) de cada modelo.

### Fairness

A auditoria de fairness usa a **taxa de falsos positivos por grupo racial** (não apenas
accuracy geral), por ser a métrica que motivou a investigação original da ProPublica sobre o
COMPAS. Um achado relevante identificado no NIJ: correlação entre `gang_affiliated` e `gender_M`
(0.736) — mantida separada na análise SHAP (não fundida com outras features correlacionadas),
por ser um potencial indício de proxy indireto de género através de outra variável, e não um
artefacto técnico a corrigir.

## Estrutura do repositório

```
clean_data/                        # Datasets originais (NIJ, COMPAS)
schema/                            # Schema partilhado (target, race_group)
src/
  finetuning/
    train_finetune_nij.py          # Fine-tuning do Qwen2.5-3B no NIJ
    train_finetune_compas.py       # Fine-tuning do Qwen2.5-3B no COMPAS
    finetune_common_functions.py   # Prompt building, parsing, geracao de previsoes
  train/
    train_baseline_nij.py          # Baseline XGBoost/LightGBM (NIJ)
    train_baseline_compas.py       # Baseline XGBoost/LightGBM (COMPAS)
    artifacts_nij.py / artifacts_compas.py
  cleaning/                        # Scripts de limpeza dos dados originais
  build/
results/
  qwen25_3b_nij_lora/              # Adaptador LoRA final (NIJ) — .safetensors nao versionado
  qwen25_3b_compas_lora/           # Adaptador LoRA final (COMPAS) — .safetensors nao versionado
  baseline_nij/                    # Modelos .pkl, X_test, explicacoes SHAP (NIJ)
  baseline_compas/                 # Modelos .pkl, X_test, explicacoes SHAP (COMPAS)
  nij_qwen25_results/              # preds_*.csv, metrics_*.csv (NIJ)
  compas_qwen25_results/           # preds_*.csv, metrics_*.csv (COMPAS)
requirements.txt
```

> Os ficheiros `adapter_model.safetensors` (adaptadores LoRA finais) **não estão versionados**
> neste repositório por excederem o limite de 100MB do GitHub. São transferidos separadamente
> (ex: pen drive, cloud storage). Todos os checkpoints intermédios de treino (`checkpoint-*`)
> também são excluídos por não serem necessários para reprodução dos resultados finais.

## Como reproduzir

### 1. Baselines (não precisa de GPU)

```bash
pip install pandas numpy scikit-learn xgboost lightgbm joblib shap
python src/train/train_baseline_nij.py
python src/train/train_baseline_compas.py
```

### 2. Fine-tuning do LLM (precisa de GPU NVIDIA com CUDA, mínimo 8GB VRAM)

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
python src/finetuning/train_finetune_nij.py
python src/finetuning/train_finetune_compas.py
```

### 3. Explicações SHAP (não precisa de GPU)

```bash
python src/train/explain_nij.py
python src/train/explain_compas.py
```

### 4. Juntar previsões + justificações

```python
import pandas as pd

preds = pd.read_csv("results/nij_qwen25_results/preds_nij_qwen25_3b.csv")
explicacoes = pd.read_csv("results/baseline_nij/explanations_nij.csv")
final = pd.concat([preds, explicacoes[["explicacao_shap"]]], axis=1)
final.to_csv("results/agente_nij_com_justificacao.csv", index=False)
```

## Requisitos

Ver `requirements.txt`. Resumo:
- **Análise/baselines:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `joblib`, `shap`
- **Fine-tuning (GPU only):** `torch==2.5.1+cu121`, `transformers`, `datasets`, `trl`, `peft`,
  `bitsandbytes`, `unsloth`

## Limitações conhecidas

- O NIJ Challenge é um problema historicamente difícil de prever bem, mesmo os modelos
  vencedores da competição original melhoraram apenas marginalmente sobre baselines ingénuos.
- As explicações SHAP são calculadas sobre o modelo baseline (XGBoost), não diretamente sobre o
  LLM, são uma aproximação razoável (mesmas features de entrada), não uma explicação exata do
  mecanismo interno do LLM.
- Features correlacionadas (ex: `prior_arrest_episodes_*` vs `prior_conviction_episodes_*`) foram
  agrupadas antes do cálculo de importância para evitar atribuições contraditórias entre
  features que medem essencialmente o mesmo conceito.
