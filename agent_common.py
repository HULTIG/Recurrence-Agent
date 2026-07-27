import pandas as pd


def build_dummy_to_readable_map(columns, categorical_columns):
    mapping = {}
    for col in columns:
        matched = False
        for cat in categorical_columns:
            prefix = f"{cat}_"
            if col.startswith(prefix):
                valor = col[len(prefix):]
                mapping[col] = (cat, valor)
                matched = True
                break
        if not matched:
            mapping[col] = (col, None)
    return mapping


def encode_raw_profile(raw_profile: dict, reference_columns, categorical_columns) -> pd.DataFrame:
    """Converte um perfil em formato cru (nomes originais das features) para o
    formato one-hot que o modelo baseline espera, usando as colunas de
    referencia (ex: X_test.columns) para garantir a mesma ordem/presenca."""
    encoded = {}
    for col in reference_columns:
        matched = False
        for cat in categorical_columns:
            prefix = f"{cat}_"
            if col.startswith(prefix):
                valor_dummy = col[len(prefix):]
                encoded[col] = 1 if str(raw_profile.get(cat)) == valor_dummy else 0
                matched = True
                break
        if not matched:
            encoded[col] = raw_profile.get(col, 0)
    return pd.DataFrame([encoded], columns=reference_columns)


def merge_correlated_shap(contributions: pd.Series, row: pd.Series, groups: dict):
    merged_contrib = contributions.copy()
    merged_values = {}
    for grupo_nome, features in groups.items():
        features_presentes = [f for f in features if f in merged_contrib.index]
        if not features_presentes:
            continue
        soma = merged_contrib[features_presentes].sum()
        merged_contrib = merged_contrib.drop(features_presentes)
        merged_contrib[grupo_nome] = soma
        merged_values[grupo_nome] = {f: row[f] for f in features_presentes}
    return merged_contrib, merged_values


def explain_contributions(contributions: pd.Series, row: pd.Series, dummy_map, groups, top_n=4):
    merged_contrib, merged_values = merge_correlated_shap(contributions, row, groups)
    top_features = merged_contrib.abs().sort_values(ascending=False).head(top_n)

    partes = []
    for col in top_features.index:
        direcao = "aumenta" if merged_contrib[col] > 0 else "reduz"
        if col in merged_values:
            detalhes = ", ".join(f"{k}={v}" for k, v in merged_values[col].items())
            partes.append(f"{col} [{detalhes}] ({direcao} o risco)")
        elif col in dummy_map and dummy_map[col][1] is not None:
            nome_original, valor_categorico = dummy_map[col]
            valor_real = row[col]
            if valor_real == 1:
                partes.append(f"{nome_original} = {valor_categorico} ({direcao} o risco)")
            else:
                partes.append(f"{nome_original} != {valor_categorico} ({direcao} o risco)")
        else:
            partes.append(f"{col} = {row[col]} ({direcao} o risco)")

    return "; ".join(partes)