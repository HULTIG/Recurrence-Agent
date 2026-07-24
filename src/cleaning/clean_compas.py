import pandas as pd

RAW_PATH = "data/compas-scores-two-years.csv"
OUT_PATH = "clean_data/compas-scores-two-years.csv"

# Identifying columns, no predictive value, should not stay in an
# anonymous profile passed to an agent
ID_COLUMNS = ["name", "first", "last", "dob"]

# Audit/process columns, not useful as classification features
AUDIT_COLUMNS = [
    "c_case_number", "r_case_number", "vr_case_number",
    "c_jail_in", "c_jail_out", "r_jail_in", "r_jail_out",
    "in_custody", "out_custody",
    "compas_screening_date", "screening_date", "v_screening_date",
    "c_offense_date", "c_arrest_date", "r_offense_date", "vr_offense_date",
    "c_charge_desc", "r_charge_desc", "vr_charge_desc",
]

# Dead column: 100% null in the original file
DEAD_COLUMNS = ["violent_recid"]

# Nulls here are structural (person did not reoffend), confirmed by
# exact match with is_recid==0 / is_violent_recid==0
STRUCTURAL_MISSING_COLUMNS = {
    "r_charge_degree": "No_Recidivism",
    "vr_charge_degree": "No_Recidivism",
}

# These columns ARE the output of the COMPAS algorithm itself, the one
# ProPublica's bias analysis targeted. Kept as a comparison baseline
# ("COMPAS said X, my model said Y, which matched two_year_recid?"),
# but must NOT be used as a training feature for your own model.
COMPAS_BASELINE_SCORE_COLUMNS = [
    "decile_score", "score_text", "type_of_assessment",
    "v_decile_score", "v_score_text", "v_type_of_assessment",
]


def to_snake_case(name: str) -> str:
    return name.strip().lower().replace(".", "_")


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH)


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    # decile_score.1 and priors_count.1 are exact copies of decile_score
    # and priors_count (confirmed by direct comparison)
    dup_cols = [c for c in df.columns if c.endswith(".1")]
    return df.drop(columns=dup_cols)


def apply_quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    # Standard filter used in the literature on this dataset: days between
    # COMPAS screening and arrest cannot exceed 30 in absolute value,
    # otherwise the record has an unreliable pairing
    mask = df["days_b_screening_arrest"].abs() <= 30
    return df[mask].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_duplicate_columns(df)
    df = apply_quality_filter(df)

    cols_to_drop = [
        c for c in ID_COLUMNS + AUDIT_COLUMNS + DEAD_COLUMNS if c in df.columns
    ]
    df = df.drop(columns=cols_to_drop)

    for col, fill_value in STRUCTURAL_MISSING_COLUMNS.items():
        df[col] = df[col].fillna(fill_value)

    # Not all nulls here coincide with absence of recidivism, so this
    # gets a flag + sentinel instead of blind imputation
    df["r_days_from_arrest_missing"] = df["r_days_from_arrest"].isnull().astype(int)
    df["r_days_from_arrest"] = df["r_days_from_arrest"].fillna(-1)

    df.columns = [to_snake_case(c) for c in df.columns]

    return df


def main():
    df = load_raw()
    n_before = df.shape

    df_clean = clean(df)

    df_clean.to_csv(OUT_PATH, index=False)

    print(f"COMPAS: {n_before} -> {df_clean.shape}")
    print(f"Guardado em: {OUT_PATH}")
    print(
        "Aviso: "
        + ", ".join(COMPAS_BASELINE_SCORE_COLUMNS)
        + " sao o output do proprio COMPAS, usar so como baseline de "
        "comparacao, NAO como feature de treino do teu modelo."
    )
    nulls = df_clean.isnull().sum()
    print(f"Nulos remanescentes por coluna:\n{nulls[nulls > 0]}")


if __name__ == "__main__":
    main()