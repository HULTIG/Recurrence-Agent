import re
import pandas as pd

RAW_PATH = "data/nij-challenge2021_full_dataset.csv"
OUT_PATH = "clean_data/nij-challenge2021.csv"

# Generic-named columns in the original file, mapped per the official
# NIJ codebook and the published errata
RENAME_BROKEN_COLUMNS = {
    "_v1": "Prior_Arrest_Episodes_PPViolationCharges",
    "_v2": "Prior_Conviction_Episodes_PPViolationCharges",
    "_v3": "Prior_Conviction_Episodes_DomesticViolenceCharges",
    "_v4": "Prior_Conviction_Episodes_GunCharges",
}

YES_NO_MAP = {"Yes": 1, "No": 0}

# Columns following the "N" / "N or more" pattern (e.g. "0","1","2","3 or more")
# handled generically by extracting the leading number, preserving order
ORDINAL_NUMERIC_PATTERN_COLUMNS = [
    "dependents",
    "prior_arrest_episodes_felony",
    "prior_arrest_episodes_misd",
    "prior_arrest_episodes_violent",
    "prior_arrest_episodes_property",
    "prior_arrest_episodes_drug",
    "prior_arrest_episodes_ppviolationcharges",
    "prior_conviction_episodes_felony",
    "prior_conviction_episodes_misd",
    "prior_conviction_episodes_prop",
    "prior_conviction_episodes_drug",
    "delinquency_reports",
    "program_attendances",
    "program_unexcusedabsences",
    "residence_changes",
]

# Columns whose categories have no direct number, need an explicit order map
PRISON_YEARS_ORDER = {
    "Less than 1 year": 0,
    "1-2 years": 1,
    "Greater than 2 to 3 years": 2,
    "More than 3 years": 3,
}

AGE_AT_RELEASE_ORDER = {
    "18-22": 0,
    "23-27": 1,
    "28-32": 2,
    "33-37": 3,
    "38-42": 4,
    "43-47": 5,
    "48 or older": 6,
}

# Drug test nulls likely mean "not tested", not "missing at random"
DRUG_TEST_COLUMNS = [
    "avg_days_per_drugtest",
    "drugtests_thc_positive",
    "drugtests_cocaine_positive",
    "drugtests_meth_positive",
    "drugtests_other_positive",
]

# Numeric nulls with no clear structural meaning, filled with the median
NUMERIC_MEDIAN_FILL_COLUMNS = [
    "supervision_risk_score_first",
    "percent_days_employed",
    "jobs_per_year",
]

# Categorical nulls with no clear structural meaning, filled with an
# explicit "Missing" category instead of guessing
CATEGORICAL_MISSING_FILL_COLUMNS = [
    "supervision_level_first",
    "prison_offense",
]


def to_snake_case(name: str) -> str:
    return name.strip().lower().replace("__", "_")


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH)


def extract_leading_number(value):
    if pd.isna(value):
        return value
    match = re.match(r"\s*(\d+)", str(value))
    return int(match.group(1)) if match else value


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME_BROKEN_COLUMNS)

    yes_no_cols = [
        c for c in df.columns
        if set(df[c].dropna().unique()) <= {"Yes", "No"}
    ]
    for c in yes_no_cols:
        df[c] = df[c].map(YES_NO_MAP)

    df.columns = [to_snake_case(c) for c in df.columns]

    for c in ORDINAL_NUMERIC_PATTERN_COLUMNS:
        df[c] = df[c].apply(extract_leading_number)

    df["prison_years"] = df["prison_years"].map(PRISON_YEARS_ORDER)
    df["age_at_release"] = df["age_at_release"].map(AGE_AT_RELEASE_ORDER)

    # gang_affiliated: do not default missing values to "No", that would
    # bias the model against people without this field recorded.
    # -1 = unknown, kept as its own category.
    df["gang_affiliated"] = df["gang_affiliated"].fillna(-1).astype(int)

    # residence_puma is a geographic zone code, not a quantity
    df["residence_puma"] = df["residence_puma"].astype(str)

    df["foi_testado_drogas"] = df["avg_days_per_drugtest"].notna().astype(int)
    for c in DRUG_TEST_COLUMNS:
        df[c] = df[c].fillna(0)

    for c in NUMERIC_MEDIAN_FILL_COLUMNS:
        df[c] = df[c].fillna(df[c].median())
    for c in CATEGORICAL_MISSING_FILL_COLUMNS:
        df[c] = df[c].fillna("Missing")

    return df


def main():
    df = load_raw()
    n_before = df.shape

    df_clean = clean(df)

    df_clean.to_csv(OUT_PATH, index=False)

    print(f"NIJ: {n_before} -> {df_clean.shape}")
    print(f"Guardado em: {OUT_PATH}")
    yes_no_cols = [c for c in df.columns if set(df[c].dropna().unique()) <= {"Yes", "No"}]
    print(f"Colunas Yes/No convertidas ({len(yes_no_cols)}): {yes_no_cols}")
    print(
        "Aviso: 'training_sample' e a flag de split 70/30 do NIJ original, "
        "NAO deve ser usada como feature de treino."
    )
    nulls = df_clean.isnull().sum()
    print(f"Nulos remanescentes por coluna:\n{nulls[nulls > 0]}")


if __name__ == "__main__":
    main()