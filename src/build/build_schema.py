import pandas as pd

NIJ_PATH = "clean_data/nij-challenge2021.csv"
COMPAS_PATH = "clean_data/compas-scores-two-years.csv"
OUT_PATH = "schema/canonical_shared_schema.csv"

NIJ_RACE_MAP = {"BLACK": "Black", "WHITE": "White"}
COMPAS_RACE_MAP = {
    "African-American": "Black",
    "Caucasian": "White",
    "Hispanic": "Other",
    "Native American": "Other",
    "Asian": "Other",
    "Other": "Other",
}

NIJ_SEX_MAP = {"M": "M", "F": "F"}
COMPAS_SEX_MAP = {"Male": "M", "Female": "F"}

# Fields both datasets share and that carry comparable meaning.
# race_group is kept for fairness auditing only, it is NOT meant to be
# used as a training feature in either model.
SHARED_COLUMNS = ["dataset_origin", "id", "race_group", "sex", "target_2yr_recid"]


def build_nij_shared(nij: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=nij.index)
    out["dataset_origin"] = "NIJ"
    out["id"] = nij["id"]
    out["race_group"] = nij["race"].map(NIJ_RACE_MAP)
    out["sex"] = nij["gender"].map(NIJ_SEX_MAP)

    # Recidivism_Arrest_Year1/2/3 are per-year flags, NOT cumulative
    # (confirmed: Year1 OR Year2 OR Year3 == Recidivism_Within_3years).
    # The 2-year equivalent is therefore Year1 OR Year2, not Year2 alone.
    out["target_2yr_recid"] = (
        (nij["recidivism_arrest_year1"] == 1) | (nij["recidivism_arrest_year2"] == 1)
    ).astype(int)

    return out[SHARED_COLUMNS]


def build_compas_shared(compas: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=compas.index)
    out["dataset_origin"] = "COMPAS"
    out["id"] = compas["id"]
    out["race_group"] = compas["race"].map(COMPAS_RACE_MAP)
    out["sex"] = compas["sex"].map(COMPAS_SEX_MAP)
    out["target_2yr_recid"] = compas["two_year_recid"]

    return out[SHARED_COLUMNS]


def main():
    nij = pd.read_csv(NIJ_PATH)
    compas = pd.read_csv(COMPAS_PATH)

    shared = pd.concat(
        [build_nij_shared(nij), build_compas_shared(compas)],
        ignore_index=True,
    )

    shared.to_csv(OUT_PATH, index=False)

    print(f"Canonical shared schema: {shared.shape}")
    print(f"Guardado em: {OUT_PATH}")
    print(shared["dataset_origin"].value_counts())
    print()
    print("Aviso: esta tabela e so para auditoria de fairness e teste de")
    print("generalizacao entre fontes. Os modelos devem treinar sobre")
    print(f"{NIJ_PATH} e {COMPAS_PATH} separadamente, nao sobre esta tabela.")


if __name__ == "__main__":
    main()