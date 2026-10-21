"""Load the private cortical-thickness data for model training."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DATA = PROJECT_ROOT / "private_data"

CLINICAL_FILE = (
    PRIVATE_DATA / "dxmerge_rsfmri_baselinedata_2022_08_22.csv"
)
LH_FILE = PRIVATE_DATA / "aparc_stats_dkt_thickness_lh.txt"
RH_FILE = PRIVATE_DATA / "aparc_stats_dkt_thickness_rh.txt"

GROUP_ORDER = ["CN", "MCI", "AD_TRAJECTORY"]
LABEL_TO_INDEX = {
    "CN": 0,
    "MCI": 1,
    "AD_TRAJECTORY": 2,
}
INDEX_TO_LABEL = {
    index: label for label, index in LABEL_TO_INDEX.items()
}


def prepare_hemisphere(path, id_column, prefix):
    """Read one hemisphere and extract scan identifiers and CT features."""
    data = pd.read_csv(path, sep="\t")

    extracted = data[id_column].astype("string").str.extract(
        r"^sub-(\d{3}S\d{4})X(\d{8})$"
    )

    data["PTID"] = extracted[0]
    data["scan_date"] = pd.to_datetime(
        extracted[1],
        format="%Y%m%d",
        errors="raise",
    )

    thickness_columns = [
        column
        for column in data.columns
        if column.startswith(f"{prefix}_")
        and column.endswith("_thickness")
        and "MeanThickness" not in column
    ]

    if len(thickness_columns) != 34:
        raise ValueError(
            f"Expected 34 {prefix.upper()} regions, "
            f"but found {len(thickness_columns)}."
        )

    if data[["PTID", "scan_date"]].isna().any().any():
        raise ValueError(f"Could not parse every identifier in {path.name}.")

    return data[["PTID", "scan_date", *thickness_columns]], thickness_columns


def load_modeling_data():
    """
    Reconstruct the three-class cortical-thickness cohort.

    Participant identifiers are used only for matching the private files.
    They are not included in the returned feature matrix.
    """
    clinical = pd.read_csv(CLINICAL_FILE)

    clinical["scan_date"] = pd.to_datetime(
        clinical["scandate"],
        format="%m/%d/%y",
        errors="raise",
    )

    left, left_columns = prepare_hemisphere(
        LH_FILE,
        "lh.aparc.thickness",
        "lh",
    )
    right, right_columns = prepare_hemisphere(
        RH_FILE,
        "rh.aparc.thickness",
        "rh",
    )

    imaging = left.merge(
        right,
        on=["PTID", "scan_date"],
        how="inner",
        validate="one_to_one",
    )

    data = clinical.merge(
        imaging,
        on=["PTID", "scan_date"],
        how="inner",
        validate="one_to_one",
    )

    if len(data) != len(clinical):
        raise ValueError(
            f"Only {len(data)} of {len(clinical)} clinical records "
            "matched cortical-thickness data."
        )

    converter = (
        data["DX.new"].isin(["CN", "MCI"])
        & data["ad_status"].eq("Dement")
    )

    data["model_group"] = data["DX.new"]

    data.loc[
        data["DX.new"].eq("Dement") | converter,
        "model_group",
    ] = "AD_TRAJECTORY"

    feature_names = left_columns + right_columns

    data = data[
        data["model_group"].isin(GROUP_ORDER)
    ].copy()

    zero_observation = data[feature_names].le(0).any(axis=1)
    excluded_zero_count = int(zero_observation.sum())
    data = data.loc[~zero_observation].copy()

    if data[feature_names].isna().any().any():
        raise ValueError(
            "Missing cortical-thickness values remain after preparation."
        )

    features = data[feature_names].astype("float32").reset_index(drop=True)
    labels = (
        data["model_group"]
        .map(LABEL_TO_INDEX)
        .astype("int64")
        .reset_index(drop=True)
    )

    metadata = {
        "total_observations": int(len(data)),
        "feature_count": len(feature_names),
        "excluded_zero_observations": excluded_zero_count,
        "class_counts": {
            group: int((data["model_group"] == group).sum())
            for group in GROUP_ORDER
        },
    }

    return features, labels, feature_names, metadata


if __name__ == "__main__":
    features, labels, feature_names, metadata = load_modeling_data()

    print("Cortical-thickness modeling data loaded successfully.")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Label vector shape: {labels.shape}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Class counts: {metadata['class_counts']}")
    print(
        "Zero-thickness observations excluded:",
        metadata["excluded_zero_observations"],
    )
    print("No participant-level data were exported.")