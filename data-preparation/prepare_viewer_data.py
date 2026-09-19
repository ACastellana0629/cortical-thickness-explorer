import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DATA = PROJECT_ROOT / "private_data"
OUTPUT_FILE = PROJECT_ROOT / "public" / "data" / "cortical_summary.json"

CLINICAL_FILE = (
    PRIVATE_DATA / "dxmerge_rsfmri_baselinedata_2022_08_22.csv"
)
LH_FILE = PRIVATE_DATA / "aparc_stats_dkt_thickness_lh.txt"
RH_FILE = PRIVATE_DATA / "aparc_stats_dkt_thickness_rh.txt"

GROUP_ORDER = ["CN", "MCI", "AD_TRAJECTORY"]


def prepare_hemisphere(path, id_column, prefix):
    """Read one hemisphere and extract participant ID and scan date."""
    data = pd.read_csv(path, sep="\t")

    extracted = data[id_column].str.extract(
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

    return data[["PTID", "scan_date", *thickness_columns]], thickness_columns


def readable_region_name(region):
    """Convert compact FreeSurfer labels into readable text."""
    replacements = {
        "bankssts": "Banks of Superior Temporal Sulcus",
        "caudalanteriorcingulate": "Caudal Anterior Cingulate",
        "caudalmiddlefrontal": "Caudal Middle Frontal",
        "cuneus": "Cuneus",
        "entorhinal": "Entorhinal",
        "fusiform": "Fusiform",
        "inferiorparietal": "Inferior Parietal",
        "inferiortemporal": "Inferior Temporal",
        "isthmuscingulate": "Isthmus Cingulate",
        "lateraloccipital": "Lateral Occipital",
        "lateralorbitofrontal": "Lateral Orbitofrontal",
        "lingual": "Lingual",
        "medialorbitofrontal": "Medial Orbitofrontal",
        "middletemporal": "Middle Temporal",
        "parahippocampal": "Parahippocampal",
        "paracentral": "Paracentral",
        "parsopercularis": "Pars Opercularis",
        "parsorbitalis": "Pars Orbitalis",
        "parstriangularis": "Pars Triangularis",
        "pericalcarine": "Pericalcarine",
        "postcentral": "Postcentral",
        "posteriorcingulate": "Posterior Cingulate",
        "precentral": "Precentral",
        "precuneus": "Precuneus",
        "rostralanteriorcingulate": "Rostral Anterior Cingulate",
        "rostralmiddlefrontal": "Rostral Middle Frontal",
        "superiorfrontal": "Superior Frontal",
        "superiorparietal": "Superior Parietal",
        "superiortemporal": "Superior Temporal",
        "supramarginal": "Supramarginal",
        "frontalpole": "Frontal Pole",
        "temporalpole": "Temporal Pole",
        "transversetemporal": "Transverse Temporal",
        "insula": "Insula",
    }

    return replacements.get(region, region.replace("_", " ").title())


def main():
    clinical = pd.read_csv(CLINICAL_FILE)

    clinical["scan_date"] = pd.to_datetime(
        clinical["scandate"],
        format="%m/%d/%y",
        errors="raise",
    )

    lh, lh_columns = prepare_hemisphere(
        LH_FILE,
        "lh.aparc.thickness",
        "lh",
    )

    rh, rh_columns = prepare_hemisphere(
        RH_FILE,
        "rh.aparc.thickness",
        "rh",
    )

    imaging = lh.merge(
        rh,
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
            f"Only {len(data)} of {len(clinical)} clinical records matched CT data."
        )

    # Preserve the original clinical trajectory for scientific interpretation.
    data["trajectory_source"] = "UNCLASSIFIED"

    data.loc[data["DX.new"] == "CN", "trajectory_source"] = "BASELINE_CN"
    data.loc[data["DX.new"] == "MCI", "trajectory_source"] = "BASELINE_MCI"
    data.loc[
        data["DX.new"] == "Dement",
        "trajectory_source",
    ] = "BASELINE_AD"

    converter = (
        data["DX.new"].isin(["CN", "MCI"])
        & data["ad_status"].eq("Dement")
    )

    data.loc[converter, "trajectory_source"] = "FUTURE_CONVERTER"

    # Updated three-class outcome used after the converter analysis.
    data["model_group"] = data["DX.new"]

    data.loc[
        data["DX.new"].eq("Dement") | converter,
        "model_group",
    ] = "AD_TRAJECTORY"

    thickness_columns = lh_columns + rh_columns

    # Exclude records without a defined baseline diagnostic group.
    data = data[data["model_group"].isin(GROUP_ORDER)].copy()

    # User-specified QC: drop an observation if any regional CT value is zero.
    zero_observation = data[thickness_columns].le(0).any(axis=1)
    excluded_zero_count = int(zero_observation.sum())
    data = data.loc[~zero_observation].copy()

    cohort_counts = {
        group: int((data["model_group"] == group).sum())
        for group in GROUP_ORDER
    }

    print("\nFinal CT viewer cohort:")
    for group, count in cohort_counts.items():
        print(f"  {group}: {count}")
    print(f"  TOTAL: {len(data)}")
    print(f"\nObservations excluded for zero CT: {excluded_zero_count}")

    long_data = data.melt(
        id_vars=["model_group"],
        value_vars=thickness_columns,
        var_name="region_full",
        value_name="thickness_mm",
    )

    parsed = long_data["region_full"].str.extract(
        r"^(lh|rh)_(.+)_thickness$"
    )

    long_data["hemisphere_code"] = parsed[0]
    long_data["region"] = parsed[1]

    summary = (
        long_data.groupby(
            ["hemisphere_code", "region", "model_group"],
            observed=True,
        )["thickness_mm"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )

    regions = []

    for (hemisphere_code, region), region_data in summary.groupby(
        ["hemisphere_code", "region"],
        sort=True,
    ):
        hemisphere = "Left" if hemisphere_code == "lh" else "Right"
        values_by_group = {}

        for group in GROUP_ORDER:
            row = region_data[region_data["model_group"] == group].iloc[0]

            values_by_group[group] = {
                "n": int(row["count"]),
                "mean": round(float(row["mean"]), 4),
                "sd": round(float(row["std"]), 4),
                "median": round(float(row["median"]), 4),
            }

        cn_mean = values_by_group["CN"]["mean"]
        mci_mean = values_by_group["MCI"]["mean"]
        ad_mean = values_by_group["AD_TRAJECTORY"]["mean"]

        regions.append(
            {
                "id": f"{hemisphere_code}_{region}",
                "hemisphere": hemisphere,
                "hemisphereCode": hemisphere_code,
                "region": region,
                "displayName": (
                    f"{hemisphere} {readable_region_name(region)}"
                ),
                "groups": values_by_group,
                "comparisons": {
                    "AD_TRAJECTORY_minus_CN": round(ad_mean - cn_mean, 4),
                    "AD_TRAJECTORY_minus_MCI": round(ad_mean - mci_mean, 4),
                    "MCI_minus_CN": round(mci_mean - cn_mean, 4),
                },
            }
        )

    payload = {
        "metadata": {
            "title": "Cortical Thickness Explorer",
            "measurement": "Regional cortical thickness",
            "unit": "mm",
            "atlas": "FreeSurfer aparc, 34 regions per hemisphere",
            "regionCount": len(regions),
            "groups": GROUP_ORDER,
            "cohortCounts": cohort_counts,
            "totalParticipants": int(len(data)),
            "excludedZeroThicknessObservations": excluded_zero_count,
            "groupDefinition": {
                "CN": "Baseline cognitively normal, without observed conversion to dementia.",
                "MCI": "Baseline mild cognitive impairment, without observed conversion to dementia.",
                "AD_TRAJECTORY": (
                    "Dementia at baseline or subsequent conversion to dementia."
                ),
            },
            "privacy": (
                "This file contains aggregate regional summaries only. "
                "It contains no participant identifiers or participant-level records."
            ),
        },
        "regions": regions,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)

    print(f"\nWrote anonymous viewer data to:\n  {OUTPUT_FILE}")
    print(f"Regions exported: {len(regions)}")


if __name__ == "__main__":
    main()