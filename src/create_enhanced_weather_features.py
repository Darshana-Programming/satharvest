from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "missouri_county_weather_2015_2025.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "missouri_county_weather_enhanced_features_2015_2025.csv"
)


# Growing season: April through September
GROWING_SEASON_START_MONTH = 4
GROWING_SEASON_END_MONTH = 9


# Temperature thresholds in Celsius
HOT_THRESHOLD_30 = 30
HOT_THRESHOLD_32 = 32
HOT_THRESHOLD_35 = 35


# Precipitation thresholds in mm/day
RAINY_DAY_THRESHOLD = 1.0
HEAVY_RAIN_THRESHOLD = 25.0


# ============================================================
# Helper Functions
# ============================================================

def longest_consecutive_dry_days(group):
    """
    Calculate the longest consecutive sequence of dry days
    for one County-Year group.

    A dry day is defined as precipitation < 1 mm.
    """

    is_dry = (
        group["PRECTOTCORR"] < RAINY_DAY_THRESHOLD
    ).astype(int)

    # Identify consecutive groups of dry/non-dry days.
    groups = is_dry.ne(is_dry.shift()).cumsum()

    # Count consecutive dry days.
    dry_lengths = (
        is_dry
        .groupby(groups)
        .sum()
    )

    if len(dry_lengths) == 0:
        return 0

    return int(dry_lengths.max())


def calculate_stage_features(group, stage_name):
    """
    Calculate weather features for one growing-season stage.

    Stages:
        early = April-May
        mid   = June-July
        late  = August-September
    """

    result = {}

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    result[f"{stage_name}_avg_temp"] = (
        group["T2M"].mean()
    )

    result[f"{stage_name}_avg_temp_max"] = (
        group["T2M_MAX"].mean()
    )

    result[f"{stage_name}_avg_temp_min"] = (
        group["T2M_MIN"].mean()
    )

    # --------------------------------------------------------
    # Precipitation
    # --------------------------------------------------------

    result[f"{stage_name}_total_precipitation"] = (
        group["PRECTOTCORR"].sum()
    )

    result[f"{stage_name}_max_daily_precipitation"] = (
        group["PRECTOTCORR"].max()
    )

    # --------------------------------------------------------
    # Other weather variables
    # --------------------------------------------------------

    result[f"{stage_name}_avg_humidity"] = (
        group["RH2M"].mean()
    )

    result[f"{stage_name}_avg_wind"] = (
        group["WS10M"].mean()
    )

    result[f"{stage_name}_avg_solar"] = (
        group["ALLSKY_SFC_SW_DWN"].mean()
    )

    # --------------------------------------------------------
    # Heat stress
    # --------------------------------------------------------

    result[f"{stage_name}_hot_days_30"] = (
        group["T2M_MAX"] > HOT_THRESHOLD_30
    ).sum()

    result[f"{stage_name}_hot_days_32"] = (
        group["T2M_MAX"] > HOT_THRESHOLD_32
    ).sum()

    result[f"{stage_name}_hot_days_35"] = (
        group["T2M_MAX"] > HOT_THRESHOLD_35
    ).sum()

    # --------------------------------------------------------
    # Rainfall events
    # --------------------------------------------------------

    result[f"{stage_name}_rainy_days"] = (
        group["PRECTOTCORR"] >= RAINY_DAY_THRESHOLD
    ).sum()

    result[f"{stage_name}_heavy_rain_days"] = (
        group["PRECTOTCORR"] >= HEAVY_RAIN_THRESHOLD
    ).sum()

    # --------------------------------------------------------
    # Dry days
    # --------------------------------------------------------

    result[f"{stage_name}_dry_days"] = (
        group["PRECTOTCORR"] < RAINY_DAY_THRESHOLD
    ).sum()

    # --------------------------------------------------------
    # Longest dry spell
    # --------------------------------------------------------

    result[f"{stage_name}_longest_dry_spell"] = (
        longest_consecutive_dry_days(group)
    )

    return result


# ============================================================
# Main Function
# ============================================================

def main():

    print("=" * 70)
    print("Creating Enhanced County-Year Weather Features")
    print("=" * 70)

    # ========================================================
    # 1. Check input file
    # ========================================================

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}\n"
        )

    print("\nInput file:")
    print(INPUT_FILE)

    # ========================================================
    # 2. Load NASA weather data
    # ========================================================

    df = pd.read_csv(INPUT_FILE)

    print("\nOriginal dataset shape:")
    print(df.shape)

    print("\nOriginal columns:")
    print(df.columns.tolist())

    # ========================================================
    # 3. Handle date column
    # ========================================================

    # Your current NASA dataset uses "date".
    # This code also supports "Date".

    if "date" in df.columns:

        df = df.rename(
            columns={
                "date": "Date"
            }
        )

    elif "Date" not in df.columns:

        raise ValueError(
            "\nThe dataset must contain either "
            "'date' or 'Date'."
        )

    # ========================================================
    # 4. Validate required columns
    # ========================================================

    required_columns = [
        "County",
        "Date",
        "T2M",
        "T2M_MAX",
        "T2M_MIN",
        "PRECTOTCORR",
        "RH2M",
        "WS10M",
        "ALLSKY_SFC_SW_DWN",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nThe following required columns are missing:\n"
            + "\n".join(missing_columns)
        )

    # ========================================================
    # 5. Convert Date column
    # ========================================================

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    invalid_dates = df["Date"].isna().sum()

    if invalid_dates > 0:

        raise ValueError(
            f"\nFound {invalid_dates} invalid dates."
        )

    # ========================================================
    # 6. Replace NASA missing-value code
    # ========================================================

    weather_columns = [
        "T2M",
        "T2M_MAX",
        "T2M_MIN",
        "PRECTOTCORR",
        "RH2M",
        "WS10M",
        "ALLSKY_SFC_SW_DWN",
    ]

    # NASA POWER may use -999 for missing values.
    df[weather_columns] = (
        df[weather_columns]
        .replace(-999, np.nan)
    )

    # ========================================================
    # 7. Create Year and Month
    # ========================================================

    df["Year"] = df["Date"].dt.year

    df["Month"] = df["Date"].dt.month

    # ========================================================
    # 8. Keep April-September growing season
    # ========================================================

    growing = df[
        (df["Month"] >= GROWING_SEASON_START_MONTH)
        & (df["Month"] <= GROWING_SEASON_END_MONTH)
    ].copy()

    print("\nGrowing-season dataset shape:")
    print(growing.shape)

    # ========================================================
    # 9. Check missing weather values
    # ========================================================

    missing_before = (
        growing[weather_columns]
        .isna()
        .sum()
        .sum()
    )

    if missing_before > 0:

        print(
            f"\nMissing weather values found: "
            f"{missing_before}"
        )

        growing = growing.dropna(
            subset=weather_columns
        )

        print(
            "Rows remaining after removing "
            "missing values:",
            len(growing)
        )

    else:

        print("\nNo missing weather values found.")

    # ========================================================
    # 10. Create daily indicator variables
    # ========================================================

    # Heat thresholds
    growing["hot_day_30"] = (
        growing["T2M_MAX"] > HOT_THRESHOLD_30
    ).astype(int)

    growing["hot_day_32"] = (
        growing["T2M_MAX"] > HOT_THRESHOLD_32
    ).astype(int)

    growing["hot_day_35"] = (
        growing["T2M_MAX"] > HOT_THRESHOLD_35
    ).astype(int)

    # Rainfall
    growing["rainy_day"] = (
        growing["PRECTOTCORR"]
        >= RAINY_DAY_THRESHOLD
    ).astype(int)

    growing["heavy_rain_day"] = (
        growing["PRECTOTCORR"]
        >= HEAVY_RAIN_THRESHOLD
    ).astype(int)

    # Dry day
    growing["dry_day"] = (
        growing["PRECTOTCORR"]
        < RAINY_DAY_THRESHOLD
    ).astype(int)

    # ========================================================
    # 11. Assign growing-season stages
    # ========================================================

    # Early season: April-May
    # Mid season: June-July
    # Late season: August-September

    growing["Stage"] = np.select(
        [
            growing["Month"].isin([4, 5]),
            growing["Month"].isin([6, 7]),
            growing["Month"].isin([8, 9]),
        ],
        [
            "early",
            "mid",
            "late",
        ],
        default="unknown",
    )

    # Remove anything unexpected.
    growing = growing[
        growing["Stage"] != "unknown"
    ].copy()

    print("\nStage distribution:")
    print(
        growing["Stage"]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # 12. Full growing-season features
    # ========================================================

    print(
        "\nCalculating full growing-season features..."
    )

    basic_features = (
        growing
        .groupby(
            ["County", "Year"]
        )
        .agg(

            # Temperature
            avg_temp=(
                "T2M",
                "mean"
            ),

            avg_temp_max=(
                "T2M_MAX",
                "mean"
            ),

            avg_temp_min=(
                "T2M_MIN",
                "mean"
            ),

            # Precipitation
            total_precipitation=(
                "PRECTOTCORR",
                "sum"
            ),

            max_daily_precipitation=(
                "PRECTOTCORR",
                "max"
            ),

            # Other weather
            avg_humidity=(
                "RH2M",
                "mean"
            ),

            avg_wind=(
                "WS10M",
                "mean"
            ),

            avg_solar=(
                "ALLSKY_SFC_SW_DWN",
                "mean"
            ),

            # Heat stress
            hot_days_30=(
                "hot_day_30",
                "sum"
            ),

            hot_days_32=(
                "hot_day_32",
                "sum"
            ),

            hot_days_35=(
                "hot_day_35",
                "sum"
            ),

            # Rainfall events
            rainy_days=(
                "rainy_day",
                "sum"
            ),

            heavy_rain_days=(
                "heavy_rain_day",
                "sum"
            ),

            # Dry days
            dry_days=(
                "dry_day",
                "sum"
            ),
        )
        .reset_index()
    )

    # ========================================================
    # 13. Longest dry spell
    # ========================================================

    print(
        "Calculating longest dry spells..."
    )

    dry_spell = (
        growing
        .groupby(
            ["County", "Year"],
            group_keys=False
        )
        .apply(
            longest_consecutive_dry_days,
            include_groups=False
        )
        .reset_index(
            name="longest_dry_spell"
        )
    )

    # ========================================================
    # 14. Merge longest dry spell
    # ========================================================

    features = basic_features.merge(
        dry_spell,
        on=[
            "County",
            "Year"
        ],
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # 15. Stage-specific features
    # ========================================================

    print(
        "Calculating stage-specific features..."
    )

    stage_results = []

    for (
        county,
        year
    ), group in growing.groupby(
        ["County", "Year"]
    ):

        row = {
            "County": county,
            "Year": year,
        }

        # ----------------------------------------------
        # Early, mid, and late stages
        # ----------------------------------------------

        for stage_name in [
            "early",
            "mid",
            "late"
        ]:

            stage_group = group[
                group["Stage"] == stage_name
            ]

            if len(stage_group) > 0:

                stage_features = (
                    calculate_stage_features(
                        stage_group,
                        stage_name
                    )
                )

                row.update(
                    stage_features
                )

            else:

                # This should normally not happen,
                # but keep the code robust.

                feature_names = [

                    "avg_temp",
                    "avg_temp_max",
                    "avg_temp_min",

                    "total_precipitation",
                    "max_daily_precipitation",

                    "avg_humidity",
                    "avg_wind",
                    "avg_solar",

                    "hot_days_30",
                    "hot_days_32",
                    "hot_days_35",

                    "rainy_days",
                    "heavy_rain_days",
                    "dry_days",

                    "longest_dry_spell",
                ]

                for feature_name in feature_names:

                    row[
                        f"{stage_name}_{feature_name}"
                    ] = np.nan

        stage_results.append(row)

    stage_features_df = pd.DataFrame(
        stage_results
    )

    # ========================================================
    # 16. Merge stage features
    # ========================================================

    features = features.merge(
        stage_features_df,
        on=[
            "County",
            "Year"
        ],
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # 17. Sort
    # ========================================================

    features = (
        features
        .sort_values(
            [
                "County",
                "Year"
            ]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 18. Validation
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print("\nFinal dataset shape:")
    print(features.shape)

    print("\nNumber of counties:")
    print(
        features["County"].nunique()
    )

    print("\nNumber of years:")
    print(
        features["Year"].nunique()
    )

    print("\nYear range:")
    print(
        features["Year"].min(),
        "to",
        features["Year"].max()
    )

    # --------------------------------------------------------
    # Duplicate County-Year records
    # --------------------------------------------------------

    duplicates = (
        features
        .duplicated(
            subset=[
                "County",
                "Year"
            ]
        )
        .sum()
    )

    print(
        "\nDuplicate County-Year records:"
    )
    print(duplicates)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing = features.isna().sum()

    missing_columns = (
        missing[missing > 0]
    )

    print(
        "\nColumns with missing values:"
    )

    if len(missing_columns) == 0:

        print("None")

    else:

        print(missing_columns)

    # --------------------------------------------------------
    # Expected combinations
    # --------------------------------------------------------

    expected_counties = 97

    expected_years = 11

    expected_combinations = (
        expected_counties
        * expected_years
    )

    print(
        "\nExpected county-year combinations:"
    )

    print(
        expected_combinations
    )

    print(
        "Actual county-year combinations:"
    )

    print(
        len(features)
    )

    # ========================================================
    # 19. Preview
    # ========================================================

    print(
        "\nSelected feature preview:"
    )

    preview_columns = [

        "County",
        "Year",

        "avg_temp",
        "avg_temp_max",
        "avg_temp_min",

        "total_precipitation",

        "hot_days_30",
        "hot_days_32",
        "hot_days_35",

        "rainy_days",
        "heavy_rain_days",

        "dry_days",
        "max_daily_precipitation",

        "longest_dry_spell",
    ]

    print(
        features[
            preview_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 20. Save output
    # ========================================================

    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # 21. Final information
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "Enhanced feature dataset created successfully."
    )
    print("=" * 70)

    print("\nOutput file:")
    print(OUTPUT_FILE)

    print("\nFinal shape:")
    print(features.shape)

    print(
        "\nTotal number of columns:"
    )

    print(
        len(features.columns)
    )

    print(
        "\nFeature columns:"
    )

    for column in features.columns:

        print(
            f"  - {column}"
        )


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main()