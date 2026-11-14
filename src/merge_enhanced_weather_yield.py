from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WEATHER_FILE = (
    PROJECT_ROOT
    / "data"
    / "missouri_county_weather_enhanced_features_2015_2025.csv"
)

YIELD_FILE = (
    PROJECT_ROOT
    / "data"
    / "missouri_corn_yield_2015_2025.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "missouri_corn_weather_enhanced_ml.csv"
)


# ============================================================
# Main Function
# ============================================================

def main():

    print("=" * 70)
    print("Merging Enhanced Weather Features with Corn Yield")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check input files
    # --------------------------------------------------------

    if not WEATHER_FILE.exists():
        raise FileNotFoundError(
            f"\nWeather file not found:\n{WEATHER_FILE}"
        )

    if not YIELD_FILE.exists():
        raise FileNotFoundError(
            f"\nYield file not found:\n{YIELD_FILE}"
        )

    # --------------------------------------------------------
    # 2. Load datasets
    # --------------------------------------------------------

    weather = pd.read_csv(WEATHER_FILE)

    yield_data = pd.read_csv(YIELD_FILE)

    print("\nEnhanced weather dataset shape:")
    print(weather.shape)

    print("\nCorn yield dataset shape:")
    print(yield_data.shape)

    # --------------------------------------------------------
    # 3. Display columns
    # --------------------------------------------------------

    print("\nWeather columns:")
    print(weather.columns.tolist())

    print("\nYield columns:")
    print(yield_data.columns.tolist())

    # --------------------------------------------------------
    # 4. Validate required columns
    # --------------------------------------------------------

    weather_required = [
        "County",
        "Year",
    ]

    yield_required = [
        "County",
        "Year",
        "Value",
    ]

    missing_weather = [
        column
        for column in weather_required
        if column not in weather.columns
    ]

    missing_yield = [
        column
        for column in yield_required
        if column not in yield_data.columns
    ]

    if missing_weather:
        raise ValueError(
            "\nMissing weather columns:\n"
            + "\n".join(missing_weather)
        )

    if missing_yield:
        raise ValueError(
            "\nMissing yield columns:\n"
            + "\n".join(missing_yield)
        )

    # --------------------------------------------------------
    # 5. Prepare yield data
    # --------------------------------------------------------

    yield_data = yield_data[
        [
            "County",
            "Year",
            "Value",
        ]
    ].copy()

    yield_data = yield_data.rename(
        columns={
            "Value": "corn_yield"
        }
    )

    # --------------------------------------------------------
    # 6. Make sure County-Year is unique
    # --------------------------------------------------------

    weather_duplicates = (
        weather
        .duplicated(
            subset=[
                "County",
                "Year",
            ]
        )
        .sum()
    )

    yield_duplicates = (
        yield_data
        .duplicated(
            subset=[
                "County",
                "Year",
            ]
        )
        .sum()
    )

    print(
        "\nWeather duplicate County-Year records:"
    )
    print(weather_duplicates)

    print(
        "\nYield duplicate County-Year records:"
    )
    print(yield_duplicates)

    if weather_duplicates > 0:
        raise ValueError(
            "Weather dataset contains duplicate "
            "County-Year records."
        )

    if yield_duplicates > 0:
        raise ValueError(
            "Yield dataset contains duplicate "
            "County-Year records."
        )

    # --------------------------------------------------------
    # 7. Make sure Year has the same type
    # --------------------------------------------------------

    weather["Year"] = pd.to_numeric(
        weather["Year"],
        errors="coerce"
    )

    yield_data["Year"] = pd.to_numeric(
        yield_data["Year"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # 8. Merge
    # --------------------------------------------------------

    print(
        "\nMerging datasets using County + Year..."
    )

    merged = weather.merge(
        yield_data,
        on=[
            "County",
            "Year",
        ],
        how="inner",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # 9. Sort
    # --------------------------------------------------------

    merged = (
        merged
        .sort_values(
            [
                "County",
                "Year",
            ]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 10. Validation
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print("\nEnhanced weather shape:")
    print(weather.shape)

    print("\nYield shape:")
    print(yield_data.shape)

    print("\nMerged shape:")
    print(merged.shape)

    # --------------------------------------------------------
    # Number of counties
    # --------------------------------------------------------

    print("\nNumber of counties:")
    print(
        merged["County"].nunique()
    )

    # --------------------------------------------------------
    # Number of years
    # --------------------------------------------------------

    print("\nNumber of years:")
    print(
        merged["Year"].nunique()
    )

    # --------------------------------------------------------
    # Year range
    # --------------------------------------------------------

    print("\nYear range:")
    print(
        merged["Year"].min(),
        "to",
        merged["Year"].max()
    )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicates = (
        merged
        .duplicated(
            subset=[
                "County",
                "Year",
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

    missing = merged.isna().sum()

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
    # Yield statistics
    # --------------------------------------------------------

    print(
        "\nCorn yield statistics:"
    )

    print(
        merged["corn_yield"]
        .describe()
    )

    # --------------------------------------------------------
    # Records by year
    # --------------------------------------------------------

    print(
        "\nRecords by year:"
    )

    print(
        merged["Year"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Yield preview
    # --------------------------------------------------------

    print(
        "\nFirst 10 records:"
    )

    preview_columns = [
        "County",
        "Year",
        "corn_yield",
        "avg_temp",
        "total_precipitation",
        "hot_days_30",
        "hot_days_32",
        "hot_days_35",
        "longest_dry_spell",
    ]

    print(
        merged[
            preview_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 11. Save dataset
    # ========================================================

    merged.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print(
        "Enhanced weather-yield dataset created successfully."
    )
    print("=" * 70)

    print("\nOutput file:")
    print(OUTPUT_FILE)

    print("\nFinal shape:")
    print(merged.shape)

    print(
        "\nTotal columns:"
    )
    print(
        len(merged.columns)
    )


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main()