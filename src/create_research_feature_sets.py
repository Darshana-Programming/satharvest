import pandas as pd
from pathlib import Path


# =========================================================
# 1. File paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "missouri_corn_weather_enhanced_ml.csv"
)

BASIC_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "missouri_corn_weather_basic_ml.csv"
)

ENHANCED_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "missouri_corn_weather_enhanced_selected_ml.csv"
)

ALL_FEATURES_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "missouri_corn_weather_all_features_ml.csv"
)


# =========================================================
# 2. Load dataset
# =========================================================

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)


# =========================================================
# 3. Define columns
# =========================================================

ID_COLUMNS = [
    "County",
    "Year"
]

TARGET = "corn_yield"


# =========================================================
# 4. Basic feature set
# =========================================================
#
# These represent simple growing-season summaries.
# This will be our baseline weather model.
# =========================================================

BASIC_FEATURES = [
    "avg_temp",
    "avg_temp_max",
    "avg_temp_min",
    "total_precipitation",
    "avg_humidity",
    "avg_wind",
    "avg_solar",
    "hot_days_30",
]


# =========================================================
# 5. Enhanced feature set
# =========================================================
#
# These features represent:
# - temperature
# - extreme heat
# - precipitation
# - rainfall frequency
# - dry spells
# - wind
# - solar radiation
# - humidity
# - temporal/stage-specific conditions
#
# This is our initial research candidate set.
# =========================================================

ENHANCED_FEATURES = [
    # Temperature
    "avg_temp_min",

    # Heat stress
    "mid_hot_days_35",
    "late_hot_days_35",

    # Precipitation
    "total_precipitation",
    "mid_max_daily_precipitation",
    "early_total_precipitation",

    # Rainfall frequency
    "rainy_days",
    "mid_rainy_days",

    # Dryness
    "late_longest_dry_spell",
    "early_longest_dry_spell",

    # Wind
    "early_avg_wind",

    # Solar radiation
    "avg_solar",
    "mid_avg_solar",
    "late_avg_solar",

    # Humidity
    "late_avg_humidity",
]


# =========================================================
# 6. All weather features
# =========================================================

ALL_FEATURES = [
    column
    for column in df.columns
    if column not in ID_COLUMNS + [TARGET]
]


# =========================================================
# 7. Check that all requested features exist
# =========================================================

def check_features(feature_list, name):

    missing = [
        feature
        for feature in feature_list
        if feature not in df.columns
    ]

    if missing:

        print(f"\nERROR: Missing {name} features:")

        for feature in missing:
            print(f"  - {feature}")

        raise ValueError(
            f"{name} feature list contains missing columns."
        )


check_features(
    BASIC_FEATURES,
    "basic"
)

check_features(
    ENHANCED_FEATURES,
    "enhanced"
)

check_features(
    ALL_FEATURES,
    "all"
)


# =========================================================
# 8. Create datasets
# =========================================================

basic_columns = (
    ID_COLUMNS
    + BASIC_FEATURES
    + [TARGET]
)

enhanced_columns = (
    ID_COLUMNS
    + ENHANCED_FEATURES
    + [TARGET]
)

all_columns = (
    ID_COLUMNS
    + ALL_FEATURES
    + [TARGET]
)


basic_df = df[basic_columns].copy()

enhanced_df = df[enhanced_columns].copy()

all_features_df = df[all_columns].copy()


# =========================================================
# 9. Save datasets
# =========================================================

basic_df.to_csv(
    BASIC_OUTPUT,
    index=False
)

enhanced_df.to_csv(
    ENHANCED_OUTPUT,
    index=False
)

all_features_df.to_csv(
    ALL_FEATURES_OUTPUT,
    index=False
)


# =========================================================
# 10. Display results
# =========================================================

print("\n==============================================")
print("RESEARCH FEATURE SETS")
print("==============================================")

print("\nBasic feature set:")
print(f"Number of weather features: {len(BASIC_FEATURES)}")
for feature in BASIC_FEATURES:
    print(f"  - {feature}")

print("\nEnhanced feature set:")
print(f"Number of weather features: {len(ENHANCED_FEATURES)}")
for feature in ENHANCED_FEATURES:
    print(f"  - {feature}")

print("\nAll-feature set:")
print(f"Number of weather features: {len(ALL_FEATURES)}")


print("\n==============================================")
print("DATASET SHAPES")
print("==============================================")

print(
    "Basic:",
    basic_df.shape
)

print(
    "Enhanced:",
    enhanced_df.shape
)

print(
    "All features:",
    all_features_df.shape
)


print("\n==============================================")
print("OUTPUT FILES")
print("==============================================")

print(BASIC_OUTPUT)
print(ENHANCED_OUTPUT)
print(ALL_FEATURES_OUTPUT)


# =========================================================
# 11. Validation
# =========================================================

for name, dataset in [
    ("Basic", basic_df),
    ("Enhanced", enhanced_df),
    ("All features", all_features_df)
]:

    missing_count = dataset.isna().sum().sum()

    duplicate_count = dataset.duplicated(
        subset=["County", "Year"]
    ).sum()

    print(
        f"\n{name} dataset validation:"
    )

    print(
        "  Missing values:",
        missing_count
    )

    print(
        "  Duplicate County-Year records:",
        duplicate_count
    )


print("\nResearch feature-set creation completed.")