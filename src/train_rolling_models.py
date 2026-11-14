"""
Rolling-origin evaluation for Missouri county-level corn yield prediction.

Research project:
    Missouri Farm Weather Intelligence

Purpose:
    Compare basic and enhanced weather features using time-based,
    rolling-origin validation.

Rolling evaluation:
    Train 2015-2019 -> Test 2020
    Train 2015-2020 -> Test 2021
    Train 2015-2021 -> Test 2022
    Train 2015-2022 -> Test 2023
    Train 2015-2023 -> Test 2024
    Train 2015-2024 -> Test 2025

Important:
    Enhanced feature selection is performed using TRAINING DATA ONLY
    for each rolling split. This prevents information from the future
    test year from influencing feature selection.

Models:
    1. Linear Regression
    2. Random Forest
    3. Gradient Boosting

Metrics:
    MAE
    RMSE
    R²
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# 1. Configuration
# ============================================================

BASIC_FILE = "data/missouri_corn_weather_basic_ml.csv"
ENHANCED_FILE = "data/missouri_corn_weather_enhanced_ml.csv"

OUTPUT_FILE = "data/rolling_model_results.csv"
SUMMARY_FILE = "data/rolling_model_summary.csv"
FEATURE_FILE = "data/rolling_selected_features.csv"

TARGET = "corn_yield"

# ------------------------------------------------------------
# Basic features
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Enhanced candidate features
# ------------------------------------------------------------

ENHANCED_CANDIDATE_FEATURES = [
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

    "early_avg_temp",
    "early_avg_temp_max",
    "early_avg_temp_min",
    "early_total_precipitation",
    "early_max_daily_precipitation",
    "early_avg_humidity",
    "early_avg_wind",
    "early_avg_solar",
    "early_hot_days_30",
    "early_hot_days_32",
    "early_hot_days_35",
    "early_rainy_days",
    "early_heavy_rain_days",
    "early_dry_days",
    "early_longest_dry_spell",

    "mid_avg_temp",
    "mid_avg_temp_max",
    "mid_avg_temp_min",
    "mid_total_precipitation",
    "mid_max_daily_precipitation",
    "mid_avg_humidity",
    "mid_avg_wind",
    "mid_avg_solar",
    "mid_hot_days_30",
    "mid_hot_days_32",
    "mid_hot_days_35",
    "mid_rainy_days",
    "mid_heavy_rain_days",
    "mid_dry_days",
    "mid_longest_dry_spell",

    "late_avg_temp",
    "late_avg_temp_max",
    "late_avg_temp_min",
    "late_total_precipitation",
    "late_max_daily_precipitation",
    "late_avg_humidity",
    "late_avg_wind",
    "late_avg_solar",
    "late_hot_days_30",
    "late_hot_days_32",
    "late_hot_days_35",
    "late_rainy_days",
    "late_heavy_rain_days",
    "late_dry_days",
    "late_longest_dry_spell",
]


# ============================================================
# 2. Rolling evaluation years
# ============================================================

ROLLING_SPLITS = [
    (2019, 2020),
    (2020, 2021),
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
    (2024, 2025),
]


# ============================================================
# 3. Model definitions
# ============================================================

def create_models():
    """
    Create fresh model objects for each experiment.
    """

    return {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_features="sqrt",
            min_samples_leaf=3,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=5,
            random_state=42,
        ),
    }


# ============================================================
# 4. Feature selection
# ============================================================

def select_features_from_training_data(
    train_df,
    candidate_features,
    target,
    correlation_threshold=0.85,
):
    """
    Select enhanced features using TRAINING DATA ONLY.

    The procedure removes one feature from highly correlated pairs.
    When two features have absolute correlation >= threshold, the
    feature with the weaker absolute correlation with the target
    is removed.

    This is intentionally simple and reproducible.

    Important:
        Feature selection is repeated separately for every
        rolling-origin training period.
    """

    available_features = [
        feature
        for feature in candidate_features
        if feature in train_df.columns
    ]

    correlation_data = train_df[
        available_features + [target]
    ].corr()

    target_correlations = (
        correlation_data[target]
        .drop(target)
        .abs()
    )

    feature_correlations = correlation_data.loc[
        available_features,
        available_features,
    ].abs()

    selected = set(available_features)

    # Repeatedly remove one member of highly correlated pairs.
    changed = True

    while changed:

        changed = False

        current_features = list(selected)

        for i in range(len(current_features)):

            feature_1 = current_features[i]

            if feature_1 not in selected:
                continue

            for j in range(i + 1, len(current_features)):

                feature_2 = current_features[j]

                if feature_2 not in selected:
                    continue

                correlation = feature_correlations.loc[
                    feature_1,
                    feature_2,
                ]

                if correlation >= correlation_threshold:

                    corr_1 = target_correlations[
                        feature_1
                    ]

                    corr_2 = target_correlations[
                        feature_2
                    ]

                    if corr_1 >= corr_2:
                        remove_feature = feature_2
                    else:
                        remove_feature = feature_1

                    selected.remove(remove_feature)

                    changed = True
                    break

            if changed:
                break

    # Preserve original feature order.
    selected_features = [
        feature
        for feature in available_features
        if feature in selected
    ]

    return selected_features


# ============================================================
# 5. Evaluation metrics
# ============================================================

def calculate_metrics(y_true, predictions):

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    return mae, rmse, r2


# ============================================================
# 6. Load datasets
# ============================================================

print("=" * 75)
print("ROLLING-ORIGIN MODEL EVALUATION")
print("=" * 75)

print("\nLoading datasets...")

basic_df = pd.read_csv(BASIC_FILE)
enhanced_df = pd.read_csv(ENHANCED_FILE)

print("\nBasic dataset:")
print(f"  Rows    : {len(basic_df)}")
print(f"  Columns : {len(basic_df.columns)}")

print("\nEnhanced dataset:")
print(f"  Rows    : {len(enhanced_df)}")
print(f"  Columns : {len(enhanced_df.columns)}")


# ============================================================
# 7. Validate datasets
# ============================================================

for name, data in [
    ("Basic", basic_df),
    ("Enhanced", enhanced_df),
]:

    if TARGET not in data.columns:
        raise ValueError(
            f"{TARGET} is missing from {name} dataset."
        )

    if "Year" not in data.columns:
        raise ValueError(
            f"Year is missing from {name} dataset."
        )

    duplicate_count = data.duplicated(
        subset=["County", "Year"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{name} dataset contains "
            f"{duplicate_count} duplicate County-Year records."
        )


# ============================================================
# 8. Run rolling experiments
# ============================================================

all_results = []
all_selected_features = []

for train_end_year, test_year in ROLLING_SPLITS:

    print("\n")
    print("=" * 75)
    print(
        f"TRAIN {basic_df['Year'].min()}-{train_end_year}"
        f"  ->  TEST {test_year}"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Basic data split
    # --------------------------------------------------------

    basic_train = basic_df[
        basic_df["Year"] <= train_end_year
    ].copy()

    basic_test = basic_df[
        basic_df["Year"] == test_year
    ].copy()

    # --------------------------------------------------------
    # Enhanced data split
    # --------------------------------------------------------

    enhanced_train = enhanced_df[
        enhanced_df["Year"] <= train_end_year
    ].copy()

    enhanced_test = enhanced_df[
        enhanced_df["Year"] == test_year
    ].copy()

    print("\nObservations:")
    print(f"  Training : {len(basic_train)}")
    print(f"  Test     : {len(basic_test)}")

    # ========================================================
    # Feature sets
    # ========================================================

    feature_sets = {
        "Basic": BASIC_FEATURES,
    }

    # --------------------------------------------------------
    # Enhanced feature selection
    # --------------------------------------------------------

    selected_enhanced_features = (
        select_features_from_training_data(
            train_df=enhanced_train,
            candidate_features=ENHANCED_CANDIDATE_FEATURES,
            target=TARGET,
            correlation_threshold=0.85,
        )
    )

    feature_sets["Enhanced"] = selected_enhanced_features

    print(
        "\nEnhanced features selected from training data:"
    )

    print(
        f"  {len(selected_enhanced_features)} "
        f"of {len(ENHANCED_CANDIDATE_FEATURES)}"
    )

    for feature in selected_enhanced_features:
        print(f"    - {feature}")

    # Save selected features for later research documentation.
    for feature in selected_enhanced_features:

        all_selected_features.append({
            "Train_End_Year": train_end_year,
            "Test_Year": test_year,
            "Feature_Set": "Enhanced",
            "Feature": feature,
        })

    # ========================================================
    # Train models
    # ========================================================

    for feature_set_name, features in feature_sets.items():

        print("\n")
        print("-" * 75)
        print(f"Feature set: {feature_set_name}")
        print(f"Number of features: {len(features)}")
        print("-" * 75)

        X_train = basic_train[features] \
            if feature_set_name == "Basic" \
            else enhanced_train[features]

        y_train = basic_train[TARGET] \
            if feature_set_name == "Basic" \
            else enhanced_train[TARGET]

        X_test = basic_test[features] \
            if feature_set_name == "Basic" \
            else enhanced_test[features]

        y_test = basic_test[TARGET] \
            if feature_set_name == "Basic" \
            else enhanced_test[TARGET]

        models = create_models()

        for model_name, model in models.items():

            print(
                f"\nTraining {model_name}..."
            )

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            model.fit(
                X_train,
                y_train,
            )

            # ------------------------------------------------
            # Predict future year
            # ------------------------------------------------

            predictions = model.predict(
                X_test
            )

            # ------------------------------------------------
            # Calculate metrics
            # ------------------------------------------------

            mae, rmse, r2 = calculate_metrics(
                y_test,
                predictions,
            )

            print(
                f"  MAE  : {mae:.3f}"
            )

            print(
                f"  RMSE : {rmse:.3f}"
            )

            print(
                f"  R²   : {r2:.3f}"
            )

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            all_results.append({

                "Feature_Set": feature_set_name,

                "Model": model_name,

                "Train_Start_Year": int(
                    basic_df["Year"].min()
                ),

                "Train_End_Year": train_end_year,

                "Test_Year": test_year,

                "Training_Observations": len(
                    X_train
                ),

                "Test_Observations": len(
                    X_test
                ),

                "Number_of_Features": len(
                    features
                ),

                "MAE": mae,

                "RMSE": rmse,

                "R2": r2,
            })


# ============================================================
# 9. Create results DataFrame
# ============================================================

results_df = pd.DataFrame(
    all_results
)

selected_features_df = pd.DataFrame(
    all_selected_features
)


# ============================================================
# 10. Save detailed results
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

selected_features_df.to_csv(
    FEATURE_FILE,
    index=False,
)


# ============================================================
# 11. Create summary
# ============================================================

summary_df = (
    results_df
    .groupby(
        ["Feature_Set", "Model"],
        as_index=False
    )
    .agg(
        Mean_MAE=("MAE", "mean"),
        Mean_RMSE=("RMSE", "mean"),
        Mean_R2=("R2", "mean"),

        Std_MAE=("MAE", "std"),
        Std_RMSE=("RMSE", "std"),
        Std_R2=("R2", "std"),

        Median_MAE=("MAE", "median"),
        Median_RMSE=("RMSE", "median"),
        Median_R2=("R2", "median"),
    )
)


# ============================================================
# 12. Save summary
# ============================================================

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# 13. Display detailed results
# ============================================================

print("\n")
print("=" * 75)
print("DETAILED ROLLING-ORIGIN RESULTS")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# ============================================================
# 14. Display summary
# ============================================================

print("\n")
print("=" * 75)
print("ROLLING-ORIGIN SUMMARY")
print("=" * 75)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# ============================================================
# 15. Output files
# ============================================================

print("\n")
print("=" * 75)
print("FILES SAVED")
print("=" * 75)

print(
    f"\nDetailed results:"
    f"\n  {OUTPUT_FILE}"
)

print(
    f"\nSummary results:"
    f"\n  {SUMMARY_FILE}"
)

print(
    f"\nSelected features:"
    f"\n  {FEATURE_FILE}"
)

print("\nRolling-origin evaluation completed successfully.")