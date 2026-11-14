"""
Train machine learning models using enhanced weather features.

Research project:
    Missouri Farm Weather Intelligence

Purpose:
    Compare enhanced weather features with the basic weather-feature
    baseline for county-level corn yield prediction in Missouri.

Important:
    The enhanced feature set used here is an exploratory candidate set.
    Feature selection was performed before this script using the complete
    dataset. Therefore, these results should be treated as exploratory.
    Final research evaluation should perform feature selection within
    each training fold to avoid information leakage.

Time-based split:
    Training   : 2015-2022
    Validation : 2023
    Test       : 2024-2025

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

INPUT_FILE = "data/missouri_corn_weather_enhanced_selected_ml.csv"
OUTPUT_FILE = "data/enhanced_model_results.csv"

TARGET = "corn_yield"

# Enhanced candidate feature set
FEATURES = [
    "avg_temp_min",
    "mid_hot_days_35",
    "late_hot_days_35",
    "total_precipitation",
    "mid_max_daily_precipitation",
    "early_total_precipitation",
    "rainy_days",
    "mid_rainy_days",
    "late_longest_dry_spell",
    "early_longest_dry_spell",
    "early_avg_wind",
    "avg_solar",
    "mid_avg_solar",
    "late_avg_solar",
    "late_avg_humidity",
]

# Time periods
TRAIN_END_YEAR = 2022
VALIDATION_YEAR = 2023
TEST_START_YEAR = 2024


# ============================================================
# 2. Helper functions
# ============================================================

def calculate_metrics(y_true, predictions):
    """
    Calculate MAE, RMSE, and R².
    """

    mae = mean_absolute_error(y_true, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_true, predictions)
    )

    r2 = r2_score(y_true, predictions)

    return mae, rmse, r2


def print_metrics(dataset_name, mae, rmse, r2):
    """
    Print model evaluation metrics.
    """

    print(f"\n{dataset_name}:")
    print(f"  MAE  : {mae:.3f}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  R²   : {r2:.3f}")


# ============================================================
# 3. Load dataset
# ============================================================

print("=" * 70)
print("ENHANCED WEATHER FEATURE MODELING")
print("=" * 70)

print("\nLoading dataset...")
print(f"Input file: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print("\nDataset information")
print("-" * 70)
print(f"Rows                : {len(df)}")
print(f"Columns             : {len(df.columns)}")
print(f"Counties            : {df['County'].nunique()}")
print(f"Minimum year        : {df['Year'].min()}")
print(f"Maximum year        : {df['Year'].max()}")
print(f"Number of features  : {len(FEATURES)}")


# ============================================================
# 4. Validate required columns
# ============================================================

required_columns = (
    ["County", "Year", TARGET]
    + FEATURES
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# 5. Check missing values
# ============================================================

print("\nChecking missing values...")

missing_values = df[required_columns].isnull().sum()

missing_values = missing_values[
    missing_values > 0
]

if len(missing_values) > 0:
    print("\nMissing values found:")
    print(missing_values)

    raise ValueError(
        "Missing values must be handled before model training."
    )

else:
    print("No missing values found.")


# ============================================================
# 6. Check duplicate County-Year observations
# ============================================================

duplicate_count = df.duplicated(
    subset=["County", "Year"]
).sum()

print(
    f"\nDuplicate County-Year records: {duplicate_count}"
)

if duplicate_count > 0:
    raise ValueError(
        "Duplicate County-Year observations found."
    )


# ============================================================
# 7. Sort data by year
# ============================================================

df = df.sort_values(
    ["Year", "County"]
).reset_index(drop=True)


# ============================================================
# 8. Create time-based datasets
# ============================================================

train = df[
    df["Year"] <= TRAIN_END_YEAR
].copy()

validation = df[
    df["Year"] == VALIDATION_YEAR
].copy()

test = df[
    df["Year"] >= TEST_START_YEAR
].copy()


print("\n" + "=" * 70)
print("TIME-BASED DATA SPLIT")
print("=" * 70)

print(
    f"\nTraining   : {train['Year'].min()}-"
    f"{train['Year'].max()}"
)

print(
    f"Validation : {validation['Year'].min()}-"
    f"{validation['Year'].max()}"
)

print(
    f"Test       : {test['Year'].min()}-"
    f"{test['Year'].max()}"
)

print("\nNumber of observations:")
print(f"Training   : {len(train)}")
print(f"Validation : {len(validation)}")
print(f"Test       : {len(test)}")


# ============================================================
# 9. Prepare X and y
# ============================================================

X_train = train[FEATURES]
y_train = train[TARGET]

X_validation = validation[FEATURES]
y_validation = validation[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


# ============================================================
# 10. Display feature list
# ============================================================

print("\n" + "=" * 70)
print("FEATURES USED")
print("=" * 70)

for number, feature in enumerate(FEATURES, start=1):
    print(f"{number:2d}. {feature}")


# ============================================================
# 11. Define models
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_features="sqrt",
        min_samples_leaf=3,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=5,
        random_state=42
    ),
}


# ============================================================
# 12. Train and evaluate models
# ============================================================

results = []

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

for model_name, model in models.items():

    print(f"\n{'-' * 70}")
    print(f"Training: {model_name}")
    print(f"{'-' * 70}")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    validation_predictions = model.predict(
        X_validation
    )

    (
        validation_mae,
        validation_rmse,
        validation_r2,
    ) = calculate_metrics(
        y_validation,
        validation_predictions
    )

    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    test_predictions = model.predict(
        X_test
    )

    (
        test_mae,
        test_rmse,
        test_r2,
    ) = calculate_metrics(
        y_test,
        test_predictions
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print_metrics(
        "Validation Results",
        validation_mae,
        validation_rmse,
        validation_r2
    )

    print_metrics(
        "Test Results",
        test_mae,
        test_rmse,
        test_r2
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "Model": model_name,

        "Training_Observations": len(train),
        "Validation_Observations": len(validation),
        "Test_Observations": len(test),

        "Validation_MAE": validation_mae,
        "Validation_RMSE": validation_rmse,
        "Validation_R2": validation_r2,

        "Test_MAE": test_mae,
        "Test_RMSE": test_rmse,
        "Test_R2": test_r2,
    })


# ============================================================
# 13. Create results DataFrame
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# 14. Save results
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 15. Display final results
# ============================================================

print("\n" + "=" * 70)
print("FINAL ENHANCED MODEL RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(f"\nOutput file:")
print(OUTPUT_FILE)

print("\nModel training completed successfully.")