import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "missouri_corn_weather_basic_ml.csv"
)


# =========================================================
# 2. Load dataset
# =========================================================

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# =========================================================
# 3. Define features and target
# =========================================================

TARGET = "corn_yield"

ID_COLUMNS = [
    "County",
    "Year"
]

FEATURES = [
    column
    for column in df.columns
    if column not in ID_COLUMNS + [TARGET]
]


X = df[FEATURES]
y = df[TARGET]


# =========================================================
# 4. Time-based split
# =========================================================
#
# Training:   2015–2022
# Validation: 2023
# Test:       2024–2025
#
# The test years are completely unseen during training.
# =========================================================

TRAIN_END = 2022
VALIDATION_YEAR = 2023
TEST_START = 2024


train_df = df[df["Year"] <= TRAIN_END].copy()

validation_df = df[
    df["Year"] == VALIDATION_YEAR
].copy()

test_df = df[
    df["Year"] >= TEST_START
].copy()


X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_validation = validation_df[FEATURES]
y_validation = validation_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# =========================================================
# 5. Display split information
# =========================================================

print("\n==============================================")
print("TIME-BASED DATA SPLIT")
print("==============================================")

print(
    f"Training years: "
    f"{train_df['Year'].min()}–{train_df['Year'].max()}"
)

print(
    f"Training observations: {len(train_df)}"
)

print(
    f"Validation year: {VALIDATION_YEAR}"
)

print(
    f"Validation observations: {len(validation_df)}"
)

print(
    f"Test years: "
    f"{test_df['Year'].min()}–{test_df['Year'].max()}"
)

print(
    f"Test observations: {len(test_df)}"
)


# =========================================================
# 6. Define models
# =========================================================

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
    )
}


# =========================================================
# 7. Evaluation function
# =========================================================

def evaluate_model(model, X_data, y_data):

    predictions = model.predict(X_data)

    mae = mean_absolute_error(
        y_data,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_data,
            predictions
        )
    )

    r2 = r2_score(
        y_data,
        predictions
    )

    return mae, rmse, r2


# =========================================================
# 8. Train and evaluate models
# =========================================================

results = []


for model_name, model in models.items():

    print("\n----------------------------------------------")
    print(model_name)
    print("----------------------------------------------")

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Validation
    val_mae, val_rmse, val_r2 = evaluate_model(
        model,
        X_validation,
        y_validation
    )

    # Test
    test_mae, test_rmse, test_r2 = evaluate_model(
        model,
        X_test,
        y_test
    )

    print(
        f"Validation MAE:  {val_mae:.3f}"
    )

    print(
        f"Validation RMSE: {val_rmse:.3f}"
    )

    print(
        f"Validation R²:   {val_r2:.3f}"
    )

    print(
        f"Test MAE:        {test_mae:.3f}"
    )

    print(
        f"Test RMSE:       {test_rmse:.3f}"
    )

    print(
        f"Test R²:         {test_r2:.3f}"
    )

    results.append(
        {
            "model": model_name,

            "validation_mae": val_mae,
            "validation_rmse": val_rmse,
            "validation_r2": val_r2,

            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2
        }
    )


# =========================================================
# 9. Results table
# =========================================================

results_df = pd.DataFrame(results)

print("\n==============================================")
print("BASELINE MODEL RESULTS")
print("==============================================")

print(
    results_df.to_string(index=False)
)


# =========================================================
# 10. Save results
# =========================================================

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "baseline_model_results.csv"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nResults saved to:\n{OUTPUT_FILE}"
)

print("\nBaseline modeling completed.")