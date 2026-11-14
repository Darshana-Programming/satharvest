import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# 1. File paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "missouri_corn_weather_enhanced_ml.csv"

OUTPUT_YIELD_CORR = PROJECT_ROOT / "data" / "enhanced_feature_yield_correlations.csv"
OUTPUT_HIGH_CORR = PROJECT_ROOT / "data" / "highly_correlated_feature_pairs.csv"


# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ---------------------------------------------------------
# 3. Identify target and predictor features
# ---------------------------------------------------------

target = "corn_yield"

# Exclude identifiers and target
exclude_columns = ["County", "Year", target]

feature_columns = [
    col for col in df.columns
    if col not in exclude_columns
]

print("\nNumber of weather features:", len(feature_columns))
print("Target:", target)


# ---------------------------------------------------------
# 4. Calculate correlation with corn yield
# ---------------------------------------------------------

yield_correlations = (
    df[feature_columns + [target]]
    .corr(numeric_only=True)[target]
    .drop(target)
    .sort_values(key=abs, ascending=False)
)

yield_corr_df = yield_correlations.reset_index()
yield_corr_df.columns = ["feature", "correlation_with_yield"]

yield_corr_df["absolute_correlation"] = (
    yield_corr_df["correlation_with_yield"].abs()
)


# ---------------------------------------------------------
# 5. Display feature-yield correlations
# ---------------------------------------------------------

print("\n==============================================")
print("FEATURE CORRELATION WITH CORN YIELD")
print("==============================================")

print(yield_corr_df.to_string(index=False))


# ---------------------------------------------------------
# 6. Save feature-yield correlations
# ---------------------------------------------------------

yield_corr_df.to_csv(OUTPUT_YIELD_CORR, index=False)

print(
    f"\nSaved yield correlation results to:\n"
    f"{OUTPUT_YIELD_CORR}"
)


# ---------------------------------------------------------
# 7. Calculate feature-to-feature correlations
# ---------------------------------------------------------

feature_corr_matrix = df[feature_columns].corr()


# ---------------------------------------------------------
# 8. Find highly correlated feature pairs
# ---------------------------------------------------------

threshold = 0.85

high_corr_pairs = []

for i in range(len(feature_columns)):
    for j in range(i + 1, len(feature_columns)):

        feature_1 = feature_columns[i]
        feature_2 = feature_columns[j]

        correlation = feature_corr_matrix.loc[
            feature_1, feature_2
        ]

        if abs(correlation) >= threshold:

            high_corr_pairs.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": correlation,
                    "absolute_correlation": abs(correlation),
                }
            )


high_corr_df = pd.DataFrame(high_corr_pairs)

if not high_corr_df.empty:
    high_corr_df = high_corr_df.sort_values(
        "absolute_correlation",
        ascending=False
    )


# ---------------------------------------------------------
# 9. Display highly correlated pairs
# ---------------------------------------------------------

print("\n==============================================")
print(f"HIGHLY CORRELATED FEATURE PAIRS (|r| >= {threshold})")
print("==============================================")

if high_corr_df.empty:
    print("No highly correlated feature pairs found.")
else:
    print(high_corr_df.to_string(index=False))


# ---------------------------------------------------------
# 10. Save highly correlated feature pairs
# ---------------------------------------------------------

high_corr_df.to_csv(OUTPUT_HIGH_CORR, index=False)

print(
    f"\nSaved high-correlation results to:\n"
    f"{OUTPUT_HIGH_CORR}"
)


# ---------------------------------------------------------
# 11. Summary
# ---------------------------------------------------------

print("\n==============================================")
print("SUMMARY")
print("==============================================")

print("Dataset shape:", df.shape)
print("Number of weather features:", len(feature_columns))
print(
    "Number of highly correlated feature pairs:",
    len(high_corr_df)
)

print("\nTop 10 features by absolute correlation with yield:")

print(
    yield_corr_df.head(10).to_string(index=False)
)

print("\nAnalysis completed.")