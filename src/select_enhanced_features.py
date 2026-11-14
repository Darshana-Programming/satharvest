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

OUTPUT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "missouri_corn_weather_selected_ml.csv"
)

OUTPUT_REPORT = (
    PROJECT_ROOT
    / "data"
    / "feature_selection_report.csv"
)


# =========================================================
# 2. Settings
# =========================================================

CORRELATION_THRESHOLD = 0.85

TARGET = "corn_yield"

ID_COLUMNS = [
    "County",
    "Year"
]


# =========================================================
# 3. Load dataset
# =========================================================

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)


# =========================================================
# 4. Identify weather features
# =========================================================

feature_columns = [
    column
    for column in df.columns
    if column not in ID_COLUMNS + [TARGET]
]

print("Number of original weather features:", len(feature_columns))


# =========================================================
# 5. Calculate correlation with yield
# =========================================================

yield_corr = (
    df[feature_columns + [TARGET]]
    .corr(numeric_only=True)[TARGET]
    .drop(TARGET)
)

yield_corr_abs = yield_corr.abs()


# =========================================================
# 6. Calculate feature-to-feature correlation matrix
# =========================================================

feature_corr = df[feature_columns].corr()


# =========================================================
# 7. Feature-selection process
# =========================================================

selected_features = feature_columns.copy()

removed_features = []


# ---------------------------------------------------------
# Process highly correlated feature pairs
# ---------------------------------------------------------

for i in range(len(feature_columns)):

    feature_1 = feature_columns[i]

    if feature_1 not in selected_features:
        continue

    for j in range(i + 1, len(feature_columns)):

        feature_2 = feature_columns[j]

        if feature_2 not in selected_features:
            continue

        correlation = feature_corr.loc[
            feature_1,
            feature_2
        ]

        if abs(correlation) >= CORRELATION_THRESHOLD:

            corr_1 = yield_corr_abs[feature_1]
            corr_2 = yield_corr_abs[feature_2]

            # Keep the feature with stronger relationship
            # with corn yield.
            if corr_1 >= corr_2:

                keep = feature_1
                remove = feature_2

            else:

                keep = feature_2
                remove = feature_1

            if remove in selected_features:

                selected_features.remove(remove)

                removed_features.append(
                    {
                        "removed_feature": remove,
                        "kept_feature": keep,
                        "feature_correlation": correlation,
                        "removed_yield_correlation":
                            yield_corr[remove],
                        "kept_yield_correlation":
                            yield_corr[keep],
                        "reason":
                            "Highly correlated feature; "
                            "kept feature with stronger "
                            "absolute correlation with yield"
                    }
                )


# =========================================================
# 8. Create selection report
# =========================================================

report_df = pd.DataFrame(removed_features)

if not report_df.empty:

    report_df = report_df.sort_values(
        by="feature_correlation",
        key=abs,
        ascending=False
    )


# =========================================================
# 9. Create selected dataset
# =========================================================

selected_columns = (
    ID_COLUMNS
    + selected_features
    + [TARGET]
)

selected_df = df[selected_columns].copy()


# =========================================================
# 10. Save files
# =========================================================

selected_df.to_csv(
    OUTPUT_DATASET,
    index=False
)

report_df.to_csv(
    OUTPUT_REPORT,
    index=False
)


# =========================================================
# 11. Display results
# =========================================================

print("\n==============================================")
print("FEATURE SELECTION RESULTS")
print("==============================================")

print(
    "Original weather features:",
    len(feature_columns)
)

print(
    "Selected weather features:",
    len(selected_features)
)

print(
    "Removed weather features:",
    len(feature_columns) - len(selected_features)
)

print("\nSelected features:")

for feature in selected_features:
    print(f"  - {feature}")


print("\nRemoved features:")

if report_df.empty:

    print("  None")

else:

    for _, row in report_df.iterrows():

        print(
            f"  - {row['removed_feature']}"
            f"  -> kept: {row['kept_feature']}"
        )


print("\nFinal selected dataset shape:")
print(selected_df.shape)

print(
    f"\nSelected dataset saved to:\n"
    f"{OUTPUT_DATASET}"
)

print(
    f"\nFeature-selection report saved to:\n"
    f"{OUTPUT_REPORT}"
)

print("\nFeature selection completed.")