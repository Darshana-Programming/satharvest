import pandas as pd

# --------------------------------------------------
# File paths
# --------------------------------------------------

weather_file = "data/missouri_county_weather_features_2015_2025.csv"
yield_file = "data/missouri_corn_yield_2015_2025.csv"
output_file = "data/missouri_corn_weather_ml.csv"


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("Loading weather features...")
weather = pd.read_csv(weather_file)

print("Loading USDA corn yield data...")
yield_data = pd.read_csv(yield_file)

print("\nWeather data shape:")
print(weather.shape)

print("\nUSDA yield data shape:")
print(yield_data.shape)


# --------------------------------------------------
# Prepare USDA data
# --------------------------------------------------

yield_data = yield_data.rename(
    columns={"Value": "corn_yield"}
)

# Keep only the columns needed
yield_data = yield_data[
    ["Year", "County", "corn_yield"]
]


# --------------------------------------------------
# Check duplicate County-Year records
# --------------------------------------------------

print("\n========================================")
print("USDA DUPLICATE CHECK")
print("========================================")

duplicates = yield_data.duplicated(
    subset=["County", "Year"]
).sum()

print("Duplicate County-Year records:")
print(duplicates)


# --------------------------------------------------
# Merge NASA weather + USDA yield
# --------------------------------------------------

print("\n========================================")
print("MERGING DATASETS")
print("========================================")

merged = pd.merge(
    weather,
    yield_data,
    on=["County", "Year"],
    how="inner"
)

print("Merged data shape:")
print(merged.shape)


# --------------------------------------------------
# Check unique counties and years
# --------------------------------------------------

print("\nNumber of counties:")
print(merged["County"].nunique())

print("\nNumber of years:")
print(merged["Year"].nunique())


# --------------------------------------------------
# Check missing values
# --------------------------------------------------

print("\n========================================")
print("MISSING VALUE CHECK")
print("========================================")

print(merged.isna().sum())


# --------------------------------------------------
# Check duplicate County-Year records
# --------------------------------------------------

print("\n========================================")
print("MERGED DUPLICATE CHECK")
print("========================================")

merged_duplicates = merged.duplicated(
    subset=["County", "Year"]
).sum()

print("Duplicate County-Year records:")
print(merged_duplicates)


# --------------------------------------------------
# Sort data
# --------------------------------------------------

merged = merged.sort_values(
    ["County", "Year"]
).reset_index(drop=True)


# --------------------------------------------------
# Save final ML dataset
# --------------------------------------------------

merged.to_csv(output_file, index=False)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n========================================")
print("MERGE COMPLETE")
print("========================================")

print("\nOutput file:")
print(output_file)

print("\nFinal shape:")
print(merged.shape)

print("\nColumns:")
print(merged.columns.tolist())

print("\nFirst 10 rows:")
print(merged.head(10))

print("\nLast 10 rows:")
print(merged.tail(10))

print("\nRecords by year:")
print(merged.groupby("Year").size())

print("\nRecords by county:")
print(merged.groupby("County").size().head(20))

print("\nCorn yield statistics:")
print(merged["corn_yield"].describe())