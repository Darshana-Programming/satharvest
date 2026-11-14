import pandas as pd

# --------------------------------------------------
# File paths
# --------------------------------------------------

weather_file = "data/missouri_county_weather_2015_2025.csv"
output_file = "data/missouri_county_weather_features_2015_2025.csv"


# --------------------------------------------------
# Load NASA weather data
# --------------------------------------------------

print("Loading NASA weather data...")

weather = pd.read_csv(weather_file)

weather["date"] = pd.to_datetime(weather["date"])

print("Weather data shape:")
print(weather.shape)


# --------------------------------------------------
# Remove NASA missing-value code
# --------------------------------------------------

weather_columns = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOTCORR",
    "RH2M",
    "WS10M",
    "ALLSKY_SFC_SW_DWN"
]

for column in weather_columns:
    weather[column] = weather[column].replace(-999, pd.NA)


# --------------------------------------------------
# Select growing season
# April 1 - September 30
# --------------------------------------------------

weather = weather[
    (weather["date"].dt.month >= 4) &
    (weather["date"].dt.month <= 9)
].copy()

print("\nGrowing-season data shape:")
print(weather.shape)


# --------------------------------------------------
# Create Year column
# --------------------------------------------------

weather["Year"] = weather["date"].dt.year


# --------------------------------------------------
# Create hot-day indicator
# Maximum temperature > 30°C
# --------------------------------------------------

weather["hot_day"] = (
    weather["T2M_MAX"] > 30
).astype(int)


# --------------------------------------------------
# Aggregate daily weather to county-year
# --------------------------------------------------

features = (
    weather
    .groupby(["County", "Year"])
    .agg(
        avg_temp=("T2M", "mean"),
        avg_temp_max=("T2M_MAX", "mean"),
        avg_temp_min=("T2M_MIN", "mean"),
        total_precipitation=("PRECTOTCORR", "sum"),
        avg_humidity=("RH2M", "mean"),
        avg_wind=("WS10M", "mean"),
        avg_solar=("ALLSKY_SFC_SW_DWN", "mean"),
        hot_days=("hot_day", "sum")
    )
    .reset_index()
)


# --------------------------------------------------
# Round numeric values
# --------------------------------------------------

numeric_columns = [
    "avg_temp",
    "avg_temp_max",
    "avg_temp_min",
    "total_precipitation",
    "avg_humidity",
    "avg_wind",
    "avg_solar"
]

features[numeric_columns] = features[numeric_columns].round(2)


# --------------------------------------------------
# Save dataset
# --------------------------------------------------

features.to_csv(output_file, index=False)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n========================================")
print("WEATHER FEATURE CREATION COMPLETE")
print("========================================")

print("\nOutput file:")
print(output_file)

print("\nShape:")
print(features.shape)

print("\nColumns:")
print(features.columns.tolist())

print("\nNumber of counties:")
print(features["County"].nunique())

print("\nNumber of years:")
print(features["Year"].nunique())

print("\nFirst 10 rows:")
print(features.head(10))

print("\nLast 10 rows:")
print(features.tail(10))

print("\nMissing values:")
print(features.isna().sum())