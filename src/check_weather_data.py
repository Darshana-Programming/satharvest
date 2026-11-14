import pandas as pd

weather_file = "data/missouri_county_weather_2015_2025.csv"

weather = pd.read_csv(weather_file)

print("========================================")
print("WEATHER DATA VALIDATION")
print("========================================")

print("\nShape:")
print(weather.shape)

print("\nColumns:")
print(weather.columns.tolist())

print("\nData types:")
print(weather.dtypes)

print("\nDate range:")
print(weather["date"].min())
print(weather["date"].max())

print("\nNumber of unique counties:")
print(weather["County"].nunique())

print("\nNumber of unique dates:")
print(weather["date"].nunique())


# --------------------------------------------------
# Missing values
# --------------------------------------------------

print("\n========================================")
print("MISSING VALUES")
print("========================================")

print(weather.isna().sum())


# --------------------------------------------------
# NASA -999 values
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

print("\n========================================")
print("NASA -999 CHECK")
print("========================================")

for column in weather_columns:
    count = (weather[column] == -999).sum()
    print(f"{column}: {count}")


# --------------------------------------------------
# Duplicate records
# --------------------------------------------------

print("\n========================================")
print("DUPLICATE CHECK")
print("========================================")

duplicates = weather.duplicated(
    subset=["County", "date"]
).sum()

print("Duplicate County-Date records:")
print(duplicates)


# --------------------------------------------------
# Records per county
# --------------------------------------------------

print("\n========================================")
print("RECORDS PER COUNTY")
print("========================================")

records_per_county = weather.groupby("County").size()

print(records_per_county)

print("\nMinimum records per county:")
print(records_per_county.min())

print("\nMaximum records per county:")
print(records_per_county.max())


# --------------------------------------------------
# Basic statistics
# --------------------------------------------------

print("\n========================================")
print("WEATHER STATISTICS")
print("========================================")

print(weather[weather_columns].describe())


# --------------------------------------------------
# Final validation
# --------------------------------------------------

print("\n========================================")
print("VALIDATION SUMMARY")
print("========================================")

print(f"Total records: {len(weather):,}")
print(f"Counties: {weather['County'].nunique()}")
print(f"Expected records: {97 * 4018:,}")
print(f"Duplicate County-Date records: {duplicates}")

if (
    len(weather) == 97 * 4018
    and weather["County"].nunique() == 97
    and duplicates == 0
):
    print("\nWeather dataset structure looks correct.")
else:
    print("\nPlease investigate the validation results.")