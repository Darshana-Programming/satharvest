import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load data
# --------------------------------------------------

file = "data/missouri_corn_weather_ml.csv"

df = pd.read_csv(file)

print("========================================")
print("DATASET OVERVIEW")
print("========================================")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())


# --------------------------------------------------
# Descriptive statistics
# --------------------------------------------------

print("\n========================================")
print("DESCRIPTIVE STATISTICS")
print("========================================")

print(df.describe())


# --------------------------------------------------
# Corn yield by year
# --------------------------------------------------

print("\n========================================")
print("CORN YIELD BY YEAR")
print("========================================")

year_summary = (
    df.groupby("Year")["corn_yield"]
    .agg(["count", "mean", "std", "min", "max"])
    .round(2)
)

print(year_summary)


# --------------------------------------------------
# Corn yield by county
# --------------------------------------------------

print("\n========================================")
print("CORN YIELD BY COUNTY")
print("========================================")

county_summary = (
    df.groupby("County")["corn_yield"]
    .agg(["count", "mean", "std", "min", "max"])
    .sort_values("mean", ascending=False)
    .round(2)
)

print(county_summary)


# --------------------------------------------------
# Correlation matrix
# --------------------------------------------------

print("\n========================================")
print("CORRELATION WITH CORN YIELD")
print("========================================")

numeric_columns = [
    "avg_temp",
    "avg_temp_max",
    "avg_temp_min",
    "total_precipitation",
    "avg_humidity",
    "avg_wind",
    "avg_solar",
    "hot_days",
    "corn_yield"
]

correlations = (
    df[numeric_columns]
    .corr()["corn_yield"]
    .sort_values(ascending=False)
)

print(correlations.round(3))


# --------------------------------------------------
# Highest and lowest yields
# --------------------------------------------------

print("\n========================================")
print("HIGHEST CORN YIELD OBSERVATIONS")
print("========================================")

print(
    df.nlargest(10, "corn_yield")[
        ["County", "Year", "corn_yield"]
    ]
)


print("\n========================================")
print("LOWEST CORN YIELD OBSERVATIONS")
print("========================================")

print(
    df.nsmallest(10, "corn_yield")[
        ["County", "Year", "corn_yield"]
    ]
)


# --------------------------------------------------
# Plot 1: Average corn yield by year
# --------------------------------------------------

year_yield = df.groupby("Year")["corn_yield"].mean()

plt.figure(figsize=(10, 5))
plt.plot(year_yield.index, year_yield.values, marker="o")
plt.xlabel("Year")
plt.ylabel("Average Corn Yield (bu/acre)")
plt.title("Average Missouri County Corn Yield by Year")
plt.grid(True)
plt.tight_layout()
plt.savefig("data/average_corn_yield_by_year.png", dpi=300)
plt.show()


# --------------------------------------------------
# Plot 2: Correlation with corn yield
# --------------------------------------------------

feature_correlations = (
    df[numeric_columns]
    .corr()["corn_yield"]
    .drop("corn_yield")
    .sort_values()
)

plt.figure(figsize=(10, 5))
feature_correlations.plot(kind="bar")
plt.xlabel("Weather Feature")
plt.ylabel("Correlation with Corn Yield")
plt.title("Weather Feature Correlations with Corn Yield")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("data/weather_yield_correlations.png", dpi=300)
plt.show()


# --------------------------------------------------
# Plot 3: Yield distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))
plt.hist(df["corn_yield"], bins=25)
plt.xlabel("Corn Yield (bu/acre)")
plt.ylabel("Number of Observations")
plt.title("Distribution of County-Level Corn Yield")
plt.tight_layout()
plt.savefig("data/corn_yield_distribution.png", dpi=300)
plt.show()


print("\n========================================")
print("EDA COMPLETE")
print("========================================")

print("\nCharts saved to data/:")
print("average_corn_yield_by_year.png")
print("weather_yield_correlations.png")
print("corn_yield_distribution.png")