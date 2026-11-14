import pandas as pd

input_file = "data/missouri_corn_yield_2015_2025.csv"

# --------------------------------------------------
# 1. Read the raw USDA data
# --------------------------------------------------


df = pd.read_csv(input_file)


# --------------------------------------------------
# 2. Display the original data
# --------------------------------------------------

print("Original shape:")
print(df.shape)

print("\nOriginal data:")
print(df.head())


# --------------------------------------------------
# 3. Check missing values
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# 4. Check data types
# --------------------------------------------------

print("\nData types:")
print(df.dtypes)

print("==================")
print("Shape:")
print(df.shape)

print("\nYears:")
print(sorted(df["Year"].unique()))

print("\nNumber of counties:")
print(df["County"].nunique())

print("\nMissing values:")
print(df.isnull().sum())

print("\nRecords by year:")
print(df.groupby("Year").size())

print("\n---unique county-----")
print(df["County"].sort_values().unique())

df = df[~df["County"].isin([
    "OTHER COUNTIES",
    "OTHER (COMBINED) COUNTIES"
])]

# Save cleaned data
df.to_csv("data/missouri_corn_yield_2015_2025.csv", index=False)

print("Shape:", df.shape)
print("Number of counties:", df["County"].nunique())

print("\nCounty names:")
print(df["County"].sort_values().unique())