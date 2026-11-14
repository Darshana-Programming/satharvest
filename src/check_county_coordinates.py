import pandas as pd

# Load USDA county data
yield_file = "data/missouri_corn_yield_2015_2025.csv"
yield_df = pd.read_csv(yield_file)

# Load county coordinates
coordinate_file = "data/missouri_county_coordinates.csv"
coordinate_df = pd.read_csv(coordinate_file)

# Get counties from USDA data
usda_counties = set(yield_df["County"].str.upper().str.strip())

# Get counties from coordinate data
coordinate_counties = set(coordinate_df["County"].str.upper().str.strip())

# Find counties that do not have coordinates
missing_coordinates = sorted(usda_counties - coordinate_counties)

print("USDA counties:", len(usda_counties))
print("Coordinate counties:", len(coordinate_counties))

print("\nCounties missing coordinates:")
print(missing_coordinates)

print("\nNumber missing:", len(missing_coordinates))