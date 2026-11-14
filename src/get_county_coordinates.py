import geopandas as gpd
import pandas as pd

# Census 2025 county boundaries
url = "https://www2.census.gov/geo/tiger/TIGER2025/COUNTY/tl_2025_us_county.zip"

print("Loading Census county boundaries...")
counties = gpd.read_file(url)

# Missouri FIPS code = 29
missouri = counties[counties["STATEFP"] == "29"].copy()

print("Missouri county/equivalent records:", len(missouri))

# Calculate representative points.
# representative_point() stays inside the county boundary.
missouri["point"] = missouri.geometry.representative_point()

# Convert to latitude/longitude
coordinates = pd.DataFrame({
    "County": missouri["NAME"].str.upper(),
    "Latitude": missouri["point"].y,
    "Longitude": missouri["point"].x
})

print("Complete the finding")
# Save
output_file = "data/missouri_county_coordinates.csv"
coordinates.to_csv(output_file, index=False)

print("\nSaved:", output_file)
print("\nFirst five rows:")
print(coordinates.head())