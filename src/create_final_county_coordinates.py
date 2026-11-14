import pandas as pd


# =========================================================
# 1. File locations
# =========================================================

usda_file = "data/missouri_corn_yield_2015_2025.csv"
coordinate_file = "data/missouri_county_coordinates.csv"

output_file = "data/missouri_corn_county_coordinates.csv"


# =========================================================
# 2. Load data
# =========================================================

usda = pd.read_csv(usda_file)
coordinates = pd.read_csv(coordinate_file)


# =========================================================
# 3. Standardize county names
# =========================================================

def standardize_county_name(name):

    name = str(name).upper().strip()

    # Remove periods
    name = name.replace(".", "")

    # Remove spaces
    name = name.replace(" ", "")

    return name


usda["County_Standard"] = (
    usda["County"]
    .apply(standardize_county_name)
)

coordinates["County_Standard"] = (
    coordinates["County"]
    .apply(standardize_county_name)
)


# =========================================================
# 4. Get the 97 USDA counties
# =========================================================

usda_counties = (
    usda[
        ["County", "County_Standard"]
    ]
    .drop_duplicates()
)


# =========================================================
# 5. Remove duplicate coordinate counties
# =========================================================

coordinates = coordinates.drop_duplicates(
    subset="County_Standard"
)


# =========================================================
# 6. Keep only counties in USDA data
# =========================================================

final_coordinates = usda_counties.merge(
    coordinates[
        [
            "County_Standard",
            "Latitude",
            "Longitude"
        ]
    ],
    on="County_Standard",
    how="left"
)


# =========================================================
# 7. Keep USDA county name
# =========================================================

final_coordinates = final_coordinates[
    [
        "County",
        "Latitude",
        "Longitude"
    ]
]


# =========================================================
# 8. Check for missing coordinates
# =========================================================

missing = final_coordinates[
    final_coordinates["Latitude"].isna()
    | final_coordinates["Longitude"].isna()
]


# =========================================================
# 9. Display results
# =========================================================

print("\n========================================")
print("FINAL COUNTY COORDINATE CHECK")
print("========================================")

print("\nNumber of USDA counties:")
print(len(usda_counties))

print("\nNumber of final coordinate counties:")
print(len(final_coordinates))

print("\nCounties missing coordinates:")
print(len(missing))

if len(missing) > 0:
    print(missing["County"].tolist())


# =========================================================
# 10. Save final file
# =========================================================

if len(missing) == 0:

    final_coordinates.to_csv(
        output_file,
        index=False
    )

    print("\nFinal coordinate file saved:")
    print(output_file)

else:

    print("\nFile was NOT saved because coordinates are missing.")


# =========================================================
# 11. Display first five records
# =========================================================

print("\nFirst five records:")

print(
    final_coordinates.head()
)