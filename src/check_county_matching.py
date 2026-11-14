import pandas as pd


# =========================================================
# 1. File locations
# =========================================================

usda_file = "data/missouri_corn_yield_2015_2025.csv"
coordinate_file = "data/missouri_county_coordinates.csv"


# =========================================================
# 2. Load the files
# =========================================================

usda = pd.read_csv(usda_file)
coordinates = pd.read_csv(coordinate_file)


print("USDA data shape:")
print(usda.shape)

print("\nCoordinate data shape:")
print(coordinates.shape)


# =========================================================
# 3. Create standardized county names
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
# 4. Get unique county names
# =========================================================

usda_counties = set(
    usda["County_Standard"].unique()
)

coordinate_counties = set(
    coordinates["County_Standard"].unique()
)


# =========================================================
# 5. Find USDA counties without coordinates
# =========================================================

missing_coordinates = sorted(
    usda_counties - coordinate_counties
)


# =========================================================
# 6. Find extra coordinate counties
# =========================================================

extra_coordinates = sorted(
    coordinate_counties - usda_counties
)


# =========================================================
# 7. Display results
# =========================================================

print("\n========================================")
print("COUNTY MATCHING RESULTS")
print("========================================")

print("\nUSDA unique counties:")
print(len(usda_counties))

print("\nCoordinate unique counties:")
print(len(coordinate_counties))


print("\n----------------------------------------")
print("USDA counties missing coordinates:")
print("----------------------------------------")

if missing_coordinates:

    for county in missing_coordinates:
        print(county)

else:

    print("None")


print("\nNumber missing:")
print(len(missing_coordinates))


print("\n----------------------------------------")
print("Extra coordinate counties:")
print("----------------------------------------")

if extra_coordinates:

    for county in extra_coordinates:
        print(county)

else:

    print("None")


print("\nNumber of extra counties:")
print(len(extra_coordinates))


# =========================================================
# 8. Show original names for mismatches
# =========================================================

print("\n========================================")
print("ORIGINAL USDA NAMES")
print("========================================")

for county in missing_coordinates:

    original_names = (
        usda.loc[
            usda["County_Standard"] == county,
            "County"
        ]
        .unique()
    )

    print(county, "->", original_names)


print("\n========================================")
print("ORIGINAL COORDINATE NAMES")
print("========================================")

for county in missing_coordinates:

    original_names = (
        coordinates.loc[
            coordinates["County_Standard"] == county,
            "County"
        ]
        .unique()
    )

    print(county, "->", original_names)