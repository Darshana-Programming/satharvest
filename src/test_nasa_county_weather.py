import time

import pandas as pd
import requests

# =========================================================
# 1. Load county coordinates
# =========================================================

coordinate_file = "data/missouri_county_coordinates.csv"

coordinates = pd.read_csv(coordinate_file)

print("Available coordinate counties:")
print(coordinates["County"].tolist())


# =========================================================
# 2. Select three counties for testing
# =========================================================

test_counties = [
    "BOONE",
    "COLE",
    "JEFFERSON"
]

coordinates = coordinates[
    coordinates["County"].isin(test_counties)
].copy()

print("\nCounties selected for testing:")
print(coordinates["County"].tolist())


# =========================================================
# 3. NASA POWER date range
# =========================================================

start_date = "20150101"
end_date = "20251231"


# =========================================================
# 4. NASA POWER weather parameters
# =========================================================

parameters = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOTCORR",
    "RH2M",
    "WS10M",
    "ALLSKY_SFC_SW_DWN"
]


# =========================================================
# 5. NASA POWER API
# =========================================================

url = "https://power.larc.nasa.gov/api/temporal/daily/point"


# List to store data from all counties
all_data = []


# =========================================================
# 6. Download weather for each county
# =========================================================

for _, row in coordinates.iterrows():

    county = row["County"]
    latitude = row["Latitude"]
    longitude = row["Longitude"]

    print("\n========================================")
    print(f"Downloading weather for {county}...")
    print(f"Latitude: {latitude}")
    print(f"Longitude: {longitude}")
    print("========================================")

    params = {
        "parameters": ",".join(parameters),
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }

    try:

        # -------------------------------------------------
        # Send request to NASA POWER
        # -------------------------------------------------

        response = requests.get(
            url,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        # Convert response to JSON
        data = response.json()

        print("NASA API request successful.")


        # -------------------------------------------------
        # Get NASA weather data
        # -------------------------------------------------

        parameter_data = data["properties"]["parameter"]


        # -------------------------------------------------
        # Get dates
        # -------------------------------------------------

        # Use the first parameter to get the list of dates
        first_parameter = parameters[0]

        dates = list(
            parameter_data[first_parameter].keys()
        )


        # -------------------------------------------------
        # Create DataFrame with dates
        # -------------------------------------------------

        weather = pd.DataFrame()

        weather["date"] = dates


        # -------------------------------------------------
        # Add each weather parameter
        # -------------------------------------------------

        for parameter in parameters:

            values = parameter_data[parameter]

            weather[parameter] = [
                values.get(date)
                for date in dates
            ]


        # -------------------------------------------------
        # Convert date
        # -------------------------------------------------

        weather["date"] = pd.to_datetime(
            weather["date"],
            format="%Y%m%d"
        )


        # -------------------------------------------------
        # Add county
        # -------------------------------------------------

        weather["County"] = county


        # -------------------------------------------------
        # Add to list
        # -------------------------------------------------

        all_data.append(weather)


        print(
            f"{county}: "
            f"{len(weather)} daily records downloaded."
        )


    except Exception as e:

        print(f"ERROR downloading {county}:")
        print(e)

        continue


    # Small pause between requests
    time.sleep(1)


# =========================================================
# 7. Check whether data was downloaded
# =========================================================

if not all_data:

    print("\nNo weather data was downloaded.")
    print("Please check the NASA POWER API request.")

    raise SystemExit


# =========================================================
# 8. Combine all counties
# =========================================================

result = pd.concat(
    all_data,
    ignore_index=True
)


# =========================================================
# 9. Reorder columns
# =========================================================

result = result[
    [
        "date",
        "County",
        "T2M",
        "T2M_MAX",
        "T2M_MIN",
        "PRECTOTCORR",
        "RH2M",
        "WS10M",
        "ALLSKY_SFC_SW_DWN"
    ]
]


# =========================================================
# 10. Check for NASA missing-value code
# =========================================================

# NASA POWER can use -999 as a fill value.
# We will identify it now rather than treating it as real weather.

weather_columns = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOTCORR",
    "RH2M",
    "WS10M",
    "ALLSKY_SFC_SW_DWN"
]

print("\nChecking for NASA missing-value code (-999):")

for column in weather_columns:

    count = (result[column] == -999).sum()

    print(f"{column}: {count}")


# =========================================================
# 11. Save the data
# =========================================================

output_file = "data/test_nasa_county_weather.csv"

result.to_csv(
    output_file,
    index=False
)


# =========================================================
# 12. Display results
# =========================================================

print("\n========================================")
print("NASA weather download completed.")
print("========================================")

print("\nSaved file:")
print(output_file)

print("\nShape:")
print(result.shape)

print("\nColumns:")
print(result.columns.tolist())

print("\nNumber of records by county:")

print(
    result.groupby("County").size()
)

print("\nDate range:")

print("Start:", result["date"].min())
print("End:  ", result["date"].max())

print("\nFirst five rows:")

print(result.head())

print("\nLast five rows:")

print(result.tail())

print("\nData types:")

print(result.dtypes)