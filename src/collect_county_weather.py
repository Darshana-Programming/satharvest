import pandas as pd
import requests
import time


# =========================================================
# 1. File locations
# =========================================================

coordinate_file = "data/missouri_corn_county_coordinates.csv"

output_file = "data/missouri_county_weather_2015_2025.csv"


# =========================================================
# 2. Load county coordinates
# =========================================================

coordinates = pd.read_csv(coordinate_file)

print("Number of counties:")
print(len(coordinates))

print("\nFirst five counties:")
print(coordinates.head())


# =========================================================
# 3. NASA POWER date range
# =========================================================

start_date = "20150101"
end_date = "20251231"


# =========================================================
# 4. NASA POWER parameters
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


# =========================================================
# 6. Storage
# =========================================================

all_data = []

successful_counties = []
failed_counties = []


# =========================================================
# 7. Download data
# =========================================================

for number, (_, row) in enumerate(
    coordinates.iterrows(),
    start=1
):

    county = row["County"]
    latitude = row["Latitude"]
    longitude = row["Longitude"]

    print("\n========================================")
    print(
        f"County {number} of {len(coordinates)}: "
        f"{county}"
    )
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


    # -----------------------------------------------------
    # Try the request up to 3 times
    # -----------------------------------------------------

    success = False

    for attempt in range(1, 4):

        try:

            print(
                f"Attempt {attempt}/3..."
            )

            response = requests.get(
                url,
                params=params,
                timeout=60
            )

            response.raise_for_status()

            data = response.json()

            parameter_data = (
                data["properties"]["parameter"]
            )


            # -------------------------------------------------
            # Get dates
            # -------------------------------------------------

            first_parameter = parameters[0]

            dates = list(
                parameter_data[
                    first_parameter
                ].keys()
            )


            # -------------------------------------------------
            # Create DataFrame
            # -------------------------------------------------

            weather = pd.DataFrame()

            weather["date"] = dates


            # -------------------------------------------------
            # Add weather parameters
            # -------------------------------------------------

            for parameter in parameters:

                values = parameter_data[
                    parameter
                ]

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
            # Reorder columns
            # -------------------------------------------------

            weather = weather[
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


            # -------------------------------------------------
            # Store data
            # -------------------------------------------------

            all_data.append(weather)

            successful_counties.append(county)

            print(
                f"SUCCESS: {county}"
            )

            print(
                f"Records: {len(weather)}"
            )

            success = True

            break


        except Exception as e:

            print(
                f"Error on attempt {attempt}:"
            )

            print(e)

            if attempt < 3:

                print(
                    "Waiting 5 seconds before retry..."
                )

                time.sleep(5)


    # -----------------------------------------------------
    # Record failed county
    # -----------------------------------------------------

    if not success:

        failed_counties.append(county)

        print(
            f"FAILED: {county}"
        )


    # -----------------------------------------------------
    # Pause between counties
    # -----------------------------------------------------

    time.sleep(1)


# =========================================================
# 8. Check results
# =========================================================

print("\n========================================")
print("DOWNLOAD SUMMARY")
print("========================================")

print(
    f"Successful counties: "
    f"{len(successful_counties)}"
)

print(
    f"Failed counties: "
    f"{len(failed_counties)}"
)


if failed_counties:

    print("\nFailed counties:")

    for county in failed_counties:
        print(county)


# =========================================================
# 9. Combine data
# =========================================================

if not all_data:

    print(
        "\nNo weather data was downloaded."
    )

    raise SystemExit


result = pd.concat(
    all_data,
    ignore_index=True
)


# =========================================================
# 10. Check missing-value code
# =========================================================

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
print("MISSING-VALUE CHECK")
print("========================================")

for column in weather_columns:

    count = (
        result[column] == -999
    ).sum()

    print(
        f"{column}: {count}"
    )


# =========================================================
# 11. Save data
# =========================================================

result.to_csv(
    output_file,
    index=False
)


# =========================================================
# 12. Final information
# =========================================================

print("\n========================================")
print("NASA WEATHER COLLECTION COMPLETE")
print("========================================")

print("\nOutput file:")
print(output_file)

print("\nShape:")
print(result.shape)

print("\nDate range:")
print(result["date"].min())
print(result["date"].max())

print("\nRecords per county:")

print(
    result.groupby("County").size()
)

print("\nFirst five rows:")

print(result.head())