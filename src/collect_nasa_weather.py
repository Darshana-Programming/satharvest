import pandas as pd
import requests

# --------------------------------------------------
# 1. Location: Jefferson City, Missouri
# --------------------------------------------------
print("in file collect nasa weather")
latitude = 38.5767
longitude = -92.1735

start_date = "20150101"
end_date = "20251231"


# --------------------------------------------------
# 2. NASA POWER parameters
# --------------------------------------------------

parameters = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOTCORR",
    "RH2M",
    "WS10M",
    "ALLSKY_SFC_SW_DWN"
]


# --------------------------------------------------
# 3. NASA POWER API
# --------------------------------------------------

url = "https://power.larc.nasa.gov/api/temporal/daily/point"

params = {
    "parameters": ",".join(parameters),
    "community": "AG",
    "longitude": longitude,
    "latitude": latitude,
    "start": start_date,
    "end": end_date,
    "format": "JSON"
}


# --------------------------------------------------
# 4. Request the data
# --------------------------------------------------

print("Downloading NASA POWER data...")

response = requests.get(url, params=params)

response.raise_for_status()

data = response.json()

print("Download completed.")


# --------------------------------------------------
# 5. Extract the weather data
# --------------------------------------------------

weather_data = data["properties"]["parameter"]

df = pd.DataFrame(weather_data)


# --------------------------------------------------
# 6. Convert date from index to a column
# --------------------------------------------------

df.index = pd.to_datetime(df.index, format="%Y%m%d")

df.index.name = "date"

df.reset_index(inplace=True)


# --------------------------------------------------
# 7. Rename columns
# --------------------------------------------------

df.rename(
    columns={
        "T2M": "temp_avg",
        "T2M_MAX": "temp_max",
        "T2M_MIN": "temp_min",
        "PRECTOTCORR": "precipitation",
        "RH2M": "humidity",
        "WS10M": "wind",
        "ALLSKY_SFC_SW_DWN": "solar"
    },
    inplace=True
)


# --------------------------------------------------
# 8. Add location information
# --------------------------------------------------

df["location"] = "Jefferson City, Missouri"


# --------------------------------------------------
# 9. Arrange columns
# --------------------------------------------------

df = df[
    [
        "date",
        "location",
        "temp_avg",
        "temp_max",
        "temp_min",
        "precipitation",
        "humidity",
        "wind",
        "solar"
    ]
]


# --------------------------------------------------
# 10. Save the data
# --------------------------------------------------

output_file = "../data/jefferson_city_weather_2015_2025.csv"

df.to_csv(output_file, index=False)

print(f"Data saved to: {output_file}")

print(f"Number of records: {len(df)}")

print("\nFirst five records:")
print(df.head())

