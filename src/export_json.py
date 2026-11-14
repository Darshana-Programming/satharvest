
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "frontend" / "public" / "data"

OUTPUT.mkdir(parents=True, exist_ok=True)

files = {
    "missouri_corn_yield_2015_2025.csv": "corn-yield.json",
    "missouri_corn_county_coordinates.csv": "counties.json",
    "missouri_county_weather_2015_2025.csv": "weather.json",
    "missouri_county_weather_enhanced_features_2015_2025.csv": "weather-features.json",
    "rolling_model_summary.csv": "model-summary.json",
    "enhanced_feature_yield_correlations.csv": "correlations.json",
}

for csv_name, json_name in files.items():
    df = pd.read_csv(DATA / csv_name)
    df.to_json(
        OUTPUT / json_name,
        orient="records",
        indent=2
    )
    print(f"Created {json_name}: {len(df)} records")
