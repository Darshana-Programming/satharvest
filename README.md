**SatHarvest — From Space to Soil**

Exploring Missouri corn yields using NASA POWER weather data (2015–2025)

SatHarvest is a project that connects historical NASA POWER weather observations with county-level Missouri corn yield records. It combines Python data processing, weather-feature engineering, exploratory analysis, machine-learning experiments, and a Next.js dashboard for exploring weather and agricultural patterns.

**Scope:** SatHarvest explores historical associations and evaluates research models. It does not establish that weather alone causes changes in corn yield.

**Project goals**
- Collect and organize daily NASA POWER weather observations for Missouri counties.
- Match county-year weather features with historical corn yield records.
- Examine growing-season temperature, precipitation, humidity, heat, and dry spells.
- Compare basic and enhanced weather features in machine-learning experiments.
- Make historical patterns accessible through an interactive dashboard.
**Data sources**
- NASA POWER: Daily meteorological observations retrieved through the NASA POWER API.
- USDA agricultural data: Missouri county-level corn yield records, expressed in bushels per acre.
The project covers 2015–2025. The prepared dashboard datasets previously verified in the project contained 681 corn-yield observations, 1,067 county-year weather-feature records, and 389,746 daily weather records. All 681 yield observations had a matching county-year weather-feature record. These counts describe the prepared datasets, not necessarily every raw file.

Workflow
County coordinates ──> NASA POWER daily weather
                              │
                              ▼
                  Growing-season weather features
                              │
USDA county corn yields ──────┤
                              ▼
                    County + Year integration
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
        Historical dashboard         Machine-learning evaluation

Growing-season features
The feature-engineering code uses April–September and divides the season into:
Period  	Months
Early	    April–May
Mid	      June–July
Late	    August–September

Derived variables include average and maximum/minimum temperature, total and maximum daily precipitation, humidity, wind, solar radiation, hot-day counts, rainy and heavy-rain days, dry-day counts, and longest dry spells. 
In the enhanced feature script, hot-day thresholds are 30°C, 32°C, and 35°C; a rainy day is at least 1 mm of precipitation, and a heavy-rain day is at least 25 mm.

**Python scripts (src/)**
**Script	Role**

get_county_coordinates.py: 	                  Extracts Missouri county coordinate information.
check_county_coordinates.py: 	                Checks coverage of county coordinates.
check_county_matching.py: 	                  Compares county names across source datasets.
create_final_county_coordinates.py: 	        Prepares coordinates used for county weather collection.
collect_county_weather.py: 	                  Requests daily NASA POWER observations for the selected county coordinates.
test_nasa_county_weather.py: 	                Tests county-level weather requests.
check_weather_data.py: 	                      Checks the downloaded weather data.
inspect_usda_corn.py: 	                      Inspects the corn yield records.
create_weather_features.py: 	                Aggregates daily weather into basic county-year features.
create_enhanced_weather_features.py: 	        Produces enhanced and seasonal weather indicators.
merge_weather_yield.py:	                      Joins basic weather features with yield data.
merge_enhanced_weather_yield.py:	            Joins enhanced weather features with yield data.
explore_data.py: 	                            Explores the combined datasets.
analyze_enhanced_features.py: 	              Examines feature–yield correlations and feature relationships.
select_enhanced_features.py: 	                Supports weather-feature selection.
create_research_feature_sets.py: 	            Prepares alternative feature sets for experiments.
train_baseline_models.py: 	                  Evaluates models using basic features.
train_enhanced_models.py: 	                  Evaluates models using enhanced features.
train_rolling_models.py: 	                    Runs time-based rolling evaluations of basic and enhanced feature sets.

**Data files (data/)
Source and collected data
File	                                   Purpose**
missouri_corn_yield_2015_2025.csv: 	    County-level corn yield observations.
corn_row_data.csv	:                     Retained corn-related source data.
missouri_county_coordinates.csv:  	    County coordinate data.
missouri_corn_county_coordinates.csv:	  Coordinates used for corn-county weather collection.
missouri_county_weather_2015_2025.csv:	Collected daily NASA POWER weather observations.
test_nasa_county_weather.csv:        	  Data from initial county API tests.


**Prepared datasets
File	                                                  Purpose**
missouri_county_weather_features_2015_2025.csv       	Basic county-year weather features.
missouri_county_weather_enhanced_features_2015_2025.csv	Enhanced seasonal and extreme-weather features.
missouri_corn_weather_ml.csv	Combined weather and yield data.
missouri_corn_weather_basic_ml.csv	Basic-feature modeling dataset.
missouri_corn_weather_enhanced_ml.csv	Enhanced-feature modeling dataset.
missouri_corn_weather_enhanced_selected_ml.csv	Selected enhanced-feature dataset.
missouri_corn_weather_all_features_ml.csv	Combined feature dataset.
missouri_corn_weather_selected_ml.csv	Selected-feature modeling dataset.


Analysis and evaluation outputs
File	Purpose
enhanced_feature_yield_correlations.csv	Feature–yield correlations.
highly_correlated_feature_pairs.csv	Highly correlated feature pairs.
feature_selection_report.csv	Feature selection results.
baseline_model_results.csv	Basic-model experiment results.
enhanced_model_results.csv	Enhanced-model experiment results.
rolling_model_results.csv	Results for individual rolling test years.
rolling_model_summary.csv	Aggregate rolling evaluation metrics.
rolling_selected_features.csv	Selected features for rolling splits.


File availability may differ in the cleaned GitHub repository; the tables document the original analysis workflow.
Machine-learning evaluation
The rolling evaluation in src/train_rolling_models.py uses expanding historical training windows:
Training years	Test year
2015–2019	2020
2015–2020	2021
2015–2021	2022
2015–2022	2023
2015–2023	2024
2015–2024	2025


It compares Linear Regression, Random Forest, and Gradient Boosting. Enhanced-feature selection is performed using training data only within each split.
Results: mean MAE across rolling test years
Model	Basic features	Enhanced features
Linear Regression	29.15	27.98
Random Forest	26.81	22.57
Gradient Boosting	26.21	21.94


MAE is in bushels per acre; lower is better. Values come from data/rolling_model_summary.csv in the original project archive.

Enhanced features reduced mean MAE for all three algorithms in these experiments. Enhanced Gradient Boosting had the lowest mean MAE, but its mean R² was approximately −0.073, so the evaluation does not demonstrate reliable future-year forecasting.
Interactive dashboard
The Next.js dashboard is designed for historical data exploration. Its developed views include:
- County selection and historical corn-yield trends.
- Annual and growing-season weather indicators.
- Weather-versus-yield scatter plots and correlation displays.
- Historical yield and weather tables.
The dashboard uses prepared JSON exports of the analysis datasets. The latest frontend source was developed in GitHub Codespaces and was not part of the original Python archive used to verify the backend details above.
Dashboard screenshots
Add screenshots from the running SatHarvest application here.

Dashboard overview
<!-- Insert screenshot here, for example: ![SatHarvest dashboard](docs/images/dashboard-overview.png) -->
Historical corn yield and weather indicators
<!-- Insert screenshot here -->
Weather-versus-yield comparison
<!-- Insert screenshot here -->Technologies
- Python: Pandas, NumPy, Requests, scikit-learn
- Data: NASA POWER API, USDA county corn yield records, CSV and JSON
- Frontend: Next.js, React, Tailwind CSS
- Development: GitHub, GitHub Codespaces
Running the project
The Python scripts generally expect to run from the Python project root because several input and output paths begin with data/.
cd satharvest
python src/train_rolling_models.py
The command above is an example for a repository containing the nested satharvest/ Python folder and the required input CSV files. It reruns the rolling evaluation and updates its output CSV files.

To run the frontend from the repository root:
cd my-app
npm install
npm run dev
Open the forwarded development URL in Codespaces. The dashboard also requires its prepared JSON files under my-app/public/data/.
Limitations
- Historical associations do not establish causation.
- The dataset does not represent every factor affecting corn yields.
- Model errors vary by evaluation year.
- The machine-learning results are experimental, not operational forecasts.
- Weather observations for future growing seasons are not supplied by the historical datasets.
Conclusion
SatHarvest demonstrates a workflow for connecting NASA POWER weather observations with Missouri agricultural records. Its data processing, seasonal feature engineering, exploratory analysis, rolling model evaluation, and dashboard provide a foundation for studying historical weather–yield relationships.

Developed for the NASA Space Apps Challenge.
