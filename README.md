# Climate Analysis Project

## Overview
This comprehensive climate analysis project utilizes OpenWeather API and NASA climate data to analyze temperature trends, precipitation patterns, and extreme weather events across multiple cities globally from 2015-2024. The project includes time series decomposition, trend analysis, seasonal pattern identification, predictive modeling for temperature forecasting, statistical tests for climate change detection, and interactive visualizations.

## Features
- **Multi-source Data Collection**: Retrieves climate data from OpenWeather API and NASA POWER API
- **Comprehensive Analysis**:
  - Time series decomposition of temperature and precipitation data
  - Trend analysis for detecting long-term climate changes
  - Seasonal pattern identification and analysis
  - Extreme weather event detection and characterization
- **Predictive Modeling**:
  - Temperature forecasting models using machine learning
  - Precipitation prediction models
  - Extreme weather event prediction
- **Statistical Testing**:
  - Mann-Kendall test for trend detection
  - Statistical significance testing for climate change
  - Distribution change analysis
- **Interactive Visualization**:
  - Regional climate variation maps
  - Time series visualizations of climate trends
  - Comparative analysis dashboards
  - Correlation heatmaps for geographic factors
- **Geographic Correlation**:
  - Analysis of relationships between weather patterns and geographic factors
  - Elevation, proximity to water, urbanization effects analysis

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/Saloni-Agg/Climate-Analysis.git
cd climate-analysis-project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys:
   - Create a `.env` file in the project root
   - Add your API keys:
   ```
   OPENWEATHER_API_KEY=your_openweather_api_key
   NASA_API_KEY=your_nasa_api_key
   ```

## Project Structure
```
climate_analysis_project/
├── README.md                 # Main project documentation
├── requirements.txt          # Python dependencies
├── config.yaml               # Configuration file
├── setup.py                  # Package setup file
├── .gitignore                # Git ignore file
├── src/                      # Source code
│   ├── data_collection/      # Data collection modules
│   ├── analysis/             # Analysis modules
│   ├── visualization/        # Visualization modules
│   └── utils/                # Utility functions
├── data/                     # Data storage
│   ├── raw/                  # Raw data
│   ├── processed/            # Processed data
│   └── external/             # External data
├── notebooks/                # Jupyter notebooks
├── tests/                    # Test suite
├── docs/                     # Documentation
└── scripts/                  # Utility scripts
```

## Usage

### Data Collection
To collect climate data from OpenWeather API and NASA POWER API:
```bash
python scripts/update_data.py --start-date 2015-01-01 --end-date 2024-06-01 --cities cities.json
```

### Running Analysis
To perform comprehensive climate analysis:
```bash
python scripts/run_analysis.py --config config.yaml
```

### Generating Reports
To generate visualizations and reports:
```bash
python scripts/generate_reports.py --output-dir reports/
```

### Interactive Dashboard
To launch the interactive dashboard:
```bash
python -m src.visualization.dashboard
```

## Examples

### Time Series Decomposition
```python
from src.analysis.time_series_analysis import decompose_time_series
import pandas as pd

# Load temperature data
temp_data = pd.read_csv('data/processed/temperature_data.csv', parse_dates=['date'], index_col='date')

# Perform time series decomposition
trend, seasonal, residual = decompose_time_series(temp_data['temperature'], period=365)

# Plot components
plot_decomposition(temp_data['temperature'], trend, seasonal, residual)
```

### Temperature Forecasting
```python
from src.analysis.forecasting_models import train_temperature_model, forecast_temperature
import pandas as pd

# Load training data
train_data = pd.read_csv('data/processed/temperature_train.csv', parse_dates=['date'], index_col='date')

# Train model
model = train_temperature_model(train_data, features=['temp', 'humidity', 'pressure'], target='next_day_temp')

# Forecast future temperatures
forecast = forecast_temperature(model, current_data, days=7)
```

### Interactive Climate Map
```python
from src.visualization.climate_maps import create_interactive_map
import pandas as pd

# Load climate data with geographic coordinates
climate_data = pd.read_csv('data/processed/global_climate_data.csv')

# Create interactive map
map_fig = create_interactive_map(climate_data, value_column='temperature_anomaly', 
                               title='Global Temperature Anomalies 2015-2024')
map_fig.show()
```

## Configuration
The project uses a `config.yaml` file for configuration. Here's an example:

```yaml
# Data collection settings
data_collection:
  openweather:
    api_url: "https://api.openweathermap.org/data/3.0/onecall"
    units: "metric"
    variables:
      - "temp"
      - "humidity"
      - "precipitation"
      - "wind_speed"
  nasa_power:
    api_url: "https://power.larc.nasa.gov/api/temporal/daily/point"
    parameters:
      - "T2M"  # Temperature at 2 Meters
      - "PRECTOTCORR"  # Precipitation
      - "WS10M"  # Wind Speed at 10 Meters

# Analysis settings
analysis:
  time_series:
    decomposition_method: "additive"
    trend_detection_threshold: 0.05
  forecasting:
    model_type: "lstm"
    train_test_split: 0.8
    features:
      - "temp"
      - "humidity"
      - "pressure"
      - "month"
      - "day"
  statistical_tests:
    significance_level: 0.05

# Visualization settings
visualization:
  color_palette: "viridis"
  map_projection: "mercator"
  default_height: 600
  default_width: 900

# Cities to analyze
cities:
  - name: "New York"
    country: "US"
    lat: 40.7128
    lon: -74.0060
  - name: "London"
    country: "GB"
    lat: 51.5074
    lon: -0.1278
  - name: "Tokyo"
    country: "JP"
    lat: 35.6762
    lon: 139.6503
  - name: "Sydney"
    country: "AU"
    lat: -33.8688
    lon: 151.2093
  - name: "Rio de Janeiro"
    country: "BR"
    lat: -22.9068
    lon: -43.1729
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- OpenWeather API for providing weather data
- NASA POWER Project for climate data
- Python data science community for the amazing tools and libraries
