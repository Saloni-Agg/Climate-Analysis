# Create the main OpenWeather API data collection module
openweather_api_code = '''"""
OpenWeather API Data Collection Module

This module provides functionality to collect weather data from the OpenWeather API
including current weather, historical data, and forecast data for climate analysis.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
import json
import time
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenWeatherAPI:
    """
    OpenWeather API client for collecting climate data.
    
    Supports One Call API 3.0 for current weather, forecasts, and historical data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenWeather API client.
        
        Args:
            api_key: OpenWeather API key. If None, will look for OPENWEATHER_API_KEY in environment.
        """
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenWeather API key is required")
        
        self.base_url = "https://api.openweathermap.org/data/3.0/onecall"
        self.historical_url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_current_weather(self, lat: float, lon: float, units: str = "metric") -> Dict:
        """
        Get current weather data for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            units: Units for temperature (metric, imperial, kelvin)
            
        Returns:
            Dictionary containing current weather data
        """
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': units,
            'exclude': 'minutely,daily,alerts'
        }
        
        async with self.session.get(self.base_url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return self._process_current_weather(data)
            else:
                logger.error(f"API request failed with status {response.status}")
                raise Exception(f"API request failed: {response.status}")
    
    async def get_historical_weather(self, lat: float, lon: float, dt: int, 
                                   units: str = "metric") -> Dict:
        """
        Get historical weather data for a specific date.
        
        Args:
            lat: Latitude
            lon: Longitude
            dt: Unix timestamp for the date
            units: Units for temperature
            
        Returns:
            Dictionary containing historical weather data
        """
        params = {
            'lat': lat,
            'lon': lon,
            'dt': dt,
            'appid': self.api_key,
            'units': units
        }
        
        async with self.session.get(self.historical_url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return self._process_historical_weather(data)
            else:
                logger.error(f"Historical API request failed with status {response.status}")
                raise Exception(f"Historical API request failed: {response.status}")
    
    async def collect_historical_data(self, locations: List[Dict], 
                                    start_date: str, end_date: str,
                                    units: str = "metric") -> pd.DataFrame:
        """
        Collect historical weather data for multiple locations and date range.
        
        Args:
            locations: List of location dictionaries with 'name', 'lat', 'lon'
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            units: Units for temperature
            
        Returns:
            DataFrame containing historical weather data
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_data = []
        
        for location in locations:
            logger.info(f"Collecting data for {location['name']}")
            
            current_date = start_dt
            while current_date <= end_dt:
                timestamp = int(current_date.timestamp())
                
                try:
                    weather_data = await self.get_historical_weather(
                        location['lat'], location['lon'], timestamp, units
                    )
                    
                    weather_data.update({
                        'location': location['name'],
                        'country': location.get('country', ''),
                        'lat': location['lat'],
                        'lon': location['lon'],
                        'date': current_date.strftime('%Y-%m-%d')
                    })
                    
                    all_data.append(weather_data)
                    
                    # Rate limiting - OpenWeather allows 1000 calls/day for free tier
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Failed to get data for {location['name']} on {current_date}: {e}")
                
                current_date += timedelta(days=1)
        
        return pd.DataFrame(all_data)
    
    def _process_current_weather(self, data: Dict) -> Dict:
        """Process current weather API response."""
        current = data.get('current', {})
        
        return {
            'timestamp': current.get('dt'),
            'temperature': current.get('temp'),
            'feels_like': current.get('feels_like'),
            'humidity': current.get('humidity'),
            'pressure': current.get('pressure'),
            'visibility': current.get('visibility'),
            'uv_index': current.get('uvi'),
            'clouds': current.get('clouds'),
            'wind_speed': current.get('wind_speed'),
            'wind_deg': current.get('wind_deg'),
            'weather_main': current.get('weather', [{}])[0].get('main'),
            'weather_description': current.get('weather', [{}])[0].get('description'),
            'precipitation': current.get('rain', {}).get('1h', 0) + current.get('snow', {}).get('1h', 0)
        }
    
    def _process_historical_weather(self, data: Dict) -> Dict:
        """Process historical weather API response."""
        current = data.get('data', [{}])[0] if data.get('data') else {}
        
        return {
            'timestamp': current.get('dt'),
            'temperature': current.get('temp'),
            'feels_like': current.get('feels_like'),
            'humidity': current.get('humidity'),
            'pressure': current.get('pressure'),
            'visibility': current.get('visibility'),
            'uv_index': current.get('uvi'),
            'clouds': current.get('clouds'),
            'wind_speed': current.get('wind_speed'),
            'wind_deg': current.get('wind_deg'),
            'weather_main': current.get('weather', [{}])[0].get('main'),
            'weather_description': current.get('weather', [{}])[0].get('description'),
            'precipitation': current.get('rain', {}).get('1h', 0) + current.get('snow', {}).get('1h', 0)
        }


async def collect_weather_data(config_file: str = "config.yaml") -> pd.DataFrame:
    """
    Main function to collect weather data using configuration file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        DataFrame containing collected weather data
    """
    import yaml
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    locations = config['cities']
    start_date = config.get('start_date', '2015-01-01')
    end_date = config.get('end_date', '2024-06-01')
    
    async with OpenWeatherAPI() as api:
        data = await api.collect_historical_data(locations, start_date, end_date)
    
    return data


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    # Example locations
    locations = [
        {"name": "New York", "country": "US", "lat": 40.7128, "lon": -74.0060},
        {"name": "London", "country": "GB", "lat": 51.5074, "lon": -0.1278},
        {"name": "Tokyo", "country": "JP", "lat": 35.6762, "lon": 139.6503}
    ]
    
    async def main():
        async with OpenWeatherAPI() as api:
            # Get current weather for New York
            current_weather = await api.get_current_weather(40.7128, -74.0060)
            print("Current weather in New York:", current_weather)
            
            # Get historical data for the last week
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # Note: For demonstration - actual historical data collection would be much larger
            historical_data = await api.collect_historical_data(
                locations[:1], 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
            print(f"Collected {len(historical_data)} historical records")
    
    # Uncomment to run the example
    # asyncio.run(main())
'''

# Save the OpenWeather API module
with open('openweather_api.py', 'w') as f:
    f.write(openweather_api_code)

print("Created openweather_api.py")
print("File size:", len(openweather_api_code), "characters")