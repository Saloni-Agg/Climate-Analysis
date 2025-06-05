"""
NASA POWER API Data Collection Module

This module provides functionality to collect climate data from the NASA POWER API
including temperature, precipitation, solar radiation, and other meteorological parameters.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any
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


class NasaPowerAPI:
    """
    NASA POWER API client for collecting climate data.

    Provides access to the NASA POWER API for collecting various climate parameters
    at different temporal resolutions (daily, monthly, climatology).
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the NASA POWER API client.

        Args:
            api_key: NASA API key. If None, will look for NASA_API_KEY in environment.
                    Note: NASA POWER API can be used without an API key but with rate limits.
        """
        self.api_key = api_key or os.getenv('NASA_API_KEY', '')

        # Base URL for NASA POWER API
        self.base_url = "https://power.larc.nasa.gov/api/temporal"

        # Session for API requests
        self.session = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def get_daily_data(self, lat: float, lon: float, 
                           start_date: str, end_date: str,
                           parameters: List[str]) -> Dict[str, Any]:
        """
        Get daily climate data for a location.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            parameters: List of NASA POWER parameters to retrieve
                        (e.g., T2M, PRECTOTCORR, WS10M)

        Returns:
            Dictionary containing climate data
        """
        url = f"{self.base_url}/daily/point"

        params = {
            'latitude': lat,
            'longitude': lon,
            'start': start_date,
            'end': end_date,
            'community': 'RE',
            'parameters': ','.join(parameters),
            'format': 'JSON'
        }

        if self.api_key:
            params['api_key'] = self.api_key

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_daily_data(data, parameters)
                else:
                    error_msg = await response.text()
                    logger.error(f"NASA POWER API request failed with status {response.status}: {error_msg}")
                    raise Exception(f"NASA POWER API request failed: {response.status}")
        except Exception as e:
            logger.error(f"Error fetching NASA POWER data: {e}")
            raise

    async def get_monthly_data(self, lat: float, lon: float, 
                             start_date: str, end_date: str,
                             parameters: List[str]) -> Dict[str, Any]:
        """
        Get monthly climate data for a location.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date in YYYYMM format
            end_date: End date in YYYYMM format
            parameters: List of NASA POWER parameters to retrieve

        Returns:
            Dictionary containing climate data
        """
        url = f"{self.base_url}/monthly/point"

        params = {
            'latitude': lat,
            'longitude': lon,
            'start': start_date,
            'end': end_date,
            'community': 'RE',
            'parameters': ','.join(parameters),
            'format': 'JSON'
        }

        if self.api_key:
            params['api_key'] = self.api_key

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_monthly_data(data, parameters)
                else:
                    error_msg = await response.text()
                    logger.error(f"NASA POWER API request failed with status {response.status}: {error_msg}")
                    raise Exception(f"NASA POWER API request failed: {response.status}")
        except Exception as e:
            logger.error(f"Error fetching NASA POWER data: {e}")
            raise

    async def get_climatology(self, lat: float, lon: float, 
                            parameters: List[str]) -> Dict[str, Any]:
        """
        Get climatology data for a location.

        Args:
            lat: Latitude
            lon: Longitude
            parameters: List of NASA POWER parameters to retrieve

        Returns:
            Dictionary containing climate data
        """
        url = f"{self.base_url}/climatology/point"

        params = {
            'latitude': lat,
            'longitude': lon,
            'community': 'RE',
            'parameters': ','.join(parameters),
            'format': 'JSON'
        }

        if self.api_key:
            params['api_key'] = self.api_key

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_climatology_data(data, parameters)
                else:
                    error_msg = await response.text()
                    logger.error(f"NASA POWER API request failed with status {response.status}: {error_msg}")
                    raise Exception(f"NASA POWER API request failed: {response.status}")
        except Exception as e:
            logger.error(f"Error fetching NASA POWER data: {e}")
            raise

    async def collect_climate_data(self, locations: List[Dict], 
                                 start_date: str, end_date: str,
                                 parameters: List[str],
                                 temporal: str = 'daily') -> pd.DataFrame:
        """
        Collect climate data for multiple locations and date range.

        Args:
            locations: List of location dictionaries with 'name', 'lat', 'lon'
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            parameters: List of NASA POWER parameters to retrieve
            temporal: Temporal resolution ('daily', 'monthly', 'climatology')

        Returns:
            DataFrame containing climate data
        """
        all_data = []

        for location in locations:
            logger.info(f"Collecting {temporal} NASA POWER data for {location['name']}")

            try:
                if temporal == 'daily':
                    data = await self.get_daily_data(
                        location['lat'], location['lon'], start_date, end_date, parameters
                    )
                elif temporal == 'monthly':
                    # Convert dates to YYYYMM format
                    start_yyyymm = start_date.replace('-', '')[0:6]
                    end_yyyymm = end_date.replace('-', '')[0:6]
                    data = await self.get_monthly_data(
                        location['lat'], location['lon'], start_yyyymm, end_yyyymm, parameters
                    )
                elif temporal == 'climatology':
                    data = await self.get_climatology(
                        location['lat'], location['lon'], parameters
                    )
                else:
                    raise ValueError(f"Unsupported temporal resolution: {temporal}")

                # Add location information
                for date, values in data.items():
                    record = {
                        'location': location['name'],
                        'country': location.get('country', ''),
                        'lat': location['lat'],
                        'lon': location['lon'],
                        'date': date
                    }
                    record.update(values)
                    all_data.append(record)

                # Rate limiting to avoid API throttling
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.warning(f"Failed to get data for {location['name']}: {e}")

        return pd.DataFrame(all_data)

    def _process_daily_data(self, data: Dict, parameters: List[str]) -> Dict[str, Dict[str, float]]:
        """Process daily data API response."""
        try:
            # Extract the data section from the response
            properties = data.get('properties', {})
            parameter_data = properties.get('parameter', {})

            # Organize by date
            result = {}

            for param in parameters:
                if param not in parameter_data:
                    logger.warning(f"Parameter {param} not found in response")
                    continue

                param_values = parameter_data[param]

                for date, value in param_values.items():
                    if date not in result:
                        result[date] = {}

                    result[date][param] = value

            return result

        except Exception as e:
            logger.error(f"Error processing daily data: {e}")
            raise

    def _process_monthly_data(self, data: Dict, parameters: List[str]) -> Dict[str, Dict[str, float]]:
        """Process monthly data API response."""
        try:
            # Extract the data section from the response
            properties = data.get('properties', {})
            parameter_data = properties.get('parameter', {})

            # Organize by date (YYYYMM)
            result = {}

            for param in parameters:
                if param not in parameter_data:
                    logger.warning(f"Parameter {param} not found in response")
                    continue

                param_values = parameter_data[param]

                for month, value in param_values.items():
                    if month not in result:
                        result[month] = {}

                    result[month][param] = value

            return result

        except Exception as e:
            logger.error(f"Error processing monthly data: {e}")
            raise

    def _process_climatology_data(self, data: Dict, parameters: List[str]) -> Dict[str, Dict[str, float]]:
        """Process climatology data API response."""
        try:
            # Extract the data section from the response
            properties = data.get('properties', {})
            parameter_data = properties.get('parameter', {})

            # Climatology data has a different structure - by month numbers
            result = {}

            for param in parameters:
                if param not in parameter_data:
                    logger.warning(f"Parameter {param} not found in response")
                    continue

                param_values = parameter_data[param]

                for month, value in param_values.items():
                    month_key = f"MONTH_{month}"
                    if month_key not in result:
                        result[month_key] = {}

                    result[month_key][param] = value

            return result

        except Exception as e:
            logger.error(f"Error processing climatology data: {e}")
            raise


# Parameter definitions for common climate variables
NASA_POWER_PARAMETERS = {
    # Temperature
    'T2M': 'Temperature at 2 Meters (°C)',
    'T2M_MAX': 'Maximum Temperature at 2 Meters (°C)',
    'T2M_MIN': 'Minimum Temperature at 2 Meters (°C)',
    'T2MDEW': 'Dew/Frost Point at 2 Meters (°C)',
    'TS': 'Earth Skin Temperature (°C)',

    # Precipitation and Humidity
    'PRECTOTCORR': 'Precipitation Corrected (mm/day)',
    'RH2M': 'Relative Humidity at 2 Meters (%)',
    'QV2M': 'Specific Humidity at 2 Meters (g/kg)',

    # Wind
    'WS10M': 'Wind Speed at 10 Meters (m/s)',
    'WD10M': 'Wind Direction at 10 Meters (degrees)',
    'WS50M': 'Wind Speed at 50 Meters (m/s)',

    # Solar Radiation
    'ALLSKY_SFC_SW_DWN': 'All Sky Surface Shortwave Downward Irradiance (W/m^2)',
    'ALLSKY_SFC_LW_DWN': 'All Sky Surface Longwave Downward Irradiance (W/m^2)',

    # Cloud Cover
    'CLOUD_AMT': 'Cloud Amount (%)',

    # Extreme Weather Indicators
    'T2M_RANGE': 'Temperature Range at 2 Meters (°C)',
    'WET_DAYS': 'Wet Days (days)',
    'FROST_DAYS': 'Frost Days (days)'
}


async def collect_nasa_climate_data(config_file: str = "config.yaml") -> pd.DataFrame:
    """
    Main function to collect NASA POWER climate data using configuration file.

    Args:
        config_file: Path to configuration file

    Returns:
        DataFrame containing collected climate data
    """
    import yaml

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    locations = config['cities']
    start_date = config.get('start_date', '2015-01-01')
    end_date = config.get('end_date', '2024-06-01')
    parameters = config.get('nasa_power', {}).get('parameters', [
        'T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'RH2M', 'WS10M'
    ])

    async with NasaPowerAPI() as api:
        data = await api.collect_climate_data(locations, start_date, end_date, parameters)

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

    # Example parameters
    parameters = ['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'RH2M', 'WS10M']

    async def main():
        async with NasaPowerAPI() as api:
            # Get daily data for New York for the last month
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            daily_data = await api.get_daily_data(
                40.7128, -74.0060, 
                start_date.strftime('%Y%m%d'), 
                end_date.strftime('%Y%m%d'),
                parameters
            )
            print(f"Collected daily data for {len(daily_data)} days")

            # Get monthly data for a year
            start_year_month = (end_date - timedelta(days=365)).strftime('%Y%m')
            end_year_month = end_date.strftime('%Y%m')

            monthly_data = await api.get_monthly_data(
                40.7128, -74.0060,
                start_year_month,
                end_year_month,
                parameters
            )
            print(f"Collected monthly data for {len(monthly_data)} months")

            # Get climatology data
            climatology_data = await api.get_climatology(
                40.7128, -74.0060,
                parameters
            )
            print(f"Collected climatology data for {len(climatology_data)} months")

    # Uncomment to run the example
    # asyncio.run(main())
