"""
Time Series Analysis Module

This module provides comprehensive time series analysis capabilities for climate data,
including decomposition, trend analysis, seasonal pattern identification, and
statistical testing for climate change detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Statistical and time series libraries
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import pingouin as pg
from arch.unitroot import PhillipsPerron

# Machine Learning libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Plotting
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClimateTimeSeriesAnalyzer:
    """
    Comprehensive time series analysis for climate data.

    Provides methods for decomposition, trend analysis, seasonal analysis,
    and statistical testing for climate change detection.
    """

    def __init__(self, data: pd.DataFrame, date_column: str = 'date',
                 value_column: str = 'temperature'):
        """
        Initialize the climate time series analyzer.

        Args:
            data: DataFrame containing time series data
            date_column: Name of the date column
            value_column: Name of the value column to analyze
        """
        self.data = data.copy()
        self.date_column = date_column
        self.value_column = value_column

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_column]):
            self.data[date_column] = pd.to_datetime(self.data[date_column])

        # Set date as index
        self.data.set_index(date_column, inplace=True)
        self.data.sort_index(inplace=True)

        # Remove any missing values
        self.data = self.data.dropna(subset=[value_column])

        # Store original series
        self.ts = self.data[value_column]

        # Initialize results storage
        self.decomposition_results = {}
        self.trend_results = {}
        self.stationarity_results = {}

    def perform_decomposition(self, model: str = 'additive', period: Optional[int] = None,
                            extrapolate_trend: str = 'freq') -> Dict:
        """
        Perform time series decomposition.

        Args:
            model: Type of decomposition ('additive' or 'multiplicative')
            period: Period for seasonal decomposition. If None, auto-detected.
            extrapolate_trend: How to extrapolate trend component

        Returns:
            Dictionary containing decomposition results
        """
        logger.info(f"Performing {model} decomposition")

        # Auto-detect period if not provided
        if period is None:
            # For daily data, assume yearly seasonality
            if len(self.ts) > 730:  # At least 2 years of data
                period = 365
            # For monthly data, assume yearly seasonality
            elif len(self.ts) > 24:  # At least 2 years of monthly data
                period = 12
            else:
                period = min(len(self.ts) // 2, 12)

        # Perform decomposition
        decomposition = seasonal_decompose(
            self.ts, 
            model=model, 
            period=period,
            extrapolate_trend=extrapolate_trend
        )

        # Store results
        self.decomposition_results = {
            'original': self.ts,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'model': model,
            'period': period
        }

        # Calculate additional statistics
        self.decomposition_results.update(self._calculate_decomposition_stats())

        return self.decomposition_results

    def analyze_trend(self, method: str = 'mann_kendall') -> Dict:
        """
        Analyze trend in the time series.

        Args:
            method: Method for trend analysis ('mann_kendall', 'linear_regression', 'theil_sen')

        Returns:
            Dictionary containing trend analysis results
        """
        logger.info(f"Performing trend analysis using {method}")

        results = {}

        if method == 'mann_kendall':
            results.update(self._mann_kendall_test())
        elif method == 'linear_regression':
            results.update(self._linear_trend_analysis())
        elif method == 'theil_sen':
            results.update(self._theil_sen_trend())
        else:
            raise ValueError(f"Unsupported trend analysis method: {method}")

        self.trend_results[method] = results
        return results

    def test_stationarity(self) -> Dict:
        """
        Test for stationarity using multiple tests.

        Returns:
            Dictionary containing stationarity test results
        """
        logger.info("Testing for stationarity")

        results = {}

        # Augmented Dickey-Fuller test
        adf_result = adfuller(self.ts.dropna())
        results['adf'] = {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        }

        # KPSS test
        kpss_result = kpss(self.ts.dropna())
        results['kpss'] = {
            'statistic': kpss_result[0],
            'p_value': kpss_result[1],
            'critical_values': kpss_result[3],
            'is_stationary': kpss_result[1] > 0.05
        }

        # Phillips-Perron test
        pp_result = PhillipsPerron(self.ts.dropna())
        results['phillips_perron'] = {
            'statistic': pp_result.stat,
            'p_value': pp_result.pvalue,
            'is_stationary': pp_result.pvalue < 0.05
        }

        # Overall assessment
        stationary_tests = [
            results['adf']['is_stationary'],
            results['kpss']['is_stationary'],
            results['phillips_perron']['is_stationary']
        ]
        results['overall_stationary'] = sum(stationary_tests) >= 2

        self.stationarity_results = results
        return results

    def detect_changepoints(self, method: str = 'pelt', min_size: int = 30) -> List[datetime]:
        """
        Detect changepoints in the time series.

        Args:
            method: Method for changepoint detection
            min_size: Minimum segment size

        Returns:
            List of detected changepoint dates
        """
        try:
            import ruptures as rpt
        except ImportError:
            logger.warning("ruptures library not available. Install with: pip install ruptures")
            return []

        logger.info(f"Detecting changepoints using {method}")

        # Prepare data
        signal = self.ts.values

        # Detect changepoints
        if method == 'pelt':
            algo = rpt.Pelt(model="rbf").fit(signal)
            result = algo.predict(pen=10)
        elif method == 'binseg':
            algo = rpt.Binseg(model="l2").fit(signal)
            result = algo.predict(n_bkps=5)
        else:
            raise ValueError(f"Unsupported changepoint detection method: {method}")

        # Convert indices to dates
        changepoints = []
        for idx in result[:-1]:  # Last point is always the end of series
            if idx < len(self.ts):
                changepoints.append(self.ts.index[idx])

        return changepoints

    def analyze_extremes(self, threshold_percentile: float = 95) -> Dict:
        """
        Analyze extreme values in the time series.

        Args:
            threshold_percentile: Percentile threshold for extreme values

        Returns:
            Dictionary containing extreme value analysis results
        """
        logger.info(f"Analyzing extreme values (threshold: {threshold_percentile}th percentile)")

        # Calculate thresholds
        upper_threshold = np.percentile(self.ts, threshold_percentile)
        lower_threshold = np.percentile(self.ts, 100 - threshold_percentile)

        # Identify extreme values
        extreme_high = self.ts[self.ts > upper_threshold]
        extreme_low = self.ts[self.ts < lower_threshold]

        # Calculate statistics
        results = {
            'upper_threshold': upper_threshold,
            'lower_threshold': lower_threshold,
            'n_extreme_high': len(extreme_high),
            'n_extreme_low': len(extreme_low),
            'extreme_high_dates': extreme_high.index.tolist(),
            'extreme_low_dates': extreme_low.index.tolist(),
            'extreme_high_values': extreme_high.tolist(),
            'extreme_low_values': extreme_low.tolist(),
            'frequency_high': len(extreme_high) / len(self.ts) * 100,
            'frequency_low': len(extreme_low) / len(self.ts) * 100
        }

        # Analyze temporal patterns in extremes
        if len(extreme_high) > 0:
            extreme_high_monthly = extreme_high.groupby(extreme_high.index.month).count()
            results['extreme_high_monthly_pattern'] = extreme_high_monthly.to_dict()

        if len(extreme_low) > 0:
            extreme_low_monthly = extreme_low.groupby(extreme_low.index.month).count()
            results['extreme_low_monthly_pattern'] = extreme_low_monthly.to_dict()

        return results

    def _calculate_decomposition_stats(self) -> Dict:
        """Calculate additional statistics from decomposition."""
        trend = self.decomposition_results['trend'].dropna()
        seasonal = self.decomposition_results['seasonal'].dropna()
        residual = self.decomposition_results['residual'].dropna()

        return {
            'trend_strength': 1 - np.var(residual) / np.var(trend + residual),
            'seasonal_strength': 1 - np.var(residual) / np.var(seasonal + residual),
            'residual_variance': np.var(residual),
            'signal_to_noise_ratio': np.var(trend + seasonal) / np.var(residual)
        }

    def _mann_kendall_test(self) -> Dict:
        """Perform Mann-Kendall test for trend."""
        try:
            result = pg.mann_kendall(self.ts.values)
            return {
                'tau': result['tau'].iloc[0],
                'p_value': result['p'].iloc[0],
                'trend': result['trend'].iloc[0],
                'significance': result['p'].iloc[0] < 0.05
            }
        except Exception as e:
            logger.warning(f"Mann-Kendall test failed: {e}")
            return {'error': str(e)}

    def _linear_trend_analysis(self) -> Dict:
        """Perform linear trend analysis."""
        # Create time index for regression
        time_index = np.arange(len(self.ts))

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, self.ts.values)

        # Calculate trend per year (assuming daily data)
        trend_per_year = slope * 365.25

        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_error': std_err,
            'trend_per_year': trend_per_year,
            'significance': p_value < 0.05,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'no trend'
        }

    def _theil_sen_trend(self) -> Dict:
        """Perform Theil-Sen trend analysis."""
        from scipy.stats import theilslopes

        time_index = np.arange(len(self.ts))

        # Theil-Sen estimator
        slope, intercept, low_slope, high_slope = theilslopes(self.ts.values, time_index)

        # Calculate trend per year
        trend_per_year = slope * 365.25

        return {
            'slope': slope,
            'intercept': intercept,
            'low_slope': low_slope,
            'high_slope': high_slope,
            'trend_per_year': trend_per_year,
            'confidence_interval': (low_slope * 365.25, high_slope * 365.25)
        }

    def plot_decomposition(self, figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        """
        Plot time series decomposition results.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if not self.decomposition_results:
            raise ValueError("Must perform decomposition first")

        fig, axes = plt.subplots(4, 1, figsize=figsize)

        # Original series
        axes[0].plot(self.decomposition_results['original'].index,
                    self.decomposition_results['original'].values)
        axes[0].set_title('Original Time Series')
        axes[0].set_ylabel(self.value_column.title())

        # Trend
        axes[1].plot(self.decomposition_results['trend'].index,
                    self.decomposition_results['trend'].values, color='red')
        axes[1].set_title('Trend Component')
        axes[1].set_ylabel('Trend')

        # Seasonal
        axes[2].plot(self.decomposition_results['seasonal'].index,
                    self.decomposition_results['seasonal'].values, color='green')
        axes[2].set_title('Seasonal Component')
        axes[2].set_ylabel('Seasonal')

        # Residual
        axes[3].plot(self.decomposition_results['residual'].index,
                    self.decomposition_results['residual'].values, color='purple')
        axes[3].set_title('Residual Component')
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')

        plt.tight_layout()
        return fig

    def plot_interactive_decomposition(self) -> go.Figure:
        """
        Create interactive plot of decomposition results using Plotly.

        Returns:
            Plotly figure
        """
        if not self.decomposition_results:
            raise ValueError("Must perform decomposition first")

        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
            vertical_spacing=0.08
        )

        # Original series
        fig.add_trace(
            go.Scatter(
                x=self.decomposition_results['original'].index,
                y=self.decomposition_results['original'].values,
                name='Original',
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        # Trend
        fig.add_trace(
            go.Scatter(
                x=self.decomposition_results['trend'].index,
                y=self.decomposition_results['trend'].values,
                name='Trend',
                line=dict(color='red')
            ),
            row=2, col=1
        )

        # Seasonal
        fig.add_trace(
            go.Scatter(
                x=self.decomposition_results['seasonal'].index,
                y=self.decomposition_results['seasonal'].values,
                name='Seasonal',
                line=dict(color='green')
            ),
            row=3, col=1
        )

        # Residual
        fig.add_trace(
            go.Scatter(
                x=self.decomposition_results['residual'].index,
                y=self.decomposition_results['residual'].values,
                name='Residual',
                line=dict(color='purple')
            ),
            row=4, col=1
        )

        fig.update_layout(
            height=800,
            title_text="Time Series Decomposition",
            showlegend=False
        )

        return fig


def analyze_climate_time_series(data: pd.DataFrame, location: str = None,
                              variable: str = 'temperature') -> Dict:
    """
    Comprehensive time series analysis for climate data.

    Args:
        data: DataFrame with climate data
        location: Location name for analysis
        variable: Climate variable to analyze

    Returns:
        Dictionary containing all analysis results
    """
    logger.info(f"Starting comprehensive time series analysis for {variable}")

    # Initialize analyzer
    analyzer = ClimateTimeSeriesAnalyzer(data, value_column=variable)

    # Perform all analyses
    results = {
        'location': location,
        'variable': variable,
        'data_summary': {
            'start_date': analyzer.ts.index.min(),
            'end_date': analyzer.ts.index.max(),
            'n_observations': len(analyzer.ts),
            'mean': analyzer.ts.mean(),
            'std': analyzer.ts.std(),
            'min': analyzer.ts.min(),
            'max': analyzer.ts.max()
        }
    }

    try:
        # Time series decomposition
        decomposition = analyzer.perform_decomposition()
        results['decomposition'] = decomposition

        # Trend analysis
        trend_mk = analyzer.analyze_trend('mann_kendall')
        trend_lr = analyzer.analyze_trend('linear_regression')
        results['trend_analysis'] = {
            'mann_kendall': trend_mk,
            'linear_regression': trend_lr
        }

        # Stationarity tests
        stationarity = analyzer.test_stationarity()
        results['stationarity'] = stationarity

        # Extreme value analysis
        extremes = analyzer.analyze_extremes()
        results['extremes'] = extremes

        # Changepoint detection
        changepoints = analyzer.detect_changepoints()
        results['changepoints'] = changepoints

    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        results['error'] = str(e)

    return results


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Generate example climate data
    dates = pd.date_range('2015-01-01', '2024-06-01', freq='D')
    np.random.seed(42)

    # Simulate temperature data with trend and seasonality
    trend = np.linspace(15, 17, len(dates))  # Warming trend
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)  # Annual cycle
    noise = np.random.normal(0, 2, len(dates))
    temperature = trend + seasonal + noise

    # Create DataFrame
    climate_data = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'location': 'Example City'
    })

    # Perform analysis
    results = analyze_climate_time_series(climate_data, location='Example City')

    print("Climate Time Series Analysis Results:")
    print(f"Location: {results['location']}")
    print(f"Data range: {results['data_summary']['start_date']} to {results['data_summary']['end_date']}")
    print(f"Number of observations: {results['data_summary']['n_observations']}")
    print(f"Mean temperature: {results['data_summary']['mean']:.2f}°C")
    print(f"Temperature trend (linear): {results['trend_analysis']['linear_regression']['trend_per_year']:.3f}°C/year")
    print(f"Trend significance: {results['trend_analysis']['linear_regression']['significance']}")
    print(f"Series is stationary: {results['stationarity']['overall_stationary']}")
