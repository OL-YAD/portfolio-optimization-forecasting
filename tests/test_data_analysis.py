# Add the parent directory to the system path 
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

import warnings
warnings.filterwarnings('ignore')

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
from scripts.data_analysis import (
    fetch_stock_data,
    clean_data,
    calculate_risk_metrics,
    perform_seasonality_analysis
)

class TestDataAnalysis(unittest.TestCase):

    @patch('scripts.data_analysis.yf.Ticker')
    def test_fetch_stock_data(self, MockTicker):
        # Mock the Ticker object
        mock_ticker = MockTicker.return_value
        mock_ticker.history.return_value = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [99, 100],
            'Close': [102, 103],
            'Volume': [1000, 1500]
        }, index=pd.DatetimeIndex([datetime(2023, 1, 1), datetime(2023, 1, 2)]))

        ticker = 'AAPL'
        start_date = '2023-01-01'
        end_date = '2023-01-02'

        df = fetch_stock_data(ticker, start_date, end_date)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 5))

    def test_clean_data(self):
        data = {
            'Open': [100, 101, np.nan],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [102, 103, 104],
            'Volume': [1000, 1500, 2000]
        }
        df = pd.DataFrame(data, index=pd.DatetimeIndex([datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]))

        cleaned_df, missing_values = clean_data(df, 'AAPL')

        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertIsInstance(missing_values, pd.Series)
        self.assertTrue(cleaned_df.isnull().values.any())  # Expecting some NaN values due to rolling calculations
        self.assertTrue('Daily_Return' in cleaned_df.columns)
        self.assertTrue('MA20' in cleaned_df.columns)
        self.assertTrue('MA50' in cleaned_df.columns)
        self.assertTrue('MA200' in cleaned_df.columns)
        self.assertTrue('Volatility' in cleaned_df.columns)


    def test_calculate_risk_metrics(self):
        data = {
            'Close': [100, 102, 101, 105, 103]
        }
        df = pd.DataFrame(data, index=pd.DatetimeIndex([datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3), datetime(2023, 1, 4), datetime(2023, 1, 5)]))
        df['Daily_Return'] = df['Close'].pct_change()

        risk_metrics = calculate_risk_metrics(df)

        self.assertIsInstance(risk_metrics, dict)
        self.assertTrue('Daily_Volatility' in risk_metrics)
        self.assertTrue('Annual_Volatility' in risk_metrics)
        self.assertTrue('Skewness' in risk_metrics)
        self.assertTrue('Kurtosis' in risk_metrics)
        self.assertTrue('Sharpe_Ratio' in risk_metrics)
        self.assertTrue('Max_Drawdown' in risk_metrics)

    @patch('scripts.data_analysis.seasonal_decompose')
    def test_perform_seasonality_analysis(self, MockDecompose):
        data = {
            'Close': [100, 102, 101, 105, 103]
        }
        df = pd.DataFrame(data, index=pd.DatetimeIndex([datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3), datetime(2023, 1, 4), datetime(2023, 1, 5)]))

        mock_decompose = MockDecompose.return_value
        mock_decompose.observed = df['Close']
        mock_decompose.trend = df['Close']
        mock_decompose.seasonal = df['Close']
        mock_decompose.resid = df['Close']

        decomposition = perform_seasonality_analysis(df, 'AAPL')

        self.assertIsNotNone(decomposition)

if __name__ == '__main__':
    unittest.main()
