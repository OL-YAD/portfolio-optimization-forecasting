import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

import warnings
warnings.filterwarnings('ignore')


import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from scripts.portfolio_optimizer import PortfolioOptimizer

class TestPortfolioOptimizer(unittest.TestCase):

    def setUp(self):
        # Create mock forecast dataframes
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        self.forecast_df_tsla = pd.DataFrame({'Date': dates, 'Forecast': np.random.randn(10)})
        self.forecast_df_bnd = pd.DataFrame({'Date': dates, 'Forecast': np.random.randn(10)})
        self.forecast_df_spy = pd.DataFrame({'Date': dates, 'Forecast': np.random.randn(10)})
        self.optimizer = PortfolioOptimizer(self.forecast_df_tsla, self.forecast_df_bnd, self.forecast_df_spy)

    def test_combine_forecasts(self):
        combined_df = self.optimizer.combine_forecasts(self.forecast_df_tsla, self.forecast_df_bnd, self.forecast_df_spy)
        self.assertIsInstance(combined_df, pd.DataFrame)
        self.assertListEqual(list(combined_df.columns), ['TSLA', 'BND', 'SPY'])
        self.assertEqual(combined_df.shape, (10, 3))

    def test_calculate_daily_returns(self):
        daily_returns = self.optimizer.calculate_daily_returns()
        self.assertIsInstance(daily_returns, pd.DataFrame)
        self.assertListEqual(list(daily_returns.columns), ['TSLA', 'BND', 'SPY'])
        self.assertEqual(daily_returns.shape, (9, 3))  # One less row due to pct_change().dropna()

    def test_portfolio_performance(self):
        daily_returns = self.optimizer.calculate_daily_returns()
        weights = np.array([0.33, 0.33, 0.34])
        portfolio_return, portfolio_volatility = self.optimizer.portfolio_performance(weights, daily_returns)
        self.assertIsInstance(portfolio_return, np.float64)
        self.assertIsInstance(portfolio_volatility, np.float64)

    @patch('scripts.portfolio_optimizer.minimize')
    def test_optimize_portfolio(self, mock_minimize):
        # Mock the minimize function
        mock_result = MagicMock()
        mock_result.x = np.array([0.4, 0.3, 0.3])
        mock_result.fun = -2.0
        mock_minimize.return_value = mock_result

        optimized_weights, max_sharpe_ratio = self.optimizer.optimize_portfolio()
        self.assertIsInstance(optimized_weights, np.ndarray)
        self.assertIsInstance(max_sharpe_ratio, float)
        self.assertEqual(max_sharpe_ratio, 2.0)

    @patch('scripts.portfolio_optimizer.PortfolioOptimizer.calculate_daily_returns')
    @patch('matplotlib.pyplot.show')
    def test_plot_portfolio_performance(self, mock_show, mock_calculate_daily_returns):
        # Mock the calculate_daily_returns method
        mock_calculate_daily_returns.return_value = pd.DataFrame(np.random.randn(9, 3), columns=['TSLA', 'BND', 'SPY'])

        optimized_weights = np.array([0.4, 0.3, 0.3])
        self.optimizer.plot_portfolio_performance(optimized_weights)
        mock_show.assert_called_once()

if __name__ == '__main__':
    unittest.main()
