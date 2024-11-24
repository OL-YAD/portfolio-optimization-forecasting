# Add the parent directory to the system path 
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

import warnings
warnings.filterwarnings('ignore')

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from scripts.time_series_forecasting import (
    adf_test,
    apply_arima,
    prepare_lstm_data,
    build_lstm_model,
    train_and_predict_lstm,
    calculate_metrics
)

class TestTimeSeriesForecasting(unittest.TestCase):

    @patch('scripts.time_series_forecasting.adfuller')
    def test_adf_test(self, mock_adfuller):
        # Mock the adfuller function
        mock_adfuller.return_value = (-2.5, 0.05, 1, 10, {'1%': -3.43, '5%': -2.86, '10%': -2.57}, 123)

        data = pd.Series(np.random.randn(100))
        ticker = 'AAPL'

        # Capture the print output
        with patch('builtins.print') as mock_print:
            adf_test(data, ticker)

        mock_print.assert_any_call(f"ADF Test for {ticker}:")
        mock_print.assert_any_call("ADF Statistic: -2.5")
        mock_print.assert_any_call("p-value: 0.05")

    @patch('scripts.time_series_forecasting.auto_arima')
    @patch('scripts.time_series_forecasting.ARIMA')
    def test_apply_arima(self, mock_ARIMA, mock_auto_arima):
        # Mock the auto_arima function
        mock_auto_arima.return_value = MagicMock(order=(1, 1, 1))

        # Mock the ARIMA model
        mock_model_arima = mock_ARIMA.return_value
        mock_model_arima.fit.return_value = MagicMock(forecast=MagicMock(return_value=np.random.randn(10)))

        train = pd.Series(np.random.randn(100))
        test = pd.Series(np.random.randn(10))

        arima_forecast, test_data, model_arima_fit = apply_arima(train, test)

        self.assertEqual(arima_forecast.shape, (10,))
        self.assertEqual(test_data.shape, (10,))

    def test_prepare_lstm_data(self):
        train = pd.Series(np.random.randn(100))
        test = pd.Series(np.random.randn(10))

        scaler, X_train, y_train, X_test, test_data = prepare_lstm_data(train, test)

        self.assertIsNotNone(scaler)
        self.assertEqual(X_train.shape, (40, 60))
        self.assertEqual(y_train.shape, (40,))
        self.assertEqual(X_test.shape, (10, 60))
        self.assertEqual(test_data.shape, (10,))

    def test_build_lstm_model(self):
        X_train = np.random.randn(40, 60, 1)
        model_lstm = build_lstm_model(X_train)

        self.assertIsNotNone(model_lstm)
        self.assertEqual(len(model_lstm.layers), 3)

    @patch('scripts.time_series_forecasting.Sequential')
    def test_train_and_predict_lstm(self, mock_Sequential):
        # Mock the Sequential model
        mock_model_lstm = mock_Sequential.return_value
        mock_model_lstm.fit.return_value = None
        mock_model_lstm.predict.return_value = np.random.randn(10, 1)

        X_train = np.random.randn(40, 60, 1)
        y_train = np.random.randn(40)
        X_test = np.random.randn(10, 60, 1)

        lstm_forecast = train_and_predict_lstm(mock_model_lstm, X_train, y_train, X_test)

        self.assertEqual(lstm_forecast.shape, (10, 1))

    def test_calculate_metrics(self):
        actual = np.random.randn(10)
        forecast = np.random.randn(10)

        mae, rmse, mape = calculate_metrics(actual, forecast)

        self.assertIsNotNone(mae)
        self.assertIsNotNone(rmse)
        self.assertIsNotNone(mape)

if __name__ == '__main__':
    unittest.main()
