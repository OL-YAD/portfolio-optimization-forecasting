import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# ADF test for stationarity for TSLA
def adf_test(data, ticker):
    print(f"ADF Test for {ticker}:")
    result = adfuller(data)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print("")

# Function to apply ARIMA model
def apply_arima(train, test):
    # Automatically find the optimal ARIMA (p, d, q) parameters
    arima_model = auto_arima(train, seasonal=False, trace=True)
    print(f"Optimal (p, d, q) parameters: {arima_model.order}")

    # Fit the ARIMA model
    model_arima = ARIMA(train, order=arima_model.order)
    model_arima_fit = model_arima.fit()

    # Forecast for the test period
    arima_forecast = model_arima_fit.forecast(steps=len(test))
    
    return arima_forecast, test, model_arima_fit


def prepare_lstm_data(train, test):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(np.array(train).reshape(-1, 1))

    # Prepare data for LSTM training
    X_train, y_train = [], []
    for i in range(60, len(train_scaled)):
        X_train.append(train_scaled[i-60:i])
        y_train.append(train_scaled[i])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Prepare test data inputs for LSTM predictions
    inputs = np.concatenate([train.iloc[-60:].values.reshape(-1, 1), test.values.reshape(-1, 1)], axis=0)
    inputs = scaler.transform(inputs)
    X_test = [inputs[i-60:i] for i in range(60, len(inputs))]
    X_test = np.array(X_test)

    return scaler, X_train, y_train, X_test, test

# LSTM model architecture
def build_lstm_model(X_train):
    model_lstm = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    return model_lstm

# Train the model and make predictions
def train_and_predict_lstm(model_lstm, X_train, y_train, X_test):
    model_lstm.fit(X_train, y_train, epochs=50, batch_size=32)
    lstm_forecast = model_lstm.predict(X_test)
    return lstm_forecast

# Define function to calculate evaluation metrics
def calculate_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = mean_squared_error(actual, forecast, squared=False)
    mape = mean_absolute_percentage_error(actual, forecast)
    return mae, rmse, mape
