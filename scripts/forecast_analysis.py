import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Trend Analysis
def analyze_trend(forecast):
    start_price = forecast.iloc[0]
    end_price = forecast.iloc[-1]
    trend_direction = "upward" if end_price > start_price else "downward" if end_price < start_price else "stable"
    price_change_percentage = ((end_price - start_price) / start_price) * 100
    return trend_direction, price_change_percentage


# Volatility and Risk Analysis
def analyze_volatility(forecast, conf_int):
    volatility = forecast.std()
    conf_interval_width = (conf_int.iloc[:, 1] - conf_int.iloc[:, 0]).mean()
    
    # Identify high volatility periods
    rolling_volatility = forecast.rolling(window=30).std()
    high_volatility_periods = rolling_volatility[rolling_volatility > rolling_volatility.mean() + rolling_volatility.std()]
    
    return {
        'overall_volatility': volatility,
        'confidence_interval_width': conf_interval_width,
        'high_volatility_periods': high_volatility_periods
    }

# Market Opportunities and Risks
def identify_market_opportunities(forecast, conf_int):
    opportunities = []
    risks = []
    
    # Potential price increases
    for i in range(1, len(forecast)):
        if forecast.iloc[i] > forecast.iloc[i-1] and conf_int.iloc[i, 0] > forecast.iloc[i-1]:
            opportunities.append({
                'start_date': forecast.index[i-1],
                'end_date': forecast.index[i],
                'potential_increase': forecast.iloc[i] - forecast.iloc[i-1]
            })
    
    # Potential price declines or high-risk periods
    for i in range(1, len(forecast)):
        if forecast.iloc[i] < forecast.iloc[i-1] and conf_int.iloc[i, 1] < forecast.iloc[i-1]:
            risks.append({
                'start_date': forecast.index[i-1],
                'end_date': forecast.index[i],
                'potential_decline': forecast.iloc[i-1] - forecast.iloc[i]
            })
    
    return opportunities, risks

# Create sequences for forecasting(LSTM)
def create_forecast_sequences(data, look_back=60):
    X_forecast = []
    for i in range(len(data) - look_back):
        X_forecast.append(data[i:i+look_back])
    return np.array(X_forecast)


