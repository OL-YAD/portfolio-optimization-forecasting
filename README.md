# Portfolio Management Optimization and Time Series Forecasting

##  Project Overview

This project implements a comprehensive portfolio optimization system using advanced time series forecasting techniques at Guide Me in Finance (GMF) Investments. It analyzes historical data for Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY) to predict market trends and optimize investment portfolios.

## Business Context
GMF Investments is a financial advisory firm specializing in personalized portfolio management. This project aims to enhance their portfolio management strategies by:
- Leveraging real-time financial data from YFinance
- Implementing predictive models for market trend analysis
- Optimizing asset allocation based on forecasted trends
- Providing data-driven investment recommendations

### Key Features
- Historical financial data analysis using YFinance
- Advanced time series forecasting models (ARIMA/SARIMA/LSTM)
- Portfolio optimization using Modern Portfolio Theory
- Risk analysis with Value at Risk (VaR) calculations
- Interactive visualizations for market trend analysis


## Dataset
The project utilizes historical financial data from three key assets:
- Tesla (TSLA): High-growth, high-risk stock in the automobile manufacturing sector
- Vanguard Total Bond Market ETF (BND): Stable bond ETF for risk management
- S&P 500 ETF (SPY): Broad market exposure through index tracking

Data period: January 1, 2015 - December 31, 2024
Source: YFinance API

## Project Structure
```
├── .github/
│   ├── workflows/            
|      ├──unittests.yml  # github actions
|         
├── notebooks/
│   ├── BND_market_forecast_analysis.ipynb
│   ├── BND_time_series_forecasting.ipynb
│   ├── data_analysis.ipynb
│   ├── market_forecast_analysis.ipynb
│   ├── portfolio_optimizer.ipynb
│   ├── SPY_time_series_forecasting.ipynb
│   ├── TSLA_time_series_forecasting.ipynb
|
|   
├── scripts/
│   ├── data_analysis.py            # Data processing script
│   ├── forecast_analysis.py        # Time series models script
│   └── portfolio_optimizer.py      # portifolio optimization script
├── tests/                # Test files
└── requirements.txt        # Project dependencies
```

## Features

### 1. Data Preprocessing
- Comprehensive data cleaning and validation
- Feature engineering for time series analysis
- Exploratory Data Analysis (EDA)
- Seasonality and trend decomposition
- Volatility analysis and risk metrics calculation

### 2. Time Series Forecasting
- Implementation of multiple forecasting models:
  - ARIMA (AutoRegressive Integrated Moving Average)
  - SARIMA (Seasonal ARIMA)
  - LSTM (Long Short-Term Memory networks)
- Model evaluation and parameter optimization
- Performance metrics tracking (MAE, RMSE, MAPE)

### 3. Portfolio Optimization
- Risk-return analysis
- Portfolio weight optimization
- Sharpe Ratio maximization
- Dynamic asset allocation strategies
- Risk management through diversification


## Installation

```bash
# Clone the repository
git clone https://github.com/OL-YAD/portfolio-optimization-forecasting.git

# Navigate to project directory
cd portfolio-optimization-forecasting

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Results

The project generates:
- Forecasted price trends for each asset
- Optimized portfolio weights
- Risk metrics and performance indicators
- Visualization of predicted market movements
- Portfolio rebalancing recommendations

## Contact

For questions and feedback:
- Email: olyadtemesgen@gmail.com
- LinkedIn: [https://www.linkedin.com/in/olyad-temesgen/]



## Acknowledgments
- 10 Academy for project guidance

