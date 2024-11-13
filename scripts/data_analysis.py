import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as stats

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing stock data
    """
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df

def clean_data(df, ticker):
    """
    Clean and preprocess financial data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw stock data
    ticker : str
        Stock ticker symbol
        
    Returns:
    --------
    pd.DataFrame
        Cleaned and preprocessed data
    """
    # Create copy to avoid modifying original data
    df = df.copy()
    
    # Check for missing values
    missing_values = df.isnull().sum()
    
    # Forward fill missing values
    df.fillna(method='ffill', inplace=True)
    
    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Calculate rolling metrics
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate volatility (20-day rolling standard deviation)
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    
    return df, missing_values

def plot_price_analysis(df, ticker):
    """
    Plot price analysis charts including price trends and volume.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed stock data
    ticker : str
        Stock ticker symbol
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Price and Moving Averages
    ax1.plot(df.index, df['Close'], label='Close Price', alpha=0.8)
    ax1.plot(df.index, df['MA20'], label='20-day MA', alpha=0.7)
    ax1.plot(df.index, df['MA50'], label='50-day MA', alpha=0.7)
    ax1.plot(df.index, df['MA200'], label='200-day MA', alpha=0.7)
    ax1.set_title(f'{ticker} Price and Moving Averages')
    ax1.legend()
    ax1.grid(True)
    
    # Volume
    ax2.bar(df.index, df['Volume'], alpha=0.7)
    ax2.set_title(f'{ticker} Trading Volume')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_returns_distribution(df, ticker):
    """
    Plot returns distribution analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed stock data
    ticker : str
        Stock ticker symbol
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Daily Returns Distribution
    sns.histplot(df['Daily_Return'].dropna(), kde=True, ax=ax1)
    ax1.set_title(f'{ticker} Daily Returns Distribution')
    
    # Q-Q Plot
    stats.probplot(df['Daily_Return'].dropna(), dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot")
    
    plt.tight_layout()
    plt.show()

def calculate_risk_metrics(df, confidence_levels=[0.95, 0.99], rf_rate=0.02):
    """
    Calculate various risk metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed stock data
    confidence_levels : list
        List of confidence levels for VaR calculation
    rf_rate : float
        Risk-free rate (annual)
        
    Returns:
    --------
    dict
        Dictionary containing risk metrics
    """
    returns = df['Daily_Return'].dropna()
    
    # Calculate VaR for each confidence level
    var_metrics = {
        f'VaR_{int(level*100)}': np.percentile(returns, (1 - level) * 100)
        for level in confidence_levels
    }
    
    # Calculate other risk metrics
    metrics = {
        'Daily_Volatility': returns.std(),
        'Annual_Volatility': returns.std() * np.sqrt(252),
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis(),
        'Sharpe_Ratio': (returns.mean() - rf_rate/252) / returns.std() * np.sqrt(252),
        'Max_Drawdown': (df['Close'] / df['Close'].expanding(min_periods=1).max() - 1).min()
    }
    
    return {**var_metrics, **metrics}

def plot_rolling_volatility(df, ticker, window=20):
    """
    Plot rolling volatility analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed stock data
    ticker : str
        Stock ticker symbol
    window : int
        Rolling window size
    """
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['Volatility'])
    plt.title(f'{ticker} {window}-day Rolling Volatility')
    plt.grid(True)
    plt.show()

def perform_seasonality_analysis(df, ticker, period=252):
    """
    Perform and plot seasonality analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed stock data
    ticker : str
        Stock ticker symbol
    period : int
        Number of periods for seasonal decomposition
        
    Returns:
    --------
    statsmodels.tsa.seasonal.DecomposeResult
        Decomposition results
    """
    decomposition = seasonal_decompose(df['Close'], period=period)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 15))
    
    # Observed
    ax1.plot(decomposition.observed)
    ax1.set_title(f'{ticker} Seasonal Decomposition - Observed')
    ax1.grid(True)
    
    # Trend
    ax2.plot(decomposition.trend)
    ax2.set_title('Trend')
    ax2.grid(True)
    
    # Seasonal
    ax3.plot(decomposition.seasonal)
    ax3.set_title('Seasonal')
    ax3.grid(True)
    
    # Residual
    ax4.plot(decomposition.resid)
    ax4.set_title('Residual')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return decomposition

def analyze_correlations(data_dict):
    """
    Analyze and plot correlations between assets.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing DataFrames for each asset
        
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    # Create DataFrame with closing prices
    close_prices = pd.DataFrame({
        ticker: data[['Close']].values.flatten() 
        for ticker, data in data_dict.items()
    }, index=next(iter(data_dict.values())).index)
    
    # Calculate returns correlation matrix
    correlation_matrix = close_prices.pct_change().corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Asset Correlation Matrix')
    plt.show()
    
    return correlation_matrix

def generate_summary_report(cleaned_data, risk_metrics, correlation_matrix):
    """
    Generate a summary report of the analysis findings.
    """
    summary = []
    
    # Price Analysis Summary
    summary.append("Price Analysis:")
    for ticker in cleaned_data.keys():
        last_price = cleaned_data[ticker]['Close'][-1]
        price_change = (last_price / cleaned_data[ticker]['Close'][0] - 1) * 100
        summary.append(f"- {ticker}: Current Price: ${last_price:.2f}, Total Return: {price_change:.2f}%")
    
    # Risk Analysis Summary
    summary.append("\nRisk Analysis:")
    for ticker, metrics in risk_metrics.items():
        summary.append(f"- {ticker}:")
        summary.append(f"  * Annual Volatility: {metrics['Annual_Volatility']*100:.2f}%")
        summary.append(f"  * Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f}")
        summary.append(f"  * Maximum Drawdown: {metrics['Max_Drawdown']*100:.2f}%")
    
    # Correlation Summary
    summary.append("\nCorrelation Analysis:")
    high_corr_pairs = []
    for i in correlation_matrix.index:
        for j in correlation_matrix.columns:
            if i < j:  # Avoid duplicate pairs
                corr = correlation_matrix.loc[i, j]
                if abs(corr) > 0.5:  # Report significant correlations
                    high_corr_pairs.append(f"- {i}-{j}: {corr:.2f}")
    
    if high_corr_pairs:
        summary.append("Significant correlations:")
        summary.extend(high_corr_pairs)
    
    # Print the summary
    print("\nAnalysis Summary:")
    print("-" * 40)
    print("\n".join(summary))