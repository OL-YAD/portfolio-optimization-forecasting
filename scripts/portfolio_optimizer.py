import os
import logging
import pandas as pd 
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

# Setup logging
log_directory = "../logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_filename = os.path.join(log_directory, f'portfolio_optimizer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename)]
)

class PortfolioOptimizer:
    def __init__(self, forecast_df_tsla, forecast_df_bnd, forecast_df_spy):
        logging.info("Initializing PortfolioOptimizer with provided forecast data.")
        self.forecast_df = self.combine_forecasts(forecast_df_tsla, forecast_df_bnd, forecast_df_spy)
        self.annual_trading_days = 252 
        self.initial_weights = np.array([0.33, 0.33, 0.34])  
        self.risk_free_rate = 0.03  
        logging.info("PortfolioOptimizer initialized.")

    def combine_forecasts(self, tsla_df, bnd_df, spy_df):
        """Combine forecast dataframes into one with columns TSLA, BND, SPY."""
        logging.info("Combining forecast dataframes for TSLA, BND, and SPY.")
        combined_df = pd.DataFrame({
            'Date': tsla_df['Date'],
            'TSLA': tsla_df['Forecast'],
            'BND': bnd_df['Forecast'],
            'SPY': spy_df['Forecast']
        })
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df.set_index('Date', inplace=True)
        logging.info("Forecast data combined successfully.")
        return combined_df
    
    def calculate_daily_returns(self):
        """Calculate daily returns for each asset in the forecast dataframe."""
        logging.info("Calculating daily returns.")
        daily_returns = self.forecast_df.pct_change().dropna()
        logging.info("Daily returns calculated.")
        return daily_returns

    def portfolio_performance(self, weights, daily_returns):
        """Calculate portfolio return and volatility (standard deviation)."""
        logging.info("Calculating portfolio performance.")
        portfolio_return = np.sum(daily_returns.mean() * weights) * self.annual_trading_days
        portfolio_variance = np.dot(weights.T, np.dot(daily_returns.cov() * self.annual_trading_days, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        logging.info(f"Portfolio performance calculated: Return = {portfolio_return:.4f}, Volatility = {portfolio_volatility:.4f}")
        return portfolio_return, portfolio_volatility

    def calculate_sharpe_ratio(self, weights, daily_returns):
        """Objective function to maximize Sharpe Ratio."""
        portfolio_return, portfolio_volatility = self.portfolio_performance(weights, daily_returns)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        logging.info(f"Calculated Sharpe Ratio: {sharpe_ratio:.4f}")
        return -sharpe_ratio  
    
    def optimize_portfolio(self):
        """Optimize portfolio to maximize Sharpe Ratio."""
        logging.info("Starting portfolio optimization to maximize Sharpe Ratio.")
        daily_returns = self.calculate_daily_returns()
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0, 1) for _ in range(len(self.initial_weights)))

        result = minimize(
            self.calculate_sharpe_ratio, 
            self.initial_weights, 
            args=(daily_returns,),
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )

        optimized_weights = result.x
        max_sharpe_ratio = -result.fun
        logging.info(f"Optimization completed. Optimized Weights: {optimized_weights}, Max Sharpe Ratio: {max_sharpe_ratio:.4f}")
        return optimized_weights, max_sharpe_ratio

    def plot_portfolio_performance(self, optimized_weights):
        """Plot daily and cumulative returns of the optimized portfolio vs individual assets."""
        logging.info("Plotting portfolio performance.")
        daily_returns = self.calculate_daily_returns()
        optimized_portfolio_daily_return = (daily_returns * optimized_weights).sum(axis=1)
        cumulative_returns = (1 + daily_returns).cumprod()
        optimized_portfolio_cumulative_return = (1 + optimized_portfolio_daily_return).cumprod()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot daily returns on the first subplot
        ax1.plot(daily_returns.index, optimized_portfolio_daily_return, color='purple', label='Optimized Portfolio (Daily Return)')
        for column in daily_returns.columns:
            ax1.plot(daily_returns.index, daily_returns[column], label=f'{column} (Daily Return)', linestyle='--')
        ax1.set_title('Daily Returns of Optimized Portfolio vs Individual Assets')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Daily Return')
        ax1.legend()
        ax1.grid(True)

        # Plot cumulative returns on the second subplot
        ax2.plot(cumulative_returns.index, optimized_portfolio_cumulative_return, color='purple', label='Optimized Portfolio (Cumulative Return)')
        for column in cumulative_returns.columns:
            ax2.plot(cumulative_returns.index, cumulative_returns[column], label=f'{column} (Cumulative Return)', linestyle='--')
        ax2.set_title('Cumulative Returns of Optimized Portfolio vs Individual Assets')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative Return')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
        logging.info("Portfolio performance plot generated.")