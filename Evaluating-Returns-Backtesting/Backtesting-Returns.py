# Load necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import empyrical as ep

futures_tickers = [
    "ES=F",   # E-mini S&P 500 Futures
    "NQ=F",   # E-mini Nasdaq-100 Futures
    "YM=F",   # E-mini Dow Jones Futures
    "RTY=F",  # E-mini Russell 2000 Futures
    "MES=F"   # Micro E-mini S&P 500 Futures
]




data = yf.download(futures_tickers)
# print(data.head())


"""
3. Resample Data
In this section, we aim to reduce the noise in the daily financial data by
 resampling it to a monthly frequency. Resampling is a common technique in 
 time series analysis, allowing us to aggregate data points over a specified 
 time period. This helps in smoothing out short-term fluctuations and making 
 the data more manageable for analysis.

"""
# 3 Resample data to monthly frequency
monthly_data = data.resample("M").last()
print(monthly_data.head())
monthly_data.index = monthly_data.index.to_period('M')
#print(monthly_data.index)

"""
4. Clean and Prepare Data
In this step, we will focus on extracting the relevant data, handling missing values, and ensuring the data is ready for analysis. Specifically, we'll subset the adjusted close prices from our dataset, fill any missing values, and drop rows with unknown prices.

Steps to Clean and Prepare Data
Subset Adjusted Close Prices
Fill Missing Values (NaNs)
Drop Rows with Remaining NaNs
Ensure Correct Date Formatting
"""

# 4 Clean and prepare data
# 1 Subset adjusted close prices
adj_close = monthly_data["Close"]
# # 2 Fill missing values
adj_close.fillna(method="ffill", inplace=True)
# 3 Drop rows with remaining NaNs
adj_close.dropna(inplace=True)
# Ensure correct date formatting
adj_close = adj_close.sort_index()



# # Subset adjusted close prices and fill NaNs
prices = monthly_data["Close"].ffill()
print(prices.head())
# # Convert index to datetime
# prices.index = pd.to_datetime(monthly_data.index)
# prices = pd.to_datetime(monthly_data.index)


# prices.plot()
# plt.show()

# prices.plot()
# plt.yscale("log")
# plt.show()

 

  
# calculate arithmetic returns
returns = prices.pct_change()
monthly_returns = monthly_data["Close"]["ES=F"].pct_change()
print("Monthly Returns", monthly_returns)   
# calculate cumulative returns
cumulative_returns = (1 + monthly_returns).cumprod() - 1
# plot returns
returns.plot()
# plt.title("Monthly Returns")
# plt.show()


# 6. Compute Risk-Parity weights 
"""
Risk-parity is an investment strategy that seeks to allocate 
portfolio weights in a way that each asset contributes equally 
to the overall portfolio risk. This method is especially useful 
in diversifying risk across different assets with varying levels of volatility.
"""

def compute_risk_parity_weights(returns, window_size=36):
    # Compute volatility known at time t
    rolling_vol  = returns.rolling(window=window_size).std()
    rolling_inverse_vol = 1 / rolling_vol
    # Divide inverse volatility by the sum of inverse volatilities
    risk_parity_weights = rolling_inverse_vol.div(rolling_inverse_vol.sum(axis=1), axis=0)
    # Shift weights by one period to use only information available at time t
    risk_parity_weights =  risk_parity_weights
    return risk_parity_weights

risk_parity_weights = compute_risk_parity_weights(returns, 36)
# risk_parity_weights.plot()
# plt.show()

# 7 Calculate weighted returns
"""
In this section, we will calculate the weighted returns for each asset in our dataset based on the risk-parity weights. By multiplying the returns by the respective weights, we can determine the contribution of each asset to the overall portfolio return.  """

# Calculate weighted returns
weighted_returns = monthly_returns * risk_parity_weights
risk_parity_portfolio_returns = weighted_returns.sum(axis=1)
print("Risk_Parity Portfolio Returns", risk_parity_portfolio_returns)


# Evalute Portfolio Performance
trading_days = 252
risk_free_rate = 0.0
# Evaluate portfolio performance
annual_mean_return = np.mean(monthly_returns) * trading_days
annual_volatility = np.std(monthly_returns) * np.sqrt(trading_days)
skewness = monthly_returns.skew()
kurtosis = monthly_returns.kurtosis()

# # Compute drawdown
log_returns = np.log(prices).diff()
cumulative_returns = (1 + monthly_returns).cumprod()
running_max = cumulative_returns.cummax()
drawdown = (cumulative_returns - running_max) / running_max
max_drawdown =  drawdown.min()

# Compute Sharpe ratio
sharpe_ratio = (annual_mean_return - risk_free_rate) / annual_volatility

# Compute Sortino ratio
downside_std = monthly_returns[monthly_returns < 0]
downside_vol = (log_returns[log_returns<0].std() * np.sqrt(252))
print("down ", downside_vol)
# sortino_ratio = (annual_mean_return - risk_free_rate)  / downside_vol 
sortino_ratio = ep.sortino_ratio(monthly_returns, required_return=0)


# Compute Calmar ratio
calmar_ratio = abs(annual_mean_return / max_drawdown) 
# Display results
print(f"Mean Annual Return: {annual_mean_return:.4f}")
print(f"Annual Volatility: {annual_volatility:.4f}")
print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis: {kurtosis:.4f}")
print(f"Maximum Drawdown: {max_drawdown:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Sortino Ratio: {sortino_ratio:.4f}")
print(f"Calmar Ratio: {calmar_ratio:.4f}")


# Plot results 
fig, ax = plt.subplots(figsize=(14, 7))
(cumulative_returns - 1).plot(ax=ax, label='Cumulative Returns', color='blue')
drawdown.plot(ax=ax, label='Drawdown', color='red')
ax.set_title('Cumulative Returns and Drawdown')
ax.set_ylabel('Cumulative Returns')
ax.set_xlabel('Date')
ax.legend()
plt.show()