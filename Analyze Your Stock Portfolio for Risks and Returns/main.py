'''
Analyse Stock Portfiolio for Risks and Returns
@Author: Henry Ha
'''
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

# Loading data
stock_prices_df = pd.read_csv("faang_stocks.csv", index_col="Date")

# Changing the index to a datetime type allows for easier filtering and plotting.
stock_prices_df.index = pd.to_datetime(stock_prices_df.index)

# Calculate daily returns
returns_df = stock_prices_df.pct_change().dropna()

# Plotting the stock prices
stock_prices_df.plot(title="FAANG Stock Prices from Years 2020–2023", figsize=(10, 6), legend=False)
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.show()

# TODO: 1st portfolio, Equally weighted portfolio
portfolio_weights = 5 * [0.2]  # Equal weights for 5 stocks
portfolio_returns = returns_df.dot(portfolio_weights)

# Calculate expected return and Sharpe ratio
benchmark_exp_return = portfolio_returns.mean() * 252  # Annualized return
benchmark_sharpe_ratio = (
    portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
)

print(f"Expected Return: {benchmark_exp_return:.4f}")
print(f"Sharpe Ratio: {benchmark_sharpe_ratio:.4f}")

# Calculate expected returns and the covariance matrix
avg_returns = returns_df.mean() * 252
cov_mat = returns_df.cov() * 252

# TODO: 2nd portfolio, Optimize for minimum volatility
ef = EfficientFrontier(avg_returns, cov_mat)
weights = ef.min_volatility()
mv_portfolio = pd.Series(weights, index=stock_prices_df.columns)
mv_portfolio_vol = ef.portfolio_performance(risk_free_rate=0)[1]

print("Minimum Volatility Portfolio Weights:")
print(mv_portfolio)
print(f"Portfolio Volatility: {mv_portfolio_vol:.4f}")

# Calculate portfolio performance (returns, volatility, Sharpe ratio)
mv_return, mv_volatility, mv_sharpe = ef.portfolio_performance(risk_free_rate=0)

print(f"Expected Return (Min Volatility): {mv_return:.4f}")
print(f"Portfolio Volatility: {mv_volatility:.4f}")
print(f"Sharpe Ratio (Min Volatility): {mv_sharpe:.4f}")


# TODO: 3rd portfolio, Optimize for maximum Sharpe ratio
ef = EfficientFrontier(avg_returns, cov_mat)
ms_weights = ef.max_sharpe(risk_free_rate=0)
ms_portfolio = pd.Series(ms_weights, index=stock_prices_df.columns)
ms_portfolio_sharpe = ef.portfolio_performance(risk_free_rate=0)[2]

print("Maximum Sharpe Ratio Portfolio Weights:")
print(ms_portfolio)
print(f"Sharpe Ratio: {ms_portfolio_sharpe:.4f}")

# Plotting the historical stock prices
# Data for the bar chart
metrics = pd.DataFrame({
    "Portfolio": ["Equally Weighted", "Min Volatility", "Max Sharpe"],
    "Expected Return": [0.2360, 0.2501, 0.2501],
    "Volatility": [0.35, 0.3031, 0.32],
    "Sharpe Ratio": [0.7222, 0.8253, 0.8822]
})

metrics.set_index("Portfolio", inplace=True)

# Plot the bar chart
ax = metrics.plot(kind="bar", figsize=(10, 6), title="Portfolio Performance Comparison", legend=True)
plt.ylabel("Value")

# Add values on top of each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', label_type='edge', fontsize=10, padding=3)

plt.tight_layout()
plt.show()


