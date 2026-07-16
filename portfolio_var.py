"""
Portfolio Value at Risk (VaR) Calculator
=========================================

Computes Historical, Parametric (Variance-Covariance), and Monte Carlo
Value at Risk for a multi-asset portfolio using historical price data
pulled from Yahoo Finance via the `yfinance` package.

Requirements:
    pip install yfinance numpy pandas scipy

Usage:
    python portfolio_var.py
    (edit the CONFIG section below to change tickers/weights/etc.)
"""

import warnings

# Suppress a harmless urllib3/LibreSSL compatibility warning triggered at
# import time on some macOS Python builds (e.g. system Python linked against
# LibreSSL instead of OpenSSL). It does not affect functionality or
# correctness of the results. Must be set BEFORE importing yfinance/urllib3.
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm



# ============================================================
# CONFIG - edit these values for your own portfolio
# ============================================================
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]
WEIGHTS = [0.25, 0.25, 0.25, 0.25]      # must sum to 1.0
PORTFOLIO_VALUE = 1_000_000             # in dollars
CONFIDENCE_LEVEL = 0.95                 # 95% confidence
TIME_HORIZON_DAYS = 1                   # 1-day VaR
HISTORY_PERIOD = "5y"                   # yfinance period string (e.g. "1y","3y","5y","10y")
NUM_SIMULATIONS = 10_000                # Monte Carlo simulation paths
RANDOM_SEED = 42


def validate_inputs(tickers, weights):
    if len(tickers) != len(weights):
        raise ValueError("Number of tickers must match number of weights.")
    if not np.isclose(sum(weights), 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1.0 (got {sum(weights)}).")


def download_prices(tickers, period):
    print(f"Downloading {period} of historical data for: {', '.join(tickers)} ...")
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how="any")
    if data.empty:
        raise RuntimeError("No price data was returned. Check tickers and internet connection.")
    return data


def compute_returns(prices):
    return prices.pct_change().dropna()


def historical_var(portfolio_returns, portfolio_value, confidence_level, horizon_days):
    """
    Historical Simulation VaR: uses the empirical distribution of past
    portfolio returns, no distributional assumption.
    """
    scaled_returns = portfolio_returns * np.sqrt(horizon_days)
    var_pct = -np.percentile(scaled_returns, (1 - confidence_level) * 100)
    var_dollar = var_pct * portfolio_value
    return var_pct, var_dollar


def parametric_var(portfolio_returns, portfolio_value, confidence_level, horizon_days):
    """
    Parametric (Variance-Covariance) VaR: assumes portfolio returns are
    normally distributed. Uses mean and standard deviation of historical
    portfolio returns.
    """
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    z_score = norm.ppf(1 - confidence_level)
    var_pct = -(mu * horizon_days + z_score * sigma * np.sqrt(horizon_days))
    var_dollar = var_pct * portfolio_value
    return var_pct, var_dollar


def monte_carlo_var(returns, weights, portfolio_value, confidence_level,
                     horizon_days, num_simulations, seed):
    """
    Monte Carlo VaR: simulates many possible future portfolio return paths
    using the historical mean/covariance of asset returns (multivariate
    normal assumption), then takes the empirical percentile of simulated
    portfolio outcomes.
    """
    rng = np.random.default_rng(seed)
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    weights = np.array(weights)

    simulated_asset_returns = rng.multivariate_normal(
        mean_returns * horizon_days,
        cov_matrix * horizon_days,
        size=num_simulations,
    )
    simulated_portfolio_returns = simulated_asset_returns @ weights

    var_pct = -np.percentile(simulated_portfolio_returns, (1 - confidence_level) * 100)
    var_dollar = var_pct * portfolio_value
    return var_pct, var_dollar


def main():
    validate_inputs(TICKERS, WEIGHTS)

    prices = download_prices(TICKERS, HISTORY_PERIOD)
    # Ensure column order matches TICKERS/WEIGHTS order
    prices = prices[TICKERS]

    returns = compute_returns(prices)
    weights = np.array(WEIGHTS)
    portfolio_returns = returns @ weights

    print("\n=== Portfolio Summary ===")
    for t, w in zip(TICKERS, WEIGHTS):
        print(f"  {t}: weight = {w:.2%}")
    print(f"  Portfolio value: ${PORTFOLIO_VALUE:,.2f}")
    print(f"  Confidence level: {CONFIDENCE_LEVEL:.0%}")
    print(f"  Time horizon: {TIME_HORIZON_DAYS} day(s)")
    print(f"  History period: {HISTORY_PERIOD}")
    print(f"  Observations used: {len(portfolio_returns)}")

    hist_pct, hist_dollar = historical_var(
        portfolio_returns, PORTFOLIO_VALUE, CONFIDENCE_LEVEL, TIME_HORIZON_DAYS
    )
    param_pct, param_dollar = parametric_var(
        portfolio_returns, PORTFOLIO_VALUE, CONFIDENCE_LEVEL, TIME_HORIZON_DAYS
    )
    mc_pct, mc_dollar = monte_carlo_var(
        returns, weights, PORTFOLIO_VALUE, CONFIDENCE_LEVEL,
        TIME_HORIZON_DAYS, NUM_SIMULATIONS, RANDOM_SEED
    )

    print("\n=== Value at Risk (VaR) Results ===")
    print(f"{'Method':<25}{'VaR %':>12}{'VaR $':>18}")
    print("-" * 55)
    print(f"{'Historical Simulation':<25}{hist_pct:>11.2%}  ${hist_dollar:>15,.2f}")
    print(f"{'Parametric (Var-Covar)':<25}{param_pct:>11.2%}  ${param_dollar:>15,.2f}")
    print(f"{'Monte Carlo':<25}{mc_pct:>11.2%}  ${mc_dollar:>15,.2f}")

    print(f"\nInterpretation: With {CONFIDENCE_LEVEL:.0%} confidence, the portfolio "
          f"is not expected to lose more than the VaR $ amount over the next "
          f"{TIME_HORIZON_DAYS} day(s), under each respective method's assumptions.")


if __name__ == "__main__":
    main()
