"""
var_engine.py
=============

Backend service module for the Portfolio Value at Risk (VaR) Streamlit
dashboard. Encapsulates all data fetching, validation, and statistical
computation logic so the frontend (`app.py`) stays purely presentational.

Public API:
    run_var_analysis(...) -> VarResult (a TypedDict-like dict)

Design notes:
    - All functions are pure and independently testable.
    - Every function that can fail raises a specific, catchable exception
      with a human-readable message; `run_var_analysis` converts these into
      a structured `success/error` result so Streamlit can render a clean
      error banner instead of crashing.
    - Handles the single-asset edge case (no covariance matrix needed).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# Suppress a harmless urllib3/LibreSSL compatibility warning triggered at
# import time on some macOS Python builds. Does not affect correctness.
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")


# ============================================================
# Custom Exceptions
# ============================================================
class ValidationError(Exception):
    """Raised when user-supplied inputs (tickers/weights/etc.) are invalid."""


class DataFetchError(Exception):
    """Raised when live market data cannot be retrieved or is unusable."""


# ============================================================
# Input Validation
# ============================================================
def validate_inputs(tickers: List[str], weights: List[float]) -> None:
    """
    Validate that tickers and weights are well-formed and compatible.

    Raises:
        ValidationError: on any mismatch or invalid values.
    """
    if not tickers:
        raise ValidationError("Please provide at least one stock ticker.")

    if len(tickers) != len(weights):
        raise ValidationError(
            f"Number of tickers ({len(tickers)}) must match number of "
            f"weights ({len(weights)})."
        )

    if any(w < 0 for w in weights):
        raise ValidationError("Weights cannot be negative.")

    weight_sum = sum(weights)
    if not np.isclose(weight_sum, 1.0, atol=1e-3):
        raise ValidationError(
            f"Weights must sum to 1.0 (100%). Current sum = {weight_sum:.4f}."
        )

    if len(set(tickers)) != len(tickers):
        raise ValidationError("Duplicate tickers detected. Please use unique tickers.")


# ============================================================
# Data Fetching
# ============================================================
def download_prices(tickers: List[str], period: str) -> pd.DataFrame:
    """
    Download historical adjusted close prices for the given tickers.

    Args:
        tickers: list of ticker symbols, e.g. ["AAPL", "MSFT"].
        period: yfinance period string, e.g. "1y", "3y", "5y".

    Returns:
        DataFrame of close prices, columns = tickers, index = dates.

    Raises:
        DataFetchError: if download fails, tickers are invalid, or data is empty.
    """
    try:
        data = yf.download(
            tickers, period=period, auto_adjust=True, progress=False
        )["Close"]
    except Exception as exc:  # network errors, bad tickers, etc.
        raise DataFetchError(
            f"Failed to download data from Yahoo Finance: {exc}"
        ) from exc

    # yfinance returns a Series for a single ticker, DataFrame for multiple.
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])

    data = data.dropna(how="all")

    if data.empty:
        raise DataFetchError(
            "No price data was returned. Please check that your ticker "
            "symbols are valid and try again."
        )

    # Detect tickers that came back entirely empty (invalid symbol).
    missing = [t for t in tickers if t not in data.columns or data[t].isna().all()]
    if missing:
        raise DataFetchError(
            f"No data found for ticker(s): {', '.join(missing)}. "
            "Please verify the symbol(s) are correct."
        )

    data = data[tickers].dropna(how="any")

    if data.empty or len(data) < 3:
        raise DataFetchError(
            "Not enough overlapping historical data across the selected "
            "tickers to run a VaR analysis. Try a longer lookback period."
        )

    return data


# ============================================================
# Core Statistical Computations
# ============================================================
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily percentage returns from a price DataFrame."""
    return prices.pct_change().dropna()


def compute_portfolio_returns(
    returns: pd.DataFrame, weights: np.ndarray
) -> pd.Series:
    """
    Compute the daily portfolio return series as a weighted combination of
    individual asset returns: r_p = R @ w
    (matrix multiplication of the returns matrix by the weight vector).
    Works for a single asset too, since weights=[1.0] collapses to a scalar.
    """
    portfolio_returns = returns.values @ weights
    return pd.Series(portfolio_returns, index=returns.index, name="portfolio_return")


def historical_var(
    portfolio_returns: pd.Series,
    portfolio_value: float,
    confidence_level: float,
    horizon_days: int,
) -> Tuple[float, float]:
    """
    Historical Simulation VaR: uses the empirical distribution of past
    portfolio returns (no distributional assumption). Scales the daily
    return series by sqrt(time) to approximate the multi-day horizon.

    Returns:
        (var_pct, var_dollar) as positive-loss magnitudes.
    """
    scaled_returns = portfolio_returns * np.sqrt(horizon_days)
    # The (1 - confidence) percentile of the loss distribution.
    var_pct = -np.percentile(scaled_returns, (1 - confidence_level) * 100)
    var_pct = max(var_pct, 0.0)
    var_dollar = var_pct * portfolio_value
    return float(var_pct), float(var_dollar)


def parametric_var(
    portfolio_returns: pd.Series,
    portfolio_value: float,
    confidence_level: float,
    horizon_days: int,
) -> Tuple[float, float]:
    """
    Parametric (Variance-Covariance) VaR: assumes portfolio returns follow a
    normal distribution N(mu, sigma^2). Uses the historical mean/std of the
    portfolio return series and the inverse normal CDF (z-score) at the
    chosen confidence level, scaled to the desired time horizon via the
    square-root-of-time rule.
    """
    mu = portfolio_returns.mean()          # mean daily portfolio return
    sigma = portfolio_returns.std()        # daily portfolio volatility (std dev)
    z_score = norm.ppf(1 - confidence_level)  # inverse normal CDF, e.g. -1.645 for 95%

    # VaR% = -(mu*t + z*sigma*sqrt(t)); z is negative so this yields a positive loss.
    var_pct = -(mu * horizon_days + z_score * sigma * np.sqrt(horizon_days))
    var_pct = max(var_pct, 0.0)
    var_dollar = var_pct * portfolio_value
    return float(var_pct), float(var_dollar)


def monte_carlo_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    confidence_level: float,
    horizon_days: int,
    num_simulations: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Monte Carlo VaR: simulates many possible future asset return paths by
    sampling from a multivariate normal distribution parameterized by the
    historical mean vector and covariance matrix of asset returns (scaled to
    the chosen time horizon), then projects each simulated path onto the
    portfolio via the weight vector and takes the empirical percentile of
    the resulting simulated portfolio returns.
    """
    rng = np.random.default_rng(seed)
    mean_returns = returns.mean().values   # per-asset average daily return
    cov_matrix = returns.cov().values       # per-asset covariance matrix (n x n)

    # Draw `num_simulations` random asset-return vectors from N(mu*t, cov*t).
    # Works for a single asset too: cov_matrix collapses to a 1x1 matrix.
    simulated_asset_returns = rng.multivariate_normal(
        mean_returns * horizon_days,
        cov_matrix * horizon_days,
        size=num_simulations,
    )
    # Project simulated asset returns onto the portfolio via weights.
    simulated_portfolio_returns = simulated_asset_returns @ weights

    var_pct = -np.percentile(simulated_portfolio_returns, (1 - confidence_level) * 100)
    var_pct = max(var_pct, 0.0)
    var_dollar = var_pct * portfolio_value
    return float(var_pct), float(var_dollar)


def compute_cumulative_returns(portfolio_returns: pd.Series) -> pd.Series:

    """Cumulative growth of $1 invested at the start of the period."""
    return (1 + portfolio_returns).cumprod()


def compute_drawdown(cumulative_returns: pd.Series) -> pd.Series:
    """
    Historical drawdown series: percentage decline from the running
    all-time-high of the cumulative return curve.
    """
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown


# ============================================================
# Orchestration
# ============================================================
def run_var_analysis(
    tickers: List[str],
    weights: List[float],
    portfolio_value: float,
    confidence_level: float,
    horizon_days: int,
    period: str = "5y",
    num_simulations: int = 10_000,
    seed: int = 42,
) -> Dict[str, Any]:

    """
    End-to-end VaR pipeline: validate -> fetch -> compute -> package results.

    Returns a structured dict:
        {
            "success": bool,
            "error": Optional[str],
            "historical": {"var_pct": float, "var_dollar": float},
            "parametric": {"var_pct": float, "var_dollar": float},
            "portfolio_returns": pd.Series,
            "prices": pd.DataFrame,
            "cumulative_returns": pd.Series,
            "drawdown": pd.Series,
            "n_observations": int,
        }
    """
    result: Dict[str, Any] = {"success": False, "error": None}

    # Clean up ticker input (strip whitespace, uppercase).
    tickers = [t.strip().upper() for t in tickers if t.strip()]

    try:
        validate_inputs(tickers, weights)
        prices = download_prices(tickers, period)
        returns = compute_returns(prices)

        weights_arr = np.array(weights, dtype=float)
        portfolio_returns = compute_portfolio_returns(returns, weights_arr)

        hist_pct, hist_dollar = historical_var(
            portfolio_returns, portfolio_value, confidence_level, horizon_days
        )
        param_pct, param_dollar = parametric_var(
            portfolio_returns, portfolio_value, confidence_level, horizon_days
        )
        mc_pct, mc_dollar = monte_carlo_var(
            returns,
            weights_arr,
            portfolio_value,
            confidence_level,
            horizon_days,
            num_simulations,
            seed,
        )

        cumulative_returns = compute_cumulative_returns(portfolio_returns)

        drawdown = compute_drawdown(cumulative_returns)

        result.update(
            {
                "success": True,
                "error": None,
                "historical": {"var_pct": hist_pct, "var_dollar": hist_dollar},
                "parametric": {"var_pct": param_pct, "var_dollar": param_dollar},
                "monte_carlo": {"var_pct": mc_pct, "var_dollar": mc_dollar},
                "portfolio_returns": portfolio_returns,

                "asset_returns": returns,
                "prices": prices,
                "cumulative_returns": cumulative_returns,
                "drawdown": drawdown,
                "n_observations": int(len(portfolio_returns)),
                "tickers": tickers,
                "weights": weights,
            }
        )
        return result

    except (ValidationError, DataFetchError) as exc:
        result["error"] = str(exc)
        return result
    except Exception as exc:  # catch-all safety net for unexpected failures
        result["error"] = f"An unexpected error occurred: {exc}"
        return result
