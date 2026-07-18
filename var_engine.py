"""
var_engine.py
=============

Finfy Core — the backend risk-analytics engine for the Finfy Portfolio
Volatility Analytics platform. Encapsulates all data fetching, validation,
and statistical computation logic so the frontend (`app.py`) stays purely
presentational and performs zero data manipulation of its own.

Public API:
    run_var_analysis(...) -> Dict[str, Any]   (a structured, nested payload)

Design notes:
    - All functions are pure and independently testable.
    - Every function that can fail raises a specific, catchable exception
      with a human-readable message; `run_var_analysis` converts these into
      a structured `success/error` result so Streamlit can render a clean
      error banner instead of crashing.
    - Handles the single-asset edge case (no covariance matrix needed).
    - Multi-day horizons (1-100 days) are handled using rigorous
      conventions: Historical VaR uses actual overlapping N-day compounded
      rolling-window returns (not a naive sqrt-time scalar on daily
      returns), while Parametric VaR uses the mathematically standard
      square-root-of-time scaling of the normal distribution's mean/std.
    - The orchestrator returns a *unified, nested* payload -- market data,
      portfolio-level series, and risk metrics are each cleanly namespaced
      so the frontend never needs to reshape or derive data itself.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# Suppress a harmless urllib3/LibreSSL compatibility warning triggered at
# import time on some macOS Python builds. Does not affect correctness.
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

# Hard bounds for the multi-day time horizon slider.
MIN_HORIZON_DAYS: int = 1
MAX_HORIZON_DAYS: int = 100

StatusCallback = Optional[Callable[[str], None]]


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
    Handles the single-asset edge case gracefully and guards against
    common malformed input (whitespace, duplicates, negative/zero-sum
    weights, mismatched counts).

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
    if weight_sum <= 0:
        raise ValidationError("Weights must sum to a positive value.")

    if not np.isclose(weight_sum, 1.0, atol=1e-3):
        raise ValidationError(
            f"Weights must sum to 1.0 (100%). Current sum = {weight_sum:.4f}."
        )

    if len(set(tickers)) != len(tickers):
        raise ValidationError("Duplicate tickers detected. Please use unique tickers.")


def validate_horizon(horizon_days: int) -> int:
    """Clamp/validate the requested time horizon to a sane [1, 100] range."""
    try:
        horizon_days = int(horizon_days)
    except (TypeError, ValueError):
        raise ValidationError("Time horizon must be an integer number of days.")

    if horizon_days < MIN_HORIZON_DAYS or horizon_days > MAX_HORIZON_DAYS:
        raise ValidationError(
            f"Time horizon must be between {MIN_HORIZON_DAYS} and "
            f"{MAX_HORIZON_DAYS} days."
        )
    return horizon_days


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


def compute_normalized_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Rebase every asset's price series to a common starting value of 100,
    so that relative performance across assets with very different price
    levels (e.g. a $50 stock vs. a $3,000 stock) can be visually compared
    on the same chart axis.
    """
    return (prices / prices.iloc[0]) * 100.0


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


def _compounded_rolling_window_returns(
    portfolio_returns: pd.Series, horizon_days: int
) -> np.ndarray:
    """
    Build the empirical distribution of actual overlapping N-day compounded
    portfolio returns, rather than approximating multi-day risk by scaling
    single-day returns by sqrt(time). This is the more rigorous convention
    for Historical Simulation VaR at horizons > 1 day, since it captures
    real historical compounding/autocorrelation effects instead of assuming
    i.i.d. daily returns.

    For a horizon of N days, each window's compounded return is:
        prod(1 + r_i) - 1   for i in [t, t+N)

    If there are not enough observations to form at least a handful of
    non-overlapping windows, this gracefully falls back to the sqrt-time
    scaled single-day series so the analysis never crashes on short lookback
    periods.
    """
    n_obs = len(portfolio_returns)

    # Need at least horizon_days+1 observations to form a single window;
    # require a modest minimum of usable windows for a meaningful empirical
    # distribution (at least 5, when available).
    min_required = horizon_days + 1
    if n_obs < min_required or n_obs - horizon_days < 5:
        # Not enough history for genuine rolling windows -> fall back to the
        # sqrt-time scalar approximation on the raw daily series.
        return portfolio_returns.values * np.sqrt(horizon_days)

    growth = (1.0 + portfolio_returns).values
    # Rolling product of `horizon_days` consecutive growth factors, using a
    # log-sum-exp style approach for numerical stability, then compounded
    # back into a simple return.
    log_growth = np.log(growth)
    cumulative_log = np.concatenate(([0.0], np.cumsum(log_growth)))
    window_log_returns = cumulative_log[horizon_days:] - cumulative_log[:-horizon_days]
    compounded_returns = np.exp(window_log_returns) - 1.0
    return compounded_returns


def historical_var(
    portfolio_returns: pd.Series,
    portfolio_value: float,
    confidence_level: float,
    horizon_days: int,
) -> Tuple[float, float]:
    """
    Historical Simulation VaR: uses the empirical distribution of past
    portfolio returns (no distributional assumption).

    For horizon_days == 1, this is simply the empirical percentile of the
    daily return series. For horizon_days > 1, actual overlapping N-day
    compounded rolling-window returns are used (see
    `_compounded_rolling_window_returns`) to more accurately reflect
    multi-day risk than a naive sqrt(time) scalar.

    Returns:
        (var_pct, var_dollar) as positive-loss magnitudes.
    """
    if horizon_days == 1:
        scaled_returns = portfolio_returns.values
    else:
        scaled_returns = _compounded_rolling_window_returns(
            portfolio_returns, horizon_days
        )

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
    square-root-of-time rule -- the mathematically standard convention for
    scaling a normal-distribution VaR estimate to multi-day horizons under
    the i.i.d. returns assumption. This scaling remains numerically stable
    even at macro-scale horizons (up to 100 days).
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
    status_callback: StatusCallback = None,
) -> Dict[str, Any]:
    """
    End-to-end Finfy risk pipeline: validate -> fetch -> compute -> package.

    Args:
        status_callback: optional callable(str) invoked with human-readable
            engineering-phrase milestones as the pipeline progresses (used
            by the Streamlit frontend to drive a live `st.status()` widget).

    Returns a unified, nested payload so the frontend performs *zero* data
    manipulation of its own:
        {
            "success": bool,
            "error": Optional[str],
            "market_data": {
                "prices": pd.DataFrame,             # raw close prices
                "normalized_prices": pd.DataFrame,   # rebased to 100
                "asset_returns": pd.DataFrame,        # daily pct returns
            },
            "portfolio": {
                "returns": pd.Series,
                "cumulative_returns": pd.Series,
                "drawdown": pd.Series,
                "tickers": List[str],
                "weights": List[float],
                "n_observations": int,
                "horizon_days": int,
                "portfolio_value": float,
                "confidence_level": float,
            },
            "risk_metrics": {
                "historical": {"var_pct": float, "var_dollar": float},
                "parametric": {"var_pct": float, "var_dollar": float},
                "monte_carlo": {"var_pct": float, "var_dollar": float},
            },
        }
    """
    result: Dict[str, Any] = {"success": False, "error": None}

    def _report(message: str) -> None:
        if status_callback is not None:
            try:
                status_callback(message)
            except Exception:
                pass  # Never let UI callback errors break the analysis.

    # Clean up ticker input (strip whitespace, uppercase, drop blanks).
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]

    try:
        _report("Finfy Core: Validating portfolio configuration...")
        horizon_days = validate_horizon(horizon_days)
        validate_inputs(tickers, weights)

        _report("Finfy Core: Accessing institutional equity endpoints via yfinance...")
        prices = download_prices(tickers, period)

        _report("Finfy Math: Reindexing historical price series and asset weights...")
        returns = compute_returns(prices)
        weights_arr = np.array(weights, dtype=float)
        portfolio_returns = compute_portfolio_returns(returns, weights_arr)
        normalized_prices = compute_normalized_prices(prices)

        _report("Finfy Math: Building dynamic variance-covariance matrices...")
        param_pct, param_dollar = parametric_var(
            portfolio_returns, portfolio_value, confidence_level, horizon_days
        )

        _report("Finfy Math: Resampling empirical historical return distributions...")
        hist_pct, hist_dollar = historical_var(
            portfolio_returns, portfolio_value, confidence_level, horizon_days
        )

        _report("Finfy Math: Running Monte Carlo simulation lattice (10,000 paths)...")
        mc_pct, mc_dollar = monte_carlo_var(
            returns,
            weights_arr,
            portfolio_value,
            confidence_level,
            horizon_days,
            num_simulations,
            seed,
        )

        _report("Finfy Visuals: Compiling performance and drawdown analytics...")
        cumulative_returns = compute_cumulative_returns(portfolio_returns)
        drawdown = compute_drawdown(cumulative_returns)

        result.update(
            {
                "success": True,
                "error": None,
                "market_data": {
                    "prices": prices,
                    "normalized_prices": normalized_prices,
                    "asset_returns": returns,
                },
                "portfolio": {
                    "returns": portfolio_returns,
                    "cumulative_returns": cumulative_returns,
                    "drawdown": drawdown,
                    "tickers": tickers,
                    "weights": weights,
                    "n_observations": int(len(portfolio_returns)),
                    "horizon_days": horizon_days,
                    "portfolio_value": float(portfolio_value),
                    "confidence_level": float(confidence_level),
                },
                "risk_metrics": {
                    "historical": {"var_pct": hist_pct, "var_dollar": hist_dollar},
                    "parametric": {"var_pct": param_pct, "var_dollar": param_dollar},
                    "monte_carlo": {"var_pct": mc_pct, "var_dollar": mc_dollar},
                },
            }
        )
        _report("Finfy Core: Analysis complete.")
        return result

    except (ValidationError, DataFetchError) as exc:
        result["error"] = str(exc)
        return result
    except Exception as exc:  # catch-all safety net for unexpected failures
        result["error"] = f"An unexpected error occurred: {exc}"
        return result
