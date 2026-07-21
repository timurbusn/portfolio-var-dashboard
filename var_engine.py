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
    - `tickers` and `weights` are expected to arrive as clean, already
      -structured Python lists straight from the frontend's `st.multiselect`
      and per-ticker numeric weight inputs -- there is no comma-splitting
      or free-text parsing performed anywhere in this module. Validation
      still guards against malformed/edge-case input (duplicates, mismatched
      lengths, non-positive weight sums, single-asset portfolios, etc.).
    - Institutional-grade tail-risk metrics: every VaR model also reports
      Expected Shortfall / Conditional VaR (the average loss magnitude
      *beyond* the VaR cutoff), and the Parametric model additionally
      reports a Cornish-Fisher-adjusted VaR that corrects the standard
      normal quantile for empirical skewness and excess kurtosis (fat
      tails), which the plain Gaussian assumption underestimates.
    - The Monte Carlo engine simulates true multi-day, correlated
      Geometric Brownian Motion price paths (via Cholesky decomposition of
      the asset covariance matrix) rather than a single sqrt-time-scaled
      normal draw, fully vectorized across all simulations and trading
      days with no Python-level simulation loop.
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
    Validate that tickers and weights (already provided as structured lists
    by the frontend's multiselect + numeric weight widgets) are well-formed
    and compatible. Handles the single-asset edge case gracefully.

    Raises:
        ValidationError: on any mismatch or invalid values.
    """
    if not tickers:
        raise ValidationError("Please select at least one stock ticker.")

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

    if len(set(tickers)) != len(tickers):
        raise ValidationError("Duplicate tickers detected. Please select unique tickers.")


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


def normalize_weights(weights: List[float]) -> List[float]:
    """
    Normalize an arbitrary list of non-negative weight values (e.g. raw
    percentage inputs from per-ticker sliders that may not sum exactly to
    100) into fractions that sum to 1.0. Robust to a single-asset portfolio.
    """
    total = sum(weights)
    if total <= 0:
        raise ValidationError("Weights must sum to a positive value.")
    return [w / total for w in weights]


# ============================================================
# Data Fetching
# ============================================================
def download_prices(tickers: List[str], period: str) -> pd.DataFrame:
    """
    Download historical adjusted close prices for the given tickers.

    Scales cleanly to enterprise-grade portfolios containing dozens of
    tickers. As the number of selected assets grows, the odds of any single
    name having sparse/misaligned trading-day history (different listing
    dates, exchange holidays, thin-volume names, etc.) rises sharply -- so
    once all tickers are aligned onto a common trading-day index, small
    interior gaps are filled via forward-fill then backward-fill rather
    than dropping otherwise-valid rows or raising, which would otherwise
    needlessly shrink the usable history for the whole portfolio.

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
            "No price data was returned. Please check that your selected "
            "tickers are valid and try again."
        )

    # Detect tickers that came back *entirely* empty (invalid symbol) --
    # these must still hard-fail, since there is nothing to fill.
    missing = [t for t in tickers if t not in data.columns or data[t].isna().all()]
    if missing:
        raise DataFetchError(
            f"No data found for ticker(s): {', '.join(missing)}. "
            "Please verify the symbol(s) are correct."
        )

    data = data[tickers]

    # Data Completeness: align all tickers onto the full common trading-day
    # index first, then fill interior/edge gaps (a later IPO, a holiday
    # mismatch across exchanges, a thinly-traded name, etc.) via forward-
    # fill followed by backward-fill, instead of dropping rows. This keeps
    # the full lookback history usable even as the portfolio scales to
    # dozens of large-cap, ETF, and crypto tickers with heterogeneous
    # trading calendars.
    data = data.ffill().bfill()

    # Any column that is still entirely NaN after fill (e.g. a ticker with
    # zero overlap with the rest of the universe) cannot be salvaged.
    still_missing = [t for t in tickers if data[t].isna().all()]
    if still_missing:
        raise DataFetchError(
            f"No usable data found for ticker(s): {', '.join(still_missing)} "
            "after gap-filling. Please verify the symbol(s) are correct."
        )

    data = data.dropna(how="any")

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

    min_required = horizon_days + 1
    if n_obs < min_required or n_obs - horizon_days < 5:
        # Not enough history for genuine rolling windows -> fall back to the
        # sqrt-time scalar approximation on the raw daily series.
        return portfolio_returns.values * np.sqrt(horizon_days)

    growth = (1.0 + portfolio_returns).values
    log_growth = np.log(growth)
    cumulative_log = np.concatenate(([0.0], np.cumsum(log_growth)))
    window_log_returns = cumulative_log[horizon_days:] - cumulative_log[:-horizon_days]
    compounded_returns = np.exp(window_log_returns) - 1.0
    return compounded_returns


def _expected_shortfall_from_sample(
    sample_returns: np.ndarray, var_pct: float
) -> float:
    """
    Empirical Expected Shortfall (Conditional VaR): the mean magnitude of
    loss across all sample observations that fall at or beyond the VaR
    cutoff (i.e. the average of the tail *worse* than the VaR boundary
    itself). Falls back to `var_pct` if the tail sample is empty (e.g. an
    extremely short lookback window), so CVaR >= VaR always holds.
    """
    tail = sample_returns[sample_returns <= -var_pct]
    if tail.size == 0:
        return var_pct
    cvar_pct = -float(tail.mean())
    return max(cvar_pct, var_pct)


def historical_var(
    portfolio_returns: pd.Series,
    portfolio_value: float,
    confidence_level: float,
    horizon_days: int,
) -> Tuple[float, float, float, float]:
    """
    Historical Simulation VaR: uses the empirical distribution of past
    portfolio returns (no distributional assumption).

    For horizon_days == 1, this is simply the empirical percentile of the
    daily return series. For horizon_days > 1, actual overlapping N-day
    compounded rolling-window returns are used (see
    `_compounded_rolling_window_returns`) to more accurately reflect
    multi-day risk than a naive sqrt(time) scalar.

    Also computes Historical Expected Shortfall (CVaR): the average
    magnitude of loss across all historical (or rolling-window) return
    observations that fall at or beyond the VaR cutoff.

    Returns:
        (var_pct, var_dollar, cvar_pct, cvar_dollar) as positive-loss
        magnitudes.
    """
    if horizon_days == 1:
        scaled_returns = portfolio_returns.values
    else:
        scaled_returns = _compounded_rolling_window_returns(
            portfolio_returns, horizon_days
        )

    var_pct = -np.percentile(scaled_returns, (1 - confidence_level) * 100)
    var_pct = max(var_pct, 0.0)
    var_dollar = var_pct * portfolio_value

    cvar_pct = _expected_shortfall_from_sample(scaled_returns, var_pct)
    cvar_dollar = cvar_pct * portfolio_value

    return float(var_pct), float(var_dollar), float(cvar_pct), float(cvar_dollar)


def parametric_var(
    portfolio_returns: pd.Series,
    portfolio_value: float,
    confidence_level: float,
    horizon_days: int,
) -> Tuple[float, float, float, float, float, float]:
    """
    Parametric (Variance-Covariance) VaR: assumes portfolio returns follow a
    normal distribution N(mu, sigma^2). Uses the historical mean/std of the
    portfolio return series and the inverse normal CDF (z-score) at the
    chosen confidence level, scaled to the desired time horizon via the
    square-root-of-time rule -- the mathematically standard convention for
    scaling a normal-distribution VaR estimate to multi-day horizons under
    the i.i.d. returns assumption. This scaling remains numerically stable
    even at macro-scale horizons (up to 100 days).

    Also reports two additional institutional-grade tail-risk metrics:

    1. Parametric Expected Shortfall (CVaR), using the closed-form normal
       tail-expectation formula:
           CVaR = -(mu*T) + sigma*sqrt(T) * phi(z_alpha) / (1 - confidence)
       where phi is the standard normal PDF and z_alpha = Phi^-1(1 - confidence).

    2. Cornish-Fisher-adjusted VaR, which corrects the standard normal
       quantile z_alpha for empirical skewness (S) and excess kurtosis (K)
       of the portfolio return series -- capturing fat tails and return
       asymmetry that the plain Gaussian assumption misses:
           z_CF = z + (S/6)(z^2 - 1) + (K/24)(z^3 - 3z) - (S^2/36)(2z^3 - 5z)
       and then applying z_CF in place of z in the standard VaR formula.

    Returns:
        (var_pct, var_dollar, cvar_pct, cvar_dollar,
         cf_var_pct, cf_var_dollar)
    """
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    z_score = norm.ppf(1 - confidence_level)

    # --- Standard Gaussian Parametric VaR ---
    var_pct = -(mu * horizon_days + z_score * sigma * np.sqrt(horizon_days))
    var_pct = max(var_pct, 0.0)
    var_dollar = var_pct * portfolio_value

    # --- Parametric Expected Shortfall (closed-form normal tail mean) ---
    tail_probability = 1 - confidence_level
    cvar_pct = -(mu * horizon_days) + sigma * np.sqrt(horizon_days) * (
        norm.pdf(z_score) / tail_probability
    )
    cvar_pct = max(cvar_pct, var_pct)
    cvar_dollar = cvar_pct * portfolio_value

    # --- Cornish-Fisher Expansion (fat tails / skewness correction) ---
    # Guard against NaN skew/kurtosis on extremely short return samples
    # (pandas returns NaN for skew/kurt with fewer than ~4 observations).
    skew = portfolio_returns.skew()
    kurt = portfolio_returns.kurt()  # excess kurtosis (pandas default, Fisher's definition)
    skew = 0.0 if pd.isna(skew) else float(skew)
    kurt = 0.0 if pd.isna(kurt) else float(kurt)

    z2 = z_score ** 2
    z3 = z_score ** 3
    z_cf = (
        z_score
        + (skew / 6.0) * (z2 - 1)
        + (kurt / 24.0) * (z3 - 3 * z_score)
        - (skew ** 2 / 36.0) * (2 * z3 - 5 * z_score)
    )

    cf_var_pct = -(mu * horizon_days + z_cf * sigma * np.sqrt(horizon_days))
    cf_var_pct = max(cf_var_pct, 0.0)
    cf_var_dollar = cf_var_pct * portfolio_value

    return (
        float(var_pct),
        float(var_dollar),
        float(cvar_pct),
        float(cvar_dollar),
        float(cf_var_pct),
        float(cf_var_dollar),
    )


def monte_carlo_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    confidence_level: float,
    horizon_days: int,
    num_simulations: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """
    Monte Carlo VaR: simulates true multi-day, correlated Geometric Brownian
    Motion (GBM) asset price paths rather than a single sqrt-time-scaled
    normal draw.

    Methodology:
        1. Decompose the daily asset covariance matrix via Cholesky
           decomposition (Sigma = L @ L.T) so independent standard-normal
           shocks can be correlated across assets: shock = Z @ L.T.
        2. For each of `horizon_days` trading days, draw a fresh
           (num_simulations x n_assets) tensor of correlated daily log-
           return shocks and apply the discrete GBM update:
               daily_log_return = (mu - 0.5*sigma^2) + sqrt(dt)*correlated_shock
           Because GBM log-returns are additive across time steps, the full
           T-day path is obtained by summing the T daily log-return
           increments -- this is mathematically identical to stepping the
           simulation day-by-day, but fully vectorized (a single NumPy
           tensor operation across the time axis) rather than a slow
           Python-level loop over each of the `num_simulations` paths.
        3. Convert each simulation's cumulative log-return to a simple
           terminal return per asset, then project onto the portfolio via
           the (normalized) weight vector to obtain `num_simulations`
           simulated portfolio-level returns over the full horizon.
        4. VaR = the empirical percentile of the simulated portfolio
           returns; CVaR = the mean loss across all simulated paths whose
           loss meets or exceeds the VaR cutoff.

    Works cleanly for a single asset too, since the covariance matrix
    collapses to a 1x1 matrix and its Cholesky factor is simply its
    standard deviation.

    Returns:
        (var_pct, var_dollar, cvar_pct, cvar_dollar)
    """
    rng = np.random.default_rng(seed)
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    n_assets = len(mean_returns)

    # Cholesky decomposition of the daily covariance matrix, with a tiny
    # diagonal jitter fallback in case of a near-singular covariance matrix
    # (e.g. two nearly perfectly-correlated assets).
    try:
        chol_factor = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        jitter = 1e-10 * np.eye(n_assets)
        chol_factor = np.linalg.cholesky(cov_matrix + jitter)

    asset_variance = np.diag(cov_matrix)
    drift = mean_returns - 0.5 * asset_variance  # GBM log-return drift term
    dt = 1.0  # one trading day per step

    if horizon_days == 1:
        # Fast path: a single correlated-shock draw is sufficient.
        z = rng.standard_normal((num_simulations, n_assets))
        correlated_shocks = z @ chol_factor.T
        cumulative_log_returns = drift + np.sqrt(dt) * correlated_shocks
    else:
        # True multi-day path simulation: draw one correlated shock tensor
        # per trading day across the full horizon, then accumulate the
        # daily log-return increments (vectorized over the time axis --
        # no Python-level loop over simulations).
        z = rng.standard_normal((horizon_days, num_simulations, n_assets))
        correlated_shocks = z @ chol_factor.T  # (horizon_days, num_simulations, n_assets)
        daily_log_returns = drift + np.sqrt(dt) * correlated_shocks
        cumulative_log_returns = daily_log_returns.sum(axis=0)  # (num_simulations, n_assets)

    simulated_asset_returns = np.exp(cumulative_log_returns) - 1.0
    simulated_portfolio_returns = simulated_asset_returns @ weights

    var_pct = -np.percentile(simulated_portfolio_returns, (1 - confidence_level) * 100)
    var_pct = max(var_pct, 0.0)
    var_dollar = var_pct * portfolio_value

    cvar_pct = _expected_shortfall_from_sample(simulated_portfolio_returns, var_pct)
    cvar_dollar = cvar_pct * portfolio_value

    return float(var_pct), float(var_dollar), float(cvar_pct), float(cvar_dollar)


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

    `tickers` and `weights` are expected to be clean Python lists sourced
    directly from the frontend's `st.multiselect` ticker picker and
    per-ticker numeric weight widgets -- no text parsing happens here.

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
                "prices": pd.DataFrame,
                "normalized_prices": pd.DataFrame,
                "asset_returns": pd.DataFrame,
            },
            "portfolio": {
                "returns": pd.Series,
                "cumulative_returns": pd.Series,
                "drawdown": pd.Series,
                "tickers": List[str],
                "weights": List[float],   # normalized fractions, sum to 1.0
                "n_observations": int,
                "horizon_days": int,
                "portfolio_value": float,
                "confidence_level": float,
            },
            "risk_metrics": {
                "historical": {
                    "var_pct": float, "var_dollar": float,
                    "cvar_pct": float, "cvar_dollar": float,
                },
                "parametric": {
                    "var_pct": float, "var_dollar": float,
                    "cvar_pct": float, "cvar_dollar": float,
                    "cf_var_pct": float, "cf_var_dollar": float,
                },
                "monte_carlo": {
                    "var_pct": float, "var_dollar": float,
                    "cvar_pct": float, "cvar_dollar": float,
                },
            },
        }
    """
    result: Dict[str, Any] = {"success": False, "error": None}

    def _report(message: str) -> None:
        if status_callback is not None:
            try:
                status_callback(message)
            except Exception:
                pass

    # Tickers arrive pre-cleaned from st.multiselect, but we defensively
    # normalize case/whitespace in case of programmatic callers.
    tickers = [t.strip().upper() for t in tickers if t and str(t).strip()]

    try:
        _report("Finfy Core: Validating portfolio configuration...")
        horizon_days = validate_horizon(horizon_days)
        validate_inputs(tickers, weights)
        normalized_weights = normalize_weights(weights)

        _report("Finfy Core: Accessing institutional equity endpoints via yfinance...")
        prices = download_prices(tickers, period)

        _report("Finfy Math: Reindexing historical price series and asset weights...")
        returns = compute_returns(prices)
        weights_arr = np.array(normalized_weights, dtype=float)
        portfolio_returns = compute_portfolio_returns(returns, weights_arr)
        normalized_prices = compute_normalized_prices(prices)

        _report("Finfy Math: Building dynamic variance-covariance matrices...")
        (
            param_pct,
            param_dollar,
            param_cvar_pct,
            param_cvar_dollar,
            cf_var_pct,
            cf_var_dollar,
        ) = parametric_var(portfolio_returns, portfolio_value, confidence_level, horizon_days)

        _report("Finfy Math: Resampling empirical historical return distributions...")
        hist_pct, hist_dollar, hist_cvar_pct, hist_cvar_dollar = historical_var(
            portfolio_returns, portfolio_value, confidence_level, horizon_days
        )

        _report("Finfy Math: Running correlated Monte Carlo GBM path simulation (10,000 paths)...")
        mc_pct, mc_dollar, mc_cvar_pct, mc_cvar_dollar = monte_carlo_var(
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
                    "weights": normalized_weights,
                    "n_observations": int(len(portfolio_returns)),
                    "horizon_days": horizon_days,
                    "portfolio_value": float(portfolio_value),
                    "confidence_level": float(confidence_level),
                },
                "risk_metrics": {
                    "historical": {
                        "var_pct": hist_pct,
                        "var_dollar": hist_dollar,
                        "cvar_pct": hist_cvar_pct,
                        "cvar_dollar": hist_cvar_dollar,
                    },
                    "parametric": {
                        "var_pct": param_pct,
                        "var_dollar": param_dollar,
                        "cvar_pct": param_cvar_pct,
                        "cvar_dollar": param_cvar_dollar,
                        "cf_var_pct": cf_var_pct,
                        "cf_var_dollar": cf_var_dollar,
                    },
                    "monte_carlo": {
                        "var_pct": mc_pct,
                        "var_dollar": mc_dollar,
                        "cvar_pct": mc_cvar_pct,
                        "cvar_dollar": mc_cvar_dollar,
                    },
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
