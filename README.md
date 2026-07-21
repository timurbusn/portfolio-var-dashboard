# 📊 Finfy — Portfolio Volatility & Risk Terminal

Finfy is a Python-based, Streamlit-powered quantitative risk terminal for
multi-asset equity portfolios. It computes institutional-grade
**Value-at-Risk (VaR)** and **Expected Shortfall (Conditional VaR)** using
three independent methodologies, and visualizes portfolio performance,
tail-risk distributions, and historical drawdowns — all backed by live
market data from Yahoo Finance.

---

## ✨ Features

- **Three VaR Models**
  - **Historical Simulation VaR** — empirical percentile of actual
    (overlapping, compounded) historical portfolio returns.
  - **Parametric (Variance-Covariance) VaR** — closed-form Gaussian VaR
    using the portfolio's mean/std and the normal quantile, scaled via
    the square-root-of-time rule.
  - **Monte Carlo VaR** — true multi-day, correlated Geometric Brownian
    Motion (GBM) price-path simulation using Cholesky decomposition of
    the asset covariance matrix, fully vectorized across simulations and
    trading days.

- **Expected Shortfall (CVaR / Tail Loss)** for all three models —
  quantifies the average magnitude of loss *beyond* the VaR cutoff.

- **Cornish-Fisher Expansion** — a fat-tail/skewness-corrected VaR that
  adjusts the standard normal quantile for the portfolio's empirical
  skewness and excess kurtosis.

- **Interactive Streamlit UI**
  - Dynamic ticker selection and per-asset dollar allocation
  - Adjustable confidence level (90% / 95% / 99%) and time horizon (1–100 days)
  - Adjustable historical lookback period (1–10 years)
  - Live Plotly charts: normalized asset performance, return-distribution
    tail chart with VaR boundary, cumulative growth, and drawdown
  - KPI metric cards for VaR, CVaR, and Cornish-Fisher-adjusted VaR

## 🏗️ Architecture

The project follows a strict separation of concerns:

| File | Responsibility |
|---|---|
| `var_engine.py` | All statistical/mathematical logic: data validation, data fetching (`yfinance`), returns computation, VaR/CVaR/Cornish-Fisher calculations, Monte Carlo GBM engine, and orchestration via `run_var_analysis()`. |
| `app.py` | Purely presentational Streamlit frontend. Renders inputs, KPI metrics, and Plotly charts using the structured payload returned by `var_engine.py`. Performs **zero** statistical computation of its own. |
| `portfolio_var.py` | Standalone CLI reference script for running a quick VaR analysis from the terminal (edit the `CONFIG` section and run directly). |

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/timurbusn/portfolio-var-dashboard.git
cd portfolio-var-dashboard
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (typically `http://localhost:8501`).

### Run the CLI Reference Script

```bash
python portfolio_var.py
```

Edit the `CONFIG` section at the top of `portfolio_var.py` to change
tickers, weights, portfolio value, confidence level, time horizon, and
lookback period.

## 📦 Dependencies

See [`requirements.txt`](requirements.txt):

- `numpy` — vectorized numerical computation
- `pandas` — time series / return data handling
- `scipy` — normal distribution functions (`norm.ppf`, `norm.pdf`)
- `yfinance` — historical market data retrieval
- `streamlit` — interactive web UI
- `plotly` — interactive charting
- `requests` — HTTP dependency for data retrieval

## ⚠️ Disclaimer

Finfy is an educational/analytical tool and does **not** constitute
financial advice. Value-at-Risk and Expected Shortfall estimates rely on
historical data and model assumptions (normality, i.i.d. returns, etc.)
that may not hold in all market conditions. Past performance is not
indicative of future results.

## 📄 License

This project is provided as-is for educational and portfolio-analysis
purposes.
