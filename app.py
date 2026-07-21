"""
app.py
======

Finfy — Portfolio Volatility Analytics.

A bulletproof, vanilla-Streamlit risk terminal. All statistical
computation and data shaping is delegated entirely to `var_engine.py`;
this module is purely presentational and uses only native Streamlit
components -- no injected custom CSS/HTML, and no external HTTP calls
(no logo CDNs, no yfinance metadata lookups) -- to eliminate the two
most common classes of frontend failure.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from var_engine import run_var_analysis, MIN_HORIZON_DAYS, MAX_HORIZON_DAYS

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Finfy // Portfolio Volatility Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Simple, Safe Local Asset Universe (no external registry/API calls)
# ============================================================
ASSET_UNIVERSE: List[str] = [
    "AAPL", "MSFT", "GOOG", "NVDA", "AMZN",
    "META", "TSLA", "JPM", "V", "WMT",
]
DEFAULT_TICKERS: List[str] = ["AAPL", "MSFT", "NVDA", "GOOG"]

CONFIDENCE_OPTIONS = ["90%", "95%", "99%"]
CONFIDENCE_MAP = {"90%": 0.90, "95%": 0.95, "99%": 0.99}
PERIOD_OPTIONS = {
    "1 Year": "1y",
    "2 Years": "2y",
    "3 Years": "3y",
    "5 Years": "5y",
    "10 Years": "10y",
}


# ============================================================
# Chart Builders (Plotly only -- no custom CSS/HTML)
# ============================================================
def build_asset_performance_chart(normalized_prices: pd.DataFrame) -> go.Figure:
    """Multi-line chart tracking each asset's normalized (base=100) path."""
    fig = go.Figure()
    for ticker in normalized_prices.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[ticker].values,
                mode="lines",
                name=ticker,
                hovertemplate=f"<b>{ticker}</b><br>%{{x|%Y-%m-%d}}<br>Index: %{{y:.2f}}<extra></extra>",
            )
        )
    fig.add_hline(y=100, line_width=1, line_dash="dot")
    fig.update_layout(
        template="plotly_dark",
        height=440,
        margin=dict(t=10, b=40, l=44, r=20),
        xaxis_title="",
        yaxis_title="Normalized Index (Base = 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def build_risk_tail_chart(
    portfolio_returns: pd.Series,
    var_pct: float,
    confidence_label: str,
    horizon_days: int,
) -> go.Figure:
    """Empirical returns distribution with a VaR cutoff marker."""
    returns_pct = portfolio_returns * 100

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=returns_pct,
            nbinsx=55,
            opacity=0.55,
            name="Return Distribution",
            histnorm="probability density",
        )
    )
    fig.add_vline(
        x=-var_pct * 100,
        line_width=3,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR Boundary ({confidence_label}, {horizon_days}d): -{var_pct:.2%}",
        annotation_position="top left",
    )
    fig.update_layout(
        template="plotly_dark",
        height=440,
        margin=dict(t=10, b=40, l=44, r=20),
        xaxis_title="Return over Horizon (%)",
        yaxis_title="Density",
        showlegend=False,
        bargap=0.02,
    )
    return fig


def build_cumulative_chart(cumulative_returns: pd.Series) -> go.Figure:
    """Line chart of cumulative portfolio growth over the lookback period."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode="lines",
            name="Cumulative Growth ($1 invested)",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(t=20, b=40, l=44, r=20),
        xaxis_title="Date",
        yaxis_title="Growth of $1",
    )
    return fig


def build_drawdown_chart(drawdown: pd.Series) -> go.Figure:
    """Area chart of historical portfolio drawdowns."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode="lines",
            fill="tozeroy",
            name="Drawdown (%)",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(t=20, b=40, l=44, r=20),
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
    )
    return fig


# ============================================================
# Header
# ============================================================
st.title("📊 Finfy — Portfolio Volatility Analytics")
st.caption("Historical · Parametric · Monte Carlo Value-at-Risk")

# ============================================================
# Sidebar — User Inputs
# ============================================================
with st.sidebar:
    st.header("Portfolio Configuration")

    tickers: List[str] = st.multiselect(
        "Select Tickers",
        options=ASSET_UNIVERSE,
        default=DEFAULT_TICKERS,
    )

    st.subheader("Dollar Allocation")

    # Dynamic dollar-amount input per selected ticker. The total portfolio
    # value is now derived automatically as the sum of these per-ticker
    # dollar amounts (no separate "Initial Investment" field needed), and
    # each amount is passed straight through as an unnormalized "weight" --
    # `run_var_analysis` normalizes any positive weight vector internally,
    # so raw dollar amounts work exactly like raw percentages did.
    raw_weights: List[float] = []
    if tickers:
        for t in tickers:
            amt = st.number_input(
                f"Amount invested in {t} ($)",
                min_value=0.0,
                value=1000.0,
                step=100.0,
                key=f"amount_{t}",
            )
            raw_weights.append(amt)
    else:
        st.caption("Select at least one ticker above to configure dollar allocations.")

    portfolio_value = float(sum(raw_weights))

    st.divider()

    confidence_label = st.select_slider(

        "Confidence Level",
        options=CONFIDENCE_OPTIONS,
        value="95%",
    )
    confidence_level = CONFIDENCE_MAP[confidence_label]

    horizon_days = st.slider(
        "Time Horizon (days)",
        min_value=MIN_HORIZON_DAYS,
        max_value=MAX_HORIZON_DAYS,
        value=1,
        step=1,
    )

    period_label = st.selectbox(
        "Historical Lookback Period", options=list(PERIOD_OPTIONS.keys()), index=3
    )
    period = PERIOD_OPTIONS[period_label]

    st.divider()
    run_clicked = st.button("Run Analysis", type="primary", disabled=not tickers)


# ============================================================
# Execution + Results
# ============================================================
if run_clicked or "last_result" in st.session_state:
    if run_clicked:
        with st.spinner("Calculating..."):
            result = run_var_analysis(
                tickers=tickers,
                weights=raw_weights,
                portfolio_value=portfolio_value,
                confidence_level=confidence_level,
                horizon_days=horizon_days,
                period=period,
            )
        st.session_state["last_result"] = result
    else:
        result = st.session_state["last_result"]

    if not result["success"]:
        st.error(result["error"])
        st.stop()

    # Unpack the unified, nested payload -- zero data manipulation here.
    market_data = result["market_data"]
    portfolio = result["portfolio"]
    risk_metrics = result["risk_metrics"]

    hist = risk_metrics["historical"]
    param = risk_metrics["parametric"]
    mc = risk_metrics["monte_carlo"]

    eff_horizon = portfolio["horizon_days"]
    eff_confidence_label = confidence_label

    # ---------------- Portfolio Composition (plain text, no images) ----------------
    st.subheader("Portfolio Composition")
    for t, w in zip(portfolio["tickers"], portfolio["weights"]):
        st.code(f" {t} | {w:.2%}")

    st.write("")

    # ---------------- Native KPI Metrics ----------------
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            f"Historical VaR ({eff_confidence_label}, {eff_horizon}D)",
            f"${hist['var_dollar']:,.2f}",
            f"-{hist['var_pct']:.2%}",
            delta_color="inverse",
        )
    with col2:
        st.metric(
            f"Parametric VaR ({eff_confidence_label}, {eff_horizon}D)",
            f"${param['var_dollar']:,.2f}",
            f"-{param['var_pct']:.2%}",
            delta_color="inverse",
        )
    with col3:
        st.metric(
            f"Monte Carlo VaR ({eff_confidence_label}, {eff_horizon}D)",
            f"${mc['var_dollar']:,.2f}",
            f"-{mc['var_pct']:.2%}",
            delta_color="inverse",
        )
    with col4:
        st.metric(label="Total Portfolio Value", value=f"${portfolio_value:,.2f}")

    with col5:
        st.metric("Observations", f"{portfolio['n_observations']:,}")

    st.write("")

    # ---------------- Expected Shortfall (CVaR) Metrics ----------------
    st.subheader("Expected Shortfall (Tail Loss / CVaR)")
    st.caption(
        "The average magnitude of loss in the worst-case scenarios that "
        "fall beyond the VaR boundary -- a deeper measure of tail risk than "
        "VaR alone."
    )
    cvar_col1, cvar_col2, cvar_col3 = st.columns(3)
    with cvar_col1:
        st.metric(
            f"Historical CVaR ({eff_confidence_label}, {eff_horizon}D)",
            f"${hist['cvar_dollar']:,.2f}",
            f"-{hist['cvar_pct']:.2%}",
            delta_color="inverse",
        )
    with cvar_col2:
        st.metric(
            f"Parametric CVaR ({eff_confidence_label}, {eff_horizon}D)",
            f"${param['cvar_dollar']:,.2f}",
            f"-{param['cvar_pct']:.2%}",
            delta_color="inverse",
        )
    with cvar_col3:
        st.metric(
            f"Monte Carlo CVaR ({eff_confidence_label}, {eff_horizon}D)",
            f"${mc['cvar_dollar']:,.2f}",
            f"-{mc['cvar_pct']:.2%}",
            delta_color="inverse",
        )

    with st.expander("Advanced: Cornish-Fisher Fat-Tail Adjusted VaR"):
        st.caption(
            "Corrects the standard Parametric VaR's normal-distribution "
            "assumption for the portfolio's empirical skewness and excess "
            "kurtosis, producing a more realistic estimate when returns "
            "exhibit fat tails or asymmetry."
        )
        st.metric(
            f"Cornish-Fisher Adjusted VaR ({eff_confidence_label}, {eff_horizon}D)",
            f"${param['cf_var_dollar']:,.2f}",
            f"-{param['cf_var_pct']:.2%}",
            delta_color="inverse",
        )

    st.write("")

    # ---------------- Charts ----------------
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Asset Performance Tracking")
        st.plotly_chart(
            build_asset_performance_chart(market_data["normalized_prices"]),
            width="stretch",
        )
    with right_col:
        st.subheader("Risk Tail Distribution")
        st.plotly_chart(
            build_risk_tail_chart(
                portfolio["returns"], hist["var_pct"], eff_confidence_label, eff_horizon
            ),
            width="stretch",
        )

    st.write("")

    # ---------------- Tabs: Performance / Drawdown / Details ----------------
    tab1, tab2, tab3 = st.tabs(["Portfolio Performance", "Drawdown", "Portfolio Details"])

    with tab1:
        st.plotly_chart(build_cumulative_chart(portfolio["cumulative_returns"]), width="stretch")

    with tab2:
        st.plotly_chart(build_drawdown_chart(portfolio["drawdown"]), width="stretch")

    with tab3:
        detail_df = pd.DataFrame(
            {
                "Ticker": portfolio["tickers"],
                "Weight": [f"{w:.2%}" for w in portfolio["weights"]],
            }
        )
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.subheader("Holdings")
            st.dataframe(detail_df, width="stretch", hide_index=True)
        with col_b:
            st.subheader("Interpretation")
            st.info(
                f"With **{eff_confidence_label}** confidence, this portfolio is not "
                f"expected to lose more than **${hist['var_dollar']:,.2f}** "
                f"({hist['var_pct']:.2%}) over the next **{eff_horizon} day(s)** "
                f"under the Historical Simulation method, "
                f"**${param['var_dollar']:,.2f}** ({param['var_pct']:.2%}) under "
                f"the Parametric (Variance-Covariance) method, or "
                f"**${mc['var_dollar']:,.2f}** ({mc['var_pct']:.2%}) under the "
                f"Monte Carlo simulation method. If losses do exceed this VaR "
                f"threshold, the expected (average) loss in those worst-case "
                f"scenarios -- the Expected Shortfall / CVaR -- is "
                f"**${hist['cvar_dollar']:,.2f}** ({hist['cvar_pct']:.2%}) under "
                f"Historical Simulation, **${param['cvar_dollar']:,.2f}** "
                f"({param['cvar_pct']:.2%}) under the Parametric method, and "
                f"**${mc['cvar_dollar']:,.2f}** ({mc['cvar_pct']:.2%}) under "
                f"Monte Carlo simulation."
            )


else:
    st.info("Select tickers and configure weights in the sidebar, then click **Run Analysis** to begin.")

