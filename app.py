"""
app.py
======

Streamlit frontend for the Portfolio Value at Risk (VaR) Dashboard.

Run locally with:
    streamlit run app.py

All heavy-lifting (data download, validation, statistics) is delegated to
`var_engine.py` — this file is purely responsible for layout & visualization.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from var_engine import run_var_analysis

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Portfolio VaR Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIDENCE_OPTIONS = {"90%": 0.90, "95%": 0.95, "99%": 0.99}
PERIOD_OPTIONS = {
    "1 Year": "1y",
    "2 Years": "2y",
    "3 Years": "3y",
    "5 Years": "5y",
    "10 Years": "10y",
}


# ============================================================
# Helper Functions
# ============================================================
def parse_tickers(raw: str) -> List[str]:
    """Turn a comma-separated ticker string into a clean list."""
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def parse_weights(raw: str, n_tickers: int) -> List[float]:
    """
    Parse a comma-separated weight string into floats. Supports both
    fractional (0.25) and percentage-style (25) entries — percentages are
    auto-normalized down to fractions if the values sum closer to 100.
    """
    if not raw.strip():
        # Default to equal weighting if the user leaves this blank.
        return [round(1.0 / n_tickers, 6)] * n_tickers if n_tickers else []

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    values = [float(p) for p in parts]

    # If it looks like percentages (sums closer to 100 than to 1), convert.
    total = sum(values)
    if total > 1.5:  # heuristic: user typed e.g. 25, 25, 25, 25
        values = [v / 100.0 for v in values]

    return values


def build_distribution_chart(
    portfolio_returns: pd.Series,
    var_pct_hist: float,
    var_pct_param: float,
    var_pct_mc: float,
) -> go.Figure:

    """Interactive histogram of historical portfolio returns with VaR markers."""
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=portfolio_returns * 100,
            nbinsx=60,
            marker=dict(color="#4C78F5", line=dict(width=0.5, color="white")),
            opacity=0.85,
            name="Daily Portfolio Returns",
        )
    )

    # Historical VaR threshold — dashed bright vertical line.
    fig.add_vline(
        x=-var_pct_hist * 100,
        line_width=3,
        line_dash="dash",
        line_color="#FF4B4B",
        annotation_text=f"Historical VaR: -{var_pct_hist:.2%}",
        annotation_position="top left",
        annotation_font=dict(color="#FF4B4B", size=13),
    )

    # Parametric VaR threshold — second dashed vertical line for comparison.
    fig.add_vline(
        x=-var_pct_param * 100,
        line_width=3,
        line_dash="dash",
        line_color="#FFA600",
        annotation_text=f"Parametric VaR: -{var_pct_param:.2%}",
        annotation_position="top right",
        annotation_font=dict(color="#FFA600", size=13),
    )

    # Monte Carlo VaR threshold — third dashed vertical line for comparison.
    fig.add_vline(
        x=-var_pct_mc * 100,
        line_width=3,
        line_dash="dash",
        line_color="#00E5A0",
        annotation_text=f"Monte Carlo VaR: -{var_pct_mc:.2%}",
        annotation_position="bottom right",
        annotation_font=dict(color="#00E5A0", size=13),
    )

    fig.update_layout(

        title="Distribution of Historical Portfolio Returns",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        bargap=0.02,
        template="plotly_dark",
        height=460,
        margin=dict(t=60, b=40, l=40, r=40),
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
            line=dict(color="#00CC96", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,204,150,0.1)",
            name="Cumulative Growth ($1 invested)",
        )
    )
    fig.update_layout(
        title="Cumulative Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        template="plotly_dark",
        height=420,
        margin=dict(t=60, b=40, l=40, r=40),
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
            line=dict(color="#EF553B", width=2),
            fill="tozeroy",
            fillcolor="rgba(239,85,59,0.15)",
            name="Drawdown (%)",
        )
    )
    fig.update_layout(
        title="Historical Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        height=420,
        margin=dict(t=60, b=40, l=40, r=40),
    )
    return fig


# ============================================================
# Sidebar — User Inputs
# ============================================================
with st.sidebar:
    st.title("⚙️ Portfolio Inputs")

    tickers_raw = st.text_input(
        "Stock Tickers (comma-separated)",
        value="AAPL, MSFT, GOOG, AMZN",
        help="Enter valid Yahoo Finance ticker symbols separated by commas.",
    )
    tickers = parse_tickers(tickers_raw)

    weights_raw = st.text_input(
        "Portfolio Weights (comma-separated, must sum to 1.0 or 100)",
        value=", ".join(["0.25"] * len(tickers)) if tickers else "",
        help="Leave blank for equal weighting. Accepts fractions (0.25) or percentages (25).",
    )

    st.divider()

    portfolio_value = st.number_input(
        "Initial Investment ($)",
        min_value=100.0,
        value=10_000.0,
        step=500.0,
        format="%.2f",
    )

    confidence_label = st.selectbox(
        "Confidence Level", options=list(CONFIDENCE_OPTIONS.keys()), index=1
    )
    confidence_level = CONFIDENCE_OPTIONS[confidence_label]

    horizon_days = st.slider(
        "Time Horizon (days)", min_value=1, max_value=20, value=1, step=1
    )

    period_label = st.selectbox(
        "Historical Lookback Period", options=list(PERIOD_OPTIONS.keys()), index=3
    )
    period = PERIOD_OPTIONS[period_label]

    st.divider()
    run_clicked = st.button("🚀 Run VaR Analysis", use_container_width=True, type="primary")


# ============================================================
# Main Dashboard
# ============================================================
st.title("📉 Portfolio Value at Risk (VaR) Dashboard")
st.caption(
    "Analyze downside risk for any custom stock portfolio using live market "
    "data, Historical Simulation, and Parametric (Variance-Covariance) VaR methods."
)

if run_clicked or "last_result" in st.session_state:
    # Parse weights fresh each run so edits are always respected.
    try:
        weights = parse_weights(weights_raw, len(tickers))
    except ValueError:
        st.error(
            "⚠️ Could not parse weights. Please enter numeric values separated by commas."
        )
        st.stop()

    if run_clicked:
        with st.spinner("Fetching live market data and computing VaR..."):
            result = run_var_analysis(
                tickers=tickers,
                weights=weights,
                portfolio_value=portfolio_value,
                confidence_level=confidence_level,
                horizon_days=horizon_days,
                period=period,
            )
        st.session_state["last_result"] = result
    else:
        result = st.session_state["last_result"]

    if not result["success"]:
        st.error(f"⚠️ {result['error']}")
        st.stop()

    # ---------------- KPI Cards ----------------
    hist = result["historical"]
    param = result["parametric"]
    mc = result["monte_carlo"]

    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(
            label=f"Historical VaR ({confidence_label}, {horizon_days}d)",
            value=f"${hist['var_dollar']:,.2f}",
            delta=f"-{hist['var_pct']:.2%}",
            delta_color="inverse",
        )
        col2.metric(
            label=f"Parametric VaR ({confidence_label}, {horizon_days}d)",
            value=f"${param['var_dollar']:,.2f}",
            delta=f"-{param['var_pct']:.2%}",
            delta_color="inverse",
        )
        col3.metric(
            label=f"Monte Carlo VaR ({confidence_label}, {horizon_days}d)",
            value=f"${mc['var_dollar']:,.2f}",
            delta=f"-{mc['var_pct']:.2%}",
            delta_color="inverse",
        )
        col4.metric(
            label="Portfolio Value",
            value=f"${portfolio_value:,.2f}",
        )
        col5.metric(
            label="Observations Used",
            value=f"{result['n_observations']:,}",
        )

    st.divider()

    # ---------------- Distribution Chart ----------------
    st.plotly_chart(
        build_distribution_chart(
            result["portfolio_returns"], hist["var_pct"], param["var_pct"], mc["var_pct"]
        ),
        use_container_width=True,
    )


    st.divider()

    # ---------------- Tabs: Performance / Drawdown / Details ----------------
    tab1, tab2, tab3 = st.tabs(
        ["📈 Portfolio Performance", "📉 Drawdown", "📋 Portfolio Details"]
    )

    with tab1:
        st.plotly_chart(
            build_cumulative_chart(result["cumulative_returns"]),
            use_container_width=True,
        )

    with tab2:
        st.plotly_chart(
            build_drawdown_chart(result["drawdown"]), use_container_width=True
        )

    with tab3:
        detail_df = pd.DataFrame(
            {
                "Ticker": result["tickers"],
                "Weight": [f"{w:.2%}" for w in result["weights"]],
            }
        )
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.subheader("Holdings")
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
        with col_b:
            st.subheader("Interpretation")
            st.info(
                f"With **{confidence_label}** confidence, this portfolio is not "
                f"expected to lose more than **${hist['var_dollar']:,.2f}** "
                f"({hist['var_pct']:.2%}) over the next **{horizon_days} day(s)** "
                f"under the Historical Simulation method, "
                f"**${param['var_dollar']:,.2f}** ({param['var_pct']:.2%}) under "
                f"the Parametric (Variance-Covariance) method, or "
                f"**${mc['var_dollar']:,.2f}** ({mc['var_pct']:.2%}) under the "
                f"Monte Carlo simulation method."
            )


else:
    st.info("👈 Configure your portfolio in the sidebar and click **Run VaR Analysis** to begin.")
