"""
app.py
======

Streamlit frontend for the Portfolio Value at Risk (VaR) Dashboard —
a premium, professional-grade quantitative analytics terminal.

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

from var_engine import run_var_analysis, MIN_HORIZON_DAYS, MAX_HORIZON_DAYS

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Portfolio VaR Terminal",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Theme Constants
# ============================================================
SURFACE_BG = "#131C33"
BORDER_COLOR = "#22345E"
ACCENT = "#00E676"
ACCENT_AMBER = "#FFA600"
ACCENT_RED = "#FF4B4B"
TEXT_MUTED = "#8FA3C7"

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
# Premium Custom CSS Injection
# ============================================================
def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #0B1120;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {SURFACE_BG};
            border-right: 1px solid {BORDER_COLOR};
        }}

        /* Headline typography */
        h1, h2, h3 {{
            font-family: 'Helvetica Neue', sans-serif;
            letter-spacing: 0.3px;
        }}

        .terminal-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 18px 24px;
            background: linear-gradient(135deg, {SURFACE_BG} 0%, #0F1830 100%);
            border: 1px solid {BORDER_COLOR};
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .terminal-header h1 {{
            font-size: 1.6rem;
            margin: 0;
            color: #EAF1FF;
        }}
        .terminal-header .accent {{
            color: {ACCENT};
        }}
        .terminal-header p {{
            margin: 2px 0 0 0;
            color: {TEXT_MUTED};
            font-size: 0.85rem;
        }}
        .live-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border: 1px solid {ACCENT};
            border-radius: 999px;
            color: {ACCENT};
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        .live-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: {ACCENT};
            box-shadow: 0 0 6px {ACCENT};
        }}

        /* Metric / KPI cards */
        .kpi-card {{
            background-color: {SURFACE_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 10px;
            padding: 16px 18px;
            height: 100%;
        }}
        .kpi-label {{
            font-size: 0.75rem;
            color: {TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #EAF1FF;
            margin-bottom: 2px;
        }}
        .kpi-delta {{
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .kpi-delta.neg {{ color: {ACCENT_RED}; }}
        .kpi-delta.neutral {{ color: {TEXT_MUTED}; }}
        .kpi-accent-bar {{
            height: 3px;
            width: 36px;
            border-radius: 2px;
            margin-bottom: 10px;
        }}

        /* Section panels around charts */
        .panel-title {{
            font-size: 0.95rem;
            font-weight: 700;
            color: #EAF1FF;
            margin-bottom: 2px;
        }}
        .panel-subtitle {{
            font-size: 0.78rem;
            color: {TEXT_MUTED};
            margin-bottom: 10px;
        }}

        /* Divider */
        hr {{
            border-color: {BORDER_COLOR} !important;
        }}

        /* Dataframe / table styling */
        [data-testid="stDataFrame"] {{
            border: 1px solid {BORDER_COLOR};
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Helper Functions
# ============================================================
def parse_tickers(raw: str) -> List[str]:
    """Turn a comma-separated ticker string into a clean, de-duplicated list."""
    seen = []
    for t in raw.split(","):
        t_clean = t.strip().upper()
        if t_clean and t_clean not in seen:
            seen.append(t_clean)
    return seen


def parse_weights(raw: str, n_tickers: int) -> List[float]:
    """
    Parse a comma-separated weight string into floats. Supports both
    fractional (0.25) and percentage-style (25) entries — percentages are
    auto-normalized down to fractions if the values sum closer to 100.
    Gracefully handles blank input (equal-weighting fallback) and single
    asset portfolios.
    """
    if not raw.strip():
        return [round(1.0 / n_tickers, 6)] * n_tickers if n_tickers else []

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return [round(1.0 / n_tickers, 6)] * n_tickers if n_tickers else []

    values = [float(p) for p in parts]

    # If it looks like percentages (sums closer to 100 than to 1), convert.
    total = sum(values)
    if total > 1.5:  # heuristic: user typed e.g. 25, 25, 25, 25
        values = [v / 100.0 for v in values]

    return values


def kpi_card(label: str, value: str, delta: str, delta_class: str = "neg", accent_color: str = ACCENT) -> str:
    """Build a styled HTML KPI card block."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-accent-bar" style="background-color:{accent_color};"></div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta {delta_class}">{delta}</div>
    </div>
    """


def build_distribution_chart(
    portfolio_returns: pd.Series,
    var_pct_hist: float,
    var_pct_param: float,
    var_pct_mc: float,
    confidence_label: str,
    horizon_days: int,
) -> go.Figure:
    """Elegant Plotly histogram of historical portfolio returns with a
    prominent styled VaR cutoff line marking the multi-day loss threshold."""
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=portfolio_returns * 100,
            nbinsx=60,
            marker=dict(color="#4C78F5", line=dict(width=0.5, color="#0B1120")),
            opacity=0.85,
            name="Daily Portfolio Returns",
        )
    )

    # Primary, prominent cutoff: Historical VaR — the neon accent line.
    fig.add_vline(
        x=-var_pct_hist * 100,
        line_width=3,
        line_dash="dash",
        line_color=ACCENT,
        annotation_text=f"Historical VaR ({confidence_label}, {horizon_days}d): -{var_pct_hist:.2%}",
        annotation_position="top left",
        annotation_font=dict(color=ACCENT, size=12),
    )

    fig.add_vline(
        x=-var_pct_param * 100,
        line_width=2,
        line_dash="dot",
        line_color=ACCENT_AMBER,
        annotation_text=f"Parametric: -{var_pct_param:.2%}",
        annotation_position="top right",
        annotation_font=dict(color=ACCENT_AMBER, size=11),
    )

    fig.add_vline(
        x=-var_pct_mc * 100,
        line_width=2,
        line_dash="dot",
        line_color=ACCENT_RED,
        annotation_text=f"Monte Carlo: -{var_pct_mc:.2%}",
        annotation_position="bottom right",
        annotation_font=dict(color=ACCENT_RED, size=11),
    )

    fig.update_layout(
        title=None,
        xaxis_title="Return over horizon (%)",
        yaxis_title="Frequency",
        bargap=0.02,
        template="plotly_dark",
        height=430,
        margin=dict(t=30, b=40, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def build_asset_performance_chart(normalized_prices: pd.DataFrame) -> go.Figure:
    """
    High-fidelity multi-line Plotly chart tracking every individual asset's
    normalized price path (rebased to 100) over the lookback period, so
    relative performance can be visually compared regardless of each
    stock's absolute price level.
    """
    palette = [
        ACCENT, "#4C78F5", ACCENT_AMBER, ACCENT_RED,
        "#B388FF", "#00B8D9", "#FF7597", "#C6FF00",
    ]
    fig = go.Figure()
    for i, ticker in enumerate(normalized_prices.columns):
        fig.add_trace(
            go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[ticker].values,
                mode="lines",
                name=ticker,
                line=dict(color=palette[i % len(palette)], width=2),
            )
        )

    fig.add_hline(y=100, line_width=1, line_dash="dot", line_color=TEXT_MUTED)

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Normalized Value (Base = 100)",
        template="plotly_dark",
        height=430,
        margin=dict(t=30, b=40, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
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
            line=dict(color=ACCENT, width=2),
            fill="tozeroy",
            fillcolor="rgba(0,230,118,0.08)",
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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
            line=dict(color=ACCENT_RED, width=2),
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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ============================================================
# Apply Theme
# ============================================================
inject_custom_css()

# ============================================================
# Sidebar — User Inputs
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Portfolio Inputs")

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

    confidence_label = st.select_slider(
        "Confidence Level",
        options=CONFIDENCE_OPTIONS,
        value="95%",
        help="Statistical confidence level for the VaR estimate.",
    )
    confidence_level = CONFIDENCE_MAP[confidence_label]

    horizon_days = st.slider(
        "Time Horizon (days)",
        min_value=MIN_HORIZON_DAYS,
        max_value=MAX_HORIZON_DAYS,
        value=1,
        step=1,
        help="Multi-day risk horizon, up to 100 days.",
    )

    period_label = st.selectbox(
        "Historical Lookback Period", options=list(PERIOD_OPTIONS.keys()), index=3
    )
    period = PERIOD_OPTIONS[period_label]

    st.divider()
    run_clicked = st.button("🚀 Run VaR Analysis", use_container_width=True, type="primary")


# ============================================================
# Main Header
# ============================================================
st.markdown(
    f"""
    <div class="terminal-header">
        <div>
            <h1>📉 Portfolio <span class="accent">VaR</span> Terminal</h1>
            <p>Historical · Parametric · Monte Carlo risk analytics, powered by live market data</p>
        </div>
        <div class="live-pill"><span class="live-dot"></span> LIVE MARKET DATA</div>
    </div>
    """,
    unsafe_allow_html=True,
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
        status_box = st.status("Initializing risk engine...", expanded=True)

        def _on_status(msg: str) -> None:
            status_box.write(msg)

        result = run_var_analysis(
            tickers=tickers,
            weights=weights,
            portfolio_value=portfolio_value,
            confidence_level=confidence_level,
            horizon_days=horizon_days,
            period=period,
            status_callback=_on_status,
        )

        if result["success"]:
            status_box.update(label="Analysis complete.", state="complete", expanded=False)
        else:
            status_box.update(label="Analysis failed.", state="error", expanded=True)

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
    effective_horizon = result.get("horizon_days", horizon_days)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            kpi_card(
                f"Historical VaR ({confidence_label}, {effective_horizon}d)",
                f"${hist['var_dollar']:,.2f}",
                f"-{hist['var_pct']:.2%}",
                "neg",
                ACCENT,
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            kpi_card(
                f"Parametric VaR ({confidence_label}, {effective_horizon}d)",
                f"${param['var_dollar']:,.2f}",
                f"-{param['var_pct']:.2%}",
                "neg",
                ACCENT_AMBER,
            ),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            kpi_card(
                f"Monte Carlo VaR ({confidence_label}, {effective_horizon}d)",
                f"${mc['var_dollar']:,.2f}",
                f"-{mc['var_pct']:.2%}",
                "neg",
                ACCENT_RED,
            ),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            kpi_card(
                "Portfolio Value",
                f"${portfolio_value:,.2f}",
                "Base capital",
                "neutral",
                "#4C78F5",
            ),
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            kpi_card(
                "Observations Used",
                f"{result['n_observations']:,}",
                f"Lookback: {period_label}",
                "neutral",
                "#B388FF",
            ),
            unsafe_allow_html=True,
        )

    st.write("")
    st.divider()

    # ---------------- Side-by-Side Grid: Distribution | Asset Performance ----------------
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown('<div class="panel-title">📊 Volatility Distribution</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-subtitle">Portfolio return distribution with VaR cutoff thresholds</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_distribution_chart(
                result["portfolio_returns"],
                hist["var_pct"],
                param["var_pct"],
                mc["var_pct"],
                confidence_label,
                effective_horizon,
            ),
            use_container_width=True,
        )

    with right_col:
        st.markdown('<div class="panel-title">📈 Asset Performance Tracking</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-subtitle">Normalized price paths (base = 100) across all holdings</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_asset_performance_chart(result["normalized_prices"]),
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
                f"({hist['var_pct']:.2%}) over the next **{effective_horizon} day(s)** "
                f"under the Historical Simulation method, "
                f"**${param['var_dollar']:,.2f}** ({param['var_pct']:.2%}) under "
                f"the Parametric (Variance-Covariance) method, or "
                f"**${mc['var_dollar']:,.2f}** ({mc['var_pct']:.2%}) under the "
                f"Monte Carlo simulation method."
            )

else:
    st.info("👈 Configure your portfolio in the sidebar and click **Run VaR Analysis** to begin.")
