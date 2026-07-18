"""
app.py
======

Finfy — Portfolio Volatility Analytics.

A premium, institutional-grade quantitative risk terminal built on
Streamlit. All statistical computation and data shaping is delegated
entirely to `var_engine.py` (Finfy Core); this module is purely
presentational — it renders the unified payload it receives and performs
zero data manipulation of its own.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from var_engine import run_var_analysis, MIN_HORIZON_DAYS, MAX_HORIZON_DAYS

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Finfy // Portfolio Volatility Analytics",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Brand & Theme Tokens
# ============================================================
BG_DEEP = "#0A0E17"
SURFACE = "rgba(19, 26, 46, 0.55)"          # glass surface (translucent)
SURFACE_SOLID = "#131A2E"
BORDER = "rgba(255, 255, 255, 0.08)"
BORDER_STRONG = "#26314F"

ACCENT_GREEN = "#39FFB0"     # vibrant neon green — positive / brand accent
ACCENT_RED = "#E8746B"       # soft desaturated red — VaR / risk metrics
ACCENT_AMBER = "#E8B76B"
ACCENT_BLUE = "#6E8CFF"
TEXT_PRIMARY = "#EDF1FB"
TEXT_MUTED = "#7C88A6"

MONO_FONT = "'JetBrains Mono', 'SFMono-Regular', 'Roboto Mono', Consolas, monospace"
SANS_FONT = "'Inter', 'Helvetica Neue', sans-serif"

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
# Custom CSS — Glassmorphism / Typography Matrix
# ============================================================
def inject_finfy_theme() -> None:
    """
    Inject scoped CSS for the Finfy brand: glass-panel cards, monospace
    numerals/tickers, layered shadows. Selectors are scoped to custom
    Finfy-prefixed classes (plus a small number of well-known Streamlit
    test-ids) so native Streamlit layout/rendering is never broken.
    """
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

        .stApp {{
            background: radial-gradient(circle at 15% 0%, #101833 0%, {BG_DEEP} 55%);
        }}

        html, body, [class*="css"] {{
            font-family: {SANS_FONT};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {SURFACE_SOLID};
            border-right: 1px solid {BORDER_STRONG};
        }}
        section[data-testid="stSidebar"] * {{
            font-family: {SANS_FONT};
        }}

        h1, h2, h3 {{
            color: {TEXT_PRIMARY};
            letter-spacing: 0.2px;
        }}

        /* ---------------- Finfy Header ---------------- */
        .finfy-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 22px 28px;
            background: linear-gradient(135deg, rgba(19,26,46,0.85) 0%, rgba(14,19,36,0.85) 100%);
            border: 1px solid {BORDER_STRONG};
            border-radius: 14px;
            margin-bottom: 22px;
            box-shadow: 0 12px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03);
            backdrop-filter: blur(8px);
        }}
        .finfy-wordmark {{
            font-family: {MONO_FONT};
            font-size: 1.9rem;
            font-weight: 700;
            color: {TEXT_PRIMARY};
            margin: 0;
            letter-spacing: -0.5px;
        }}
        .finfy-wordmark span {{
            color: {ACCENT_GREEN};
        }}
        .finfy-subline {{
            font-family: {MONO_FONT};
            font-size: 0.78rem;
            color: {TEXT_MUTED};
            margin-top: 4px;
            letter-spacing: 0.4px;
            text-transform: uppercase;
        }}
        .finfy-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border: 1px solid {ACCENT_GREEN};
            border-radius: 999px;
            color: {ACCENT_GREEN};
            font-family: {MONO_FONT};
            font-size: 0.72rem;
            font-weight: 500;
            letter-spacing: 0.6px;
            background: rgba(57, 255, 176, 0.06);
        }}
        .finfy-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: {ACCENT_GREEN};
            box-shadow: 0 0 8px {ACCENT_GREEN};
            animation: finfy-pulse 2.2s ease-in-out infinite;
        }}
        @keyframes finfy-pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.35; }}
        }}

        /* ---------------- Glass Metric Cards ---------------- */
        .finfy-card {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 18px 20px;
            height: 100%;
            box-shadow: 0 8px 24px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.04);
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }}
        .finfy-card::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: var(--accent-color, {ACCENT_GREEN});
            opacity: 0.85;
        }}
        .finfy-card-label {{
            font-family: {MONO_FONT};
            font-size: 0.68rem;
            color: {TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 10px;
        }}
        .finfy-card-value {{
            font-family: {MONO_FONT};
            font-size: 1.55rem;
            font-weight: 700;
            color: {TEXT_PRIMARY};
            margin-bottom: 4px;
            line-height: 1.15;
        }}
        .finfy-card-delta {{
            font-family: {MONO_FONT};
            font-size: 0.82rem;
            font-weight: 500;
        }}
        .finfy-card-delta.risk {{ color: {ACCENT_RED}; }}
        .finfy-card-delta.positive {{ color: {ACCENT_GREEN}; }}
        .finfy-card-delta.neutral {{ color: {TEXT_MUTED}; }}

        /* ---------------- Section / Panel Headers ---------------- */
        .finfy-panel-head {{
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            margin-bottom: 6px;
        }}
        .finfy-panel-title {{
            font-family: {SANS_FONT};
            font-size: 1.0rem;
            font-weight: 700;
            color: {TEXT_PRIMARY};
        }}
        .finfy-panel-tag {{
            font-family: {MONO_FONT};
            font-size: 0.68rem;
            color: {TEXT_MUTED};
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}
        .finfy-panel-sub {{
            font-family: {SANS_FONT};
            font-size: 0.8rem;
            color: {TEXT_MUTED};
            margin-bottom: 12px;
        }}
        .finfy-panel-wrap {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 18px 18px 6px 18px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.04);
            backdrop-filter: blur(10px);
            margin-bottom: 8px;
        }}

        /* ---------------- Ticker chips (monospace) ---------------- */
        .finfy-chip {{
            display: inline-block;
            font-family: {MONO_FONT};
            font-size: 0.75rem;
            padding: 3px 9px;
            border-radius: 6px;
            border: 1px solid {BORDER_STRONG};
            color: {TEXT_PRIMARY};
            background: rgba(255,255,255,0.03);
            margin: 2px 4px 2px 0;
        }}

        hr {{ border-color: {BORDER_STRONG} !important; }}

        [data-testid="stDataFrame"] {{
            border: 1px solid {BORDER_STRONG};
            border-radius: 10px;
            font-family: {MONO_FONT};
        }}

        div[data-testid="stStatusWidget"] {{
            font-family: {MONO_FONT};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Input Helpers
# ============================================================
def parse_tickers(raw: str) -> List[str]:
    """Turn a comma-separated ticker string into a clean, de-duplicated list."""
    seen: List[str] = []
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

    total = sum(values)
    if total > 1.5:  # heuristic: user typed e.g. 25, 25, 25, 25
        values = [v / 100.0 for v in values]

    return values


def render_kpi_card(label: str, value: str, delta: str, tone: str, accent_color: str) -> None:
    """Render a single glass-morphic KPI card with monospace numerals."""
    st.markdown(
        f"""
        <div class="finfy-card" style="--accent-color:{accent_color};">
            <div class="finfy-card-label">{label}</div>
            <div class="finfy-card-value">{value}</div>
            <div class="finfy-card-delta {tone}">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Chart Builders
# ============================================================
def build_asset_performance_chart(normalized_prices: pd.DataFrame) -> go.Figure:
    """
    Dominant-canvas multi-line chart tracking every individual asset's
    normalized price path (rebased to 100) over the lookback period.
    """
    palette = [
        ACCENT_GREEN, ACCENT_BLUE, ACCENT_AMBER, ACCENT_RED,
        "#B388FF", "#4DD0E1", "#FF8FB3", "#C6FF6B",
    ]
    fig = go.Figure()
    for i, ticker in enumerate(normalized_prices.columns):
        fig.add_trace(
            go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[ticker].values,
                mode="lines",
                name=ticker,
                line=dict(color=palette[i % len(palette)], width=2.2),
                hovertemplate=f"<b>{ticker}</b><br>%{{x|%Y-%m-%d}}<br>Index: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.add_hline(y=100, line_width=1, line_dash="dot", line_color=TEXT_MUTED)

    fig.update_layout(
        template="plotly_dark",
        height=460,
        margin=dict(t=10, b=40, l=44, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO_FONT, color=TEXT_PRIMARY, size=11),
        xaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Normalized Index (Base = 100)", gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
        hovermode="x unified",
    )
    return fig


def build_risk_tail_chart(
    portfolio_returns: pd.Series,
    var_pct: float,
    confidence_label: str,
    horizon_days: int,
) -> go.Figure:
    """
    Risk Tail Distribution Canvas: the empirical returns distribution curve
    with clean transparency fill and a single prominent vertical line
    marking the downside risk boundary (Historical VaR cutoff).
    """
    returns_pct = portfolio_returns * 100

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=returns_pct,
            nbinsx=55,
            marker=dict(color=ACCENT_BLUE, line=dict(width=0)),
            opacity=0.35,
            name="Return Distribution",
            histnorm="probability density",
        )
    )

    # KDE-like smoothed curve via a simple density estimate for a clean
    # tail silhouette on top of the histogram.
    try:
        from scipy.stats import gaussian_kde
        import numpy as np

        kde = gaussian_kde(returns_pct.dropna())
        x_grid = np.linspace(returns_pct.min(), returns_pct.max(), 300)
        y_grid = kde(x_grid)
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_grid,
                mode="lines",
                line=dict(color=ACCENT_BLUE, width=2),
                fill="tozeroy",
                fillcolor="rgba(110, 140, 255, 0.12)",
                name="Density",
            )
        )
    except Exception:
        pass

    # Single, prominent downside risk boundary line.
    fig.add_vline(
        x=-var_pct * 100,
        line_width=3,
        line_dash="dash",
        line_color=ACCENT_RED,
        annotation_text=f"VaR Boundary ({confidence_label}, {horizon_days}d): -{var_pct:.2%}",
        annotation_position="top left",
        annotation_font=dict(color=ACCENT_RED, size=11, family=MONO_FONT),
    )

    fig.update_layout(
        template="plotly_dark",
        height=460,
        margin=dict(t=10, b=40, l=44, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO_FONT, color=TEXT_PRIMARY, size=11),
        xaxis=dict(title="Return over Horizon (%)", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Density", gridcolor="rgba(255,255,255,0.05)"),
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
            line=dict(color=ACCENT_GREEN, width=2),
            fill="tozeroy",
            fillcolor="rgba(57,255,176,0.08)",
            name="Cumulative Growth ($1 invested)",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(t=20, b=40, l=44, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO_FONT, color=TEXT_PRIMARY, size=11),
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
            line=dict(color=ACCENT_RED, width=2),
            fill="tozeroy",
            fillcolor="rgba(232,116,107,0.15)",
            name="Drawdown (%)",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(t=20, b=40, l=44, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO_FONT, color=TEXT_PRIMARY, size=11),
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
    )
    return fig


# ============================================================
# Apply Theme
# ============================================================
inject_finfy_theme()

# ============================================================
# Sidebar — User Inputs
# ============================================================
with st.sidebar:
    st.markdown(
        f"<div style='font-family:{MONO_FONT}; font-weight:700; font-size:1.05rem; "
        f"color:{TEXT_PRIMARY}; margin-bottom:2px;'>FINFY<span style='color:{ACCENT_GREEN};'>.</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-family:{MONO_FONT}; font-size:0.7rem; color:{TEXT_MUTED}; "
        f"text-transform:uppercase; letter-spacing:0.5px; margin-bottom:16px;'>Portfolio Configuration</div>",
        unsafe_allow_html=True,
    )

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
    run_clicked = st.button("Run Analysis", use_container_width=True, type="primary")


# ============================================================
# Finfy Header
# ============================================================
st.markdown(
    f"""
    <div class="finfy-header">
        <div>
            <div class="finfy-wordmark">Finfy<span>.</span> Engine</div>
            <div class="finfy-subline">Portfolio Volatility Analytics // Historical · Parametric · Monte Carlo</div>
        </div>
        <div class="finfy-pill"><span class="finfy-dot"></span> LIVE MARKET FEED</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if run_clicked or "last_result" in st.session_state:
    try:
        weights = parse_weights(weights_raw, len(tickers))
    except ValueError:
        st.error("Could not parse weights. Please enter numeric values separated by commas.")
        st.stop()

    if run_clicked:
        status_box = st.status("Finfy Core: Initializing risk engine...", expanded=True)

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
            status_box.update(label="Finfy Core: Analysis complete.", state="complete", expanded=False)
        else:
            status_box.update(label="Finfy Core: Analysis failed.", state="error", expanded=True)

        st.session_state["last_result"] = result
    else:
        result = st.session_state["last_result"]

    if not result["success"]:
        st.error(result["error"])
        st.stop()

    # Unpack the unified, nested payload — zero data manipulation here.
    market_data = result["market_data"]
    portfolio = result["portfolio"]
    risk_metrics = result["risk_metrics"]

    hist = risk_metrics["historical"]
    param = risk_metrics["parametric"]
    mc = risk_metrics["monte_carlo"]

    eff_horizon = portfolio["horizon_days"]
    eff_confidence_label = confidence_label

    # ---------------- Glass KPI Grid ----------------
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card(
            f"Historical VaR · {eff_confidence_label} · {eff_horizon}D",
            f"${hist['var_dollar']:,.2f}",
            f"-{hist['var_pct']:.2%}",
            "risk",
            ACCENT_RED,
        )
    with col2:
        render_kpi_card(
            f"Parametric VaR · {eff_confidence_label} · {eff_horizon}D",
            f"${param['var_dollar']:,.2f}",
            f"-{param['var_pct']:.2%}",
            "risk",
            ACCENT_AMBER,
        )
    with col3:
        render_kpi_card(
            f"Monte Carlo VaR · {eff_confidence_label} · {eff_horizon}D",
            f"${mc['var_dollar']:,.2f}",
            f"-{mc['var_pct']:.2%}",
            "risk",
            ACCENT_BLUE,
        )
    with col4:
        render_kpi_card(
            "Portfolio Value",
            f"${portfolio['portfolio_value']:,.2f}",
            "Base Capital",
            "positive",
            ACCENT_GREEN,
        )
    with col5:
        render_kpi_card(
            "Observations",
            f"{portfolio['n_observations']:,}",
            f"Lookback: {period_label}",
            "neutral",
            TEXT_MUTED,
        )

    st.write("")

    # ---------------- Asymmetric Grid: Performance (dominant) | Risk Tail ----------------
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown('<div class="finfy-panel-wrap">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="finfy-panel-head">
                <div class="finfy-panel-title">Asset Performance Tracking</div>
                <div class="finfy-panel-tag">Normalized · Base 100</div>
            </div>
            <div class="finfy-panel-sub">Individual holding trajectories over the selected lookback window</div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_asset_performance_chart(market_data["normalized_prices"]),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="finfy-panel-wrap">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="finfy-panel-head">
                <div class="finfy-panel-title">Risk Tail Distribution</div>
                <div class="finfy-panel-tag">Empirical · VaR Cutoff</div>
            </div>
            <div class="finfy-panel-sub">Portfolio return density with downside risk boundary</div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_risk_tail_chart(
                portfolio["returns"], hist["var_pct"], eff_confidence_label, eff_horizon
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # ---------------- Tabs: Performance / Drawdown / Details ----------------
    tab1, tab2, tab3 = st.tabs(
        ["Portfolio Performance", "Drawdown", "Portfolio Details"]
    )

    with tab1:
        st.plotly_chart(
            build_cumulative_chart(portfolio["cumulative_returns"]),
            use_container_width=True,
        )

    with tab2:
        st.plotly_chart(
            build_drawdown_chart(portfolio["drawdown"]), use_container_width=True
        )

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
            chips = "".join(f'<span class="finfy-chip">{t}</span>' for t in portfolio["tickers"])
            st.markdown(chips, unsafe_allow_html=True)
            st.write("")
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
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
                f"Monte Carlo simulation method."
            )

else:
    st.info("Configure your portfolio in the sidebar and click **Run Analysis** to begin.")
