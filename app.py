"""
app.py
======

Finfy — Portfolio Volatility Analytics.

A premium, institutional-grade quantitative risk terminal built on
Streamlit. All statistical computation and data shaping is delegated
entirely to `var_engine.py` (Finfy Core); this module is purely
presentational — it renders the unified payload it receives and performs
zero data manipulation of its own.

Frictionless UX notes:
    - Ticker selection uses `st.multiselect` (typeahead / autosuggest)
      against a pre-populated institutional asset registry, replacing the
      old fragile comma-separated free-text ticker/weight inputs.
    - Each selected ticker gets a per-asset weight number input plus a
      dynamically resolved corporate logo avatar (Clearbit logo CDN,
      keyed off a ticker->domain map) with a clean monogram fallback.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


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
BG_DEEP = "#0A0F1D"
SURFACE = "rgba(19, 28, 51, 0.55)"          # glass surface (translucent)
SURFACE_SOLID = "#131C33"
BORDER = "rgba(255, 255, 255, 0.08)"
BORDER_STRONG = "#26314F"

ACCENT_GREEN = "#00E676"     # vibrant neon green — positive / brand accent
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

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOG"]

# ============================================================
# Institutional Asset Registry (typeahead universe)
# ============================================================
# A representative, high-volume institutional universe: mega/large-cap
# equities across sectors, dominant broad-market & sector ETFs, and a
# crypto anchor. Not the full S&P 500 constituent list, but broad enough
# to drive a realistic, responsive typeahead search experience.
ASSET_REGISTRY: List[str] = sorted(set([
    # Mega-cap Technology
    "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO",
    "ORCL", "ADBE", "CRM", "AMD", "INTC", "CSCO", "IBM", "QCOM", "TXN",
    "NOW", "INTU", "UBER", "SHOP", "PANW", "SNOW", "PLTR", "NET", "CRWD",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "V", "MA",
    "PYPL",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY",
    # Consumer
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "HD", "LOW",
    "TGT", "DIS", "NFLX",
    # Industrials & Energy
    "XOM", "CVX", "CAT", "BA", "GE", "HON", "UPS", "RTX", "LMT", "DE",
    # Broad Market & Sector ETFs
    "SPY", "VOO", "QQQ", "DIA", "IWM", "VTI", "ARKK", "XLF", "XLE", "XLK",
    "XLV", "GLD", "SLV", "TLT", "HYG",
    # Crypto Anchors
    "BTC-USD", "ETH-USD", "SOL-USD",
]))

# Ticker -> corporate domain map, used to resolve logo avatars via the
# Clearbit Logo API (https://logo.clearbit.com/<domain>). Not every ticker
# in the registry needs an entry — anything missing falls back to a clean
# monogram avatar automatically.
TICKER_DOMAINS: Dict[str, str] = {
    "AAPL": "apple.com", "MSFT": "microsoft.com", "GOOG": "google.com",
    "GOOGL": "google.com", "AMZN": "amazon.com", "META": "meta.com",
    "NVDA": "nvidia.com", "TSLA": "tesla.com", "AVGO": "broadcom.com",
    "ORCL": "oracle.com", "ADBE": "adobe.com", "CRM": "salesforce.com",
    "AMD": "amd.com", "INTC": "intel.com", "CSCO": "cisco.com",
    "IBM": "ibm.com", "QCOM": "qualcomm.com", "TXN": "ti.com",
    "NOW": "servicenow.com", "INTU": "intuit.com", "UBER": "uber.com",
    "SHOP": "shopify.com", "PANW": "paloaltonetworks.com",
    "SNOW": "snowflake.com", "PLTR": "palantir.com", "NET": "cloudflare.com",
    "CRWD": "crowdstrike.com",
    "JPM": "jpmorganchase.com", "BAC": "bankofamerica.com",
    "WFC": "wellsfargo.com", "GS": "goldmansachs.com", "MS": "morganstanley.com",
    "C": "citigroup.com", "AXP": "americanexpress.com", "BLK": "blackrock.com",
    "SCHW": "schwab.com", "V": "visa.com", "MA": "mastercard.com",
    "PYPL": "paypal.com",
    "UNH": "unitedhealthgroup.com", "JNJ": "jnj.com", "LLY": "lilly.com",
    "PFE": "pfizer.com", "MRK": "merck.com", "ABBV": "abbvie.com",
    "TMO": "thermofisher.com", "ABT": "abbott.com", "DHR": "danaher.com",
    "BMY": "bms.com",
    "WMT": "walmart.com", "COST": "costco.com", "PG": "pg.com",
    "KO": "coca-cola.com", "PEP": "pepsico.com", "MCD": "mcdonalds.com",
    "NKE": "nike.com", "SBUX": "starbucks.com", "HD": "homedepot.com",
    "LOW": "lowes.com", "TGT": "target.com", "DIS": "disney.com",
    "NFLX": "netflix.com",
    "XOM": "exxonmobil.com", "CVX": "chevron.com", "CAT": "caterpillar.com",
    "BA": "boeing.com", "GE": "ge.com", "HON": "honeywell.com",
    "UPS": "ups.com", "RTX": "rtx.com", "LMT": "lockheedmartin.com",
    "DE": "deere.com",
}

CLEARBIT_LOGO_URL = "https://logo.clearbit.com/{domain}"


# ============================================================
# Custom CSS — Glassmorphism / Typography Matrix
# ============================================================
def inject_finfy_theme() -> None:
    """
    Inject scoped CSS for the Finfy brand: glass-panel cards, monospace
    numerals/tickers, layered shadows, and logo-avatar chips. Selectors
    are scoped to custom Finfy-prefixed classes so native Streamlit
    layout/rendering is never broken.
    """
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

        .stApp {{
            background: radial-gradient(circle at 15% 0%, #111c3a 0%, {BG_DEEP} 55%);
        }}

        /* NOTE: intentionally scoped -- do NOT use a blanket [class*="css"]
           wildcard selector here. Streamlit's own internal icon elements
           (e.g. the sidebar collapse/expand control) rely on Material
           Symbols icon-font ligatures like "keyboard_double_arrow_right"
           to render as glyphs. A wildcard font-family override strips
           that icon font and causes the raw ligature text to leak onto
           the page as literal text. Scope to real text-bearing containers
           only, and explicitly exclude icon containers. */
        html, body, .stApp, .stApp p, .stApp span, .stApp div, .stApp label,
        .stApp li, .stMarkdown, [data-testid="stMarkdownContainer"] {{
            font-family: {SANS_FONT};
        }}
        [data-testid="stIconMaterial"],
        [data-testid="collapsedControl"],
        [data-testid="collapsedControl"] * {{
            font-family: unset !important;
        }}

        section[data-testid="stSidebar"] {{
            background-color: {SURFACE_SOLID};
            border-right: 1px solid {BORDER_STRONG};
        }}
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] li {{
            font-family: {SANS_FONT};
        }}
        section[data-testid="stSidebar"] [data-testid="stIconMaterial"] {{
            font-family: unset !important;
        }}

        /* Safety net: guarantee the sidebar collapse/expand control never
           leaks raw icon-ligature text, while keeping the icon glyph
           itself fully visible and functional. */
        [data-testid="collapsedControl"] {{
            color: transparent !important;
        }}
        [data-testid="collapsedControl"] svg,
        [data-testid="collapsedControl"] [data-testid="stIconMaterial"] {{
            color: {TEXT_PRIMARY} !important;
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
            background: linear-gradient(135deg, rgba(19,28,51,0.85) 0%, rgba(12,17,32,0.85) 100%);
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
        .finfy-wordmark span {{ color: {ACCENT_GREEN}; }}
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
            background: rgba(0, 230, 118, 0.06);
        }}
        .finfy-dot {{
            width: 7px; height: 7px; border-radius: 50%;
            background: {ACCENT_GREEN};
            box-shadow: 0 0 8px {ACCENT_GREEN};
            animation: finfy-pulse 2.2s ease-in-out infinite;
        }}
        @keyframes finfy-pulse {{ 0%,100% {{opacity:1;}} 50% {{opacity:0.35;}} }}

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
            content: ""; position: absolute; top:0; left:0; right:0; height:2px;
            background: var(--accent-color, {ACCENT_GREEN}); opacity: 0.85;
        }}
        .finfy-card-label {{
            font-family: {MONO_FONT}; font-size: 0.68rem; color: {TEXT_MUTED};
            text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px;
        }}
        .finfy-card-value {{
            font-family: {MONO_FONT}; font-size: 1.55rem; font-weight: 700;
            color: {TEXT_PRIMARY}; margin-bottom: 4px; line-height: 1.15;
        }}
        .finfy-card-delta {{ font-family: {MONO_FONT}; font-size: 0.82rem; font-weight: 500; }}
        .finfy-card-delta.risk {{ color: {ACCENT_RED}; }}
        .finfy-card-delta.positive {{ color: {ACCENT_GREEN}; }}
        .finfy-card-delta.neutral {{ color: {TEXT_MUTED}; }}

        /* ---------------- Section / Panel Headers ---------------- */
        .finfy-panel-head {{
            display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 6px;
        }}
        .finfy-panel-title {{ font-family: {SANS_FONT}; font-size: 1.0rem; font-weight: 700; color: {TEXT_PRIMARY}; }}
        .finfy-panel-tag {{
            font-family: {MONO_FONT}; font-size: 0.68rem; color: {TEXT_MUTED};
            letter-spacing: 0.5px; text-transform: uppercase;
        }}
        .finfy-panel-sub {{ font-family: {SANS_FONT}; font-size: 0.8rem; color: {TEXT_MUTED}; margin-bottom: 12px; }}
        .finfy-panel-wrap {{
            background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px;
            padding: 18px 18px 6px 18px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.04);
            backdrop-filter: blur(10px); margin-bottom: 8px;
        }}

        /* ---------------- Ticker Logo Avatars (Portfolio Composition) ---------------- */
        .finfy-asset-row {{
            display: flex; align-items: center; gap: 10px;
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 8px 12px;
            margin-bottom: 8px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.22);
        }}
        .finfy-avatar {{
            width: 28px; height: 28px; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            background: {SURFACE_SOLID};
            border: 1px solid {BORDER_STRONG};
            overflow: hidden;
            flex-shrink: 0;
        }}
        .finfy-avatar img {{ width: 100%; height: 100%; object-fit: contain; background: #fff; }}
        .finfy-avatar-fallback {{
            width: 100%; height: 100%;
            display: flex; align-items: center; justify-content: center;
            font-family: {MONO_FONT}; font-size: 0.65rem; font-weight: 700;
            color: {ACCENT_GREEN}; background: rgba(0,230,118,0.08);
        }}
        .finfy-asset-ticker {{
            font-family: {MONO_FONT}; font-size: 0.82rem; font-weight: 700; color: {TEXT_PRIMARY};
            min-width: 78px;
        }}
        .finfy-asset-weight {{
            font-family: {MONO_FONT}; font-size: 0.78rem; color: {ACCENT_GREEN}; margin-left: auto;
        }}

        .finfy-chip {{
            display: inline-flex; align-items: center; gap: 6px;
            font-family: {MONO_FONT}; font-size: 0.75rem;
            padding: 3px 10px 3px 4px;
            border-radius: 999px;
            border: 1px solid {BORDER_STRONG};
            color: {TEXT_PRIMARY};
            background: rgba(255,255,255,0.03);
            margin: 2px 4px 2px 0;
        }}
        .finfy-chip .finfy-avatar {{ width: 18px; height: 18px; border-radius: 5px; }}

        hr {{ border-color: {BORDER_STRONG} !important; }}

        [data-testid="stDataFrame"] {{
            border: 1px solid {BORDER_STRONG}; border-radius: 10px; font-family: {MONO_FONT};
        }}
        div[data-testid="stStatusWidget"] {{ font-family: {MONO_FONT}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Logo / Avatar Helpers — Robust 3-Tier Resolution Pipeline
# ============================================================
# Tier 1: static Clearbit CDN lookup keyed off a curated ticker->domain map.
# Tier 2: yfinance upstream metadata (`Ticker.info["logo_url"]`), wrapped in
#         a strict try/except so a bad network response never hangs render.
# Tier 3: pure CSS/HTML circular monogram avatar -- guaranteed to render,
#         never a broken <img> tag.
#
# Every candidate URL is validated server-side (HEAD request, short
# timeout) *before* it is ever embedded in HTML, so the browser is never
# asked to load a URL that might fail -- eliminating broken-image icons
# entirely. Results are cached so repeated Streamlit reruns never re-hit
# the network for the same ticker.

LOGO_REQUEST_TIMEOUT_SECONDS = 2.5


def _tier1_clearbit_url(ticker: str) -> Optional[str]:
    """Tier 1: static Clearbit CDN URL from the curated domain map."""
    domain = TICKER_DOMAINS.get(ticker.upper())
    if not domain:
        return None
    return CLEARBIT_LOGO_URL.format(domain=domain)


def _tier2_yfinance_url(ticker: str) -> Optional[str]:
    """
    Tier 2: fall back to yfinance's own upstream metadata for a logo URL.
    Strictly wrapped in try/except -- any network error, missing key, or
    malformed response is swallowed and treated as "no logo available"
    rather than ever crashing or hanging the page render.
    """
    try:
        info = yf.Ticker(ticker).get_info()
        logo_url = info.get("logo_url") or info.get("logoUrl")
        if logo_url and isinstance(logo_url, str) and logo_url.startswith("http"):
            return logo_url
    except Exception:
        pass
    return None


def _url_resolves_to_image(url: str) -> bool:
    """
    Validate that a candidate logo URL actually resolves successfully
    (HTTP 200) within a short timeout, so we never embed a dead link in
    the page. Any exception (timeout, DNS failure, connection error) is
    treated as a failed resolution.
    """
    try:
        response = requests.head(
            url, timeout=LOGO_REQUEST_TIMEOUT_SECONDS, allow_redirects=True
        )
        if response.status_code == 200:
            return True
        # Some CDNs don't support HEAD cleanly -- fall back to a light GET.
        if response.status_code in (403, 405):
            get_resp = requests.get(
                url, timeout=LOGO_REQUEST_TIMEOUT_SECONDS, stream=True
            )
            return get_resp.status_code == 200
        return False
    except Exception:
        return False


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def get_robust_logo_url(ticker: str) -> Optional[str]:
    """
    Resolve a validated, guaranteed-working logo URL for `ticker` using the
    3-tier pipeline (Clearbit -> yfinance metadata -> None). Returns None
    if no tier produces a URL that actually resolves to a live image --
    callers must render the Tier 3 CSS monogram fallback in that case.
    Cached per-ticker for 6 hours to avoid redundant network calls on
    every Streamlit rerun.
    """
    for candidate in (_tier1_clearbit_url(ticker), _tier2_yfinance_url(ticker)):
        if candidate and _url_resolves_to_image(candidate):
            return candidate
    return None


def render_avatar_html(ticker: str, size: int = 28) -> str:
    """
    Build an HTML micro-avatar. Renders a crisp corporate logo image only
    if a URL has already been server-side validated; otherwise renders the
    premium circular CSS/HTML monogram fallback -- never a broken <img>.
    """
    logo_url = get_robust_logo_url(ticker)
    monogram = ticker[:2].upper()
    if logo_url:
        return (
            f'<div class="finfy-avatar" style="width:{size}px;height:{size}px;">'
            f'<img src="{logo_url}" alt="{monogram}" '
            f'style="width:100%;height:100%;object-fit:contain;background:#fff;" />'
            f"</div>"
        )
    return (
        f'<span class="finfy-avatar-fallback" '
        f'style="display:inline-flex;align-items:center;justify-content:center;'
        f"width:{size}px;height:{size}px;line-height:1;border-radius:50%;"
        f'background-color:{BORDER_STRONG};color:#FFFFFF;text-align:center;'
        f"font-family:{MONO_FONT};font-size:{max(9, int(size * 0.4))}px;"
        f'font-weight:700;flex-shrink:0;">{monogram}</span>'
    )



def render_portfolio_composition(tickers: List[str], weights: List[float]) -> None:
    """Render the 'Portfolio Composition' panel: logo avatar + ticker + weight per row."""
    rows = []
    for t, w in zip(tickers, weights):
        avatar_html = render_avatar_html(t, size=28)
        rows.append(
            f'<div class="finfy-asset-row">{avatar_html}'
            f'<span class="finfy-asset-ticker">{t}</span>'
            f'<span class="finfy-asset-weight">{w:.2%}</span></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_ticker_chips(tickers: List[str]) -> None:
    """Render a compact row of logo+ticker chips (used in Portfolio Details tab)."""
    chips = []
    for t in tickers:
        avatar_html = render_avatar_html(t, size=18)
        chips.append(f'<span class="finfy-chip">{avatar_html}{t}</span>')
    st.markdown("".join(chips), unsafe_allow_html=True)


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
    """Dominant-canvas multi-line chart tracking each asset's normalized (base=100) path."""
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
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=10)),
        hovermode="x unified",
    )
    return fig


def build_risk_tail_chart(
    portfolio_returns: pd.Series,
    var_pct: float,
    confidence_label: str,
    horizon_days: int,
) -> go.Figure:
    """Risk Tail Distribution Canvas: empirical returns density with a VaR cutoff marker."""
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

    try:
        from scipy.stats import gaussian_kde
        import numpy as np

        kde = gaussian_kde(returns_pct.dropna())
        x_grid = np.linspace(returns_pct.min(), returns_pct.max(), 300)
        y_grid = kde(x_grid)
        fig.add_trace(
            go.Scatter(
                x=x_grid, y=y_grid, mode="lines",
                line=dict(color=ACCENT_BLUE, width=2),
                fill="tozeroy", fillcolor="rgba(110, 140, 255, 0.12)",
                name="Density",
            )
        )
    except Exception:
        pass

    fig.add_vline(
        x=-var_pct * 100,
        line_width=3, line_dash="dash", line_color=ACCENT_RED,
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
            x=cumulative_returns.index, y=cumulative_returns.values, mode="lines",
            line=dict(color=ACCENT_GREEN, width=2),
            fill="tozeroy", fillcolor="rgba(0,230,118,0.08)",
            name="Cumulative Growth ($1 invested)",
        )
    )
    fig.update_layout(
        template="plotly_dark", height=420,
        margin=dict(t=20, b=40, l=44, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO_FONT, color=TEXT_PRIMARY, size=11),
        xaxis_title="Date", yaxis_title="Growth of $1",
    )
    return fig


def build_drawdown_chart(drawdown: pd.Series) -> go.Figure:
    """Area chart of historical portfolio drawdowns."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown.values * 100, mode="lines",
            line=dict(color=ACCENT_RED, width=2),
            fill="tozeroy", fillcolor="rgba(232,116,107,0.15)",
            name="Drawdown (%)",
        )
    )
    fig.update_layout(
        template="plotly_dark", height=420,
        margin=dict(t=20, b=40, l=44, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO_FONT, color=TEXT_PRIMARY, size=11),
        xaxis_title="Date", yaxis_title="Drawdown (%)",
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

    tickers: List[str] = st.multiselect(
        "Select Tickers",
        options=ASSET_REGISTRY,
        default=DEFAULT_TICKERS,
        help="Type to search the institutional asset registry (equities, ETFs, and crypto anchors).",
    )

    st.markdown(
        f"<div style='font-family:{MONO_FONT}; font-size:0.7rem; color:{TEXT_MUTED}; "
        f"text-transform:uppercase; letter-spacing:0.5px; margin: 10px 0 4px 0;'>Allocation Weights</div>",
        unsafe_allow_html=True,
    )

    # Per-ticker numeric weight input, replacing the old comma-split text
    # field entirely. Defaults to an equal-weight split.
    raw_weights: List[float] = []
    if tickers:
        equal_share = round(100.0 / len(tickers), 2)
        for t in tickers:
            w = st.number_input(
                f"{t} weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=equal_share,
                step=1.0,
                key=f"weight_{t}",
            )
            raw_weights.append(w)
    else:
        st.caption("Select at least one ticker above to configure weights.")

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
    run_clicked = st.button("Run Analysis", use_container_width=True, type="primary", disabled=not tickers)


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
    if run_clicked:
        status_box = st.status("Finfy Core: Initializing risk engine...", expanded=True)

        def _on_status(msg: str) -> None:
            status_box.write(msg)

        result = run_var_analysis(
            tickers=tickers,
            weights=raw_weights,
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

    # ---------------- Portfolio Composition (Logo Grid) ----------------
    st.markdown('<div class="finfy-panel-wrap">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="finfy-panel-head">
            <div class="finfy-panel-title">Portfolio Composition</div>
            <div class="finfy-panel-tag">Live Holdings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    comp_cols = st.columns(min(len(portfolio["tickers"]), 6) or 1)
    for idx, (t, w) in enumerate(zip(portfolio["tickers"], portfolio["weights"])):
        with comp_cols[idx % len(comp_cols)]:
            render_portfolio_composition([t], [w])
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # ---------------- Glass KPI Grid ----------------
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card(
            f"Historical VaR · {eff_confidence_label} · {eff_horizon}D",
            f"${hist['var_dollar']:,.2f}", f"-{hist['var_pct']:.2%}", "risk", ACCENT_RED,
        )
    with col2:
        render_kpi_card(
            f"Parametric VaR · {eff_confidence_label} · {eff_horizon}D",
            f"${param['var_dollar']:,.2f}", f"-{param['var_pct']:.2%}", "risk", ACCENT_AMBER,
        )
    with col3:
        render_kpi_card(
            f"Monte Carlo VaR · {eff_confidence_label} · {eff_horizon}D",
            f"${mc['var_dollar']:,.2f}", f"-{mc['var_pct']:.2%}", "risk", ACCENT_BLUE,
        )
    with col4:
        render_kpi_card(
            "Portfolio Value", f"${portfolio['portfolio_value']:,.2f}",
            "Base Capital", "positive", ACCENT_GREEN,
        )
    with col5:
        render_kpi_card(
            "Observations", f"{portfolio['n_observations']:,}",
            f"Lookback: {period_label}", "neutral", TEXT_MUTED,
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
            build_risk_tail_chart(portfolio["returns"], hist["var_pct"], eff_confidence_label, eff_horizon),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # ---------------- Tabs: Performance / Drawdown / Details ----------------
    tab1, tab2, tab3 = st.tabs(["Portfolio Performance", "Drawdown", "Portfolio Details"])

    with tab1:
        st.plotly_chart(build_cumulative_chart(portfolio["cumulative_returns"]), use_container_width=True)

    with tab2:
        st.plotly_chart(build_drawdown_chart(portfolio["drawdown"]), use_container_width=True)

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
            render_ticker_chips(portfolio["tickers"])
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
    st.info("Select tickers and configure weights in the sidebar, then click **Run Analysis** to begin.")
