"""Streamlit dashboard for the statistical arbitrage research engine.

Loads precomputed results.json (produced by scripts/precompute_results.py,
which runs the real pipeline against real market data) and presents them
visually. No numbers here are fabricated for the UI; this is a display
layer on top of the same backtest/walk-forward code in src/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

st.set_page_config(
    page_title="Statistical Arbitrage Research Engine",
    page_icon="\U0001F4C8",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0d1117; }
    h1, h2, h3, p, span, div, label { color: #c9d1d9 !important; }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 13px; color: #8b949e !important; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-value { font-size: 28px; font-weight: 700; margin-top: 4px; }
    .positive { color: #3fb950 !important; }
    .negative { color: #f85149 !important; }
    .neutral { color: #c9d1d9 !important; }
    .section-tag {
        display: inline-block;
        background-color: #1f6feb22;
        color: #58a6ff !important;
        border: 1px solid #1f6feb55;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 8px;
    }
    .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }
    #MainMenu, header, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def metric_card(label: str, value: str, tone: str = "neutral") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {tone}">{value}</div>
    </div>
    """


def tone_for(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


results = load_results()

st.markdown('<span class="section-tag">RESEARCH ENGINE</span>', unsafe_allow_html=True)
st.title("Statistical Arbitrage & Pairs Trading")
st.caption(
    f"Universe: {', '.join(results['universe'])}  •  "
    f"{results['date_range']['start']} to {results['date_range']['end']}  •  "
    "Cointegration screening → cost-aware backtest → walk-forward validation"
)

st.divider()

# ---- Cointegration scan ----
st.subheader("Cointegration Scan")
st.caption("Engle-Granger test across all pairs in the universe, significance p < 0.05")
pairs_df = pd.DataFrame(results["pairs"]).rename(columns={
    "asset_a": "Asset A", "asset_b": "Asset B", "p_value": "p-value",
    "hedge_ratio": "Hedge Ratio (β)", "half_life": "Half-Life (days)",
})
st.dataframe(pairs_df, use_container_width=True, hide_index=True)
st.caption(f"Best pair selected on full history: **{results['best_pair']}**")

st.divider()

# ---- Metrics: naive vs walk-forward ----
st.subheader("Naive vs. Walk-Forward Performance")
st.caption("The naive backtest selects a pair and evaluates it on the same data — optimistic by construction. Walk-forward re-selects the pair on rolling out-of-sample windows.")

col_naive, col_wf = st.columns(2)

with col_naive:
    st.markdown("##### Naive (in-sample selected)")
    n = results["naive"]["summary"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(metric_card("Sharpe Ratio", f"{n['sharpe_ratio']:.3f}", tone_for(n["sharpe_ratio"])), unsafe_allow_html=True)
        st.markdown(metric_card("CAGR", f"{n['cagr']*100:.1f}%", tone_for(n["cagr"])), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Max Drawdown", f"{n['max_drawdown']*100:.1f}%", "negative"), unsafe_allow_html=True)
        st.markdown(metric_card("Total Return", f"{n['total_return']*100:.1f}%", tone_for(n["total_return"])), unsafe_allow_html=True)

with col_wf:
    st.markdown("##### Walk-Forward (honest, out-of-sample)")
    w = results["walk_forward"]["summary"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(metric_card("Sharpe Ratio", f"{w['sharpe_ratio']:.3f}", tone_for(w["sharpe_ratio"])), unsafe_allow_html=True)
        st.markdown(metric_card("CAGR", f"{w['cagr']*100:.1f}%", tone_for(w["cagr"])), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Max Drawdown", f"{w['max_drawdown']*100:.1f}%", "negative"), unsafe_allow_html=True)
        st.markdown(metric_card("Total Return", f"{w['total_return']*100:.1f}%", tone_for(w["total_return"])), unsafe_allow_html=True)

st.divider()

# ---- Equity curve ----
st.subheader("Equity Curve")
naive_ec = results["naive"]["equity_curve"]
wf_ec = results["walk_forward"]["equity_curve"]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=naive_ec["dates"], y=naive_ec["values"], name=f"Naive in-sample ({results['best_pair']})",
    line=dict(color="#58a6ff", width=2),
))
fig.add_trace(go.Scatter(
    x=wf_ec["dates"], y=wf_ec["values"], name="Walk-forward out-of-sample",
    line=dict(color="#f0883e", width=2),
))
fig.update_layout(
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
    font=dict(color="#c9d1d9"),
    xaxis=dict(gridcolor="#21262d", title="Date"),
    yaxis=dict(gridcolor="#21262d", title="Equity ($)"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=40, b=10),
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---- Walk-forward windows ----
st.subheader("Walk-Forward Windows")
st.caption("Each row is a separate out-of-sample trading window with its own pair re-selection.")
windows_df = pd.DataFrame(results["walk_forward"]["windows"])[
    ["window_start", "window_end", "pair", "sharpe_ratio", "total_return", "n_trades"]
].rename(columns={
    "window_start": "Start", "window_end": "End", "pair": "Pair",
    "sharpe_ratio": "Sharpe", "total_return": "Return", "n_trades": "Trades",
})
st.dataframe(windows_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "This is a research tool, not a live trading system. All numbers above are the real, "
    "unedited output of the pipeline in src/ — nothing here is mocked for the UI."
)
