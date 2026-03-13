import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import requests

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA vs UVXY 負相關儀表板",
    page_icon="📊",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0e1117; }
  .metric-card {
      background: #1c1f26;
      border-radius: 10px;
      padding: 16px 20px;
      border: 1px solid #2d3139;
      text-align: center;
  }
  .metric-label { color: #8b8fa8; font-size: 0.78rem; letter-spacing: 0.05em; }
  .metric-value { font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
  .metric-sub   { font-size: 0.85rem; font-weight: 600; margin-top: 2px; }
  .status-ok   { color: #00d97e; }
  .status-warn { color: #f6c90e; }
  .status-bad  { color: #e84045; }
  .section-title {
      font-size: 1rem; font-weight: 600; color: #c9cdd8;
      border-left: 3px solid #5c7cfa;
      padding-left: 8px; margin-bottom: 8px;
  }
</style>
""", unsafe_allow_html=True)

# ── Telegram helper ───────────────────────────────────────────────────────────
def send_telegram(message: str):
    try:
        token   = st.secrets["telegram"]["bot_token"]
        chat_id = st.secrets["telegram"]["chat_id"]
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=8)
    except Exception as e:
        st.sidebar.warning(f"Telegram 發送失敗: {e}")

# ── Data fetch ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=55)
def fetch_candles(ticker: str, bars: int = 30) -> pd.DataFrame:
    """Fetch 1-minute candles; returns last `bars` rows."""
    df = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
    if df.empty:
        return df
    df = df.tail(bars).copy()
    df.index = pd.to_datetime(df.index)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

# ── Candle chart builder ──────────────────────────────────────────────────────
def make_candle_chart(df: pd.DataFrame, title: str, color_up="#00d97e", color_dn="#e84045") -> go.Figure:
    fig = go.Figure(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],  close=df["Close"],
        increasing_line_color=color_up, decreasing_line_color=color_dn,
        increasing_fillcolor=color_up, decreasing_fillcolor=color_dn,
        name=title,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#c9cdd8")),
        paper_bgcolor="#1c1f26", plot_bgcolor="#1c1f26",
        xaxis=dict(gridcolor="#2d3139", showgrid=True, rangeslider=dict(visible=False), color="#8b8fa8"),
        yaxis=dict(gridcolor="#2d3139", showgrid=True, color="#8b8fa8"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
    )
    return fig

# ── Correlation chart ─────────────────────────────────────────────────────────
def make_corr_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["time"], y=history["corr"],
        mode="lines+markers",
        line=dict(color="#5c7cfa", width=2),
        marker=dict(size=4),
        name="皮爾森相關係數",
    ))
    fig.add_hline(y=-0.5, line_dash="dash", line_color="#f6c90e",
                  annotation_text="警戒線 −0.5", annotation_font_color="#f6c90e")
    fig.add_hline(y=0, line_dash="dot", line_color="#8b8fa8")
    fig.update_layout(
        title=dict(text="歷史相關係數走勢", font=dict(size=14, color="#c9cdd8")),
        paper_bgcolor="#1c1f26", plot_bgcolor="#1c1f26",
        xaxis=dict(gridcolor="#2d3139", color="#8b8fa8"),
        yaxis=dict(gridcolor="#2d3139", color="#8b8fa8", range=[-1.1, 1.1]),
        margin=dict(l=10, r=10, t=40, b=10),
        height=240,
        legend=dict(font=dict(color="#8b8fa8"), bgcolor="#1c1f26"),
    )
    return fig

# ── Session state init ────────────────────────────────────────────────────────
if "corr_history" not in st.session_state or not isinstance(st.session_state.corr_history, pd.DataFrame):
    st.session_state.corr_history = pd.DataFrame({"time": pd.Series(dtype="datetime64[ns]"), "corr": pd.Series(dtype="float64")})
if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = None
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 控制面板")
    detect_window = st.slider("偵測視窗（根K線）", 3, 10, 5)
    cooldown_min  = st.slider("提醒冷卻時間（分鐘）", 1, 30, 5)
    display_bars  = st.slider("K線顯示數量", 10, 30, 15)
    auto_refresh  = st.toggle("每分鐘自動刷新", value=True)
    if st.button("🔄 立即手動刷新"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.markdown("### 📋 提醒紀錄")
    if st.session_state.alert_log:
        for entry in reversed(st.session_state.alert_log[-10:]):
            st.markdown(f"<small style='color:#f6c90e'>{entry}</small>", unsafe_allow_html=True)
    else:
        st.markdown("<small style='color:#8b8fa8'>尚無提醒</small>", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("# 📊 TSLA vs UVXY 實時負相關儀表板")
st.markdown("<small style='color:#8b8fa8'>每分鐘自動更新 · 1分K線 · 負相關失效即時 Telegram 提醒</small>", unsafe_allow_html=True)
st.divider()

# ── Fetch data ────────────────────────────────────────────────────────────────
with st.spinner("載入最新市場數據…"):
    tsla_df = fetch_candles("TSLA", bars=max(display_bars + 5, 30))
    uvxy_df = fetch_candles("UVXY", bars=max(display_bars + 5, 30))

data_ok = not tsla_df.empty and not uvxy_df.empty

# ── Compute correlation ───────────────────────────────────────────────────────
corr_value = None
direction_diverge = False

if data_ok:
    tsla_show = tsla_df.tail(display_bars)
    uvxy_show = uvxy_df.tail(display_bars)

    # Align on common timestamps
    common_idx = tsla_df.index.intersection(uvxy_df.index)
    if len(common_idx) >= detect_window:
        tsla_aligned = tsla_df.loc[common_idx, "Close"]
        uvxy_aligned = uvxy_df.loc[common_idx, "Close"]
        corr_value = float(tsla_aligned.corr(uvxy_aligned))

        # Append to history (keep last 120 points)
        now = datetime.now()
        new_row = pd.DataFrame({"time": [pd.Timestamp(now)], "corr": [float(corr_value)]})
        existing = st.session_state.corr_history
        if not isinstance(existing, pd.DataFrame):
            existing = pd.DataFrame({"time": pd.Series(dtype="datetime64[ns]"), "corr": pd.Series(dtype="float64")})
        st.session_state.corr_history = pd.concat(
            [existing, new_row], ignore_index=True
        ).tail(120).reset_index(drop=True)

        # Direction check over detect_window
        tsla_recent = tsla_aligned.iloc[-detect_window:]
        uvxy_recent = uvxy_aligned.iloc[-detect_window:]
        tsla_up = tsla_recent.iloc[-1] > tsla_recent.iloc[0]
        uvxy_up = uvxy_recent.iloc[-1] > uvxy_recent.iloc[0]
        direction_diverge = (tsla_up == uvxy_up)  # same direction → diverge

        # Alert logic
        alert_needed = direction_diverge or corr_value > -0.5
        cooldown_ok = (
            st.session_state.last_alert_time is None or
            (now - st.session_state.last_alert_time).total_seconds() > cooldown_min * 60
        )
        if alert_needed and cooldown_ok:
            tsla_price = float(tsla_aligned.iloc[-1])
            uvxy_price = float(uvxy_aligned.iloc[-1])
            direction_str = "同漲 🔺" if tsla_up else "同跌 🔻"
            msg = (
                f"⚠️ *TSLA vs UVXY 負相關失效*\n"
                f"時間：`{now.strftime('%Y-%m-%d %H:%M')}`\n"
                f"皮爾森係數：`{corr_value:.3f}`\n"
                f"方向偏離：{direction_str if direction_diverge else '未同向'}\n"
                f"TSLA：`${tsla_price:.2f}` | UVXY：`${uvxy_price:.2f}`\n"
                f"偵測視窗：{detect_window} 根K線"
            )
            send_telegram(msg)
            st.session_state.last_alert_time = now
            log_entry = f"{now.strftime('%H:%M')} 係數={corr_value:.2f} {'同向偏離' if direction_diverge else '係數偏高'}"
            st.session_state.alert_log.append(log_entry)

# ── Metric cards ──────────────────────────────────────────────────────────────
tsla_price = float(tsla_df["Close"].iloc[-1]) if data_ok else None
uvxy_price = float(uvxy_df["Close"].iloc[-1]) if data_ok else None

# Compute % change from first candle in display window
def pct_change(df, bars):
    tail = df["Close"].tail(bars)
    if len(tail) < 2:
        return None
    return (tail.iloc[-1] - tail.iloc[0]) / tail.iloc[0] * 100

tsla_pct = pct_change(tsla_df, display_bars) if data_ok else None
uvxy_pct = pct_change(uvxy_df, display_bars) if data_ok else None

def fmt_pct(v):
    if v is None:
        return "—"
    arrow = "▲" if v >= 0 else "▼"
    return f"{arrow} {abs(v):.2f}%"

def pct_color(v):
    if v is None:
        return "#8b8fa8"
    return "#00d97e" if v >= 0 else "#e84045"

if corr_value is not None and corr_value < -0.7:
    corr_color = "#00d97e"
    status_text = "✅ 負相關正常"
elif corr_value is not None and corr_value < -0.5:
    corr_color = "#f6c90e"
    status_text = "⚠️ 相關性偏弱"
else:
    corr_color = "#e84045"
    status_text = "🚨 負相關失效"

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

def metric_card(col, label, value, color, sub=None, sub_color="#8b8fa8"):
    sub_html = f'<div class="metric-sub" style="color:{sub_color}">{sub}</div>' if sub else ""
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{color}">{value}</div>
          {sub_html}
        </div>""", unsafe_allow_html=True)

metric_card(c1, "TSLA 最新價",   f"${tsla_price:.2f}" if tsla_price else "—", "#5c7cfa",
            sub=fmt_pct(tsla_pct), sub_color=pct_color(tsla_pct))
metric_card(c2, "TSLA 漲跌幅",   fmt_pct(tsla_pct), pct_color(tsla_pct))
metric_card(c3, "UVXY 最新價",   f"${uvxy_price:.2f}" if uvxy_price else "—", "#f6c90e",
            sub=fmt_pct(uvxy_pct), sub_color=pct_color(uvxy_pct))
metric_card(c4, "UVXY 漲跌幅",   fmt_pct(uvxy_pct), pct_color(uvxy_pct))
metric_card(c5, "皮爾森係數",    f"{corr_value:.3f}" if corr_value is not None else "—", corr_color)
metric_card(c6, "相關性狀態",    status_text if corr_value is not None else "計算中",
            "#e84045" if direction_diverge else "#00d97e")
metric_card(c7, "最後更新",      datetime.now().strftime("%H:%M:%S"), "#8b8fa8")

st.markdown("<br>", unsafe_allow_html=True)

# ── Dual candlestick charts ───────────────────────────────────────────────────
st.markdown('<div class="section-title">雙 K 線圖（最新 {} 根 1 分鐘蠟燭）</div>'.format(display_bars), unsafe_allow_html=True)

if data_ok:
    col_tsla, col_uvxy = st.columns(2)
    with col_tsla:
        fig_tsla = make_candle_chart(tsla_show, f"TSLA  —  ${tsla_price:.2f}")
        st.plotly_chart(fig_tsla, use_container_width=True, config={"displayModeBar": False})
    with col_uvxy:
        fig_uvxy = make_candle_chart(uvxy_show, f"UVXY  —  ${uvxy_price:.2f}", color_up="#f6c90e", color_dn="#e84045")
        st.plotly_chart(fig_uvxy, use_container_width=True, config={"displayModeBar": False})
else:
    st.error("⚠️ 無法載入市場數據，請稍後再試（盤後或非交易時段）")

# ── Correlation history chart ─────────────────────────────────────────────────
st.markdown('<div class="section-title">歷史皮爾森相關係數走勢</div>', unsafe_allow_html=True)

if len(st.session_state.corr_history) >= 2:
    fig_corr = make_corr_chart(st.session_state.corr_history)
    st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("📈 累積相關係數紀錄中，稍後將顯示走勢圖…")

# ── Direction divergence banner ───────────────────────────────────────────────
if direction_diverge:
    st.error(f"🚨 **方向同步偏離警報**：TSLA 與 UVXY 在最新 {detect_window} 根K線方向相同，負相關關係異常！")

# ── Auto refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()
