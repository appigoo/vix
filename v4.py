import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
import pytz

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA vs VIX 實時負相關儀表板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { color: #8899aa; font-size: 13px; letter-spacing: 1px; text-transform: uppercase; }
    .metric-value { font-size: 28px; font-weight: 700; margin: 6px 0 2px; }
    .metric-delta { font-size: 13px; }
    .positive { color: #00d084; }
    .negative { color: #ff4d6d; }
    .neutral  { color: #aab0c0; }
    .alert-box {
        background: #1f0a0a;
        border-left: 4px solid #ff4d6d;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 14px;
        color: #ffb3b3;
    }
    .ok-box {
        background: #051a10;
        border-left: 4px solid #00d084;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 14px;
        color: #9effd4;
    }
    .section-title {
        color: #c9d1e0;
        font-size: 16px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }
    div[data-testid="stMetric"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Telegram helper (reads from st.secrets)
# ─────────────────────────────────────────────
def send_telegram(message: str) -> bool:
    """Send a Telegram message via Bot API. Returns True on success."""
    try:
        token   = st.secrets["telegram"]["bot_token"]
        chat_id = st.secrets["telegram"]["chat_id"]
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        st.sidebar.warning(f"Telegram 發送失敗: {e}")
        return False

# ─────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────
@st.cache_data(ttl=55)          # refresh just under 1 min
def fetch_1min_data(ticker: str, bars: int = 30) -> pd.DataFrame:
    """Fetch 1-minute OHLCV bars for the last session."""
    try:
        df = yf.download(
            ticker,
            period="1d",
            interval="1m",
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return pd.DataFrame()
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.tail(bars).copy()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()


def compute_correlation(tsla: pd.DataFrame, vix: pd.DataFrame) -> float | None:
    """Pearson correlation between TSLA close and VIX close on shared timestamps."""
    try:
        ts = tsla["Close"].rename("TSLA")
        vs = vix["Close"].rename("VIX")
        combined = pd.concat([ts, vs], axis=1).dropna()
        if len(combined) < 5:
            return None
        return float(combined["TSLA"].corr(combined["VIX"]))
    except Exception:
        return None


def detect_divergence(tsla: pd.DataFrame, vix: pd.DataFrame, window: int = 5) -> tuple[bool, str]:
    """
    Returns (divergence_detected, description).
    Divergence = both series moving in the SAME direction over the last `window` bars.
    """
    try:
        ts = tsla["Close"].tail(window)
        vs = vix["Close"].tail(window)
        if len(ts) < window or len(vs) < window:
            return False, "數據不足"

        tsla_dir =  1 if ts.iloc[-1] > ts.iloc[0] else (-1 if ts.iloc[-1] < ts.iloc[0] else 0)
        vix_dir  =  1 if vs.iloc[-1] > vs.iloc[0] else (-1 if vs.iloc[-1] < vs.iloc[0] else 0)

        if tsla_dir == 0 or vix_dir == 0:
            return False, "價格持平，無明顯方向"

        if tsla_dir == vix_dir:
            direction = "同步上漲 ↑↑" if tsla_dir == 1 else "同步下跌 ↓↓"
            desc = f"⚠️ 負相關失效！TSLA 與 VIX {direction}（過去 {window} 根K線）"
            return True, desc

        direction = ("TSLA↑ VIX↓" if tsla_dir == 1 else "TSLA↓ VIX↑")
        return False, f"✅ 負相關正常：{direction}"
    except Exception:
        return False, "分析出錯"

# ─────────────────────────────────────────────
# Candlestick chart builder
# ─────────────────────────────────────────────
def make_candle_chart(tsla: pd.DataFrame, vix: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        subplot_titles=("TSLA  1-Min K線（最新15根）", "VIX  1-Min K線（最新15根）"),
        vertical_spacing=0.14,
        row_heights=[0.5, 0.5],
    )

    def add_candles(df: pd.DataFrame, row: int, color_up: str, color_dn: str, name: str):
        tail = df.tail(15)
        fig.add_trace(
            go.Candlestick(
                x=tail.index,
                open=tail["Open"],
                high=tail["High"],
                low=tail["Low"],
                close=tail["Close"],
                name=name,
                increasing_line_color=color_up,
                decreasing_line_color=color_dn,
                increasing_fillcolor=color_up,
                decreasing_fillcolor=color_dn,
                line_width=1,
            ),
            row=row, col=1,
        )

    if not tsla.empty:
        add_candles(tsla, 1, "#00d084", "#ff4d6d", "TSLA")
    if not vix.empty:
        add_candles(vix, 2, "#f4c542", "#ff6e40", "VIX")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#13171f",
        font=dict(family="Inter, sans-serif", color="#c9d1e0", size=12),
        margin=dict(l=10, r=10, t=50, b=10),
        height=620,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e2533", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1e2533", zeroline=False)

    # Colour subplot titles
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=13, color="#8899aa")

    return fig


def make_correlation_chart(corr_history: list[dict]) -> go.Figure:
    if not corr_history:
        return go.Figure()
    df = pd.DataFrame(corr_history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["corr"],
        mode="lines+markers",
        line=dict(color="#5b8af5", width=2),
        marker=dict(size=5, color=df["corr"].apply(
            lambda v: "#ff4d6d" if v > -0.3 else "#00d084"
        )),
        name="皮爾森相關係數",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#555")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="#f4c542",
                  annotation_text="弱負相關閾值 -0.5", annotation_position="bottom right")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#13171f",
        font=dict(color="#c9d1e0", size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        height=220,
        yaxis=dict(range=[-1.1, 1.1]),
        showlegend=False,
    )
    return fig

# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────
if "corr_history"       not in st.session_state: st.session_state.corr_history       = []
if "alert_history"      not in st.session_state: st.session_state.alert_history      = []
if "last_alert_time"    not in st.session_state: st.session_state.last_alert_time    = None
if "last_refresh"       not in st.session_state: st.session_state.last_refresh       = None
if "auto_refresh"       not in st.session_state: st.session_state.auto_refresh       = True
if "alert_cooldown_min" not in st.session_state: st.session_state.alert_cooldown_min = 5
if "diverge_window"     not in st.session_state: st.session_state.diverge_window     = 5

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 控制面板")
    st.session_state.auto_refresh = st.toggle(
        "每分鐘自動更新", value=st.session_state.auto_refresh
    )
    st.session_state.diverge_window = st.slider(
        "偏離偵測視窗（K線數）", 3, 10, st.session_state.diverge_window
    )
    st.session_state.alert_cooldown_min = st.slider(
        "Telegram 提醒冷卻（分鐘）", 1, 30, st.session_state.alert_cooldown_min
    )
    manual_refresh = st.button("🔄 立即刷新")

    st.divider()
    st.markdown("### 📱 Telegram 設定")
    st.caption("請在 `.streamlit/secrets.toml` 中設定：")
    st.code("""[telegram]
bot_token = "YOUR_BOT_TOKEN"
chat_id   = "YOUR_CHAT_ID"
""", language="toml")

    tg_ok = "telegram" in st.secrets
    st.markdown(
        f'<span style="color:{"#00d084" if tg_ok else "#ff4d6d"}">{"✅ Telegram 已設定" if tg_ok else "❌ Telegram 未設定"}</span>',
        unsafe_allow_html=True,
    )
    if tg_ok and st.button("發送測試訊息"):
        ok = send_telegram("🔔 <b>TSLA vs VIX 儀表板</b>\n連線測試成功 ✅")
        st.success("已發送！") if ok else st.error("發送失敗")

    st.divider()
    if st.button("🗑️ 清除歷史紀錄"):
        st.session_state.corr_history  = []
        st.session_state.alert_history = []

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<h1 style="text-align:center; font-size:26px; font-weight:700; color:#c9d1e0; margin-bottom:4px;">
    📊 TSLA × VIX 實時負相關監控儀表板
</h1>
<p style="text-align:center; color:#556070; font-size:13px; margin-bottom:20px;">
    每分鐘自動追蹤 · K線圖 · 相關係數歷史 · Telegram 異常提醒
</p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Main refresh logic
# ─────────────────────────────────────────────
now_et = datetime.now(pytz.timezone("America/New_York"))
should_refresh = (
    manual_refresh
    or st.session_state.last_refresh is None
    or (
        st.session_state.auto_refresh
        and (datetime.now() - st.session_state.last_refresh).total_seconds() >= 60
    )
)

# Placeholder containers (defined before data fetch so we can show spinners)
status_row   = st.container()
chart_col1, chart_col2 = st.columns([3, 2])
corr_section = st.container()
alert_section = st.container()

if should_refresh:
    with st.spinner("正在拉取最新市場數據…"):
        tsla_df = fetch_1min_data("TSLA", 30)
        vix_df  = fetch_1min_data("^VIX",  30)
    st.session_state.last_refresh = datetime.now()

    corr = compute_correlation(tsla_df, vix_df)
    if corr is not None:
        st.session_state.corr_history.append({
            "time": now_et.strftime("%H:%M"),
            "corr": round(corr, 4),
        })
        # Keep last 60 data points
        st.session_state.corr_history = st.session_state.corr_history[-60:]

    diverged, div_desc = detect_divergence(
        tsla_df, vix_df, st.session_state.diverge_window
    )

    if diverged:
        # Cooldown check
        cooldown_ok = True
        if st.session_state.last_alert_time is not None:
            elapsed = (datetime.now() - st.session_state.last_alert_time).total_seconds() / 60
            cooldown_ok = elapsed >= st.session_state.alert_cooldown_min

        ts_price = float(tsla_df["Close"].iloc[-1]) if not tsla_df.empty else 0
        vx_price = float(vix_df["Close"].iloc[-1])  if not vix_df.empty  else 0

        alert_msg = (
            f"⚠️ <b>TSLA × VIX 負相關失效！</b>\n"
            f"時間：{now_et.strftime('%Y-%m-%d %H:%M')} ET\n"
            f"TSLA：${ts_price:.2f}　VIX：{vx_price:.2f}\n"
            f"相關係數：{corr:.3f}\n"
            f"{div_desc}"
        )

        if cooldown_ok:
            sent = send_telegram(alert_msg)
            st.session_state.last_alert_time = datetime.now()
            st.session_state.alert_history.append({
                "time": now_et.strftime("%H:%M"),
                "msg": div_desc,
                "sent": sent,
            })
else:
    # Use cached values for display
    tsla_df = fetch_1min_data("TSLA", 30)
    vix_df  = fetch_1min_data("^VIX",  30)
    corr = compute_correlation(tsla_df, vix_df)
    diverged, div_desc = detect_divergence(
        tsla_df, vix_df, st.session_state.diverge_window
    )

# ─────────────────────────────────────────────
# Status row — metric cards
# ─────────────────────────────────────────────
with status_row:
    c1, c2, c3, c4, c5 = st.columns(5)

    def price_delta(df):
        if df.empty or len(df) < 2:
            return 0.0, 0.0
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2])
        return last, last - prev

    tsla_price, tsla_chg = price_delta(tsla_df)
    vix_price,  vix_chg  = price_delta(vix_df)
    corr_str  = f"{corr:.3f}" if corr is not None else "—"
    corr_cls  = "negative" if corr and corr > -0.3 else "positive"

    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">TSLA 最新價</div>
            <div class="metric-value {"positive" if tsla_chg>=0 else "negative"}">${tsla_price:.2f}</div>
            <div class="metric-delta {"positive" if tsla_chg>=0 else "negative"}">{("+" if tsla_chg>=0 else "")}{tsla_chg:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">VIX 最新價</div>
            <div class="metric-value {"negative" if vix_chg>=0 else "positive"}">{vix_price:.2f}</div>
            <div class="metric-delta {"negative" if vix_chg>=0 else "positive"}">{("+" if vix_chg>=0 else "")}{vix_chg:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">皮爾森相關係數</div>
            <div class="metric-value {corr_cls}">{corr_str}</div>
            <div class="metric-delta neutral">目標 &lt; −0.5</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        status_cls  = "negative" if diverged else "positive"
        status_text = "⚠️ 偏離" if diverged else "✅ 正常"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">負相關狀態</div>
            <div class="metric-value {status_cls}" style="font-size:22px;">{status_text}</div>
            <div class="metric-delta neutral">視窗 {st.session_state.diverge_window} 根</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        refresh_str = st.session_state.last_refresh.strftime("%H:%M:%S") if st.session_state.last_refresh else "—"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">最後更新</div>
            <div class="metric-value neutral" style="font-size:20px;">{refresh_str}</div>
            <div class="metric-delta neutral">ET {now_et.strftime("%m/%d")}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Candlestick + Status
# ─────────────────────────────────────────────
with chart_col1:
    st.markdown('<div class="section-title">📈 K線圖（每根 = 1 分鐘）</div>', unsafe_allow_html=True)
    if not tsla_df.empty and not vix_df.empty:
        st.plotly_chart(
            make_candle_chart(tsla_df, vix_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.warning("無法獲取K線數據，請確認市場是否開盤")

with chart_col2:
    st.markdown('<div class="section-title">🔍 即時分析</div>', unsafe_allow_html=True)
    if diverged:
        st.markdown(f'<div class="alert-box">🚨 <b>負相關失效警告</b><br>{div_desc}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ok-box">{div_desc}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 提醒記錄</div>', unsafe_allow_html=True)
    if st.session_state.alert_history:
        for a in reversed(st.session_state.alert_history[-8:]):
            icon = "📤" if a["sent"] else "⚠️"
            st.markdown(
                f'<div class="alert-box">{icon} [{a["time"]}] {a["msg"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="ok-box">暫無異常提醒記錄</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Correlation history chart
# ─────────────────────────────────────────────
with corr_section:
    st.markdown('<div class="section-title">📉 相關係數歷史（每分鐘）</div>', unsafe_allow_html=True)
    if st.session_state.corr_history:
        st.plotly_chart(
            make_correlation_chart(st.session_state.corr_history),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        df_hist = pd.DataFrame(st.session_state.corr_history)
        avg_c = df_hist["corr"].mean()
        min_c = df_hist["corr"].min()
        max_c = df_hist["corr"].max()
        h1, h2, h3 = st.columns(3)
        h1.metric("平均相關係數", f"{avg_c:.3f}")
        h2.metric("最低（最強負相關）", f"{min_c:.3f}")
        h3.metric("最高（最弱負相關）", f"{max_c:.3f}")
    else:
        st.info("等待數據累積中…（每分鐘更新一次）")

# ─────────────────────────────────────────────
# Auto-refresh every 60 seconds
# ─────────────────────────────────────────────
if st.session_state.auto_refresh:
    time.sleep(60)
    st.rerun()
