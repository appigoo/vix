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
# CSS
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
    .metric-label { color: #8899aa; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; }
    .metric-value { font-size: 26px; font-weight: 700; margin: 6px 0 2px; }
    .metric-delta { font-size: 13px; }
    .positive { color: #00d084; }
    .negative { color: #ff4d6d; }
    .neutral  { color: #aab0c0; }
    .alert-box {
        background: #1f0a0a; border-left: 4px solid #ff4d6d;
        border-radius: 8px; padding: 12px 16px; margin: 6px 0;
        font-size: 13px; color: #ffb3b3;
    }
    .ok-box {
        background: #051a10; border-left: 4px solid #00d084;
        border-radius: 8px; padding: 12px 16px; margin: 6px 0;
        font-size: 13px; color: #9effd4;
    }
    .info-box {
        background: #0d1a2e; border-left: 4px solid #5b8af5;
        border-radius: 8px; padding: 12px 16px; margin: 6px 0;
        font-size: 13px; color: #a8c4ff;
    }
    .mode-badge-regular {
        display:inline-block; background:#0d2e1a; border:1px solid #00d084;
        color:#00d084; border-radius:20px; padding:4px 14px;
        font-size:13px; font-weight:600;
    }
    .mode-badge-premarket {
        display:inline-block; background:#1a1a0d; border:1px solid #f4c542;
        color:#f4c542; border-radius:20px; padding:4px 14px;
        font-size:13px; font-weight:600;
    }
    .section-title {
        color: #c9d1e0; font-size: 15px; font-weight: 600;
        letter-spacing: 0.5px; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────
defaults = {
    "mode":               "regular",
    "corr_history":       [],
    "alert_history":      [],
    "last_alert_time":    None,
    "last_refresh":       None,
    "auto_refresh":       True,
    "alert_cooldown_min": 5,
    "diverge_window":     5,
    "vix_patch_info":     None,
    "vix_latest_quote":   None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# Mode config
# ─────────────────────────────────────────────
MODE_CONFIG = {
    "regular": {
        "label":        "📈 盤中模式",
        "badge":        "regular",
        "fear_label":   "VIX 恐慌指數",
        "fear_display": "VIX",
        "subtitle":     "數據源：Yahoo Finance 實時 API  ·  盤中 09:30–16:00 ET",
        "note":         None,
    },
    "premarket": {
        "label":        "🌅 盤前/盤後模式",
        "badge":        "premarket",
        "fear_label":   "VIX 恐慌指數（含盤前）",
        "fear_display": "VIX",
        "subtitle":     "數據源：Yahoo Finance 實時 API  ·  盤前 07:30 ET起  ·  盤後 16:00–20:00 ET",
        "note":         "盤前模式使用 Yahoo v8 Chart API，從美東 07:30 開始較穩定。Alpaca 為備援。",
    },
}

# ─────────────────────────────────────────────
# Telegram
# ─────────────────────────────────────────────
def send_telegram(message: str) -> bool:
    try:
        token   = st.secrets["telegram"]["bot_token"]
        chat_id = st.secrets["telegram"]["chat_id"]
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        st.sidebar.warning(f"Telegram 失敗: {e}")
        return False

# ─────────────────────────────────────────────
# Data fetching functions
# ─────────────────────────────────────────────

@st.cache_data(ttl=45)
def fetch_yfinance(ticker: str, bars: int = 30, prepost: bool = False) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker, period="1d", interval="1m",
            prepost=prepost, progress=False, auto_adjust=True,
        )
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.tail(bars).copy()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=25)
def fetch_yahoo_chart_api(ticker: str, bars: int = 30) -> pd.DataFrame:
    urls = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1m&range=1d&includePrePost=true",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1m&range=1d&includePrePost=true",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://finance.yahoo.com/",
    }

    for url in urls:
        try:
            session = requests.Session()
            session.headers.update(headers)
            session.get("https://finance.yahoo.com", timeout=5)
            r = session.get(url, timeout=10)
            if r.status_code != 200:
                continue
            data = r.json()
            result = data.get("chart", {}).get("result", [])
            if not result:
                continue
            res = result[0]
            ts = res.get("timestamp", [])
            if not ts:
                continue
            quote = res["indicators"]["quote"][0]
            adjclose = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose", [])
            df = pd.DataFrame({
                "Open":   quote.get("open",   [None]*len(ts)),
                "High":   quote.get("high",   [None]*len(ts)),
                "Low":    quote.get("low",    [None]*len(ts)),
                "Close":  adjclose if adjclose else quote.get("close", [None]*len(ts)),
                "Volume": quote.get("volume", [0]*len(ts)),
            }, index=pd.to_datetime(ts, unit="s", utc=True))
            df = df.dropna(subset=["Close"])
            if df.empty:
                continue
            df.index = df.index.tz_convert("America/New_York")
            st.session_state[f"yahoo_api_{ticker}_ok"] = f"✅ {ticker}: {len(df)} bars"
            return df.tail(bars)
        except Exception as e:
            pass
    st.session_state[f"yahoo_api_{ticker}_ok"] = f"❌ {ticker} 失敗"
    return pd.DataFrame()

def fetch_yahoo_realtime_quote(ticker: str) -> dict:
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://finance.yahoo.com/",
        }
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            return {}
        result = r.json().get("quoteResponse", {}).get("result", [])
        if not result:
            return {}
        q = result[0]
        price = None
        ts = None
        source = "regular"
        if q.get("preMarketPrice") and q.get("preMarketTime"):
            price, ts, source = q["preMarketPrice"], q["preMarketTime"], "pre"
        elif q.get("postMarketPrice") and q.get("postMarketTime"):
            price, ts, source = q["postMarketPrice"], q["postMarketTime"], "post"
        elif q.get("regularMarketPrice") and q.get("regularMarketTime"):
            price, ts = q["regularMarketPrice"], q["regularMarketTime"]
        if price and ts:
            dt = pd.Timestamp(ts, unit="s", tz="UTC").tz_convert("America/New_York")
            return {
                "price": price,
                "time": dt,
                "source": source,
                "changePct": q.get("regularMarketChangePercent", 0),
            }
        return {}
    except:
        return {}

def patch_vix_latest(vix_df: pd.DataFrame) -> pd.DataFrame:
    quote = fetch_yahoo_realtime_quote("%5EVIX")
    if not quote or "price" not in quote:
        st.session_state["vix_patch_info"] = "無法取得 VIX 即時報價"
        return vix_df

    now_et = datetime.now(pytz.timezone("America/New_York")).floor("T")
    qt = quote["time"].floor("T")

    if vix_df.empty:
        last_t = now_et - timedelta(minutes=30)
    else:
        last_t = vix_df.index[-1].floor("T")

    minutes_gap = int((qt - last_t).total_seconds() / 60)

    if minutes_gap <= 1:
        if not vix_df.empty:
            vix_df.iloc[-1, vix_df.columns.get_loc("Close")] = quote["price"]
        st.session_state["vix_latest_quote"] = quote
        return vix_df

    # 補齊
    price = quote["price"]
    new_rows = []
    t = last_t + timedelta(minutes=1)
    while t <= qt:
        new_rows.append({"Open": price, "High": price, "Low": price, "Close": price, "Volume": 0})
        t += timedelta(minutes=1)

    if new_rows:
        new_index = pd.date_range(start=last_t + timedelta(minutes=1), periods=len(new_rows), freq="T", tz="America/New_York")
        new_df = pd.DataFrame(new_rows, index=new_index)
        vix_df = pd.concat([vix_df, new_df])

    st.session_state["vix_patch_info"] = f"補齊 {len(new_rows)} 分鐘至 {qt.strftime('%H:%M')}"
    st.session_state["vix_latest_quote"] = quote
    return vix_df

def fetch_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    is_pre = st.session_state.mode == "premarket"
    tsla = fetch_yahoo_chart_api("TSLA", 30)
    if tsla.empty:
        tsla = fetch_yfinance("TSLA", 30, prepost=is_pre)

    vix = fetch_yahoo_chart_api("%5EVIX", 30)
    if vix.empty:
        vix = fetch_yahoo_chart_api("^VIX", 30)
    if vix.empty:
        vix = fetch_yfinance("^VIX", 30, prepost=is_pre)

    vix = patch_vix_latest(vix)
    return tsla, vix

# ─────────────────────────────────────────────
# Analysis functions (簡化版，保留核心)
# ─────────────────────────────────────────────

def get_vix_lag(tsla_df, vix_df):
    if tsla_df.empty or vix_df.empty:
        return 999
    t = pd.to_datetime(tsla_df.index[-1]).tz_localize(None).floor("T")
    v = pd.to_datetime(vix_df.index[-1]).tz_localize(None).floor("T")
    return max(0, int((t - v).total_seconds() / 60))

def compute_correlation(df1, df2):
    if df1.empty or df2.empty:
        return None
    common = df1.index.intersection(df2.index)
    if len(common) < 5:
        return None
    a = df1.loc[common, "Close"]
    b = df2.loc[common, "Close"]
    return a.corr(b)

def detect_divergence(df1, df2, window=5):
    if df1.empty or df2.empty:
        return False, "數據不足"
    a = df1["Close"].tail(window)
    b = df2["Close"].tail(window)
    if len(a) < window:
        return False, "數據不足"
    d1 = 1 if a.iloc[-1] > a.iloc[0] else (-1 if a.iloc[-1] < a.iloc[0] else 0)
    d2 = 1 if b.iloc[-1] > b.iloc[0] else (-1 if b.iloc[-1] < b.iloc[0] else 0)
    if d1 == 0 or d2 == 0:
        return False, "價格持平"
    if d1 == d2:
        return True, f"⚠️ 同步{'上漲' if d1==1 else '下跌'}（過去{window}分鐘）"
    return False, f"負相關正常：{'TSLA↑ VIX↓' if d1==1 else 'TSLA↓ VIX↑'}"

def price_delta(df):
    if df.empty or len(df) < 2:
        return 0.0, 0.0
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2])
    return last, last - prev

# ─────────────────────────────────────────────
# Charts (簡化版)
# ─────────────────────────────────────────────

def make_candle_chart(tsla_df, vix_df, fear_label):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.15,
                        subplot_titles=("TSLA 1分K", f"{fear_label} 1分K"))
    if not tsla_df.empty:
        t = tsla_df.tail(15)
        fig.add_trace(go.Candlestick(x=t.index, open=t.Open, high=t.High, low=t.Low, close=t.Close,
                                     name="TSLA", increasing_line_color="#00d084", decreasing_line_color="#ff4d6d"),
                      row=1, col=1)
    if not vix_df.empty:
        v = vix_df.tail(15)
        fig.add_trace(go.Candlestick(x=v.index, open=v.Open, high=v.High, low=v.Low, close=v.Close,
                                     name=fear_label, increasing_line_color="#f4c542", decreasing_line_color="#ff6e40"),
                      row=2, col=1)
    fig.update_layout(template="plotly_dark", height=580, showlegend=False,
                      margin=dict(l=10,r=10,t=50,b=10))
    return fig

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 控制")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("盤中模式", type="primary" if st.session_state.mode == "regular" else "secondary"):
            st.session_state.mode = "regular"
            st.rerun()
    with col2:
        if st.button("盤前/後", type="primary" if st.session_state.mode == "premarket" else "secondary"):
            st.session_state.mode = "premarket"
            st.rerun()

    cfg = MODE_CONFIG[st.session_state.mode]
    st.markdown(f'<div class="mode-badge-{cfg["badge"]}">{cfg["label"]}</div>', unsafe_allow_html=True)

    st.session_state.auto_refresh = st.toggle("自動更新（60秒）", st.session_state.auto_refresh)
    manual_refresh = st.button("立即刷新")

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
cfg = MODE_CONFIG[st.session_state.mode]
st.markdown(f"<h2 style='text-align:center'>TSLA × {cfg['fear_display']} 負相關監控</h2>", unsafe_allow_html=True)
st.caption(cfg["subtitle"])

# ─────────────────────────────────────────────
# Refresh logic
# ─────────────────────────────────────────────
now_et = datetime.now(pytz.timezone("America/New_York"))

should_refresh = (
    manual_refresh or
    st.session_state.last_refresh is None or
    (st.session_state.auto_refresh and
     (datetime.now() - st.session_state.last_refresh).total_seconds() >= 60)
)

status_row = st.container()
chart_col1, chart_col2 = st.columns([3, 2])
corr_section = st.container()

if should_refresh:
    with st.spinner("更新中..."):
        tsla_df, fear_df = fetch_pair()
    st.session_state.last_refresh = datetime.now()
else:
    # 避免第一次載入空白，使用最後一次資料或空
    if "tsla_df" not in globals() or "fear_df" not in globals():
        tsla_df, fear_df = fetch_pair()
    else:
        # 使用快取或先前資料
        pass

corr = compute_correlation(tsla_df, fear_df)
diverged, div_desc = detect_divergence(tsla_df, fear_df, st.session_state.diverge_window)

# ─────────────────────────────────────────────
# Metric cards
# ─────────────────────────────────────────────
with status_row:
    c1, c2, c3, c4 = st.columns(4)
    tsla_p, tsla_chg = price_delta(tsla_df)
    fear_p, fear_chg = price_delta(fear_df)

    with c1:
        cls = "positive" if tsla_chg >= 0 else "negative"
        st.metric("TSLA", f"${tsla_p:.2f}", f"{tsla_chg:+.2f}")

    with c2:
        lag = get_vix_lag(tsla_df, fear_df)
        lag_text = f"落後 {lag} min" if lag > 2 else "同步"
        st.metric("VIX", f"{fear_p:.2f}", lag_text)

    with c3:
        corr_str = f"{corr:.3f}" if corr is not None else "—"
        st.metric("相關係數", corr_str)

    with c4:
        status = "⚠️ 偏離" if diverged else "正常"
        st.metric("狀態", status)

# Charts
with chart_col1:
    st.plotly_chart(make_candle_chart(tsla_df, fear_df, cfg["fear_display"]), use_container_width=True)

with chart_col2:
    if diverged:
        st.markdown(f'<div class="alert-box">{div_desc}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ok-box">{div_desc}</div>', unsafe_allow_html=True)

# Auto refresh
if st.session_state.auto_refresh:
    time.sleep(60)
    st.rerun()
