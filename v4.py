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
# CSS (保持原樣)
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
    "vix_patch_info":     None,          # 新增：顯示補齊資訊
    "vix_latest_quote":   None,          # 新增：儲存最新 quote 資訊
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# Mode config (保持原樣)
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
        "subtitle":     "數據源：Yahoo Finance 實時 API  ·  盤前 07:30 ET（倫敦 12:30）起  ·  盤後 16:00–20:00 ET",
        "note":         "盤前模式使用 Yahoo Finance v8 Chart API 獲取 VIX 實時 1 分鐘數據（與 TradingView Yahoo feed 同源），從美東 07:30（倫敦 12:30）開始可用。Alpaca 僅作最後備用。",
    },
}

# ─────────────────────────────────────────────
# Telegram (保持原樣)
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
# Data fetching
# ─────────────────────────────────────────────

@st.cache_data(ttl=45)  # 稍微縮短 ttl
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

@st.cache_data(ttl=25)   # 縮短到 25s，讓感覺更即時
def fetch_yahoo_chart_api(ticker: str, bars: int = 30) -> pd.DataFrame:
    urls = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1m&range=1d&includePrePost=true",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1m&range=1d&includePrePost=true",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com/",
    }

    last_err = ""
    for url in urls:
        try:
            session = requests.Session()
            session.headers.update(headers)
            session.get("https://finance.yahoo.com", timeout=5)
            r = session.get(url, timeout=10)

            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                continue

            data = r.json()
            result = data.get("chart", {}).get("result") or []
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
            return df.tail(bars)

        except Exception as e:
            last_err = str(e)
            continue

    st.session_state[f"yahoo_api_{ticker}_ok"] = f"❌ {ticker}: {last_err}"
    return pd.DataFrame()

# Alpaca 部分保持原樣（略過貼出以節省空間，若需要可保留原版）

def fetch_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    is_pre = st.session_state.mode == "premarket"

    # TSLA
    tsla = fetch_yahoo_chart_api("TSLA", 30)
    if tsla.empty:
        tsla = fetch_yfinance("TSLA", 30, prepost=is_pre)

    # VIX
    vix = fetch_yahoo_chart_api("%5EVIX", 30)
    if vix.empty:
        vix = fetch_yahoo_chart_api("^VIX", 30)
    if vix.empty:
        vix = fetch_yfinance("^VIX", 30, prepost=is_pre)

    # 強制 patch VIX（核心改進）
    if not vix.empty or tsla.empty:  # 即使 vix 空也嘗試 patch 出最新
        vix = patch_vix_latest(vix)

    return tsla, vix

# ─────────────────────────────────────────────
# 加強版 patch_vix_latest：強制補到「現在」
# ─────────────────────────────────────────────
def fetch_yahoo_realtime_quote(ticker: str) -> dict:
    # 移除 cache 或設 ttl=10，讓 quote 更即時
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}"
        headers = {
            "User-Agent": "Mozilla/5.0 ...",  # 同上
            "Referer": "https://finance.yahoo.com/",
        }
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            return {}
        result = r.json().get("quoteResponse", {}).get("result", [])
        if not result:
            return {}
        q = result[0]

        # 優先取 preMarket / postMarket / regular
        price = None
        ts = None
        source = "regular"
        if q.get("preMarketPrice") and q.get("preMarketTime"):
            price = q["preMarketPrice"]
            ts = q["preMarketTime"]
            source = "pre"
        elif q.get("postMarketPrice") and q.get("postMarketTime"):
            price = q["postMarketPrice"]
            ts = q["postMarketTime"]
            source = "post"
        elif q.get("regularMarketPrice") and q.get("regularMarketTime"):
            price = q["regularMarketPrice"]
            ts = q["regularMarketTime"]
            source = "regular"

        if price and ts:
            dt = pd.Timestamp(ts, unit="s", tz="UTC").tz_convert("America/New_York")
            return {
                "price": price,
                "time": dt,
                "change": q.get("regularMarketChange", 0),
                "changePct": q.get("regularMarketChangePercent", 0),
                "source": source
            }
        return {}
    except:
        return {}

def patch_vix_latest(vix_df: pd.DataFrame) -> pd.DataFrame:
    quote = fetch_yahoo_realtime_quote("%5EVIX")
    if not quote or "price" not in quote:
        st.session_state["vix_patch_info"] = "無法取得 VIX 即時報價"
        return vix_df

    now_et = datetime.now(pytz.timezone("America/New_York")).floor("T")  # floor to minute
    qt = quote["time"].floor("T")

    # 取最後一根 bar 的時間（若空則假設從 30 分前開始）
    if vix_df.empty:
        last_t = now_et - timedelta(minutes=30)
    else:
        last_t = vix_df.index[-1].floor("T")

    minutes_to_patch = int((qt - last_t).total_seconds() / 60)

    if minutes_to_patch <= 1:
        # 已夠新，僅更新最後一根 close
        vix_df.iloc[-1, vix_df.columns.get_loc("Close")] = quote["price"]
        st.session_state["vix_latest_quote"] = quote
        return vix_df

    # 強制補齊到 quote 時間
    price = quote["price"]
    new_rows = []
    t = last_t + timedelta(minutes=1)
    while t <= qt:
        new_rows.append({
            "Open": price, "High": price,
            "Low": price, "Close": price,
            "Volume": 0
        })
        t += timedelta(minutes=1)

    if new_rows:
        new_df = pd.DataFrame(new_rows, index=pd.date_range(start=last_t + timedelta(minutes=1), periods=len(new_rows), freq="T", tz="America/New_York"))
        vix_df = pd.concat([vix_df, new_df])

    # 最後再更新最新 quote 資訊
    st.session_state["vix_patch_info"] = f"已補齊 {len(new_rows)} 分鐘至 {qt.strftime('%H:%M')}"
    st.session_state["vix_latest_quote"] = quote

    return vix_df

# 其他函數（align_timestamps, compute_correlation, detect_divergence 等）保持原樣，略過重貼

# ─────────────────────────────────────────────
# 主程式邏輯（部分修改）
# ─────────────────────────────────────────────
# ... (sidebar, header 等保持原樣)

if should_refresh:
    with st.spinner("正在拉取最新市場數據…"):
        tsla_df, fear_df = fetch_pair()
    st.session_state.last_refresh = datetime.now()

    # ... (計算 corr, 偵測偏離, 發送警報 等保持原樣)

# ─────────────────────────────────────────────
# Metric cards – 新增 VIX quote 時間顯示
# ─────────────────────────────────────────────
with status_row:
    c1, c2, c3, c4, c5 = st.columns(5)
    tsla_p, tsla_chg = price_delta(tsla_df)
    fear_p, fear_chg = price_delta(fear_df)
    corr_str = f"{corr:.3f}" if corr is not None else "—"
    corr_cls = "negative" if corr and corr > -0.3 else "positive"

    # TSLA card (原樣)
    with c1:
        cls = "positive" if tsla_chg >= 0 else "negative"
        st.markdown(f"""<div class="metric-card">...""", unsafe_allow_html=True)  # 原內容

    # VIX card – 加強顯示 quote 時間與 patch 狀態
    with c2:
        lag = get_vix_lag(tsla_df, fear_df)
        lag_str = f"落後 {lag} 分鐘" if lag > 2 else "同步 ✓"
        lag_cls = "negative" if lag > 5 else ("neutral" if lag > 2 else "positive")

        quote_time_str = "—"
        if st.session_state.vix_latest_quote:
            qt = st.session_state.vix_latest_quote["time"]
            source = st.session_state.vix_latest_quote.get("source", "?")
            quote_time_str = f"{qt.strftime('%H:%M')} ({source})"

        patch_str = st.session_state.get("vix_patch_info", "")

        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">{cfg['fear_display']} 最新價</div>
            <div class="metric-value {cls}">{fear_p:.2f}</div>
            <div class="metric-delta {lag_cls}">{lag_str}</div>
            <div style="font-size:11px; color:#8899aa; margin-top:4px;">
                Quote 時間：{quote_time_str}<br>{patch_str}
            </div>
        </div>""", unsafe_allow_html=True)

    # 其他 card (相關係數、狀態、模式) 保持原樣

# ... 其餘圖表、歷史相關係數等部分保持原樣

# Auto-refresh
if st.session_state.auto_refresh:
    time.sleep(60)
    st.rerun()
