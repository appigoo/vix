import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
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
        "subtitle":     "數據源：Yahoo Finance 實時 API  ·  盤前 07:30 ET（倫敦 12:30）起  ·  盤後 16:00–20:00 ET",
        "note":         "盤前模式使用 Yahoo Finance v8 Chart API 獲取 VIX 實時 1 分鐘數據（與 TradingView Yahoo feed 同源），從美東 07:30（倫敦 12:30）開始可用。Alpaca 僅作最後備用。",
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
# Data fetching
# ─────────────────────────────────────────────

@st.cache_data(ttl=55)
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


@st.cache_data(ttl=30)   # 30s cache for real-time feel
def fetch_yahoo_chart_api(ticker: str, bars: int = 30) -> pd.DataFrame:
    """
    Fetch 1-min OHLCV directly from Yahoo Finance v8 chart API.
    This endpoint returns real-time data including pre/post-market
    for indices like ^VIX — same source TradingView Yahoo feed uses.
    No API key required.
    """
    try:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?interval=1m&range=1d&includePrePost=true"
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        }
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()

        data = r.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return pd.DataFrame()

        res      = result[0]
        ts       = res.get("timestamp", [])
        quote    = res["indicators"]["quote"][0]
        adjclose = res["indicators"].get("adjclose", [{}])[0].get("adjclose", quote.get("close"))

        df = pd.DataFrame({
            "Open":   quote.get("open",   []),
            "High":   quote.get("high",   []),
            "Low":    quote.get("low",    []),
            "Close":  adjclose if adjclose else quote.get("close", []),
            "Volume": quote.get("volume", []),
        }, index=pd.to_datetime(ts, unit="s", utc=True))

        df = df.dropna(subset=["Open", "Close"])
        df.index = df.index.tz_convert("America/New_York")
        return df.tail(bars)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=55)
def fetch_alpaca(ticker: str, bars: int = 30) -> pd.DataFrame:
    """
    1-minute bars from Alpaca supporting pre/post-market.
    Uses start= param to fetch from 4 hours ago so we always
    catch whatever bars exist even in early pre-market.
    """
    try:
        key    = st.secrets["alpaca"]["api_key"]
        secret = st.secrets["alpaca"]["api_secret"]
        headers = {
            "APCA-API-KEY-ID":     key,
            "APCA-API-SECRET-KEY": secret,
        }
        # Use start= 4 hours ago in RFC3339 to capture early pre-market bars
        from datetime import timezone, timedelta
        start_dt = (datetime.now(timezone.utc) - timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = (
            f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
            f"?timeframe=1Min&limit={bars}&feed=iex&adjustment=raw"
            f"&start={start_dt}&sort=desc"
        )
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            st.session_state["alpaca_last_error"] = f"HTTP {r.status_code}：{r.text[:300]}"
            return pd.DataFrame()
        data = r.json().get("bars") or []
        if not data:
            # Try wider window — 8 hours (catches very slow pre-market)
            start_dt2 = (datetime.now(timezone.utc) - timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
            url2 = (
                f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
                f"?timeframe=1Min&limit={bars}&feed=iex&adjustment=raw"
                f"&start={start_dt2}&sort=desc"
            )
            r2 = requests.get(url2, headers=headers, timeout=10)
            data = (r2.json().get("bars") or []) if r2.status_code == 200 else []

        if not data:
            now_et = datetime.now(pytz.timezone("America/New_York"))
            st.session_state["alpaca_last_error"] = (
                f"bars 為空 [{ticker}]  現在美東 {now_et.strftime('%H:%M')} ET\n"
                f"IEX 盤前數據從 07:00 ET（倫敦 12:00）開始，請稍後再試"
            )
            return pd.DataFrame()

        st.session_state["alpaca_last_error"] = None
        # sort=desc means newest first — reverse to chronological
        data = list(reversed(data))
        df = pd.DataFrame(data).rename(columns={
            "t": "Datetime", "o": "Open", "h": "High",
            "l": "Low",      "c": "Close","v": "Volume",
        })
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        st.session_state["alpaca_last_error"] = f"例外錯誤：{type(e).__name__}: {e}"
        return pd.DataFrame()


def test_alpaca_connection() -> tuple[bool, str]:
    """Diagnostic call — returns (success, message)."""
    lines_out = []

    # Step 1: read + validate secrets
    try:
        key    = st.secrets["alpaca"]["api_key"]
        secret = st.secrets["alpaca"]["api_secret"]
    except Exception as e:
        return False, f"❌ secrets.toml 讀取失敗：{e}"

    key    = str(key).strip()    if key    is not None else ""
    secret = str(secret).strip() if secret is not None else ""

    lines_out.append(f"📋 api_key    長度：{len(key)} 字元  前綴：{key[:6] if len(key)>=6 else key}...")
    lines_out.append(f"📋 api_secret 長度：{len(secret)} 字元  前綴：{secret[:4] if len(secret)>=4 else '(空)'}...")

    if not key:
        lines_out.append("❌ api_key 是空值！請檢查 secrets.toml")
        return False, "\n".join(lines_out)
    if not secret:
        lines_out.append("❌ api_secret 是空值！")
        lines_out.append("請在 Alpaca 後台點 Regenerate，重新複製 Secret 填入 secrets.toml")
        return False, "\n".join(lines_out)

    lines_out.append("")
    try:
        headers = {
            "APCA-API-KEY-ID":     key,
            "APCA-API-SECRET-KEY": secret,
        }
        # Step 2: Auth check
        acct = requests.get(
            "https://paper-api.alpaca.markets/v2/account",
            headers=headers, timeout=10,
        )
        lines_out.append(f"🔑 Auth 端點：HTTP {acct.status_code}")
        if acct.status_code == 200:
            info = acct.json()
            lines_out.append(f"   帳戶狀態：{info.get('status','?')}  類型：{info.get('account_type','paper')}")
        else:
            lines_out.append(f"   回應：{acct.text[:300]}")

        # Step 3: Data check
        data_r = requests.get(
            "https://data.alpaca.markets/v2/stocks/TSLA/bars"
            "?timeframe=1Min&limit=3&feed=iex&adjustment=raw",
            headers=headers, timeout=10,
        )
        lines_out.append(f"📊 Data 端點：HTTP {data_r.status_code}")
        if data_r.status_code == 200:
            bars = data_r.json().get("bars") or []
            lines_out.append(f"   取得 TSLA bars：{len(bars)} 根")
            if bars:
                lines_out.append(f"   最新時間：{bars[-1].get('t','?')}")
                lines_out.append(f"   收盤價：${bars[-1].get('c','?')}")
            else:
                lines_out.append("   ⚠️ bars 為空（市場休市或盤前無數據）")
        else:
            lines_out.append(f"   回應：{data_r.text[:300]}")

        success = data_r.status_code == 200
        return success, "\n".join(lines_out)
    except Exception as e:
        lines_out.append(f"❌ 網絡/請求錯誤：{type(e).__name__}: {e}")
        return False, "\n".join(lines_out)



def fetch_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (tsla_df, vix_df) according to current mode.

    Both modes now use Yahoo Finance v8 chart API (fetch_yahoo_chart_api)
    which returns real-time 1-min bars including pre/post-market — same
    data source as TradingView Yahoo feed.

    Fallback chain:
      1. Yahoo chart API  (real-time, pre+post)
      2. yfinance library (slight delay, pre+post)
      3. Alpaca UVXY      (pre-market mode only, if both above fail)
    """
    is_pre = st.session_state.mode == "premarket"

    # ── TSLA ──────────────────────────────────────────
    tsla = fetch_yahoo_chart_api("TSLA", 30)
    if tsla.empty:
        tsla = fetch_yfinance("TSLA", 30, prepost=is_pre)

    # ── VIX ───────────────────────────────────────────
    vix = fetch_yahoo_chart_api("%5EVIX", 30)   # %5E = ^ URL-encoded
    if vix.empty:
        vix = fetch_yfinance("^VIX", 30, prepost=is_pre)
    # Last resort in pre-market: Alpaca UVXY
    if vix.empty and is_pre and "alpaca" in st.secrets:
        vix = fetch_alpaca("UVXY", 30)

    return tsla, vix

# ─────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────
def align_timestamps(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strip timezone info and align two DataFrames to their common timestamps.
    Rounds each index to the nearest minute to handle minor timestamp offsets.
    """
    try:
        d1 = df1.copy()
        d2 = df2.copy()
        # Normalize to UTC-naive, rounded to minute
        d1.index = pd.to_datetime(d1.index).tz_localize(None).round("1min")
        d2.index = pd.to_datetime(d2.index).tz_localize(None).round("1min")
        # Keep only shared timestamps
        common = d1.index.intersection(d2.index)
        return d1.loc[common], d2.loc[common]
    except Exception:
        return df1, df2


def get_vix_lag(tsla_df: pd.DataFrame, vix_df: pd.DataFrame) -> int:
    """Return how many minutes VIX lags behind TSLA (0 = in sync)."""
    try:
        t = pd.to_datetime(tsla_df.index[-1]).tz_localize(None)
        v = pd.to_datetime(vix_df.index[-1]).tz_localize(None)
        return max(0, int((t - v).total_seconds() / 60))
    except Exception:
        return 0


def compute_correlation(df1: pd.DataFrame, df2: pd.DataFrame) -> float | None:
    try:
        a, b = align_timestamps(df1, df2)
        combined = pd.concat(
            [a["Close"].rename("A"), b["Close"].rename("B")], axis=1
        ).dropna()
        if len(combined) < 5:
            return None
        return float(combined["A"].corr(combined["B"]))
    except Exception:
        return None


def detect_divergence(df1: pd.DataFrame, df2: pd.DataFrame,
                      window: int = 5) -> tuple[bool, str]:
    try:
        # Use aligned timestamps for fair comparison
        a, b = align_timestamps(df1, df2)
        t1 = a["Close"].tail(window)
        t2 = b["Close"].tail(window)
        if len(t1) < window or len(t2) < window:
            lag = get_vix_lag(df1, df2)
            if lag > 5:
                return False, f"⏳ VIX 數據落後 {lag} 分鐘，等待同步中…"
            return False, "數據不足"
        d1 = 1 if t1.iloc[-1] > t1.iloc[0] else (-1 if t1.iloc[-1] < t1.iloc[0] else 0)
        d2 = 1 if t2.iloc[-1] > t2.iloc[0] else (-1 if t2.iloc[-1] < t2.iloc[0] else 0)
        if d1 == 0 or d2 == 0:
            return False, "價格持平，無明顯方向"
        if d1 == d2:
            return True, f"⚠️ 負相關失效！{'同步上漲 ↑↑' if d1==1 else '同步下跌 ↓↓'}（過去 {window} 根K線）"
        return False, f"✅ 負相關正常：{'TSLA↑ 恐慌↓' if d1==1 else 'TSLA↓ 恐慌↑'}"
    except Exception:
        return False, "分析出錯"

# ─────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────
def make_candle_chart(tsla_df: pd.DataFrame, fear_df: pd.DataFrame,
                      fear_label: str) -> go.Figure:
    """
    Align both charts to the same time window so they stay in sync.
    Uses the intersection of the two timeframes for the X-axis range.
    """
    # ── find common time window (latest 15 bars of the SLOWER series) ──
    def tail15(df):
        return df.tail(15) if not df.empty else df

    t_df = tail15(tsla_df)
    f_df = tail15(fear_df)

    # Use the overlapping time range so both charts show same period
    if not t_df.empty and not f_df.empty:
        # Convert to UTC-naive for comparison
        t_idx = t_df.index.tz_localize(None) if t_df.index.tzinfo else t_df.index
        f_idx = f_df.index.tz_localize(None) if f_df.index.tzinfo else f_df.index
        x_start = max(t_idx[0],  f_idx[0])
        x_end   = max(t_idx[-1], f_idx[-1])  # show up to latest of either
        lag_mins = int((t_idx[-1] - f_idx[-1]).total_seconds() / 60)
        lag_note = f"  ⚠️ VIX 數據落後 {lag_mins} 分鐘" if lag_mins > 2 else ""
    else:
        x_start, x_end, lag_note = None, None, ""

    tsla_title = "TSLA  1-Min K線（最新15根）"
    fear_title = f"{fear_label}  1-Min K線（最新15根）{lag_note}"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,          # independent axes — different update freq
        subplot_titles=(tsla_title, fear_title),
        vertical_spacing=0.14,
        row_heights=[0.5, 0.5],
    )

    def add_candles(df, row, cu, cd, name):
        t = df.tail(15)
        fig.add_trace(go.Candlestick(
            x=t.index, open=t["Open"], high=t["High"],
            low=t["Low"], close=t["Close"], name=name,
            increasing_line_color=cu, decreasing_line_color=cd,
            increasing_fillcolor=cu, decreasing_fillcolor=cd, line_width=1,
        ), row=row, col=1)

    if not tsla_df.empty: add_candles(tsla_df, 1, "#00d084", "#ff4d6d", "TSLA")
    if not fear_df.empty: add_candles(fear_df, 2, "#f4c542", "#ff6e40", fear_label)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#13171f",
        font=dict(family="Inter, sans-serif", color="#c9d1e0", size=12),
        margin=dict(l=10, r=10, t=50, b=10), height=620, showlegend=False,
        xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e2533", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1e2533", zeroline=False)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=13, color="#8899aa" if "⚠️" not in ann.text else "#f4c542")
    return fig


def make_corr_chart(history: list) -> go.Figure:
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["corr"], mode="lines+markers",
        line=dict(color="#5b8af5", width=2),
        marker=dict(size=5, color=[
            "#ff4d6d" if v > -0.3 else "#00d084" for v in df["corr"]
        ]),
    ))
    fig.add_hline(y=0,    line_dash="dot",  line_color="#555")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="#f4c542",
                  annotation_text="弱負相關閾值 −0.5",
                  annotation_position="bottom right")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#13171f",
        font=dict(color="#c9d1e0", size=11),
        margin=dict(l=10, r=10, t=20, b=10), height=220,
        yaxis=dict(range=[-1.1, 1.1]), showlegend=False,
    )
    return fig

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 控制面板")

    # ── Mode toggle ──
    st.markdown("### 🔀 切換交易時段")
    col_a, col_b = st.columns(2)
    with col_a:
        reg_type = "primary" if st.session_state.mode == "regular" else "secondary"
        if st.button("📈 盤中\nyfinance", use_container_width=True, type=reg_type):
            if st.session_state.mode != "regular":
                st.session_state.mode          = "regular"
                st.session_state.corr_history  = []
                st.session_state.alert_history = []
                st.session_state.last_refresh  = None
                st.cache_data.clear()
                st.rerun()
    with col_b:
        pre_type = "primary" if st.session_state.mode == "premarket" else "secondary"
        if st.button("🌅 盤前後\nyfinance", use_container_width=True, type=pre_type):
            if st.session_state.mode != "premarket":
                st.session_state.mode          = "premarket"
                st.session_state.corr_history  = []
                st.session_state.alert_history = []
                st.session_state.last_refresh  = None
                st.cache_data.clear()
                st.rerun()

    cfg = MODE_CONFIG[st.session_state.mode]
    st.markdown(
        f'<div class="mode-badge-{cfg["badge"]}" style="margin-top:8px">{cfg["label"]}</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.session_state.auto_refresh = st.toggle(
        "每分鐘自動更新", value=st.session_state.auto_refresh)
    st.session_state.diverge_window = st.slider(
        "偏離偵測視窗（K線數）", 3, 10, st.session_state.diverge_window)
    st.session_state.alert_cooldown_min = st.slider(
        "Telegram 冷卻時間（分鐘）", 1, 30, st.session_state.alert_cooldown_min)
    manual_refresh = st.button("🔄 立即刷新")

    st.divider()
    st.markdown("### 📱 Telegram")
    tg_ok = "telegram" in st.secrets
    st.markdown(
        f'<span style="color:{"#00d084" if tg_ok else "#ff4d6d"}">'
        f'{"✅ 已設定" if tg_ok else "❌ 未設定"}</span>',
        unsafe_allow_html=True,
    )
    if tg_ok and st.button("發送測試訊息"):
        ok = send_telegram("🔔 <b>TSLA 監控儀表板</b>\n連線測試成功 ✅")
        st.success("已發送！") if ok else st.error("發送失敗")

    if st.session_state.mode == "premarket":
        st.divider()
        st.markdown("### 🔑 Alpaca API")
        ap_ok = "alpaca" in st.secrets
        st.markdown(
            f'<span style="color:{"#00d084" if ap_ok else "#ff4d6d"}">'
            f'{"✅ 已設定" if ap_ok else "❌ 未設定"}</span>',
            unsafe_allow_html=True,
        )
        if not ap_ok:
            st.caption("請在 secrets.toml 加入：")
            st.code("""[alpaca]
api_key    = "PKXXXXXXXXXXXXXXXX"
api_secret = "XXXXXXXXXXXXXXXXXXXXXXXX"
""", language="toml")
            st.caption("免費註冊 → alpaca.markets")
        else:
            # Show last fetch error if any
            last_err = st.session_state.get("alpaca_last_error")
            if last_err:
                st.markdown(
                    f'<div style="background:#1f0a0a;border-left:3px solid #ff4d6d;'
                    f'border-radius:6px;padding:8px 10px;margin:6px 0;'
                    f'font-size:11px;color:#ffb3b3;word-break:break-all;">'
                    f'⚠️ 錯誤詳情：<br>{last_err}</div>',
                    unsafe_allow_html=True,
                )
            # Diagnostic test button
            if st.button("🔍 測試 Alpaca 連線"):
                with st.spinner("診斷中…"):
                    ok, msg = test_alpaca_connection()
                if ok:
                    st.success("✅ 連線成功！數據正常")
                else:
                    st.error("❌ 連線失敗")
                st.code(msg, language="text")

    st.divider()
    if st.button("🗑️ 清除歷史"):
        st.session_state.corr_history  = []
        st.session_state.alert_history = []

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
cfg = MODE_CONFIG[st.session_state.mode]

st.markdown(f"""
<h1 style="text-align:center;font-size:24px;font-weight:700;
           color:#c9d1e0;margin-bottom:2px;">
    📊 TSLA × {cfg['fear_display']} 實時負相關監控儀表板
</h1>
<p style="text-align:center;color:#556070;font-size:12px;margin-bottom:6px;">
    {cfg['subtitle']}
</p>
""", unsafe_allow_html=True)

if cfg["note"]:
    st.markdown(f'<div class="info-box">ℹ️ {cfg["note"]}</div>',
                unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Refresh logic
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

status_row = st.container()
chart_col1, chart_col2 = st.columns([3, 2])
corr_section = st.container()

if should_refresh:
    with st.spinner("正在拉取最新市場數據…"):
        tsla_df, fear_df = fetch_pair()
    st.session_state.last_refresh = datetime.now()

    corr = compute_correlation(tsla_df, fear_df)
    if corr is not None:
        st.session_state.corr_history.append({
            "time": now_et.strftime("%H:%M"),
            "corr": round(corr, 4),
        })
        st.session_state.corr_history = st.session_state.corr_history[-60:]

    diverged, div_desc = detect_divergence(
        tsla_df, fear_df, st.session_state.diverge_window)

    if diverged:
        cooldown_ok = True
        if st.session_state.last_alert_time is not None:
            elapsed = (datetime.now() - st.session_state.last_alert_time
                       ).total_seconds() / 60
            cooldown_ok = elapsed >= st.session_state.alert_cooldown_min

        ts_p = float(tsla_df["Close"].iloc[-1]) if not tsla_df.empty else 0
        fe_p = float(fear_df["Close"].iloc[-1]) if not fear_df.empty else 0
        tag  = "【盤前/盤後 UVXY】" if st.session_state.mode == "premarket" else "【盤中 VIX】"

        alert_msg = (
            f"⚠️ <b>TSLA × {cfg['fear_display']} 負相關失效！</b> {tag}\n"
            f"時間：{now_et.strftime('%Y-%m-%d %H:%M')} ET\n"
            f"TSLA：${ts_p:.2f}　{cfg['fear_display']}：{fe_p:.2f}\n"
            f"相關係數：{corr:.3f}\n{div_desc}"
        )
        if cooldown_ok:
            sent = send_telegram(alert_msg)
            st.session_state.last_alert_time = datetime.now()
            st.session_state.alert_history.append({
                "time": now_et.strftime("%H:%M"),
                "msg":  div_desc,
                "sent": sent,
            })
else:
    tsla_df, fear_df = fetch_pair()
    corr = compute_correlation(tsla_df, fear_df)
    diverged, div_desc = detect_divergence(
        tsla_df, fear_df, st.session_state.diverge_window)

# ─────────────────────────────────────────────
# Metric cards
# ─────────────────────────────────────────────
def price_delta(df):
    if df.empty or len(df) < 2:
        return 0.0, 0.0
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2])
    return last, last - prev

with status_row:
    c1, c2, c3, c4, c5 = st.columns(5)
    tsla_p, tsla_chg = price_delta(tsla_df)
    fear_p, fear_chg = price_delta(fear_df)
    corr_str = f"{corr:.3f}" if corr is not None else "—"
    corr_cls = "negative" if corr and corr > -0.3 else "positive"

    with c1:
        cls = "positive" if tsla_chg >= 0 else "negative"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">TSLA 最新價</div>
            <div class="metric-value {cls}">${tsla_p:.2f}</div>
            <div class="metric-delta {cls}">{("+" if tsla_chg>=0 else "")}{tsla_chg:.2f}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        cls = "negative" if fear_chg >= 0 else "positive"
        lag = get_vix_lag(tsla_df, fear_df)
        lag_str = f"落後 {lag} 分鐘" if lag > 2 else "即時同步 ✓"
        lag_cls = "negative" if lag > 5 else ("neutral" if lag > 2 else "positive")
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">{cfg['fear_display']} 最新價</div>
            <div class="metric-value {cls}">{fear_p:.2f}</div>
            <div class="metric-delta {lag_cls}">{lag_str}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">皮爾森相關係數</div>
            <div class="metric-value {corr_cls}">{corr_str}</div>
            <div class="metric-delta neutral">目標 &lt; −0.5</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        s_cls  = "negative" if diverged else "positive"
        s_text = "⚠️ 偏離" if diverged else "✅ 正常"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">負相關狀態</div>
            <div class="metric-value {s_cls}" style="font-size:20px;">{s_text}</div>
            <div class="metric-delta neutral">視窗 {st.session_state.diverge_window} 根</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        mode_label  = "🌅 盤前/盤後" if st.session_state.mode == "premarket" else "📈 盤中"
        refresh_str = st.session_state.last_refresh.strftime("%H:%M:%S") \
                      if st.session_state.last_refresh else "—"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">當前模式</div>
            <div class="metric-value neutral" style="font-size:16px;">{mode_label}</div>
            <div class="metric-delta neutral">更新 {refresh_str}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Candle charts + analysis panel
# ─────────────────────────────────────────────
with chart_col1:
    st.markdown(
        f'<div class="section-title">📈 K線圖（每根 = 1 分鐘）— {cfg["fear_label"]}</div>',
        unsafe_allow_html=True)
    if not tsla_df.empty and not fear_df.empty:
        st.plotly_chart(
            make_candle_chart(tsla_df, fear_df, cfg["fear_display"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        if st.session_state.mode == "premarket":
            st.warning("⚠️ Alpaca 無數據 — 請確認 API Key 是否設定，或當前時段無交易活動。")
        else:
            st.warning("⚠️ 無法獲取K線數據 — 請確認市場是否開盤（09:30–16:00 ET）。")

with chart_col2:
    st.markdown('<div class="section-title">🔍 即時分析</div>', unsafe_allow_html=True)
    if diverged:
        st.markdown(
            f'<div class="alert-box">🚨 <b>負相關失效警告</b><br>{div_desc}</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ok-box">{div_desc}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 提醒記錄</div>', unsafe_allow_html=True)
    if st.session_state.alert_history:
        for a in reversed(st.session_state.alert_history[-8:]):
            icon = "📤" if a["sent"] else "⚠️"
            st.markdown(
                f'<div class="alert-box">{icon} [{a["time"]}] {a["msg"]}</div>',
                unsafe_allow_html=True)
    else:
        st.markdown('<div class="ok-box">暫無異常提醒記錄</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Correlation history
# ─────────────────────────────────────────────
with corr_section:
    st.markdown(
        f'<div class="section-title">📉 相關係數歷史（TSLA × {cfg["fear_display"]}）</div>',
        unsafe_allow_html=True)
    if st.session_state.corr_history:
        st.plotly_chart(
            make_corr_chart(st.session_state.corr_history),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        df_hist = pd.DataFrame(st.session_state.corr_history)
        h1, h2, h3 = st.columns(3)
        h1.metric("平均相關係數",       f"{df_hist['corr'].mean():.3f}")
        h2.metric("最低（最強負相關）",  f"{df_hist['corr'].min():.3f}")
        h3.metric("最高（最弱負相關）",  f"{df_hist['corr'].max():.3f}")
    else:
        st.info("等待數據累積中…（每分鐘更新一次）")

# ─────────────────────────────────────────────
# secrets.toml reference
# ─────────────────────────────────────────────
with st.expander("📄 secrets.toml 完整設定範例"):
    st.code("""# .streamlit/secrets.toml

[telegram]
bot_token = "1234567890:ABCxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
chat_id   = "987654321"

# 盤前/盤後模式需要 Alpaca（免費帳號即可）
# 免費註冊：https://alpaca.markets
[alpaca]
api_key    = "PKXXXXXXXXXXXXXXXXXXXXXX"
api_secret = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
""", language="toml")

# ─────────────────────────────────────────────
# Auto-refresh every 60s
# ─────────────────────────────────────────────
if st.session_state.auto_refresh:
    time.sleep(60)
    st.rerun()
