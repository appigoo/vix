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
    page_title="TSLA vs VIX 實時同步儀表板",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────
# CSS 樣式
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
    .metric-value { font-size: 26px; font-weight: 700; margin: 6px 0 2px; }
    .positive { color: #00d084; }
    .negative { color: #ff4d6d; }
    .info-box {
        background: #0d1a2e; border-left: 4px solid #5b8af5;
        border-radius: 8px; padding: 12px 16px; margin: 10px 0;
        font-size: 13px; color: #a8c4ff;
    }
    .mode-badge-premarket {
        display:inline-block; background:#1a1a0d; border:1px solid #f4c542;
        color:#f4c542; border-radius:20px; padding:4px 14px; font-size:13px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
if "mode" not in st.session_state: st.session_state.mode = "regular"
if "corr_history" not in st.session_state: st.session_state.corr_history = []
if "last_refresh" not in st.session_state: st.session_state.last_refresh = None

# ─────────────────────────────────────────────
# 數據拉取核心 (修正時差邏輯)
# ─────────────────────────────────────────────

def fetch_yahoo_realtime_quote(ticker: str) -> dict:
    """獲取 Yahoo 實時報價 (無 15 分鐘延遲)"""
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        result = r.json().get("quoteResponse", {}).get("result", [])
        if not result: return {}
        q = result[0]
        # 優先取盤前/盤後價，若無則取常規價
        price = q.get("preMarketPrice") or q.get("regularMarketPrice") or q.get("postMarketPrice")
        ts = q.get("preMarketTime") or q.get("regularMarketTime") or q.get("postMarketTime")
        if price and ts:
            dt = pd.Timestamp(ts, unit="s", tz="UTC").tz_convert("America/New_York")
            return {"price": price, "time": dt}
    except: pass
    return {}

@st.cache_data(ttl=30)
def fetch_yahoo_chart_api(ticker: str, bars: int = 40) -> pd.DataFrame:
    """獲取 1 分鐘 K 線數據"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1m&range=1d&includePrePost=true"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        data = r.json()["chart"]["result"][0]
        ts = data["timestamp"]
        quote = data["indicators"]["quote"][0]
        df = pd.DataFrame({
            "Open": quote["open"], "High": quote["high"],
            "Low": quote["low"], "Close": quote["close"],
        }, index=pd.to_datetime(ts, unit="s", utc=True))
        df.index = df.index.tz_convert("America/New_York")
        return df.dropna().tail(bars)
    except: return pd.DataFrame()

def patch_vix_latest(vix_df: pd.DataFrame, tsla_latest_time: datetime) -> pd.DataFrame:
    """
    核心修正：解決 16 分鐘時差。
    如果 VIX K 線落後於 TSLA，則使用實時 Quote 補齊中間的分鐘。
    """
    if vix_df.empty: return vix_df
    
    # 1. 獲取 VIX 真正的實時現價
    quote = fetch_yahoo_realtime_quote("%5EVIX")
    if not quote: return vix_df
    
    # 2. 確定對齊目標 (以 TSLA 的最後時間與 Quote 時間的較小值為準，防止過度超前)
    target_time = min(tsla_latest_time, quote["time"]).replace(second=0, microsecond=0)
    last_vix_time = vix_df.index[-1].replace(second=0, microsecond=0)
    
    # 3. 計算落後分鐘數
    lag_mins = int((target_time - last_vix_time).total_seconds() / 60)
    
    if lag_mins > 0:
        new_rows = []
        current_price = quote["price"]
        for i in range(1, lag_mins + 1):
            new_time = last_vix_time + timedelta(minutes=i)
            new_rows.append({
                "Open": current_price, "High": current_price,
                "Low": current_price, "Close": current_price,
            })
        new_df = pd.DataFrame(new_rows, index=pd.to_datetime([last_vix_time + timedelta(minutes=i) for i in range(1, lag_mins+1)]))
        vix_df = pd.concat([vix_df, new_df])
        vix_df = vix_df[~vix_df.index.duplicated(keep='last')]
        st.session_state["vix_patch_info"] = f"⚡ 已補齊 {lag_mins} 分鐘延遲 (VIX -> {target_time.strftime('%H:%M')})"
    else:
        st.session_state["vix_patch_info"] = "✅ VIX 數據已同步"
        
    return vix_df

def fetch_pair():
    # 獲取 TSLA
    tsla = fetch_yahoo_chart_api("TSLA")
    if tsla.empty: return pd.DataFrame(), pd.DataFrame()
    
    # 獲取 VIX
    vix = fetch_yahoo_chart_api("%5EVIX")
    if vix.empty: vix = fetch_yahoo_chart_api("^VIX")
    
    # 執行對齊補丁
    tsla_last_t = tsla.index[-1]
    vix = patch_vix_latest(vix, tsla_last_t)
    
    return tsla, vix

# ─────────────────────────────────────────────
# 分析與繪圖
# ─────────────────────────────────────────────

def align_and_corr(df1, df2):
    """嚴格對齊時間戳並計算相關性"""
    d1 = df1.copy(); d2 = df2.copy()
    d1.index = d1.index.round("1min")
    d2.index = d2.index.round("1min")
    common = d1.index.intersection(d2.index)
    if len(common) < 5: return None, None, None
    
    c1 = d1.loc[common, "Close"]
    c2 = d2.loc[common, "Close"]
    corr = c1.corr(c2)
    return corr, d1.loc[common], d2.loc[common]

# ─────────────────────────────────────────────
# 主界面
# ─────────────────────────────────────────────

st.markdown(f'<h2 style="text-align:center;">TSLA × VIX 負相關同步監控</h2>', unsafe_allow_html=True)

# 側邊欄控制
with st.sidebar:
    st.header("⚙️ 設定")
    st.session_state.mode = st.radio("交易時段", ["regular", "premarket"], format_func=lambda x: "📈 盤中" if x=="regular" else "🌅 盤前/盤後")
    if st.button("🔄 立即刷新"): st.cache_data.clear()
    st.divider()
    if "vix_patch_info" in st.session_state:
        st.info(st.session_state.vix_patch_info)

# 數據處理
tsla_raw, vix_raw = fetch_pair()

if not tsla_raw.empty and not vix_raw.empty:
    corr_val, tsla_aligned, vix_aligned = align_and_corr(tsla_raw, vix_raw)
    
    # 指標欄
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("TSLA 最新價", f"${tsla_raw['Close'].iloc[-1]:.2f}")
    with c2:
        st.metric("VIX 最新價", f"{vix_raw['Close'].iloc[-1]:.2f}")
    with c3:
        color = "positive" if (corr_val or 0) < -0.7 else "negative"
        st.markdown(f'<div class="metric-card"><div style="color:#889;">相關係數 (1m)</div>'
                    f'<div class="metric-value {color}">{corr_val:.4f if corr_val else "計算中"}</div></div>', unsafe_allow_html=True)

    # 圖表
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("TSLA (1-Min)", "VIX (1-Min Aligned)"))
    
    # TSLA Candlestick
    fig.add_trace(go.Candlestick(x=tsla_aligned.index, open=tsla_aligned['Open'], high=tsla_aligned['High'], 
                                 low=tsla_aligned['Low'], close=tsla_aligned['Close'], name="TSLA"), row=1, col=1)
    # VIX Candlestick
    fig.add_trace(go.Candlestick(x=vix_aligned.index, open=vix_aligned['Open'], high=vix_aligned['High'], 
                                 low=vix_aligned['Low'], close=vix_aligned['Close'], name="VIX"), row=2, col=1)
    
    fig.update_layout(height=700, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # 歷史警報邏輯 (簡化)
    if corr_val and corr_val > -0.2:
        st.warning(f"⚠️ 負相關失效警報！當前相關性為 {corr_val:.2f}，兩者可能正在同向變動。")
else:
    st.error("無法獲取數據，請檢查網絡或 API 狀態。")

# 自動刷新
time.sleep(60)
st.rerun()
