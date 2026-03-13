import gradio as gr
import yfinance as yf
import pandas as pd
import ta

def analyze(symbol):

    df = yf.download(symbol, period="3mo")

    macd = ta.trend.MACD(df["Close"])

    df["macd"] = macd.macd()

    signal = "BUY" if df["macd"].iloc[-1] > 0 else "SELL"

    return f"{symbol} signal: {signal}"

demo = gr.Interface(
    fn=analyze,
    inputs="text",
    outputs="text",
    title="AI Stock Analyzer"
)

demo.launch()