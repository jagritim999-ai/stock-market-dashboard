import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import time
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(
    page_title="Real Time Stock Dashboard",
    layout="wide"
)

st.title("📊 Real Time Stock Market Dashboard")
st.subheader("📅 Date Range")

period = st.selectbox(
    "Select Period",
    ["1mo", "3mo", "6mo", "1y", "2y"]
)
if st.button("Refesh Data"):
    st.rerun()

# ==========================
# MAIN STOCK SELECTION
# ==========================

stock = st.selectbox(
    "Choose Main Stock",
    ["AAPL", "MSFT", "GOOGL", "TSLA", "INFY.NS", "TCS.NS"]
)

# ==========================
# DOWNLOAD DATA
# ==========================
with st.spinner("📡 Fetching latest stock data..."):
    data = yf.download(
        stock,
        period=period,
        auto_adjust=True,
        progress=False
    )


ticker = yf.Ticker(stock)
info = ticker.info

st.subheader("🏢 Company Information")

st.write("Company:", info.get("longName", "N/A"))
st.write("Sector:", info.get("sector", "N/A"))
st.write("Industry:", info.get("industry", "N/A"))

if data.empty:
    st.error("❌ No data found")
else:
    ...
if data.empty:
    st.error("❌ No data found")

else:

    # Fix MultiIndex
    if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
        data.columns = data.columns.get_level_values(0)

    # Current Price
    current_price = float(data["Close"].iloc[-1])
    previous_close = float(data["Close"].iloc[-2])

    change_percent = (
       (current_price - previous_close)
        / previous_close
    ) * 100

    # Moving Averages
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["BB_MA20"] = data["Close"].rolling(20).mean()
    data["BB_STD"] = data["Close"].rolling(20).std()

    data["Upper_Band"] = data["BB_MA20"] + (2 * data["BB_STD"])
    data["Lower_Band"] = data["BB_MA20"] - (2 * data["BB_STD"])

    # RSI
    delta = data["Close"].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    data["RSI"] = 100 - (100 / (1 + rs))
    # Risk Analysis
    data["Returns"] = data["Close"].pct_change()

    risk = data["Returns"].std()

    # High / Low
    high_price = float(data["High"].max())
    low_price = float(data["Low"].min())

    # ==========================
    # METRICS
    # ==========================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📌 Current Price", f"${current_price:.2f}")

    with col2:
        st.metric("📈 Highest Price", f"${high_price:.2f}")

    with col3:
        st.metric("📉 Lowest Price", f"${low_price:.2f}")
    with col4:
        st.metric(
          "📊 Daily Change %",
          f"{change_percent:.2f}%"
    )
        st.metric(
          "⚠️ Risk (Volatility)",
          f"{risk:.4f}"
       )
    # ==========================
    # BUY / SELL SIGNAL
    # ==========================

    latest_ma20 = data["MA20"].iloc[-1]
    latest_ma50 = data["MA50"].iloc[-1]

    if latest_ma20 > latest_ma50:
        st.success("🟢 BUY SIGNAL")
    else:
        st.error("🔴 SELL SIGNAL")

    # ==========================
    # DATA TABLE
    # ==========================

    st.subheader("📄 Stock Data")
    st.dataframe(data.tail())

    # ==========================
    # CANDLESTICK CHART
    # ==========================

    st.subheader("🕯️ Candlestick Chart")

    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"]
    )])

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # MOVING AVERAGE CHART
    # ==========================

    st.subheader("📈 Closing Price with Moving Averages")
    st.line_chart(data[["Close", "MA20", "MA50"]])

    # ==========================
    # VOLUME CHART
    # ==========================

    st.subheader("📊 Trading Volume")
    st.bar_chart(data["Volume"])

    # ==========================
    # RSI CHART
    # ==========================

    st.subheader("📈 RSI Indicator")
    st.line_chart(data["RSI"])

    latest_rsi = data["RSI"].iloc[-1]

    if latest_rsi > 70:
        st.warning("⚠️ Overbought Zone (RSI > 70)")
    elif latest_rsi < 30:
        st.success("🟢 Oversold Zone (RSI < 30)")
    else:
        st.info("ℹ️ Neutral Zone")
        st.subheader("📈 Bollinger Bands")

        st.line_chart(
            data[["Close", "Upper_Band", "Lower_Band"]]
        )
    # =========================
    # Portfolio Tracker
    # ==========================

    st.subheader("💼 Portfolio Tracker")

    shares = st.number_input(
        "Number of Shares",
         min_value=1,
         value=10
    )

    buy_price = st.number_input(
        "Buy Price",
         value=100.0
    )

    investment = shares * buy_price
    current_value = shares * current_price

    profit_loss = current_value - investment

    st.metric(
       "💰 Profit / Loss",
       f"${profit_loss:.2f}"
    ) 
    st.write(data)
    st.write("Shape:", data.shape)
    st.subheader("🤖 Stock Price Prediction")

    ml_data = data.dropna()

    #  🔥 SAFE CHECK
    if ml_data.empty:
         st.error("❌ No data available. Check stock symbol or internet.")
         st.stop()

    X = np.arange(len(ml_data)).reshape(-1, 1)
    y = ml_data["Close"].values

    model = LinearRegression()
    model.fit(X, y)

    future_day = np.array([[len(ml_data)]])

    prediction = model.predict(future_day)

    st.metric(
        "🔮 Predicted Next Price",
        f"${prediction[0]:.2f}"
    )
    # 📥 DOWNLOAD BUTTON (HERE)
    st.download_button(
        "📥 Download Data",
        data.to_csv(),
        file_name="stock_data.csv",
        mime="text/csv"
    )
    # ==========================
    # DOWNLOAD CSV
    # ==========================

    st.subheader("📥 Download Stock Data")

    csv = data.to_csv().encode("utf-8")

    st.download_button(
        label="Download CSV Report",
        data=csv,
        file_name=f"{stock}_stock_data.csv",
        mime="text/csv"
    )

# ==========================
# STOCK COMPARISON
# ==========================

st.subheader("📊 Compare Multiple Stocks")

compare_stocks = st.multiselect(
    "Select Stocks for Comparison",
    ["AAPL", "MSFT", "GOOGL", "TSLA", "INFY.NS", "TCS.NS"],
    default=["AAPL", "MSFT"]
)

comparison_data = pd.DataFrame()

for s in compare_stocks:

    df = yf.download(
        s,
        period="6mo",
        auto_adjust=True,
        progress=False
    )

    if not df.empty:

        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)

        comparison_data[s] = df["Close"]

if not comparison_data.empty:
    st.line_chart(comparison_data)