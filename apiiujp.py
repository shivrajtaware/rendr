
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from data_fetcher import get_stock_data
from indicators import apply_all
from predictor import advice_engine
import ml_predict_lstm, ui_helpers
import plotly.graph_objects as go

st.set_page_config(page_title='Stock Dashboard', layout='wide')
st.title('Stock Dashboard — With Date Range')

# Sidebar: symbol and interval
with st.sidebar:
    symbol = st.text_input('Stock (smart search)', value='TCS')
    interval = st.selectbox('Interval', ['1d','1wk','1mo'], index=0)

# Date presets above chart
st.subheader("Date Range")
cols = st.columns(7)
preset = st.empty()
today = datetime.date.today()

# buttons
if cols[0].button("1M"): preset = '1M'
elif cols[1].button("3M"): preset = '3M'
elif cols[2].button("6M"): preset = '6M'
elif cols[3].button("YTD"): preset = 'YTD'
elif cols[4].button("1Y"): preset = '1Y'
elif cols[5].button("5Y"): preset = '5Y'
elif cols[6].button("MAX"): preset = 'MAX'
else: preset = None

# compute default start
if preset == '1M':
    default_start = today - datetime.timedelta(days=30)
elif preset == '3M':
    default_start = today - datetime.timedelta(days=90)
elif preset == '6M':
    default_start = today - datetime.timedelta(days=180)
elif preset == 'YTD':
    default_start = datetime.date(today.year,1,1)
elif preset == '1Y':
    default_start = today - datetime.timedelta(days=365)
elif preset == '5Y':
    default_start = today - datetime.timedelta(days=365*5)
elif preset == 'MAX':
    default_start = today - datetime.timedelta(days=365*20)
else:
    default_start = today - datetime.timedelta(days=365)

start_date, end_date = st.date_input("Select Custom Range", (default_start, today))

# Fetch data
df = get_stock_data(symbol, interval=interval)
if df is None or df.empty:
    st.error("No data.")
else:
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    if df.empty:
        st.warning("No data in selected range.")
    else:
        df = apply_all(df, indicators=['EMA','MACD','RSI'])

        # Candle chart with spacing
        plot_df = df.reset_index()
        plot_df['Date'] = pd.to_datetime(plot_df['Date'])
        plot_df['Date'] = plot_df['Date'] + pd.to_timedelta(np.arange(len(plot_df)), unit='m')

        fig = go.Figure(data=[go.Candlestick(
            x=plot_df['Date'],
            open=plot_df['Open'],
            high=plot_df['High'],
            low=plot_df['Low'],
            close=plot_df['Close']
        )])

        for col in plot_df.columns:
            if col.startswith('EMA'):
                fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df[col], mode='lines', name=col))
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Prediction
        try:
            rb = advice_engine(df)
            st.write("Advice:", rb.get('action'))
        except:
            pass

        try:
            ml_res = ml_predict_lstm.predict_next(symbol, interval=interval)
            st.write("ML Forecast:", ml_res)
        except:
            st.info("ML prediction failed.")
