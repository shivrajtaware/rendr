
import streamlit as st
import pandas as pd
import numpy as np
from data_fetcher import get_stock_data
from indicators import apply_all
from predictor import advice_engine
import ml_predict_lstm, ui_helpers

import plotly.graph_objects as go

st.set_page_config(page_title='Stock Dashboard — Pro Advice', layout='wide')
st.title('Stock Dashboard — Professional Long & Medium Term Advice')

with st.sidebar:
    symbol = st.text_input('Stock (smart search)', value='TCS')
    interval = st.selectbox('Interval', ['1d','60m','30m','15m','5m','1m'], index=0)
    max_candles = st.number_input('Max candles', min_value=50, max_value=2000, value=300)
    indicators = st.multiselect('Indicators', ['EMA','MACD','RSI','Bollinger','ATR','VWAP','OBV','VolSurge','ADX'], default=['EMA','MACD','RSI','Bollinger'])
    ema_periods = st.multiselect('EMA periods', [9,12,20,50,100,200], default=[20,50])
    st.markdown('---')
    st.write('Model predictions')
    enable_ml = st.checkbox('Enable LSTM prediction (requires trained model in /models)', value=True)
    st.markdown('---')
    st.write('Trade advice')
    own_stock = st.checkbox('I own this stock', value=False)
    if own_stock:
        entry_price = st.number_input('Entry price', value=0.0, format='%.2f')
        qty = st.number_input('Qty', value=0, step=1)
    st.markdown('---')
    if st.button('Refresh Data'):
        st.session_state['refresh'] = True

if 'refresh' not in st.session_state:
    st.session_state['refresh'] = False

if st.session_state['refresh'] or 'df' not in st.session_state:
    df = get_stock_data(symbol, interval=interval, limit=max_candles)
    if df.empty:
        st.error('No data found for symbol. Try another one.'); st.stop()
    df = apply_all(df, indicators, ema_periods)
    st.session_state['df'] = df.copy()
    st.session_state['refresh'] = False
else:
    df = st.session_state['df']

tabs = st.tabs(['Overview','Technical Analysis','Predictions','Trade Advice','Export'])
with tabs[0]:
    st.subheader(f'{symbol} — Overview')
    last = df.iloc[-1]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Last Price', f'{last.Close:.2f}', delta=f'{(last.Close - df.Close.iloc[-2]):.2f}')
    c2.metric('High', f'{last.High:.2f}'); c3.metric('Low', f'{last.Low:.2f}'); c4.metric('Volume', f'{int(last.Volume)}')

with tabs[1]:
    st.subheader('Technical Analysis')
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candles'))
    if 'EMA' in indicators:
        for p in ema_periods:
            col = f'EMA{p}'
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1)))
    if 'Bollinger' in indicators and 'BB_UPPER' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], name='BB_UPPER', line=dict(width=1), opacity=0.5))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], name='BB_LOWER', line=dict(width=1), opacity=0.5))
    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=700)
    st.plotly_chart(fig, use_container_width=True)

    if 'MACD' in indicators and 'MACD' in df.columns:
        st.subheader('MACD'); st.line_chart(df[['MACD','Signal']].tail(200))
    if 'RSI' in indicators and 'RSI' in df.columns:
        st.subheader('RSI'); st.line_chart(df['RSI'].tail(200))
    if 'ATR' in indicators and 'ATR' in df.columns:
        st.subheader('ATR'); st.line_chart(df['ATR'].tail(200))

with tabs[2]:
    st.subheader('Predictions')
    try:
        rb = advice_engine(df)
        st.write('Rule-based market action:', rb['action'])
        if rb['reason']: st.write('Reasons:'); st.write(rb['reason'])
    except Exception as e:
        st.error(f'Prediction error: {e}')
    ml_res = None
    if enable_ml:
        try:
            ml_res = ml_predict_lstm.predict_next(symbol, interval=interval)
        except Exception:
            ml_res = None
    if ml_res:
        st.write(f"LSTM probability up: {ml_res['prob_up']*100:.1f}%")
    else:
        st.info('No trained model found or not enough history. Use ml_trainer_lstm.py to train.')

with tabs[3]:
    st.subheader('Trade Advice (Professional)')
    entry = entry_price if own_stock and 'entry_price' in locals() else None
    q = int(qty) if own_stock and 'qty' in locals() else 0
    adv = advice_engine(df, entry_price if own_stock else None, qty if own_stock else 0)
    st.write('Market Action:', adv['action'])
    st.write('Rationale:'); st.write(adv['reason'])
    st.write('Support:', adv['support'], 'Resistance:', adv['resistance'])
    st.write('Medium-term Target:', adv['targets'][0]['price'], 'Long-term Target:', adv['targets'][1]['price'])
    if adv['atr'] is not None:
        st.write('ATR (latest):', round(adv['atr'],4))
        sl = round(df['Close'].iloc[-1] - 1.5*adv['atr'],2)
        st.write('Suggested ATR-based Stop-loss:', sl)
    if adv['portfolio']:
        st.write('Your Position P&L (%):', adv['portfolio']['pnl_percent'])
        st.write('Suggested Position Action:', adv['suggestion'])

with tabs[4]:
    st.subheader('Export important snapshot')
    export_df = pd.DataFrame({
        'Close': [df['Close'].iloc[-1]],
        'High': [df['High'].iloc[-1]],
        'Low': [df['Low'].iloc[-1]],
        'Volume': [int(df['Volume'].iloc[-1]) if 'Volume' in df.columns else None],
        'Rule_Action': [advice_engine(df)['action']]
    }, index=[df.index[-1]])
    # ensure timezone-naive
    export_df.index = export_df.index.tz_localize(None)
    xlsx_bytes = ui_helpers.df_to_excel_bytes({'snapshot': export_df})
    st.download_button('Download snapshot as Excel', xlsx_bytes, file_name=f'{symbol}_snapshot.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

st.markdown('---')
st.markdown('**Notes:** Real data via Yahoo Finance. Refresh data only when you click "Refresh Data".')
