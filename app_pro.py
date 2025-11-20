
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from data_fetcher import get_stock_data, smart_symbol_search, STOCKS
from indicators import apply_all
from predictor import advice_engine
import ml_predict_lstm, ui_helpers
import plotly.graph_objects as go

st.set_page_config(page_title='Stock Dashboard — Pro (Upgraded)', layout='wide')
st.title('Stock Dashboard — Professional — Pro Upgrade')

# ---------------------- Sidebar controls ----------------------
with st.sidebar:
    st.header("Controls")
    symbol_input = st.text_input('Symbol (smart search)', value='TCS')
    symbol = smart_symbol_search(symbol_input) if isinstance(symbol_input, str) else symbol_input

    # Timeframe selector (common terminal timeframes)
    interval = st.selectbox("Timeframe / Interval", options=['1d','1wk','1mo','1m','5m','15m','30m','60m'], index=0,
                            help="Choose data interval. (1m,5m,15m,30m,60m for intraday if supported by source)")

    # Date range presets
    today = datetime.date.today()
    presets = st.radio("Quick range", options=['1M','3M','6M','YTD','1Y','2Y','5Y','MAX','Custom'], index=4, horizontal=True)

    # Default start based on preset
    if presets == '1M':
        default_start = today - datetime.timedelta(days=30)
    elif presets == '3M':
        default_start = today - datetime.timedelta(days=90)
    elif presets == '6M':
        default_start = today - datetime.timedelta(days=182)
    elif presets == 'YTD':
        default_start = datetime.date(today.year,1,1)
    elif presets == '1Y':
        default_start = today - datetime.timedelta(days=365)
    elif presets == '2Y':
        default_start = today - datetime.timedelta(days=365*2)
    elif presets == '5Y':
        default_start = today - datetime.timedelta(days=365*5)
    elif presets == 'MAX':
        default_start = today - datetime.timedelta(days=365*20)
    else:
        default_start = today - datetime.timedelta(days=365)

    start_date, end_date = st.date_input("Select date range", value=(default_start, today))

    # Multi-chart layout
    layout_mode = st.radio("Chart layout", options=['Single','Two-up','Four-grid'], index=0)

    # Indicators selection
    st.markdown("**Indicators**")
    enable_ema = st.checkbox('EMA (12,26,50)', value=True)
    enable_macd = st.checkbox('MACD', value=True)
    enable_rsi = st.checkbox('RSI', value=True)
    enable_bbands = st.checkbox('Bollinger Bands', value=False)
    enable_vwap = st.checkbox('VWAP', value=False)
    enable_atr = st.checkbox('ATR', value=False)
    enable_obv = st.checkbox('OBV', value=False)

    enable_ml = st.checkbox('Enable ML predictions', value=True)

    run_refresh = st.button('Refresh Data')

# ---------------------- Helpers ----------------------
def safe_get_stock(symbol, interval):
    try:
        # Some projects expect interval named 'period' or 'interval' - try common signature
        df = get_stock_data(symbol, interval=interval)
    except TypeError:
        try:
            df = get_stock_data(symbol)
        except Exception as e:
            st.error(f'Error fetching data: {e}')
            return pd.DataFrame()
    except Exception as e:
        st.error(f'Error fetching data: {e}')
        return pd.DataFrame()
    return df.copy()

def apply_indicators_to_df(df):
    indicators = []
    if enable_ema: indicators.append('EMA')
    if enable_macd: indicators.append('MACD')
    if enable_rsi: indicators.append('RSI')
    if enable_bbands: indicators.append('Bollinger')
    if enable_atr: indicators.append('ATR')
    if enable_vwap: indicators.append('VWAP')
    if enable_obv: indicators.append('OBV')
    if len(indicators)==0:
        return df
    try:
        df2 = apply_all(df.copy(), indicators=indicators)
        return df2
    except Exception as e:
        st.warning(f'Indicator calculation failed: {e}')
        return df

def spaced_dates_index(df):
    # Create a small spacing by adding incremental seconds to the index.
    # Works if index is a DatetimeIndex.
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # Add spacing proportional to row number (in seconds) to create visible gap
    offsets = pd.to_timedelta(np.arange(len(df)), unit='s')
    return df.index + offsets

def plot_candles(df, title="Candles", show_indicators=True):
    if df.empty:
        st.write("No data to plot.")
        return
    plot_df = df.copy()
    # Use spaced index for visual gaps
    plot_df = plot_df.reset_index().rename(columns={'index':'Date'})
    plot_df['Date'] = pd.to_datetime(plot_df['Date'])
    plot_df['Date'] = plot_df['Date'] + pd.to_timedelta(np.arange(len(plot_df)), unit='m')

    fig = go.Figure(data=[go.Candlestick(
        x=plot_df['Date'],
        open=plot_df['Open'],
        high=plot_df['High'],
        low=plot_df['Low'],
        close=plot_df['Close'],
        increasing_line_color='green', decreasing_line_color='red',
        )])

    # Overlays: EMA lines if present
    for col in plot_df.columns:
        if show_indicators and col.startswith('EMA'):
            fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df[col], mode='lines', name=col))
        if show_indicators and col=='UpperBB':
            fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['UpperBB'], mode='lines', name='UpperBB', opacity=0.6))
        if show_indicators and col=='LowerBB':
            fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['LowerBB'], mode='lines', name='LowerBB', opacity=0.6))
        if show_indicators and col=='VWAP':
            fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['VWAP'], mode='lines', name='VWAP', opacity=0.8))

    fig.update_layout(xaxis_rangeslider_visible=False, title=title, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- Main ----------------------
df = safe_get_stock(symbol, interval)
if df is None or df.empty:
    st.warning("No data returned. Check the symbol or internet connection.")
else:
    # filter by selected date range
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df[(df.index>=start_ts) & (df.index<=end_ts)]
    if df.empty:
        st.warning("No data in the selected date range.")
    else:
        # compute indicators
        df_ind = apply_indicators_to_df(df)
        # Layout handling
        if layout_mode == 'Single':
            plot_candles(df_ind, title=f"{symbol} — {interval}")
        elif layout_mode == 'Two-up':
            col1, col2 = st.columns(2)
            with col1:
                plot_candles(df_ind, title=f"{symbol} — Left — {interval}")
            with col2:
                # show Heikin-Ashi or daily aggregated
                ha = df_ind.copy()
                ha['Close'] = df_ind['Close']  # keep closure for now
                plot_candles(ha, title=f"{symbol} — Right — Aggregated")
        else:
            # Four-grid: price, volume, RSI, MACD
            cols = st.columns(2)
            with cols[0]:
                plot_candles(df_ind, title=f"{symbol} — {interval}")
            with cols[1]:
                st.subheader("Volume")
                st.bar_chart(df_ind['Volume'] if 'Volume' in df_ind.columns else pd.Series())

            st.markdown("### Indicators")
            if 'RSI' in df_ind.columns:
                st.line_chart(df_ind['RSI'])
            if 'MACD' in df_ind.columns:
                st.line_chart(df_ind[['MACD','Signal']])

        # ---------------- Predictions & Backtest ----------------
        st.markdown('---')
        st.header('Predictions & Backtesting')
        try:
            rb = advice_engine(df_ind)
            st.write('Rule-based market action:', rb.get('action'))
            if rb.get('reason'): st.write('Reasons:'); st.write(rb.get('reason'))
        except Exception as e:
            st.warning(f'Rule-based advice failed: {e}')

        if enable_ml:
            try:
                ml_res = ml_predict_lstm.predict_next(symbol, interval=interval)
                st.subheader('ML forecast (next steps)')
                st.write(ml_res)
            except Exception as e:
                st.info('ML prediction unavailable or failed.')

        # Simple backtester: SMA 50/200 crossover
        st.subheader('Quick backtest — SMA(50/200) crossover')
        try:
            df_bt = df_ind.copy()
            df_bt['SMA50'] = df_bt['Close'].rolling(50).mean()
            df_bt['SMA200'] = df_bt['Close'].rolling(200).mean()
            df_bt.dropna(inplace=True)
            df_bt['position'] = np.where(df_bt['SMA50']>df_bt['SMA200'], 1, 0)
            df_bt['returns'] = df_bt['Close'].pct_change().fillna(0)
            df_bt['strategy'] = df_bt['position'].shift(1) * df_bt['returns']
            cum_ret = (1+df_bt['strategy']).cumprod().iloc[-1] if not df_bt['strategy'].empty else np.nan
            buy_and_hold = (1+df_bt['returns']).cumprod().iloc[-1] if not df_bt['returns'].empty else np.nan
            st.write({'Strategy CAGR_approx': round((cum_ret**(252/len(df_bt))-1) if len(df_bt)>0 else np.nan,4),
                      'BuyHold_approx': round((buy_and_hold**(252/len(df_bt))-1) if len(df_bt)>0 else np.nan,4)})
        except Exception as e:
            st.info(f'Backtest failed: {e}')

        # ---------------- Screener ----------------
        st.markdown('---')
        st.header('Screener (quick)')
        try:
            stocks = STOCKS if isinstance(STOCKS, dict) else {}
            screener_df = []
            for sym, meta in list(stocks.items())[:120]:
                try:
                    sdf = safe_get_stock(sym, interval='1d')
                    if sdf is None or sdf.empty: continue
                    # compute simple momentum = pct change over period
                    period_days = int((pd.to_datetime(end_date)-pd.to_datetime(start_date)).days) or 1
                    pct = (sdf['Close'].iloc[-1] - sdf['Close'].iloc[0]) / max(1e-9, sdf['Close'].iloc[0])
                    screener_df.append({'symbol': sym, 'pct_change': pct, 'last': sdf['Close'].iloc[-1]})
                except Exception:
                    continue
            screener_df = pd.DataFrame(screener_df).sort_values('pct_change', ascending=False)
            st.dataframe(screener_df.head(50))
        except Exception as e:
            st.info(f'Screener failed: {e}')

        # ---------------- Market Info ----------------
        st.markdown('---')
        st.header('Market Info & Fundamentals')
        try:
            import yfinance as yf
            t = yf.Ticker(symbol)
            info = t.info if hasattr(t, 'info') else {}
            basics = {k: info.get(k) for k in ['sector','industry','marketCap','previousClose','open','volume','averageVolume','forwardPE','trailingPE']}
            st.write(basics)
        except Exception as e:
            st.info('Fundamentals fetch failed.')

        # ---------------- Export snapshot ----------------
        export_df = pd.DataFrame({
            'Close': [df['Close'].iloc[-1]],
            'High': [df['High'].iloc[-1]],
            'Low': [df['Low'].iloc[-1]],
            'Volume': [int(df['Volume'].iloc[-1]) if 'Volume' in df.columns else None],
            'Rule_Action': [advice_engine(df_ind).get('action')]
        }, index=[df.index[-1]])
        # ensure timezone-naive
        export_df.index = export_df.index.tz_localize(None)
        xlsx_bytes = ui_helpers.df_to_excel_bytes({'snapshot': export_df})
        st.download_button('Download snapshot as Excel', xlsx_bytes, file_name=f'{symbol}_snapshot.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

st.markdown('---')
st.caption('Pro Upgrade: Date presets, timeframes, multi-layouts, indicators, screener, prediction & quick backtest. (No drawing tools, no alerts/watchlist, no news)')
