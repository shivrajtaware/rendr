
import pandas as pd
import numpy as np

def EMA(series, span):
    return series.ewm(span=span, adjust=False).mean()

def SMA(series, window):
    return series.rolling(window).mean()

def MACD(df):
    df['EMA12'] = EMA(df['Close'], 12)
    df['EMA26'] = EMA(df['Close'], 26)
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_HIST'] = df['MACD'] - df['Signal']
    return df

def RSI(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def Bollinger(df, period=20):
    df['BB_MID'] = df['Close'].rolling(period).mean()
    df['BB_STD'] = df['Close'].rolling(period).std()
    df['BB_UPPER'] = df['BB_MID'] + 2 * df['BB_STD']
    df['BB_LOWER'] = df['BB_MID'] - 2 * df['BB_STD']
    return df

def ATR(df, period=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
    tr = df[['H-L','H-PC','L-PC']].max(axis=1)
    df['ATR'] = tr.rolling(period).mean()
    df.drop(columns=['H-L','H-PC','L-PC'], inplace=True)
    return df

def VWAP(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (tp * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

def OBV(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iat[i] > df['Close'].iat[i-1]:
            obv.append(obv[-1] + df['Volume'].iat[i])
        elif df['Close'].iat[i] < df['Close'].iat[i-1]:
            obv.append(obv[-1] - df['Volume'].iat[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    return df

def VolSurge(df, window=20, mult=2):
    df['AvgVol'] = df['Volume'].rolling(window).mean()
    df['VolSurge'] = df['Volume'] > (df['AvgVol'] * mult)
    return df

def ADX(df, period=14):
    up = df['High'].diff()
    down = -df['Low'].diff()
    plus_dm = up.where((up>down)&(up>0),0)
    minus_dm = down.where((down>up)&(down>0),0)
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus = (plus_dm.rolling(period).sum()/atr) * 100
    minus = (minus_dm.rolling(period).sum()/atr) * 100
    dx = ( (plus-minus).abs() / (plus+minus) ) * 100
    df['ADX'] = dx.rolling(period).mean()
    return df

def apply_all(df, indicators=None, ema_periods=[20,50,100]):
    if df is None or df.empty: return df
    if indicators is None: indicators = ['EMA','MACD','RSI','Bollinger','ATR','VWAP','OBV','VolSurge','ADX']
    if 'EMA' in indicators:
        for p in ema_periods:
            df[f'EMA{p}'] = EMA(df['Close'], p)
    if 'MACD' in indicators: df = MACD(df)
    if 'RSI' in indicators: df = RSI(df)
    if 'Bollinger' in indicators: df = Bollinger(df)
    if 'ATR' in indicators: df = ATR(df)
    if 'VWAP' in indicators: df = VWAP(df)
    if 'OBV' in indicators: df = OBV(df)
    if 'VolSurge' in indicators: df = VolSurge(df)
    if 'ADX' in indicators: df = ADX(df)
    df.fillna(value=pd.NA, inplace=True)
    return df
