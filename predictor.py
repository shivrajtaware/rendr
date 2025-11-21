
import numpy as np
import pandas as pd
def compute_support_resistance(df, lookback=60):
    highs = df['High'].tail(lookback)
    lows = df['Low'].tail(lookback)
    support = float(lows.min())
    resistance = float(highs.max())
    return support, resistance

def advice_engine(df, entry_price=None, qty=0):
    if df is None or df.empty:
        return {'action':'NO_DATA','reason':['No data'],'support':None,'resistance':None,'targets':[]}
    last = df.iloc[-1]
    support, resistance = compute_support_resistance(df, lookback=90)
    reason = []
    score = 0
    # MACD
    if 'MACD' in df.columns and 'Signal' in df.columns:
        if last['MACD'] > last['Signal']:
            score += 1; reason.append('MACD above Signal')
        else:
            score -= 1; reason.append('MACD below Signal')
    # EMA trend
    if 'EMA20' in df.columns and 'EMA50' in df.columns:
        if last['EMA20'] > last['EMA50']:
            score += 1; reason.append('EMA20>EMA50 (trend up)')
        else:
            score -= 1; reason.append('EMA20<EMA50 (trend down)')
    # RSI
    if 'RSI' in df.columns:
        r = last['RSI']
        if r < 30:
            score += 1; reason.append('RSI oversold')
        elif r > 70:
            score -= 1; reason.append('RSI overbought')
    # Vol surge
    if 'VolSurge' in df.columns and last['VolSurge']:
        score += 1; reason.append('Volume surge')
    # ADX strength
    strength = None
    if 'ADX' in df.columns and pd.notna(df['ADX'].iloc[-1]):
        adx = df['ADX'].iloc[-1]
        strength = float(adx)
        if adx > 25:
            reason.append('ADX indicates strong trend')
    # ATR for stop-loss
    atr = None
    if 'ATR' in df.columns and pd.notna(df['ATR'].iloc[-1]):
        atr = float(df['ATR'].iloc[-1])
    # compute targets: medium-term and long-term
    current = float(last['Close'])
    mt_target = current * 1.05  # medium-term +5%
    lt_target = current * 1.12  # long-term +12%
    targets = [{'label':'Medium-term','price':round(mt_target,2)},{'label':'Long-term','price':round(lt_target,2)}]
    # decide action
    if score >= 2:
        action = 'BUY'
    elif score <= -2:
        action = 'SELL'
    else:
        action = 'HOLD'
    # portfolio-aware suggestions (if entry given)
    portfolio = None
    if entry_price and qty>0:
        pnl = (current - entry_price) / entry_price * 100
        portfolio = {'entry':entry_price,'qty':qty,'pnl_percent':round(pnl,2)}
        # rules for holding/selling based on pnl and action
        if action == 'BUY' and pnl > 10:
            suggestion = 'HOLD (consider trailing stop to protect profits)'
        elif action == 'SELL' and pnl > 5:
            suggestion = 'PARTIAL_EXIT (book partial profits)'
        elif action == 'SELL' and pnl <= 0:
            suggestion = 'CUT_LOSS (consider exit)'
        else:
            suggestion = 'HOLD'
    else:
        suggestion = None
    out = {'action':action,'reason':reason,'support':round(support,2),'resistance':round(resistance,2),'targets':targets,'atr':atr,'strength':strength,'portfolio':portfolio,'suggestion':suggestion}
    return out
