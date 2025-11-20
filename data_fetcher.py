
import yfinance as yf
import pandas as pd, os, json

def load_stock_list():
    path = os.path.join(os.path.dirname(__file__), "stocks.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

STOCKS = load_stock_list()

def smart_symbol_search(query):
    if not isinstance(query, str): return query
    q = query.strip().lower()
    for name, symbol in STOCKS.items():
        if q == name.lower() or q == symbol.lower().replace('.ns',''):
            return symbol
    for name, symbol in STOCKS.items():
        if q in name.lower() or q in symbol.lower():
            return symbol
    if not query.endswith('.NS') and not query.endswith('.ns'):
        return query.upper() + '.NS'
    return query.upper()

def get_stock_data(symbol, interval='1d', limit=1000):
    symbol = smart_symbol_search(symbol)
    interval_map = {'1m':'7d','2m':'60d','5m':'60d','15m':'60d','30m':'60d','60m':'730d','1h':'730d','1d':'max'}
    period = interval_map.get(interval, '60d')
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    except Exception as e:
        print('yfinance error:', e)
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.tail(limit)
    for c in ['Open','High','Low','Close','Volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['Open','High','Low','Close'], inplace=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df
