
import os, joblib, numpy as np
from tensorflow.keras.models import load_model
from data_fetcher import get_stock_data
from indicators import apply_all

MODEL_DIR='models'
MODEL_FILE=os.path.join(MODEL_DIR,'lstm_next_dir.h5')
SCALER_FILE=os.path.join(MODEL_DIR,'lstm_scaler.pkl')
SEQ=30
FEATURES=['Open','High','Low','Close','Volume','EMA20','EMA50','RSI','MACD','Signal','BB_UPPER','BB_LOWER','AvgVol']

def load_artifacts():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        raise FileNotFoundError('Train model first')
    model=load_model(MODEL_FILE)
    scaler=joblib.load(SCALER_FILE)
    return model, scaler

def prepare(symbol, interval='1d'):
    df = get_stock_data(symbol, interval=interval, limit=5000)
    if df.empty: return None, None
    df = apply_all(df)
    for c in FEATURES:
        if c not in df.columns: df[c]=0
    df = df[FEATURES].fillna(0)
    X = df.values
    if len(X) < SEQ: return None, None
    scaler = joblib.load(SCALER_FILE)
    Xs = scaler.transform(X)
    seq = Xs[-SEQ:]
    return seq.reshape(1, SEQ, len(FEATURES)), FEATURES

def predict_next(symbol, interval='1d'):
    try:
        model, scaler = load_artifacts()
    except Exception:
        return None
    seq, _ = prepare(symbol, interval=interval)
    if seq is None: return None
    prob = float(model.predict(seq, verbose=0)[0][0])
    return {'prob_up': prob, 'direction':1 if prob>=0.5 else 0}
