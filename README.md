
Stock Dashboard — Professional Advice (Long & Medium Term)
---------------------------------------------------------
How to run:
1. Unzip folder.
2. (Optional) create venv and activate.
3. Install: pip install -r requirements.txt
4. Run: python -m streamlit run app.py
5. In sidebar, type a stock (e.g. TCS) and click Refresh Data to fetch real data from Yahoo Finance.

Notes:
- This app uses Yahoo Finance via yfinance (free) — data may be delayed by 1-3 minutes.
- Trade Advice tab shows professional long-term and medium-term targets and suggestion.
- No simulator, no websocket, no broker required.
Included screenshot asset at: assets/screenshot.png


---
PRO UPGRADE: On 2025-11-20T13:13:14.639484, app.py was replaced with a stable professional upgrade (app_pro.py).
Features included: Date range presets, timeframe selector, candle spacing, indicator toggles,
multi-layout (Single/Two-up), ML prediction toggle, quick backtest, screener, market fundamentals,
and snapshot export. Drawing tools, alerts/watchlist, and news were intentionally omitted.
