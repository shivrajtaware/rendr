
import pandas as pd, io
def df_to_excel_bytes(df_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for name, df in df_dict.items():
            # ensure timezone-naive index
            df = df.copy()
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.to_excel(writer, sheet_name=name[:31], index=True)
    output.seek(0)
    return output.read()
