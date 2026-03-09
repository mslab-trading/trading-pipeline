import numpy as np
import pandas as pd
from data.data_dates import get_next_trading_date

import warnings
warnings.filterwarnings("ignore")

#  Input: df of data of a stock
# Output: series of its ADX
def calculate_adx(df: pd.DataFrame, period=14):
    # 計算 TR (True Range)
    df['high_low'] = df['etl:adj_high'] - df['etl:adj_low']
    df['high_close'] = abs(df['etl:adj_high'] - df['etl:adj_close'].shift(1))
    df['low_close'] = abs(df['etl:adj_low'] - df['etl:adj_close'].shift(1))
    df['TR'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    
    # 計算 +DM 和 -DM
    df['+DM'] = np.where((df['etl:adj_high'].diff() > 0) & (df['etl:adj_high'].diff() > -df['etl:adj_low'].diff()), df['etl:adj_high'].diff(), 0)
    df['-DM'] = np.where((-df['etl:adj_low'].diff() > 0) & (-df['etl:adj_low'].diff() > df['etl:adj_high'].diff()), -df['etl:adj_low'].diff(), 0)
    
    # 取 14 期加總值
    df['TR_smooth'] = df['TR'].rolling(window=period).sum()
    df['+DM_smooth'] = df['+DM'].rolling(window=period).sum()
    df['-DM_smooth'] = df['-DM'].rolling(window=period).sum()
    
    # 計算 +DI 和 -DI
    df['+DI'] = (df['+DM_smooth'] / df['TR_smooth']) * 100
    df['-DI'] = (df['-DM_smooth'] / df['TR_smooth']) * 100
    
    # 計算 DX
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    
    # 計算 ADX
    df['ADX'] = df['DX'].rolling(window=period).mean()
    
    # 選擇需要的欄位
    return df[['ADX']]

#  Input: Targets = [stock ids]
# Output: df of stockid*date, with true/false (whether ADX>40) as the values
def get_ADX_df(Target, cfg):
    df_ADX = pd.DataFrame()
    for target in Target:
        df_stock = pd.read_csv(f"{cfg['root_path']}/{target}.csv")
        df_stock.index = df_stock["date"]
        df_stock = calculate_adx(df_stock).fillna(0)
        df_ADX = pd.concat([df_ADX, df_stock], axis=1)
    df_ADX = df_ADX.fillna(False)
    df_ADX.columns = Target
    df_ADX = df_ADX.sort_index()

    unique_dates = df_ADX.index.unique()
    date_mapping = {date: get_next_trading_date(date) for date in unique_dates}
    df_ADX.index = df_ADX.index.map(date_mapping)

    return df_ADX

def get_buy_signals(pred_df: pd.DataFrame, val_df: pd.DataFrame, cfg):
    Target = pred_df.columns
    df_ADX = get_ADX_df(Target, cfg)
    df_canbuy = df_ADX > cfg["ADX_threshold"]
    df_canbuy = df_canbuy.reindex(index=pred_df.index, columns=pred_df.columns, fill_value=False)
    thresholds = val_df.stack().quantile(cfg["buy_percentile"])
    return (pred_df > thresholds) & df_canbuy

def get_sell_signals(pred_df: pd.DataFrame, val_df: pd.DataFrame, cfg):
    thresholds = val_df.stack().quantile(cfg["sell_percentile"])
    return pred_df < thresholds

